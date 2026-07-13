"""§3.1 step 7: the LLM Gateway's agent-backed prose Definition-of-Done judge
(`LLMGateway.judge_prose`) and its end-to-end gating. Covers the forced-tool
and guided judgement paths, index→criterion mapping, fail-closed omission, the
unknown-`evaluated_by` config error, the repair loop / its exhaustion, and the
whole path (engine gate → async evaluate_dod → judge_prose → verdict → COMPLETE
or FAIL). No network: scripted FakeAdapters only, reusing test_gateway's fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.base import AgentOutputError
from ravana.runtime.gateway import SUBMIT_VERDICT, LLMGateway
from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderResponse,
)
from ravana.schema.models import WorkflowDoc
from tests.test_gateway import FakeAdapter, _guided_text, _submit


def _judge_graph(*, provider: str = "anthropic", model: str = "m", criteria: list[str] | None = None):
    """A one-agent workflow whose evaluated_by agent is `pm`. When `criteria` is
    given, it also carries a prose DoD (for the end-to-end gate tests)."""
    spec: dict[str, Any] = {
        "state": {"schema": {"done": {"type": "boolean"}}, "initial": {}},
        "agents": [
            {
                "id": "pm",
                "name": "PM",
                "llm": {"provider": provider, "model": model},
                "system_prompt": "you are the pm",
                "output_schema": {"type": "object", "additionalProperties": True},
            }
        ],
        "graph": {"entry": "n", "nodes": [{"id": "n", "agent": "pm"}], "edges": []},
    }
    if criteria is not None:
        spec["definition_of_done"] = {"evaluated_by": "pm", "criteria": criteria}
    return compile_workflow(
        WorkflowDoc.model_validate(
            {"apiVersion": "ravana/v1", "kind": "Workflow", "metadata": {"name": "judge", "version": 1}, "spec": spec}
        )
    )


def _verdict(items: Any) -> ProviderResponse:
    """A forced-tool judgement response: a submit_verdict tool call carrying
    `items` as its `verdicts`."""
    return _verdict_call({"verdicts": items})


def _verdict_call(args: dict[str, Any]) -> ProviderResponse:
    """A submit_verdict tool call with arbitrary arguments (for the malformed /
    missing-key cases the repair loop must catch)."""
    return ProviderResponse(text="", tool_calls=[NormalizedToolCall(id="v", tool=SUBMIT_VERDICT, arguments=args)])


# --- judge_prose in isolation ----------------------------------------------
def test_judge_prose_forced_tool_maps_verdicts_by_index():
    graph = _judge_graph()
    adapter = FakeAdapter(
        caps={Capability.NATIVE_STRUCTURED_OUTPUT},
        responses=[_verdict([{"index": 0, "met": True}, {"index": 1, "met": False}])],
    )
    gateway = LLMGateway(graph, {"anthropic": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["c-zero", "c-one"], {"x": 1}))
    assert out == {"c-zero": True, "c-one": False}
    # A judgement offers ONLY the submit_verdict tool (no toolkits) and forces it.
    req = adapter.requests[0]
    assert req.force_tool == SUBMIT_VERDICT
    assert [t.name for t in req.tools] == [SUBMIT_VERDICT]


def test_judge_prose_guided_path_parses_json_text():
    graph = _judge_graph(provider="local", model="local-model")
    adapter = FakeAdapter(
        caps={Capability.GUIDED_DECODING},
        responses=[_guided_text({"verdicts": [{"index": 0, "met": True}]})],
    )
    gateway = LLMGateway(graph, {"local": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["only"], {}))
    assert out == {"only": True}
    # Guided path constrains the whole response — no forced tool, no tools offered.
    assert adapter.requests[0].force_tool is None
    assert adapter.requests[0].output_schema is not None


def test_judge_prose_omitted_criterion_is_fail_closed():
    # The model rules on index 0 but never mentions index 1 — that criterion is
    # treated as NOT met, so an incomplete verdict can't sneak a run to COMPLETE.
    graph = _judge_graph()
    adapter = FakeAdapter(responses=[_verdict([{"index": 0, "met": True}])])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["c0", "c1"], {}))
    assert out == {"c0": True, "c1": False}


def test_judge_prose_non_true_met_is_fail_closed():
    # Only an explicit boolean True passes. A truthy-but-not-True `met` (a
    # "true" STRING, or 1) must read as NOT met — otherwise Python truthiness
    # would fail open on garbled output.
    graph = _judge_graph()
    adapter = FakeAdapter(
        responses=[_verdict([{"index": 0, "met": True}, {"index": 1, "met": "true"}, {"index": 2, "met": 1}])]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["c0", "c1", "c2"], {}))
    assert out == {"c0": True, "c1": False, "c2": False}


def test_judge_prose_garbage_verdicts_value_fails_closed():
    # The shallow schema check only guarantees the 'verdicts' key exists, not
    # that it's a well-formed list. A non-list value must fail closed (every
    # criterion not met), never crash.
    graph = _judge_graph()
    adapter = FakeAdapter(responses=[_verdict("not-a-list")])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["c0"], {}))
    assert out == {"c0": False}


def test_judge_prose_unknown_evaluated_by_raises():
    graph = _judge_graph()
    gateway = LLMGateway(graph, {"anthropic": FakeAdapter(responses=[])})
    with pytest.raises(ValueError, match="unknown agent"):
        asyncio.run(gateway.judge_prose("nobody", ["c0"], {}))


def test_judge_prose_repairs_then_succeeds():
    graph = _judge_graph()
    bad = _verdict_call({"wrong": "shape"})  # missing the required 'verdicts' key — repairable
    good = _verdict([{"index": 0, "met": True}])
    adapter = FakeAdapter(responses=[bad, good])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    out = asyncio.run(gateway.judge_prose("pm", ["c0"], {}))
    assert out == {"c0": True}
    assert len(adapter.requests) == 2  # one repair round-trip


def test_judge_prose_exhausts_repairs_then_raises():
    # An adapter that never returns a valid verdict must not loop forever: after
    # guards.max_output_repairs it raises AgentOutputError (non-transient — the
    # DoD gate turns it into a fail-closed run).
    graph = _judge_graph()
    adapter = FakeAdapter(responses=[_verdict_call({"wrong": "shape"})])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    with pytest.raises(AgentOutputError):
        asyncio.run(gateway.judge_prose("pm", ["c0"], {}))
    assert len(adapter.requests) == 3  # initial + max_output_repairs (2)


def test_judge_prose_missing_tool_call_is_repaired_not_crashed():
    # A non-compliant provider that ignores the forced tool returns no
    # submit_verdict — treated as an invalid output (repairable), then exhausted.
    graph = _judge_graph()
    adapter = FakeAdapter(responses=[ProviderResponse(text="I refuse to use the tool", tool_calls=[])])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    with pytest.raises(AgentOutputError):
        asyncio.run(gateway.judge_prose("pm", ["c0"], {}))


# --- end-to-end gate through the gateway -----------------------------------
def _run_e2e(con, graph, node_turn: ProviderResponse, judgement: ProviderResponse) -> str:
    adapter = FakeAdapter(responses=[node_turn, judgement])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    return asyncio.run(
        start_run(
            con, graph, gateway, org_id="test", workflow_id=workflow_id,
            dod_prose_verdict=gateway.judge_prose,
        )
    )


def _status(con, run_id: str) -> str:
    return con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"]


def test_run_completes_when_agent_judges_prose_met(con):
    graph = _judge_graph(criteria=["All acceptance criteria are met"])
    run_id = _run_e2e(con, graph, _submit({"done": True}), _verdict([{"index": 0, "met": True}]))
    assert _status(con, run_id) == "COMPLETED"


def test_run_fails_when_agent_judges_prose_unmet(con):
    graph = _judge_graph(criteria=["All acceptance criteria are met"])
    run_id = _run_e2e(con, graph, _submit({"done": True}), _verdict([{"index": 0, "met": False}]))
    assert _status(con, run_id) == "FAILED"
