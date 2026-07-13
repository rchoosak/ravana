"""§3.1 step 7: the LLM Gateway's agent-backed prose Definition-of-Done judge
(`LLMGateway.judge_prose`) and its end-to-end gating. Covers the forced-tool
and guided paths, POSITION-ALIGNED verdict mapping and its fail-closed
defenses (omission, non-`true` `met`, garbage `verdicts`, bool/out-of-range/
duplicate index), token usage, the §3.6 fallback chain, the unknown-
`evaluated_by` config error, the repair loop / its exhaustion, and the whole
gate (engine → evaluate_dod → judge_prose → COMPLETE or FAIL). No network:
scripted FakeAdapters only, reusing test_gateway's fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.base import AgentOutputError, ProseJudgementError
from ravana.runtime.gateway import SUBMIT_VERDICT, LLMGateway
from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderResponse,
)
from ravana.schema.models import WorkflowDoc
from tests.conftest import RecordingSleep
from tests.test_gateway import FakeAdapter, _guided_text, _submit


def _judge_graph(*, provider: str = "anthropic", model: str = "m", criteria: list[str] | None = None, fallback=None):
    """A one-agent workflow whose evaluated_by agent is `pm`. When `criteria` is
    given it also carries a prose DoD (for the end-to-end gate tests)."""
    llm: dict[str, Any] = {"provider": provider, "model": model}
    if fallback is not None:
        llm["fallback"] = fallback
    spec: dict[str, Any] = {
        "state": {"schema": {"done": {"type": "boolean"}}, "initial": {}},
        "agents": [
            {
                "id": "pm",
                "name": "PM",
                "llm": llm,
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


def _verdict_call(args: dict[str, Any], **usage: int) -> ProviderResponse:
    """A submit_verdict tool call with arbitrary arguments (and optional token
    usage) — for the malformed / missing-key / usage cases."""
    return ProviderResponse(
        text="", tool_calls=[NormalizedToolCall(id="v", tool=SUBMIT_VERDICT, arguments=args)], **usage
    )


def _verdict(items: Any, **usage: int) -> ProviderResponse:
    return _verdict_call({"verdicts": items}, **usage)


def _judge(gateway: LLMGateway, criteria: list[str], state: dict[str, Any] | None = None):
    return asyncio.run(gateway.judge_prose("pm", criteria, state or {}))


# --- verdict mapping (position-aligned, fail-closed) -----------------------
def test_judge_prose_forced_tool_maps_verdicts_by_index():
    adapter = FakeAdapter(
        caps={Capability.NATIVE_STRUCTURED_OUTPUT},
        responses=[_verdict([{"index": 0, "met": True}, {"index": 1, "met": False}])],
    )
    gateway = LLMGateway(_judge_graph(), {"anthropic": adapter})
    out = _judge(gateway, ["c-zero", "c-one"], {"x": 1})
    assert out.verdicts == [True, False]  # position-aligned to the criteria
    # A judgement offers ONLY the submit_verdict tool (no toolkits) and forces it.
    req = adapter.requests[0]
    assert req.force_tool == SUBMIT_VERDICT
    assert [t.name for t in req.tools] == [SUBMIT_VERDICT]


def test_judge_prose_guided_path_parses_json_text():
    adapter = FakeAdapter(caps={Capability.GUIDED_DECODING}, responses=[_guided_text({"verdicts": [{"index": 0, "met": True}]})])
    gateway = LLMGateway(_judge_graph(provider="local", model="local-model"), {"local": adapter})
    out = _judge(gateway, ["only"])
    assert out.verdicts == [True]
    assert adapter.requests[0].force_tool is None and adapter.requests[0].output_schema is not None


def test_judge_prose_omitted_criterion_is_fail_closed():
    adapter = FakeAdapter(responses=[_verdict([{"index": 0, "met": True}])])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0", "c1"])
    assert out.verdicts == [True, False]  # c1 omitted -> not met


def test_judge_prose_non_true_met_is_fail_closed():
    # Only an explicit boolean True passes — a "true" STRING or 1 must not.
    adapter = FakeAdapter(responses=[_verdict([{"index": 0, "met": True}, {"index": 1, "met": "true"}, {"index": 2, "met": 1}])])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0", "c1", "c2"])
    assert out.verdicts == [True, False, False]


def test_judge_prose_garbage_verdicts_value_fails_closed():
    adapter = FakeAdapter(responses=[_verdict("not-a-list")])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [False]


def test_judge_prose_bool_index_is_rejected_not_read_as_zero():
    # bool is a subclass of int; index=false must NOT be read as criterion 0.
    adapter = FakeAdapter(responses=[_verdict([{"index": False, "met": True}])])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [False]  # the false-indexed ruling does not land on c0


def test_judge_prose_out_of_range_index_is_dropped():
    adapter = FakeAdapter(responses=[_verdict([{"index": 5, "met": True}])])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [False]


def test_judge_prose_duplicate_index_fails_closed():
    # Two rulings on the same index are unreliable; that criterion fails closed
    # even if one of them said met.
    adapter = FakeAdapter(responses=[_verdict([{"index": 0, "met": True}, {"index": 0, "met": False}])])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [False]


def test_judge_prose_carries_token_usage():
    adapter = FakeAdapter(responses=[_verdict([{"index": 0, "met": True}], input_tokens=30, output_tokens=8)])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [True] and out.usage.input_tokens == 30 and out.usage.output_tokens == 8


def test_judge_prose_usage_accumulates_across_repairs():
    # Tokens from an attempt that turned out invalid are NOT lost — the
    # judgement's usage is the sum across the whole logical judgement.
    bad = _verdict_call({"wrong": "shape"}, input_tokens=50, output_tokens=10)
    good = _verdict([{"index": 0, "met": True}], input_tokens=53, output_tokens=17)
    adapter = FakeAdapter(responses=[bad, good])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [True]
    assert out.usage.input_tokens == 103 and out.usage.output_tokens == 27


def test_judge_prose_multiple_submit_verdict_is_fail_closed():
    # Two submit_verdict calls in one response (first true, second false) must
    # NOT last-write-wins or take-first — it's ambiguous, so repair, then (here)
    # exhaust and raise. Never a silent [True].
    two = ProviderResponse(
        text="",
        tool_calls=[
            NormalizedToolCall(id="a", tool=SUBMIT_VERDICT, arguments={"verdicts": [{"index": 0, "met": True}]}),
            NormalizedToolCall(id="b", tool=SUBMIT_VERDICT, arguments={"verdicts": [{"index": 0, "met": False}]}),
        ],
    )
    adapter = FakeAdapter(responses=[two])
    with pytest.raises(ProseJudgementError) as ei:
        asyncio.run(LLMGateway(_judge_graph(), {"anthropic": adapter}).judge_prose("pm", ["c0"], {}))
    assert isinstance(ei.value.__cause__, AgentOutputError)


# --- config error, repair, fallback ----------------------------------------
def test_judge_prose_unknown_evaluated_by_raises():
    gateway = LLMGateway(_judge_graph(), {"anthropic": FakeAdapter(responses=[])})
    with pytest.raises(ValueError, match="unknown agent"):
        asyncio.run(gateway.judge_prose("nobody", ["c0"], {}))


def test_judge_prose_repairs_then_succeeds():
    bad = _verdict_call({"wrong": "shape"})  # missing the required 'verdicts' key — repairable
    good = _verdict([{"index": 0, "met": True}])
    adapter = FakeAdapter(responses=[bad, good])
    out = _judge(LLMGateway(_judge_graph(), {"anthropic": adapter}), ["c0"])
    assert out.verdicts == [True]
    assert len(adapter.requests) == 2  # one repair round-trip


def test_judge_prose_exhausts_repairs_then_raises():
    adapter = FakeAdapter(responses=[_verdict_call({"wrong": "shape"})])
    with pytest.raises(ProseJudgementError) as ei:
        asyncio.run(LLMGateway(_judge_graph(), {"anthropic": adapter}).judge_prose("pm", ["c0"], {}))
    assert isinstance(ei.value.__cause__, AgentOutputError)  # underlying cause preserved
    assert len(adapter.requests) == 3  # initial + max_output_repairs (2)


def test_judge_prose_failed_judgement_still_carries_usage():
    # A judgement that fails outright (repairs exhausted) still reports the
    # tokens it spent, so the engine can account for a failed judgement's cost.
    adapter = FakeAdapter(responses=[_verdict_call({"wrong": "shape"}, input_tokens=100, output_tokens=20)])
    with pytest.raises(ProseJudgementError) as ei:
        asyncio.run(LLMGateway(_judge_graph(), {"anthropic": adapter}).judge_prose("pm", ["c0"], {}))
    # 3 attempts (initial + 2 repairs), each 100/20 -> 300/60.
    assert ei.value.usage.input_tokens == 300 and ei.value.usage.output_tokens == 60


def test_judge_prose_missing_tool_call_is_repaired_not_crashed():
    adapter = FakeAdapter(responses=[ProviderResponse(text="I refuse to use the tool", tool_calls=[])])
    with pytest.raises(ProseJudgementError):
        asyncio.run(LLMGateway(_judge_graph(), {"anthropic": adapter}).judge_prose("pm", ["c0"], {}))


def test_judge_prose_uses_fallback_chain_on_transient_primary():
    # §3.6: a judgement survives a primary-provider outage the same way a node
    # does — the transient primary yields to a working fallback entry.
    graph = _judge_graph(fallback=[{"provider": "openai", "model": "gpt"}])
    primary = FakeAdapter(name="anthropic", fail=True, fail_retryable=True)
    fb = FakeAdapter(name="openai", responses=[_verdict([{"index": 0, "met": True}])])
    gateway = LLMGateway(graph, {"anthropic": primary, "openai": fb}, retry_sleep=RecordingSleep())
    out = _judge(gateway, ["c0"])
    assert out.verdicts == [True]  # fallback served the judgement


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
