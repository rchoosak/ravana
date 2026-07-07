"""LLM Gateway tests — all against fake provider adapters, so nothing here
touches the network or needs an API key. Covers §3.4's strategy selection,
the submit_result contract, the within-turn tool loop, the repair loop,
§3.6's fallback chain, and §1.4's temperature normalization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.runtime.base import TransientAgentError
from ravana.runtime.gateway import SUBMIT_RESULT, LLMGateway
from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)
from ravana.schema.loader import load_workflow_yaml
from tests.conftest import SDLC_WORKFLOW


@dataclass
class FakeAdapter:
    """Scriptable provider adapter. `responses` is a list of ProviderResponse
    returned in order per complete() call; `caps` sets declared capabilities;
    `requests` records what the gateway sent (for assertions)."""

    name: str = "fake"
    caps: set[Capability] = field(default_factory=lambda: {Capability.NATIVE_STRUCTURED_OUTPUT})
    responses: list[ProviderResponse] = field(default_factory=list)
    fail: bool = False
    requests: list[ProviderRequest] = field(default_factory=list)
    _i: int = 0

    def capabilities(self, model: str) -> set[Capability]:
        return self.caps

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.fail:
            raise ProviderError(f"{self.name} is down")
        resp = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return resp


def _submit(payload: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse(text="done", tool_calls=[NormalizedToolCall(id="t1", tool=SUBMIT_RESULT, arguments=payload)], output_tokens=5)


def _guided_text(payload: dict[str, Any]) -> ProviderResponse:
    """Guided decoding returns the schema-conformant JSON as message text, not
    a tool call — this is what the gateway's guided path reads."""
    import json as _json

    return ProviderResponse(text=_json.dumps(payload), tool_calls=[], output_tokens=5)


@pytest.fixture
def graph():
    return compile_workflow(load_workflow_yaml(SDLC_WORKFLOW))


def _run(gateway: LLMGateway, agent_id: str, state: dict[str, Any] | None = None):
    return asyncio.run(
        gateway.run_turn(run_id="r1", node_id="n1", attempt=1, agent_id=agent_id, shared_state=state or {})
    )


def test_submit_result_becomes_structured_payload(graph):
    adapter = FakeAdapter(responses=[_submit({"requirement_clarity": "HIGH", "milestone_plan": {}})])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    result = _run(gateway, "pm")
    assert result.structured_payload == {"requirement_clarity": "HIGH", "milestone_plan": {}}
    assert result.content == "done"
    assert result.output_tokens == 5


def test_native_provider_offers_but_does_not_force_on_normal_turn(graph):
    adapter = FakeAdapter(caps={Capability.NATIVE_STRUCTURED_OUTPUT}, responses=[_submit({"system_spec": {}})])
    gateway = LLMGateway(graph, {"anthropic": adapter})
    _run(gateway, "sa")
    # §3.4.4: submit_result is offered as a tool, but tool_choice is NOT forced
    # on a normal turn — forcing every turn would discard the model's free-text
    # reasoning (§3.4.2). Forcing is reserved for the budget-exhaustion escape.
    assert adapter.requests[0].tools[0].name == SUBMIT_RESULT
    assert adapter.requests[0].force_tool is None


def test_free_text_reasoning_captured_alongside_structured_payload(graph):
    # §3.4.2: reasoning -> message.content, schema object -> structured_payload.
    # A normal (unforced) turn returns both.
    adapter = FakeAdapter(
        responses=[
            ProviderResponse(
                text="I reviewed the spec and it's clear.",
                tool_calls=[NormalizedToolCall(id="t1", tool=SUBMIT_RESULT, arguments={"requirement_clarity": "HIGH"})],
            )
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter})
    result = _run(gateway, "pm")
    assert result.content == "I reviewed the spec and it's clear."
    assert result.structured_payload == {"requirement_clarity": "HIGH"}


def test_budget_exhaustion_forces_submit_result(graph):
    # A model that keeps calling a real tool and never submits must eventually
    # be hard-forced to submit_result once max_tool_calls_per_turn is hit.
    guards = graph.doc.spec.graph.guards
    cap = guards.max_tool_calls_per_turn

    class LoopExec:
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            return "keep going"

    # Every response is a real (non-submit) tool call, except the fake's last
    # entry which is a submit — but the gateway should force submit via
    # force_tool before it ever relies on that, once the budget is spent.
    keep_calling = ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="tc", tool="web_search", arguments={})])
    adapter = FakeAdapter(responses=[keep_calling] * (cap) + [_submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=LoopExec())
    result = _run(gateway, "pm")
    assert result.structured_payload == {"requirement_clarity": "HIGH"}
    # The request made once the budget was exhausted must have forced submit_result.
    assert any(r.force_tool == SUBMIT_RESULT for r in adapter.requests)


def test_guided_provider_reads_schema_json_from_text(graph):
    # A guided-capable local model (strongest tier) gets output_schema passed
    # for grammar-constrained decoding, is NOT sent a submit_result tool, and
    # its schema JSON is read from message text — not from a tool call.
    adapter = FakeAdapter(
        caps={Capability.GUIDED_DECODING, Capability.NATIVE_STRUCTURED_OUTPUT},
        responses=[_guided_text({"system_spec": {"stack": "python"}})],
    )
    gateway = LLMGateway(graph, {"local": adapter})
    result = _run(gateway, "dev")  # dev is provider: local
    assert adapter.requests[0].output_schema is not None
    assert adapter.requests[0].force_tool is None
    assert adapter.requests[0].tools == []  # no submit_result tool in the guided path
    assert result.structured_payload == {"system_spec": {"stack": "python"}}


def test_within_turn_tool_loop_then_submit(graph):
    seen: list[tuple[str, str]] = []

    class Exec:
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            # §3.6: the key is present at execution time so a side-effecting
            # connector can dedupe — recording it here proves it arrived before
            # the side effect, not after.
            seen.append((tool, idempotency_key))
            return "tool output"

    adapter = FakeAdapter(
        responses=[
            ProviderResponse(text="thinking", tool_calls=[NormalizedToolCall(id="tc1", tool="web_search", arguments={"q": "x"})]),
            _submit({"requirement_clarity": "HIGH"}),
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=Exec())
    result = _run(gateway, "pm")
    assert [t for t, _ in seen] == ["web_search"]  # the real tool ran before submit_result
    assert result.tool_call_count == 1
    assert result.structured_payload == {"requirement_clarity": "HIGH"}
    # The executed tool is recorded WITH the same content-addressed key that
    # was passed to execute (§3.6) — one key, computed once, before execution.
    from ravana.runtime.idempotency import compute_idempotency_key

    expected_key = compute_idempotency_key("r1", "n1", "web_search", {"q": "x"})
    assert seen == [("web_search", expected_key)]
    assert result.tool_calls == [{"tool": "web_search", "arguments": {"q": "x"}, "idempotency_key": expected_key}]


def test_second_model_turn_sees_normalized_transcript(graph):
    # After a tool call, the next request's transcript must carry (a) the
    # assistant's tool_calls turn and (b) the tool_result — normalized shapes
    # the adapter later translates per-provider. Without the assistant turn,
    # real OpenAI rejects the tool result (P2 finding).
    class Exec:
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            return "result-text"

    adapter = FakeAdapter(
        responses=[
            ProviderResponse(text="calling a tool", tool_calls=[NormalizedToolCall(id="tc1", tool="web_search", arguments={})]),
            _submit({"requirement_clarity": "HIGH"}),
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=Exec())
    _run(gateway, "pm")
    # Second request (index 1) is what the gateway sent after the tool ran.
    second = adapter.requests[1].messages
    roles = [m.role for m in second]
    assert "assistant" in roles and "tool_result" in roles
    assistant = next(m for m in second if m.role == "assistant")
    assert assistant.tool_calls[0].tool == "web_search"
    tool_result = next(m for m in second if m.role == "tool_result")
    assert tool_result.tool_call_id == "tc1" and tool_result.content == "result-text"


def test_repair_loop_on_invalid_output(graph):
    # QA's output_schema requires qa_status enum PASS|FAIL. First submit is
    # invalid (bad enum), gateway re-prompts, second is valid.
    adapter = FakeAdapter(
        responses=[
            _submit({"qa_status": "MAYBE", "qa_report": {}}),
            _submit({"qa_status": "PASS", "qa_report": {}}),
        ]
    )
    gateway = LLMGateway(graph, {"openai": adapter})
    result = _run(gateway, "qa")
    assert result.structured_payload["qa_status"] == "PASS"
    assert result.repair_count == 1
    assert len(adapter.requests) == 2  # original + one repair


def test_repair_budget_exhausted_fails(graph):
    # max_output_repairs is 2 in the SDLC example; 3 invalid submits exhausts it.
    adapter = FakeAdapter(responses=[_submit({"qa_status": "NOPE", "qa_report": {}})])
    gateway = LLMGateway(graph, {"openai": adapter})
    with pytest.raises(TransientAgentError, match="failed validation"):
        _run(gateway, "qa")


def test_primary_retried_once_before_falling_back(graph):
    # §3.6: each entry gets its own small retry budget (1). A primary that
    # fails once then succeeds must recover on its own retry, never reaching
    # the fallback.
    class FlakyThenOK:
        name = "local"
        _calls = 0

        def capabilities(self, model):
            return {Capability.NATIVE_STRUCTURED_OUTPUT}

        async def complete(self, request):
            self._calls += 1
            if self._calls == 1:
                raise ProviderError("transient blip")
            return _submit({"system_spec": {}})

    primary = FlakyThenOK()
    fallback = FakeAdapter(name="anthropic", responses=[_submit({"system_spec": {"via": "fallback"}})])
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback})
    result = _run(gateway, "dev")
    assert result.structured_payload == {"system_spec": {}}  # primary's retry, not the fallback
    assert len(fallback.requests) == 0  # fallback never used


def test_fallback_chain_used_when_primary_fails(graph):
    # dev has llm.fallback = [{anthropic, claude-sonnet-5}]; make the primary
    # (local) fail and confirm the run still succeeds via the fallback.
    primary = FakeAdapter(name="local", fail=True)
    fallback = FakeAdapter(name="anthropic", responses=[_submit({"system_spec": {}})])
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback})
    result = _run(gateway, "dev")
    assert result.structured_payload == {"system_spec": {}}
    assert len(fallback.requests) == 1  # fallback actually served the turn


def test_all_entries_exhausted_raises(graph):
    primary = FakeAdapter(name="local", fail=True)
    fallback = FakeAdapter(name="anthropic", fail=True)
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback})
    with pytest.raises(TransientAgentError, match="all LLM entries exhausted"):
        _run(gateway, "dev")


def test_openai_adapter_caches_client_per_endpoint():
    # P2: one adapter instance serving agents at different endpoints must not
    # reuse the first-resolved client for all of them.
    from ravana.runtime.providers.base import UserMessage
    from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

    made: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    # Patch the lazily-imported AsyncOpenAI symbol the adapter constructs.
    import openai

    orig = openai.AsyncOpenAI
    openai.AsyncOpenAI = FakeClient
    try:
        adapter = OpenAICompatibleAdapter(name="local")  # no injected client -> lazy per-endpoint construction
        adapter._resolve_client(ProviderRequest(model="m", system="", messages=[UserMessage(text="x")], endpoint="http://a:11434/v1", api_key_ref="k1"))
        adapter._resolve_client(ProviderRequest(model="m", system="", messages=[UserMessage(text="x")], endpoint="http://b:11434/v1", api_key_ref="k2"))
        adapter._resolve_client(ProviderRequest(model="m", system="", messages=[UserMessage(text="x")], endpoint="http://a:11434/v1", api_key_ref="k1"))
    finally:
        openai.AsyncOpenAI = orig

    # Two distinct endpoints -> two clients; the repeat of endpoint a reuses.
    assert len(made) == 2
    assert {m["base_url"] for m in made} == {"http://a:11434/v1", "http://b:11434/v1"}


def test_temperature_dropped_for_no_sampling_param_models():
    from ravana.runtime.providers.anthropic_adapter import _accepts_temperature

    assert not _accepts_temperature("claude-opus-4-8")
    assert not _accepts_temperature("claude-sonnet-5")
    assert not _accepts_temperature("claude-fable-5")
    assert _accepts_temperature("claude-3-haiku-20240307")  # older model still accepts it


def test_prompt_assembler_includes_skills_and_state(graph):
    adapter = FakeAdapter(responses=[_submit({"system_spec": {}})])
    gateway = LLMGateway(graph, {"local": adapter})
    _run(gateway, "dev", state={"requirement": "build X"})
    system = adapter.requests[0].system
    # dev has skills conventional_commits + secure_coding_checklist (§4);
    # the assembler injects each skill's description + instructions.
    assert "Conventional Commits" in system
    assert "Baseline security checks" in system
    # shared_state injected as context
    assert "build X" in system
