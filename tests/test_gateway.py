"""LLM Gateway tests. Nothing here touches the network or needs an API key:
most tests script fake provider adapters, while the credential/SDK-boundary
probes use the REAL OpenAI/Anthropic adapter classes with only the SDK-client
constructor patched or failing before any request is built. Covers §3.4's
strategy selection, the submit_result contract, the within-turn tool loop,
the repair loop, §3.6's fallback chain + error taxonomy, and §1.4's
temperature normalization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

import yaml

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.base import AgentOutputError, TransientAgentError
from ravana.runtime.gateway import SUBMIT_RESULT, LLMGateway
from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    Tool,
)
from ravana.schema.loader import load_workflow_yaml
from ravana.schema.models import WorkflowDoc
from tests.conftest import SDLC_WORKFLOW, RecordingSleep


class _SurfacingExec:
    """Base for tool-executor fakes: surfaces one Tool per declared toolkit id,
    mirroring RavanaToolExecutor.tools_for so the gateway can offer them. Fakes
    add their own execute()."""

    def tools_for(self, toolkit_ids):
        return [Tool(name=t, description=f"fake {t}", input_schema={"type": "object"}) for t in toolkit_ids]




# A minimal single-agent workflow whose agent declares NO toolkits — used to
# exercise the one-shot guided path, which is only taken for toolkit-free
# agents (an agent WITH toolkits must run the tool loop instead).
_GUIDED_MIN_WORKFLOW = """
apiVersion: ravana/v1
kind: Workflow
metadata: { name: guided-min }
spec:
  agents:
    - id: solo
      name: Solo
      llm: { provider: local, model: local-model }
      system_prompt: "Return structured output."
      output_schema:
        type: object
        properties: { system_spec: { type: object } }
        required: [system_spec]
        additionalProperties: false
  graph:
    entry: n
    nodes: [{ id: n, agent: solo }]
    edges: []
"""


@pytest.fixture
def guided_graph():
    return compile_workflow(WorkflowDoc.model_validate(yaml.safe_load(_GUIDED_MIN_WORKFLOW)))


@dataclass
class FakeAdapter:
    """Scriptable provider adapter. `responses` is a list of ProviderResponse
    returned in order per complete() call; `caps` sets declared capabilities;
    `requests` records what the gateway sent (for assertions)."""

    name: str = "fake"
    caps: set[Capability] = field(default_factory=lambda: {Capability.NATIVE_STRUCTURED_OUTPUT})
    responses: list[ProviderResponse] = field(default_factory=list)
    fail: bool = False
    fail_retryable: bool = True  # False = permanent failure (auth/bad-request shaped)
    requests: list[ProviderRequest] = field(default_factory=list)
    _i: int = 0

    def capabilities(self, model: str) -> set[Capability]:
        return self.caps

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if self.fail:
            raise ProviderError(f"{self.name} is down", retryable=self.fail_retryable)
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

    class LoopExec(_SurfacingExec):
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


def test_guided_provider_reads_schema_json_from_text(guided_graph):
    # A guided-capable local model (strongest tier) gets output_schema passed
    # for grammar-constrained decoding, is NOT sent a submit_result tool, and
    # its schema JSON is read from message text — not from a tool call. The
    # guided one-shot path is only valid for a toolkit-free agent, so this uses
    # the minimal single-agent graph (see test_guided_agent_with_toolkits_*).
    adapter = FakeAdapter(
        caps={Capability.GUIDED_DECODING, Capability.NATIVE_STRUCTURED_OUTPUT},
        responses=[_guided_text({"system_spec": {"stack": "python"}})],
    )
    gateway = LLMGateway(guided_graph, {"local": adapter})
    result = _run(gateway, "solo")
    assert adapter.requests[0].output_schema is not None
    assert adapter.requests[0].force_tool is None
    assert adapter.requests[0].tools == []  # no submit_result tool in the guided path
    assert result.structured_payload == {"system_spec": {"stack": "python"}}


def test_guided_agent_with_toolkits_uses_tool_loop_not_guided(graph):
    # dev is on a guided-capable provider AND has toolkits. Guided one-shot
    # can't express tool calls, so the gateway MUST run the tool loop instead —
    # offering submit_result (not the guided read path). Otherwise dev would
    # silently never see its tools.
    adapter = FakeAdapter(
        caps={Capability.GUIDED_DECODING, Capability.NATIVE_STRUCTURED_OUTPUT},
        responses=[_submit({"system_spec": {}})],
    )
    gateway = LLMGateway(graph, {"local": adapter}, tool_executor=_SurfacingExec())
    result = _run(gateway, "dev")
    offered = {t.name for t in adapter.requests[0].tools}
    assert SUBMIT_RESULT in offered  # tool loop, not the guided one-shot
    assert {"code_interpreter", "git_connector"} <= offered  # dev's toolkits surfaced
    assert result.structured_payload == {"system_spec": {}}


def test_agent_toolkits_offered_alongside_submit_result(graph):
    # pm (anthropic, native forced-tool) declares toolkits=[web_search]; the
    # gateway must offer web_search AND submit_result to the model.
    adapter = FakeAdapter(responses=[_submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=_SurfacingExec())
    _run(gateway, "pm")
    offered = {t.name for t in adapter.requests[0].tools}
    assert offered == {"web_search", SUBMIT_RESULT}


def test_no_tool_executor_offers_only_submit_result(graph):
    # Without a real executor, _NoToolExecutor surfaces no tools — the turn is
    # submit_result-only even for an agent that declares toolkits.
    adapter = FakeAdapter(responses=[_submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter})  # default _NoToolExecutor
    _run(gateway, "pm")
    assert [t.name for t in adapter.requests[0].tools] == [SUBMIT_RESULT]


def test_toolkit_id_colliding_with_submit_result_is_rejected(graph):
    class Collide(_SurfacingExec):
        def tools_for(self, toolkit_ids):
            return [Tool(name=SUBMIT_RESULT, description="x", input_schema={"type": "object"})]

        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            return "x"

    adapter = FakeAdapter(responses=[_submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=Collide())
    with pytest.raises(ValueError, match="collides with the reserved submit tool"):
        _run(gateway, "pm")


def test_tool_not_granted_to_agent_is_refused_not_executed(graph):
    # §8/§916 per-node boundary: pm's toolkits are [web_search]. If the provider
    # calls git_connector (a tool registered for the run but NOT granted to pm),
    # the gateway must refuse it — never dispatch — and feed the model an error.
    executed: list[str] = []

    class RecordingExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            executed.append(tool)
            return "ran"

    adapter = FakeAdapter(
        responses=[
            ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="tc1", tool="git_connector", arguments={})]),
            _submit({"requirement_clarity": "HIGH"}),
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=RecordingExec())
    _run(gateway, "pm")
    assert executed == []  # the ungranted tool was never executed
    tool_result = next(m for m in adapter.requests[1].messages if m.role == "tool_result")
    assert "not available to this agent" in tool_result.content


def test_multi_call_response_is_capped_at_the_per_turn_budget(graph):
    # A single response carrying more tool calls than max_tool_calls_per_turn
    # must not fire side effects past the budget (P1b).
    cap = graph.doc.spec.graph.guards.max_tool_calls_per_turn
    executed: list[str] = []

    class RecordingExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            executed.append(tool)
            return "ran"

    many = ProviderResponse(
        text=None,
        tool_calls=[NormalizedToolCall(id=f"tc{i}", tool="web_search", arguments={"i": i}) for i in range(cap + 2)],
    )
    adapter = FakeAdapter(responses=[many, _submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=RecordingExec())
    result = _run(gateway, "pm")
    assert len(executed) == cap  # exactly the budget fired; the surplus was refused
    # P2a: the reported count is the EXECUTED count (== cap), not the inflated
    # attempt count (cap + 2) — so the engine's post-turn guard won't fail a run
    # the gateway already capped.
    assert result.tool_call_count == cap


def test_submit_alongside_tool_calls_defers_submit_and_runs_the_tool(graph):
    # P2b: a response carrying BOTH a real tool call and submit_result must not
    # accept the premature submit (the model hasn't seen the tool result yet) —
    # the tool runs, the submit is deferred with an "ignored" notice, and the
    # NEXT submit is honored. The real tool call must never be silently dropped.
    executed: list[str] = []

    class RecordingExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            executed.append(tool)
            return "tool output"

    adapter = FakeAdapter(
        responses=[
            ProviderResponse(
                text="doing both",
                tool_calls=[
                    NormalizedToolCall(id="tc1", tool="web_search", arguments={"q": "x"}),
                    NormalizedToolCall(id="tc2", tool=SUBMIT_RESULT, arguments={"requirement_clarity": "HIGH"}),
                ],
            ),
            _submit({"requirement_clarity": "HIGH"}),
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=RecordingExec())
    result = _run(gateway, "pm")
    assert executed == ["web_search"]  # the real tool ran; not dropped for the submit
    assert len(adapter.requests) == 2  # the mixed turn did NOT return — submit was deferred
    ignored = [m for m in adapter.requests[1].messages if m.role == "tool_result" and "submit_result ignored" in m.content]
    assert ignored  # the model was told its premature submit was ignored
    assert result.structured_payload == {"requirement_clarity": "HIGH"}


def test_multiple_mixed_submit_results_keep_the_transcript_balanced(graph):
    # A provider can emit more than one submit_result alongside a real tool
    # call. EVERY tool_use in the recorded assistant turn must get a matching
    # tool_result, or a strict provider rejects the next request. (Earlier only
    # the first submit_result was answered, leaving orphan tool_use blocks.)
    class RecordingExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            return "tool output"

    mixed = ProviderResponse(
        text="both, twice",
        tool_calls=[
            NormalizedToolCall(id="tc1", tool="web_search", arguments={"q": "x"}),
            NormalizedToolCall(id="tc2", tool=SUBMIT_RESULT, arguments={"requirement_clarity": "HIGH"}),
            NormalizedToolCall(id="tc3", tool=SUBMIT_RESULT, arguments={"requirement_clarity": "LOW"}),
        ],
    )
    adapter = FakeAdapter(responses=[mixed, _submit({"requirement_clarity": "HIGH"})])
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=RecordingExec())
    _run(gateway, "pm")

    second = adapter.requests[1].messages
    assistant = next(m for m in second if m.role == "assistant")
    tool_use_ids = {tc.id for tc in assistant.tool_calls}
    answered_ids = {m.tool_call_id for m in second if m.role == "tool_result"}
    assert tool_use_ids <= answered_ids  # every tool_use (incl. both submits) has a result


def test_model_addressable_tool_error_is_fed_back_not_raised(graph):
    # §3.6 taxonomy: a MODEL-ADDRESSABLE tool failure (404/422/bad args —
    # neither retryable nor fatal) is fed back as an error tool_result so the
    # model can adjust its call or route around it. It must NOT end the turn.
    from ravana.runtime.toolkits.base import ToolkitError

    class FailingExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            raise ToolkitError("HTTP 404 from /nope")  # default kind: MODEL_ADDRESSABLE

    adapter = FakeAdapter(
        responses=[
            ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="tc1", tool="web_search", arguments={})]),
            _submit({"requirement_clarity": "HIGH"}),
        ]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=FailingExec())
    result = _run(gateway, "pm")  # must not raise
    tool_result = next(m for m in adapter.requests[1].messages if m.role == "tool_result")
    assert "failed: HTTP 404" in tool_result.content
    assert result.structured_payload == {"requirement_clarity": "HIGH"}


def test_transient_tool_error_ends_the_turn_as_transient(graph):
    # §3.6 lists "tool timeout" as TRANSIENT: a retryable ToolkitError must end
    # the turn as TransientAgentError so the ENGINE retries a fresh
    # node_execution attempt with backoff — not be silently fed back with no
    # backoff anywhere.
    from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError

    class TimeoutExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            raise ToolkitError("request timed out", kind=ToolFailureKind.TRANSIENT)

    adapter = FakeAdapter(
        responses=[ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="tc1", tool="web_search", arguments={})])]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=TimeoutExec())
    with pytest.raises(TransientAgentError, match="failed transiently"):
        _run(gateway, "pm")


def test_fatal_tool_error_is_a_hard_failure(graph):
    # §3.6 lists "tool auth failure" as NON-transient: a fatal ToolkitError
    # (401/403) fails the run — neither fed back nor retried.
    from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError

    class AuthFailExec(_SurfacingExec):
        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            raise ToolkitError("HTTP 401 unauthorized", kind=ToolFailureKind.FATAL)

    adapter = FakeAdapter(
        responses=[ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="tc1", tool="web_search", arguments={})])]
    )
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=AuthFailExec())
    with pytest.raises(RuntimeError, match="failed fatally"):
        _run(gateway, "pm")


def _single_node_gateway_graph(output_schema: dict | None = None, toolkits: list[str] | None = None):
    """Minimal one-node workflow compiled for engine+gateway e2e tests."""
    spec: dict = {
        "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
        "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
    }
    if output_schema is not None:
        spec["agents"][0]["output_schema"] = output_schema
    if toolkits:
        spec["toolkits"] = [{"id": t, "type": "web_search"} for t in toolkits]
        spec["agents"][0]["toolkits"] = toolkits
    doc = WorkflowDoc.model_validate(
        {"apiVersion": "ravana/v1", "kind": "Workflow", "metadata": {"name": "gw-e2e", "version": 1}, "spec": spec}
    )
    return compile_workflow(doc)


def test_engine_retries_a_transient_tool_failure_with_backoff(con):
    # End-to-end §3.6: a tool timeout fails the attempt transiently; the engine
    # backs off, dispatches a NEW attempt, and the retry succeeds.
    from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError

    class FlakyToolExec(_SurfacingExec):
        calls = 0

        async def execute(self, *, run_id, node_id, tool, arguments, idempotency_key):
            self.calls += 1
            if self.calls == 1:
                raise ToolkitError("connect timeout", kind=ToolFailureKind.TRANSIENT)
            return "recovered"

    graph = _single_node_gateway_graph(toolkits=["web_search"])
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    calls_tool = ProviderResponse(text=None, tool_calls=[NormalizedToolCall(id="t1", tool="web_search", arguments={"q": "x"})])
    # Attempt 1 consumes response[0] and dies at the tool; attempt 2 replays
    # the tool call (response[1], tool now recovers) then submits (response[2]).
    adapter = FakeAdapter(name="anthropic", responses=[calls_tool, calls_tool, _submit({"done": True})])
    sleeper = RecordingSleep()
    executor = FlakyToolExec()
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=executor, retry_sleep=sleeper)

    run_id = asyncio.run(
        start_run(con, graph, gateway, org_id="test", workflow_id=workflow_id, retry_sleep=sleeper)
    )
    attempts = con.execute(
        "SELECT attempt, status FROM node_execution WHERE run_id = ? ORDER BY attempt", (run_id,)
    ).fetchall()
    assert [r["status"] for r in attempts] == ["FAILED", "SUCCEEDED"]  # a NEW attempt, per §3.6
    assert executor.calls == 2  # the tool actually re-ran on the retry
    sleeper.assert_delays(1.0)  # and the engine backed off between attempts
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"


def test_repair_exhaustion_fails_the_run_without_node_retries(con):
    # §3.6 "repair budget exhausted" is NON-transient: the run FAILs on the
    # first node_execution — no backoff, no extra attempts, and the model is
    # not called beyond the repair budget.
    graph = _single_node_gateway_graph(
        output_schema={"type": "object", "properties": {"verdict": {"type": "string", "enum": ["OK"]}}, "required": ["verdict"], "additionalProperties": False}
    )
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    max_repairs = graph.doc.spec.graph.guards.max_output_repairs
    adapter = FakeAdapter(name="anthropic", responses=[_submit({"verdict": "NOPE"})])  # invalid enum, forever
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"anthropic": adapter}, retry_sleep=sleeper)

    run_id = asyncio.run(
        start_run(con, graph, gateway, org_id="test", workflow_id=workflow_id, retry_sleep=sleeper)
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    rows = con.execute("SELECT COUNT(*) c FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()["c"]
    assert rows == 1  # no node retries for exhausted repairs
    assert sleeper.delays == []  # and no backoff spent
    assert len(adapter.requests) == max_repairs + 1  # model called exactly budget+1 times, never more


def test_non_compliant_provider_that_never_submits_is_bounded(graph):
    # A provider that keeps answering with text and never submits (ignoring the
    # forced submit_result) must NOT loop forever — the gateway bounds total
    # model round-trips and fails the turn. §3.6 "guard exceeded" is
    # NON-transient: the same provider would ignore the forced submit again on
    # a retried attempt, so this is AgentOutputError, not TransientAgentError.
    never_submits = ProviderResponse(text="still thinking, no tool call", tool_calls=[])
    adapter = FakeAdapter(responses=[never_submits])  # returned repeatedly
    gateway = LLMGateway(graph, {"anthropic": adapter}, tool_executor=_SurfacingExec())
    with pytest.raises(AgentOutputError, match="did not produce a valid submit_result"):
        _run(gateway, "pm")


def test_within_turn_tool_loop_then_submit(graph):
    seen: list[tuple[str, str]] = []

    class Exec(_SurfacingExec):
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
    class Exec(_SurfacingExec):
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
    # max_output_repairs is 2 in the SDLC example; 3 invalid submits exhausts
    # it. §3.6: repair exhaustion is NON-transient (AgentOutputError, not
    # TransientAgentError) — the engine must fail the run, not retry the node
    # past a budget that already expired.
    adapter = FakeAdapter(responses=[_submit({"qa_status": "NOPE", "qa_report": {}})])
    gateway = LLMGateway(graph, {"openai": adapter})
    with pytest.raises(AgentOutputError, match="failed validation"):
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
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    result = _run(gateway, "dev")
    assert result.structured_payload == {"system_spec": {}}  # primary's retry, not the fallback
    assert len(fallback.requests) == 0  # fallback never used
    # §3.6: the same-entry retry backed off once (first-failure band, base=1s).
    sleeper.assert_delays(1.0)


def test_fallback_chain_used_when_primary_fails(graph):
    # dev has llm.fallback = [{anthropic, claude-sonnet-5}]; make the primary
    # (local) fail and confirm the run still succeeds via the fallback.
    primary = FakeAdapter(name="local", fail=True)
    fallback = FakeAdapter(name="anthropic", responses=[_submit({"system_spec": {}})])
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    result = _run(gateway, "dev")
    assert result.structured_payload == {"system_spec": {}}
    assert len(fallback.requests) == 1  # fallback actually served the turn
    # Only the primary's own retry backed off; SWITCHING to the fallback did
    # not add a sleep — a different provider needs no breathing room.
    assert len(sleeper.delays) == 1


def test_all_entries_exhausted_raises(graph):
    primary = FakeAdapter(name="local", fail=True)
    fallback = FakeAdapter(name="anthropic", fail=True)
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    with pytest.raises(TransientAgentError, match="all LLM entries exhausted"):
        _run(gateway, "dev")
    # One backoff per entry's internal retry (2 entries), none on chain moves.
    assert len(sleeper.delays) == 2


# --- §3.6 error taxonomy: retryable vs permanent -----------------------------
def test_classify_retryable_by_status_code():
    from ravana.runtime.providers.base import classify_retryable

    class FakeSdkStatusError(Exception):
        """Shaped like both SDKs' API errors: carries an HTTP status_code."""

        def __init__(self, status_code=None):
            self.status_code = status_code

    assert classify_retryable(FakeSdkStatusError(429)) is True  # rate limit: breathing room helps
    assert classify_retryable(FakeSdkStatusError(500)) is True
    assert classify_retryable(FakeSdkStatusError(503)) is True
    assert classify_retryable(FakeSdkStatusError(408)) is True  # request timeout
    assert classify_retryable(FakeSdkStatusError()) is True  # no status (connection blip): availability bias
    assert classify_retryable(FakeSdkStatusError(401)) is False  # auth: a retry can't fix credentials
    assert classify_retryable(FakeSdkStatusError(400)) is False  # invalid request
    assert classify_retryable(FakeSdkStatusError(404)) is False  # unknown model
    assert classify_retryable(FakeSdkStatusError(422)) is False
    # Builtin programming/config errors carry no status_code but a retry
    # re-runs the same broken config — permanent (the anthropic SDK surfaces a
    # missing credential as TypeError; review probe caught it classified
    # transient and wasting a backoff).
    assert classify_retryable(TypeError("could not resolve authentication")) is False
    assert classify_retryable(ValueError("bad config")) is False
    assert classify_retryable(KeyError("missing")) is False
    assert classify_retryable(AttributeError("nope")) is False


def _fallback_chain_graph(name: str, primary_provider: str, fallback_provider: str):
    """Single-node workflow whose agent runs on `primary_provider` (hosted
    shape: no endpoint, no api_key_ref) with one fallback entry — the shared
    testbed for the credential/SDK-boundary probes."""
    doc = WorkflowDoc.model_validate(
        {
            "apiVersion": "ravana/v1",
            "kind": "Workflow",
            "metadata": {"name": name, "version": 1},
            "spec": {
                "agents": [
                    {
                        "id": "a",
                        "name": "A",
                        "llm": {"provider": primary_provider, "model": "m", "fallback": [{"provider": fallback_provider, "model": "m2"}]},
                        "system_prompt": "p",
                    }
                ],
                "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
            },
        }
    )
    return compile_workflow(doc)


def _assert_falls_back_cleanly(cred_graph, primary, primary_name: str, fallback_name: str):
    """Shared probe assertion: the primary's client-init failure normalizes,
    the fallback serves the turn, and no same-entry backoff is wasted."""
    fallback = FakeAdapter(name=fallback_name, responses=[_submit({"done": True})])
    sleeper = RecordingSleep()
    gateway = LLMGateway(cred_graph, {primary_name: primary, fallback_name: fallback}, retry_sleep=sleeper)
    result = _run(gateway, "a")
    assert result.structured_payload == {"done": True}
    assert len(fallback.requests) == 1  # the fallback actually ran
    assert sleeper.delays == []  # permanent: no same-entry backoff wasted


def test_openai_missing_credential_is_normalized_and_falls_back(monkeypatch):
    # Review probe: with no api_key anywhere, AsyncOpenAI() raises OpenAIError
    # at client construction — previously OUTSIDE the normalization boundary,
    # so the raw SDK error skipped the §3.6 fallback chain entirely (fallback
    # called 0 times). Now it normalizes to ProviderError(retryable=False).
    # (No network: the failure happens before any request is built.)
    from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    graph = _fallback_chain_graph("openai-cred-test", "openai", "anthropic")
    _assert_falls_back_cleanly(graph, OpenAICompatibleAdapter(name="openai"), "openai", "anthropic")


def test_anthropic_client_init_failure_is_normalized_and_falls_back(monkeypatch):
    # Review probe: AnthropicAdapter built its SDK client eagerly in __init__,
    # BEFORE any normalization boundary — a malformed proxy/credential config
    # escaped as a raw ValueError and the fallback chain never ran. The client
    # is now built lazily inside complete()'s boundary.
    import anthropic

    from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter

    def exploding_client(*args, **kwargs):
        raise ValueError("malformed proxy configuration")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", exploding_client)
    graph = _fallback_chain_graph("anthropic-init-test", "anthropic", "openai")
    _assert_falls_back_cleanly(graph, AnthropicAdapter(), "anthropic", "openai")


def test_sdk_internal_retries_are_disabled(monkeypatch):
    # Review finding: both SDKs default to max_retries=2, which STACKS with the
    # gateway's per-entry retry (up to 6 HTTP attempts per entry) and sleeps
    # outside the injected backoff waiter. The gateway owns retry policy —
    # clients must be constructed with max_retries=0.
    import anthropic
    import openai

    from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
    from ravana.runtime.providers.base import UserMessage
    from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

    made: dict[str, dict[str, Any]] = {}

    class FakeAnthropicClient:
        def __init__(self, **kwargs):
            made["anthropic"] = kwargs

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            made["openai"] = kwargs

    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropicClient)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAIClient)

    AnthropicAdapter()._resolve_client()
    OpenAICompatibleAdapter(name="local")._resolve_client(
        ProviderRequest(model="m", system="", messages=[UserMessage(text="x")], endpoint="http://a/v1")
    )
    assert made["anthropic"]["max_retries"] == 0
    assert made["openai"]["max_retries"] == 0


def test_entry_terminal_outcome_decides_transient_vs_hard(graph):
    # Review probe: a historical retryable failure must not mask a chain that
    # ENDED all-permanent. Primary fails 500 (retryable) then 401 (permanent)
    # on its retry; fallback fails permanent. Every entry's TERMINAL outcome is
    # permanent => hard error, not TransientAgentError (which would burn
    # engine node retries re-running a hopeless chain).
    class RetryableThenPermanent:
        name = "local"
        calls = 0

        def capabilities(self, model):
            return {Capability.NATIVE_STRUCTURED_OUTPUT}

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("500 server error", retryable=True)
            raise ProviderError("401 unauthorized", retryable=False)

    primary = RetryableThenPermanent()
    fallback = FakeAdapter(name="anthropic", fail=True, fail_retryable=False)
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    with pytest.raises(RuntimeError, match="failed permanently"):
        _run(gateway, "dev")
    assert primary.calls == 2  # the retryable failure did get its one retry
    sleeper.assert_delays(1.0)  # ...with its backoff — but the OUTCOME stayed permanent


def test_permanent_error_skips_same_entry_retry_and_goes_to_fallback(graph):
    # An auth-shaped (permanent) primary failure must NOT be retried at the
    # same entry — no backoff sleep — and the chain moves straight to the
    # fallback, which serves the turn.
    primary = FakeAdapter(name="local", fail=True, fail_retryable=False)
    fallback = FakeAdapter(name="anthropic", responses=[_submit({"system_spec": {}})])
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    result = _run(gateway, "dev")
    assert result.structured_payload == {"system_spec": {}}
    assert len(primary.requests) == 1  # exactly one try: permanent means no same-entry retry
    assert sleeper.delays == []  # and no backoff was spent on it


def test_all_permanent_chain_raises_hard_error_not_transient(graph):
    # Every entry failing permanently (auth/config) must NOT surface as
    # TransientAgentError — the engine would burn max_retries_per_node
    # re-running a hopeless chain. It surfaces as a hard error instead.
    primary = FakeAdapter(name="local", fail=True, fail_retryable=False)
    fallback = FakeAdapter(name="anthropic", fail=True, fail_retryable=False)
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=sleeper)
    with pytest.raises(RuntimeError, match="failed permanently"):
        _run(gateway, "dev")
    assert sleeper.delays == []  # nothing to wait for anywhere in the chain


def test_mixed_chain_with_any_retryable_failure_stays_transient(graph):
    # Primary fails permanently, fallback fails transiently: the turn is still
    # transient (the engine's own retry may recover), not a hard error.
    primary = FakeAdapter(name="local", fail=True, fail_retryable=False)
    fallback = FakeAdapter(name="anthropic", fail=True, fail_retryable=True)
    gateway = LLMGateway(graph, {"local": primary, "anthropic": fallback}, retry_sleep=RecordingSleep())
    with pytest.raises(TransientAgentError, match="all LLM entries exhausted"):
        _run(gateway, "dev")


def test_all_permanent_chain_fails_the_run_without_node_retries(con):
    # Engine-level consequence: a permanently-misconfigured agent fails the
    # run on the FIRST node_execution — no §3.6 node retries, no backoff.
    graph = compile_workflow(load_workflow_yaml(SDLC_WORKFLOW))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    adapter = FakeAdapter(name="anthropic", fail=True, fail_retryable=False)
    sleeper = RecordingSleep()
    gateway = LLMGateway(graph, {p: adapter for p in ("anthropic", "local", "openai")}, retry_sleep=sleeper)

    run_id = asyncio.run(
        start_run(con, graph, gateway, org_id="test", workflow_id=workflow_id, input_payload={"repository": "r"}, retry_sleep=sleeper)
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    rows = con.execute("SELECT COUNT(*) c FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()["c"]
    assert rows == 1  # failed immediately: no wrong-type node retries
    assert sleeper.delays == []  # and zero backoff spent anywhere


def test_no_tool_executor_wired_is_a_hard_error_not_provider_fault():
    # Defense-in-depth taxonomy: _NoToolExecutor.execute is a WIRING bug, so it
    # raises a hard error — NOT ProviderError, which the fallback loop would
    # retry/fall through (burning full LLM turns per pass). Note it is normally
    # unreachable through the tool loop: _NoToolExecutor surfaces zero tools,
    # so the per-node boundary refuses any tool call before execute — this
    # pins the taxonomy for whatever future path reaches it directly.
    from ravana.runtime.gateway import _NoToolExecutor

    with pytest.raises(RuntimeError, match="no ToolExecutor is wired"):
        asyncio.run(_NoToolExecutor().execute(run_id="r", node_id="n", tool="x", arguments={}, idempotency_key="k"))


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


def test_hosted_openai_omits_api_key_so_sdk_reads_env():
    # P1c: hosted OpenAI (no endpoint, no api_key_ref) must NOT get a dummy key
    # — passing one blocks the SDK's OPENAI_API_KEY fallback and misauthenticates.
    from ravana.runtime.providers.base import UserMessage
    from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

    made: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    import openai

    orig = openai.AsyncOpenAI
    openai.AsyncOpenAI = FakeClient
    try:
        adapter = OpenAICompatibleAdapter(name="openai")
        adapter._resolve_client(ProviderRequest(model="gpt-4o", system="", messages=[UserMessage(text="x")]))
    finally:
        openai.AsyncOpenAI = orig

    assert len(made) == 1
    assert "api_key" not in made[0]  # SDK falls back to OPENAI_API_KEY
    assert "base_url" not in made[0]  # hosted default, no override


def test_local_runtime_without_key_gets_placeholder():
    # A local endpoint with no api_key_ref still needs a nonempty key for the
    # SDK; the placeholder is supplied only in that case (contrast P1c).
    from ravana.runtime.providers.base import UserMessage
    from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

    made: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    import openai

    orig = openai.AsyncOpenAI
    openai.AsyncOpenAI = FakeClient
    try:
        adapter = OpenAICompatibleAdapter(name="local")
        adapter._resolve_client(
            ProviderRequest(model="m", system="", messages=[UserMessage(text="x")], endpoint="http://localhost:11434/v1")
        )
    finally:
        openai.AsyncOpenAI = orig

    assert made[0]["base_url"] == "http://localhost:11434/v1"
    assert made[0]["api_key"] == "not-needed-for-local"


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
