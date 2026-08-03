"""Opt-in live-provider smoke tests for the structured-output contract (§3.4).

The offline tests in this module always run. They prove that the real adapter
capabilities drive the gateway to the expected request shape without touching
the network. The live tests additionally execute a minimal compiled workflow
through the engine and LLMGateway, and are skipped unless explicitly enabled:

    RAVANA_LIVE_SMOKE=1 ANTHROPIC_API_KEY=sk-... uv run pytest tests/test_live_smoke.py
    RAVANA_LIVE_SMOKE=1 OPENAI_API_KEY=sk-... uv run pytest tests/test_live_smoke.py

For a guided-decoding smoke test against a local Ollama/vLLM endpoint, both
the endpoint and its model name are required:

    RAVANA_LIVE_SMOKE=1 \
      RAVANA_LIVE_OPENAI_ENDPOINT=http://localhost:11434/v1 \
      RAVANA_LIVE_OPENAI_MODEL=qwen2.5-coder:7b \
      uv run pytest tests/test_live_smoke.py

Each live run sends only an instruction to return {"answer": "ping"}; no
repository content leaves the machine. Adapters are closed even when a live
assertion fails.
"""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from ravana.compiler.graph import CompiledGraph, compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.gateway import SUBMIT_RESULT, LLMGateway
from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
    ProviderTarget,
)
from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter
from ravana.schema.models import WorkflowDoc
from ravana.schema.util import loads

_SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["ping"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

_LIVE = pytest.mark.skipif(
    os.environ.get("RAVANA_LIVE_SMOKE") != "1",
    reason="live-API smoke test; set RAVANA_LIVE_SMOKE=1 to run",
)


class _RecordingAdapter:
    """Use a real adapter's capabilities while returning an offline response."""

    def __init__(self, delegate: ProviderAdapter, response: ProviderResponse):
        self.name = delegate.name
        self._delegate = delegate
        self._response = response
        self.requests: list[ProviderRequest] = []

    def capabilities(self, target: ProviderTarget) -> set[Capability]:
        return self._delegate.capabilities(target)

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return self._response

    async def aclose(self) -> None:
        await self._delegate.aclose()


def _require(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} not set")
    return value


def _smoke_graph(*, provider: str, model: str, endpoint: str | None = None) -> CompiledGraph:
    llm: dict[str, object] = {
        "provider": provider,
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
    }
    if endpoint is not None:
        llm["endpoint"] = endpoint

    return compile_workflow(
        WorkflowDoc.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": f"live-smoke-{provider}", "version": 1},
                "spec": {
                    "agents": [
                        {
                            "id": "smoke",
                            "name": "Live smoke",
                            "llm": llm,
                            "system_prompt": (
                                "Return exactly one object whose answer field is the single word ping. "
                                "Do not add any other fields."
                            ),
                            "output_schema": _SUBMIT_SCHEMA,
                        }
                    ],
                    "graph": {
                        "entry": "only",
                        "nodes": [{"id": "only", "agent": "smoke"}],
                        "edges": [],
                        # Force native providers to submit on their first and only
                        # turn, and fail a malformed live response without billing
                        # a repair call. Guided providers remain one-shot too.
                        "guards": {"max_tool_calls_per_turn": 0, "max_output_repairs": 0},
                    },
                },
            }
        )
    )


async def _run_gateway_turn(graph: CompiledGraph, adapter: ProviderAdapter):
    gateway = LLMGateway(graph, {adapter.name: adapter})
    try:
        result = await gateway.run_turn(
            run_id="offline-smoke",
            node_id="only",
            attempt=1,
            logical_visit_id="visit-1",
            agent_id="smoke",
            shared_state={},
        )
        return result
    finally:
        await gateway.aclose()


async def _run_live_workflow(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    adapter: ProviderAdapter,
) -> None:
    workflow_id = get_or_create_workflow(con, graph, org_id="live-smoke", created_by="pytest")
    gateway = LLMGateway(graph, {adapter.name: adapter})
    try:
        run_id = await start_run(
            con,
            graph,
            gateway,
            org_id="live-smoke",
            workflow_id=workflow_id,
            triggered_by="pytest-live-smoke",
            input_payload={},
        )
    finally:
        await gateway.aclose()

    run = con.execute("SELECT status, shared_state FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    assert loads(run["shared_state"]) == {"answer": "ping"}

    execution = con.execute(
        "SELECT input_tokens, output_tokens FROM node_execution WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    assert execution["input_tokens"] > 0
    assert execution["output_tokens"] > 0


def test_adapter_capabilities_are_asserted_offline():
    anthropic = AnthropicAdapter()
    hosted_openai = OpenAICompatibleAdapter(name="openai")
    local_openai = OpenAICompatibleAdapter(name="local", guided_decoding=True)

    assert anthropic.capabilities(
        ProviderTarget(provider="anthropic", model="test-model")
    ) == {Capability.NATIVE_STRUCTURED_OUTPUT}
    assert hosted_openai.capabilities(
        ProviderTarget(provider="openai", model="test-model")
    ) == {Capability.NATIVE_STRUCTURED_OUTPUT}
    assert local_openai.capabilities(
        ProviderTarget(provider="local", model="test-model", endpoint="http://localhost/v1")
    ) == {Capability.NATIVE_STRUCTURED_OUTPUT, Capability.GUIDED_DECODING}


async def test_anthropic_native_request_shape_is_selected_offline():
    graph = _smoke_graph(provider="anthropic", model="test-model")
    response = ProviderResponse(
        text=None,
        tool_calls=[
            NormalizedToolCall(
                id="submit-1",
                tool=SUBMIT_RESULT,
                arguments={"answer": "ping"},
            )
        ],
        input_tokens=1,
        output_tokens=1,
    )
    adapter = _RecordingAdapter(AnthropicAdapter(), response)

    result = await _run_gateway_turn(graph, adapter)

    assert result.structured_payload == {"answer": "ping"}
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert request.output_schema is None
    assert [tool.name for tool in request.tools] == [SUBMIT_RESULT]
    assert request.force_tool == SUBMIT_RESULT


async def test_local_guided_request_shape_is_selected_offline():
    endpoint = "http://localhost:11434/v1"
    graph = _smoke_graph(provider="local", model="test-model", endpoint=endpoint)
    response = ProviderResponse(
        text=json.dumps({"answer": "ping"}),
        input_tokens=1,
        output_tokens=1,
    )
    adapter = _RecordingAdapter(
        OpenAICompatibleAdapter(name="local", guided_decoding=True),
        response,
    )

    result = await _run_gateway_turn(graph, adapter)

    assert result.structured_payload == {"answer": "ping"}
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert request.endpoint == endpoint
    assert request.output_schema == _SUBMIT_SCHEMA
    assert request.tools == []
    assert request.force_tool is None


@_LIVE
async def test_anthropic_native_gateway_round_trip(con: sqlite3.Connection):
    _require("ANTHROPIC_API_KEY")
    model = os.environ.get("RAVANA_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    graph = _smoke_graph(provider="anthropic", model=model)

    await _run_live_workflow(con, graph, AnthropicAdapter())


@_LIVE
async def test_openai_hosted_native_gateway_round_trip(con: sqlite3.Connection):
    _require("OPENAI_API_KEY")
    model = os.environ.get("RAVANA_LIVE_OPENAI_MODEL", "gpt-4o-mini")
    graph = _smoke_graph(provider="openai", model=model)

    await _run_live_workflow(con, graph, OpenAICompatibleAdapter(name="openai"))


@_LIVE
async def test_openai_compatible_local_guided_gateway_round_trip(con: sqlite3.Connection):
    endpoint = _require("RAVANA_LIVE_OPENAI_ENDPOINT")
    model = _require("RAVANA_LIVE_OPENAI_MODEL")
    graph = _smoke_graph(provider="local", model=model, endpoint=endpoint)
    adapter = OpenAICompatibleAdapter(name="local", guided_decoding=True)

    await _run_live_workflow(con, graph, adapter)
