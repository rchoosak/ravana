"""`ravana run ... --backend llm` wiring. These cover the pure selection/build
helpers — which providers become which adapters, and that the gateway is
constructed with the graph's toolkits — without any network call (adapters
defer their SDK-client construction to the first complete() call, so building
one needs neither a credential nor a connection). The end-to-end run against
real models/APIs is intentionally not exercised here (it needs real
credentials); that remains a manual smoke test, tracked in TASKS.md.
"""

from __future__ import annotations

import asyncio
from typing import Any

import click
import pytest

import ravana.cli as cli_module
from ravana.cli import (
    _adapters_for_graph,
    _build_llm_gateway,
    _build_runtime,
    _make_adapter,
    _prose_verdict_for,
    _providers_in_graph,
    _resume_with_cleanup,
    _start_with_cleanup,
)
from ravana.runtime.base import AgentTurnResult
from ravana.runtime.gateway import LLMGateway
from ravana.runtime.mock import MockAgentRuntime
from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter


def test_providers_in_graph_collects_agent_and_fallback_providers(sdlc_graph):
    # SDLC: pm/sa anthropic, dev local (fallback anthropic), qa openai.
    assert _providers_in_graph(sdlc_graph) == {"anthropic", "local", "openai"}


def test_make_adapter_maps_non_anthropic_to_openai_compatible():
    local = _make_adapter("local")
    assert isinstance(local, OpenAICompatibleAdapter)
    assert local.name == "local"  # name preserved so the gateway keys match llm.provider
    assert isinstance(_make_adapter("openai"), OpenAICompatibleAdapter)


def test_make_adapter_maps_anthropic_to_anthropic_adapter():
    # AnthropicAdapter defers its SDK-client construction to the first
    # complete() call (inside the normalization boundary), so building the
    # adapter needs no credential at all.
    assert isinstance(_make_adapter("anthropic"), AnthropicAdapter)


def test_adapters_for_graph_covers_every_provider(sdlc_graph):
    adapters = _adapters_for_graph(sdlc_graph)
    assert set(adapters) == {"anthropic", "local", "openai"}


def test_build_llm_gateway_wires_graph_toolkits(sdlc_graph, con):
    gateway = _build_llm_gateway(con, sdlc_graph)
    assert isinstance(gateway, LLMGateway)
    # The gateway's executor surfaces an agent's declared toolkits as tools.
    specs = gateway._tools.tools_for(["git_connector"])
    assert [t.name for t in specs] == ["git_connector"]
    # §8c: the SAME resolver serves toolkit auth_refs AND llm.api_key_refs —
    # the gateway can resolve per-agent LLM keys at dispatch.
    assert gateway._secret_resolver is not None


def test_build_runtime_mock_requires_fixture(sdlc_graph, con):
    with pytest.raises(click.ClickException, match="requires --mock-fixture"):
        _build_runtime(con, sdlc_graph, "mock", None)


def test_prose_verdict_wired_only_for_llm_backend(sdlc_graph, con):
    # §3.1 step 7: the DoD prose judge is the gateway's own judge_prose under
    # --backend llm; the mock backend has no judge, so prose stays advisory.
    gateway = _build_llm_gateway(con, sdlc_graph)
    assert _prose_verdict_for(gateway) == gateway.judge_prose
    assert _prose_verdict_for(MockAgentRuntime({})) is None


def test_cli_start_scope_forwards_prose_verdict(monkeypatch, sdlc_graph, con):
    # The start scope threads a real prose judge into start_run when the runtime
    # is the LLM gateway — otherwise the terminal DoD gate would silently skip
    # prose criteria even on a real run.
    seen: dict[str, Any] = {}

    async def fake_start(*args, **kwargs):
        seen["dod_prose_verdict"] = kwargs.get("dod_prose_verdict")
        return "run-1"

    monkeypatch.setattr(cli_module, "start_run", fake_start)
    gateway = _build_llm_gateway(con, sdlc_graph)
    asyncio.run(
        _start_with_cleanup(
            con, sdlc_graph, gateway, org_id="test", workflow_id="workflow-1", input_payload={}
        )
    )
    assert seen["dod_prose_verdict"] == gateway.judge_prose


class _ClosableRuntime:
    def __init__(self):
        self.closed = False

    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        logical_visit_id: str,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult:
        raise AssertionError("test patched the engine entry point")

    async def aclose(self) -> None:
        self.closed = True


def test_cli_start_scope_closes_runtime_on_success(monkeypatch, sdlc_graph, con):
    async def fake_start(*args, **kwargs):
        return "run-1"

    monkeypatch.setattr(cli_module, "start_run", fake_start)
    runtime = _ClosableRuntime()
    run_id = asyncio.run(
        _start_with_cleanup(
            con,
            sdlc_graph,
            runtime,
            org_id="test",
            workflow_id="workflow-1",
            input_payload={},
        )
    )
    assert run_id == "run-1"
    assert runtime.closed


def test_cli_start_scope_closes_runtime_when_preparation_fails(
    sdlc_graph, con
):
    class FailingPreparationRuntime(_ClosableRuntime):
        async def prepare_run(self, run_id: str) -> None:
            raise RuntimeError("workspace preparation failed")

    runtime = FailingPreparationRuntime()
    with pytest.raises(RuntimeError, match="workspace preparation failed"):
        asyncio.run(
            _start_with_cleanup(
                con,
                sdlc_graph,
                runtime,
                org_id="test",
                workflow_id="workflow-1",
                input_payload={},
            )
        )
    assert runtime.closed


def test_cli_resume_scope_closes_runtime_on_failure(monkeypatch, sdlc_graph, con):
    async def failing_resume(*args, **kwargs):
        raise RuntimeError("resume failed")

    monkeypatch.setattr(cli_module, "resume_hitl", failing_resume)
    runtime = _ClosableRuntime()
    with pytest.raises(RuntimeError, match="resume failed"):
        asyncio.run(
            _resume_with_cleanup(
                con,
                sdlc_graph,
                runtime,
                "run-1",
                "hitl-1",
                {"answer": "yes"},
            )
        )
    assert runtime.closed


# --- §10.1 git workspace provisioning wiring -------------------------------
def _no_toolkit_graph():
    from ravana.compiler.graph import compile_workflow
    from ravana.schema.models import WorkflowDoc

    return compile_workflow(
        WorkflowDoc.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": "x", "version": 1},
                "spec": {
                    "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
                    "graph": {"entry": "n", "nodes": [{"id": "n", "agent": "a"}], "edges": []},
                },
            }
        )
    )


def test_engine_invokes_runtime_run_preparation(con):
    from ravana.compiler.persist import get_or_create_workflow
    from ravana.engine.loop import start_run

    class PreparingRuntime(_ClosableRuntime):
        prepared_run_id: str | None = None

        async def prepare_run(self, run_id: str) -> None:
            self.prepared_run_id = run_id

        async def run_turn(self, **kwargs) -> AgentTurnResult:
            return AgentTurnResult(structured_payload={})

    graph = _no_toolkit_graph()
    workflow_id = get_or_create_workflow(
        con, graph, org_id="test", created_by="test"
    )
    runtime = PreparingRuntime()
    run_id = asyncio.run(
        start_run(
            con,
            graph,
            runtime,
            org_id="test",
            workflow_id=workflow_id,
        )
    )
    assert runtime.prepared_run_id == run_id


def test_llm_runtime_prepares_requested_git_base(
    tmp_path, sdlc_graph, con, monkeypatch
):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    project = tmp_path
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "T"], check=True)
    (project / "f.txt").write_text("first")
    subprocess.run(["git", "-C", str(project), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "first"], check=True)
    first_commit = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
    ).strip()
    (project / "f.txt").write_text("second")
    subprocess.run(["git", "-C", str(project), "commit", "-qam", "second"], check=True)
    ravana = project / ".ravana"
    (ravana / "runs").mkdir(parents=True)
    monkeypatch.setattr(cli_module, "find_ravana_dir", lambda: ravana)

    gateway = _build_llm_gateway(con, sdlc_graph, git_base_ref=first_commit)

    async def prepare_and_close() -> None:
        try:
            await gateway.prepare_run("run-x")
        finally:
            await gateway.aclose()

    asyncio.run(prepare_and_close())
    ws = ravana / "runs" / "run-x" / "workspace"
    assert (ws / ".git").exists()
    assert (ws / "f.txt").read_text() == "first"


def test_llm_runtime_prepares_plain_workspace_for_non_git_project(
    tmp_path, sdlc_graph, con, monkeypatch
):
    ravana = tmp_path / ".ravana"
    (ravana / "runs").mkdir(parents=True)
    monkeypatch.setattr(cli_module, "find_ravana_dir", lambda: ravana)
    gateway = _build_llm_gateway(con, sdlc_graph)

    async def prepare_and_close() -> None:
        try:
            await gateway.prepare_run("run-x")
        finally:
            await gateway.aclose()

    asyncio.run(prepare_and_close())
    workspace = ravana / "runs" / "run-x" / "workspace"
    assert workspace.is_dir()
    assert not (workspace / ".git").exists()
