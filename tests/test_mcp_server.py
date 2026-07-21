"""`mcp_server` toolkit (§1.7) against a real MCP stdio subprocess.

The §8 requirements are the load-bearing assertions: the endpoint allow-list
(an arbitrary `config.command` must not be launchable by whoever can edit a
workflow) and tool-list pinning (a server that grows a tool after approval must
not have it offered or invoked).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from tests.test_tool_execution import _seed_run

from ravana.runtime.toolkits.base import ToolkitError
from ravana.runtime.toolkits.mcp_server import (
    McpServerHandler,
    check_endpoint_allowed,
    qualified_tool_name,
)

PROBE = str(Path(__file__).parent / "fixtures" / "mcp_probe_server.py")


def _config(**overrides):
    config = {"transport": "stdio", "command": sys.executable, "args": [PROBE]}
    config.update(overrides)
    return config


async def _prepared(toolkit_id="probe_mcp", **overrides):
    handler = McpServerHandler(toolkit_id, _config(**overrides))
    await handler.prepare_run("run-1")
    return handler


async def test_server_tools_are_discovered_and_qualified():
    handler = await _prepared()
    try:
        names = sorted(t.name for t in handler.sub_tools)
        assert names == [
            "probe_mcp__add",
            "probe_mcp__echo_env",
            "probe_mcp__explode",
        ]
        add = next(t for t in handler.sub_tools if t.name.endswith("__add"))
        assert add.description == "Add two numbers."
        assert add.input_schema["type"] == "object"  # server JSON Schema passes through
        assert handler.executable is True
    finally:
        await handler.aclose()


async def test_calling_a_pinned_tool_returns_its_result():
    handler = await _prepared()
    try:
        out = await handler.call(
            arguments={"a": 2, "b": 3},
            idempotency_key="k1",
            run_id="run-1",
            tool=qualified_tool_name("probe_mcp", "add"),
        )
        assert out == "5"
    finally:
        await handler.aclose()


async def test_tool_added_after_preparation_cannot_be_invoked():
    # §8 tool poisoning: the pinned set is the authority, not the live server.
    handler = await _prepared()
    try:
        handler._pinned.pop("add")  # simulate: never approved at preparation
        with pytest.raises(ToolkitError, match="not in the tool list pinned"):
            await handler.call(
                arguments={"a": 1, "b": 1},
                idempotency_key="k",
                run_id="run-1",
                tool="probe_mcp__add",
            )
    finally:
        await handler.aclose()


async def test_allowed_tools_narrows_what_is_exposed():
    handler = await _prepared(allowed_tools=["add"])
    try:
        assert [t.name for t in handler.sub_tools] == ["probe_mcp__add"]
        with pytest.raises(ToolkitError, match="not in the tool list pinned"):
            await handler.call(
                arguments={"name": "PATH"},
                idempotency_key="k",
                run_id="run-1",
                tool="probe_mcp__echo_env",
            )
    finally:
        await handler.aclose()


async def test_allow_listed_tool_the_server_does_not_offer_is_fatal():
    handler = McpServerHandler("probe_mcp", _config(allowed_tools=["add", "not_offered"]))
    with pytest.raises(ToolkitError, match="not offered by the server"):
        await handler.prepare_run("run-1")


async def test_a_failing_tool_comes_back_as_an_addressable_result():
    # §3.6: a remote tool's own error is fed to the model, not raised as a run
    # failure — the model can adjust its arguments or route around it.
    handler = await _prepared()
    try:
        out = await handler.call(
            arguments={},
            idempotency_key="k",
            run_id="run-1",
            tool="probe_mcp__explode",
        )
        assert "ERROR from probe_mcp.explode" in out
    finally:
        await handler.aclose()


async def test_session_survives_prepare_call_and_close_in_different_tasks():
    # The reason the session lives in its own worker task. `stdio_client` is an
    # anyio cancel scope, and exiting one from a different task than entered
    # raises "Attempted to exit cancel scope in a different task"; holding the
    # context open across these three calls directly would work only while they
    # share a task, which the single-process engine happens to do today.
    handler = McpServerHandler("probe_mcp", _config())
    await asyncio.create_task(handler.prepare_run("run-1"))

    async def call_it():
        return await handler.call(
            arguments={"a": 7, "b": 1},
            idempotency_key="k",
            run_id="run-1",
            tool="probe_mcp__add",
        )

    assert await asyncio.create_task(call_it()) == "8"
    await asyncio.create_task(handler.aclose())  # third distinct task


async def test_a_server_that_refuses_to_start_is_transient_not_silent():
    handler = McpServerHandler(
        "probe_mcp", _config(env={"RAVANA_PROBE_REFUSE_START": "1"})
    )
    with pytest.raises(ToolkitError, match="failed to start"):
        await handler.prepare_run("run-1")
    assert handler.executable is False


async def test_auth_token_reaches_the_server_via_env_only():
    # §8(c): the credential is resolved at dispatch and handed to the child
    # process, never written into config or the tool arguments.
    handler = McpServerHandler(
        "probe_mcp", _config(), get_auth_token=lambda: "s3cret-token"
    )
    await handler.prepare_run("run-1")
    try:
        out = await handler.call(
            arguments={"name": "MCP_AUTH_TOKEN"},
            idempotency_key="k",
            run_id="run-1",
            tool="probe_mcp__echo_env",
        )
        assert out == "s3cret-token"
    finally:
        await handler.aclose()


# --- §8 endpoint allow-list -------------------------------------------------
def test_unconfigured_allowlist_fails_closed():
    # "No allow-list" must not mean "any command": config.command comes from a
    # workflow file, so allowing it would be arbitrary local execution by
    # whoever can edit workflows.
    with pytest.raises(ToolkitError, match="no MCP server allow-list is configured"):
        check_endpoint_allowed("probe_mcp", {"command": "/bin/sh"}, None)
    with pytest.raises(ToolkitError, match="no MCP server allow-list is configured"):
        check_endpoint_allowed("probe_mcp", {"command": "/bin/sh"}, set())


def test_command_outside_the_allowlist_is_refused():
    with pytest.raises(ToolkitError, match="not in the admin-curated allow-list"):
        check_endpoint_allowed("probe_mcp", {"command": "/bin/sh"}, {sys.executable})


def test_allow_listed_command_passes():
    check_endpoint_allowed("probe_mcp", {"command": sys.executable}, {sys.executable})


def test_unsupported_transport_is_refused():
    with pytest.raises(ToolkitError, match="transport"):
        McpServerHandler("probe_mcp", {"transport": "http", "url": "https://x"})


# --- registry / executor wiring ---------------------------------------------
def _mcp_graph(command, **config_extra):
    from ravana.compiler.graph import compile_workflow
    from ravana.schema.models import WorkflowDoc

    return compile_workflow(
        WorkflowDoc.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": "m", "version": 1},
                "spec": {
                    "toolkits": [
                        {
                            "id": "probe_mcp",
                            "type": "mcp_server",
                            "config": {"transport": "stdio", "command": command,
                                       "args": [PROBE], **config_extra},
                        }
                    ],
                    "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"},
                                "system_prompt": "p", "toolkits": ["probe_mcp"]}],
                    "graph": {"entry": "n", "nodes": [{"id": "n", "agent": "a"}], "edges": []},
                },
            }
        )
    )


def _registry(graph, allowlist):
    from ravana.runtime.secrets import EnvSecretResolver
    from ravana.runtime.toolkits.registry import build_registry

    return build_registry(graph, EnvSecretResolver(), mcp_allowlist=allowlist)


def test_registry_refuses_an_unlisted_endpoint_without_exploding(con):
    # Fail-closed, but at the same seam every other unusable toolkit fails at:
    # building the registry must not explode over a toolkit no executed agent
    # declares. tools_for is where it becomes loud.
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph("/bin/sh"), {sys.executable})
    assert handlers["probe_mcp"].executable is False
    executor = RavanaToolExecutor(con, handlers)
    with pytest.raises(ToolkitError, match="not in the admin-curated allow-list"):
        executor.tools_for(["probe_mcp"])


def test_registry_refuses_every_mcp_server_when_no_allowlist_configured(con):
    handlers = _registry(_mcp_graph(sys.executable), None)
    assert handlers["probe_mcp"].executable is False
    assert "no MCP server allow-list is configured" in handlers["probe_mcp"].description


async def test_executor_surfaces_and_routes_mcp_sub_tools(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph(sys.executable), {sys.executable})
    executor = RavanaToolExecutor(con, handlers)
    await executor.prepare_run("run-1")
    try:
        names = sorted(t.name for t in executor.tools_for(["probe_mcp"]))
        assert names == ["probe_mcp__add", "probe_mcp__echo_env", "probe_mcp__explode"]

        _seed_run(con, run_id="run-1")
        out = await executor.execute(
            run_id="run-1", node_id="n", tool="probe_mcp__add",
            arguments={"a": 4, "b": 5}, idempotency_key="k1",
        )
        assert out == "9"
    finally:
        await executor.aclose()


async def test_executor_rejects_bad_arguments_against_the_sub_tool_schema(con):
    # §8(a): the schema enforced is the SUB-tool's, published by the server —
    # the handler's own is a permissive placeholder that would let anything by.
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph(sys.executable), {sys.executable})
    executor = RavanaToolExecutor(con, handlers)
    await executor.prepare_run("run-1")
    try:
        _seed_run(con, run_id="run-1")
        with pytest.raises(ToolkitError, match="invalid arguments"):
            await executor.execute(
                run_id="run-1", node_id="n", tool="probe_mcp__add",
                arguments={"a": 1}, idempotency_key="k2",  # 'b' missing
            )
    finally:
        await executor.aclose()
