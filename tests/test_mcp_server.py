"""`mcp_server` toolkit (§1.7) against a real MCP stdio subprocess.

The §8 requirements are the load-bearing assertions: the endpoint allow-list
(workflow config may only reference an admin-owned server definition) and
tool-list pinning (a server that grows a tool after approval must not have it
offered or invoked).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_tool_execution import _seed_run

from ravana.runtime.providers.base import Tool
from ravana.runtime.toolkits.base import ToolRetrySafeCancellation, ToolkitError
from ravana.runtime.secrets import ResolvedSecret
from ravana.runtime.toolkits.base import ToolOutcomeUnknown
from ravana.runtime.toolkits.mcp_server import (
    McpServerDefinition,
    McpServerHandler,
    _close_stack,
    check_endpoint_allowed,
    parse_server_allowlist,
    qualified_tool_name,
)
from ravana.schema.util import now_iso

PROBE = str(Path(__file__).parent / "fixtures" / "mcp_probe_server.py")


def _server(
    env=None,
    *,
    auth_env="MCP_AUTH_TOKEN",
    authenticate_discovery=False,
    read_only_tools=(),
    cwd=None,
):
    return McpServerDefinition(
        name="probe",
        command=sys.executable,
        cwd=cwd or str(Path(PROBE).parent),
        args=(PROBE,),
        env=tuple(sorted((env or {}).items())),
        auth_env=auth_env,
        authenticate_discovery=authenticate_discovery,
        read_only_tools=read_only_tools,
    )


def _config(**overrides):
    config = {"transport": "stdio", "server": "probe"}
    config.update(overrides)
    return config


async def _prepared(toolkit_id="probe_mcp", **overrides):
    server_env = overrides.pop("_server_env", None)
    handler = McpServerHandler(toolkit_id, _config(**overrides), server=_server(server_env))
    await handler.prepare_run("run-1")
    return handler


async def test_server_tools_are_discovered_and_qualified():
    handler = await _prepared()
    try:
        tools = handler.sub_tools_for("run-1")
        names = sorted(t.name for t in tools)
        assert names == [
            "probe_mcp__add",
            "probe_mcp__auth_is_set",
            "probe_mcp__current_directory",
            "probe_mcp__echo_env",
            "probe_mcp__explode",
        ]
        add = next(t for t in tools if t.name.endswith("__add"))
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


async def test_admin_working_directory_controls_server_cwd(tmp_path):
    handler = McpServerHandler(
        "probe_mcp",
        _config(allowed_tools=["current_directory"]),
        server=_server(cwd=str(tmp_path)),
    )
    await handler.prepare_run("run-1")
    try:
        assert await handler.call(
            arguments={},
            idempotency_key="cwd",
            run_id="run-1",
            tool="probe_mcp__current_directory",
        ) == str(tmp_path)
    finally:
        await handler.aclose()


async def test_tool_added_after_preparation_cannot_be_invoked():
    # §8 tool poisoning: the pinned set is the authority, not the live server.
    handler = await _prepared()
    try:
        handler._pinned_by_run["run-1"].pop("add")  # simulate: never approved at preparation
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
        assert [t.name for t in handler.sub_tools_for("run-1")] == [
            "probe_mcp__add"
        ]
        with pytest.raises(ToolkitError, match="not in the tool list pinned"):
            await handler.call(
                arguments={"name": "PATH"},
                idempotency_key="k",
                run_id="run-1",
                tool="probe_mcp__echo_env",
            )
    finally:
        await handler.aclose()


async def test_read_only_tools_are_admin_declared_not_server_hints():
    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(read_only_tools=("echo_env",)),
    )
    await handler.prepare_run("run-1")
    try:
        assert handler.is_side_effecting_for_tool("echo_env", {}) is False
        assert handler.is_side_effecting_for_tool("add", {}) is True
    finally:
        await handler.aclose()


async def test_allow_listed_tool_the_server_does_not_offer_is_fatal():
    handler = McpServerHandler(
        "probe_mcp", _config(allowed_tools=["add", "not_offered"]), server=_server()
    )
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


async def test_each_dispatch_owns_its_session_task():
    # `stdio_client` is an anyio cancel scope. Each operation enters and exits
    # its short-lived session in the same task, so callers may dispatch from
    # unrelated tasks without crossing cancel-scope ownership.
    handler = McpServerHandler("probe_mcp", _config(), server=_server())
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
        "probe_mcp",
        _config(),
        server=_server({"RAVANA_PROBE_REFUSE_START": "1"}),
    )
    with pytest.raises(ToolkitError, match="failed to start"):
        await handler.prepare_run("run-1")
    assert handler.executable is False


async def test_auth_token_reaches_the_server_via_env_only():
    # §8(c): ResolvedSecret is opened only at the child-process boundary and
    # is resolved again for the dispatch, never written into config/arguments.
    calls = 0

    def token():
        nonlocal calls
        calls += 1
        return ResolvedSecret("s3cret-token")

    handler = McpServerHandler(
        "probe_mcp", _config(), server=_server(), get_auth_token=token
    )
    await handler.prepare_run("run-1")
    try:
        out = await handler.call(
            arguments={},
            idempotency_key="k",
            run_id="run-1",
            tool="probe_mcp__auth_is_set",
        )
        assert out == "true"
        assert calls == 1  # only the actual dispatch opens the credential
    finally:
        await handler.aclose()


async def test_dispatch_credential_never_reaches_host_stderr(capfd):
    token = "stderr-sentinel-secret"
    handler = McpServerHandler(
        "probe_mcp",
        _config(allowed_tools=["auth_is_set"]),
        server=_server({"RAVANA_PROBE_ECHO_AUTH_STDERR": "1"}),
        get_auth_token=lambda: ResolvedSecret(token),
    )
    await handler.prepare_run("run-1")
    capfd.readouterr()
    try:
        assert await handler.call(
            arguments={},
            idempotency_key="stderr",
            run_id="run-1",
            tool="probe_mcp__auth_is_set",
        ) == "true"
        captured = capfd.readouterr()
        assert token not in captured.out
        assert token not in captured.err
    finally:
        await handler.aclose()


async def test_authenticated_discovery_is_explicit_and_dispatch_re_resolves():
    calls = 0

    def token():
        nonlocal calls
        calls += 1
        return ResolvedSecret(f"token-{calls}")

    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(authenticate_discovery=True),
        get_auth_token=token,
    )
    await handler.prepare_run("run-1")
    assert calls == 1
    try:
        assert await handler.call(
            arguments={},
            idempotency_key="k-auth-discovery",
            run_id="run-1",
            tool="probe_mcp__auth_is_set",
        ) == "true"
        assert calls == 2
    finally:
        await handler.aclose()


async def test_mcp_rejects_a_tool_result_that_echoes_the_dispatch_credential():
    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(),
        get_auth_token=lambda: ResolvedSecret("arbitrary-token"),
    )
    await handler.prepare_run("run-1")
    try:
        with pytest.raises(ToolOutcomeUnknown, match="outcome is unknown"):
            await handler.call(
                arguments={"name": "MCP_AUTH_TOKEN"},
                idempotency_key="k-leak",
                run_id="run-1",
                tool="probe_mcp__echo_env",
            )
    finally:
        await handler.aclose()


def test_auth_ref_uses_the_admin_selected_environment_name():
    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(auth_env="GITHUB_PERSONAL_ACCESS_TOKEN"),
        get_auth_token=lambda: ResolvedSecret("github-token"),
    )

    env, secret_values = handler._child_environment(include_auth=True)
    assert env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "github-token"
    assert env["PATH"] == ""
    assert secret_values == ("github-token",)


def test_child_environment_does_not_inherit_unrelated_host_secrets(monkeypatch):
    monkeypatch.setenv("RAVANA_SECRET_UNRELATED", "must-not-cross-process-boundary")
    monkeypatch.setenv("HOME", "/host/home")
    monkeypatch.setenv("PATH", "/host/path")
    handler = McpServerHandler("probe_mcp", _config(), server=_server())

    env, secret_values = handler._child_environment(include_auth=False)
    assert env["HOME"] == ""
    assert env["PATH"] == ""
    assert env["USER"] == ""
    assert "RAVANA_SECRET_UNRELATED" not in env
    assert secret_values == ()


def test_child_environment_allows_only_explicit_admin_overrides():
    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(env={"PATH": "/admin/path", "RAVANA_PROBE_MODE": "test"}),
    )

    env, _ = handler._child_environment(include_auth=False)
    assert env["PATH"] == "/admin/path"
    assert env["RAVANA_PROBE_MODE"] == "test"
    assert env["HOME"] == ""


def test_cli_loads_complete_admin_server_definitions(tmp_path):
    from ravana.cli import _mcp_allowlist

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "mcp": {
                    "allowed_servers": {
                        "probe": {
                            "command": sys.executable,
                            "args": [PROBE],
                            "cwd": str(tmp_path),
                            "env": {"RAVANA_PROBE_MODE": "test"},
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    allowlist = _mcp_allowlist(tmp_path)
    assert allowlist is not None
    assert allowlist["probe"].args == (PROBE,)
    assert allowlist["probe"].working_directory == str(tmp_path)
    assert allowlist["probe"].environment == {"RAVANA_PROBE_MODE": "test"}


# --- §8 endpoint allow-list -------------------------------------------------
def test_unconfigured_allowlist_fails_closed():
    # "No allow-list" must not mean "any command": a workflow may only name an
    # install-owned definition.
    with pytest.raises(ToolkitError, match="no MCP server allow-list is configured"):
        check_endpoint_allowed("probe_mcp", {"server": "probe"}, None)
    with pytest.raises(ToolkitError, match="no MCP server allow-list is configured"):
        check_endpoint_allowed("probe_mcp", {"server": "probe"}, {})


def test_command_outside_the_allowlist_is_refused():
    with pytest.raises(ToolkitError, match="not in the admin-curated allow-list"):
        check_endpoint_allowed(
            "probe_mcp", {"server": "unlisted"}, {"probe": _server()}
        )


def test_allow_listed_command_passes():
    assert check_endpoint_allowed(
        "probe_mcp", {"server": "probe"}, {"probe": _server()}
    ).name == "probe"


def test_workflow_cannot_override_admin_server_launch_definition():
    with pytest.raises(ToolkitError, match="must be supplied by the admin"):
        check_endpoint_allowed(
            "probe_mcp",
            {"server": "probe", "args": ["-c", "malicious"]},
            {"probe": _server()},
        )
    with pytest.raises(ToolkitError, match="must be supplied by the admin"):
        check_endpoint_allowed(
            "probe_mcp",
            {"server": "probe", "cwd": "/tmp"},
            {"probe": _server()},
        )


def test_server_allowlist_requires_complete_definitions():
    parsed = parse_server_allowlist(
        {
            "probe": {
                "command": sys.executable,
                "args": [PROBE],
                "cwd": str(Path(PROBE).parent),
                "env": {},
                "auth_env": "GITHUB_PERSONAL_ACCESS_TOKEN",
                "authenticate_discovery": True,
                "read_only_tools": ["list_issues"],
            }
        }
    )
    assert parsed is not None
    # NOT `Path(...).resolve()`: the command must keep the interpreter it was
    # given. Resolving the final symlink turns a virtualenv interpreter into its
    # base one, which cannot import the server's dependencies — see
    # test_parsed_definition_from_a_venv_interpreter_actually_starts.
    assert parsed["probe"].command == sys.executable
    assert parsed["probe"].working_directory == str(Path(PROBE).parent.resolve())
    assert parsed["probe"].auth_env == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert parsed["probe"].authenticate_discovery is True
    assert parsed["probe"].read_only_tools == ("list_issues",)


def test_named_server_commands_require_an_admin_owned_path():
    with pytest.raises(ToolkitError, match="env.PATH"):
        parse_server_allowlist(
            {
                "probe": {
                    "command": "python",
                    "args": [],
                    "cwd": str(Path(PROBE).parent),
                }
            }
        )


def test_server_definition_requires_an_explicit_working_directory():
    with pytest.raises(ToolkitError, match="cwd must be an existing absolute directory"):
        parse_server_allowlist({"probe": {"command": sys.executable}})
    with pytest.raises(ToolkitError, match="cwd must be an existing absolute directory"):
        McpServerDefinition(name="probe", command=sys.executable, cwd="relative")


def test_server_definition_rejects_a_relative_working_directory():
    with pytest.raises(ToolkitError, match="existing absolute directory"):
        parse_server_allowlist(
            {"probe": {"command": sys.executable, "cwd": "relative"}}
        )


def test_unsupported_transport_is_refused():
    with pytest.raises(ToolkitError, match="transport"):
        McpServerHandler(
            "probe_mcp", {"transport": "http", "url": "https://x"}, server=_server()
        )


# --- registry / executor wiring ---------------------------------------------
def _mcp_graph(server="probe", **config_extra):
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
                            "config": {"transport": "stdio", "server": server, **config_extra},
                        }
                    ],
                    "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"},
                                "system_prompt": "p", "toolkits": ["probe_mcp"]}],
                    "graph": {"entry": "n", "nodes": [{"id": "n", "agent": "a"}], "edges": []},
                },
            }
        )
    )


def _registry(graph, allowlist, con=None):
    from ravana.runtime.secrets import EnvSecretResolver
    from ravana.runtime.toolkits.registry import build_registry

    return build_registry(
        graph, EnvSecretResolver(), mcp_allowlist=allowlist, mcp_snapshot_con=con
    )


def test_registry_refuses_an_unlisted_endpoint_without_exploding(con):
    # Fail-closed, but at the same seam every other unusable toolkit fails at:
    # building the registry must not explode over a toolkit no executed agent
    # declares. tools_for is where it becomes loud.
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph("missing"), {"probe": _server()})
    assert handlers["probe_mcp"].executable is False
    executor = RavanaToolExecutor(con, handlers)
    with pytest.raises(ToolkitError, match="not in the admin-curated allow-list"):
        executor.tools_for(["probe_mcp"])


def test_registry_refuses_every_mcp_server_when_no_allowlist_configured(con):
    handlers = _registry(_mcp_graph("probe"), None)
    assert handlers["probe_mcp"].executable is False
    assert "no MCP server allow-list is configured" in handlers["probe_mcp"].description


async def test_executor_surfaces_and_routes_mcp_sub_tools(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph("probe"), {"probe": _server()})
    executor = RavanaToolExecutor(con, handlers)
    await executor.prepare_run("run-1")
    await executor.prepare_tools("run-1", ["probe_mcp"])
    try:
        with pytest.raises(ToolkitError, match="run_id is required"):
            executor.tools_for(["probe_mcp"])
        names = sorted(t.name for t in executor.tools_for(["probe_mcp"], run_id="run-1"))
        assert names == [
            "probe_mcp__add",
            "probe_mcp__auth_is_set",
            "probe_mcp__current_directory",
            "probe_mcp__echo_env",
            "probe_mcp__explode",
        ]

        _seed_run(con, run_id="run-1")
        out = await executor.execute(
            run_id="run-1", node_id="n", tool="probe_mcp__add",
            arguments={"a": 4, "b": 5}, idempotency_key="k1",
        )
        assert out == "9"
    finally:
        await executor.aclose()


async def test_executor_prepares_only_mcp_toolkits_requested_by_the_active_node(
    monkeypatch, con
):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    prepared: list[tuple[str, str]] = []
    used = McpServerHandler("used", _config(), server=_server())
    unused = McpServerHandler("unused", _config(), server=_server())

    async def prepare_used(run_id):
        prepared.append(("used", run_id))

    async def prepare_unused(run_id):
        prepared.append(("unused", run_id))

    monkeypatch.setattr(used, "prepare_run", prepare_used)
    monkeypatch.setattr(unused, "prepare_run", prepare_unused)
    executor = RavanaToolExecutor(con, {"used": used, "unused": unused})

    await executor.prepare_run("run-1")
    assert prepared == []

    await executor.prepare_tools("run-1", ["used"])
    assert prepared == [("used", "run-1")]


async def test_executor_applies_admin_read_only_to_qualified_tool(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(
        _mcp_graph("probe"), {"probe": _server(read_only_tools=("echo_env",))}, con
    )
    executor = RavanaToolExecutor(con, handlers)
    await executor.prepare_run("run-read-only")
    await executor.prepare_tools("run-read-only", ["probe_mcp"])
    try:
        _seed_run(con, run_id="run-read-only")
        for key in ("read-1", "read-2"):
            assert await executor.execute(
                run_id="run-read-only",
                node_id="n",
                tool="probe_mcp__echo_env",
                arguments={"name": "PATH"},
                idempotency_key=key,
            )
        count = con.execute(
            "SELECT COUNT(*) AS count FROM tool_invocation WHERE run_id = ?",
            ("run-read-only",),
        ).fetchone()["count"]
        assert count == 0
    finally:
        await executor.aclose()


async def test_executor_keeps_credential_echo_as_indeterminate(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(),
        get_auth_token=lambda: ResolvedSecret("arbitrary-token"),
    )
    executor = RavanaToolExecutor(con, {"probe_mcp": handler})
    await executor.prepare_run("run-secret")
    await executor.prepare_tools("run-secret", ["probe_mcp"])
    try:
        _seed_run(con, run_id="run-secret")
        with pytest.raises(ToolOutcomeUnknown):
            await executor.execute(
                run_id="run-secret",
                node_id="n",
                tool="probe_mcp__echo_env",
                arguments={"name": "MCP_AUTH_TOKEN"},
                idempotency_key="secret-echo",
            )
        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("secret-echo",),
        ).fetchone()
        assert row["status"] == "STARTED"
    finally:
        await executor.aclose()


async def test_executor_marks_pre_dispatch_cancellation_retry_safe(monkeypatch, con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handler = await _prepared()

    async def cancel_before_dispatch(tool, arguments, idempotency_key, phase):
        raise asyncio.CancelledError

    monkeypatch.setattr(handler, "_call_once", cancel_before_dispatch)
    executor = RavanaToolExecutor(con, {"probe_mcp": handler})
    try:
        _seed_run(con, run_id="run-1")
        with pytest.raises(ToolRetrySafeCancellation):
            await executor.execute(
                run_id="run-1",
                node_id="n",
                tool="probe_mcp__add",
                arguments={"a": 1, "b": 2},
                idempotency_key="cancel-before",
            )
        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("cancel-before",),
        ).fetchone()
        assert row["status"] == "FAILED"
    finally:
        await executor.aclose()


async def test_mcp_tool_snapshot_is_restored_by_a_fresh_runtime(monkeypatch, con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    graph = _mcp_graph("probe")
    first = RavanaToolExecutor(con, _registry(graph, {"probe": _server()}, con))
    await first.prepare_run("run-snapshot")
    await first.prepare_tools("run-snapshot", ["probe_mcp"])
    _seed_run(con, run_id="run-snapshot")
    con.execute("UPDATE run SET status = 'WAITING_HUMAN' WHERE id = ?", ("run-snapshot",))
    con.commit()
    await first.release_run("run-snapshot")
    await first.aclose()

    second_handlers = _registry(graph, {"probe": _server()}, con)
    second_handler = second_handlers["probe_mcp"]

    async def should_not_discover():
        raise AssertionError("a fresh runtime must restore the durable snapshot")

    monkeypatch.setattr(second_handler, "_discover_tools", should_not_discover)
    second = RavanaToolExecutor(con, second_handlers)
    try:
        await second.prepare_run("run-snapshot")
        await second.prepare_tools("run-snapshot", ["probe_mcp"])
        assert [tool.name for tool in second.tools_for(["probe_mcp"], run_id="run-snapshot")]
    finally:
        con.execute("UPDATE run SET status = 'COMPLETED' WHERE id = ?", ("run-snapshot",))
        con.commit()
        await second.release_run("run-snapshot")
        await second.aclose()
    assert con.execute(
        "SELECT COUNT(*) AS count FROM mcp_tool_snapshot WHERE run_id = ?",
        ("run-snapshot",),
    ).fetchone()["count"] == 0


async def test_legacy_snapshot_fingerprint_fails_closed(monkeypatch, con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    run_id = "run-legacy-snapshot"
    toolkit_id = "probe_mcp"
    server = _server(cwd=str(Path(PROBE).parent))
    legacy_server_payload = {
        "name": server.name,
        "command": server.command,
        "args": server.args,
        "env": server.env,
        "auth_env": server.auth_env,
        "authenticate_discovery": server.authenticate_discovery,
        "read_only_tools": server.read_only_tools,
    }
    legacy_server_fingerprint = hashlib.sha256(
        json.dumps(
            legacy_server_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    legacy_snapshot_fingerprint = hashlib.sha256(
        json.dumps(
            {"server": legacy_server_fingerprint, "allowed_tools": None},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    _seed_run(con, run_id=run_id)
    con.execute("UPDATE run SET status = 'WAITING_HUMAN' WHERE id = ?", (run_id,))
    con.execute(
        """INSERT INTO mcp_tool_snapshot
           (run_id, toolkit_id, server_fingerprint, tool_name, description, input_schema,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            toolkit_id,
            legacy_snapshot_fingerprint,
            "add",
            "Add two integers",
            "{}",
            now_iso(),
        ),
    )
    con.commit()

    handlers = _registry(_mcp_graph("probe"), {"probe": server}, con)
    handler = handlers[toolkit_id]

    async def should_not_discover():
        raise AssertionError("a mismatched snapshot must fail before rediscovery")

    monkeypatch.setattr(handler, "_discover_tools", should_not_discover)
    executor = RavanaToolExecutor(con, handlers)
    try:
        with pytest.raises(ToolkitError, match="admin definition or tool grant changed"):
            await executor.prepare_tools(run_id, [toolkit_id])
        stored = con.execute(
            """SELECT DISTINCT server_fingerprint
               FROM mcp_tool_snapshot WHERE run_id = ? AND toolkit_id = ?""",
            (run_id, toolkit_id),
        ).fetchone()["server_fingerprint"]
        assert stored == legacy_snapshot_fingerprint
    finally:
        await executor.aclose()


async def test_restored_snapshot_revalidates_provider_tool_names(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    run_id = "run-invalid-snapshot-name"
    toolkit_id = "probe_mcp"
    handlers = _registry(_mcp_graph("probe"), {"probe": _server()}, con)
    handler = handlers[toolkit_id]
    _seed_run(con, run_id=run_id)
    con.execute("UPDATE run SET status = 'WAITING_HUMAN' WHERE id = ?", (run_id,))
    con.execute(
        """INSERT INTO mcp_tool_snapshot
           (run_id, toolkit_id, server_fingerprint, tool_name, description, input_schema,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            toolkit_id,
            handler._snapshot_fingerprint(),
            "invalid.tool name",
            "invalid",
            "{}",
            now_iso(),
        ),
    )
    con.commit()

    executor = RavanaToolExecutor(con, handlers)
    try:
        with pytest.raises(ToolkitError, match="provider-compatible"):
            await executor.prepare_tools(run_id, [toolkit_id])
    finally:
        await executor.aclose()


async def test_mcp_tool_list_pagination_and_workflow_filtering_are_independent():
    handler = McpServerHandler(
        "probe_mcp",
        _config(allowed_tools=["add"]),
        server=_server(read_only_tools=("echo_env",)),
    )

    class PagedSession:
        def __init__(self):
            self.cursors = []

        async def list_tools(self, *, cursor=None):
            self.cursors.append(cursor)
            if cursor is None:
                specs = [SimpleNamespace(name="add", description="add", inputSchema={})]
                return SimpleNamespace(tools=specs, nextCursor="page-2")
            specs = [SimpleNamespace(name="echo_env", description="read", inputSchema={})]
            return SimpleNamespace(tools=specs, nextCursor=None)

    session = PagedSession()
    pinned = await handler._pin_tools(session)

    assert list(pinned) == ["add"]
    assert session.cursors == [None, "page-2"]


async def test_authenticated_discovery_rejects_a_credential_in_a_tool_name():
    handler = McpServerHandler(
        "probe_mcp",
        _config(),
        server=_server(authenticate_discovery=True),
    )

    class CredentialEchoSession:
        async def list_tools(self, *, cursor=None):
            spec = SimpleNamespace(
                name="discovery-secret",
                description="ordinary description",
                inputSchema={},
            )
            return SimpleNamespace(tools=[spec], nextCursor=None)

    with pytest.raises(ToolkitError, match="credential material"):
        await handler._pin_tools(
            CredentialEchoSession(), secret_values=("discovery-secret",)
        )


@pytest.mark.parametrize("tool_name", ["invalid tool.name", "x" * 64])
async def test_discovery_rejects_provider_incompatible_qualified_names(tool_name):
    handler = McpServerHandler("probe_mcp", _config(), server=_server())

    class IncompatibleNameSession:
        async def list_tools(self, *, cursor=None):
            spec = SimpleNamespace(
                name=tool_name,
                description="ordinary description",
                inputSchema={},
            )
            return SimpleNamespace(tools=[spec], nextCursor=None)

    with pytest.raises(ToolkitError, match="provider-compatible"):
        await handler._pin_tools(IncompatibleNameSession())


def test_mcp_snapshot_cleanup_keeps_recent_orphans_during_preparation(con):
    con.execute(
        """INSERT INTO mcp_tool_snapshot
           (run_id, toolkit_id, server_fingerprint, tool_name, description, input_schema,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "preparing-run",
            "probe_mcp",
            "fingerprint",
            "add",
            "add",
            "{}",
            now_iso(),
        ),
    )
    con.commit()

    _registry(_mcp_graph("probe"), {"probe": _server()}, con)

    assert con.execute(
        "SELECT COUNT(*) AS count FROM mcp_tool_snapshot WHERE run_id = ?",
        ("preparing-run",),
    ).fetchone()["count"] == 1

    con.execute(
        "UPDATE mcp_tool_snapshot SET created_at = ? WHERE run_id = ?",
        ("1970-01-01T00:00:00+00:00", "preparing-run"),
    )
    con.commit()
    _registry(_mcp_graph("probe"), {"probe": _server()}, con)

    assert con.execute(
        "SELECT COUNT(*) AS count FROM mcp_tool_snapshot WHERE run_id = ?",
        ("preparing-run",),
    ).fetchone()["count"] == 0


async def test_executor_rejects_bad_arguments_against_the_sub_tool_schema(con):
    # §8(a): the schema enforced is the SUB-tool's, published by the server —
    # the handler's own is a permissive placeholder that would let anything by.
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handlers = _registry(_mcp_graph("probe"), {"probe": _server()})
    executor = RavanaToolExecutor(con, handlers)
    await executor.prepare_run("run-1")
    await executor.prepare_tools("run-1", ["probe_mcp"])
    try:
        _seed_run(con, run_id="run-1")
        with pytest.raises(ToolkitError, match="invalid arguments"):
            await executor.execute(
                run_id="run-1", node_id="n", tool="probe_mcp__add",
                arguments={"a": 1}, idempotency_key="k2",  # 'b' missing
            )
    finally:
        await executor.aclose()


async def test_each_run_has_its_own_pinned_snapshot_and_dispatch_credential():
    tokens = iter(
        [
            ResolvedSecret("call-one"),
            ResolvedSecret("call-two"),
        ]
    )
    handler = McpServerHandler(
        "probe_mcp", _config(), server=_server(), get_auth_token=lambda: next(tokens)
    )
    await handler.prepare_run("run-1")
    await handler.prepare_run("run-2")
    assert handler.sub_tools_for("run-1") == handler.sub_tools_for("run-2")
    try:
        assert await handler.call(
            arguments={},
            idempotency_key="k1",
            run_id="run-1",
            tool="probe_mcp__auth_is_set",
        ) == "true"
        assert await handler.call(
            arguments={},
            idempotency_key="k2",
            run_id="run-2",
            tool="probe_mcp__auth_is_set",
        ) == "true"
    finally:
        await handler.aclose()


async def test_releasing_one_run_keeps_other_run_snapshot_available():
    handler = await _prepared()
    await handler.prepare_run("run-2")
    try:
        await handler.release_run("run-1")
        with pytest.raises(ToolkitError, match="not prepared for run 'run-1'"):
            handler.sub_tools_for("run-1")
        assert handler.sub_tools_for("run-2")
        assert handler.executable is True
    finally:
        await handler.aclose()


async def test_mcp_timeout_is_outcome_unknown_not_retryable(monkeypatch):
    handler = await _prepared()
    try:
        async def never_finishes(tool, arguments, idempotency_key, phase):
            phase.dispatched = True
            await asyncio.sleep(1)
            return "unreachable"

        monkeypatch.setattr(
            "ravana.runtime.toolkits.mcp_server._CALL_TIMEOUT_SECONDS", 0.001
        )
        handler._call_once = never_finishes
        with pytest.raises(ToolOutcomeUnknown):
            await handler.call(
                arguments={"a": 1, "b": 2},
                idempotency_key="k-timeout",
                run_id="run-1",
                tool="probe_mcp__add",
            )
    finally:
        await handler.aclose()


async def test_prefixed_subtool_timeout_keeps_side_effect_claim_started(
    monkeypatch, con
):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handler = McpServerHandler(
        "a",
        _config(),
        server=_server(read_only_tools=("x",)),
    )
    handler._pinned_by_run["run-1"] = {
        "a__x": Tool(name="a__x", description="side effect", input_schema={})
    }
    handler.executable = True

    async def never_finishes(tool, arguments, idempotency_key, phase):
        phase.dispatched = True
        await asyncio.sleep(1)

    monkeypatch.setattr(
        "ravana.runtime.toolkits.mcp_server._CALL_TIMEOUT_SECONDS", 0.001
    )
    monkeypatch.setattr(handler, "_call_once", never_finishes)
    executor = RavanaToolExecutor(con, {"a": handler})
    _seed_run(con, run_id="run-1")

    try:
        with pytest.raises(ToolOutcomeUnknown):
            await executor.execute(
                run_id="run-1",
                node_id="n",
                tool="a__a__x",
                arguments={},
                idempotency_key="prefixed-side-effect",
            )
        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("prefixed-side-effect",),
        ).fetchone()
        assert row["status"] == "STARTED"
    finally:
        await executor.aclose()


async def test_post_dispatch_cleanup_failure_keeps_the_ledger_indeterminate(
    monkeypatch, con
):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    handler = await _prepared()
    executor = RavanaToolExecutor(con, {"probe_mcp": handler})
    _seed_run(con, run_id="run-1")

    async def close_then_fail(stack):
        await stack.aclose()
        raise RuntimeError("session cleanup failed")

    monkeypatch.setattr(
        "ravana.runtime.toolkits.mcp_server._close_stack", close_then_fail
    )
    try:
        with pytest.raises(ToolOutcomeUnknown, match="outcome is unknown"):
            await executor.execute(
                run_id="run-1",
                node_id="n",
                tool="probe_mcp__add",
                arguments={"a": 1, "b": 2},
                idempotency_key="cleanup-failure",
            )
        row = con.execute(
            "SELECT status FROM tool_invocation WHERE idempotency_key = ?",
            ("cleanup-failure",),
        ).fetchone()
        assert row["status"] == "STARTED"
    finally:
        await executor.aclose()


async def test_session_cleanup_timeout_propagates_for_call_classification(
    monkeypatch,
):
    class SlowStack:
        async def aclose(self):
            await asyncio.sleep(1)

    monkeypatch.setattr(
        "ravana.runtime.toolkits.mcp_server._SHUTDOWN_TIMEOUT_SECONDS", 0.001
    )
    with pytest.raises(asyncio.TimeoutError):
        await _close_stack(SlowStack())


async def test_session_cleanup_cancellation_propagates_for_call_classification():
    class CancelledStack:
        async def aclose(self):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _close_stack(CancelledStack())


async def test_mcp_failure_before_tool_dispatch_is_transient(monkeypatch):
    handler = await _prepared()
    try:
        async def never_dispatches(tool, arguments, idempotency_key, phase):
            await asyncio.sleep(1)

        monkeypatch.setattr(
            "ravana.runtime.toolkits.mcp_server._CALL_TIMEOUT_SECONDS", 0.001
        )
        handler._call_once = never_dispatches
        with pytest.raises(ToolkitError, match="failed before tool dispatch") as exc_info:
            await handler.call(
                arguments={"a": 1, "b": 2},
                idempotency_key="k-before-call",
                run_id="run-1",
                tool="probe_mcp__add",
            )
        assert exc_info.value.kind.value == "transient"
    finally:
        await handler.aclose()


async def test_executor_rejects_a_qualified_name_collision(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    class RegularHandler:
        executable = True
        description = "regular"
        input_schema = {"type": "object"}

        def is_side_effecting(self, arguments):
            return False

        async def call(self, *, arguments, idempotency_key, run_id):
            return "regular"

        async def aclose(self):
            return None

    async def prepare():
        mcp = McpServerHandler("a", _config(), server=_server())
        await mcp.prepare_run("run-1")
        return mcp

    mcp = await prepare()
    executor = RavanaToolExecutor(con, {"a": mcp, "a__add": RegularHandler()})
    try:
        with pytest.raises(ToolkitError, match="collision"):
            # The colliding direct toolkit need not be granted to this agent;
            # otherwise the model would see a name that execute cannot resolve.
            executor.tools_for(["a"], run_id="run-1")
    finally:
        await executor.aclose()


async def test_executor_rejects_a_collision_from_an_ungranted_mcp_handler(con):
    from ravana.runtime.tool_executor import RavanaToolExecutor

    granted = await _prepared(toolkit_id="a")
    ungranted = await _prepared(toolkit_id="a__b")
    granted._pinned_by_run["run-1"]["b__add"] = granted._pinned_by_run[
        "run-1"
    ].pop("add")
    executor = RavanaToolExecutor(con, {"a": granted, "a__b": ungranted})
    try:
        with pytest.raises(ToolkitError, match="collision"):
            executor.tools_for(["a"], run_id="run-1")
    finally:
        await executor.aclose()


# --- parser -> launch seam ---------------------------------------------------
# Every other launch test builds an McpServerDefinition directly, and every
# other parser test stops at the parsed value. Nothing joined the two, which is
# how a command rewrite that made the parsed definition unlaunchable passed a
# green suite. These tests run what the parser actually stores.
async def test_parsed_definition_from_a_venv_interpreter_actually_starts():
    # sys.executable under a virtualenv IS a symlink to a base interpreter.
    # Resolving it discards sys.prefix, so the server loses its dependencies and
    # never starts — the ordinary configuration for a Python MCP server.
    parsed = parse_server_allowlist(
        {
            "probe": {
                "command": sys.executable,
                "args": [PROBE],
                "cwd": str(Path(PROBE).parent),
            }
        }
    )
    definition = parsed["probe"]

    handler = McpServerHandler("probe_mcp", _config(), server=definition)
    await handler.prepare_run("run-1")
    try:
        assert [t.name for t in handler.sub_tools_for("run-1")], "server offered no tools"
        assert await handler.call(
            arguments={"a": 2, "b": 3},
            idempotency_key="k1",
            run_id="run-1",
            tool=qualified_tool_name("probe_mcp", "add"),
        ) == "5"
    finally:
        await handler.aclose()


def test_parsing_keeps_the_interpreter_it_was_given():
    # The narrower unit assertion behind the launch test above, so a regression
    # names the cause instead of only the symptom.
    parsed = parse_server_allowlist(
        {"probe": {"command": sys.executable, "args": [], "cwd": str(Path(PROBE).parent)}}
    )
    assert parsed["probe"].command == sys.executable


def test_parsing_still_resolves_a_bare_name_to_an_absolute_path():
    # `which` resolution is the part that defeats a later PATH change, and it
    # must survive the fix above: the child is spawned by absolute path.
    bin_dir = str(Path(sys.executable).parent)
    name = Path(sys.executable).name
    parsed = parse_server_allowlist(
        {
            "probe": {
                "command": name,
                "args": [],
                "cwd": str(Path(PROBE).parent),
                "env": {"PATH": bin_dir},
            }
        }
    )
    stored = parsed["probe"].command
    assert Path(stored).is_absolute()
    assert Path(stored).parent == Path(bin_dir)
