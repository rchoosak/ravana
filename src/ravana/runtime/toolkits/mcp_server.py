"""`mcp_server` toolkit (§1.7) — one MCP server exposing many sub-tools.

§1.7 makes MCP the default path for third-party integrations, so a single
`mcp_server` Toolkit stands for however many tools that server publishes,
rather than the one-toolkit-one-tool shape every other handler has.

Two §8 requirements shape this file more than the protocol does:

- **Endpoint allow-list.** A workflow selects an admin-owned server definition
  by name. The definition fixes the complete executable, argv, and environment
  before anything is launched; workflow config cannot supply any of them.
- **Tool-list pinning ("tool poisoning" / rug-pull).** A server's advertised
  tool list can change *after* it was approved. The list is read once during
  run preparation and pinned; calls are checked against that pinned set, and it
  is never re-read mid-run. A server that grows a `read_all_secrets` tool after
  preparation cannot have it offered to the model or invoked.

Each discovery and call owns a short-lived session in its current task. This
keeps anyio's cancel scope enter/exit task-affine, gives every dispatch its
fresh credential environment, and prevents server-side session state crossing
run boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ravana.runtime.providers.base import Tool
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolRetrySafeCancellation,
    ToolOutcomeUnknown,
    ToolkitError,
)
from ravana.runtime.secrets import (
    ResolvedSecret,
    SecretLeakError,
    ensure_secret_free,
    redact_secrets,
)
from ravana.schema.util import now_iso

# A qualified tool name is "<toolkit_id>SEP<sub_tool>". Two underscores keep it
# readable and match what provider tool-name charsets accept ([A-Za-z0-9_-]).
TOOL_NAME_SEPARATOR = "__"

_STARTUP_TIMEOUT_SECONDS = 30.0
_CALL_TIMEOUT_SECONDS = 120.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_DEFAULT_AUTH_ENV = "MCP_AUTH_TOKEN"
_IDEMPOTENCY_META_KEY = "ravana/idempotency_key"
_ORPHAN_SNAPSHOT_GRACE_SECONDS = 3600
# `mcp.client.stdio.stdio_client` merges these host variables into the child
# environment even when `StdioServerParameters.env` is supplied. Empty values
# neutralize that SDK default while still allowing the admin definition below
# to explicitly provide one of them.
_STDIO_CLIENT_INHERITED_ENV_VARS = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")


def qualified_tool_name(toolkit_id: str, sub_tool: str) -> str:
    return f"{toolkit_id}{TOOL_NAME_SEPARATOR}{sub_tool}"


@dataclass(frozen=True)
class McpServerDefinition:
    """An administrator-owned, immutable stdio launch definition.

    `auth_env` maps the toolkit's resolved `auth_ref` to the environment name
    the selected server actually understands. `authenticate_discovery` is an
    explicit opt-in for servers that require that credential during MCP
    initialize/list_tools; the default keeps discovery lazy.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    auth_env: str = _DEFAULT_AUTH_ENV
    authenticate_discovery: bool = False
    read_only_tools: tuple[str, ...] = ()

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.env)

    @property
    def fingerprint(self) -> str:
        """Stable identity for the admin-owned launch definition."""
        payload = {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "auth_env": self.auth_env,
            "authenticate_discovery": self.authenticate_discovery,
            "read_only_tools": self.read_only_tools,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def parse_server_allowlist(raw: Any) -> dict[str, McpServerDefinition] | None:
    """Parse complete server definitions from install-owned configuration.

    The workflow may select a definition by name, but cannot contribute argv or
    environment values. Invalid configuration fails closed for the whole list.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    parsed: dict[str, McpServerDefinition] = {}
    for name, definition in raw.items():
        if not isinstance(name, str) or not name or not isinstance(definition, dict):
            raise ToolkitError("mcp.allowed_servers must map names to server definitions", kind=ToolFailureKind.FATAL)
        command = definition.get("command")
        args = definition.get("args", [])
        env = definition.get("env", {})
        auth_env = definition.get("auth_env", _DEFAULT_AUTH_ENV)
        authenticate_discovery = definition.get("authenticate_discovery", False)
        read_only_tools = definition.get("read_only_tools", [])
        if not isinstance(command, str) or not command:
            raise ToolkitError(f"MCP server definition '{name}' needs a command", kind=ToolFailureKind.FATAL)
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ToolkitError(f"MCP server definition '{name}' args must be strings", kind=ToolFailureKind.FATAL)
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
        ):
            raise ToolkitError(f"MCP server definition '{name}' env must be string-valued", kind=ToolFailureKind.FATAL)
        if not isinstance(auth_env, str) or not auth_env or "=" in auth_env or "\x00" in auth_env:
            raise ToolkitError(
                f"MCP server definition '{name}' auth_env must be a valid environment name",
                kind=ToolFailureKind.FATAL,
            )
        if not isinstance(authenticate_discovery, bool):
            raise ToolkitError(
                f"MCP server definition '{name}' authenticate_discovery must be boolean",
                kind=ToolFailureKind.FATAL,
            )
        if not isinstance(read_only_tools, list) or any(
            not isinstance(tool, str) or not tool for tool in read_only_tools
        ):
            raise ToolkitError(
                f"MCP server definition '{name}' read_only_tools must be tool names",
                kind=ToolFailureKind.FATAL,
            )
        if not os.path.isabs(command) and "PATH" not in env:
            raise ToolkitError(
                f"MCP server definition '{name}' must provide env.PATH for command {command!r}",
                kind=ToolFailureKind.FATAL,
            )
        resolved = shutil.which(command, path=env.get("PATH", ""))
        if resolved is None:
            raise ToolkitError(
                f"MCP server definition '{name}' command {command!r} was not found on PATH",
                kind=ToolFailureKind.FATAL,
            )
        parsed[name] = McpServerDefinition(
            name=name,
            command=os.path.realpath(resolved),
            args=tuple(args),
            env=tuple(sorted(env.items())),
            auth_env=auth_env,
            authenticate_discovery=authenticate_discovery,
            read_only_tools=tuple(read_only_tools),
        )
    return parsed


@dataclass
class _CallPhase:
    dispatched: bool = False


class McpServerHandler:
    """Runs one MCP stdio server and exposes its (allow-listed) tools.

    `executable` is True only once `prepare_run` has pinned a tool list: before
    that there is nothing to offer, and `tools_for` refuses non-executable
    handlers rather than advertising a tool that cannot run.
    """

    # Unused: this handler routes per sub-tool, each with the schema the server
    # published. Kept because ToolkitHandler declares it.
    input_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}

    def __init__(
        self,
        toolkit_id: str,
        config: dict[str, Any],
        *,
        server: McpServerDefinition,
        get_auth_token: Any = None,
        snapshot_con: sqlite3.Connection | None = None,
    ):
        transport = config.get("transport", "stdio")
        if transport != "stdio":
            # §1.7 pairs http/SSE with the hosted tiers; the Local tier is stdio.
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': transport {transport!r} is not supported "
                "in this build (stdio only; http/SSE lands with the hosted tier)",
                kind=ToolFailureKind.FATAL,
            )
        self._toolkit_id = toolkit_id
        self._server = server
        self._allowed_tools = _allowed_tools(config.get("allowed_tools"), toolkit_id)
        self._get_auth_token = get_auth_token
        self._snapshot_con = snapshot_con

        self._pinned_by_run: dict[str, dict[str, Tool]] = {}
        # Only executable once `prepare_run` has pinned a tool list: before
        # that there is nothing to offer, and tools_for refuses a
        # non-executable handler rather than advertising an unusable tool.
        self.executable = False
        self.description = f"[mcp_server] {toolkit_id} (not yet prepared)"

    def sub_tools_for(self, run_id: str) -> list[Tool]:
        if run_id not in self._pinned_by_run:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' was not prepared for run '{run_id}'",
                kind=ToolFailureKind.FATAL,
            )
        return [
            Tool(
                name=qualified_tool_name(self._toolkit_id, name),
                description=spec.description,
                input_schema=spec.input_schema,
            )
            for name, spec in sorted(self._pinned_by_run.get(run_id, {}).items())
        ]

    def input_schema_for_tool(self, run_id: str, tool: str) -> dict[str, Any]:
        sub_tool = self._resolve_sub_tool(run_id, tool)
        return self._pinned_by_run[run_id][sub_tool].input_schema

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # MCP publishes an optional `annotations.readOnlyHint`, but it is a hint
        # from an untrusted server, not a guarantee. Treating every call as
        # side-effecting costs a re-read on retry; trusting a wrong hint would
        # double-fire a real side effect, so this fails safe.
        return True

    def is_side_effecting_for_tool(self, tool: str, arguments: dict[str, Any]) -> bool:
        """Use only the admin-owned read-only declaration for MCP tools."""
        prefix = f"{self._toolkit_id}{TOOL_NAME_SEPARATOR}"
        if tool.startswith(prefix):
            tool = tool[len(prefix):]
        return tool not in self._server.read_only_tools

    async def prepare_run(self, run_id: str) -> None:
        """Discover and pin tools for one run without retaining a session."""
        if run_id in self._pinned_by_run:
            return
        pinned = self._restore_snapshot(run_id)
        if pinned is None:
            try:
                pinned = await asyncio.wait_for(
                    self._discover_tools(), timeout=_STARTUP_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError as exc:
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}' did not start within "
                    f"{_STARTUP_TIMEOUT_SECONDS:g}s",
                    kind=ToolFailureKind.TRANSIENT,
                ) from exc
            self._persist_snapshot(run_id, pinned)
        self._pinned_by_run[run_id] = pinned
        self.executable = True
        self.description = (
            f"[mcp_server] {self._toolkit_id}: {len(pinned)} tool(s) pinned at preparation"
        )

    async def call(
        self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str, tool: str | None = None
    ) -> str:
        """Invoke one pinned sub-tool. `tool` is the qualified name the model
        called; the executor passes it through because one handler serves many."""
        sub_tool = self._resolve_sub_tool(run_id, tool)
        if run_id not in self._pinned_by_run:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' was not prepared for run '{run_id}'",
                kind=ToolFailureKind.FATAL,
            )
        try:
            phase = _CallPhase()
            return await asyncio.wait_for(
                self._call_once(sub_tool, arguments, idempotency_key, phase),
                timeout=_CALL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            if phase.dispatched:
                if self.is_side_effecting_for_tool(sub_tool, arguments):
                    raise ToolOutcomeUnknown(
                        f"mcp_server '{self._toolkit_id}' tool {sub_tool!r} timed out after "
                        f"{_CALL_TIMEOUT_SECONDS:g}s",
                    ) from exc
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}' read-only tool {sub_tool!r} timed out after "
                    f"{_CALL_TIMEOUT_SECONDS:g}s",
                    kind=ToolFailureKind.TRANSIENT,
                ) from exc
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' failed before tool dispatch after "
                f"{_CALL_TIMEOUT_SECONDS:g}s",
                kind=ToolFailureKind.TRANSIENT,
            ) from exc
        except asyncio.CancelledError as exc:
            if phase.dispatched:
                if self.is_side_effecting_for_tool(sub_tool, arguments):
                    raise ToolOutcomeUnknown(
                        f"mcp_server '{self._toolkit_id}' tool {sub_tool!r} was cancelled "
                        "after dispatch; outcome is unknown",
                    ) from exc
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}' read-only tool {sub_tool!r} was cancelled "
                    "after dispatch",
                    kind=ToolFailureKind.TRANSIENT,
                ) from exc
            raise ToolRetrySafeCancellation(
                f"mcp_server '{self._toolkit_id}' tool {sub_tool!r} was cancelled before dispatch"
            ) from exc

    def _resolve_sub_tool(self, run_id: str, tool: str | None) -> str:
        if tool is None:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' requires a qualified tool name",
                kind=ToolFailureKind.FATAL,
            )
        prefix = f"{self._toolkit_id}{TOOL_NAME_SEPARATOR}"
        sub_tool = tool[len(prefix):] if tool.startswith(prefix) else tool
        # The pinned set is the authority: a tool the server added after
        # preparation is refused here even though the server would serve it.
        if sub_tool not in self._pinned_by_run.get(run_id, {}):
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': tool {sub_tool!r} was not in the "
                "tool list pinned at run preparation"
            )
        return sub_tool

    async def aclose(self) -> None:
        self._pinned_by_run.clear()
        self.executable = False

    async def release_run(self, run_id: str) -> None:
        """Drop only one run's pinned snapshot from a shared runtime."""
        self._pinned_by_run.pop(run_id, None)
        self.executable = bool(self._pinned_by_run)
        if self._snapshot_con is not None:
            row = self._snapshot_con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
            if row is None or row["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                self._snapshot_con.execute("DELETE FROM mcp_tool_snapshot WHERE run_id = ?", (run_id,))
                self._snapshot_con.commit()

    def _snapshot_fingerprint(self) -> str:
        payload = {
            "server": self._server.fingerprint,
            "allowed_tools": sorted(self._allowed_tools) if self._allowed_tools is not None else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _restore_snapshot(self, run_id: str) -> dict[str, Tool] | None:
        if self._snapshot_con is None:
            return None
        rows = self._snapshot_con.execute(
            """SELECT server_fingerprint, tool_name, description, input_schema
               FROM mcp_tool_snapshot
               WHERE run_id = ? AND toolkit_id = ?
               ORDER BY tool_name""",
            (run_id, self._toolkit_id),
        ).fetchall()
        if not rows:
            return None
        expected = self._snapshot_fingerprint()
        if any(row["server_fingerprint"] != expected for row in rows):
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': admin definition or tool grant changed "
                f"after run '{run_id}' was prepared",
                kind=ToolFailureKind.FATAL,
            )
        pinned: dict[str, Tool] = {}
        try:
            for row in rows:
                schema = json.loads(row["input_schema"])
                if not isinstance(schema, dict):
                    raise ValueError("input schema is not an object")
                pinned[row["tool_name"]] = Tool(
                    name=row["tool_name"],
                    description=row["description"],
                    input_schema=schema,
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': stored tool snapshot is invalid",
                kind=ToolFailureKind.FATAL,
            ) from exc
        return pinned

    def _persist_snapshot(self, run_id: str, pinned: dict[str, Tool]) -> None:
        if self._snapshot_con is None:
            return
        fingerprint = self._snapshot_fingerprint()
        self._snapshot_con.execute(
            "DELETE FROM mcp_tool_snapshot WHERE run_id = ? AND toolkit_id = ?",
            (run_id, self._toolkit_id),
        )
        self._snapshot_con.executemany(
            """INSERT INTO mcp_tool_snapshot
               (run_id, toolkit_id, server_fingerprint, tool_name, description, input_schema,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    self._toolkit_id,
                    fingerprint,
                    name,
                    spec.description,
                    json.dumps(spec.input_schema, sort_keys=True, separators=(",", ":")),
                    now_iso(),
                )
                for name, spec in pinned.items()
            ],
        )
        self._snapshot_con.commit()

    async def _discover_tools(self) -> dict[str, Tool]:
        """Open a short-lived session solely to pin its advertised tools."""
        try:
            # Tool discovery must not read an auth_ref for a toolkit that may
            # never be called. The dispatch session below resolves it lazily.
            env, secret_values = self._child_environment(
                include_auth=self._server.authenticate_discovery
            )
            async with self._open_session(env) as session:
                return await self._pin_tools(session, secret_values=secret_values)
        except ToolkitError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize startup failures
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' failed to start ({type(exc).__name__})",
                kind=ToolFailureKind.TRANSIENT,
            ) from exc

    async def _call_once(
        self, tool: str, arguments: dict[str, Any], idempotency_key: str, phase: _CallPhase
    ) -> str:
        """Run one call in a session owned and closed by this task."""
        secret_values: tuple[str, ...] = ()
        try:
            env, secret_values = self._child_environment(include_auth=True)
            async with self._open_session(env) as session:
                phase.dispatched = True
                result = await session.call_tool(
                    tool,
                    arguments,
                    meta={_IDEMPOTENCY_META_KEY: idempotency_key},
                )
                rendered = _render(result, self._toolkit_id, tool)
                try:
                    ensure_secret_free(
                        rendered,
                        context=f"mcp_server '{self._toolkit_id}' tool result",
                        values=secret_values,
                    )
                except SecretLeakError as exc:
                    if phase.dispatched:
                        raise ToolOutcomeUnknown(
                            f"mcp_server '{self._toolkit_id}' tool {tool!r} returned credential material; "
                            "outcome is unknown",
                        ) from None
                    raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None
                return rendered
        except ToolkitError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize process/tool failures
            safe_error = redact_secrets(str(exc), values=secret_values)
            if phase.dispatched and self.is_side_effecting_for_tool(tool, arguments):
                raise ToolOutcomeUnknown(
                    f"mcp_server '{self._toolkit_id}' tool {tool!r} failed "
                    f"({type(exc).__name__}); outcome is unknown: {safe_error}"
                ) from None
            if phase.dispatched:
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}' read-only tool {tool!r} failed "
                    f"after dispatch ({type(exc).__name__}): {safe_error}",
                    kind=ToolFailureKind.TRANSIENT,
                ) from None
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' failed before tool dispatch "
                f"({type(exc).__name__}): {safe_error}",
                kind=ToolFailureKind.TRANSIENT,
            ) from None

    @asynccontextmanager
    async def _open_session(self, env: dict[str, str]) -> AsyncIterator[Any]:
        """Open, initialize, and close one task-affine MCP session."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        try:
            params = StdioServerParameters(
                command=self._server.command,
                args=list(self._server.args),
                env=env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            yield session
        finally:
            # This remains inside each caller's classification boundary. A
            # post-dispatch cleanup error must become outcome-unknown, never an
            # ordinary FAILED invocation that is safe to retry.
            await _close_stack(stack)

    def _child_environment(self, *, include_auth: bool) -> tuple[dict[str, str], tuple[str, ...]]:
        # The server definition is the complete environment boundary. Do not
        # inherit the host process environment, which may contain unrelated
        # Ravana/provider credentials; only the dispatch credential below is
        # intentionally added at the child-process seam.
        env = {name: "" for name in _STDIO_CLIENT_INHERITED_ENV_VARS}
        env.update(self._server.environment)
        if include_auth and self._get_auth_token is not None:
            try:
                token = self._get_auth_token()
            except Exception as exc:  # noqa: BLE001 - credential errors are fatal
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}' credential resolution failed "
                    f"({type(exc).__name__})",
                    kind=ToolFailureKind.FATAL,
                ) from None
            if token is not None:
                if isinstance(token, ResolvedSecret):
                    value = token.value()
                elif isinstance(token, str):
                    value = token
                else:
                    raise ToolkitError(
                        f"mcp_server '{self._toolkit_id}' credential provider returned "
                        f"unsupported type {type(token).__name__}",
                        kind=ToolFailureKind.FATAL,
                    )
                env[self._server.auth_env] = value
                return env, (value,)
        return env, ()

    async def _pin_tools(self, session: Any, *, secret_values: tuple[str, ...] = ()) -> dict[str, Tool]:
        pinned: dict[str, Tool] = {}
        offered_names: set[str] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        while True:
            if cursor is not None:
                if cursor in seen_cursors:
                    raise ToolkitError(
                        f"mcp_server '{self._toolkit_id}': server returned a looping tool-list cursor",
                        kind=ToolFailureKind.FATAL,
                    )
                seen_cursors.add(cursor)
            listed = await session.list_tools(cursor=cursor)
            for spec in listed.tools:
                try:
                    ensure_secret_free(
                        spec.name,
                        context=f"mcp_server '{self._toolkit_id}' tool name",
                        values=secret_values,
                    )
                except SecretLeakError as exc:
                    raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None
                offered_names.add(spec.name)
                if self._allowed_tools is not None and spec.name not in self._allowed_tools:
                    continue
                schema = spec.inputSchema if isinstance(spec.inputSchema, dict) else {}
                description = spec.description or f"{spec.name} (via {self._toolkit_id})"
                try:
                    ensure_secret_free(
                        description,
                        context=f"mcp_server '{self._toolkit_id}' tool description",
                        values=secret_values,
                    )
                    ensure_secret_free(
                        schema,
                        context=f"mcp_server '{self._toolkit_id}' tool schema",
                        values=secret_values,
                    )
                except SecretLeakError as exc:
                    raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None
                pinned[spec.name] = Tool(
                    name=spec.name,
                    # A server-supplied description is untrusted text the model
                    # reads (§8 prompt injection); it is passed through as the tool
                    # description but never treated as instruction by Ravana itself.
                    description=description,
                    input_schema=schema,
                )
            next_cursor = getattr(listed, "nextCursor", None)
            if next_cursor is None:
                break
            if (
                not isinstance(next_cursor, str)
                or not next_cursor
                or next_cursor == cursor
                or next_cursor in seen_cursors
            ):
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}': server returned an invalid tool-list cursor",
                    kind=ToolFailureKind.FATAL,
                )
            cursor = next_cursor
        if self._allowed_tools is not None:
            missing = sorted(self._allowed_tools - set(pinned))
            if missing:
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}': allow-listed tools not offered "
                    f"by the server: {', '.join(missing)}",
                    kind=ToolFailureKind.FATAL,
                )
        # A workflow allow-list may intentionally hide an admin-declared
        # read-only tool. Validate the admin declaration against the complete
        # server advertisement, not the workflow-filtered exposed set.
        missing_read_only = sorted(set(self._server.read_only_tools) - offered_names)
        if missing_read_only:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': admin read-only tools not offered "
                f"by the server: {', '.join(missing_read_only)}",
                kind=ToolFailureKind.FATAL,
            )
        if not pinned:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': server offered no usable tools",
                kind=ToolFailureKind.FATAL,
            )
        return pinned


def cleanup_mcp_tool_snapshots(con: sqlite3.Connection) -> int:
    """Remove snapshots that cannot be resumed by a live run.

    Preparation precedes run insertion, so the snapshot table cannot use a
    foreign key without making normal preparation impossible. This cleanup is
    the crash-recovery counterpart: a fresh runtime removes terminal rows and
    orphan rows older than the preparation grace period.
    """
    orphan_cutoff = (
        datetime.now(timezone.utc)
        - timedelta(seconds=_ORPHAN_SNAPSHOT_GRACE_SECONDS)
    ).isoformat()
    cursor = con.execute(
        """DELETE FROM mcp_tool_snapshot
           WHERE (NOT EXISTS (
                      SELECT 1 FROM run WHERE run.id = mcp_tool_snapshot.run_id
                  )
                  AND (created_at = '' OR created_at < ?))
              OR run_id IN (
                     SELECT id FROM run WHERE status IN ('COMPLETED', 'FAILED', 'CANCELLED')
                 )""",
        (orphan_cutoff,),
    )
    con.commit()
    return cursor.rowcount


async def _close_stack(stack: AsyncExitStack) -> None:
    # `wait_for(coro)` schedules `coro` in a new task. AnyIO's stdio cancel
    # scopes are task-affine, so cleanup stays in this task. Timeout and
    # cancellation deliberately propagate to the caller's dispatch-phase
    # classifier: after dispatch they must retain an indeterminate STARTED
    # ledger claim rather than being mistaken for successful cleanup.
    async with asyncio.timeout(_SHUTDOWN_TIMEOUT_SECONDS):
        await stack.aclose()

def _allowed_tools(raw: Any, toolkit_id: str) -> set[str] | None:
    """`config.allowed_tools` narrows what the server may expose. None means
    "whatever the server publishes", which is only as trustworthy as the
    allow-listed endpoint itself."""
    if raw is None:
        return None
    if not isinstance(raw, list) or any(not isinstance(t, str) or not t for t in raw):
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': config.allowed_tools must be a list of tool names",
            kind=ToolFailureKind.FATAL,
        )
    if not raw:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': config.allowed_tools is empty, which would "
            "expose nothing — omit it to allow every tool the server publishes",
            kind=ToolFailureKind.FATAL,
        )
    return set(raw)


def _render(result: Any, toolkit_id: str, tool: str) -> str:
    """Flatten an MCP CallToolResult into the string a tool result must be.

    A server-reported error is returned as text, not raised: §3.6 routes a
    tool's own error back to the model as MODEL_ADDRESSABLE so it can adjust,
    and a remote tool refusing a bad argument is exactly that case.
    """
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[{getattr(item, 'type', 'content')} omitted]")
    body = "\n".join(parts).strip() or "(no content)"
    if getattr(result, "isError", False):
        return f"ERROR from {toolkit_id}.{tool}: {body}"
    return body


def check_endpoint_allowed(
    toolkit_id: str,
    config: dict[str, Any],
    allowlist: dict[str, McpServerDefinition] | None,
) -> McpServerDefinition:
    """§8: refuse an MCP endpoint that an admin has not curated.

    The workflow contributes only a server name and safe tool narrowing. The
    admin-owned definition supplies the complete command, arguments, and env.
    """
    if allowlist is None or not allowlist:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}' is not usable: no MCP server allow-list is "
            "configured. §8 requires complete server definitions in "
            "`mcp.allowed_servers` in .ravana/config.yaml.",
            kind=ToolFailureKind.FATAL,
        )
    if any(key in config for key in ("command", "args", "env")):
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': command, args, and env must be supplied "
            "by the admin server definition, not the workflow",
            kind=ToolFailureKind.FATAL,
        )
    server_name = config.get("server")
    if not isinstance(server_name, str) or not server_name:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': config.server must name an admin-curated "
            "MCP server definition",
            kind=ToolFailureKind.FATAL,
        )
    definition = allowlist.get(server_name)
    if definition is None:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': server {server_name!r} is not in the "
            "admin-curated allow-list (`mcp.allowed_servers` in .ravana/config.yaml)",
            kind=ToolFailureKind.FATAL,
        )
    return definition
