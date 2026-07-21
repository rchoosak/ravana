"""`mcp_server` toolkit (§1.7) — one MCP server exposing many sub-tools.

§1.7 makes MCP the default path for third-party integrations, so a single
`mcp_server` Toolkit stands for however many tools that server publishes,
rather than the one-toolkit-one-tool shape every other handler has.

Two §8 requirements shape this file more than the protocol does:

- **Endpoint allow-list.** The server command is checked against an
  admin-curated allow-list before anything is launched. §8: MCP servers "must
  come from an admin-curated allow-list of server endpoints, never an arbitrary
  URL any workflow author can paste into a Toolkit's `config`" — a workflow file
  is authored by whoever can edit workflows, and launching a subprocess named
  there would let that person run arbitrary local commands.
- **Tool-list pinning ("tool poisoning" / rug-pull).** A server's advertised
  tool list can change *after* it was approved. The list is read once during
  run preparation and pinned; calls are checked against that pinned set, and it
  is never re-read mid-run. A server that grows a `read_all_secrets` tool after
  preparation cannot have it offered to the model or invoked.

The session lives in its own task. `stdio_client` is an anyio cancel scope, and
anyio refuses an exit from a different task than the entry ("Attempted to exit
cancel scope in a different task than it was entered in" — reproduced). Holding
it open across `prepare_run` → `call` → `aclose` would therefore only work while
those happen to share a task, which is true of the single-process Local engine
today and silently false the moment anything dispatches concurrently. A worker
task that opens the session, serves requests off a queue, and closes it keeps
every enter/exit in one task by construction.
"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from ravana.runtime.providers.base import Tool
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolkitError,
)

# A qualified tool name is "<toolkit_id>SEP<sub_tool>". Two underscores keep it
# readable and match what provider tool-name charsets accept ([A-Za-z0-9_-]).
TOOL_NAME_SEPARATOR = "__"

_STARTUP_TIMEOUT_SECONDS = 30.0
_CALL_TIMEOUT_SECONDS = 120.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def qualified_tool_name(toolkit_id: str, sub_tool: str) -> str:
    return f"{toolkit_id}{TOOL_NAME_SEPARATOR}{sub_tool}"


@dataclass
class _Request:
    """One `call_tool` in flight, with the future its caller is waiting on."""

    tool: str
    arguments: dict[str, Any]
    future: asyncio.Future[str] = field(default_factory=asyncio.Future)


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
        get_auth_token: Any = None,
    ):
        transport = config.get("transport", "stdio")
        if transport != "stdio":
            # §1.7 pairs http/SSE with the hosted tiers; the Local tier is stdio.
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': transport {transport!r} is not supported "
                "in this build (stdio only; http/SSE lands with the hosted tier)",
                kind=ToolFailureKind.FATAL,
            )
        command = config.get("command")
        if not isinstance(command, str) or not command:
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': config.command is required for stdio transport",
                kind=ToolFailureKind.FATAL,
            )
        args = config.get("args", [])
        if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': config.args must be a list of strings",
                kind=ToolFailureKind.FATAL,
            )

        self._toolkit_id = toolkit_id
        self._command = command
        self._args = list(args)
        self._env = config.get("env") if isinstance(config.get("env"), dict) else None
        self._allowed_tools = _allowed_tools(config.get("allowed_tools"), toolkit_id)
        self._get_auth_token = get_auth_token

        self._pinned: dict[str, Tool] = {}
        # Only executable once `prepare_run` has pinned a tool list: before
        # that there is nothing to offer, and tools_for refuses a
        # non-executable handler rather than advertising an unusable tool.
        self.executable = False
        self._requests: asyncio.Queue[_Request | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self.description = f"[mcp_server] {toolkit_id} (not yet prepared)"

    @property
    def sub_tools(self) -> list[Tool]:
        """The pinned tool specs, qualified with this toolkit's id so two
        servers exposing a `search` tool stay distinguishable to the model."""
        return [
            Tool(
                name=qualified_tool_name(self._toolkit_id, name),
                description=spec.description,
                input_schema=spec.input_schema,
            )
            for name, spec in sorted(self._pinned.items())
        ]

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # MCP publishes an optional `annotations.readOnlyHint`, but it is a hint
        # from an untrusted server, not a guarantee. Treating every call as
        # side-effecting costs a re-read on retry; trusting a wrong hint would
        # double-fire a real side effect, so this fails safe.
        return True

    async def prepare_run(self, run_id: str) -> None:
        """Start the server and pin its tool list for the whole run."""
        if self._worker is not None:
            return  # already prepared for this handler's lifetime
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._worker = asyncio.create_task(self._serve(), name=f"mcp:{self._toolkit_id}")
        try:
            await asyncio.wait_for(self._ready, timeout=_STARTUP_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            await self.aclose()
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' did not start within "
                f"{_STARTUP_TIMEOUT_SECONDS:g}s",
                kind=ToolFailureKind.TRANSIENT,
            ) from exc
        except ToolkitError:
            await self.aclose()
            raise

    async def call(
        self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str, tool: str | None = None
    ) -> str:
        """Invoke one pinned sub-tool. `tool` is the qualified name the model
        called; the executor passes it through because one handler serves many."""
        sub_tool = self._resolve_sub_tool(tool)
        if self._worker is None or self._worker.done():
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' is not running",
                kind=ToolFailureKind.FATAL,
            )
        request = _Request(tool=sub_tool, arguments=arguments)
        await self._requests.put(request)
        try:
            return await asyncio.wait_for(request.future, timeout=_CALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            request.future.cancel()
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' tool {sub_tool!r} timed out after "
                f"{_CALL_TIMEOUT_SECONDS:g}s",
                kind=ToolFailureKind.TRANSIENT,
            ) from exc

    def _resolve_sub_tool(self, tool: str | None) -> str:
        if tool is None:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}' requires a qualified tool name",
                kind=ToolFailureKind.FATAL,
            )
        prefix = f"{self._toolkit_id}{TOOL_NAME_SEPARATOR}"
        sub_tool = tool[len(prefix):] if tool.startswith(prefix) else tool
        # The pinned set is the authority: a tool the server added after
        # preparation is refused here even though the server would serve it.
        if sub_tool not in self._pinned:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': tool {sub_tool!r} was not in the "
                "tool list pinned at run preparation"
            )
        return sub_tool

    async def aclose(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None:
            return
        await self._requests.put(None)  # sentinel: drain and shut down
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            worker.cancel()
        except Exception:  # noqa: BLE001 - a server dying during shutdown is not a run failure
            pass

    async def _serve(self) -> None:
        """Own the session for its entire lifetime, inside ONE task.

        Every anyio cancel scope this opens is also closed here, which is the
        constraint the module docstring describes.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert self._ready is not None
        stack = AsyncExitStack()
        try:
            env = dict(self._env) if self._env else None
            if self._get_auth_token is not None:
                token = self._get_auth_token()
                if token is not None:
                    # §8(c): resolved at dispatch, passed to the child by env
                    # only — never logged, never written into config.
                    env = {**(env or {}), "MCP_AUTH_TOKEN": str(token)}
            params = StdioServerParameters(
                command=self._command, args=self._args, env=env
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._pinned = await self._pin_tools(session)
            self.executable = bool(self._pinned)
            self.description = (
                f"[mcp_server] {self._toolkit_id}: {len(self._pinned)} tool(s) pinned at preparation"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to prepare_run's waiter
            if not self._ready.done():
                # A ToolkitError raised in here has already classified itself —
                # an allow-listed tool the server doesn't offer is FATAL config,
                # not a TRANSIENT startup blip. Re-wrapping would both bury the
                # actionable message and downgrade the kind, so the engine would
                # retry a misconfiguration that can never succeed.
                self._ready.set_exception(
                    exc
                    if isinstance(exc, ToolkitError)
                    else ToolkitError(
                        f"mcp_server '{self._toolkit_id}' failed to start "
                        f"({type(exc).__name__})",
                        kind=ToolFailureKind.TRANSIENT,
                    )
                )
            await stack.aclose()
            return

        if not self._ready.done():
            self._ready.set_result(None)
        try:
            while True:
                request = await self._requests.get()
                if request is None:
                    return
                if request.future.cancelled():
                    continue
                try:
                    result = await session.call_tool(request.tool, request.arguments)
                    request.future.set_result(_render(result, self._toolkit_id, request.tool))
                except Exception as exc:  # noqa: BLE001 - one call's failure, not the session's
                    if not request.future.done():
                        request.future.set_exception(
                            ToolkitError(
                                f"mcp_server '{self._toolkit_id}' tool {request.tool!r} "
                                f"failed ({type(exc).__name__})",
                                kind=ToolFailureKind.TRANSIENT,
                            )
                        )
        finally:
            await stack.aclose()

    async def _pin_tools(self, session: Any) -> dict[str, Tool]:
        listed = await session.list_tools()
        pinned: dict[str, Tool] = {}
        for spec in listed.tools:
            if self._allowed_tools is not None and spec.name not in self._allowed_tools:
                continue
            schema = spec.inputSchema if isinstance(spec.inputSchema, dict) else {}
            pinned[spec.name] = Tool(
                name=spec.name,
                # A server-supplied description is untrusted text the model
                # reads (§8 prompt injection); it is passed through as the tool
                # description but never treated as instruction by Ravana itself.
                description=(spec.description or f"{spec.name} (via {self._toolkit_id})"),
                input_schema=schema,
            )
        if self._allowed_tools is not None:
            missing = sorted(self._allowed_tools - set(pinned))
            if missing:
                raise ToolkitError(
                    f"mcp_server '{self._toolkit_id}': allow-listed tools not offered "
                    f"by the server: {', '.join(missing)}",
                    kind=ToolFailureKind.FATAL,
                )
        if not pinned:
            raise ToolkitError(
                f"mcp_server '{self._toolkit_id}': server offered no usable tools",
                kind=ToolFailureKind.FATAL,
            )
        return pinned


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
    toolkit_id: str, config: dict[str, Any], allowlist: set[str] | None
) -> None:
    """§8: refuse an MCP endpoint that an admin has not curated.

    Fails CLOSED on an unconfigured allow-list. A workflow file is editable by
    anyone who can author workflows, so treating "no allow-list configured" as
    "allow anything" would turn `config.command` into arbitrary local command
    execution by whoever can open a PR against a workflow YAML.
    """
    command = config.get("command")
    if allowlist is None or not allowlist:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}' is not usable: no MCP server allow-list is "
            "configured. §8 requires MCP endpoints be admin-curated; add the command "
            "to `mcp.allowed_servers` in .ravana/config.yaml.",
            kind=ToolFailureKind.FATAL,
        )
    if not isinstance(command, str) or command not in allowlist:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': command {command!r} is not in the "
            "admin-curated allow-list (`mcp.allowed_servers` in .ravana/config.yaml)",
            kind=ToolFailureKind.FATAL,
        )
    resolved = shutil.which(command)
    if resolved is None:
        raise ToolkitError(
            f"mcp_server '{toolkit_id}': allow-listed command {command!r} was not "
            "found on PATH",
            kind=ToolFailureKind.FATAL,
        )
