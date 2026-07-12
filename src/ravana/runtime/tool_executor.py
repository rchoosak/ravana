"""RavanaToolExecutor — the real `ToolExecutor` (the gateway's injection seam,
§3.4/§3.6). It maps the tool name the model called to a toolkit handler and
executes it, with the idempotency ledger (§3.6/§8) wrapped around
*side-effecting* handlers:

  - For a side-effecting handler, look up `idempotency_key` in
    `tool_invocation`. A prior SUCCEEDED row => return its stored result
    WITHOUT re-executing — this is how a retried attempt (same
    content-addressed key) avoids double-firing a side effect (a second git
    push / ticket / email). Read-only handlers skip the ledger entirely so a
    repeated read returns live state, not the first cached response (§3.6
    scopes the dedup mandate to side effects).
  - A prior FAILED row is never reused: the point of a retry is to try again.

In this slice one toolkit == one callable tool (its id); mcp_server, which
exposes many sub-tools under one toolkit, gets its finer name->tool mapping
in the sandbox slice.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ravana.runtime.providers.base import Tool
from ravana.runtime.schema_validate import validate_json
from ravana.runtime.secrets import redact_secrets
from ravana.runtime.toolkits.base import ToolkitError, ToolkitHandler
from ravana.schema.util import now_iso


class RavanaToolExecutor:
    def __init__(self, con: sqlite3.Connection, handlers: dict[str, ToolkitHandler]):
        self._con = con
        self._handlers = handlers

    def tools_for(self, toolkit_ids: list[str]) -> list[Tool]:
        """The callable-tool specs (name/description/input_schema) the gateway
        offers the model for an agent, one per declared toolkit id. The
        compiler already rejects an agent referencing an unknown toolkit, so a
        missing handler here would be an internal bug — surface it loudly
        rather than silently dropping a tool the agent expects to have.

        A NON-executable (deferred) toolkit is refused rather than advertised:
        surfacing it would only invite the model to call a tool guaranteed to
        fail. Failing fast here tells the operator this build can't run the
        workflow yet (the deferred type lands in the sandbox slice)."""
        tools: list[Tool] = []
        for tid in toolkit_ids:
            handler = self._handlers.get(tid)
            if handler is None:
                raise ToolkitError(f"agent references toolkit '{tid}' with no registered handler")
            if not getattr(handler, "executable", True):
                raise ToolkitError(
                    f"toolkit '{tid}' is not executable in this build, so it cannot be offered to the model: "
                    f"{getattr(handler, 'description', 'deferred')}"
                )
            tools.append(Tool(name=tid, description=handler.description, input_schema=handler.input_schema))
        return tools

    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str:
        handler = self._handlers.get(tool)
        if handler is None:
            raise ToolkitError(f"agent called unknown tool '{tool}' (no toolkit registered under that id)")

        # §8(a): enforce the handler's declared input schema here, in the
        # runtime — a provider-side tool schema is a hint the model usually
        # follows, not a safety boundary. Rejecting bad args before dispatch
        # keeps a malformed/injected call from reaching the connector at all.
        schema_error = validate_json(arguments, getattr(handler, "input_schema", None))
        if schema_error is not None:
            raise ToolkitError(f"tool '{tool}' called with invalid arguments: {schema_error}")

        # §3.6 scopes dedup to side-effecting calls, and it's method-aware: an
        # api_connector POST dedupes (a retry must not double-fire), a GET does
        # not (a poll should re-read live state, not replay the first
        # response). A prior FAILED row is never reused: a retry should retry.
        side_effecting = handler.is_side_effecting(arguments)
        if side_effecting:
            prior = self._con.execute(
                "SELECT status, result FROM tool_invocation WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if prior is not None and prior["status"] == "SUCCEEDED":
                return prior["result"]

        try:
            result = await handler.call(arguments=arguments, idempotency_key=idempotency_key)
        except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise for the gateway
            if side_effecting:
                self._record(run_id, node_id, tool, idempotency_key, status="FAILED", error=redact_secrets(str(exc)))
            raise
        if side_effecting:
            self._record(run_id, node_id, tool, idempotency_key, status="SUCCEEDED", result=result)
        return result

    def _record(
        self, run_id: str, node_id: str, tool: str, key: str, *, status: str, result: str | None = None, error: str | None = None
    ) -> None:
        # INSERT OR REPLACE so a retried key that previously FAILED can be
        # overwritten with a SUCCEEDED row on a later successful attempt.
        self._con.execute(
            """INSERT OR REPLACE INTO tool_invocation
               (idempotency_key, run_id, node_id, tool, status, result, error, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (key, run_id, node_id, tool, status, result, error, now_iso()),
        )
        self._con.commit()
