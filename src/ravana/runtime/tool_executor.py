"""RavanaToolExecutor — the real `ToolExecutor` (the gateway's injection seam,
§3.4/§3.6). It maps the tool name the model called to a toolkit handler and
executes it, with the idempotency ledger (§3.6/§8) wrapped around
*side-effecting* handlers:

  - Before a side-effecting handler runs, atomically claim `idempotency_key` as
    STARTED. A prior SUCCEEDED row => return its stored result WITHOUT
    re-executing — this is how a retried attempt (same
    logical-invocation key) avoids double-firing a side effect (a second git
    push / ticket / email). Read-only handlers skip the ledger entirely so a
    repeated read returns live state, not the first cached response (§3.6
    scopes the dedup mandate to side effects).
  - A prior FAILED row is never reused: the point of a retry is to try again.
    A STARTED row is an indeterminate outcome left by a process crash; Ravana
    fails closed instead of risking the same side effect a second time.

Most toolkits are one callable tool named after the toolkit id. An
`mcp_server` toolkit (§1.7) instead surfaces the set of sub-tools it pinned
during run preparation, qualified as `<toolkit_id>__<tool>`; `_resolve` maps a
called name back by membership in that pinned set, never by string-splitting.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from ravana.runtime.providers.base import Tool
from ravana.runtime.schema_validate import validate_json
from ravana.runtime.secrets import redact_secrets
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolRetrySafeCancellation,
    ToolkitError,
    ToolkitHandler,
    ToolOutcomeUnknown,
)
from ravana.schema.util import now_iso


def _sub_tools_for(
    handler: ToolkitHandler,
    run_id: str | None,
    *,
    require_prepared: bool = True,
) -> list[Tool] | None:
    """Read a handler's optional run-scoped multi-tool capability."""
    sub_tools_for = getattr(handler, "sub_tools_for", None)
    if sub_tools_for is None:
        return None
    if run_id is None:
        raise ToolkitError("run_id is required to access run-scoped toolkit tools")
    is_prepared_for = getattr(handler, "is_prepared_for", None)
    if (
        not require_prepared
        and is_prepared_for is not None
        and not is_prepared_for(run_id)
    ):
        return []
    return sub_tools_for(run_id)


class RavanaToolExecutor:
    def __init__(self, con: sqlite3.Connection, handlers: dict[str, ToolkitHandler]):
        self._con = con
        self._handlers = handlers

    def _distinct_handlers(self) -> Iterator[ToolkitHandler]:
        """Yield each shared handler object once, preserving registry order."""
        seen: set[int] = set()
        for handler in self._handlers.values():
            identity = id(handler)
            if identity in seen:
                continue
            seen.add(identity)
            yield handler

    def tools_for(self, toolkit_ids: list[str], *, run_id: str | None = None) -> list[Tool]:
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
        sub_tool_names: set[str] = set()
        for tid in toolkit_ids:
            handler = self._handlers.get(tid)
            if handler is None:
                raise ToolkitError(f"agent references toolkit '{tid}' with no registered handler")
            if not getattr(handler, "executable", True):
                raise ToolkitError(
                    f"toolkit '{tid}' is not executable in this build, so it cannot be offered to the model: "
                    f"{getattr(handler, 'description', 'deferred')}"
                )
            # §1.7: one `mcp_server` toolkit stands for however many tools that
            # server publishes, so a handler may surface a *set* of qualified
            # tools instead of one named after the toolkit. The list is whatever
            # that handler pinned during run preparation — never re-read here.
            sub_tools = _sub_tools_for(handler, run_id)
            if sub_tools is not None:
                tools.extend(sub_tools)
                sub_tool_names.update(tool.name for tool in sub_tools)
                continue
            tools.append(Tool(name=tid, description=handler.description, input_schema=handler.input_schema))
        names = [tool.name for tool in tools]
        global_sub_tool_counts: dict[str, int] = {}
        if run_id is not None:
            for candidate in self._handlers.values():
                for spec in _sub_tools_for(
                    candidate, run_id, require_prepared=False
                ) or []:
                    global_sub_tool_counts[spec.name] = (
                        global_sub_tool_counts.get(spec.name, 0) + 1
                    )
        collisions = sorted(
            name
            for name in names
            if (name in sub_tool_names and global_sub_tool_counts.get(name, 0) != 1)
            or (name in self._handlers and name in sub_tool_names)
            or (name not in sub_tool_names and global_sub_tool_counts.get(name, 0) > 0)
        )
        if collisions:
            raise ToolkitError("callable tool name collision: " + ", ".join(collisions))
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ToolkitError(f"callable tool name collision: {', '.join(duplicates)}")
        return tools

    def _resolve(self, tool: str, run_id: str) -> tuple[ToolkitHandler, str | None]:
        """Map a called tool name to (handler, sub-tool or None).

        Resolution is by *membership*, not by splitting on the separator: a
        toolkit id may itself contain the separator, and a sub-tool is accepted
        only if the handler actually pinned it.
        """
        direct = self._handlers.get(tool)
        matches = [
            candidate
            for candidate in self._handlers.values()
            if any(
                spec.name == tool
                for spec in (
                    _sub_tools_for(candidate, run_id, require_prepared=False) or []
                )
            )
        ]
        if direct is not None and matches:
            raise ToolkitError(f"callable tool name collision for '{tool}'")
        if direct is not None:
            return direct, None
        if len(matches) == 1:
            return matches[0], tool
        if len(matches) > 1:
            raise ToolkitError(f"callable tool name collision for '{tool}'")
        raise ToolkitError(f"agent called unknown tool '{tool}' (no toolkit registered under that id)")

    async def prepare_run(self, run_id: str) -> None:
        """Prepare eager run resources; node-scoped tools are prepared later."""
        for handler in self._distinct_handlers():
            if getattr(handler, "prepare_on_demand", False):
                continue
            prepare = getattr(handler, "prepare_run", None)
            if prepare is not None:
                await prepare(run_id)

    async def prepare_tools(self, run_id: str, toolkit_ids: list[str]) -> None:
        """Prepare only on-demand handlers granted to the active node."""
        seen: set[int] = set()
        for toolkit_id in toolkit_ids:
            handler = self._handlers.get(toolkit_id)
            if handler is None:
                raise ToolkitError(
                    f"agent references toolkit '{toolkit_id}' with no registered handler"
                )
            identity = id(handler)
            if identity in seen or not getattr(handler, "prepare_on_demand", False):
                continue
            seen.add(identity)
            prepare = getattr(handler, "prepare_run", None)
            if prepare is not None:
                await prepare(run_id)

    async def hand_off_run(self, run_id: str) -> str | None:
        """Hand back the run's workspace via whichever handler owns it (§10.1).

        Only the workspace-owning handler implements this, so the first summary
        returned is the run's handoff; the loop keeps going past handlers that
        have nothing to say rather than assuming a position in the registry.
        """
        for handler in self._distinct_handlers():
            hand_off = getattr(handler, "hand_off_run", None)
            if hand_off is None:
                continue
            summary: str | None = await hand_off(run_id)
            if summary is not None:
                return summary
        return None

    async def aclose(self) -> None:
        """Close each distinct handler owned by this execution scope."""
        first_error: RuntimeError | None = None
        for handler in self._distinct_handlers():
            close = getattr(handler, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - continue closing siblings
                first_error = first_error or RuntimeError(
                    f"toolkit cleanup failed ({type(exc).__name__})"
                )
        if first_error is not None:
            raise first_error

    async def release_run(self, run_id: str) -> None:
        """Release only run-scoped handler state in a shared runtime."""
        for handler in self._distinct_handlers():
            release = getattr(handler, "release_run", None)
            if release is not None:
                await release(run_id)

    async def execute(
        self, *, run_id: str, node_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str
    ) -> str:
        handler, sub_tool = self._resolve(tool, run_id)

        # §8(a): enforce the handler's declared input schema here, in the
        # runtime — a provider-side tool schema is a hint the model usually
        # follows, not a safety boundary. Rejecting bad args before dispatch
        # keeps a malformed/injected call from reaching the connector at all.
        # For a multi-tool handler the schema is the SUB-tool's, since the
        # handler's own is a permissive placeholder.
        schema = getattr(handler, "input_schema", None)
        if sub_tool is not None:
            schema_for_tool = getattr(handler, "input_schema_for_tool", None)
            if schema_for_tool is not None:
                schema = schema_for_tool(run_id, sub_tool)
            else:
                schema = next(
                    (s.input_schema for s in (_sub_tools_for(handler, run_id) or []) if s.name == sub_tool),
                    schema,
                )
        schema_error = validate_json(arguments, schema)
        if schema_error is not None:
            raise ToolkitError(f"tool '{tool}' called with invalid arguments: {schema_error}")

        # §3.6 scopes dedup to side-effecting calls, and it's method-aware: an
        # api_connector POST dedupes (a retry must not double-fire), a GET does
        # not (a poll should re-read live state, not replay the first
        # response). A prior FAILED row is never reused: a retry should retry.
        side_effecting_for_tool = getattr(handler, "is_side_effecting_for_tool", None)
        if sub_tool is not None and side_effecting_for_tool is not None:
            side_effecting = side_effecting_for_tool(sub_tool, arguments)
        else:
            side_effecting = handler.is_side_effecting(arguments)
        if side_effecting:
            prior_result = self._claim(run_id, node_id, tool, idempotency_key)
            if prior_result is not None:
                return prior_result

        try:
            call_kwargs: dict[str, Any] = {}
            if sub_tool is not None:
                call_kwargs["tool"] = sub_tool
            result = await handler.call(
                arguments=arguments, idempotency_key=idempotency_key, run_id=run_id, **call_kwargs
            )
        except ToolOutcomeUnknown as exc:
            if side_effecting:
                self._record_indeterminate(
                    run_id,
                    node_id,
                    tool,
                    idempotency_key,
                    error=redact_secrets(str(exc)),
                )
            raise
        except ToolRetrySafeCancellation as exc:
            if side_effecting:
                self._record(
                    run_id,
                    node_id,
                    tool,
                    idempotency_key,
                    status="FAILED",
                    error=redact_secrets(str(exc)),
                )
            raise
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
        cursor = self._con.execute(
            """UPDATE tool_invocation
               SET status = ?, result = ?, error = ?
               WHERE idempotency_key = ? AND run_id = ? AND node_id = ?
                 AND tool = ? AND status = 'STARTED'""",
            (status, result, error, key, run_id, node_id, tool),
        )
        if cursor.rowcount != 1:
            self._con.rollback()
            raise RuntimeError("tool invocation lost its STARTED idempotency claim")
        self._con.commit()

    def _record_indeterminate(
        self,
        run_id: str,
        node_id: str,
        tool: str,
        key: str,
        *,
        error: str,
    ) -> None:
        cursor = self._con.execute(
            """UPDATE tool_invocation
               SET error = ?
               WHERE idempotency_key = ? AND run_id = ? AND node_id = ?
                 AND tool = ? AND status = 'STARTED'""",
            (error, key, run_id, node_id, tool),
        )
        if cursor.rowcount != 1:
            self._con.rollback()
            raise RuntimeError("tool invocation lost its indeterminate STARTED claim")
        self._con.commit()

    def _claim(self, run_id: str, node_id: str, tool: str, key: str) -> str | None:
        """Claim a side effect before dispatch, or replay/refuse a prior claim.

        The committed STARTED row closes the crash window where a handler could
        mutate external state and the process could die before any ledger row
        existed. An unresolved STARTED outcome is deliberately at-most-once:
        operator recovery is safer than silently firing the effect again.
        """
        cursor = self._con.execute(
            """INSERT OR IGNORE INTO tool_invocation
               (idempotency_key, run_id, node_id, tool, status, result, error, created_at)
               VALUES (?,?,?,?, 'STARTED', NULL, NULL, ?)""",
            (key, run_id, node_id, tool, now_iso()),
        )
        if cursor.rowcount == 1:
            self._con.commit()
            return None

        prior = self._load_claim(key)
        self._validate_claim_scope(prior, run_id, node_id, tool)
        if prior["status"] == "FAILED":
            retry = self._con.execute(
                """UPDATE tool_invocation
                   SET status = 'STARTED', result = NULL, error = NULL, created_at = ?
                   WHERE idempotency_key = ? AND status = 'FAILED'""",
                (now_iso(), key),
            )
            if retry.rowcount == 1:
                self._con.commit()
                return None
            prior = self._load_claim(key)
            self._validate_claim_scope(prior, run_id, node_id, tool)

        self._con.commit()
        if prior["status"] == "SUCCEEDED":
            result = prior["result"]
            if result is None:
                raise RuntimeError("SUCCEEDED tool invocation has no stored result")
            return str(result)
        if prior["status"] == "STARTED":
            raise ToolkitError(
                "a prior attempt may already have executed this side effect; refusing to execute it again",
                kind=ToolFailureKind.FATAL,
            )
        raise RuntimeError(f"unknown tool invocation status {prior['status']!r}")

    def _load_claim(self, key: str) -> sqlite3.Row:
        prior = self._con.execute(
            """SELECT run_id, node_id, tool, status, result
               FROM tool_invocation WHERE idempotency_key = ?""",
            (key,),
        ).fetchone()
        if prior is None:
            self._con.rollback()
            raise RuntimeError("idempotency claim disappeared")
        return prior

    def _validate_claim_scope(
        self, prior: sqlite3.Row, run_id: str, node_id: str, tool: str
    ) -> None:
        if (prior["run_id"], prior["node_id"], prior["tool"]) != (
            run_id,
            node_id,
            tool,
        ):
            self._con.commit()
            raise ToolkitError(
                "idempotency key collides with a different tool invocation",
                kind=ToolFailureKind.FATAL,
            )
