"""Builds the set of toolkit handlers for a run from the compiled graph's
toolkit configs (§2.2 `toolkit` rows) + a secret resolver. Only the toolkit
types implemented in this slice are wired; the heavyweight, external-service
types (`code_interpreter` Docker sandbox, `mcp_server` stdio, `web_search`)
raise a clear ToolkitError pointing at the slice that owns them, rather than
silently accepting a config they can't honor.
"""

from __future__ import annotations

from typing import Any

from ravana.compiler.graph import CompiledGraph
from ravana.runtime.secrets import ResolvedSecret, SecretResolver, ensure_resolved
from ravana.runtime.toolkits.api_connector import ApiConnectorHandler
from ravana.runtime.toolkits.base import ToolkitError, ToolkitHandler

# Toolkit types deferred to the sandbox slice, with the reason each needs it.
_DEFERRED = {
    "code_interpreter": "Docker sandbox execution — highest blast radius (§8), lands in the sandbox slice",
    "mcp_server": "stdio MCP subprocess via the mcp SDK — lands in the sandbox slice",
    "web_search": "provider HTTP call (e.g. Tavily) — lands with the connector-providers slice",
}


def build_registry(
    graph: CompiledGraph,
    resolver: SecretResolver,
    *,
    clients: dict[str, Any] | None = None,
) -> dict[str, ToolkitHandler]:
    """Returns {toolkit_id: handler}. `clients` optionally injects a per-toolkit
    transport (tests pass fakes keyed by toolkit id)."""
    clients = clients or {}
    handlers: dict[str, ToolkitHandler] = {}
    for toolkit_id, toolkit in graph.toolkits_by_id.items():
        if toolkit.type == "api_connector":
            # §8(c): the runtime injects a credential *provider* that resolves
            # lazily at dispatch — the connector never receives the `auth_ref`
            # pointer or the resolver, and a run that never calls this toolkit
            # never needs its secret present.
            handlers[toolkit_id] = ApiConnectorHandler(
                config=toolkit.config,
                get_auth_token=_auth_provider(resolver, toolkit.auth_ref),
                client=clients.get(toolkit_id),
            )
        elif toolkit.type in _DEFERRED:
            handlers[toolkit_id] = _DeferredHandler(toolkit_id, toolkit.type)
        else:
            handlers[toolkit_id] = _DeferredHandler(toolkit_id, toolkit.type, reason=f"unknown toolkit type '{toolkit.type}'")
    return handlers


def _auth_provider(resolver: SecretResolver, auth_ref: str | None):
    """A dispatch-time credential provider closing over the resolver and ref —
    the connector receives only this callable and resolves ON EVERY CALL
    (§8c "at dispatch time"): no handler-lifetime memo, so a rotated/expired
    token is picked up on the next tool call, matching the LLM path. A run
    whose path never touches this toolkit still never reads its secret.
    Returns ResolvedSecret so the §8 invariants (non-empty, self-redacting,
    redaction-registered) hold for toolkit tokens too."""
    if auth_ref is None:
        return lambda: None

    def provider() -> ResolvedSecret:
        return ensure_resolved(resolver.resolve(auth_ref))

    return provider


class _DeferredHandler:
    """Stands in for a not-yet-implemented toolkit type so a workflow that
    references it still compiles/persists, but any actual call fails loudly
    and specifically rather than mystifyingly."""

    input_schema: dict = {"type": "object", "additionalProperties": True}
    executable = False  # registered so a workflow compiles, but never surfaced/run

    def __init__(self, toolkit_id: str, toolkit_type: str, reason: str | None = None):
        self._toolkit_id = toolkit_id
        self._reason = reason or _DEFERRED.get(toolkit_type, "unimplemented")
        self.description = f"[{toolkit_type}] not executable in this slice: {self._reason}"

    def is_side_effecting(self, arguments) -> bool:  # never reached — call() always raises
        return False

    async def call(self, *, arguments, idempotency_key):
        raise ToolkitError(f"toolkit '{self._toolkit_id}' is not executable in this slice: {self._reason}")
