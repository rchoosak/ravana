"""§3.6 tool-failure taxonomy for HTTP-backed toolkits.

Shared by every connector that speaks HTTP (`api_connector`, `web_search`, …)
so the transient/fatal/model-addressable classification lives in ONE place. A
second copy of this is a second place a security-relevant routing decision — a
401 must be FATAL, not a backed-off retry — could drift.
"""

from __future__ import annotations

from ravana.runtime.toolkits.base import ToolFailureKind


def classify_status(status: int) -> ToolFailureKind:
    """§3.6's taxonomy by HTTP status: 401/403 is the "tool auth failure"
    (fatal, fails the run); 5xx/429/408 may recover (transient — engine retries
    the attempt with backoff); any other 4xx is something the model can adjust
    to (bad query, validation) — fed back to it."""
    if status in (401, 403):
        return ToolFailureKind.FATAL
    if status in (408, 429) or status >= 500:
        return ToolFailureKind.TRANSIENT
    return ToolFailureKind.MODEL_ADDRESSABLE


def classify_exception(exc: Exception) -> ToolFailureKind | None:
    """Classify a client-raised exception per §3.6, or None for "not ours —
    propagate raw" (a programming/config bug the engine should fail hard on).

    `httpx.HTTPStatusError` is checked FIRST and routed by its response status:
    a client configured with `raise_for_status()` surfaces a 401 as an
    exception, and blanket-treating the httpx hierarchy as transient would turn
    that auth failure (FATAL) into a backed-off retry. Only `TransportError`
    (timeouts, connection failures) and the builtin OS-level types are
    transient."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a direct dependency
        return ToolFailureKind.TRANSIENT if isinstance(exc, (OSError, TimeoutError)) else None
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_status(exc.response.status_code)
    if isinstance(exc, (OSError, TimeoutError)):
        return ToolFailureKind.TRANSIENT
    if isinstance(exc, httpx.TransportError):
        return ToolFailureKind.TRANSIENT
    return None
