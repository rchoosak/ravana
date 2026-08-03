"""Shared failure/credential handling for HTTP-backed connector toolkits.

Every connector that speaks HTTP (`api_connector`, `web_search`, …) shares the
same handful of security-relevant decisions, kept here in ONE place so a copy
can't drift from the others:
  - the §3.6 transient/fatal/model-addressable status taxonomy (a 401 must be
    FATAL, not a backed-off retry);
  - raising a failed request's exception with its secrets redacted;
  - resolving the dispatch-time credential (§8c), turning a resolver failure
    into a FATAL error.
"""

from __future__ import annotations

from typing import Callable, NoReturn

from ravana.runtime.secrets import ResolvedSecret, redact_secrets
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError


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


def redacted_exception_for_rethrow(
    exc: Exception, *, safe_message: str, context: str
) -> Exception | None:
    """Return a secret-safe replacement, or None when the original is safe.

    Callers can use a bare ``raise`` for None, preserving the original type and
    traceback. If redaction changed the message, reconstruct the same exception
    type where possible and fall back to a context-only RuntimeError.
    """
    if safe_message == str(exc):
        return None
    try:
        return type(exc)(safe_message)
    except Exception:  # noqa: BLE001 - third-party exception constructors vary
        return RuntimeError(f"{context} ({type(exc).__name__}): {safe_message}")


def raise_for_request_exception(
    exc: Exception, *, secret_values: tuple[str, ...], context: str
) -> NoReturn:
    """Classify a client-raised request exception per §3.6 and raise.

    A recognised transport/status failure becomes a `ToolkitError` with the
    classified kind. An unrecognised one (a programming/config bug) is re-raised
    — redacted and reconstructed if its message carried a secret, else the
    original — so the engine's terminal boundary fails the run hard rather than
    a wrong-type transient retry re-running broken code.

    Extracted because `api_connector` and `web_search` ran this exact block; a
    second copy is a second place the classify/redact decision could drift.
    """
    kind = classify_exception(exc)
    safe_error = redact_secrets(str(exc), values=secret_values)
    if kind is None:
        replacement = redacted_exception_for_rethrow(
            exc, safe_message=safe_error, context=context
        )
        if replacement is None:
            raise exc
        raise replacement from None
    raise ToolkitError(f"{context}: {safe_error}", kind=kind) from None


def resolve_dispatch_token(
    get_auth_token: Callable[[], ResolvedSecret | None], *, context: str
) -> str | None:
    """Resolve the dispatch-time credential (§8c) to plaintext, or None.

    A resolver that RAISES is fatal — a rotated/absent secret is not something a
    model or a retry can fix. Whether a *None* result (no auth configured) is
    acceptable is the caller's decision: `api_connector` allows it, `web_search`
    treats a missing key as FATAL. That divergence stays at the call site; only
    the shared resolve-or-FATAL shape lives here.
    """
    try:
        token = get_auth_token()
    except Exception as exc:  # noqa: BLE001 - credential failure is fatal
        raise ToolkitError(
            f"{context} credential resolution failed ({type(exc).__name__})",
            kind=ToolFailureKind.FATAL,
        ) from None
    return token.value() if token is not None else None
