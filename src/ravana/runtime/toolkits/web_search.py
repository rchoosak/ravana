"""`web_search` toolkit (§1.7) — a read-only search against a provider.

Shaped like `api_connector` (provider HTTP, dispatch-time credential injection
per §8c, the shared §3.6 failure taxonomy) but narrowed to one operation: the
model supplies a `query`, not a method/path/body, so it cannot turn a search
tool into an arbitrary HTTP client. The provider endpoint and auth style are
fixed by the handler, not chosen by the model.

**Untrusted output.** Search results are arbitrary internet text — the most
attacker-influenceable input in the system, read by a model that also holds
tool credentials and a code sandbox. This handler does NOT wrap results in an
injection boundary: that boundary is a cross-cutting §8 concern for EVERY tool
result (api_connector bodies, code_interpreter stdout, MCP output), tracked
separately, and wrapping only web_search would be an inconsistent half-measure.
What it does do is run the response through the same secret-output gate as
api_connector, so the provider can't echo the API key back into the transcript.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from ravana.runtime.secrets import (
    ResolvedSecret,
    SecretLeakError,
    ensure_secret_free,
    redact_secrets,
)
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.http_errors import classify_exception, classify_status

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["query"],
    "additionalProperties": False,
}

# provider -> (endpoint, how its request/response are shaped). Only Tavily today
# (§4's example). A new provider is a new entry, not a config free-for-all.
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_SUPPORTED_PROVIDERS = frozenset({"tavily"})

_DEFAULT_MAX_RESULTS = 5


class WebSearchHandler:
    input_schema = INPUT_SCHEMA
    executable = True

    def __init__(
        self,
        config: dict[str, Any],
        get_auth_token: Callable[[], ResolvedSecret | None] = lambda: None,
        client: Any | None = None,
    ):
        provider = config.get("provider")
        if provider not in _SUPPORTED_PROVIDERS:
            raise ToolkitError(
                f"web_search: unsupported provider {provider!r} "
                f"(one of {sorted(_SUPPORTED_PROVIDERS)})",
                kind=ToolFailureKind.FATAL,
            )
        self._provider = provider
        self._get_auth_token = get_auth_token
        self._client = client
        self._owns_client = client is None
        self.description = (
            f"Search the web via {provider}. Provide a 'query' string and an optional "
            "'max_results' (1-20). Returns a list of result titles, URLs, and snippets."
        )

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # A search is a read: no dedup, and a retry re-runs it for live results
        # rather than replaying a cached response (§3.6 scopes dedup to effects).
        return False

    def _resolve_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        if not self._owns_client or self._client is None:
            return
        client, self._client = self._client, None
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def call(
        self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str | None = None
    ) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolkitError("web_search: 'query' must be a non-empty string")
        max_results = arguments.get("max_results", _DEFAULT_MAX_RESULTS)

        # §8c: resolved at dispatch, opened to plaintext only here. A search
        # provider needs its key to answer at all, so a missing key is FATAL,
        # not something the model can adjust.
        try:
            token = self._get_auth_token()
        except Exception as exc:  # noqa: BLE001 - credential failure is fatal
            raise ToolkitError(
                f"web_search credential resolution failed ({type(exc).__name__})",
                kind=ToolFailureKind.FATAL,
            ) from None
        api_key = token.value() if token is not None else None
        if not api_key:
            raise ToolkitError(
                f"web_search: no API key configured for {self._provider} (set the toolkit's auth_ref)",
                kind=ToolFailureKind.FATAL,
            )
        secret_values = (api_key,)

        # Tavily takes the key in the JSON body, not a bearer header.
        body = {"api_key": api_key, "query": query, "max_results": max_results}
        client = self._resolve_client()
        try:
            response = await client.post(_TAVILY_ENDPOINT, json=body)
        except Exception as exc:
            kind = classify_exception(exc)
            safe_error = redact_secrets(str(exc), values=secret_values)
            if kind is None:
                # Not a recognised transport/status failure — a programming or
                # config bug. Fail the run hard rather than retry broken code;
                # re-raise redacted so the key can't ride out in the message.
                raise ToolkitError(
                    f"web_search request failed ({type(exc).__name__}): {safe_error}",
                    kind=ToolFailureKind.FATAL,
                ) from None
            raise ToolkitError(f"web_search request failed: {safe_error}", kind=kind) from None

        status = getattr(response, "status_code", None)
        if status is not None and status >= 400:
            detail = redact_secrets(_safe_text(response), values=secret_values)[:500]
            raise ToolkitError(
                f"web_search got HTTP {status} from {self._provider}: {detail}",
                kind=classify_status(status),
            )
        return _format_results(response, secret_values=secret_values)


def _safe_text(response: Any) -> str:
    return str(getattr(response, "text", ""))


def _format_results(response: Any, *, secret_values: tuple[str, ...]) -> str:
    """Render the provider's JSON into a compact, model-readable list.

    Runs through the secret-output gate first (§8): the request carried the API
    key, and a hostile/buggy provider echoing it back must not land it in the
    transcript. A gate hit is FATAL — a leak, not a search result.
    """
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON 2xx is a provider contract break
        raise ToolkitError(
            f"web_search: {getattr(response, 'status_code', '?')} response was not JSON",
            kind=ToolFailureKind.TRANSIENT,
        ) from None

    try:
        ensure_secret_free(payload, context="web_search response", values=secret_values)
    except SecretLeakError as exc:
        raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return "No results."
    lines: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip() or "(untitled)"
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        lines.append(f"- {title}\n  {url}\n  {content}")
    return "\n".join(lines) if lines else json.dumps(payload)[:2000]
