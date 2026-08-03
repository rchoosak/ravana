"""`web_search` toolkit (§1.7) — a read-only search against a provider.

Shaped like `api_connector` (provider HTTP, dispatch-time credential injection
per §8c, the shared §3.6 failure taxonomy) but narrowed to one operation: the
model supplies a `query`, not a method/path/body, so it cannot turn a search
tool into an arbitrary HTTP client. The provider endpoint and auth style are
fixed by the handler, not chosen by the model.

**Untrusted output.** Search results are arbitrary internet text — the most
attacker-influenceable input in the system. The provider-neutral
`ToolResultMessage` boundary wraps every tool result before either LLM adapter
reads it, while this handler also runs the complete response through the
secret-output gate so the provider cannot echo its API key into the transcript.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ravana.runtime.secrets import (
    ResolvedSecret,
    SecretLeakError,
    ensure_secret_free,
)
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.http_client import LazyAsyncHttpClient
from ravana.runtime.toolkits.http_errors import (
    classify_status,
    raise_for_request_exception,
    resolve_dispatch_token,
)

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
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_ERROR_BYTES = 4096
_MAX_RENDERED_CHARS = 24_000
_MAX_TITLE_CHARS = 300
_MAX_URL_CHARS = 2048
_MAX_CONTENT_CHARS = 4000
_TRUNCATION_MARKER = "... [truncated]"


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
        self._http_client = LazyAsyncHttpClient(client)
        self.description = (
            f"Search the web via {provider}. Provide a 'query' string and an optional "
            "'max_results' (1-20). Returns a list of result titles, URLs, and snippets."
        )

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        # A search is a read: no dedup, and a retry re-runs it for live results
        # rather than replaying a cached response (§3.6 scopes dedup to effects).
        return False

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def call(
        self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str | None = None
    ) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolkitError("web_search: 'query' must be a non-empty string")
        max_results = arguments.get("max_results", _DEFAULT_MAX_RESULTS)
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or not 1 <= max_results <= 20
        ):
            raise ToolkitError("web_search: 'max_results' must be an integer from 1 to 20")

        # §8c: resolved at dispatch, opened to plaintext only here. A search
        # provider needs its key to answer at all, so a missing key is FATAL,
        # not something the model can adjust.
        api_key = resolve_dispatch_token(self._get_auth_token, context="web_search")
        if not api_key:
            raise ToolkitError(
                f"web_search: no API key configured for {self._provider} (set the toolkit's auth_ref)",
                kind=ToolFailureKind.FATAL,
            )
        secret_values = (api_key,)

        headers = {"Authorization": f"Bearer {api_key}"}
        body = {"query": query, "max_results": max_results}
        client = self._http_client.get()
        try:
            async with client.stream(
                "POST",
                _TAVILY_ENDPOINT,
                headers=headers,
                json=body,
            ) as response:
                status = getattr(response, "status_code", None)
                body_limit = (
                    _MAX_ERROR_BYTES
                    if isinstance(status, int) and status >= 400
                    else _MAX_RESPONSE_BYTES
                )
                raw_body, truncated = await _read_limited(response, body_limit)
        except Exception as exc:
            # §3.6 classification + secret-safe rethrow, shared with api_connector.
            raise_for_request_exception(
                exc,
                secret_values=secret_values,
                context="web_search request failed",
                status_classifier=_classify_tavily_status,
            )

        response_text = _checked_response_text(
            raw_body,
            secret_values=secret_values,
        )
        if not isinstance(status, int):
            raise ToolkitError(
                "web_search: provider response had no integer HTTP status",
                kind=ToolFailureKind.FATAL,
            )
        if status >= 400:
            detail = response_text[:500]
            if truncated:
                detail += f" {_TRUNCATION_MARKER}"
            raise ToolkitError(
                f"web_search got HTTP {status} from {self._provider}: {detail}",
                kind=_classify_tavily_status(status),
            )
        if truncated:
            raise ToolkitError(
                f"web_search: provider response exceeded {_MAX_RESPONSE_BYTES} bytes",
                kind=ToolFailureKind.FATAL,
            )
        return _format_results(
            raw_body,
            secret_values=secret_values,
            max_results=max_results,
        )


async def _read_limited(response: Any, limit: int) -> tuple[bytes, bool]:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit + 1 - len(body)
        if remaining <= 0:
            return bytes(body[:limit]), True
        body.extend(chunk[:remaining])
        if len(body) > limit:
            return bytes(body[:limit]), True
    return bytes(body), False


def _checked_response_text(
    raw_body: bytes, *, secret_values: tuple[str, ...]
) -> str:
    text = raw_body.decode("utf-8", errors="replace")
    try:
        ensure_secret_free(
            text,
            context="web_search response",
            values=secret_values,
        )
    except SecretLeakError as exc:
        raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None
    return text


def _classify_tavily_status(status: int) -> ToolFailureKind:
    # The model can adjust a rejected query, but not Tavily's fixed endpoint,
    # method, authentication, account state, or usage-plan limits (432/433).
    if status in (400, 422):
        return ToolFailureKind.MODEL_ADDRESSABLE
    kind = classify_status(status)
    return ToolFailureKind.FATAL if kind is ToolFailureKind.MODEL_ADDRESSABLE else kind


def _format_results(
    raw_body: bytes,
    *,
    secret_values: tuple[str, ...],
    max_results: int,
) -> str:
    """Render the provider's JSON into a compact, model-readable list.

    Runs through the secret-output gate first (§8): the request carried the API
    key, and a hostile/buggy provider echoing it back must not land it in the
    transcript. A gate hit is FATAL — a leak, not a search result.
    """
    try:
        payload = json.loads(raw_body)
    except Exception:  # noqa: BLE001 - a non-JSON 2xx is a provider contract break
        raise ToolkitError(
            "web_search: successful provider response was not JSON",
            kind=ToolFailureKind.TRANSIENT,
        ) from None

    try:
        ensure_secret_free(payload, context="web_search response", values=secret_values)
    except SecretLeakError as exc:
        raise ToolkitError(str(exc), kind=ToolFailureKind.FATAL) from None

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ToolkitError(
            "web_search: successful provider response had an invalid 'results' field",
            kind=ToolFailureKind.TRANSIENT,
        )
    results = payload["results"]
    if not results:
        return "No results."
    lines: list[str] = []
    for item in results[:max_results]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) for field in ("title", "url", "content")
        ):
            raise ToolkitError(
                "web_search: successful provider response contained an invalid result",
                kind=ToolFailureKind.TRANSIENT,
            )
        title = _truncate(item["title"].strip() or "(untitled)", _MAX_TITLE_CHARS)
        url = _truncate(item["url"].strip(), _MAX_URL_CHARS)
        content = _truncate(item["content"].strip(), _MAX_CONTENT_CHARS)
        lines.append(f"- {title}\n  {url}\n  {content}")
    if len(results) > max_results:
        lines.append(f"[truncated to requested max_results={max_results}]")
    rendered = "\n".join(lines)
    return _truncate(rendered, _MAX_RENDERED_CHARS)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    keep = max(0, limit - len(_TRUNCATION_MARKER))
    return value[:keep] + _TRUNCATION_MARKER
