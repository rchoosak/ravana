"""`web_search` toolkit (§1.7) — a read-only, provider-shaped search.

Exercised with a fake httpx client (no network), like the api_connector tests.
The load-bearing checks: the model's `query` reaches the provider, the API key
is injected at dispatch and never leaks back out, a search is read-only (no
dedup), and the §3.6 failure taxonomy routes provider errors correctly.
"""

from __future__ import annotations

import asyncio
import json as json_module

import pytest

from ravana.runtime.secrets import ResolvedSecret
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.web_search import (
    _MAX_CONTENT_CHARS,
    _MAX_RENDERED_CHARS,
    _MAX_RESPONSE_BYTES,
    WebSearchHandler,
)

_TAVILY_RESULTS = {
    "results": [
        {"title": "Python asyncio", "url": "https://ex/1", "content": "event loop basics"},
        {"title": "Trio", "url": "https://ex/2", "content": "structured concurrency"},
    ]
}

_STATUS_CASES = [
    (401, ToolFailureKind.FATAL),        # bad/absent provider key
    (429, ToolFailureKind.TRANSIENT),    # rate limited — retry with backoff
    (500, ToolFailureKind.TRANSIENT),
    (404, ToolFailureKind.FATAL),        # endpoint is fixed, not model-owned
    (432, ToolFailureKind.FATAL),        # Tavily plan usage limit
    (433, ToolFailureKind.FATAL),        # Tavily pay-as-you-go limit
    (400, ToolFailureKind.MODEL_ADDRESSABLE),
    (422, ToolFailureKind.MODEL_ADDRESSABLE),  # the model can adjust the query
]


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else _TAVILY_RESULTS
        self._raw_body = (
            text.encode()
            if text
            else (
                b"not json"
                if self._payload is _RAISE
                else json_module.dumps(self._payload).encode()
            )
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield self._raw_body


_RAISE = object()


class _FakeClient:
    def __init__(self, response=None, raises=None):
        self.calls: list[dict] = []
        self._response = response or _FakeResponse()
        self._raises = raises

    def stream(self, method, url, *, headers=None, json=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        if self._raises is not None:
            raise self._raises
        return self._response


def _handler(client, *, key="tavily-key-XYZ", provider="tavily"):
    return WebSearchHandler(
        {"provider": provider},
        get_auth_token=lambda: ResolvedSecret(key) if key is not None else None,
        client=client,
    )


def _call(handler, arguments, key="k1"):
    return asyncio.run(handler.call(arguments=arguments, idempotency_key=key, run_id="r"))


def test_query_and_bearer_key_reach_the_provider_and_results_are_formatted():
    client = _FakeClient()
    out = _call(_handler(client), {"query": "async python", "max_results": 2})

    assert client.calls[0]["method"] == "POST"
    assert client.calls[0]["url"].startswith("https://api.tavily.com")
    body = client.calls[0]["json"]
    assert body["query"] == "async python" and body["max_results"] == 2
    assert "api_key" not in body
    assert client.calls[0]["headers"] == {
        "Authorization": "Bearer tavily-key-XYZ"
    }
    assert "Python asyncio" in out and "https://ex/1" in out


def test_real_httpx_wire_request_uses_bearer_header_not_body_key():
    import httpx

    captured = {}

    async def respond(request):
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json_module.loads(request.content)
        return httpx.Response(200, json=_TAVILY_RESULTS)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            handler = _handler(client)
            return await handler.call(
                arguments={"query": "wire shape", "max_results": 1},
                idempotency_key="k1",
                run_id="r",
            )

    out = asyncio.run(scenario())
    assert captured["authorization"] == "Bearer tavily-key-XYZ"
    assert captured["body"] == {"query": "wire shape", "max_results": 1}
    assert "Python asyncio" in out


def test_search_is_read_only():
    # §3.6: a search is not side-effecting, so a retry re-runs for live results
    # rather than replaying a cached response.
    assert _handler(_FakeClient()).is_side_effecting({"query": "x"}) is False


def test_unsupported_provider_is_fatal():
    with pytest.raises(ToolkitError) as exc:
        WebSearchHandler({"provider": "bing"})
    assert exc.value.kind is ToolFailureKind.FATAL


def test_missing_api_key_is_fatal_not_model_addressable():
    handler = _handler(_FakeClient(), key=None)
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.FATAL


def test_empty_query_is_refused_before_any_request():
    client = _FakeClient()
    with pytest.raises(ToolkitError, match="query"):
        _call(_handler(client), {"query": "   "})
    assert client.calls == []  # rejected before the key was even resolved


@pytest.mark.parametrize(
    "status,kind",
    _STATUS_CASES,
)
def test_http_error_status_routes_per_taxonomy(status, kind):
    handler = _handler(_FakeClient(_FakeResponse(status_code=status, payload={"detail": "x"})))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is kind


@pytest.mark.parametrize("status,kind", _STATUS_CASES)
def test_raised_http_status_uses_the_same_tavily_taxonomy(status, kind):
    import httpx

    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(status, request=request)
    error = httpx.HTTPStatusError(
        f"HTTP {status}", request=request, response=response
    )
    handler = _handler(_FakeClient(raises=error))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is kind


def test_transport_failure_is_transient():
    import httpx

    handler = _handler(_FakeClient(raises=httpx.ConnectError("refused")))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.TRANSIENT


def test_api_key_never_leaks_into_an_error_message():
    import httpx

    # The key appears in a transport error string; it must be redacted before
    # the message becomes a model-facing tool error.
    handler = _handler(_FakeClient(raises=httpx.ConnectError("failed with tavily-key-XYZ")))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert "tavily-key-XYZ" not in str(exc.value)


@pytest.mark.parametrize(
    "leaky_payload",
    [
        # Key as a whole field value.
        {"results": [{"title": "x", "url": "u", "content": "tavily-key-XYZ"}]},
        # Key EMBEDDED in a larger string — the realistic exfil shape (a URL
        # param). The gate must match substrings, not just whole fields.
        {"results": [{"title": "x", "url": "https://x?token=tavily-key-XYZ", "content": "ok"}]},
        # Key in a field web_search never formats into its output. The gate runs
        # on the whole payload, so an un-rendered field can't smuggle it either.
        {"results": [{"title": "x", "url": "u", "content": "ok"}], "answer": "key=tavily-key-XYZ"},
    ],
)
def test_provider_echoing_the_key_back_is_a_fatal_leak_not_a_result(leaky_payload):
    # §8 secret-output gate: a hostile/buggy provider that reflects the API key
    # in its response must fail closed, not surface it in the transcript —
    # wherever in the payload it appears, whole-field or embedded.
    handler = _handler(_FakeClient(_FakeResponse(payload=leaky_payload)))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.FATAL
    assert "tavily-key-XYZ" not in str(exc.value)


@pytest.mark.parametrize(
    "status,text",
    [
        (400, "bad request echoed tavily-key-XYZ"),
        (200, "not json but echoed tavily-key-XYZ"),
    ],
)
def test_every_raw_provider_body_crosses_the_secret_gate(status, text):
    handler = _handler(_FakeClient(_FakeResponse(status_code=status, text=text)))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.FATAL
    assert "tavily-key-XYZ" not in str(exc.value)


def test_no_results_is_reported_not_an_error():
    handler = _handler(_FakeClient(_FakeResponse(payload={"results": []})))
    assert _call(handler, {"query": "x"}) == "No results."


def test_non_json_success_is_a_transient_provider_contract_error():
    handler = _handler(_FakeClient(_FakeResponse(payload=_RAISE)))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.TRANSIENT


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "provider contract changed"},
        {"results": "not-a-list"},
        {"results": [{"title": "x", "url": "u"}]},
        {"results": ["not-an-object"]},
    ],
)
def test_malformed_success_is_not_reported_as_no_results(payload):
    handler = _handler(_FakeClient(_FakeResponse(payload=payload)))
    with pytest.raises(ToolkitError) as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.TRANSIENT


def test_provider_cannot_return_more_results_than_requested():
    payload = {
        "results": [
            {"title": f"result-{i}", "url": f"https://ex/{i}", "content": "x"}
            for i in range(3)
        ]
    }
    out = _call(
        _handler(_FakeClient(_FakeResponse(payload=payload))),
        {"query": "x", "max_results": 2},
    )
    assert "result-0" in out and "result-1" in out
    assert "result-2" not in out
    assert "truncated to requested max_results=2" in out


def test_rendered_result_fields_and_total_output_are_bounded():
    payload = {
        "results": [
            {
                "title": "title",
                "url": "https://ex",
                "content": "x" * (_MAX_CONTENT_CHARS + 100),
            }
            for _ in range(20)
        ]
    }
    out = _call(
        _handler(_FakeClient(_FakeResponse(payload=payload))),
        {"query": "x", "max_results": 20},
    )
    assert len(out) <= _MAX_RENDERED_CHARS
    assert "[truncated]" in out


def test_oversized_provider_response_fails_closed():
    response = _FakeResponse(text="x" * (_MAX_RESPONSE_BYTES + 1))
    handler = _handler(_FakeClient(response))
    with pytest.raises(ToolkitError, match="exceeded") as exc:
        _call(handler, {"query": "x"})
    assert exc.value.kind is ToolFailureKind.FATAL


@pytest.mark.parametrize("value", [True, 0, 21, "2"])
def test_invalid_max_results_is_refused_before_any_request(value):
    client = _FakeClient()
    with pytest.raises(ToolkitError, match="max_results"):
        _call(_handler(client), {"query": "x", "max_results": value})
    assert client.calls == []


def test_unclassified_programming_error_preserves_its_type():
    handler = _handler(_FakeClient(raises=TypeError("bad fake client")))
    with pytest.raises(TypeError, match="bad fake client"):
        _call(handler, {"query": "x"})
