"""`web_search` toolkit (§1.7) — a read-only, provider-shaped search.

Exercised with a fake httpx client (no network), like the api_connector tests.
The load-bearing checks: the model's `query` reaches the provider, the API key
is injected at dispatch and never leaks back out, a search is read-only (no
dedup), and the §3.6 failure taxonomy routes provider errors correctly.
"""

from __future__ import annotations

import asyncio

import pytest

from ravana.runtime.secrets import ResolvedSecret
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.runtime.toolkits.web_search import WebSearchHandler

_TAVILY_RESULTS = {
    "results": [
        {"title": "Python asyncio", "url": "https://ex/1", "content": "event loop basics"},
        {"title": "Trio", "url": "https://ex/2", "content": "structured concurrency"},
    ]
}


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else _TAVILY_RESULTS
        self.text = text or str(self._payload)

    def json(self):
        if self._payload is _RAISE:
            raise ValueError("not json")
        return self._payload


_RAISE = object()


class _FakeClient:
    def __init__(self, response=None, raises=None):
        self.calls: list[dict] = []
        self._response = response or _FakeResponse()
        self._raises = raises

    async def post(self, url, *, json=None):
        self.calls.append({"url": url, "json": json})
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


def test_query_and_key_reach_the_provider_and_results_are_formatted():
    client = _FakeClient()
    out = _call(_handler(client), {"query": "async python", "max_results": 2})

    assert client.calls[0]["url"].startswith("https://api.tavily.com")
    body = client.calls[0]["json"]
    assert body["query"] == "async python" and body["max_results"] == 2
    assert body["api_key"] == "tavily-key-XYZ"  # injected at dispatch
    assert "Python asyncio" in out and "https://ex/1" in out


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
    [
        (401, ToolFailureKind.FATAL),        # bad/absent provider key
        (429, ToolFailureKind.TRANSIENT),    # rate limited — retry with backoff
        (500, ToolFailureKind.TRANSIENT),
        (422, ToolFailureKind.MODEL_ADDRESSABLE),  # the model can adjust the query
    ],
)
def test_http_error_status_routes_per_taxonomy(status, kind):
    handler = _handler(_FakeClient(_FakeResponse(status_code=status, payload={"detail": "x"})))
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


def test_no_results_is_reported_not_an_error():
    handler = _handler(_FakeClient(_FakeResponse(payload={"results": []})))
    assert _call(handler, {"query": "x"}) == "No results."
