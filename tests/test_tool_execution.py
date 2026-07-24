"""Tool-execution + idempotency-ledger tests. All against fake HTTP clients
and an in-memory SQLite ledger — no network. Covers the §3.6/§8 dedup
guarantee, api_connector request shaping / auth injection, registry wiring,
and secret resolution.
"""

from __future__ import annotations

import asyncio

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.runtime.secrets import EnvSecretResolver, SecretNotFound
from ravana.runtime.tool_executor import RavanaToolExecutor
from ravana.runtime.toolkits.base import (
    ToolFailureKind,
    ToolkitError,
    ToolOutcomeUnknown,
)
from ravana.runtime.toolkits.registry import build_registry
from ravana.schema.loader import load_workflow_yaml
from ravana.schema.util import now_iso
from tests.conftest import SDLC_WORKFLOW


@pytest.fixture
def graph():
    return compile_workflow(load_workflow_yaml(SDLC_WORKFLOW))


def _seed_run(con, run_id="r1"):
    # tool_invocation.run_id FKs to run(id), which FKs to workflow(id) —
    # insert the workflow first, then the run.
    con.execute(
        "INSERT INTO workflow (id, org_id, name, version, state_schema, entry_node_id, created_by, created_at) VALUES (?,?,?,?,?,?,?,?)",
        ("w1", "o1", "wf", 1, "{}", "n1", "t", now_iso()),
    )
    con.execute(
        """INSERT INTO run (id, org_id, workflow_id, workflow_version, status, started_at)
           VALUES (?,?,?,?,?,?)""",
        (run_id, "o1", "w1", 1, "RUNNING", now_iso()),
    )
    con.commit()


# --- Fake HTTP client -------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, response=None):
        self.calls: list[dict] = []
        self._response = response or FakeResponse()

    async def request(self, method, path, *, headers=None, json=None, params=None):
        self.calls.append({"method": method, "path": path, "headers": headers, "json": json, "params": params})
        return self._response


class CountingHandler:
    """Records how many times it actually executed — the dedup assertion."""

    input_schema = {"type": "object"}

    def __init__(self, side_effecting: bool = True):
        self._side_effecting = side_effecting
        self.calls = 0

    def is_side_effecting(self, arguments) -> bool:
        return self._side_effecting

    async def call(self, *, arguments, idempotency_key, run_id=None):
        self.calls += 1
        return f"executed #{self.calls} for {idempotency_key}"


def _exec(con, handlers):
    return RavanaToolExecutor(con, handlers)


def test_idempotency_ledger_dedupes_retried_call(con, graph):
    _seed_run(con)
    handler = CountingHandler()
    executor = _exec(con, {"git_connector": handler})

    # Same key twice = one logical call retried. The handler must run once;
    # the second execute returns the stored result (§3.6).
    r1 = asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={"x": 1}, idempotency_key="k1"))
    r2 = asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={"x": 1}, idempotency_key="k1"))
    assert handler.calls == 1  # side effect fired exactly once
    assert r1 == r2

    # A different key is a different logical call — executes again.
    asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={"x": 2}, idempotency_key="k2"))
    assert handler.calls == 2

    rows = con.execute("SELECT idempotency_key, status FROM tool_invocation ORDER BY idempotency_key").fetchall()
    assert [(r["idempotency_key"], r["status"]) for r in rows] == [("k1", "SUCCEEDED"), ("k2", "SUCCEEDED")]


def test_failed_call_is_not_deduped_and_can_be_retried(con, graph):
    _seed_run(con)

    class FlakyThenOK:
        input_schema = {"type": "object"}
        calls = 0

        def is_side_effecting(self, arguments) -> bool:
            return True

        async def call(self, *, arguments, idempotency_key, run_id=None):
            self.calls += 1
            if self.calls == 1:
                raise ToolkitError("remote 503")
            return "recovered"

    handler = FlakyThenOK()
    executor = _exec(con, {"git_connector": handler})

    with pytest.raises(ToolkitError):
        asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={}, idempotency_key="k1"))
    # FAILED row must NOT dedupe a genuine retry — second attempt runs and succeeds.
    result = asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={}, idempotency_key="k1"))
    assert result == "recovered"
    assert handler.calls == 2
    row = con.execute("SELECT status FROM tool_invocation WHERE idempotency_key = 'k1'").fetchone()
    assert row["status"] == "SUCCEEDED"


def test_started_claim_prevents_duplicate_side_effect_after_process_crash(con, graph):
    _seed_run(con)

    class SimulatedProcessCrash(BaseException):
        pass

    class CrashAfterSideEffect:
        input_schema = {"type": "object"}

        def __init__(self):
            self.effects = 0
            self.crash = True

        def is_side_effecting(self, arguments) -> bool:
            return True

        async def call(self, *, arguments, idempotency_key, run_id=None):
            self.effects += 1
            if self.crash:
                raise SimulatedProcessCrash()
            return "must not run again"

    handler = CrashAfterSideEffect()
    executor = _exec(con, {"git_connector": handler})

    with pytest.raises(SimulatedProcessCrash):
        asyncio.run(
            executor.execute(
                run_id="r1", node_id="n1", tool="git_connector", arguments={}, idempotency_key="k1"
            )
        )
    row = con.execute(
        "SELECT status FROM tool_invocation WHERE idempotency_key = 'k1'"
    ).fetchone()
    assert row["status"] == "STARTED"

    handler.crash = False
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(
            executor.execute(
                run_id="r1", node_id="n1", tool="git_connector", arguments={}, idempotency_key="k1"
            )
        )
    assert exc_info.value.kind is ToolFailureKind.FATAL
    assert handler.effects == 1


def test_indeterminate_tool_error_keeps_started_claim(con, graph):
    _seed_run(con)

    class UnknownAfterSideEffect:
        input_schema = {"type": "object"}

        def __init__(self):
            self.effects = 0

        def is_side_effecting(self, arguments) -> bool:
            return True

        async def call(self, *, arguments, idempotency_key, run_id=None):
            self.effects += 1
            raise ToolOutcomeUnknown("container cleanup failed")

    handler = UnknownAfterSideEffect()
    executor = _exec(con, {"code_interpreter": handler})

    with pytest.raises(ToolOutcomeUnknown):
        asyncio.run(
            executor.execute(
                run_id="r1",
                node_id="n1",
                tool="code_interpreter",
                arguments={},
                idempotency_key="k1",
            )
        )

    row = con.execute(
        "SELECT status, error FROM tool_invocation WHERE idempotency_key = 'k1'"
    ).fetchone()
    assert row["status"] == "STARTED"
    assert "cleanup failed" in row["error"]

    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(
            executor.execute(
                run_id="r1",
                node_id="n1",
                tool="code_interpreter",
                arguments={},
                idempotency_key="k1",
            )
        )
    assert exc_info.value.kind is ToolFailureKind.FATAL
    assert handler.effects == 1


def test_unknown_tool_raises(con, graph):
    _seed_run(con)
    executor = _exec(con, {})
    with pytest.raises(ToolkitError, match="unknown tool"):
        asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="nope", arguments={}, idempotency_key="k1"))


def test_read_only_handler_is_not_deduped(con, graph):
    # §3.6 scopes dedup to side effects: a read-only handler (e.g. a GET/poll)
    # must re-run on the same key, returning live state, not a cached replay —
    # and it writes nothing to the ledger.
    _seed_run(con)
    handler = CountingHandler(side_effecting=False)
    executor = _exec(con, {"web_search": handler})
    asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="web_search", arguments={}, idempotency_key="k1"))
    asyncio.run(executor.execute(run_id="r1", node_id="n1", tool="web_search", arguments={}, idempotency_key="k1"))
    assert handler.calls == 2  # re-ran; not deduped
    assert con.execute("SELECT COUNT(*) c FROM tool_invocation").fetchone()["c"] == 0  # ledger untouched


def test_tools_for_surfaces_specs_for_executable_toolkits(graph):
    # The gateway asks the executor to describe an agent's toolkits as callable
    # tools (name = toolkit id, plus description + input_schema).
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": FakeHttpClient()})
    executor = RavanaToolExecutor(None, handlers)  # con unused for describing tools
    specs = executor.tools_for(["git_connector"])
    by_name = {t.name: t for t in specs}
    assert set(by_name) == {"git_connector"}
    assert by_name["git_connector"].input_schema["required"] == ["path"]
    assert by_name["git_connector"].description  # non-empty, model-facing


def test_web_search_is_advertised_as_a_callable_tool(graph):
    # web_search ships now (was deferred). It is executable, so tools_for
    # surfaces it to the model rather than refusing it.
    resolver = EnvSecretResolver({})
    executor = RavanaToolExecutor(None, build_registry(graph, resolver))
    specs = executor.tools_for(["web_search"])
    assert [t.name for t in specs] == ["web_search"]
    assert specs[0].input_schema["required"] == ["query"]


def test_tools_for_refuses_to_advertise_a_non_executable_toolkit(graph):
    # The refusal mechanism still holds for a genuinely-unavailable toolkit —
    # here an MCP server with no admin allow-list configured. Surfacing it would
    # invite the model to call a tool guaranteed to fail, so tools_for raises.
    resolver = EnvSecretResolver({})
    executor = RavanaToolExecutor(None, build_registry(graph, resolver))
    with pytest.raises(ToolkitError, match="not executable in this build"):
        executor.tools_for(["github_mcp"])


def test_tools_for_raises_on_unregistered_toolkit(graph):
    resolver = EnvSecretResolver({})
    executor = RavanaToolExecutor(None, build_registry(graph, resolver))
    with pytest.raises(ToolkitError, match="no registered handler"):
        executor.tools_for(["does_not_exist"])


def test_api_connector_declares_input_schema(graph):
    # §8(a): every connector declares an input JSON schema.
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": FakeHttpClient()})
    schema = handlers["git_connector"].input_schema
    assert schema["type"] == "object"
    assert "path" in schema["required"]


def test_api_connector_closes_the_client_it_constructs(monkeypatch):
    import httpx

    from ravana.runtime.toolkits.api_connector import ApiConnectorHandler

    made = []

    class Client:
        def __init__(self, **kwargs):
            self.closed = False
            made.append(self)

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    handler = ApiConnectorHandler({"base_url": "https://api.test"})
    handler._resolve_client()
    asyncio.run(handler.aclose())
    assert made and made[0].closed


def test_api_connector_shapes_request_with_auth_and_idempotency_header(graph):
    fake = FakeHttpClient(FakeResponse(200, {"created": "ticket-42"}))
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "ghp_realtoken"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})

    result = asyncio.run(
        handlers["git_connector"].call(
            arguments={"method": "post", "path": "/issues", "json": {"title": "bug"}}, idempotency_key="abc123"
        )
    )
    assert result == '{"created": "ticket-42"}'
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/issues"
    assert call["headers"]["Idempotency-Key"] == "abc123"
    # auth_ref resolved from the env and injected as a bearer token — never the raw ref.
    assert call["headers"]["Authorization"] == "Bearer ghp_realtoken"
    assert call["json"] == {"title": "bug"}


def test_api_connector_raises_on_http_error(graph):
    fake = FakeHttpClient(FakeResponse(500, {"error": "boom"}))
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})
    with pytest.raises(ToolkitError, match="HTTP 500"):
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))


def test_api_connector_rejects_success_body_that_echoes_exact_token(graph):
    secret = 'zzz-"QUOTED"-TOKEN'
    fake = FakeHttpClient(FakeResponse(200, {"nested": {"token": secret}}))
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": secret})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})

    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(
            handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k")
        )
    assert exc_info.value.kind is ToolFailureKind.FATAL
    assert secret not in str(exc_info.value)
    assert "credential material" in str(exc_info.value)


def test_api_connector_hides_resolver_exception_text(graph):
    class LeakyResolver:
        def resolve(self, ref):
            raise RuntimeError("resolver exposed zzz-UNPATTERNED-SECRET")

    handlers = build_registry(
        graph, LeakyResolver(), clients={"git_connector": FakeHttpClient()}
    )
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(
            handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k")
        )
    assert exc_info.value.kind is ToolFailureKind.FATAL
    assert "zzz-UNPATTERNED-SECRET" not in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (500, ToolFailureKind.TRANSIENT),  # server error: §3.6 transient
        (503, ToolFailureKind.TRANSIENT),
        (429, ToolFailureKind.TRANSIENT),  # rate limit
        (408, ToolFailureKind.TRANSIENT),  # request timeout
        (404, ToolFailureKind.MODEL_ADDRESSABLE),  # the model can fix its path
        (422, ToolFailureKind.MODEL_ADDRESSABLE),
        (401, ToolFailureKind.FATAL),  # §3.6 "tool auth failure": non-transient
        (403, ToolFailureKind.FATAL),
    ],
)
def test_api_connector_classifies_http_failures(graph, status, kind):
    # §3.6 taxonomy on tool failures: transient (retry attempt w/ backoff),
    # fatal (auth — fails the run), or model-addressable (fed back).
    fake = FakeHttpClient(FakeResponse(status, {"error": "x"}))
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "tool-test-token"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))
    assert exc_info.value.kind is kind


def test_api_connector_transport_failure_is_transient(graph):
    # A connection-level failure (reset/timeout) is §3.6's "tool timeout" —
    # transient, so the engine retries the attempt with backoff.
    class ExplodingClient:
        async def request(self, *args, **kwargs):
            raise OSError("connection reset by peer")

    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": ExplodingClient()})
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))
    assert exc_info.value.kind is ToolFailureKind.TRANSIENT


def test_api_connector_httpx_timeout_is_transient(graph):
    # httpx's own exceptions don't subclass OSError — the transport check must
    # cover the httpx hierarchy too.
    import httpx

    class TimeoutClient:
        async def request(self, *args, **kwargs):
            raise httpx.ReadTimeout("read timed out")

    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": TimeoutClient()})
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))
    assert exc_info.value.kind is ToolFailureKind.TRANSIENT


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ToolFailureKind.FATAL),  # review probe: was blanket-TRANSIENT via httpx.HTTPError
        (403, ToolFailureKind.FATAL),
        (500, ToolFailureKind.TRANSIENT),
        (404, ToolFailureKind.MODEL_ADDRESSABLE),
    ],
)
def test_api_connector_raise_for_status_client_routes_by_status(graph, status, kind):
    # A client configured to raise_for_status() surfaces HTTP errors as
    # httpx.HTTPStatusError EXCEPTIONS instead of returned responses. These
    # must route by their response status (§3.6) — a 401 is FATAL, never a
    # backed-off transient retry. (No network: exceptions built directly.)
    import httpx

    class RaiseForStatusClient:
        async def request(self, method, path, **kwargs):
            request = httpx.Request(method, f"http://api.test{path}")
            response = httpx.Response(status, request=request)
            raise httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)

    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": RaiseForStatusClient()})
    with pytest.raises(ToolkitError) as exc_info:
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))
    assert exc_info.value.kind is kind


def test_api_connector_programming_bug_propagates_raw_not_transient(graph):
    # Review finding: the transport catch was `except Exception`, so a
    # TypeError from a programming/config bug was classified TRANSIENT and the
    # engine retried broken code with backoff. A non-transport exception now
    # propagates raw — the engine's terminal boundary fails the run hard.
    class BuggyClient:
        async def request(self, *args, **kwargs):
            raise TypeError("request() got an unexpected keyword argument")

    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": BuggyClient()})
    with pytest.raises(TypeError):  # NOT ToolkitError — no wrong-type transient retry
        asyncio.run(handlers["git_connector"].call(arguments={"path": "/x"}, idempotency_key="k"))


@pytest.mark.parametrize(
    "bad_path",
    [
        "https://evil.example/steal",  # absolute URL — httpx base_url is bypassed
        "//evil.example/steal",  # scheme-relative — netloc still overrides the host
        "http://evil.example",  # absolute, no path
        "relative/no/leading/slash",  # ambiguous; require an explicit base-relative path
    ],
)
def test_api_connector_rejects_offbase_path_and_never_sends_token(graph, bad_path):
    # §8 P1: a model-controlled absolute/scheme-relative path would send the
    # bearer token to an attacker host. It must be rejected BEFORE the request
    # is built, and the fake client must never see the call.
    fake = FakeHttpClient()
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "ghp_secret"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})
    with pytest.raises(ToolkitError, match="base_url-relative|start with"):
        asyncio.run(handlers["git_connector"].call(arguments={"method": "POST", "path": bad_path}, idempotency_key="k"))
    assert fake.calls == []  # request never dispatched → token never left the process


def test_api_connector_allows_base_relative_path(graph):
    # The allow-list counterpart: a normal base-relative path is accepted.
    fake = FakeHttpClient(FakeResponse(200, {"ok": True}))
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})
    asyncio.run(handlers["git_connector"].call(arguments={"method": "GET", "path": "/repos/x/issues"}, idempotency_key="k"))
    assert fake.calls[0]["path"] == "/repos/x/issues"


def test_api_connector_rejects_disallowed_method(graph):
    fake = FakeHttpClient()
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": fake})
    with pytest.raises(ToolkitError, match="not allowed"):
        asyncio.run(handlers["git_connector"].call(arguments={"method": "TRACE", "path": "/x"}, idempotency_key="k"))
    assert fake.calls == []


def test_api_connector_get_is_not_side_effecting_post_is(graph):
    # §3.6 P2: dedup is method-aware. A GET is read-only; a POST mutates.
    resolver = EnvSecretResolver({"RAVANA_SECRET_GITHUB_PAT": "x"})
    handlers = build_registry(graph, resolver, clients={"git_connector": FakeHttpClient()})
    h = handlers["git_connector"]
    assert h.is_side_effecting({"method": "GET", "path": "/x"}) is False
    assert h.is_side_effecting({"method": "POST", "path": "/x"}) is True


def test_executor_rejects_args_violating_input_schema(con, graph):
    # §8(a) P3: the executor enforces the handler's declared schema before
    # dispatch — a bad/injected call never reaches the connector.
    _seed_run(con)

    class StrictHandler:
        input_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
        calls = 0

        def is_side_effecting(self, arguments) -> bool:
            return True

        async def call(self, *, arguments, idempotency_key, run_id=None):
            self.calls += 1
            return "ran"

    handler = StrictHandler()
    executor = _exec(con, {"git_connector": handler})
    with pytest.raises(ToolkitError, match="invalid arguments"):
        asyncio.run(
            executor.execute(run_id="r1", node_id="n1", tool="git_connector", arguments={"nope": 1}, idempotency_key="k1")
        )
    assert handler.calls == 0  # never dispatched
    # ...and nothing written to the ledger for a request that never ran.
    assert con.execute("SELECT COUNT(*) c FROM tool_invocation").fetchone()["c"] == 0


def test_registry_wires_web_search_and_defers_unconfigured_mcp(graph):
    resolver = EnvSecretResolver({})
    handlers = build_registry(graph, resolver)
    # web_search now ships: a real executable handler, not a deferral stub.
    from ravana.runtime.toolkits.web_search import WebSearchHandler

    assert isinstance(handlers["web_search"], WebSearchHandler)
    assert handlers["web_search"].executable is True
    # An MCP server with no admin allow-list stays unavailable and refuses to run.
    with pytest.raises(ToolkitError, match="not executable in this slice|allow-list"):
        asyncio.run(handlers["github_mcp"].call(arguments={}, idempotency_key="k"))


def test_toolkit_token_is_reresolved_every_dispatch(graph):
    # §8c "at dispatch time" for toolkits too (review finding: the old
    # provider memoized for the handler's lifetime, so a rotated auth token
    # was never picked up). Each call re-resolves; rotation lands immediately.
    calls: list[int] = []

    class RotatingResolver:
        def resolve(self, ref: str):
            calls.append(1)
            from ravana.runtime.secrets import ResolvedSecret

            return ResolvedSecret(f"token-gen-{len(calls)}")

    fake = FakeHttpClient()
    handlers = build_registry(graph, RotatingResolver(), clients={"git_connector": fake})
    asyncio.run(handlers["git_connector"].call(arguments={"method": "GET", "path": "/a"}, idempotency_key="k1"))
    asyncio.run(handlers["git_connector"].call(arguments={"method": "GET", "path": "/b"}, idempotency_key="k2"))
    assert len(calls) == 2  # re-resolved per dispatch, not memoized for the handler's lifetime
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer token-gen-1"
    assert fake.calls[1]["headers"]["Authorization"] == "Bearer token-gen-2"  # rotation picked up


def test_missing_secret_raises_clearly():
    resolver = EnvSecretResolver({})  # empty env
    with pytest.raises(SecretNotFound, match="RAVANA_SECRET_GITHUB_PAT"):
        resolver.resolve("secrets://github_pat")


def test_secret_requires_scheme():
    resolver = EnvSecretResolver({"RAVANA_SECRET_X": "v"})
    with pytest.raises(SecretNotFound, match="scheme"):
        resolver.resolve("github_pat")  # missing secrets:// prefix
