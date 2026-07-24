"""`validate_json` — real JSON-Schema conformance (§3.4.0).

The load-bearing change from the previous shallow check is **property-level type
enforcement**: it did none, so a wrongly-typed value passed and surfaced later
as a routing error one step from its cause. These pin that, the behaviour that
was already correct (so the upgrade didn't regress it), and the contract both
callers depend on — that a violation is *returned*, never raised, even for a
malformed schema.
"""

from __future__ import annotations

import pytest

from ravana.runtime.schema_validate import validate_json

_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "iteration": {"type": "integer"},
        "files": {"type": "array", "items": {"type": "string"}},
        "meta": {"type": "object", "properties": {"score": {"type": "number"}}},
    },
    "required": ["status"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "PASS"},
        {"status": "FAIL", "iteration": 3, "files": ["a.py", "b.py"]},
        {"status": "PASS", "meta": {"score": 0.9}},
    ],
)
def test_valid_payloads_pass(payload):
    assert validate_json(payload, _SCHEMA) is None


# The whole point of the upgrade: the shallow validator accepted every one of
# these. Each names the offending field so the §3.4 repair prompt is actionable.
@pytest.mark.parametrize(
    "payload,field",
    [
        ({"status": "PASS", "iteration": "five"}, "iteration"),   # str where int
        ({"status": "PASS", "files": [1, 2]}, "files.0"),         # wrong array item
        ({"status": "PASS", "files": "a.py"}, "files"),           # scalar where array
        ({"status": "PASS", "meta": {"score": "high"}}, "meta.score"),  # nested type
        ({"status": "PASS", "iteration": None}, "iteration"),     # null where int
    ],
)
def test_property_type_violations_are_caught_and_located(payload, field):
    error = validate_json(payload, _SCHEMA)
    assert error is not None
    assert field in error


@pytest.mark.parametrize(
    "payload,needle",
    [
        ({"iteration": 1}, "status"),                    # missing required
        ({"status": "PASS", "extra": 1}, "extra"),       # additionalProperties
        ({"status": "MAYBE"}, "PASS"),                   # enum, error names the choices
    ],
)
def test_previously_correct_checks_are_preserved(payload, needle):
    error = validate_json(payload, _SCHEMA)
    assert error is not None and needle in error


def test_no_schema_requires_an_object():
    # A schema-less node's whole payload becomes the state_delta, which must be
    # a mapping (§3.4).
    assert validate_json({"anything": 1}, None) is None
    assert validate_json(["not", "an", "object"], None) == "expected a JSON object"


@pytest.mark.parametrize(
    "bad_schema",
    [
        {"type": "object", "properties": {"a": {"type": "nonsense"}}},  # unknown type, raised lazily
        {"type": "object", "required": "should-be-a-list"},             # wrong keyword shape
        {"type": 123},                                                  # type not a string/list
        {"type": "object", "properties": ["a", "b"]},                   # properties not an object
        {"type": "object", "properties": {"a": {"$ref": "#/nope"}}},    # unresolvable local $ref
    ],
)
def test_malformed_schema_is_an_authoring_error_never_raises(bad_schema):
    # Both callers treat the result as a value (a repair prompt / a tool error),
    # so this MUST NOT raise. It must also be reported as an AUTHORING error,
    # not blamed on the payload — the assertion is `authoring error` only, with
    # no escape hatch. An earlier version allowed `or "required"`, which matched
    # the payload-blaming message `required: "abc"` produced and so passed for
    # the wrong reason (the exact non-discriminating-test trap this repo keeps
    # hitting). `check_schema` up front makes every one of these an authoring
    # error, so the escape hatch is gone.
    error = validate_json({"a": 1}, bad_schema)
    assert isinstance(error, str)
    assert "authoring error" in error


def test_remote_ref_makes_no_outbound_request():
    # SSRF/LFI surface: a $ref to a URL must never make the validator reach
    # out. A closed-port timing check does NOT prove this — it passes on a fast
    # connection refusal even if a fetch was attempted (that mistake shipped
    # once). Stand up a real listener and assert it receives ZERO requests.
    import http.server
    import socketserver
    import threading

    hits: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            body = b'{"type": "object"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            return

    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        schema = {"type": "object", "properties": {"a": {"$ref": f"http://127.0.0.1:{port}/schema"}}}
        error = validate_json({"a": 1}, schema)
    finally:
        server.shutdown()

    assert hits == [], f"validator fetched the remote $ref: {hits}"
    assert error is not None and "authoring error" in error


def test_error_collection_is_bounded_not_just_the_report(monkeypatch):
    # `_MAX_REPORTED_ERRORS` caps the message either way, so asserting "<=3
    # reported" does NOT prove the *materialisation* bound — remove the islice
    # and it still passes. Spy on how many errors are actually pulled: the
    # island cap means the pull stops at _MAX_COLLECTED_ERRORS, not the full
    # (here effectively unbounded) error stream.
    import ravana.runtime.schema_validate as mod

    pulled = 0
    real_iter = mod.Draft202012Validator.iter_errors

    def counting_iter(self, instance):
        nonlocal pulled
        for err in real_iter(self, instance):
            pulled += 1
            yield err

    monkeypatch.setattr(mod.Draft202012Validator, "iter_errors", counting_iter)

    schema = {"type": "object", "properties": {f"f{i}": {"type": "integer"} for i in range(500)}}
    payload = {f"f{i}": "not-an-int" for i in range(500)}  # 500 violations available
    error = validate_json(payload, schema)

    assert error is not None and error.count("field '") <= 3  # report still capped
    assert pulled <= mod._MAX_COLLECTED_ERRORS, f"materialised {pulled}, bound not applied"


def test_deep_payload_does_not_raise(monkeypatch):
    # A ~2000-deep nested value blows the recursion limit inside the payload
    # walk / iter_errors; the never-raise backstop must turn it into a string.
    deep: object = 1
    for _ in range(2000):
        deep = [deep]
    result = validate_json(deep, {"type": "object"})
    assert isinstance(result, str) and "too deep" in result


@pytest.mark.parametrize(
    "schema,label",
    [
        ({"type": "object", "properties": {"x": {"multipleOf": float("nan")}}}, "multipleOf raises ValueError"),
        ({"type": "object", "properties": {"x": {"minimum": float("nan")}}}, "minimum silently accepts"),
    ],
)
def test_non_finite_number_in_the_schema_is_an_authoring_error(schema, label):
    # jsonschema won't flag these — `minimum: NaN` accepts everything, and
    # `multipleOf: NaN` raises ValueError mid-validation. Both are authoring
    # errors, returned not raised.
    error = validate_json({"x": 1}, schema)
    assert error is not None and "authoring error" in error


def test_giant_offending_value_does_not_bloat_the_message():
    # jsonschema embeds the instance in its message; a 1MB wrong value would
    # otherwise become a 1MB repair prompt spent on every retry.
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    error = validate_json({"x": "A" * 1_000_000}, schema)
    assert error is not None and len(error) < 1000


def test_schemaless_payload_still_rejects_non_finite(monkeypatch):
    # Pins the guard ORDER: the non-finite check must run before the
    # `schema is None` early return, or a schema-less node smuggles a NaN into
    # durable state. Moving the check after that return makes this fail.
    assert validate_json({"score": float("nan")}, None) is not None
    assert validate_json({"score": float("inf")}, None) is not None
    assert validate_json({"ok": 1}, None) is None


@pytest.mark.parametrize(
    "payload,field",
    [
        ({"status": "PASS", "meta": {"score": float("nan")}}, "meta.score"),
        ({"status": "PASS", "meta": {"score": float("inf")}}, "meta.score"),
        ({"status": "PASS", "files": [float("-inf")]}, "files.0"),
    ],
)
def test_non_finite_numbers_are_rejected_with_their_path(payload, field):
    # NaN is invisible to a numeric bound (`NaN <= max` is False, not raised),
    # so it would slip past the schema into durable state. Rejected up front.
    error = validate_json(payload, _SCHEMA)
    assert error is not None
    assert field in error and "finite" in error
