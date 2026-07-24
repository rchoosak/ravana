"""`validate_json` — real JSON-Schema conformance (§3.4.0).

The load-bearing change from the previous shallow check is **property-level type
enforcement**: it did none, so a wrongly-typed value passed and surfaced later
as a routing error one step from its cause. These pin that, the behaviour that
was already correct (so the upgrade didn't regress it), and the contract both
callers depend on — that a violation is *returned*, never raised, even for a
malformed schema.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from ravana.runtime.schema_validate import validate_json

_DRAFT202012_DIALECT = "https://json-schema.org/draft/2020-12/schema"

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


def test_schema_authoring_error_takes_precedence_over_bad_payload():
    error = validate_json(
        {"x": float("nan")},
        {"type": "object", "required": "not-an-array"},
    )
    assert error is not None
    assert "authoring error" in error and "array" in error
    assert "finite JSON number" not in error


def test_deep_schema_is_tagged_as_an_authoring_error():
    schema: dict = {"type": "object"}
    for _ in range(2000):
        schema = {"allOf": [schema]}
    error = validate_json({}, schema)
    assert error is not None
    assert "authoring error" in error and "too deep" in error


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
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        schema = {"type": "object", "properties": {"a": {"$ref": f"http://127.0.0.1:{port}/schema"}}}
        # The optional field is absent: ref preflight must reject the schema
        # independently of which branches this payload exercises.
        error = validate_json({}, schema)
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()

    assert hits == [], f"validator fetched the remote $ref: {hits}"
    assert error is not None and "authoring error" in error


def test_broken_local_ref_is_rejected_before_its_optional_field_is_seen():
    schema = {
        "type": "object",
        "properties": {"optional": {"$ref": "#/$defs/missing"}},
    }
    error = validate_json({}, schema)
    assert error is not None and "authoring error" in error


def test_broken_dynamic_ref_is_preflighted_too():
    schema = {
        "type": "object",
        "properties": {"optional": {"$dynamicRef": "#missing"}},
    }
    error = validate_json({}, schema)
    assert error is not None and "authoring error" in error


def test_valid_inline_refs_and_anchors_survive_preflight():
    pointer_schema = {
        "$defs": {"integer": {"type": "integer"}},
        "$ref": "#/$defs/integer",
    }
    anchor_schema = {
        "$defs": {"integer": {"$id": "urn:ravana:integer", "type": "integer"}},
        "$ref": "urn:ravana:integer",
    }
    assert validate_json(1, pointer_schema) is None
    assert validate_json("wrong", pointer_schema) is not None
    assert validate_json(1, anchor_schema) is None


def test_error_collection_is_bounded_not_just_the_report(monkeypatch):
    # `_MAX_REPORTED_ERRORS` caps the message either way, so asserting "<=3
    # reported" does NOT prove the *materialisation* bound — remove the islice
    # and it still passes. Spy on how many errors are actually pulled: the
    # island cap means the pull stops at _MAX_COLLECTED_ERRORS, not the full
    # (here effectively unbounded) error stream.
    import ravana.runtime.schema_validate as mod

    pulled = 0
    real_iter = mod._SafeDraft202012Validator.iter_errors

    def counting_iter(self, instance):
        nonlocal pulled
        for err in real_iter(self, instance):
            pulled += 1
            yield err

    monkeypatch.setattr(mod._SafeDraft202012Validator, "iter_errors", counting_iter)

    schema = {"type": "object", "properties": {f"f{i}": {"type": "integer"} for i in range(500)}}
    payload = {f"f{i}": "not-an-int" for i in range(500)}  # 500 violations available
    error = validate_json(payload, schema)

    assert error is not None and error.count("field '") <= 3  # report still capped
    assert pulled <= mod._MAX_COLLECTED_ERRORS, f"materialised {pulled}, bound not applied"


def test_deep_payload_does_not_raise():
    # A ~2000-deep nested value blows the recursion limit inside the payload
    # walk / iter_errors; the never-raise backstop must turn it into a string.
    deep: object = 1
    for _ in range(2000):
        deep = [deep]
    result = validate_json(deep, {"type": "object"})
    assert isinstance(result, str) and "too deep" in result


@pytest.mark.parametrize(
    "schema,_label",
    [
        ({"type": "object", "properties": {"x": {"multipleOf": float("nan")}}}, "multipleOf raises ValueError"),
        ({"type": "object", "properties": {"x": {"minimum": float("nan")}}}, "minimum silently accepts"),
    ],
)
def test_non_finite_number_in_the_schema_is_an_authoring_error(schema, _label):
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
    assert "integer" in error  # the actionable reason survives truncation


def test_giant_field_path_does_not_bypass_the_diagnostic_cap():
    key = "K" * 1_000_000
    schema = {
        "type": "object",
        "patternProperties": {".*": {"type": "integer"}},
    }
    error = validate_json({key: "wrong"}, schema)
    assert error is not None
    assert len(error) < 1000 and "integer" in error


def test_regex_keywords_keep_normal_draft_2020_behaviour():
    assert validate_json("aaa", {"type": "string", "pattern": "^a+$"}) is None
    error = validate_json("bbb", {"type": "string", "pattern": "^a+$"})
    assert error is not None and "pattern" in error

    schema = {
        "type": "object",
        "patternProperties": {"^count_": {"type": "integer"}},
    }
    assert validate_json({"count_ok": 1}, schema) is None
    error = validate_json({"count_bad": "one"}, schema)
    assert error is not None and "count_bad" in error


def test_regex_evaluation_has_a_hard_cpu_deadline():
    # Run in a child so removing the timeout fails in bounded time
    # instead of hanging the whole suite in Python's backtracking `re` engine.
    code = (
        "from ravana.runtime.schema_validate import validate_json; "
        "print(validate_json('a' * 100 + '!', "
        "{'type': 'string', 'pattern': '(a|aa)+$'})); "
        "print(validate_json({'a' * 100 + '!': 1}, "
        "{'type': 'object', 'patternProperties': {'(a|aa)+$': "
        "{'type': 'integer'}}}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0
    assert completed.stdout.count("authoring error") == 2
    assert completed.stdout.count("regular-expression") == 2


@pytest.mark.parametrize(
    "keyword",
    ["additionalProperties", "unevaluatedProperties"],
)
def test_regex_deadline_cannot_be_bypassed_by_object_helpers(keyword):
    # jsonschema's stock helpers call stdlib `re.search` while discovering
    # additional/evaluated keys. Put the helper before patternProperties so a
    # vulnerable validator hangs there before reaching our keyword override.
    code = f"""
from ravana.runtime.schema_validate import validate_json
key = "a" * 100 + "!"
schema = {{
    "type": "object",
    {keyword!r}: False,
    "patternProperties": {{"(a|aa)+$": {{"type": "integer"}}}},
}}
print(validate_json({{key: 1}}, schema))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0
    assert "authoring error" in completed.stdout
    assert "regular-expression" in completed.stdout


def test_embedded_resource_cannot_evolve_back_to_unbounded_validator():
    # jsonschema's generated evolve() dispatches from an embedded `$schema`.
    # Without our override, even the same dialect silently selects the stock
    # validator and bypasses both regex and keyword-work budgets.
    code = f"""
from ravana.runtime.schema_validate import validate_json
dialect = {_DRAFT202012_DIALECT!r}
regex_schema = {{
    "$defs": {{
        "embedded": {{
            "$id": "urn:ravana:embedded-regex",
            "$schema": dialect,
            "$dynamicAnchor": "bounded",
            "type": "string",
            "pattern": "(a|aa)+$",
        }},
    }},
    "$dynamicRef": "urn:ravana:embedded-regex#bounded",
}}
print(validate_json("a" * 100 + "!", regex_schema))

branch = {{"type": "array", "items": {{"type": "integer"}}}}
work_schema = {{
    "$defs": {{
        "embedded": {{
            "$id": "urn:ravana:embedded-work",
            "$schema": dialect,
            "anyOf": [branch for _ in range(50)],
        }},
    }},
    "$ref": "urn:ravana:embedded-work",
}}
print(validate_json(["wrong"] * 3_000, work_schema))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0
    assert "regular-expression" in completed.stdout
    assert "resource limit" in completed.stdout


def test_jsonschema_compatibility_matches_generated_constructor_contract():
    import ravana.runtime.schema_validate as mod

    actual_fields = tuple(
        (attribute.name, attribute.alias)
        for attribute in mod._SafeDraft202012Validator.__attrs_attrs__
        if attribute.init
    )
    assert actual_fields == mod._Jsonschema426Compatibility.EVOLVE_FIELDS


def test_non_draft_202012_embedded_resource_is_rejected():
    schema = {
        "$defs": {
            "embedded": {
                "$id": "urn:ravana:draft7",
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "string",
            }
        },
        "$ref": "urn:ravana:draft7",
    }
    error = validate_json("ok", schema)
    assert error is not None
    assert "authoring error" in error and "unsupported $schema dialect" in error


def test_regex_budget_counts_only_time_spent_matching(monkeypatch):
    import ravana.runtime.schema_validate as mod

    # A non-regex keyword can legitimately consume more than the regex budget.
    # The old absolute deadline then blamed the following harmless pattern.
    def slow_keyword(validator, value, instance, schema):
        time.sleep(mod._REGEX_BUDGET_SECONDS * 1.2)
        if False:
            yield

    monkeypatch.setitem(
        mod._SafeDraft202012Validator.VALIDATORS,
        "x-test-slow",
        slow_keyword,
    )
    assert (
        validate_json(
            "ok",
            {
                "x-test-slow": True,
                "pattern": "^ok$",
            },
        )
        is None
    )


@pytest.mark.parametrize(
    "payload,schema",
    [
        (
            {"pattern": "a" * 2_000},
            {"const": {"pattern": "a" * 2_000}},
        ),
        (
            {"pattern": "a" * 2_000},
            {"enum": [{"pattern": "a" * 2_000}]},
        ),
        (
            None,
            {"default": {"pattern": "a" * 2_000}},
        ),
    ],
)
def test_pattern_size_preflight_ignores_non_schema_literals(payload, schema):
    assert validate_json(payload, schema) is None


def test_nested_validation_work_is_bounded_before_eager_error_collection():
    # `anyOf` materialises every branch's errors internally before yielding its
    # own error, so the outer islice cannot stop it. Repeating a modest payload
    # across many branches makes removal of the keyword-level budget time out.
    code = """
from ravana.runtime.schema_validate import validate_json
branch = {"type": "array", "items": {"type": "integer"}}
schema = {"anyOf": [branch for _ in range(50)]}
print(validate_json(["wrong"] * 3_000, schema))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0
    assert "resource limit" in completed.stdout
    assert "authoring error" not in completed.stdout


def test_unique_items_cannot_fall_back_to_quadratic_work():
    # jsonschema's generic equality fallback is O(n^2) for arrays of objects.
    # This size leaves enough steps to enter the implementation: the bounded
    # canonicaliser then stops, while the stock helper runs for seconds.
    code = """
from ravana.runtime.schema_validate import validate_json
payload = [{"i": i} for i in range(4_998)]
print(validate_json(payload, {"type": "array", "uniqueItems": True}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    assert completed.returncode == 0
    assert "resource limit" in completed.stdout


@pytest.mark.parametrize(
    "payload,valid",
    [
        ([True, 1], True),
        ([1, 1.0], False),
        ([{"a": 1, "b": 2}, {"b": 2, "a": 1}], False),
    ],
)
def test_bounded_unique_items_preserves_json_schema_equality(payload, valid):
    error = validate_json(payload, {"type": "array", "uniqueItems": True})
    assert (error is None) is valid


def test_safe_object_helpers_preserve_evaluated_property_semantics():
    schema = {
        "allOf": [
            {
                "type": "object",
                "patternProperties": {"^count_": {"type": "integer"}},
            }
        ],
        "unevaluatedProperties": False,
    }
    assert validate_json({"count_ok": 1}, schema) is None
    error = validate_json({"count_ok": 1, "extra": 2}, schema)
    assert error is not None and "extra" in error

    # The empty regex matches every key. A truthiness check on the joined
    # patterns would incorrectly classify this property as additional.
    assert (
        validate_json(
            {"anything": 1},
            {
                "type": "object",
                "patternProperties": {"": {"type": "integer"}},
                "additionalProperties": False,
            },
        )
        is None
    )


@pytest.mark.parametrize(
    "ref_keyword,anchor_keyword",
    [
        ("$ref", "$anchor"),
        ("$dynamicRef", "$dynamicAnchor"),
    ],
)
def test_evaluated_property_compatibility_handles_local_references(
    ref_keyword,
    anchor_keyword,
):
    schema = {
        "$defs": {
            "base": {
                anchor_keyword: "base",
                "properties": {"count": {"type": "integer"}},
            }
        },
        ref_keyword: "#base",
        "unevaluatedProperties": False,
    }
    assert validate_json({"count": 1}, schema) is None
    error = validate_json({"count": 1, "extra": 2}, schema)
    assert error is not None and "extra" in error


def test_giant_pattern_is_rejected_before_compilation():
    error = validate_json(
        "anything",
        {"type": "string", "pattern": "a" * 10_000},
    )
    assert error is not None
    assert "authoring error" in error and "pattern exceeds" in error


def test_schemaless_payload_still_rejects_non_finite():
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
