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


def test_remote_ref_is_rejected_without_fetching():
    # SSRF surface: a $ref to a URL must never make the validator reach out.
    # The registry does not retrieve — a remote ref resolves to an authoring
    # error, no network. Timed so a silent fetch attempt (connect/timeout)
    # would show up as elapsed wall-clock.
    import time

    schema = {"type": "object", "properties": {"a": {"$ref": "http://127.0.0.1:9/x"}}}
    start = time.monotonic()
    error = validate_json({"a": 1}, schema)
    assert error is not None and "authoring error" in error
    assert time.monotonic() - start < 0.5, "a remote $ref must not be fetched"


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
