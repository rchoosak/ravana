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
    ],
)
def test_malformed_schema_returns_an_error_never_raises(bad_schema):
    # Both callers treat the result as a value (a repair prompt / a tool error),
    # so this MUST NOT raise. `jsonschema` reports some schema defects only
    # during validation, not construction — an unknown `type` is the case that
    # slipped past a construction-only guard when this was first written.
    error = validate_json({"a": 1}, bad_schema)
    assert isinstance(error, str)
    assert "authoring error" in error or "required" in error
