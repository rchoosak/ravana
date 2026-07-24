"""Shared JSON-Schema validation for agent structured output (gateway, §3.4)
and toolkit input before dispatch (§8a).

Backed by the real `jsonschema` validator rather than a hand-rolled subset.
The previous shallow version checked only top-level object-ness, `required`,
`additionalProperties` and *top-level* enums — it did no `type` checking on
properties at all, so `{"iteration": "five"}` satisfied
`{"iteration": {"type": "integer"}}`. Both call sites care:

- §3.4 output: a wrongly-typed value passed validation and landed in
  `shared_state`, where a routing condition (`iteration_count >= 5`) then
  compared `str` to `int` at edge-evaluation time — a schema violation
  surfacing as a routing error one step later, far from its cause.
- §8a tool input: arguments reached the handler — and for `mcp_server`, a
  third-party process — without their declared types being enforced.

Errors are returned, never raised: both callers treat a violation as a
*result* (a repair-loop prompt, a model-addressable tool error), not an
exception. The message is written to be actionable by the model, since it is
fed back verbatim in the §3.4 repair prompt.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions


def validate_json(payload: Any, schema: dict[str, Any] | None) -> str | None:
    """Returns an error string if payload violates the schema, else None."""
    if schema is None:
        # No declared schema still means "an object" — the whole payload
        # becomes a state_delta (§3.4), which has to be a mapping.
        return None if isinstance(payload, dict) else "expected a JSON object"

    # A malformed *schema* is an authoring bug, not a model mistake. Say so
    # plainly rather than blaming the payload, which would send the repair loop
    # chasing a fault it cannot fix.
    #
    # Both construction AND iteration are guarded: `jsonschema` reports some
    # schema defects lazily. An unknown `type` passes the constructor and raises
    # `UnknownType` mid-validation, so catching only at construction would let it
    # escape as an exception — from a function both callers rely on to *return*
    # its errors.
    try:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path))
    except jsonschema_exceptions.SchemaError as exc:
        return f"invalid output schema (workflow authoring error): {exc.message}"
    except jsonschema_exceptions.UnknownType as exc:
        return f"invalid output schema (workflow authoring error): unknown type {exc.type!r}"
    if not errors:
        return None
    return "; ".join(_describe(error) for error in errors[:_MAX_REPORTED_ERRORS])


# Enough to fix a response in one repair round without burying the model in a
# wall of text — the message is spent as prompt tokens on every retry.
_MAX_REPORTED_ERRORS = 3


def _describe(error: jsonschema_exceptions.ValidationError) -> str:
    """One short, model-actionable line naming the offending field.

    `jsonschema`'s own message is precise but positionless ("'five' is not of
    type 'integer'"), so the path is prepended: with several fields wrong, the
    model otherwise cannot tell which one to fix.
    """
    location = ".".join(str(part) for part in error.absolute_path)
    return f"field '{location}': {error.message}" if location else error.message
