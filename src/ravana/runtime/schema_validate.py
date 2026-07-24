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

import itertools
import math
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable

# Context-neutral: this validator serves BOTH §3.4 agent output and §8a tool
# input, so the message must not claim "output schema" on the tool-input path.
_SCHEMA_ERROR_PREFIX = "invalid schema (workflow authoring error): "


def _refuse_retrieval(uri: str) -> Any:
    """A registry `retrieve` that fetches nothing.

    jsonschema 4.x keeps legacy HTTP/file retrieval for a `$ref` to a URL — a
    live listener confirmed a `$ref: "http://.../schema"` triggers a real GET.
    An `output_schema` / toolkit input schema is workflow- or MCP-authored, so
    that is an SSRF/LFI surface: a `$ref` could pull an internal URL or a local
    file. This registry raises instead of retrieving, so ANY ref the schema
    doesn't define inline is an unresolved-ref authoring error, never a fetch.
    """
    # `ref=` is the real attrs field and `retrieve=` a real Registry param
    # (both runtime-verified: the no-fetch listener test depends on them).
    # `referencing`'s attrs-generated __init__ signatures confuse mypy here.
    raise NoSuchResource(ref=uri)  # type: ignore[call-arg]


# Shared, immutable, and empty-but-non-retrieving. Reused across calls so every
# validation resolves refs the same way: inline only, no network, no disk.
_NO_FETCH_REGISTRY: Registry = Registry(retrieve=_refuse_retrieval)  # type: ignore[call-arg]


def validate_json(payload: Any, schema: dict[str, Any] | None) -> str | None:
    """Returns an error string if payload violates the schema, else None.

    Never raises — the contract both callers (the §3.4 repair loop, §8a tool
    dispatch) depend on, since they treat the result as a value. Enforced
    STRUCTURALLY: the whole body runs under one guard that converts any escaping
    exception into a returned string. Three earlier rounds each patched one more
    exception type (`SchemaError`, `UnknownType`, `Unresolvable`, then
    `RecursionError` on one call) and the next round found another leak — a deep
    payload's `RecursionError`, a `multipleOf: NaN`'s `ValueError`. Enumerating
    raisers was the wrong shape for a never-raise contract; this backstop is the
    right one, with the specific handlers below kept for actionable messages.
    """
    try:
        return _validate(payload, schema)
    except RecursionError:
        # A pathologically deep schema OR payload blows the stack anywhere in
        # here (the payload walk, check_schema, or iter_errors). Too deep to
        # validate is a rejection, not a crash.
        return "schema or payload nesting is too deep to validate"
    except Exception as exc:  # noqa: BLE001 - the never-raise backstop is the point
        # Anything jsonschema throws that the specific handlers didn't name
        # (e.g. `multipleOf: NaN` -> ValueError). Class only, no payload text.
        return f"{_SCHEMA_ERROR_PREFIX}could not be validated ({type(exc).__name__})"


def _validate(payload: Any, schema: dict[str, Any] | None) -> str | None:
    # Non-finite floats are not JSON (Draft 2020-12) and are invisible to a
    # numeric bound: NaN compares False to everything, so `NaN <= maximum` is
    # False-not-raised and a NaN slips past any constraint into durable state /
    # tool arguments. Checked FIRST, before the schema-less early return, so a
    # schema-less node's payload can't smuggle a NaN into state either.
    nonfinite = _first_nonfinite_path(payload)
    if nonfinite is not None:
        return f"field '{nonfinite}': not a finite JSON number (NaN/Infinity are not valid JSON)"

    if schema is None:
        # No declared schema still means "an object" — the whole payload
        # becomes a state_delta (§3.4), which has to be a mapping.
        return None if isinstance(payload, dict) else "expected a JSON object"

    # A non-finite number in the SCHEMA is an authoring error jsonschema won't
    # flag: `minimum: NaN` silently accepts everything (all comparisons False),
    # `multipleOf: NaN` raises ValueError mid-validation. Reject before either.
    schema_nonfinite = _first_nonfinite_path(schema)
    if schema_nonfinite is not None:
        return f"{_SCHEMA_ERROR_PREFIX}non-finite number at '{schema_nonfinite}'"

    # Validate the schema against its meta-schema FIRST. Without this, a
    # malformed schema is either silently misread — `required: "abc"` becomes
    # three single-letter required fields, a payload-blaming message the model
    # can never satisfy — or raised (`type: 123` -> TypeError, `properties: []`
    # -> AttributeError). `check_schema` turns those into one authoring error.
    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema_exceptions.SchemaError as exc:
        return f"{_SCHEMA_ERROR_PREFIX}{_truncate(exc.message)}"

    try:
        validator = Draft202012Validator(schema, registry=_NO_FETCH_REGISTRY)
        # `iter_errors` is lazy; `sorted()` would materialise EVERY error for a
        # huge/deeply-wrong payload just to report three. Cap the window first.
        collected = list(
            itertools.islice(validator.iter_errors(payload), _MAX_COLLECTED_ERRORS)
        )
    except jsonschema_exceptions.UnknownType as exc:
        # An unknown `type` passes check_schema but raises during validation.
        return f"{_SCHEMA_ERROR_PREFIX}unknown type {exc.type!r}"
    except Unresolvable as exc:
        # A `$ref` the schema can't resolve (including any remote URL — the
        # registry does NOT fetch it, it raises here) is an authoring error,
        # not a model mistake. Catching it keeps the no-network, never-raise
        # contract regardless of what a workflow/MCP schema puts in a $ref.
        return f"{_SCHEMA_ERROR_PREFIX}unresolvable $ref ({_truncate(str(exc))})"

    if not collected:
        return None
    errors = sorted(collected, key=lambda e: list(e.absolute_path))
    return "; ".join(_describe(error) for error in errors[:_MAX_REPORTED_ERRORS])


# Enough to fix a response in one repair round without burying the model in a
# wall of text — the message is spent as prompt tokens on every retry.
_MAX_REPORTED_ERRORS = 3

# Upper bound on errors materialised before reporting the first few. Bounds the
# work a pathological payload can force without changing the reported output.
_MAX_COLLECTED_ERRORS = 64

# Per-message cap. `jsonschema` embeds the offending instance in its message
# ("<1MB blob> is not of type 'integer'"), and the whole result is spent as
# repair-prompt tokens on every retry — so a 1MB wrong value would otherwise
# become a 1MB error string. Generous enough to keep a normal message intact.
_MAX_MESSAGE_CHARS = 300


def _truncate(text: str) -> str:
    if len(text) <= _MAX_MESSAGE_CHARS:
        return text
    return text[:_MAX_MESSAGE_CHARS] + f"… (+{len(text) - _MAX_MESSAGE_CHARS} chars)"


def _first_nonfinite_path(value: Any, path: str = "") -> str | None:
    """Dotted path of the first NaN/Infinity float in `value`, or None.

    `bool` is a subclass of `int`, not `float`, so True/False are never flagged;
    only genuine non-finite floats (from `json.loads` of `NaN`/`Infinity`) are.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return path or "<root>"
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_nonfinite_path(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _first_nonfinite_path(item, f"{path}.{index}" if path else str(index))
            if found is not None:
                return found
    return None


def _describe(error: jsonschema_exceptions.ValidationError) -> str:
    """One short, model-actionable line naming the offending field.

    `jsonschema`'s own message is precise but positionless ("'five' is not of
    type 'integer'"), so the path is prepended: with several fields wrong, the
    model otherwise cannot tell which one to fix.
    """
    location = ".".join(str(part) for part in error.absolute_path)
    message = _truncate(error.message)
    return f"field '{location}': {message}" if location else message
