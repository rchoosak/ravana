"""Shared JSON-Schema validation for agent output and toolkit input.

Backed by a real Draft 2020-12 validator rather than the previous shallow
top-level check. Errors are returned, never raised: the gateway feeds payload
violations into its repair loop, while the tool executor turns them into a
model-addressable tool error.

Schemas may come from workflow authors or an MCP server. They are therefore
validated before payloads, cannot retrieve external references, and cannot run
unbounded regular-expression work in the runtime process.
"""

from __future__ import annotations

from contextvars import ContextVar
import itertools
import math
import time
from typing import Any

import regex
from jsonschema import Draft202012Validator
from jsonschema import exceptions as jsonschema_exceptions
from jsonschema import validators
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource, Unresolvable
from referencing.jsonschema import DRAFT202012

_SCHEMA_ERROR_PREFIX = "invalid schema (workflow authoring error): "

# Bound both validation work and the repair-prompt text it produces.
_MAX_REPORTED_ERRORS = 3
_MAX_COLLECTED_ERRORS = 64
_MAX_MESSAGE_CHARS = 300
_MAX_PATH_CHARS = 120
_MAX_DIAGNOSTIC_CHARS = 1_500
_MAX_PATTERN_CHARS = 1_500

# Python's built-in `re` has no timeout. The alternate engine applies this
# shared wall-clock budget to every pattern/patternProperties match in one
# validation, so many individually cheap patterns cannot multiply the limit.
_REGEX_BUDGET_SECONDS = 0.1
_REGEX_DEADLINE: ContextVar[float | None] = ContextVar(
    "ravana_schema_regex_deadline",
    default=None,
)


class _RegexBudgetExceeded(RuntimeError):
    pass


class _SchemaPolicyError(ValueError):
    pass


def _regex_matches(pattern: str, value: str) -> bool:
    deadline = _REGEX_DEADLINE.get()
    if deadline is None:
        raise RuntimeError("schema regex evaluated outside validation boundary")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _RegexBudgetExceeded
    try:
        return regex.search(pattern, value, timeout=remaining) is not None
    except TimeoutError as exc:
        raise _RegexBudgetExceeded from exc


def _safe_pattern(validator, pattern, instance, schema):
    if validator.is_type(instance, "string") and not _regex_matches(pattern, instance):
        # The field path already identifies the value. Avoid embedding the
        # entire instance as jsonschema's stock pattern message does.
        yield jsonschema_exceptions.ValidationError(
            f"does not match pattern {_truncate(pattern, _MAX_PATH_CHARS)!r}"
        )


def _safe_pattern_properties(validator, pattern_properties, instance, schema):
    if not validator.is_type(instance, "object"):
        return
    for pattern, subschema in pattern_properties.items():
        for key, value in instance.items():
            if _regex_matches(pattern, key):
                yield from validator.descend(
                    value,
                    subschema,
                    path=key,
                    schema_path=pattern,
                )


_SafeDraft202012Validator = validators.extend(
    Draft202012Validator,
    validators={
        "pattern": _safe_pattern,
        "patternProperties": _safe_pattern_properties,
    },
)


def _refuse_retrieval(uri: str) -> Any:
    """Registry retrieval hook which rejects every network or file lookup."""
    raise NoSuchResource(ref=uri)  # type: ignore[call-arg]


_NO_FETCH_REGISTRY: Registry = Registry(retrieve=_refuse_retrieval)  # type: ignore[call-arg]


def validate_json(payload: Any, schema: dict[str, Any] | None) -> str | None:
    """Return an actionable validation error, or None; never raise."""
    try:
        result = _validate(payload, schema)
    except RecursionError:
        # Schema recursion is handled inside `_validate_schema`, so an escape
        # here is payload/validator traversal rather than an authoring defect.
        result = "payload nesting is too deep to validate"
    except Exception as exc:  # noqa: BLE001 - this is the never-raise boundary
        # Keep exception text out: third-party validators or payload objects may
        # include credentials or the complete offending value in it.
        result = f"{_SCHEMA_ERROR_PREFIX}could not be validated ({type(exc).__name__})"
    return None if result is None else _truncate(result, _MAX_DIAGNOSTIC_CHARS)


def _validate(payload: Any, schema: dict[str, Any] | None) -> str | None:
    if schema is None:
        # A schema-less node still rejects non-JSON floats before accepting its
        # whole object as a state delta.
        nonfinite = _first_nonfinite_path(payload)
        if nonfinite is not None:
            return _field_error(
                nonfinite,
                "not a finite JSON number (NaN/Infinity are not valid JSON)",
            )
        return None if isinstance(payload, dict) else "expected a JSON object"

    # Authoring defects take precedence. Otherwise a NaN payload can mask a
    # malformed schema and make the repair loop blame the model.
    schema_error = _validate_schema(schema)
    if schema_error is not None:
        return schema_error

    nonfinite = _first_nonfinite_path(payload)
    if nonfinite is not None:
        return _field_error(
            nonfinite,
            "not a finite JSON number (NaN/Infinity are not valid JSON)",
        )

    token = _REGEX_DEADLINE.set(time.monotonic() + _REGEX_BUDGET_SECONDS)
    try:
        validator = _SafeDraft202012Validator(schema, registry=_NO_FETCH_REGISTRY)
        collected = list(
            itertools.islice(validator.iter_errors(payload), _MAX_COLLECTED_ERRORS)
        )
    except _RegexBudgetExceeded:
        return (
            f"{_SCHEMA_ERROR_PREFIX}regular-expression evaluation exceeded "
            f"{_REGEX_BUDGET_SECONDS:g}s limit"
        )
    except jsonschema_exceptions.UnknownType as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unknown type {exc.type!r}"
    except Unresolvable as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unresolvable $ref ({_truncate(str(exc))})"
    finally:
        _REGEX_DEADLINE.reset(token)

    if not collected:
        return None
    errors = sorted(collected, key=lambda error: list(error.absolute_path))
    return "; ".join(_describe(error) for error in errors[:_MAX_REPORTED_ERRORS])


def _validate_schema(schema: dict[str, Any]) -> str | None:
    """Validate author-controlled structure and every reference."""
    try:
        # jsonschema accepts `minimum: NaN` and crashes on `multipleOf: NaN`.
        schema_nonfinite = _first_nonfinite_path(schema)
        if schema_nonfinite is not None:
            return (
                f"{_SCHEMA_ERROR_PREFIX}non-finite number at "
                f"'{_truncate(schema_nonfinite, _MAX_PATH_CHARS)}'"
            )

        # Compile-sized patterns are bounded before check_schema's regex format
        # check; matching is separately protected by the runtime deadline.
        _preflight_pattern_sizes(schema)
        _SafeDraft202012Validator.check_schema(schema)
        _preflight_refs(schema)
    except jsonschema_exceptions.SchemaError as exc:
        return f"{_SCHEMA_ERROR_PREFIX}{_truncate(exc.message)}"
    except Unresolvable as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unresolvable $ref ({_truncate(str(exc))})"
    except _SchemaPolicyError as exc:
        return f"{_SCHEMA_ERROR_PREFIX}{exc}"
    except RecursionError:
        return f"{_SCHEMA_ERROR_PREFIX}schema nesting is too deep to validate"
    return None


def _preflight_pattern_sizes(value: Any) -> None:
    """Reject patterns large enough to make compilation a resource risk."""
    if isinstance(value, dict):
        pattern = value.get("pattern")
        if isinstance(pattern, str) and len(pattern) > _MAX_PATTERN_CHARS:
            raise _SchemaPolicyError(
                f"pattern exceeds {_MAX_PATTERN_CHARS} characters"
            )
        pattern_properties = value.get("patternProperties")
        if isinstance(pattern_properties, dict):
            for candidate in pattern_properties:
                if (
                    isinstance(candidate, str)
                    and len(candidate) > _MAX_PATTERN_CHARS
                ):
                    raise _SchemaPolicyError(
                        "patternProperties key exceeds "
                        f"{_MAX_PATTERN_CHARS} characters"
                    )
        for child in value.values():
            _preflight_pattern_sizes(child)
    elif isinstance(value, list):
        for child in value:
            _preflight_pattern_sizes(child)


def _preflight_refs(schema: dict[str, Any]) -> None:
    """Resolve every schema ref through the non-retrieving registry."""
    root = Resource.from_contents(schema, default_specification=DRAFT202012)
    resolver = _NO_FETCH_REGISTRY.resolver_with_root(root)
    _walk_refs(root, resolver)


def _walk_refs(resource, resolver) -> None:
    contents = resource.contents
    if isinstance(contents, dict):
        for keyword in ("$ref", "$dynamicRef"):
            ref = contents.get(keyword)
            if isinstance(ref, str):
                resolver.lookup(ref)
    for subresource in resource.subresources():
        _walk_refs(subresource, resolver.in_subresource(subresource))


def _field_error(path: str, message: str) -> str:
    return f"field '{_truncate(path, _MAX_PATH_CHARS)}': {message}"


def _truncate(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    """Bound text while preserving context at the start and reason at the end."""
    if len(text) <= limit:
        return text
    marker = "... [truncated] ..."
    available = limit - len(marker)
    head = available // 2
    tail = available - head
    return text[:head] + marker + text[-tail:]


def _first_nonfinite_path(value: Any, path: str = "") -> str | None:
    """Return the dotted path of the first NaN/Infinity float, if any."""
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
    """Return one bounded, model-actionable line with the offending field."""
    location = ".".join(str(part) for part in error.absolute_path)
    message = _truncate(error.message)
    return _field_error(location, message) if location else message
