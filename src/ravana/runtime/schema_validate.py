"""Shared JSON-Schema validation for agent output and toolkit input.

Backed by a real Draft 2020-12 validator rather than the previous shallow
top-level check. Errors are returned, never raised: the gateway feeds payload
violations into its repair loop, while the tool executor turns them into a
model-addressable tool error.

Schemas may come from workflow authors or an MCP server. They are therefore
validated before payloads, cannot retrieve external references, and cannot run
unbounded validation or regular-expression work in the runtime process.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import itertools
import math
import time
from typing import Any, Iterator

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
_MAX_VALIDATION_STEPS = 10_000
_VALIDATION_BUDGET_SECONDS = 0.5
_SUPPORTED_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Python's built-in `re` has no timeout. The alternate engine applies this
# shared regex-execution budget to every regex match in one validation, so many
# individually cheap patterns cannot multiply the limit.
_REGEX_BUDGET_SECONDS = 0.1


@dataclass
class _ValidationBudget:
    deadline: float
    regex_seconds_remaining: float
    steps_remaining: int


_VALIDATION_BUDGET: ContextVar[_ValidationBudget | None] = ContextVar(
    "ravana_schema_validation_budget",
    default=None,
)


class _RegexBudgetExceeded(RuntimeError):
    pass


class _ValidationBudgetExceeded(RuntimeError):
    pass


class _SchemaPolicyError(ValueError):
    pass


@contextmanager
def _validation_budget() -> Iterator[None]:
    token = _VALIDATION_BUDGET.set(
        _ValidationBudget(
            deadline=time.monotonic() + _VALIDATION_BUDGET_SECONDS,
            regex_seconds_remaining=_REGEX_BUDGET_SECONDS,
            steps_remaining=_MAX_VALIDATION_STEPS,
        )
    )
    try:
        yield
    finally:
        _VALIDATION_BUDGET.reset(token)


def _consume_validation_step() -> None:
    budget = _VALIDATION_BUDGET.get()
    if budget is None:
        return
    if time.monotonic() >= budget.deadline:
        raise _ValidationBudgetExceeded
    budget.steps_remaining -= 1
    if budget.steps_remaining < 0:
        raise _ValidationBudgetExceeded


def _regex_matches(pattern: str, value: str) -> bool:
    budget = _VALIDATION_BUDGET.get()
    if budget is None:
        raise RuntimeError("schema regex evaluated outside validation boundary")

    _consume_validation_step()
    overall_remaining = budget.deadline - time.monotonic()
    if overall_remaining <= 0:
        raise _ValidationBudgetExceeded
    if budget.regex_seconds_remaining <= 0:
        raise _RegexBudgetExceeded

    timeout = min(overall_remaining, budget.regex_seconds_remaining)
    limited_by_regex = budget.regex_seconds_remaining <= overall_remaining
    started = time.monotonic()
    try:
        return regex.search(pattern, value, timeout=timeout) is not None
    except TimeoutError as exc:
        if limited_by_regex:
            raise _RegexBudgetExceeded from exc
        raise _ValidationBudgetExceeded from exc
    finally:
        budget.regex_seconds_remaining -= time.monotonic() - started


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


def _additional_property_names(instance, schema):
    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    for key in instance:
        _consume_validation_step()
        if key in properties:
            continue
        if any(_regex_matches(pattern, key) for pattern in patterns):
            continue
        yield key


def _safe_additional_properties(
    validator,
    additional_properties,
    instance,
    schema,
):
    if not validator.is_type(instance, "object"):
        return

    extras = list(_additional_property_names(instance, schema))
    if validator.is_type(additional_properties, "object"):
        for key in extras:
            yield from validator.descend(
                instance[key],
                additional_properties,
                path=key,
            )
    elif additional_properties is False and extras:
        yield jsonschema_exceptions.ValidationError(
            f"additional properties are not allowed ({_summarize_keys(extras)})"
        )


def _errors_are_empty(errors) -> bool:
    return next(errors, None) is None


class _Jsonschema426Compatibility:
    """One boundary for private state relied on from jsonschema 4.26."""

    EVOLVE_FIELDS = (
        ("schema", "schema"),
        ("_ref_resolver", "resolver"),
        ("format_checker", "format_checker"),
        ("_registry", "registry"),
        ("_resolver", "_resolver"),
    )

    @staticmethod
    def evolve(validator, **changes):
        for attribute, argument in _Jsonschema426Compatibility.EVOLVE_FIELDS:
            if argument not in changes:
                changes[argument] = getattr(validator, attribute)
        return validator.__class__(**changes)

    @staticmethod
    def lookup_reference(validator, ref):
        resolver = getattr(validator, "_resolver", None)
        lookup = getattr(resolver, "lookup", None)
        if not callable(lookup):
            raise RuntimeError("unsupported jsonschema resolver API")
        return lookup(ref)


def _evaluated_property_keys(validator, instance, schema) -> set[Any]:
    """Draft 2020-12 evaluated-key discovery without stdlib regex calls.

    This mirrors jsonschema 4.26's helper because its implementation calls
    unbounded stdlib regex. Keep the dependency constrained in pyproject.toml
    and the reference/combinator conformance cases in test_schema_validate.py
    in sync with this compatibility copy.
    """
    _consume_validation_step()
    if validator.is_type(schema, "boolean"):
        return set()

    evaluated: set[Any] = set()
    for keyword in ("$ref", "$dynamicRef"):
        ref = schema.get(keyword)
        if ref is None:
            continue
        resolved = _Jsonschema426Compatibility.lookup_reference(validator, ref)
        evaluated.update(
            _evaluated_property_keys(
                validator.evolve(
                    schema=resolved.contents,
                    _resolver=resolved.resolver,
                ),
                instance,
                resolved.contents,
            )
        )

    properties = schema.get("properties")
    if validator.is_type(properties, "object"):
        evaluated.update(properties.keys() & instance.keys())

    for keyword in ("additionalProperties", "unevaluatedProperties"):
        subschema = schema.get(keyword)
        if subschema is None:
            continue
        for key, value in instance.items():
            _consume_validation_step()
            if _errors_are_empty(validator.descend(value, subschema)):
                evaluated.add(key)

    pattern_properties = schema.get("patternProperties", {})
    for key in instance:
        for pattern in pattern_properties:
            if _regex_matches(pattern, key):
                evaluated.add(key)

    for key, subschema in schema.get("dependentSchemas", {}).items():
        _consume_validation_step()
        if key in instance:
            evaluated.update(
                _evaluated_property_keys(validator, instance, subschema)
            )

    for keyword in ("allOf", "oneOf", "anyOf"):
        for subschema in schema.get(keyword, []):
            _consume_validation_step()
            if _errors_are_empty(validator.descend(instance, subschema)):
                evaluated.update(
                    _evaluated_property_keys(validator, instance, subschema)
                )

    if_schema = schema.get("if")
    if if_schema is not None:
        if validator.evolve(schema=if_schema).is_valid(instance):
            evaluated.update(
                _evaluated_property_keys(validator, instance, if_schema)
            )
            then_schema = schema.get("then")
            if then_schema is not None:
                evaluated.update(
                    _evaluated_property_keys(validator, instance, then_schema)
                )
        else:
            else_schema = schema.get("else")
            if else_schema is not None:
                evaluated.update(
                    _evaluated_property_keys(validator, instance, else_schema)
                )

    return evaluated


def _safe_unevaluated_properties(
    validator,
    unevaluated_properties,
    instance,
    schema,
):
    if not validator.is_type(instance, "object"):
        return

    evaluated = _evaluated_property_keys(validator, instance, schema)
    invalid = []
    for key, value in instance.items():
        _consume_validation_step()
        if key not in evaluated and not _errors_are_empty(
            validator.descend(
                value,
                unevaluated_properties,
                path=key,
                schema_path=key,
            )
        ):
            invalid.append(key)

    if invalid:
        if unevaluated_properties is False:
            message = "unevaluated properties are not allowed"
        else:
            message = "unevaluated properties do not satisfy the schema"
        yield jsonschema_exceptions.ValidationError(
            f"{message} ({_summarize_keys(invalid)})"
        )


def _freeze_json(value: Any) -> Any:
    """Build a hashable key with JSON Schema's equality semantics."""
    _consume_validation_step()
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_freeze_json(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            frozenset(
                (_freeze_json(key), _freeze_json(item))
                for key, item in value.items()
            ),
        )
    return ("python-object", type(value), id(value))


def _safe_unique_items(validator, unique_items, instance, schema):
    if not unique_items or not validator.is_type(instance, "array"):
        return

    seen = set()
    for item in instance:
        frozen = _freeze_json(item)
        if frozen in seen:
            yield jsonschema_exceptions.ValidationError(
                "array has non-unique elements"
            )
            return
        seen.add(frozen)


def _bounded_keyword(validate):
    def bounded(validator, keyword_value, instance, schema):
        _consume_validation_step()
        yield from validate(validator, keyword_value, instance, schema)

    return bounded


_KEYWORD_VALIDATORS = dict(Draft202012Validator.VALIDATORS)
_KEYWORD_VALIDATORS.update(
    {
        "additionalProperties": _safe_additional_properties,
        "pattern": _safe_pattern,
        "patternProperties": _safe_pattern_properties,
        "unevaluatedProperties": _safe_unevaluated_properties,
        "uniqueItems": _safe_unique_items,
    }
)

_SafeDraft202012Validator = validators.extend(
    Draft202012Validator,
    validators={
        keyword: _bounded_keyword(validate)
        for keyword, validate in _KEYWORD_VALIDATORS.items()
    },
)


# jsonschema's generated evolve() dispatches on an embedded `$schema` and would
# silently switch this class back to an unbounded stock validator. Subclassing
# generated validator classes is unsupported, so install the compatibility
# hook directly and keep jsonschema constrained to the verified minor release.
setattr(
    _SafeDraft202012Validator,
    "evolve",
    _Jsonschema426Compatibility.evolve,
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
        try:
            with _validation_budget():
                # A schema-less node still rejects non-JSON floats before
                # accepting its whole object as a state delta.
                nonfinite = _first_nonfinite_path(payload)
                if nonfinite is not None:
                    return _field_error(
                        nonfinite,
                        "not a finite JSON number "
                        "(NaN/Infinity are not valid JSON)",
                    )
                return (
                    None
                    if isinstance(payload, dict)
                    else "expected a JSON object"
                )
        except _ValidationBudgetExceeded:
            return "payload exceeds the validation resource limit"

    # Authoring defects take precedence. Otherwise a NaN payload can mask a
    # malformed schema and make the repair loop blame the model.
    schema_error = _validate_schema(schema)
    if schema_error is not None:
        return schema_error

    try:
        with _validation_budget():
            nonfinite = _first_nonfinite_path(payload)
            if nonfinite is not None:
                return _field_error(
                    nonfinite,
                    "not a finite JSON number "
                    "(NaN/Infinity are not valid JSON)",
                )

            validator = _SafeDraft202012Validator(
                schema,
                registry=_NO_FETCH_REGISTRY,
            )
            collected = list(
                itertools.islice(
                    validator.iter_errors(payload),
                    _MAX_COLLECTED_ERRORS,
                )
            )
    except _RegexBudgetExceeded:
        return (
            f"{_SCHEMA_ERROR_PREFIX}regular-expression evaluation exceeded "
            f"{_REGEX_BUDGET_SECONDS:g}s limit"
        )
    except _ValidationBudgetExceeded:
        return "payload exceeds the validation resource limit"
    except jsonschema_exceptions.UnknownType as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unknown type {exc.type!r}"
    except Unresolvable as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unresolvable $ref ({_truncate(str(exc))})"

    if not collected:
        return None
    errors = sorted(collected, key=lambda error: list(error.absolute_path))
    return "; ".join(_describe(error) for error in errors[:_MAX_REPORTED_ERRORS])


def _validate_schema(schema: dict[str, Any]) -> str | None:
    """Validate author-controlled structure and every reference."""
    try:
        with _validation_budget():
            # jsonschema accepts `minimum: NaN` and crashes on `multipleOf: NaN`.
            schema_nonfinite = _first_nonfinite_path(schema)
            if schema_nonfinite is not None:
                return (
                    f"{_SCHEMA_ERROR_PREFIX}non-finite number at "
                    f"'{_truncate(schema_nonfinite, _MAX_PATH_CHARS)}'"
                )

            root = Resource.from_contents(
                schema,
                default_specification=DRAFT202012,
            )
            # Bound compile-sized patterns before check_schema's regex format
            # check, but inspect only actual subschema locations.
            _preflight_schema_resources(root)
            _SafeDraft202012Validator.check_schema(schema)
            _preflight_refs(root)
    except jsonschema_exceptions.SchemaError as exc:
        return f"{_SCHEMA_ERROR_PREFIX}{_truncate(exc.message)}"
    except Unresolvable as exc:
        return f"{_SCHEMA_ERROR_PREFIX}unresolvable $ref ({_truncate(str(exc))})"
    except _RegexBudgetExceeded:
        return (
            f"{_SCHEMA_ERROR_PREFIX}regular-expression evaluation exceeded "
            f"{_REGEX_BUDGET_SECONDS:g}s limit"
        )
    except _ValidationBudgetExceeded:
        return f"{_SCHEMA_ERROR_PREFIX}schema exceeds the validation resource limit"
    except _SchemaPolicyError as exc:
        return f"{_SCHEMA_ERROR_PREFIX}{exc}"
    except RecursionError:
        return f"{_SCHEMA_ERROR_PREFIX}schema nesting is too deep to validate"
    return None


def _preflight_schema_resources(resource) -> None:
    """Enforce dialect and compile-size policy on real schema resources."""
    _consume_validation_step()
    contents = resource.contents
    if isinstance(contents, dict):
        dialect = contents.get("$schema")
        if (
            isinstance(dialect, str)
            and dialect.removesuffix("#") != _SUPPORTED_DIALECT
        ):
            raise _SchemaPolicyError(
                f"unsupported $schema dialect {_truncate(dialect)!r}"
            )

        pattern = contents.get("pattern")
        if isinstance(pattern, str) and len(pattern) > _MAX_PATTERN_CHARS:
            raise _SchemaPolicyError(
                f"pattern exceeds {_MAX_PATTERN_CHARS} characters"
            )
        pattern_properties = contents.get("patternProperties")
        if isinstance(pattern_properties, dict):
            for candidate in pattern_properties:
                _consume_validation_step()
                if (
                    isinstance(candidate, str)
                    and len(candidate) > _MAX_PATTERN_CHARS
                ):
                    raise _SchemaPolicyError(
                        "patternProperties key exceeds "
                        f"{_MAX_PATTERN_CHARS} characters"
                    )
    for subresource in resource.subresources():
        _preflight_schema_resources(subresource)


def _preflight_refs(root) -> None:
    """Resolve every schema ref through the non-retrieving registry."""
    resolver = _NO_FETCH_REGISTRY.resolver_with_root(root)
    _walk_refs(root, resolver)


def _walk_refs(resource, resolver) -> None:
    _consume_validation_step()
    contents = resource.contents
    if isinstance(contents, dict):
        for keyword in ("$ref", "$dynamicRef"):
            ref = contents.get(keyword)
            if isinstance(ref, str):
                resolver.lookup(ref)
    for subresource in resource.subresources():
        _walk_refs(subresource, resolver.in_subresource(subresource))


def _summarize_keys(keys: list[Any]) -> str:
    shown = ", ".join(
        repr(_truncate(str(key), _MAX_PATH_CHARS))
        for key in keys[:_MAX_REPORTED_ERRORS]
    )
    remaining = len(keys) - _MAX_REPORTED_ERRORS
    return f"{shown}, and {remaining} more" if remaining > 0 else shown


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
    _consume_validation_step()
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
