"""Shared shallow JSON-Schema check used both for agent structured output
(gateway, §3.4) and toolkit input validation before dispatch (§8a). Minimal
by design for Phase 0a/0b — type, required keys, top-level enums, and
`additionalProperties: false` — which is what the §4 example schemas use. A
full `jsonschema` validator is a drop-in upgrade when conformance becomes
load-bearing (tracked as a deferred item).
"""

from __future__ import annotations

from typing import Any


def validate_json(payload: Any, schema: dict[str, Any] | None) -> str | None:
    """Returns an error string if payload violates the schema, else None."""
    if schema is None:
        return None if isinstance(payload, dict) else "expected a JSON object"
    if schema.get("type") == "object" and not isinstance(payload, dict):
        return "expected a JSON object"
    if isinstance(payload, dict):
        for key in schema.get("required", []):
            if key not in payload:
                return f"missing required field '{key}'"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(payload) - set(properties)
            if extra:
                return f"unexpected field(s): {sorted(extra)}"
        for key, spec in properties.items():
            if key in payload and "enum" in spec and payload[key] not in spec["enum"]:
                return f"field '{key}' must be one of {spec['enum']}, got {payload[key]!r}"
    return None
