"""§3.5's merge-policy-aware commit. Phase 0a is single-process (no real
contention on `state_version`, per §10.1), but the merge logic itself — which
policy applies to which key — is exercised identically to how the hosted
tiers would use it, so a broadcast branch's two writers combine correctly
rather than one clobbering the other.
"""

from __future__ import annotations

from typing import Any

from ravana.schema.models import StateConfig


def merge_delta(
    shared_state: dict[str, Any],
    delta: dict[str, Any],
    state_config: StateConfig,
) -> dict[str, Any]:
    """Returns a new dict; does not mutate shared_state in place, so callers
    can commit-or-discard atomically (mirrors the CAS commit at the DB layer)."""
    merged = dict(shared_state)
    for key, value in delta.items():
        field_schema = state_config.fields.get(key)
        policy = field_schema.merge if field_schema else "overwrite"
        if policy == "overwrite" or key not in merged:
            merged[key] = value
        elif policy == "merge-object":
            existing = merged.get(key) or {}
            if not isinstance(existing, dict) or not isinstance(value, dict):
                raise ValueError(f"merge-object policy on '{key}' requires object values, got {type(existing)}/{type(value)}")
            merged[key] = {**existing, **value}
        elif policy == "append":
            existing = merged.get(key) or []
            if not isinstance(existing, list):
                raise ValueError(f"append policy on '{key}' requires a list, got {type(existing)}")
            merged[key] = [*existing, *(value if isinstance(value, list) else [value])]
        else:  # pragma: no cover - StateFieldSchema.merge is a validated Literal
            raise ValueError(f"unknown merge policy: {policy}")
    return merged
