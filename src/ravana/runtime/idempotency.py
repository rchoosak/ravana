"""§3.6's logical-invocation idempotency key.

`logical_visit_id` stays stable across retry attempts but changes when the
graph enters the node again. `tool_call_ordinal` distinguishes two intentional
identical calls within that visit. The command content remains in the hash so a
retry whose model changes the call at the same position cannot replay a stale
result.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_idempotency_key(
    run_id: str,
    node_id: str,
    logical_visit_id: str,
    tool_call_ordinal: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    payload = f"{run_id}:{node_id}:{logical_visit_id}:{tool_call_ordinal}:{tool_name}:{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
