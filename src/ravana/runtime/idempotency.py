"""§3.6's content-addressed tool-call idempotency key — the fix for the P1
finding that `hash(run_id, node_id, attempt)` changes on every retry by
construction (a retry *is* a new attempt), making "dedupe on this key"
impossible exactly when it's needed. This key depends on what's being asked
for, not on which dispatch asked for it, so a retry that reissues the
identical call reproduces the identical key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_idempotency_key(run_id: str, node_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    payload = f"{run_id}:{node_id}:{tool_name}:{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
