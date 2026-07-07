"""§2.2's audit_log — every config-plane mutation and manual operator action,
distinct from state_transition_log (runtime routing history inside a Run).
Phase 0a has no RBAC (§12 defers that to Phase 2), so `actor` is just
whoever the CLI was invoked as, but the trail still starts from day one
per the roadmap note in ARCHITECTURE.md §12.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ravana.schema.util import dumps, new_id, now_iso


def write_audit(
    con: sqlite3.Connection,
    org_id: str,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    con.execute(
        """INSERT INTO audit_log (id, org_id, actor, action, entity_type, entity_id, before, after, metadata, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (new_id(), org_id, actor, action, entity_type, entity_id, dumps(before), dumps(after), dumps(metadata), now_iso()),
    )
    con.commit()
