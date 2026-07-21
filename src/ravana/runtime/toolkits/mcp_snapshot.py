"""Durable per-run MCP tool snapshots and their lifecycle."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ravana.runtime.providers.base import Tool
from ravana.runtime.toolkits.base import ToolFailureKind, ToolkitError
from ravana.schema.util import now_iso

_ORPHAN_SNAPSHOT_GRACE_SECONDS = 3600
_TERMINAL_RUN_STATUSES = ("COMPLETED", "FAILED", "CANCELLED")


class McpToolSnapshotStore:
    """Own SQLite persistence independently from MCP process transport."""

    def __init__(self, con: sqlite3.Connection):
        self._con = con

    def restore(
        self,
        run_id: str,
        toolkit_id: str,
        expected_fingerprint: str,
    ) -> dict[str, Tool] | None:
        rows = self._con.execute(
            """SELECT server_fingerprint, tool_name, description, input_schema
               FROM mcp_tool_snapshot
               WHERE run_id = ? AND toolkit_id = ?
               ORDER BY tool_name""",
            (run_id, toolkit_id),
        ).fetchall()
        if not rows:
            return None
        stored_fingerprints = {row["server_fingerprint"] for row in rows}
        if stored_fingerprints != {expected_fingerprint}:
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': admin definition or tool grant changed "
                f"after run '{run_id}' was prepared",
                kind=ToolFailureKind.FATAL,
            )

        pinned: dict[str, Tool] = {}
        try:
            for row in rows:
                schema = json.loads(row["input_schema"])
                if not isinstance(schema, dict):
                    raise ValueError("input schema is not an object")
                pinned[row["tool_name"]] = Tool(
                    name=row["tool_name"],
                    description=row["description"],
                    input_schema=schema,
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolkitError(
                f"mcp_server '{toolkit_id}': stored tool snapshot is invalid",
                kind=ToolFailureKind.FATAL,
            ) from exc
        return pinned

    def persist(
        self,
        run_id: str,
        toolkit_id: str,
        fingerprint: str,
        pinned: dict[str, Tool],
    ) -> None:
        created_at = now_iso()
        self._con.execute(
            "DELETE FROM mcp_tool_snapshot WHERE run_id = ? AND toolkit_id = ?",
            (run_id, toolkit_id),
        )
        self._con.executemany(
            """INSERT INTO mcp_tool_snapshot
               (run_id, toolkit_id, server_fingerprint, tool_name, description, input_schema,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    toolkit_id,
                    fingerprint,
                    name,
                    spec.description,
                    json.dumps(spec.input_schema, sort_keys=True, separators=(",", ":")),
                    created_at,
                )
                for name, spec in pinned.items()
            ],
        )
        self._con.commit()

    def release(self, run_id: str) -> None:
        row = self._con.execute(
            "SELECT status FROM run WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None or row["status"] in _TERMINAL_RUN_STATUSES:
            self._con.execute(
                "DELETE FROM mcp_tool_snapshot WHERE run_id = ?", (run_id,)
            )
            self._con.commit()

    def cleanup(self) -> int:
        """Remove terminal snapshots and abandoned preparations after grace."""
        orphan_cutoff = (
            datetime.now(timezone.utc)
            - timedelta(seconds=_ORPHAN_SNAPSHOT_GRACE_SECONDS)
        ).isoformat()
        cursor = self._con.execute(
            """DELETE FROM mcp_tool_snapshot
               WHERE (NOT EXISTS (
                          SELECT 1 FROM run WHERE run.id = mcp_tool_snapshot.run_id
                      )
                      AND (created_at = '' OR created_at < ?))
                  OR run_id IN (
                         SELECT id FROM run WHERE status IN ('COMPLETED', 'FAILED', 'CANCELLED')
                     )""",
            (orphan_cutoff,),
        )
        self._con.commit()
        return cursor.rowcount
