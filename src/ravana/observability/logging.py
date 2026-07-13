"""Minimal structured logging per §9's Runtime Logging & Correlation —
JSON lines to stdout, always tagged with run_id/node_execution_id when
available, since that correlation key is the whole point (§9: without it,
the infra log and the domain log are two disconnected debugging dead ends).
Phase 0a doesn't need Loki/OTel wiring yet, just the tagging convention
established from the start so nothing needs retrofitting later (§12).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ravana.runtime.secrets import redact_record
from ravana.schema.util import now_iso


def log_event(level: str, message: str, *, run_id: str | None = None, node_execution_id: str | None = None, **extra: Any) -> None:
    record = {
        "timestamp": now_iso(),
        "level": level,
        "message": message,
        "run_id": run_id,
        "node_execution_id": node_execution_id,
        **extra,
    }
    # §8's logging backstop: "logging must actively redact anything matching a
    # known secret pattern." Applied to the WHOLE record — message and every
    # string `**extra` field — at the single point every log line passes
    # through, so a new caller can't route a secret around it.
    try:
        line = json.dumps(redact_record(record))
    except Exception:  # noqa: BLE001 - a non-serializable extra field must not break the run
        line = json.dumps({"timestamp": record["timestamp"], "level": level, "message": str(message), "run_id": run_id})
    # BEST-EFFORT: an observability write (a closed/broken stderr, a full disk)
    # must NEVER propagate. This logger is called from inside the engine's
    # fail-closed paths — a raise here would escape past the run-status commit
    # and strand the run at RUNNING, which is the opposite of what a log line is
    # for. Swallow it; losing a log line is strictly better than losing the run.
    try:
        print(line, file=sys.stderr)
    except Exception:  # noqa: BLE001 - see above
        pass
