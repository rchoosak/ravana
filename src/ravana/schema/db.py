"""SQLite translation of ARCHITECTURE.md §2.2's Postgres schema, for the
Local/Embedded tier (§10.1). Translation rules applied uniformly:

- UUID          -> TEXT (str(uuid4()))
- TIMESTAMPTZ   -> TEXT (ISO-8601 UTC, set by the application, not a SQL DEFAULT)
- JSONB         -> TEXT (json.dumps at the app layer; see ravana.schema.jsonutil)
- TEXT[]/UUID[] -> TEXT (JSON-array-encoded, same as JSONB)
- NUMERIC(p,s)  -> REAL
- BOOLEAN       -> INTEGER (0/1)
- BIGINT        -> INTEGER (SQLite INTEGER is already 8-byte)

Two invariants from §2.2 are preserved as real CHECK/FOREIGN KEY constraints,
matching the fixes applied to the Postgres DDL after the P2 review:
`workflow_node_backing_xor` and `workflow_edge`'s source_node_id FK / non-empty
target list. `target_node_ids` element-level integrity (each id resolves to a
real node or the sentinel '__terminal__') is deliberately NOT a DB constraint
here either, for the same reason noted in ARCHITECTURE.md: it's the Workflow
Compiler's job (see ravana.compiler.validate).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agent (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    role            TEXT,
    system_prompt   TEXT NOT NULL,
    llm_provider    TEXT NOT NULL,
    llm_model       TEXT NOT NULL,
    llm_endpoint    TEXT,
    llm_api_key_ref TEXT,
    llm_fallback    TEXT,
    temperature     REAL DEFAULT 0.2,
    max_tokens      INTEGER,
    output_schema   TEXT,
    toolkit_ids     TEXT NOT NULL DEFAULT '[]',
    skill_ids       TEXT NOT NULL DEFAULT '[]',
    version         INTEGER NOT NULL DEFAULT 1,
    created_by      TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS toolkit (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    config          TEXT NOT NULL,
    auth_ref        TEXT
);

CREATE TABLE IF NOT EXISTS skill (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT NOT NULL,
    instructions    TEXT NOT NULL,
    resources       TEXT,
    version         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    UNIQUE (org_id, name, version)
);

CREATE TABLE IF NOT EXISTS workflow (
    id              TEXT PRIMARY KEY,
    org_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    description     TEXT,
    version         INTEGER NOT NULL DEFAULT 1,
    state_schema    TEXT NOT NULL,
    entry_node_id   TEXT NOT NULL,
    dod_criteria    TEXT,
    guards          TEXT,
    concurrency     TEXT,
    status          TEXT NOT NULL DEFAULT 'DRAFT',
    created_by      TEXT NOT NULL,
    published_by    TEXT,
    published_at    TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (org_id, name, version)
);

CREATE TABLE IF NOT EXISTS workflow_node (
    id              TEXT NOT NULL,
    workflow_id     TEXT NOT NULL REFERENCES workflow(id),
    agent_id        TEXT REFERENCES agent(id),
    sub_workflow_id TEXT REFERENCES workflow(id),
    on_enter        TEXT,
    join_policy     TEXT NOT NULL DEFAULT 'any',
    hitl_config     TEXT,
    PRIMARY KEY (workflow_id, id),
    CONSTRAINT workflow_node_backing_xor CHECK (
        (agent_id IS NOT NULL AND sub_workflow_id IS NULL) OR
        (agent_id IS NULL AND sub_workflow_id IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS workflow_edge (
    id                TEXT PRIMARY KEY,
    workflow_id       TEXT NOT NULL REFERENCES workflow(id),
    source_node_id    TEXT NOT NULL,
    target_node_ids   TEXT NOT NULL,
    condition_expr    TEXT,
    is_default        INTEGER NOT NULL DEFAULT 0,
    priority          INTEGER DEFAULT 0,
    FOREIGN KEY (workflow_id, source_node_id) REFERENCES workflow_node (workflow_id, id),
    CONSTRAINT workflow_edge_has_targets CHECK (json_array_length(target_node_ids) > 0)
);

CREATE TABLE IF NOT EXISTS run (
    id                TEXT PRIMARY KEY,
    org_id            TEXT NOT NULL,
    workflow_id       TEXT NOT NULL REFERENCES workflow(id),
    workflow_version  INTEGER NOT NULL,
    status            TEXT NOT NULL,
    current_nodes     TEXT NOT NULL DEFAULT '[]',
    shared_state      TEXT NOT NULL DEFAULT '{}',
    state_version     INTEGER NOT NULL DEFAULT 0,
    concurrency_group TEXT,
    parent_run_id     TEXT REFERENCES run(id),
    parent_node_execution_id TEXT,
    triggered_by      TEXT,
    input_payload     TEXT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_concurrency ON run (workflow_id, concurrency_group, status);

CREATE TABLE IF NOT EXISTS node_execution (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES run(id),
    node_id             TEXT NOT NULL,
    attempt             INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'QUEUED',
    leased_by           TEXT,
    leased_until        TEXT,
    error               TEXT,
    repair_count        INTEGER NOT NULL DEFAULT 0,
    tool_call_count     INTEGER NOT NULL DEFAULT 0,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL NOT NULL DEFAULT 0,
    started_at          TEXT,
    finished_at         TEXT,
    UNIQUE (run_id, node_id, attempt)
);

CREATE TABLE IF NOT EXISTS message (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES run(id),
    node_id             TEXT NOT NULL,
    sender_agent_id     TEXT REFERENCES agent(id),
    recipient           TEXT,
    role                TEXT NOT NULL,
    content             TEXT,
    structured_payload  TEXT,
    tool_calls          TEXT,
    parent_message_id   TEXT REFERENCES message(id),
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact (
    id                TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES run(id),
    produced_by_node  TEXT NOT NULL,
    type              TEXT NOT NULL,
    storage_uri       TEXT NOT NULL,
    version           INTEGER DEFAULT 1,
    metadata          TEXT,
    checksum          TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_transition_log (
    id                    TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL REFERENCES run(id),
    sequence              INTEGER NOT NULL,
    node_execution_id     TEXT REFERENCES node_execution(id),
    event_type            TEXT NOT NULL,
    from_node             TEXT,
    to_node               TEXT,
    condition_evaluated   TEXT,
    result                INTEGER,
    state_diff            TEXT,
    state_version_before  INTEGER,
    state_version_after   INTEGER,
    created_at            TEXT NOT NULL,
    UNIQUE (run_id, sequence)
);

CREATE TABLE IF NOT EXISTS hitl_request (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES run(id),
    node_id         TEXT NOT NULL,
    question        TEXT NOT NULL,
    options         TEXT,
    assignee        TEXT,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    response        TEXT,
    responded_by    TEXT,
    responded_at    TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            TEXT PRIMARY KEY,
    org_id        TEXT NOT NULL,
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    entity_type   TEXT NOT NULL,
    entity_id     TEXT NOT NULL,
    before        TEXT,
    after         TEXT,
    metadata      TEXT,
    created_at    TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas Ravana's invariants depend on.

    foreign_keys must be turned on per-connection in SQLite (unlike Postgres,
    it defaults off) or workflow_edge's source_node_id FK and the run/agent/
    workflow REFERENCES are silently unenforced.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create the schema (idempotent — CREATE TABLE IF NOT EXISTS throughout)."""
    con = connect(db_path)
    con.executescript(SCHEMA_SQL)
    con.commit()
    return con
