"""Persist validated workflow aggregates and their audit history (§2.2/§7)."""

from __future__ import annotations

import sqlite3

from ravana.compiler.graph import CompiledGraph
from ravana.observability.audit import write_audit
from ravana.schema.models import ToolkitConfig
from ravana.schema.util import dumps, loads, new_id, now_iso
from ravana.schema.workflow_snapshot import WorkflowSnapshot


class WorkflowPersistenceError(Exception):
    """A persisted workflow cannot accept the requested mutation."""


# The toolkit columns that carry the authored definition — everything except
# the surrogate `id`/`org_id`. SELECT, INSERT, and UPDATE all derive their
# column list and value order from this one source, so a column added later
# cannot be written by one statement yet silently missed by the reader that
# rebuilds the row — the SELECT/INSERT drift a hand-copied list invites. The
# `name` column holds the author-facing toolkit id (`toolkit.id`).
_TOOLKIT_CONTENT_COLUMNS = ("name", "type", "config", "auth_ref", "description")


def _toolkit_content_values(toolkit: ToolkitConfig) -> tuple[str | None, ...]:
    """Values for `_TOOLKIT_CONTENT_COLUMNS`, in that exact column order."""
    return (
        toolkit.id,
        toolkit.type,
        dumps(toolkit.config),
        toolkit.auth_ref,
        toolkit.description,
    )


def _persisted_toolkits(
    con: sqlite3.Connection, toolkit_ids: list[str]
) -> tuple[dict[str, dict], dict[str, str]]:
    if not toolkit_ids:
        return {}, {}
    placeholders = ",".join("?" for _ in toolkit_ids)
    content_columns = ", ".join(_TOOLKIT_CONTENT_COLUMNS)
    rows = con.execute(
        f"SELECT id, {content_columns} FROM toolkit WHERE id IN ({placeholders})",
        toolkit_ids,
    ).fetchall()
    snapshot = {
        row["name"]: {
            "type": row["type"],
            "config": loads(row["config"]),
            "auth_ref": row["auth_ref"],
            "description": row["description"],
        }
        for row in rows
    }
    return snapshot, {row["name"]: row["id"] for row in rows}


def _claimed_toolkit_ids(con: sqlite3.Connection, workflow_id: str) -> set[str]:
    claimed: set[str] = set()
    for row in con.execute("SELECT toolkit_ids FROM workflow WHERE id != ?", (workflow_id,)):
        claimed.update(loads(row["toolkit_ids"]) or [])
    return claimed


def _claim_legacy_toolkits(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    org_id: str,
    workflow_id: str,
) -> list[str]:
    """Lazily associate pre-ownership toolkit rows with one legacy workflow.

    Old rows have no workflow FK. Prefer an exact persisted definition, then a
    same-name/type row so a description-only edit can retain the real `before`.
    Rows already claimed by another workflow are never reused.
    """
    claimed = _claimed_toolkit_ids(con, workflow_id)
    selected: list[str] = []
    content_columns = ", ".join(_TOOLKIT_CONTENT_COLUMNS)
    for toolkit in graph.doc.spec.toolkits:
        candidates = con.execute(
            f"""SELECT rowid AS row_order, id, {content_columns}
               FROM toolkit WHERE org_id = ? AND name = ? AND type = ? ORDER BY rowid""",
            (org_id, toolkit.id, toolkit.type),
        ).fetchall()
        available = [row for row in candidates if row["id"] not in claimed]
        if not available:
            continue

        def score(row: sqlite3.Row) -> tuple[int, int]:
            same_config = loads(row["config"]) == toolkit.config
            same_auth = row["auth_ref"] == toolkit.auth_ref
            same_description = row["description"] == toolkit.description
            if same_config and same_auth and same_description:
                rank = 0
            elif same_config and same_auth:
                rank = 1
            else:
                rank = 2
            return rank, row["row_order"]

        chosen = min(available, key=score)
        selected.append(chosen["id"])
        claimed.add(chosen["id"])
    return selected


def _sync_toolkits(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    org_id: str,
    owned_ids: list[str],
) -> list[str]:
    _, persisted_ids = _persisted_toolkits(con, owned_ids)
    current_ids: list[str] = []
    for toolkit in graph.doc.spec.toolkits:
        toolkit_db_id = persisted_ids.pop(toolkit.id) if toolkit.id in persisted_ids else new_id()
        current_ids.append(toolkit_db_id)
        if toolkit_db_id in owned_ids:
            set_clause = ", ".join(f"{column} = ?" for column in _TOOLKIT_CONTENT_COLUMNS)
            con.execute(
                f"UPDATE toolkit SET {set_clause} WHERE id = ?",
                (*_toolkit_content_values(toolkit), toolkit_db_id),
            )
        else:
            columns = ("id", "org_id", *_TOOLKIT_CONTENT_COLUMNS)
            column_list = ", ".join(columns)
            placeholders = ", ".join("?" for _ in columns)
            con.execute(
                f"INSERT INTO toolkit ({column_list}) VALUES ({placeholders})",
                (toolkit_db_id, org_id, *_toolkit_content_values(toolkit)),
            )
    for removed_id in persisted_ids.values():
        con.execute("DELETE FROM toolkit WHERE id = ?", (removed_id,))
    return current_ids


def _insert_agents(
    con: sqlite3.Connection, graph: CompiledGraph, org_id: str, actor: str, now: str
) -> dict[str, str]:
    agent_db_ids: dict[str, str] = {}
    # New rows preserve sender_agent_id history for Runs pinned to the prior
    # DRAFT while workflow_node moves to the edited aggregate.
    for agent in graph.doc.spec.agents:
        agent_db_id = new_id()
        agent_db_ids[agent.id] = agent_db_id
        con.execute(
            """INSERT INTO agent (id, org_id, name, system_prompt, llm_provider, llm_model, llm_endpoint,
                                   llm_api_key_ref, llm_fallback, temperature, max_tokens, output_schema,
                                   toolkit_ids, skill_ids, created_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                agent_db_id,
                org_id,
                agent.name,
                agent.system_prompt,
                agent.llm.provider,
                agent.llm.model,
                agent.llm.endpoint,
                agent.llm.api_key_ref,
                dumps([fallback.model_dump() for fallback in agent.llm.fallback]),
                agent.llm.temperature,
                agent.llm.max_tokens,
                dumps(agent.output_schema),
                dumps(agent.toolkits),
                dumps(agent.skills),
                actor,
                now,
            ),
        )
    return agent_db_ids


def _upsert_skills(con: sqlite3.Connection, graph: CompiledGraph, org_id: str, now: str) -> None:
    for skill in graph.doc.spec.skills:
        existing = con.execute(
            "SELECT id FROM skill WHERE org_id = ? AND name = ? AND version = 1",
            (org_id, skill.id),
        ).fetchone()
        if existing:
            con.execute(
                """UPDATE skill SET description = ?, instructions = ?, resources = ? WHERE id = ?""",
                (skill.description, skill.instructions, dumps(skill.resources), existing["id"]),
            )
        else:
            con.execute(
                """INSERT INTO skill (id, org_id, name, description, instructions, resources, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (new_id(), org_id, skill.id, skill.description, skill.instructions, dumps(skill.resources), now),
            )


def _replace_graph_rows(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    workflow_id: str,
    agent_db_ids: dict[str, str],
) -> None:
    con.execute("DELETE FROM workflow_edge WHERE workflow_id = ?", (workflow_id,))
    con.execute("DELETE FROM workflow_node WHERE workflow_id = ?", (workflow_id,))

    for node in graph.doc.spec.graph.nodes:
        contract = graph.contract_for_node(node.id) if node.agent else None
        con.execute(
            """INSERT INTO workflow_node
               (id, workflow_id, agent_id, sub_workflow_id, on_enter, join_policy,
                toolkit_ids, hitl_config, output_schema)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                node.id,
                workflow_id,
                agent_db_ids.get(node.agent) if node.agent else None,
                None,
                node.on_enter,
                node.join,
                dumps(list(contract.toolkits)) if contract else dumps([]),
                dumps(contract.hitl.model_dump() if contract and contract.hitl else None),
                dumps(contract.output_schema if contract else None),
            ),
        )

    for edge in graph.doc.spec.graph.edges:
        con.execute(
            """INSERT INTO workflow_edge (id, workflow_id, source_node_id, target_node_ids,
                                           condition_expr, is_default, priority)
               VALUES (?,?,?,?,?,?,?)""",
            (
                new_id(),
                workflow_id,
                edge.from_,
                dumps(edge.to),
                edge.condition,
                int(edge.is_default),
                edge.priority,
            ),
        )


def _update_workflow_row(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    workflow_id: str,
    toolkit_ids: list[str],
    snapshot: WorkflowSnapshot,
) -> None:
    doc = graph.doc
    con.execute(
        """UPDATE workflow SET description = ?, state_schema = ?, entry_node_id = ?,
           dod_criteria = ?, guards = ?, concurrency = ?, toolkit_ids = ?, definition_snapshot = ?
           WHERE id = ?""",
        (
            doc.metadata.description,
            dumps(doc.spec.state.model_dump(by_alias=True)),
            doc.spec.graph.entry,
            dumps(doc.spec.definition_of_done.model_dump() if doc.spec.definition_of_done else None),
            dumps(doc.spec.graph.guards.model_dump()),
            dumps(doc.spec.concurrency.model_dump() if doc.spec.concurrency else None),
            dumps(toolkit_ids),
            snapshot.to_json(),
            workflow_id,
        ),
    )


def _save_existing_workflow(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    workflow_row: sqlite3.Row,
    org_id: str,
    actor: str,
) -> str:
    incoming = WorkflowSnapshot.from_doc(graph.doc)
    stored_json = workflow_row["definition_snapshot"]
    if stored_json is not None:
        stored = WorkflowSnapshot.from_json(stored_json)
        if stored == incoming:
            return workflow_row["id"]
        before: dict = stored.payload
    else:
        if workflow_row["status"] != "DRAFT":
            raise WorkflowPersistenceError(
                "legacy published workflow has no verifiable definition snapshot; "
                "create a new version before running it"
            )
        owned_ids = loads(workflow_row["toolkit_ids"]) or _claim_legacy_toolkits(
            con, graph, org_id, workflow_row["id"]
        )
        legacy_toolkits, _ = _persisted_toolkits(con, owned_ids)
        before = {"legacy_definition_unavailable": True, "toolkits": legacy_toolkits}

    if workflow_row["status"] != "DRAFT":
        raise WorkflowPersistenceError(
            f"workflow '{graph.doc.metadata.name}' version {graph.doc.metadata.version} "
            f"is {workflow_row['status']} and cannot be edited in place"
        )

    owned_ids = loads(workflow_row["toolkit_ids"]) or _claim_legacy_toolkits(
        con, graph, org_id, workflow_row["id"]
    )
    now = now_iso()
    with con:
        toolkit_ids = _sync_toolkits(con, graph, org_id, owned_ids)
        agent_db_ids = _insert_agents(con, graph, org_id, actor, now)
        _upsert_skills(con, graph, org_id, now)
        _replace_graph_rows(con, graph, workflow_row["id"], agent_db_ids)
        _update_workflow_row(con, graph, workflow_row["id"], toolkit_ids, incoming)
        write_audit(
            con,
            org_id,
            actor,
            "workflow.draft_saved",
            "workflow",
            workflow_row["id"],
            before=before,
            after=incoming.payload,
            commit=False,
        )
    return workflow_row["id"]


def get_or_create_workflow(
    con: sqlite3.Connection, graph: CompiledGraph, org_id: str, actor: str
) -> str:
    """Create a workflow or atomically save a changed DRAFT aggregate."""
    existing = con.execute(
        """SELECT id, status, toolkit_ids, definition_snapshot FROM workflow
           WHERE org_id = ? AND name = ? AND version = ?""",
        (org_id, graph.doc.metadata.name, graph.doc.metadata.version),
    ).fetchone()
    if existing:
        return _save_existing_workflow(con, graph, existing, org_id, actor)
    return persist_workflow(con, graph, org_id, actor)


def persist_workflow(
    con: sqlite3.Connection, graph: CompiledGraph, org_id: str, actor: str
) -> str:
    doc = graph.doc
    now = now_iso()
    workflow_id = new_id()
    snapshot = WorkflowSnapshot.from_doc(doc)

    with con:
        toolkit_ids = _sync_toolkits(con, graph, org_id, [])
        agent_db_ids = _insert_agents(con, graph, org_id, actor, now)
        _upsert_skills(con, graph, org_id, now)
        con.execute(
            """INSERT INTO workflow
               (id, org_id, name, description, version, state_schema, entry_node_id,
                dod_criteria, guards, concurrency, toolkit_ids, definition_snapshot,
                status, created_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                workflow_id,
                org_id,
                doc.metadata.name,
                doc.metadata.description,
                doc.metadata.version,
                dumps(doc.spec.state.model_dump(by_alias=True)),
                doc.spec.graph.entry,
                dumps(doc.spec.definition_of_done.model_dump() if doc.spec.definition_of_done else None),
                dumps(doc.spec.graph.guards.model_dump()),
                dumps(doc.spec.concurrency.model_dump() if doc.spec.concurrency else None),
                dumps(toolkit_ids),
                snapshot.to_json(),
                "DRAFT",
                actor,
                now,
            ),
        )
        _replace_graph_rows(con, graph, workflow_id, agent_db_ids)
        write_audit(
            con,
            org_id,
            actor,
            "workflow.draft_saved",
            "workflow",
            workflow_id,
            after=snapshot.payload,
            commit=False,
        )
    return workflow_id
