"""Writes a compiled, in-memory graph into the DB rows (agent/toolkit/skill/
workflow/workflow_node/workflow_edge, §2.2) that `run.workflow_id` and
friends actually reference. This is the "workflow save" side of `POST
/v1/workflows` (§7) — for Phase 0a's CLI there's no separate save/publish
step exposed yet, `ravana run start` just persists-then-runs in one shot.
"""

from __future__ import annotations

import sqlite3

from ravana.compiler.graph import CompiledGraph
from ravana.observability.audit import write_audit
from ravana.schema.util import dumps, loads, new_id, now_iso


class WorkflowPersistenceError(Exception):
    """A persisted workflow cannot accept the requested mutation."""


def _toolkit_snapshot(graph: CompiledGraph) -> dict[str, dict]:
    return {
        toolkit.id: {
            "type": toolkit.type,
            "config": toolkit.config,
            "auth_ref": toolkit.auth_ref,
            "description": toolkit.description,
        }
        for toolkit in graph.doc.spec.toolkits
    }


def _persisted_toolkits(con: sqlite3.Connection, toolkit_ids: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    if not toolkit_ids:
        return {}, {}
    placeholders = ",".join("?" for _ in toolkit_ids)
    rows = con.execute(
        f"SELECT id, name, type, config, auth_ref, description FROM toolkit WHERE id IN ({placeholders})",
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


def _sync_draft_toolkits(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    workflow_row: sqlite3.Row,
    org_id: str,
    actor: str,
) -> None:
    toolkit_ids = loads(workflow_row["toolkit_ids"]) or []
    before, persisted_ids = _persisted_toolkits(con, toolkit_ids)
    after = _toolkit_snapshot(graph)
    if before == after:
        return
    if workflow_row["status"] != "DRAFT":
        raise WorkflowPersistenceError(
            f"workflow '{graph.doc.metadata.name}' version {graph.doc.metadata.version} "
            f"is {workflow_row['status']} and cannot be edited in place"
        )

    current_ids: list[str] = []
    for toolkit in graph.doc.spec.toolkits:
        toolkit_db_id = persisted_ids.pop(toolkit.id) if toolkit.id in persisted_ids else new_id()
        current_ids.append(toolkit_db_id)
        if toolkit.id in before:
            con.execute(
                """UPDATE toolkit SET type = ?, config = ?, auth_ref = ?, description = ?
                   WHERE id = ?""",
                (toolkit.type, dumps(toolkit.config), toolkit.auth_ref, toolkit.description, toolkit_db_id),
            )
        else:
            con.execute(
                """INSERT INTO toolkit (id, org_id, name, type, config, auth_ref, description)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    toolkit_db_id, org_id, toolkit.id, toolkit.type,
                    dumps(toolkit.config), toolkit.auth_ref, toolkit.description,
                ),
            )

    for removed_id in persisted_ids.values():
        con.execute("DELETE FROM toolkit WHERE id = ?", (removed_id,))
    con.execute(
        "UPDATE workflow SET toolkit_ids = ? WHERE id = ?",
        (dumps(current_ids), workflow_row["id"]),
    )
    write_audit(
        con,
        org_id,
        actor,
        "workflow.draft_saved",
        "workflow",
        workflow_row["id"],
        before={"toolkits": before},
        after={"toolkits": after},
    )


def get_or_create_workflow(con: sqlite3.Connection, graph: CompiledGraph, org_id: str, created_by: str) -> str:
    """Reuse an identical persisted workflow or update its mutable DRAFT
    toolkit rows, preserving the (org_id, name, version) identity (§2.2)."""
    existing = con.execute(
        "SELECT id, status, toolkit_ids FROM workflow WHERE org_id = ? AND name = ? AND version = ?",
        (org_id, graph.doc.metadata.name, graph.doc.metadata.version),
    ).fetchone()
    if existing:
        _sync_draft_toolkits(con, graph, existing, org_id, created_by)
        return existing["id"]
    return persist_workflow(con, graph, org_id, created_by)


def persist_workflow(con: sqlite3.Connection, graph: CompiledGraph, org_id: str, created_by: str) -> str:
    doc = graph.doc
    now = now_iso()

    toolkit_db_ids = {toolkit.id: new_id() for toolkit in doc.spec.toolkits}

    agent_db_ids: dict[str, str] = {}
    for agent in doc.spec.agents:
        agent_db_id = new_id()
        agent_db_ids[agent.id] = agent_db_id
        con.execute(
            """INSERT INTO agent (id, org_id, name, system_prompt, llm_provider, llm_model, llm_endpoint,
                                   llm_api_key_ref, llm_fallback, temperature, max_tokens, output_schema,
                                   toolkit_ids, skill_ids, created_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                agent_db_id, org_id, agent.name, agent.system_prompt, agent.llm.provider, agent.llm.model,
                agent.llm.endpoint, agent.llm.api_key_ref, dumps([f.model_dump() for f in agent.llm.fallback]),
                agent.llm.temperature, agent.llm.max_tokens, dumps(agent.output_schema),
                dumps(agent.toolkits), dumps(agent.skills), created_by, now,
            ),
        )

    for toolkit in doc.spec.toolkits:
        toolkit_db_id = toolkit_db_ids[toolkit.id]
        con.execute(
            "INSERT INTO toolkit (id, org_id, name, type, config, auth_ref, description) VALUES (?,?,?,?,?,?,?)",
            (
                toolkit_db_id, org_id, toolkit.id, toolkit.type,
                dumps(toolkit.config), toolkit.auth_ref, toolkit.description,
            ),
        )

    for skill in doc.spec.skills:
        con.execute(
            """INSERT INTO skill (id, org_id, name, description, instructions, resources, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (new_id(), org_id, skill.id, skill.description, skill.instructions, dumps(skill.resources), now),
        )

    workflow_id = new_id()
    con.execute(
        """INSERT INTO workflow (id, org_id, name, description, version, state_schema, entry_node_id,
                                  dod_criteria, guards, concurrency, toolkit_ids, status, created_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            workflow_id, org_id, doc.metadata.name, doc.metadata.description, doc.metadata.version,
            dumps(doc.spec.state.model_dump(by_alias=True)), doc.spec.graph.entry,
            dumps(doc.spec.definition_of_done.model_dump() if doc.spec.definition_of_done else None),
            dumps(doc.spec.graph.guards.model_dump()),
            dumps(doc.spec.concurrency.model_dump() if doc.spec.concurrency else None),
            dumps(list(toolkit_db_ids.values())),
            "DRAFT", created_by, now,
        ),
    )

    for node in doc.spec.graph.nodes:
        contract = graph.contract_for_node(node.id) if node.agent else None
        con.execute(
            """INSERT INTO workflow_node
               (id, workflow_id, agent_id, sub_workflow_id, on_enter, join_policy,
                toolkit_ids, hitl_config, output_schema)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                node.id, workflow_id,
                agent_db_ids.get(node.agent) if node.agent else None,
                None,  # sub_workflow_id: not resolvable without a workflow name->id registry; out of scope for 0a
                node.on_enter,
                node.join,
                dumps(list(contract.toolkits)) if contract else dumps([]),
                dumps(contract.hitl.model_dump() if contract and contract.hitl else None),
                dumps(contract.output_schema if contract else None),
            ),
        )

    for edge in doc.spec.graph.edges:
        con.execute(
            """INSERT INTO workflow_edge (id, workflow_id, source_node_id, target_node_ids, condition_expr,
                                           is_default, priority)
               VALUES (?,?,?,?,?,?,?)""",
            (new_id(), workflow_id, edge.from_, dumps(edge.to), edge.condition, int(edge.is_default), edge.priority),
        )

    write_audit(
        con, org_id, created_by, "workflow.draft_saved", "workflow", workflow_id,
        after={
            "name": doc.metadata.name,
            "version": doc.metadata.version,
            "toolkits": _toolkit_snapshot(graph),
        },
    )
    return workflow_id
