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
from ravana.schema.util import dumps, new_id, now_iso


def get_or_create_workflow(con: sqlite3.Connection, graph: CompiledGraph, org_id: str, created_by: str) -> str:
    """Idempotent wrapper: repeated `ravana run start` invocations against the
    same workflow file (same org/name/version) should reuse the persisted
    row rather than hit the (org_id, name, version) UNIQUE constraint (§2.2)."""
    existing = con.execute(
        "SELECT id FROM workflow WHERE org_id = ? AND name = ? AND version = ?",
        (org_id, graph.doc.metadata.name, graph.doc.metadata.version),
    ).fetchone()
    if existing:
        return existing["id"]
    return persist_workflow(con, graph, org_id, created_by)


def persist_workflow(con: sqlite3.Connection, graph: CompiledGraph, org_id: str, created_by: str) -> str:
    doc = graph.doc
    now = now_iso()

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

    toolkit_db_ids: dict[str, str] = {}
    for toolkit in doc.spec.toolkits:
        toolkit_db_id = new_id()
        toolkit_db_ids[toolkit.id] = toolkit_db_id
        con.execute(
            "INSERT INTO toolkit (id, org_id, name, type, config, auth_ref) VALUES (?,?,?,?,?,?)",
            (toolkit_db_id, org_id, toolkit.id, toolkit.type, dumps(toolkit.config), toolkit.auth_ref),
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
                                  dod_criteria, guards, concurrency, status, created_by, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            workflow_id, org_id, doc.metadata.name, doc.metadata.description, doc.metadata.version,
            dumps(doc.spec.state.model_dump(by_alias=True)), doc.spec.graph.entry,
            dumps(doc.spec.definition_of_done.model_dump() if doc.spec.definition_of_done else None),
            dumps(doc.spec.graph.guards.model_dump()),
            dumps(doc.spec.concurrency.model_dump() if doc.spec.concurrency else None),
            "DRAFT", created_by, now,
        ),
    )

    for node in doc.spec.graph.nodes:
        con.execute(
            """INSERT INTO workflow_node (id, workflow_id, agent_id, sub_workflow_id, on_enter, join_policy, hitl_config)
               VALUES (?,?,?,?,?,?,?)""",
            (
                node.id, workflow_id,
                agent_db_ids.get(node.agent) if node.agent else None,
                None,  # sub_workflow_id: not resolvable without a workflow name->id registry; out of scope for 0a
                node.on_enter,
                node.join,
                dumps(_agent_hitl_config(doc, node.agent)),
            ),
        )

    for edge in doc.spec.graph.edges:
        con.execute(
            """INSERT INTO workflow_edge (id, workflow_id, source_node_id, target_node_ids, condition_expr,
                                           is_default, priority)
               VALUES (?,?,?,?,?,?,?)""",
            (new_id(), workflow_id, edge.from_, dumps(edge.to), edge.condition, int(edge.is_default), edge.priority),
        )

    con.commit()
    write_audit(
        con, org_id, created_by, "workflow.draft_saved", "workflow", workflow_id,
        after={"name": doc.metadata.name, "version": doc.metadata.version},
    )
    return workflow_id


def _agent_hitl_config(doc, agent_id: str | None) -> dict | None:
    if agent_id is None:
        return None
    agent = next((a for a in doc.spec.agents if a.id == agent_id), None)
    if agent is None or agent.hitl is None:
        return None
    return agent.hitl.model_dump()
