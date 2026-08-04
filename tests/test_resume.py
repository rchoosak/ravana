"""Tests for the v0.14 fix: a HITL response dispatches a *new* node_execution
attempt for the paused node (the agent actually re-runs with the human's
answer in context), not a bare re-route of the stale first-turn output."""

from __future__ import annotations

import asyncio
import sqlite3

import click
import pytest

from ravana.cli import _compiled_graph_for_run
from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import resume_hitl, start_run
from ravana.schema.db import init_db
from ravana.schema.util import loads, now_iso


class _PreparedRuntime:
    def __init__(self, inner):
        self.inner = inner
        self.prepared: list[str] = []

    async def prepare_run(self, run_id: str) -> None:
        self.prepared.append(run_id)

    async def run_turn(self, **kwargs):
        return await self.inner.run_turn(**kwargs)

    async def aclose(self) -> None:
        await self.inner.aclose()


def test_resume_creates_new_attempt_not_a_bare_reroute(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()

    attempts_before = con.execute(
        "SELECT attempt, status FROM node_execution WHERE run_id = ? AND node_id = 'pm_intake' ORDER BY attempt",
        (run_id,),
    ).fetchall()
    assert [dict(r) for r in attempts_before] == [{"attempt": 1, "status": "WAITING_HUMAN"}]

    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "it's clear now"}))

    attempts_after = con.execute(
        "SELECT attempt, status FROM node_execution WHERE run_id = ? AND node_id = 'pm_intake' ORDER BY attempt",
        (run_id,),
    ).fetchall()
    assert [r["attempt"] for r in attempts_after] == [1, 2]
    assert attempts_after[0]["status"] == "WAITING_HUMAN"  # the paused attempt is left as-is, not mutated in place
    assert attempts_after[1]["status"] == "SUCCEEDED"  # the new attempt is what actually resolved things

    # The old attempt's output (requirement_clarity=LOW) must not be what
    # routing acted on — the new attempt's HIGH is what let the run proceed.
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert loads(run["shared_state"])["requirement_clarity"] == "HIGH"


def test_resume_appends_human_response_to_message_thread(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "it's clear now"}))

    messages = con.execute(
        "SELECT role, structured_payload FROM message WHERE run_id = ? AND node_id = 'pm_intake' ORDER BY created_at",
        (run_id,),
    ).fetchall()
    roles = [m["role"] for m in messages]
    assert roles == ["agent", "user", "agent"]  # first turn, human's answer, second (resumed) turn
    assert loads(messages[1]["structured_payload"]) == {"answer": "it's clear now"}


def test_resume_reprepares_run_scoped_resources(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    runtime = _PreparedRuntime(sdlc_runtime)
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    assert runtime.prepared == [run_id]

    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, runtime, run_id, hitl["id"], {"answer": "clear"}))

    assert runtime.prepared == [run_id, run_id]


def test_resuming_an_already_answered_hitl_request_is_rejected(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "ok"}))

    try:
        asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "again"}))
        assert False, "expected ValueError for double-answering the same hitl_request"
    except ValueError:
        pass


def test_resume_snapshot_preserves_descriptions_and_inherited_node_contracts(
    con, sdlc_graph, sdlc_runtime
):
    doc = sdlc_graph.doc.model_copy(deep=True)
    toolkit = next(t for t in doc.spec.toolkits if t.id == "git_connector")
    toolkit.description = "Description pinned when the run started."
    graph = compile_workflow(doc)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="test")

    run_id = asyncio.run(
        start_run(
            con,
            graph,
            sdlc_runtime,
            org_id="test",
            workflow_id=workflow_id,
            input_payload={"repository": "r"},
        )
    )
    run_row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()

    resumed_graph = _compiled_graph_for_run(con, run_row)
    resumed = next(t for t in resumed_graph.doc.spec.toolkits if t.id == "git_connector")
    assert resumed.description == "Description pinned when the run started."
    for node in graph.doc.spec.graph.nodes:
        if node.agent is not None:
            assert resumed_graph.contract_for_node(node.id) == graph.contract_for_node(node.id)


def test_resume_rejects_a_graph_that_differs_from_the_run_snapshot(
    con, sdlc_graph, sdlc_workflow_id, sdlc_runtime
):
    run_id = asyncio.run(
        start_run(
            con,
            sdlc_graph,
            sdlc_runtime,
            org_id="test",
            workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    hitl = con.execute(
        "SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)
    ).fetchone()

    changed_doc = sdlc_graph.doc.model_copy(deep=True)
    next(t for t in changed_doc.spec.toolkits if t.id == "git_connector").description = (
        "Changed after the run started."
    )
    changed_graph = compile_workflow(changed_doc)

    with pytest.raises(ValueError, match="must resume with its pinned workflow snapshot"):
        asyncio.run(
            resume_hitl(
                con,
                changed_graph,
                sdlc_runtime,
                run_id,
                hitl["id"],
                {"answer": "clear"},
            )
        )
    assert con.execute(
        "SELECT status FROM hitl_request WHERE id = ?", (hitl["id"],)
    ).fetchone()["status"] == "PENDING"


def test_draft_edit_cannot_change_the_agent_identity_pinned_to_an_inflight_run(
    con, sdlc_graph, sdlc_workflow_id, sdlc_runtime
):
    run_id = asyncio.run(
        start_run(
            con,
            sdlc_graph,
            sdlc_runtime,
            org_id="test-org",
            workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    hitl = con.execute(
        "SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)
    ).fetchone()
    run_row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    pinned_agent_id = loads(run_row["agent_db_ids"])["pm_intake"]

    changed_doc = sdlc_graph.doc.model_copy(deep=True)
    changed_doc.spec.agents[0].system_prompt = "Edited while the old run waits for HITL"
    changed_graph = compile_workflow(changed_doc)
    get_or_create_workflow(con, changed_graph, org_id="test-org", actor="editor")
    current_agent_id = con.execute(
        "SELECT agent_id FROM workflow_node WHERE workflow_id = ? AND id = 'pm_intake'",
        (sdlc_workflow_id,),
    ).fetchone()[0]
    assert current_agent_id != pinned_agent_id

    pinned_graph = _compiled_graph_for_run(con, run_row)
    asyncio.run(
        resume_hitl(
            con,
            pinned_graph,
            sdlc_runtime,
            run_id,
            hitl["id"],
            {"answer": "clear"},
        )
    )
    sender_ids = [
        row["sender_agent_id"]
        for row in con.execute(
            """SELECT sender_agent_id FROM message
               WHERE run_id = ? AND node_id = 'pm_intake' AND role = 'agent'
               ORDER BY created_at""",
            (run_id,),
        )
    ]
    assert sender_ids == [pinned_agent_id, pinned_agent_id]


def test_legacy_run_without_snapshot_fails_closed_on_every_resume_path(
    tmp_path, sdlc_graph, sdlc_runtime
):
    db_path = tmp_path / "legacy_run.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE workflow (id TEXT PRIMARY KEY);
        CREATE TABLE run (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            workflow_version INTEGER NOT NULL,
            status TEXT NOT NULL,
            current_nodes TEXT NOT NULL DEFAULT '[]',
            shared_state TEXT NOT NULL DEFAULT '{}',
            state_version INTEGER NOT NULL DEFAULT 0,
            concurrency_group TEXT,
            parent_run_id TEXT,
            parent_node_execution_id TEXT,
            triggered_by TEXT,
            input_payload TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT
        );
        """
    )
    legacy.close()
    con = init_db(db_path)
    run_id = "legacy-run"
    con.execute("INSERT INTO workflow (id) VALUES ('legacy-workflow')")
    con.execute(
        """INSERT INTO run
           (id, org_id, workflow_id, workflow_version, status, started_at)
           VALUES (?,?,?,?,?,?)""",
        (run_id, "test", "legacy-workflow", 1, "WAITING_HUMAN", now_iso()),
    )
    con.execute(
        """INSERT INTO hitl_request
           (id, run_id, node_id, question, status, created_at)
           VALUES (?,?,?,?,?,?)""",
        ("legacy-hitl", run_id, "pm_intake", "Continue?", "PENDING", now_iso()),
    )
    con.commit()
    run_row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()

    with pytest.raises(click.ClickException, match="cannot resume safely"):
        _compiled_graph_for_run(con, run_row)
    with pytest.raises(ValueError, match="cannot resume safely"):
        asyncio.run(
            resume_hitl(
                con,
                sdlc_graph,
                sdlc_runtime,
                run_id,
                "legacy-hitl",
                {"answer": "clear"},
            )
        )
    assert con.execute(
        "SELECT status FROM hitl_request WHERE id = 'legacy-hitl'"
    ).fetchone()["status"] == "PENDING"
    con.close()
