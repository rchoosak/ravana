"""Tests for the v0.14 fix: a HITL response dispatches a *new* node_execution
attempt for the paused node (the agent actually re-runs with the human's
answer in context), not a bare re-route of the stale first-turn output."""

from __future__ import annotations

import asyncio

from ravana.engine.loop import resume_hitl, start_run
from ravana.schema.util import loads


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
