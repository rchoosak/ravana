"""End-to-end: the full software-development-team.yaml example (§4) driven
by mocked agents, from pm_intake to COMPLETED, including one HITL round-trip
and at least one bugfix-loop iteration."""

from __future__ import annotations

import asyncio

from ravana.engine.loop import resume_hitl, start_run
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.util import loads
from tests.conftest import RecordingSleep


def test_full_sdlc_run_reaches_completed(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            triggered_by="test", input_payload={"requirement": "build X", "repository": "org/repo"},
        )
    )

    # First turn: PM sets clarity=LOW, which the corrected §3.1 ordering
    # must route to HITL rather than fail-fast (also covered directly by
    # test_routing.py, asserted again here as part of the full flow).
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "WAITING_HUMAN"

    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "unambiguous now"}))

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    state = loads(run["shared_state"])
    assert state["requirement_clarity"] == "HIGH"
    assert state["qa_status"] == "PASS"
    assert state["pm_verdict"] == "COMPLETE"

    # At least one real bugfix-loop iteration happened (qa_test found FAIL
    # at least once before the eventual PASS) — proves the loop mechanism,
    # not just a straight-through pass.
    fail_then_pass = con.execute(
        "SELECT COUNT(*) c FROM state_transition_log WHERE run_id = ? AND event_type = 'ROUTE' AND from_node = 'qa_test' AND to_node = 'dev_code'",
        (run_id,),
    ).fetchone()["c"]
    assert fail_then_pass >= 1

    # The event log has everything §3.5's replay claim depends on.
    events = con.execute(
        "SELECT sequence, state_version_before, state_version_after FROM state_transition_log WHERE run_id = ? ORDER BY sequence",
        (run_id,),
    ).fetchall()
    sequences = [e["sequence"] for e in events]
    assert sequences == sorted(sequences)
    assert sequences == list(range(1, len(sequences) + 1))  # gapless, matching §3.1's "atomic with the CAS" claim


def test_broadcast_merge_policy_keeps_both_writers_state(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    """sa_design broadcasts to [dev_code, qa_test] in parallel (§3.5). Both
    eventually write to shared_state; qa_report uses merge-object specifically
    so a later qa_test dispatch's report doesn't just clobber an unrelated
    key some other concurrent writer touched."""
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "org/repo"},
        )
    )
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "ok"}))

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    state = loads(run["shared_state"])
    # system_spec (from sa_design) and qa_report (from qa_test) both survive
    # to the end — neither broadcast branch's write was lost.
    assert state["system_spec"] == {"stack": "python"}
    assert "qa_report" in state


def test_max_total_steps_guard_fails_a_runaway_run(con, sdlc_graph, sdlc_workflow_id):
    """A graph that never lets qa_status settle should hit guards.max_total_steps
    and fail loudly rather than looping forever."""
    fixture = {
        "pm_intake": [{"structured_payload": {"requirement_clarity": "HIGH", "milestone_plan": {}}}],
        "sa_design": [{"structured_payload": {"system_spec": {}}}],
        "dev_code": [{"structured_payload": {}}],
        "qa_test": [{"structured_payload": {"qa_status": "FAIL", "qa_report": {}}}] * 50,
        # Reached via is_default once max_loop_iterations caps the bugfix
        # loop — and then PM keeps sending it back INCOMPLETE, so the run
        # still never settles and must hit max_total_steps instead.
        "pm_final_review": [{"structured_payload": {"pm_verdict": "INCOMPLETE"}}] * 50,
    }
    runtime = MockAgentRuntime(fixture)
    run_id = asyncio.run(
        start_run(con, sdlc_graph, runtime, org_id="test", workflow_id=sdlc_workflow_id, input_payload={"repository": "r"})
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"

    total_dispatches = con.execute("SELECT COUNT(*) c FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()["c"]
    assert total_dispatches <= sdlc_graph.doc.spec.graph.guards.max_total_steps + 1


def test_transient_failure_retries_then_succeeds(con, sdlc_graph, sdlc_workflow_id):
    """§3.6: a transient failure gets retried (new node_execution attempt,
    with exponential backoff before the retry) up to max_retries_per_node,
    rather than failing the run on the first hiccup."""
    fixture = {
        "pm_intake": [
            {"transient_error": True},
            {"structured_payload": {"requirement_clarity": "HIGH", "milestone_plan": {}}},
        ],
        "sa_design": [{"structured_payload": {"system_spec": {}}}],
        "dev_code": [{"structured_payload": {}}],
        "qa_test": [{"structured_payload": {"qa_status": "PASS", "qa_report": {}}}],
        "pm_final_review": [{"structured_payload": {"pm_verdict": "COMPLETE"}}],
    }
    runtime = MockAgentRuntime(fixture)
    sleeper = RecordingSleep()

    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"}, retry_sleep=sleeper,
        )
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"

    pm_attempts = con.execute(
        "SELECT attempt, status FROM node_execution WHERE run_id = ? AND node_id = 'pm_intake' ORDER BY attempt",
        (run_id,),
    ).fetchall()
    assert pm_attempts[0]["status"] == "FAILED"
    assert pm_attempts[1]["status"] == "SUCCEEDED"
    # §3.6: the retry backed off (one failure => one delay, equal jitter around
    # base=1s: 0.5 <= d <= 1.0), instead of re-dispatching immediately.
    assert len(sleeper.delays) == 1 and 0.5 <= sleeper.delays[0] <= 1.0
