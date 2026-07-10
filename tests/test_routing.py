"""Tests for §3.1's route-or-pause-or-fail sequence: conditional edges, then
HITL, then is_default, then fail-fast. These specifically re-create the two
real bugs found while writing ARCHITECTURE.md/EXAMPLES.md, so a regression
here is exactly the class of mistake this suite exists to catch.
"""

from __future__ import annotations

import asyncio

import yaml

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import WorkflowDoc
from tests.conftest import SDLC_WORKFLOW


def _load_raw() -> dict:
    with open(SDLC_WORKFLOW) as f:
        return yaml.safe_load(f)


def test_qa_test_dead_end_is_caught_without_is_default(con):
    """Reproduces the exact bug found in ARCHITECTURE.md §4: with the
    is_default edge removed, qa_status == FAIL and iteration_count >= 5
    matches neither of qa_test's two conditional edges. Before the §3.1 fix,
    this silently stalled forever; now it must fail fast with a clear error
    naming the node — not hang, not raise an unrelated exception."""
    raw = _load_raw()
    raw["spec"]["graph"]["edges"] = [e for e in raw["spec"]["graph"]["edges"] if not e.get("is_default")]
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")

    fixture = {
        "pm_intake": [{"structured_payload": {"requirement_clarity": "HIGH", "milestone_plan": {}}}],
        "sa_design": [{"structured_payload": {"system_spec": {}}}],
        "dev_code": [{"structured_payload": {}}],
        # Always FAIL, forcing iteration_count to exceed the loop cap (5) and
        # hit the dead end once neither conditional edge matches anymore.
        "qa_test": [{"structured_payload": {"qa_status": "FAIL", "qa_report": {}}}] * 10,
    }
    runtime = MockAgentRuntime(fixture)

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, input_payload={"repository": "r"})
    )

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"

    fail_events = con.execute(
        "SELECT * FROM state_transition_log WHERE run_id = ? AND event_type = 'FAIL'", (run_id,)
    ).fetchall()
    assert len(fail_events) == 1
    assert fail_events[0]["from_node"] == "qa_test"

    failed_node_execution = con.execute(
        "SELECT * FROM node_execution WHERE run_id = ? AND node_id = 'qa_test' ORDER BY attempt DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    assert failed_node_execution["status"] == "FAILED"
    assert "no matching route" in failed_node_execution["error"]


def test_is_default_prevents_the_dead_end(con, sdlc_graph, sdlc_workflow_id):
    """The published example (with is_default intact) must NOT dead-end in
    the same scenario — it should route to pm_final_review instead."""
    fixture = {
        "pm_intake": [{"structured_payload": {"requirement_clarity": "HIGH", "milestone_plan": {}}}],
        "sa_design": [{"structured_payload": {"system_spec": {}}}],
        "dev_code": [{"structured_payload": {}}],
        "qa_test": [{"structured_payload": {"qa_status": "FAIL", "qa_report": {}}}] * 10,
        "pm_final_review": [{"structured_payload": {"pm_verdict": "COMPLETE"}}],
    }
    runtime = MockAgentRuntime(fixture)

    run_id = asyncio.run(
        start_run(con, sdlc_graph, runtime, org_id="test", workflow_id=sdlc_workflow_id, input_payload={"repository": "r"})
    )

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    # is_default routed qa_test -> pm_final_review, so the run reached the
    # terminal rather than hitting the routing dead-end this test guards
    # against. Its final status is FAILED (not COMPLETED) — but at the *DoD*
    # gate, not the router: this scenario pins qa_status='FAIL' to force the
    # loop-cap/default path, so definition_of_done's `state.qa_status == 'PASS'`
    # is legitimately unmet. That's the DoD gate working, and it's distinct from
    # a routing dead-end (which would carry a "no matching route" error).
    assert run["status"] == "FAILED"

    default_route = con.execute(
        "SELECT * FROM state_transition_log WHERE run_id = ? AND event_type = 'ROUTE' AND from_node = 'qa_test' AND to_node = 'pm_final_review'",
        (run_id,),
    ).fetchall()
    assert len(default_route) >= 1  # the routing point this test exists to prove

    dod_event = con.execute(
        "SELECT result FROM state_transition_log WHERE run_id = ? AND event_type = 'DOD_EVALUATED'", (run_id,)
    ).fetchone()
    assert dod_event is not None and dod_event["result"] == 0  # failed at the DoD gate, not a dead-end


def test_hitl_takes_priority_over_fail_fast(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    """pm_intake's only conditional edge requires requirement_clarity=='HIGH'.
    When the mock leaves it at the initial 'LOW', §3.1 must check HITL
    *before* concluding there's no matching route — this is the ordering
    fix from v0.14 (HITL must fire here, not a fail-fast dead end)."""
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"repository": "r"},
        )
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "WAITING_HUMAN"

    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    assert hitl is not None
    assert hitl["node_id"] == "pm_intake"
    assert hitl["assignee"] == "role:operator"

    fail_events = con.execute(
        "SELECT * FROM state_transition_log WHERE run_id = ? AND event_type = 'FAIL'", (run_id,)
    ).fetchall()
    assert not fail_events
