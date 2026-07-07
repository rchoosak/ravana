"""Tests for §3.8's join primitive — the fix for the design gap found during
Phase 0a implementation: a node that is both a broadcast target and a
loop-reconvergence target (qa_test in the §4 example) was dispatched once
per arriving edge, burning iteration_count twice as fast as intended and
being correct only by accident.

Semantics under test:
- join: all holds dispatch until every inbound source has delivered (first wave)
- loop re-entries with only a subset of sources fire at quiescence
- arrivals are derived from state_transition_log, so they survive a HITL
  pause/resume (which crosses engine invocations)
- a pending HITL defers quiescence (answering it may deliver missing arrivals)
"""

from __future__ import annotations

import asyncio

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.compiler.validate import validate
from ravana.engine.loop import resume_hitl, start_run
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import WorkflowDoc
from ravana.schema.util import loads


def _run_sdlc_to_completion(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime) -> str:
    run_id = asyncio.run(
        start_run(
            con, sdlc_graph, sdlc_runtime, org_id="test", workflow_id=sdlc_workflow_id,
            input_payload={"requirement": "build X", "repository": "org/repo"},
        )
    )
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, sdlc_graph, sdlc_runtime, run_id, hitl["id"], {"answer": "ok"}))
    return run_id


def test_join_eliminates_double_dispatch(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    """qa_test must be dispatched exactly once per logical QA cycle (2 total:
    first wave + one bugfix loop), not once per arriving edge (which was 4)."""
    run_id = _run_sdlc_to_completion(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime)

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"

    qa_dispatches = con.execute(
        "SELECT COUNT(*) c FROM node_execution WHERE run_id = ? AND node_id = 'qa_test'", (run_id,)
    ).fetchone()["c"]
    assert qa_dispatches == 2

    # iteration_count is the designed-correctness check: on_enter increments
    # once per dispatch, so with the join it must equal the number of real
    # QA cycles, not double it.
    state = loads(run["shared_state"])
    assert state["iteration_count"] == 2


def test_join_waits_for_all_sources_in_first_wave(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    """qa_test's first dispatch must come after BOTH sa_design's broadcast
    arrival and dev_code's arrival — verified by log sequence ordering."""
    run_id = _run_sdlc_to_completion(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime)

    first_qa_commit = con.execute(
        """SELECT MIN(sequence) s FROM state_transition_log
           WHERE run_id = ? AND event_type = 'COMMIT' AND from_node = 'qa_test'""",
        (run_id,),
    ).fetchone()["s"]
    arrivals_before_first_dispatch = con.execute(
        """SELECT DISTINCT from_node FROM state_transition_log
           WHERE run_id = ? AND event_type = 'ROUTE' AND to_node = 'qa_test' AND sequence < ?""",
        (run_id, first_qa_commit),
    ).fetchall()
    assert {r["from_node"] for r in arrivals_before_first_dispatch} == {"sa_design", "dev_code"}


def test_loop_reentry_fires_at_quiescence_with_partial_arrivals(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime):
    """The bugfix loop's second QA cycle only receives dev_code's arrival
    (sa_design doesn't re-fire) — the join must still dispatch instead of
    deadlocking waiting for sa_design forever."""
    run_id = _run_sdlc_to_completion(con, sdlc_graph, sdlc_workflow_id, sdlc_runtime)

    # Reaching COMPLETED at all proves no deadlock; additionally the second
    # QA dispatch's arrival window must contain only dev_code.
    commits = con.execute(
        """SELECT sequence FROM state_transition_log
           WHERE run_id = ? AND event_type = 'COMMIT' AND from_node = 'qa_test' ORDER BY sequence""",
        (run_id,),
    ).fetchall()
    assert len(commits) >= 2
    second_wave_start = commits[0]["sequence"]
    second_commit_seq = commits[-1]["sequence"]
    arrivals_between = con.execute(
        """SELECT DISTINCT from_node FROM state_transition_log
           WHERE run_id = ? AND event_type = 'ROUTE' AND to_node = 'qa_test'
             AND sequence > ? AND sequence < ?""",
        (run_id, second_wave_start, second_commit_seq),
    ).fetchall()
    assert {r["from_node"] for r in arrivals_between} == {"dev_code"}


def test_pending_hitl_defers_quiescence_flush(con):
    """A join must NOT fire on partial arrivals while a HITL pause is
    outstanding on another branch — answering it may deliver the missing
    arrival. Graph: entry broadcasts to [a, b]; a -> j, b -> j (j join: all);
    b pauses on HITL before delivering."""
    doc = WorkflowDoc.model_validate(
        {
            "apiVersion": "ravana/v1",
            "kind": "Workflow",
            "metadata": {"name": "join-hitl-test", "version": 1},
            "spec": {
                "state": {
                    "schema": {"b_ready": {"type": "boolean", "merge": "overwrite"}},
                    "initial": {"b_ready": False},
                },
                "agents": [
                    {"id": "plain", "name": "P", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"},
                    {
                        "id": "gated", "name": "G", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p",
                        "hitl": {"enabled": True, "trigger_condition": "!state.b_ready"},
                    },
                ],
                "graph": {
                    "entry": "start",
                    "nodes": [
                        {"id": "start", "agent": "plain"},
                        {"id": "a", "agent": "plain"},
                        {"id": "b", "agent": "gated"},
                        {"id": "j", "agent": "plain", "join": "all"},
                    ],
                    "edges": [
                        {"from": "start", "to": ["a", "b"]},
                        {"from": "a", "to": ["j"]},
                        {"from": "b", "to": ["j"], "condition": "state.b_ready"},
                        {"from": "j", "to": ["__terminal__"]},
                    ],
                },
            },
        }
    )
    graph = compile_workflow(doc)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime(
        {
            "start": [{"structured_payload": {}}],
            "a": [{"structured_payload": {}}],
            "b": [{"structured_payload": {}}, {"structured_payload": {"b_ready": True}}],
            "j": [{"structured_payload": {}}],
        }
    )

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))

    # b paused on HITL; a already delivered its arrival to j. j must NOT have
    # dispatched yet — quiescence is deferred while HITL is pending.
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "WAITING_HUMAN"
    j_dispatches = con.execute(
        "SELECT COUNT(*) c FROM node_execution WHERE run_id = ? AND node_id = 'j'", (run_id,)
    ).fetchone()["c"]
    assert j_dispatches == 0

    # Answering the HITL re-runs b, which now delivers — j fires exactly once
    # with BOTH arrivals, and the run completes.
    hitl = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchone()
    asyncio.run(resume_hitl(con, graph, runtime, run_id, hitl["id"], {"answer": "go"}))

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    j_dispatches = con.execute(
        "SELECT COUNT(*) c FROM node_execution WHERE run_id = ? AND node_id = 'j'", (run_id,)
    ).fetchone()["c"]
    assert j_dispatches == 1


def test_validator_warns_on_pointless_join(sdlc_graph):
    """join: all with <2 inbound sources, or on the entry node, gets a warning."""
    import yaml

    from tests.conftest import SDLC_WORKFLOW

    with open(SDLC_WORKFLOW) as f:
        raw = yaml.safe_load(f)
    for node in raw["spec"]["graph"]["nodes"]:
        if node["id"] == "pm_intake":  # entry node, zero inbound sources
            node["join"] = "all"
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    issues = validate(graph)
    messages = [i.message for i in issues if i.severity == "warning"]
    assert any("entry node 'pm_intake'" in m for m in messages)
    assert any("pm_intake" in m and "inbound source" in m for m in messages)

    # And the published example itself (qa_test join: all, 2 sources) is clean.
    assert validate(sdlc_graph) == []
