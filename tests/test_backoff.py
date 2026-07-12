"""Exponential backoff (§3.6): the pure delay function, and the engine's
per-node retry actually growing its delays across consecutive transient
failures. (The gateway's per-entry backoff and the single-retry engine path
are asserted in test_gateway.py / test_e2e_sdlc.py alongside the behaviors
they modify.)"""

from __future__ import annotations

import asyncio

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.backoff import backoff_delay
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import WorkflowDoc
from tests.conftest import RecordingSleep


def _mid(a: float, b: float) -> float:
    """Rng stub pinned to the interval midpoint, for exact delay assertions."""
    return (a + b) / 2


def test_delay_doubles_per_consecutive_failure():
    # With rng pinned to the midpoint, equal jitter yields exactly 3/4 of the
    # exponential: failure 1 -> 0.75*base, 2 -> 1.5*base, 3 -> 3*base.
    assert backoff_delay(1, base=1.0, cap=30.0, rng=_mid) == pytest.approx(0.75)
    assert backoff_delay(2, base=1.0, cap=30.0, rng=_mid) == pytest.approx(1.5)
    assert backoff_delay(3, base=1.0, cap=30.0, rng=_mid) == pytest.approx(3.0)


def test_delay_is_capped():
    # Failure 10 of base=1 would be 512s uncapped; the cap bounds it.
    assert backoff_delay(10, base=1.0, cap=30.0, rng=_mid) == pytest.approx(22.5)  # 0.75 * cap


def test_jitter_stays_within_equal_jitter_bounds():
    # Real rng: delay must live in [exp/2, exp] — never below half (equal
    # jitter's deterministic floor), never above the full exponential.
    for failure_number in (1, 2, 3, 4):
        exp = min(30.0, 1.0 * 2 ** (failure_number - 1))
        for _ in range(50):
            d = backoff_delay(failure_number, base=1.0, cap=30.0)
            assert exp / 2 <= d <= exp


def test_failure_number_is_one_indexed():
    with pytest.raises(ValueError, match="1-indexed"):
        backoff_delay(0, base=1.0, cap=30.0)


def _single_node_doc(name: str, guards: dict | None = None) -> WorkflowDoc:
    """One agent, one node, no edges — the minimal engine-retry testbed.
    `guards` overrides (e.g. {"max_retries_per_node": 1}) apply to the graph."""
    graph: dict = {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []}
    if guards:
        graph["guards"] = guards
    return WorkflowDoc.model_validate(
        {
            "apiVersion": "ravana/v1",
            "kind": "Workflow",
            "metadata": {"name": name, "version": 1},
            "spec": {
                "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
                "graph": graph,
            },
        }
    )


def test_engine_delays_grow_exponentially_across_retries(con):
    # Two consecutive transient failures on one node: the recorded backoffs
    # must double (attempt 1 ~base, attempt 2 ~2*base), per §3.6.
    graph = compile_workflow(_single_node_doc("backoff-test"))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime(
        {"only": [{"transient_error": True}, {"transient_error": True}, {"structured_payload": {}}]}
    )
    sleeper = RecordingSleep()

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, retry_sleep=sleeper)
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"
    # 1st consecutive failure: exp=1s; 2nd: exp=2s — grew, not flat.
    sleeper.assert_delays(1.0, 2.0)
    visits = {
        row["logical_visit_id"]
        for row in con.execute(
            "SELECT logical_visit_id FROM node_execution WHERE run_id = ?", (run_id,)
        )
    }
    assert len(visits) == 1  # all retry attempts belong to one logical visit


def _cyclic_two_node_doc() -> WorkflowDoc:
    # a -> b, and b loops back to a while state.done != true; a's 3rd visit
    # sets done. Gives node 'a' SUCCEEDED attempts before a later failure.
    return WorkflowDoc.model_validate(
        {
            "apiVersion": "ravana/v1",
            "kind": "Workflow",
            "metadata": {"name": "backoff-loop-test", "version": 1},
            "spec": {
                "state": {"schema": {"done": {"type": "boolean"}}, "initial": {}},
                "agents": [
                    {"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"},
                    {"id": "b", "name": "B", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"},
                ],
                "graph": {
                    "entry": "worker",
                    "nodes": [{"id": "worker", "agent": "a"}, {"id": "checker", "agent": "b"}],
                    "edges": [
                        {"from": "worker", "to": ["checker"]},
                        {"from": "checker", "to": ["worker"], "condition": "state.done != true"},
                        {"from": "checker", "to": ["__terminal__"], "is_default": True},
                    ],
                },
            },
        }
    )


def test_backoff_keys_on_consecutive_failures_not_lifetime_attempts(con):
    # A node re-entered by a §3.7 loop accumulates SUCCEEDED attempts. Its
    # FIRST transient failure on a later visit must back off ~base (streak=1),
    # not the inflated lifetime attempt number (which would jump toward the
    # cap). Regression for the review finding.
    graph = compile_workflow(_cyclic_two_node_doc())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime(
        {
            # worker: succeeds twice (visits 1-2), transient-fails once on
            # visit 3 (lifetime attempt 3), then succeeds and finishes the loop.
            "worker": [
                {"structured_payload": {}},
                {"structured_payload": {}},
                {"transient_error": True},
                {"structured_payload": {"done": True}},
            ],
            "checker": [{"structured_payload": {}}] * 3,
        }
    )
    sleeper = RecordingSleep()
    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, retry_sleep=sleeper)
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"
    # One failure => one backoff, and it's a FIRST-failure delay (exp=1s),
    # NOT backoff_delay(3) (exp=4s) off the lifetime attempt.
    sleeper.assert_delays(1.0)
    worker_visits = [
        row["logical_visit_id"]
        for row in con.execute(
            """SELECT logical_visit_id FROM node_execution
               WHERE run_id = ? AND node_id = 'worker' ORDER BY attempt""",
            (run_id,),
        )
    ]
    assert len(set(worker_visits[:2])) == 2  # two successful graph entries
    assert worker_visits[2] == worker_visits[3]  # retry keeps visit identity


def test_no_sleep_before_a_guaranteed_budget_failure(con):
    # When the failure that just happened already exhausts max_retries_per_node,
    # the re-queued dispatch will fail the run without running a turn — so the
    # engine must NOT spend a backoff sleep first. Regression for the review
    # finding (the final sleep bought nothing).
    graph = compile_workflow(_single_node_doc("backoff-exhaust-test", guards={"max_retries_per_node": 1}))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"transient_error": True}] * 5})
    sleeper = RecordingSleep()
    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, retry_sleep=sleeper)
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "FAILED"
    # max_retries_per_node=1: failure #1 backs off (a retry follows); failure #2
    # exhausts the budget => NO second sleep before the guard fails the run.
    assert len(sleeper.delays) == 1
