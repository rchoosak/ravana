"""Definition-of-Done evaluation (§3.1 step 7): the pure evaluator
(engine/dod.py) and its engine gating (a terminal only COMPLETEs if the DoD is
met, else FAILs). Expression criteria are enforced deterministically; prose
criteria are advisory (recorded, not gating) until an evaluator agent is wired.
"""

from __future__ import annotations

import asyncio

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.dod import evaluate_dod
from ravana.engine.loop import start_run
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import DefinitionOfDone, WorkflowDoc


def _dod(criteria: list[str]) -> DefinitionOfDone:
    return DefinitionOfDone(evaluated_by="a", criteria=criteria)


# --- pure evaluator --------------------------------------------------------
def test_expression_criteria_pass_and_fail():
    met = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    assert met.met and not met.unmet
    unmet = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {"qa_status": "FAIL"})
    assert not unmet.met
    assert unmet.unmet == ["state.qa_status == 'PASS'"]


def test_missing_state_key_expression_is_false_not_an_error():
    # StateProxy returns None for an unset key, so `None == 'PASS'` is a clean
    # False (criterion not yet met) — not a raise that would misclassify it.
    result = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {})
    assert not result.met
    assert result.results[0].kind == "expression"


def test_prose_criterion_is_advisory_when_no_evaluator():
    result = evaluate_dod(_dod(["All acceptance criteria are met"]), {})
    assert result.met  # unevaluated prose does not gate
    assert result.unevaluated == ["All acceptance criteria are met"]
    assert result.results[0].kind == "prose" and result.results[0].passed is None


def test_prose_criterion_enforced_when_verdict_wired():
    crit = "All acceptance criteria are met"
    passed = evaluate_dod(_dod([crit]), {}, prose_verdict=lambda who, cs, st: {crit: True})
    assert passed.met
    failed = evaluate_dod(_dod([crit]), {}, prose_verdict=lambda who, cs, st: {crit: False})
    assert not failed.met and failed.unmet == [crit]


def test_prose_verdict_receives_only_prose_criteria_and_evaluated_by():
    seen: dict = {}

    def verdict(who, criteria, state):
        seen["who"] = who
        seen["criteria"] = criteria
        return {c: True for c in criteria}

    evaluate_dod(
        DefinitionOfDone(evaluated_by="pm", criteria=["a prose line", "state.x == 1"]),
        {"x": 1},
        prose_verdict=verdict,
    )
    assert seen["who"] == "pm"
    assert seen["criteria"] == ["a prose line"]  # the expression criterion is NOT sent to the agent


def test_mixed_expression_and_prose_classified_correctly():
    prose = "No open defects in state.qa_report"  # references state. but isn't a valid expression
    result = evaluate_dod(_dod(["state.qa_status == 'PASS'", prose]), {"qa_status": "PASS"})
    kinds = {r.criterion: r.kind for r in result.results}
    assert kinds["state.qa_status == 'PASS'"] == "expression"
    assert kinds[prose] == "prose"
    assert result.met  # expression passed; prose advisory


def test_empty_criteria_is_vacuously_met():
    assert evaluate_dod(_dod([]), {}).met


def test_ordering_expression_on_unset_key_is_a_failing_expression_not_prose():
    # `state.count > 5` PARSES as an expression; on an unset key it errors at
    # eval (None > 5). It must be a FAILING expression (gating), NOT silently
    # downgraded to advisory prose — otherwise the gate is defeated.
    result = evaluate_dod(_dod(["state.count > 5"]), {})
    assert result.results[0].kind == "expression"
    assert result.results[0].passed is False
    assert not result.met and result.unmet == ["state.count > 5"]


def test_ordering_expression_met_when_key_present():
    result = evaluate_dod(_dod(["state.count > 5"]), {"count": 10})
    assert result.met and result.results[0].passed is True


def test_membership_expression_on_unset_key_is_a_failing_expression():
    # `'x' in state.items` parses; `'x' in None` raises -> failing expression.
    result = evaluate_dod(_dod(["'defect' in state.qa_report"]), {})
    assert result.results[0].kind == "expression"
    assert not result.met


# --- engine gating ---------------------------------------------------------
def _dod_workflow(criteria: list[str], with_dod: bool = True) -> WorkflowDoc:
    spec: dict = {
        "state": {"schema": {"qa_status": {"type": "string"}}, "initial": {}},
        "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
        "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
    }
    if with_dod:
        spec["definition_of_done"] = {"evaluated_by": "a", "criteria": criteria}
    return WorkflowDoc.model_validate(
        {"apiVersion": "ravana/v1", "kind": "Workflow", "metadata": {"name": "dod-test", "version": 1}, "spec": spec}
    )


def _run_single(con, workflow, payload):
    graph = compile_workflow(workflow)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": payload}]})
    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))
    return run_id


def _dod_event(con, run_id):
    return con.execute(
        "SELECT result, state_diff FROM state_transition_log WHERE run_id = ? AND event_type = 'DOD_EVALUATED'",
        (run_id,),
    ).fetchone()


def test_run_fails_at_terminal_when_dod_expression_unmet(con):
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "FAIL"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"  # reached terminal but DoD not met
    event = _dod_event(con, run_id)
    assert event["result"] == 0
    assert "state.qa_status == 'PASS'" in event["state_diff"]  # unmet criterion recorded


def test_run_completes_when_dod_expression_met(con):
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    assert _dod_event(con, run_id)["result"] == 1


def test_prose_criterion_does_not_block_completion_but_is_recorded(con):
    run_id = _run_single(
        con, _dod_workflow(["state.qa_status == 'PASS'", "All acceptance criteria are met"]), {"qa_status": "PASS"}
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"  # expression met; unevaluated prose is advisory
    assert "All acceptance criteria are met" in _dod_event(con, run_id)["state_diff"]  # recorded as unevaluated


def test_run_with_no_dod_completes_without_a_dod_event(con):
    run_id = _run_single(con, _dod_workflow([], with_dod=False), {"qa_status": "whatever"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    assert _dod_event(con, run_id) is None  # gate no-ops when there's no definition_of_done


def test_run_fails_when_ordering_expression_unmet_on_unset_key(con):
    # End-to-end version of the classification fix: an erroring expression must
    # gate the run (FAILED), not pass through as advisory prose (COMPLETED).
    run_id = _run_single(con, _dod_workflow(["state.count > 5"]), {"qa_status": "PASS"})
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "FAILED"


def _start_with_verdict(con, workflow, payload, verdict):
    graph = compile_workflow(workflow)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": payload}]})
    return asyncio.run(
        start_run(
            con, graph, runtime, org_id="test", workflow_id=workflow_id, dod_prose_verdict=verdict
        )
    )


def test_engine_enforces_prose_when_a_verdict_is_injected(con):
    # P2: with a prose_verdict wired at the engine boundary (start_run), a prose
    # criterion gates the run end-to-end — not only in the pure evaluator.
    crit = "All acceptance criteria are met"
    failed = _start_with_verdict(con, _dod_workflow([crit]), {"qa_status": "PASS"}, lambda who, cs, st: {c: False for c in cs})
    assert con.execute("SELECT status FROM run WHERE id = ?", (failed,)).fetchone()["status"] == "FAILED"

    passed = _start_with_verdict(con, _dod_workflow([crit]), {"qa_status": "PASS"}, lambda who, cs, st: {c: True for c in cs})
    assert con.execute("SELECT status FROM run WHERE id = ?", (passed,)).fetchone()["status"] == "COMPLETED"


def test_engine_prose_verdict_receives_evaluated_by_and_final_state(con):
    seen: dict = {}

    def verdict(evaluated_by, criteria, state):
        seen["evaluated_by"] = evaluated_by
        seen["state"] = dict(state)
        return {c: True for c in criteria}

    _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, verdict)
    assert seen["evaluated_by"] == "a"  # the DoD's evaluated_by agent id
    assert seen["state"]["qa_status"] == "PASS"  # final shared_state, post-commit
