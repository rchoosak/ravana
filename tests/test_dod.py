"""Definition-of-Done evaluation (§3.1 step 7): the pure evaluator
(engine/dod.py, now synchronous) and its engine gating (a terminal only
COMPLETEs if the DoD is met, else FAILs). Expression criteria are enforced
deterministically; prose criteria are resolved at the engine boundary by an
injected async verdict returning a *position-aligned* ProseJudgement, whose
token usage the gate meters against guards.max_tokens_total.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.dod import evaluate_dod
from ravana.engine.loop import TerminalPersistenceError, start_run
from ravana.runtime.base import LLMUsage, ProseJudgement, ProseJudgementError
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import DefinitionOfDone, WorkflowDoc

# A test double maps (evaluated_by, prose_criteria, state) -> position-aligned bools.
VerdictFn = Callable[[str, list[str], dict[str, Any]], list[bool]]


def _dod(criteria: list[str]) -> DefinitionOfDone:
    return DefinitionOfDone(evaluated_by="a", criteria=criteria)


def _averdict(fn: VerdictFn):
    """Adapt a plain (sync) verdict function into the async ProseVerdict the
    engine awaits — returning a ProseJudgement. A raising `fn` still raises when
    awaited (the throwing-verdict test depends on that)."""

    async def verdict(who: str, criteria: list[str], state: dict[str, Any]) -> ProseJudgement:
        return ProseJudgement(verdicts=fn(who, criteria, state))

    return verdict


# --- pure evaluator (sync) -------------------------------------------------
def test_expression_criteria_pass_and_fail():
    met = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    assert met.met and not met.unmet
    unmet = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {"qa_status": "FAIL"})
    assert not unmet.met
    assert unmet.unmet == ["state.qa_status == 'PASS'"]


def test_missing_state_key_expression_is_false_not_an_error():
    result = evaluate_dod(_dod(["state.qa_status == 'PASS'"]), {})
    assert not result.met
    assert result.results[0].kind == "expression"


def test_prose_criterion_is_advisory_when_no_verdict_applied():
    result = evaluate_dod(_dod(["All acceptance criteria are met"]), {})
    assert result.met  # unevaluated prose does not gate
    assert result.unevaluated == ["All acceptance criteria are met"]
    assert result.results[0].kind == "prose" and result.results[0].passed is None


def test_prose_criteria_property_excludes_expressions():
    result = evaluate_dod(_dod(["a prose line", "state.x == 1"]), {"x": 1})
    assert result.prose_criteria == ["a prose line"]  # only what a verdict is asked to judge


def test_apply_prose_verdict_gates_fail_closed():
    crit = "All acceptance criteria are met"
    passed = evaluate_dod(_dod([crit]), {})
    passed.apply_prose_verdict([True])
    assert passed.met
    failed = evaluate_dod(_dod([crit]), {})
    failed.apply_prose_verdict([False])
    assert not failed.met and failed.unmet == [crit]


def test_apply_prose_verdict_short_list_is_fail_closed():
    # A verdict missing an entry for a criterion leaves it not met (never passes).
    result = evaluate_dod(_dod(["prose zero", "prose one"]), {})
    result.apply_prose_verdict([True])  # nothing for the second
    assert not result.met and result.unmet == ["prose one"]


def test_apply_prose_verdict_only_exact_true_passes():
    # apply is `is True` — a truthy non-True entry does not pass.
    result = evaluate_dod(_dod(["a prose criterion"]), {})
    result.apply_prose_verdict([1])  # type: ignore[list-item]  # deliberately not a bool
    assert not result.met


def test_apply_prose_verdict_preserves_position_for_duplicate_text():
    # Two criteria with identical text keep independent rulings — nothing is
    # keyed by (collidable) criterion text.
    crit = "same text"
    result = evaluate_dod(_dod([crit, crit]), {})
    result.apply_prose_verdict([True, False])
    assert not result.met  # the second one is not met -> gate fails
    assert result.unmet == [crit]


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
    result = evaluate_dod(_dod(["state.count > 5"]), {})
    assert result.results[0].kind == "expression"
    assert result.results[0].passed is False
    assert not result.met and result.unmet == ["state.count > 5"]


def test_ordering_expression_met_when_key_present():
    result = evaluate_dod(_dod(["state.count > 5"]), {"count": 10})
    assert result.met and result.results[0].passed is True


def test_membership_expression_on_unset_key_is_a_failing_expression():
    result = evaluate_dod(_dod(["'defect' in state.qa_report"]), {})
    assert result.results[0].kind == "expression"
    assert not result.met


# --- engine gating ---------------------------------------------------------
def _dod_workflow(criteria: list[str], with_dod: bool = True, guards: dict | None = None) -> WorkflowDoc:
    graph: dict[str, Any] = {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []}
    if guards is not None:
        graph["guards"] = guards
    spec: dict = {
        "state": {"schema": {"qa_status": {"type": "string"}}, "initial": {}},
        "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
        "graph": graph,
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
    return asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))


def _dod_event(con, run_id):
    return con.execute(
        "SELECT result, state_diff FROM state_transition_log WHERE run_id = ? AND event_type = 'DOD_EVALUATED'",
        (run_id,),
    ).fetchone()


def test_run_fails_at_terminal_when_dod_expression_unmet(con):
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "FAIL"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    event = _dod_event(con, run_id)
    assert event["result"] == 0
    assert "state.qa_status == 'PASS'" in event["state_diff"]  # unmet criterion recorded
    assert json.loads(event["state_diff"])["outcome"] == "criteria_unmet"


def test_run_completes_when_dod_expression_met(con):
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    event = _dod_event(con, run_id)
    assert event["result"] == 1 and json.loads(event["state_diff"])["outcome"] == "met"


def test_prose_criterion_does_not_block_completion_but_is_recorded(con):
    run_id = _run_single(
        con, _dod_workflow(["state.qa_status == 'PASS'", "All acceptance criteria are met"]), {"qa_status": "PASS"}
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"  # expression met; unevaluated prose is advisory (mock has no judge)
    assert "All acceptance criteria are met" in _dod_event(con, run_id)["state_diff"]


def test_run_with_no_dod_completes_without_a_dod_event(con):
    run_id = _run_single(con, _dod_workflow([], with_dod=False), {"qa_status": "whatever"})
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    assert _dod_event(con, run_id) is None


def test_run_fails_when_ordering_expression_unmet_on_unset_key(con):
    run_id = _run_single(con, _dod_workflow(["state.count > 5"]), {"qa_status": "PASS"})
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "FAILED"


def _start_with_verdict(con, workflow, payload, verdict, *, raw=None):
    graph = compile_workflow(workflow)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": payload}]})
    return asyncio.run(
        start_run(
            con, graph, runtime, org_id="test", workflow_id=workflow_id,
            dod_prose_verdict=raw if raw is not None else _averdict(verdict),
        )
    )


def test_engine_enforces_prose_when_a_verdict_is_injected(con):
    crit = "All acceptance criteria are met"
    failed = _start_with_verdict(con, _dod_workflow([crit]), {"qa_status": "PASS"}, lambda who, cs, st: [False for _ in cs])
    assert con.execute("SELECT status FROM run WHERE id = ?", (failed,)).fetchone()["status"] == "FAILED"

    passed = _start_with_verdict(con, _dod_workflow([crit]), {"qa_status": "PASS"}, lambda who, cs, st: [True for _ in cs])
    assert con.execute("SELECT status FROM run WHERE id = ?", (passed,)).fetchone()["status"] == "COMPLETED"


def test_throwing_prose_verdict_fails_the_run_not_strands_it(con):
    async def boom(evaluated_by, criteria, state):
        raise RuntimeError("evaluator exploded")

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=boom)
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"  # not stranded at RUNNING
    assert run["ended_at"] is not None
    event = _dod_event(con, run_id)
    assert event["result"] == 0
    # Durable cause: an evaluator failure is distinguishable from an unmet criterion.
    assert json.loads(event["state_diff"])["outcome"] == "evaluator_error"


def test_engine_prose_verdict_receives_evaluated_by_and_final_state(con):
    seen: dict = {}

    async def verdict(evaluated_by, criteria, state):
        seen["evaluated_by"] = evaluated_by
        seen["state"] = dict(state)
        return ProseJudgement(verdicts=[True for _ in criteria])

    _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    assert seen["evaluated_by"] == "a"
    assert seen["state"]["qa_status"] == "PASS"


def test_dod_event_records_judgement_usage(con):
    async def verdict(evaluated_by, criteria, state):
        return ProseJudgement(verdicts=[True for _ in criteria], usage=LLMUsage(input_tokens=40, output_tokens=12))

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    usage = json.loads(_dod_event(con, run_id)["state_diff"])["usage"]
    assert usage == {"input_tokens": 40, "output_tokens": 12}


class _BrokenStderr:
    def write(self, *a, **k):
        raise ValueError("stderr is closed")

    def flush(self, *a, **k):
        pass


def test_log_event_is_best_effort_when_stderr_is_broken(monkeypatch):
    import sys

    from ravana.observability.logging import log_event

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    log_event("ERROR", "boom", run_id="r1")  # must NOT raise


def test_log_event_fallback_redacts_message_and_keeps_correlation(monkeypatch):
    # When an unserializable `extra` forces the fallback path, the message must
    # STILL be redacted (a secret must never reach stderr just because an extra
    # field couldn't serialize) and BOTH correlation keys must survive.
    import io
    import sys

    from ravana.observability.logging import log_event

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    log_event("ERROR", "leak sk-DO-NOT-LOG-abc123def456ghi", run_id="r1", node_execution_id="n1", bad=object())
    out = buf.getvalue()
    assert "sk-DO-NOT-LOG" not in out and "***REDACTED***" in out  # redacted, not raw
    assert "r1" in out and "n1" in out  # both correlation keys preserved


def test_logging_failure_does_not_strand_the_run(con, monkeypatch):
    # A broken stderr must not strand a run: the DoD gate logs an ERROR line on
    # the FAILED path, but log_event is best-effort, so the run still commits
    # FAILED with ended_at set rather than escaping past the commit as RUNNING.
    import sys

    monkeypatch.setattr(sys, "stderr", _BrokenStderr())
    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, lambda who, cs, st: [False for _ in cs]
    )
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None


def test_dod_event_write_failure_does_not_strand_run(con):
    # If persisting the DOD_EVALUATED itself fails (here: a trigger rejects the
    # insert), the run must STILL resolve — status-only FAILED with ended_at set
    # — not strand at RUNNING with start_run raising. This is why the DoD
    # decision is pure and separate from the (single, fallback-guarded) write.
    con.execute(
        "CREATE TRIGGER reject_dod BEFORE INSERT ON state_transition_log "
        "WHEN NEW.event_type = 'DOD_EVALUATED' "
        "BEGIN SELECT RAISE(ABORT, 'no dod events'); END"
    )
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"  # resolved status-only, not stranded at RUNNING
    assert run["ended_at"] is not None
    assert _dod_event(con, run_id) is None  # the event write was rejected; the run resolved anyway


def test_terminal_status_write_failure_is_observable(con):
    # If even the status-only fallback is unwritable, persistence cannot truthfully
    # claim the run resolved. Surface a typed error so a supervisor can recover it.
    con.execute(
        "CREATE TRIGGER reject_terminal_status BEFORE UPDATE OF status ON run "
        "WHEN OLD.status = 'RUNNING' AND NEW.status IN ('COMPLETED', 'FAILED') "
        "BEGIN SELECT RAISE(ABORT, 'no terminal statuses'); END"
    )

    with pytest.raises(TerminalPersistenceError):
        _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})

    run = con.execute("SELECT id, status, ended_at FROM run ORDER BY started_at DESC LIMIT 1").fetchone()
    assert run["status"] == "RUNNING" and run["ended_at"] is None
    assert _dod_event(con, run["id"]) is None


def test_cancel_during_dod_judgement_is_not_overwritten(con):
    # The judge is an await point. A CANCELLED committed while it runs is terminal
    # and must win over the stale completion decision, including its staged event.
    async def cancel_then_pass(evaluated_by, criteria, state):
        run = con.execute("SELECT id FROM run WHERE status = 'RUNNING' ORDER BY started_at DESC LIMIT 1").fetchone()
        con.execute("UPDATE run SET status = 'CANCELLED', ended_at = 'cancelled' WHERE id = ?", (run["id"],))
        con.commit()
        return ProseJudgement(verdicts=[True for _ in criteria], usage=LLMUsage(1, 1))

    run_id = _start_with_verdict(
        con,
        _dod_workflow(["All acceptance criteria are met"]),
        {"qa_status": "PASS"},
        None,
        raw=cancel_then_pass,
    )

    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "CANCELLED" and run["ended_at"] == "cancelled"
    assert _dod_event(con, run_id) is None


def test_unforeseen_gate_error_still_stages_event(con, monkeypatch):
    # An UNFORESEEN error inside the DoD decision (here: evaluate_dod raises)
    # must still produce a FAILED run WITH a DOD_EVALUATED — the decision is
    # total (fails closed to evaluator_error) and _write_final persists it.
    import ravana.engine.loop as loop_mod

    def boom(*a, **k):
        raise RuntimeError("unexpected gate bug")

    monkeypatch.setattr(loop_mod, "evaluate_dod", boom)
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    count = con.execute(
        "SELECT COUNT(*) AS c FROM state_transition_log WHERE run_id = ? AND event_type = 'DOD_EVALUATED'", (run_id,)
    ).fetchone()["c"]
    assert count == 1  # exactly one event, not zero (contract) and not double
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "evaluator_error"


def test_event_builder_failure_uses_minimal_fail_closed_event(con, monkeypatch):
    # The total-decision fallback must not call the same builder that just failed.
    import ravana.engine.loop as loop_mod

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected builder bug")

    monkeypatch.setattr(loop_mod, "_build_dod_event", boom)
    run_id = _run_single(con, _dod_workflow(["state.qa_status == 'PASS'"]), {"qa_status": "PASS"})

    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    event = _dod_event(con, run_id)
    payload = json.loads(event["state_diff"])
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    assert event["result"] == 0
    assert payload["outcome"] == "evaluator_error" and payload["detail"] == "RuntimeError"


def test_dod_finalization_is_idempotent_on_reentry(con):
    # A re-entry after a run is already terminal must NOT re-judge, must not
    # append a second DOD_EVALUATED, and must not flip the terminal status.
    from ravana.engine.loop import _RunCtx, _finalize_status

    calls = {"n": 0}

    async def verdict(evaluated_by, criteria, state):
        calls["n"] += 1
        # Flip the verdict on the 2nd call, so a non-idempotent re-judge would
        # visibly change COMPLETED -> FAILED.
        return ProseJudgement(verdicts=[calls["n"] == 1 for _ in criteria])

    graph = compile_workflow(_dod_workflow(["a prose criterion"]))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {"qa_status": "PASS"}}]})
    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, dod_prose_verdict=verdict)
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"
    assert calls["n"] == 1

    # Re-enter finalization on the already-terminal run.
    ctx = _RunCtx(
        con=con, graph=graph, run_id=run_id, org_id="test", workflow_id=workflow_id,
        runtime=runtime, terminal_reached=True, dod_prose_verdict=verdict,
    )
    asyncio.run(_finalize_status(ctx))

    assert calls["n"] == 1  # judge NOT called again
    n_events = con.execute(
        "SELECT COUNT(*) AS c FROM state_transition_log WHERE run_id = ? AND event_type = 'DOD_EVALUATED'", (run_id,)
    ).fetchone()["c"]
    assert n_events == 1  # no second event
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"  # not flipped


def test_dod_judgement_usage_is_metered_against_max_tokens_total(con):
    # A judgement whose tokens push the run over guards.max_tokens_total FAILs
    # the run (cost cap), even though every criterion was judged met.
    async def verdict(evaluated_by, criteria, state):
        return ProseJudgement(verdicts=[True for _ in criteria], usage=LLMUsage(input_tokens=100, output_tokens=100))

    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"], guards={"max_tokens_total": 10}),
        {"qa_status": "PASS"}, None, raw=verdict,
    )
    run = con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "cost_cap_exceeded"


def test_malformed_judge_returning_none_fails_closed_not_stranded(con):
    # A runtime that returns None (not a ProseJudgement) must FAIL the run with a
    # durable cause — never throw past the gate and strand it at RUNNING.
    async def verdict(evaluated_by, criteria, state):
        return None

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "evaluator_error"


def test_negative_usage_verdict_fails_closed(con):
    # A judge reporting negative tokens (to duck the cost cap) can't even
    # construct its ProseJudgement — LLMUsage rejects it — so the run FAILs
    # rather than metering a negative into the total.
    async def verdict(evaluated_by, criteria, state):
        return ProseJudgement(verdicts=[True], usage=LLMUsage(input_tokens=-100, output_tokens=0))

    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"], guards={"max_tokens_total": 10}),
        {"qa_status": "PASS"}, None, raw=verdict,
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "FAILED"


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, float("nan"), float("inf")])
def test_llm_usage_rejects_non_int_or_negative(bad):
    # A token count is a non-negative *int*: bool (an int subclass), float, and
    # NaN/inf must all be rejected — NaN especially, since `NaN > cap` is False
    # and would make the cost cap un-triggerable.
    with pytest.raises(ValueError, match="non-negative int"):
        LLMUsage(input_tokens=bad, output_tokens=0)


def test_llm_usage_add_rejects_negative_delta_not_netted():
    # A -100 delta must raise, not be absorbed against a larger positive total
    # (100 + -100 = 0 would otherwise pass a result-only check).
    with pytest.raises(ValueError, match="non-negative int"):
        LLMUsage(100, 20).add(-100, -20)


def test_llm_usage_is_immutable():
    # A post-construction `usage.input_tokens = -100` (a cost-cap bypass) must
    # fail — the non-negative invariant is held by frozen-ness, not just at
    # construction.
    usage = LLMUsage(input_tokens=5, output_tokens=5)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        usage.input_tokens = -100  # type: ignore[misc]


def test_corrupted_judgement_usage_fails_closed_not_over_cap(con):
    # A runtime handing back a duck-typed usage with a NaN count must NOT let the
    # run COMPLETE by making `NaN > cap` False — the engine rebuilds usage through
    # LLMUsage, which rejects it, and the run FAILs closed.
    import types

    async def verdict(evaluated_by, criteria, state):
        judgement = ProseJudgement(verdicts=[True])
        judgement.usage = types.SimpleNamespace(input_tokens=float("nan"), output_tokens=0, total=float("nan"))
        return judgement

    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"], guards={"max_tokens_total": 10}),
        {"qa_status": "PASS"}, None, raw=verdict,
    )
    run = con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "evaluator_error"


def test_prose_judgement_error_rejects_non_usage():
    # The failure carrier must not trust a raw payload — a runtime raising
    # `ProseJudgementError("oops")` fails at construction, not later as an
    # AttributeError in the gate.
    with pytest.raises(TypeError, match="must be an LLMUsage"):
        ProseJudgementError("not-usage")  # type: ignore[arg-type]


def test_runtime_raising_malformed_error_fails_closed(con):
    # A verdict that raises ProseJudgementError with a bad payload → the
    # construction raises TypeError, which the gate treats as an evaluator error
    # and FAILs the run closed (ended_at set, DOD event present) — not stranded.
    async def verdict(evaluated_by, criteria, state):
        raise ProseJudgementError("not-usage")  # type: ignore[arg-type]

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "evaluator_error"


def test_mutated_error_usage_fails_closed_not_stranded(con):
    # Belt-and-suspenders for a post-construction mutation of exc.usage: the
    # engine's isinstance guard must keep the run from stranding at RUNNING.
    async def verdict(evaluated_by, criteria, state):
        err = ProseJudgementError(LLMUsage(5, 5))
        err.usage = "not-usage"  # type: ignore[assignment]  # mutate past the constructor
        raise err

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    assert json.loads(_dod_event(con, run_id)["state_diff"])["outcome"] == "evaluator_error"


def test_failure_path_revalidates_corrupted_usage(con):
    # A ProseJudgementError carrying an LLMUsage whose frozen field was corrupted
    # (past the guard, via object.__setattr__) to a non-serializable value must
    # be re-validated: the engine rebuilds it through LLMUsage, records NO usage,
    # and still writes the DOD_EVALUATED event + FAILs closed — rather than
    # throwing on serialization and leaving no event.
    async def verdict(evaluated_by, criteria, state):
        u = LLMUsage(5, 5)
        object.__setattr__(u, "input_tokens", object())  # corrupt past frozen + validation
        raise ProseJudgementError(u)

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    run = con.execute("SELECT status, ended_at FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED" and run["ended_at"] is not None
    event = _dod_event(con, run_id)
    assert event is not None  # the event WAS written (corrupt usage didn't break serialization)
    sd = json.loads(event["state_diff"])
    assert sd["outcome"] == "evaluator_error"
    assert "usage" not in sd  # corrupted usage recorded as no usage


def test_failed_judgement_over_cap_is_cost_cap_not_evaluator_error(con):
    # A judgement that FAILS but whose spend breaches the hard cap is labelled
    # cost_cap_exceeded (the governance-salient outcome), not evaluator_error —
    # the cap applies to any usage that enters the tally, success or failure.
    async def verdict(evaluated_by, criteria, state):
        raise ProseJudgementError(LLMUsage(input_tokens=300, output_tokens=60))

    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"], guards={"max_tokens_total": 10}),
        {"qa_status": "PASS"}, None, raw=verdict,
    )
    run = con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    sd = json.loads(_dod_event(con, run_id)["state_diff"])
    assert sd["outcome"] == "cost_cap_exceeded"
    assert sd["usage"] == {"input_tokens": 300, "output_tokens": 60}


def test_failed_judgement_under_cap_stays_evaluator_error(con):
    # ...but a failed judgement whose spend is WITHIN the cap keeps the
    # evaluator_error label (the cap check doesn't reclassify every failure).
    async def verdict(evaluated_by, criteria, state):
        raise ProseJudgementError(LLMUsage(input_tokens=2, output_tokens=1))

    run_id = _start_with_verdict(
        con, _dod_workflow(["a prose criterion"], guards={"max_tokens_total": 100}),
        {"qa_status": "PASS"}, None, raw=verdict,
    )
    sd = json.loads(_dod_event(con, run_id)["state_diff"])
    assert sd["outcome"] == "evaluator_error"


def test_failed_judgement_usage_is_recorded_on_event(con):
    # A judgement that fails outright (ProseJudgementError) still spent tokens;
    # the engine records them on the DOD_EVALUATED event so a failed judgement's
    # cost isn't invisible to accounting, and fails the run closed.
    async def verdict(evaluated_by, criteria, state):
        raise ProseJudgementError(LLMUsage(input_tokens=300, output_tokens=60))

    run_id = _start_with_verdict(con, _dod_workflow(["a prose criterion"]), {"qa_status": "PASS"}, None, raw=verdict)
    run = con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    event = json.loads(_dod_event(con, run_id)["state_diff"])
    assert event["outcome"] == "evaluator_error"
    assert event["usage"] == {"input_tokens": 300, "output_tokens": 60}
