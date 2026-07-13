"""Definition-of-Done evaluation (§3.1 step 7). A run that reaches a terminal
only COMPLETES if its `definition_of_done` is met; otherwise it FAILs with the
unmet criteria named. Previously `definition_of_done` was parsed and persisted
but never evaluated (flagged in the v0.16 design review).

Criteria come in two kinds (ARCHITECTURE §4):
  - EXPRESSION criteria (e.g. `state.qa_status == 'PASS'`) — evaluated
    deterministically through the same sandboxed condition engine the router
    uses for edges (`engine.expr.eval_condition`).
  - PROSE criteria (e.g. "All acceptance criteria are met") — natural language
    needing the `evaluated_by` agent's judgement. A `prose_verdict` function
    can be injected to supply that judgement; when none is wired, prose
    criteria are RECORDED as *unevaluated* (surfaced in the DoD outcome) rather
    than silently passed, and they do NOT gate COMPLETED. Wiring a real
    agent-backed prose evaluator is a tracked follow-up.

Classification is by PARSE, not by evaluation: a criterion is an expression iff
it parses as one (`is_boolean_expression`); a sentence like "No open defects in
state.qa_report" fails to parse and is prose. A parseable expression that only
ERRORS at evaluation (e.g. `state.count > 5` on an unset key, where `None > 5`
raises) is still an expression — a *failing* one — not prose; treating such a
raise as prose would silently downgrade a genuinely-unmet criterion to advisory
and defeat the gate.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ravana.engine.expr import eval_condition, is_boolean_expression
from ravana.schema.models import DefinitionOfDone

# Given (evaluated_by agent id, the prose criteria, final shared_state), returns
# a {criterion: passed} verdict for those criteria. Injected by the caller that
# has a runtime to ask; unset here (the default) means prose isn't evaluated.
# Async because a real verdict runs an agent turn against the LLM Gateway — an
# awaitable the (already-async) engine loop threads through to the DoD gate.
ProseVerdict = Callable[[str, list[str], dict[str, Any]], Awaitable[dict[str, bool]]]


@dataclass
class CriterionResult:
    criterion: str
    kind: Literal["expression", "prose"]
    passed: bool | None  # None => not evaluated (prose with no evaluator wired)


@dataclass
class DodResult:
    met: bool
    results: list[CriterionResult]

    @property
    def unmet(self) -> list[str]:
        return [r.criterion for r in self.results if r.passed is False]

    @property
    def unevaluated(self) -> list[str]:
        return [r.criterion for r in self.results if r.passed is None]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable summary for the state_transition_log DOD event."""
        return {
            "met": self.met,
            "results": [{"criterion": r.criterion, "kind": r.kind, "passed": r.passed} for r in self.results],
            "unmet": self.unmet,
            "unevaluated": self.unevaluated,
        }


def _evaluate_expression(criterion: str, state: dict[str, Any]) -> bool:
    """A parseable expression's boolean value. A raise at evaluation (e.g. an
    ordering comparison against an unset key) means the criterion is NOT
    demonstrably met — fail closed to False rather than error out, so an
    erroring criterion FAILs the DoD gate instead of crashing the run."""
    try:
        return eval_condition(criterion, state)
    except Exception:  # noqa: BLE001 - a parseable expression that errors is "not met", not prose
        return False


async def evaluate_dod(
    dod: DefinitionOfDone,
    state: dict[str, Any],
    *,
    prose_verdict: ProseVerdict | None = None,
) -> DodResult:
    """Evaluate every criterion against the final `state`. `met` is True iff
    every *evaluated* criterion passed — unevaluated prose (no verdict wired)
    is advisory and does not gate. An empty criteria list is vacuously met.

    Expression criteria are evaluated deterministically and synchronously; only
    the injected prose verdict is awaited, so a run with no prose criteria (or
    no verdict wired) makes no provider call."""
    results: list[CriterionResult] = []
    prose_criteria: list[str] = []
    for criterion in dod.criteria:
        if is_boolean_expression(criterion):
            results.append(CriterionResult(criterion, "expression", _evaluate_expression(criterion, state)))
        else:
            prose_criteria.append(criterion)
            results.append(CriterionResult(criterion, "prose", None))

    if prose_criteria and prose_verdict is not None:
        verdicts = await prose_verdict(dod.evaluated_by, prose_criteria, state)
        for result in results:
            if result.kind == "prose":
                result.passed = bool(verdicts.get(result.criterion, False))

    met = all(r.passed for r in results if r.passed is not None)
    return DodResult(met=met, results=results)
