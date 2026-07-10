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

Classification is by behaviour, not by a marker: a criterion is an expression
iff it evaluates cleanly through the condition engine; anything that fails to
parse/evaluate (a sentence like "No open defects in state.qa_report") is prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ravana.engine.expr import eval_condition
from ravana.schema.models import DefinitionOfDone

# Given (evaluated_by agent id, the prose criteria, final shared_state), returns
# a {criterion: passed} verdict for those criteria. Injected by the caller that
# has a runtime to ask; unset here (the default) means prose isn't evaluated.
ProseVerdict = Callable[[str, list[str], dict[str, Any]], dict[str, bool]]


@dataclass
class CriterionResult:
    criterion: str
    kind: str  # "expression" | "prose"
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


def _try_expression(criterion: str, state: dict[str, Any]) -> tuple[bool, bool]:
    """(is_expression, value). A clean evaluation through the condition engine
    means the criterion IS an expression; a raise (unparseable prose) means it
    is not."""
    try:
        return True, eval_condition(criterion, state)
    except Exception:  # noqa: BLE001 - any eval failure just means "this is prose"
        return False, False


def evaluate_dod(
    dod: DefinitionOfDone,
    state: dict[str, Any],
    *,
    prose_verdict: ProseVerdict | None = None,
) -> DodResult:
    """Evaluate every criterion against the final `state`. `met` is True iff
    every *evaluated* criterion passed — unevaluated prose (no verdict wired)
    is advisory and does not gate. An empty criteria list is vacuously met."""
    results: list[CriterionResult] = []
    prose_criteria: list[str] = []
    for criterion in dod.criteria:
        is_expr, value = _try_expression(criterion, state)
        if is_expr:
            results.append(CriterionResult(criterion, "expression", bool(value)))
        else:
            prose_criteria.append(criterion)
            results.append(CriterionResult(criterion, "prose", None))

    if prose_criteria and prose_verdict is not None:
        verdicts = prose_verdict(dod.evaluated_by, prose_criteria, state)
        for result in results:
            if result.kind == "prose":
                result.passed = bool(verdicts.get(result.criterion, False))

    met = all(r.passed for r in results if r.passed is not None)
    return DodResult(met=met, results=results)
