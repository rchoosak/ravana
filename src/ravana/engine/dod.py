"""Definition-of-Done evaluation (§3.1 step 7). A run that reaches a terminal
only COMPLETES if its `definition_of_done` is met, otherwise it FAILs with the
unmet criteria named. Previously `definition_of_done` was parsed and persisted
but never evaluated (flagged in the v0.16 design review).

Criteria come in two kinds (ARCHITECTURE §4):
  - EXPRESSION criteria (e.g. `state.qa_status == 'PASS'`) — evaluated
    deterministically through the same sandboxed condition engine the router
    uses for edges (`engine.expr.eval_condition`).
  - PROSE criteria (e.g. "All acceptance criteria are met") — natural language
    needing the `evaluated_by` agent's judgement.

`evaluate_dod` is **pure and synchronous**: it classifies every criterion and
resolves the expression ones, leaving prose criteria *unevaluated* (`passed is
None`). Prose judgement is I/O (it runs an agent turn), so it lives at the
engine boundary: the caller fetches `DodResult.prose_criteria`, obtains a
position-aligned `ProseJudgement`, and calls `apply_prose_verdict`. Keeping the
evaluator sync means the deterministic half stays trivially testable without an
event loop, and the one place that touches the network is the injected verdict.

Classification is by PARSE, not by evaluation: a criterion is an expression iff
it parses as one (`is_boolean_expression`); a sentence like "No open defects in
state.qa_report" fails to parse and is prose. A parseable expression that only
ERRORS at evaluation (e.g. `state.count > 5` on an unset key, where `None > 5`
raises) is still an expression — a *failing* one — not prose; treating such a
raise as prose would silently downgrade a genuinely-unmet criterion to advisory
and defeat the gate.

Verdicts are applied FAIL-CLOSED throughout: a prose criterion counts as met
only when its verdict is *exactly* boolean `True`. An omitted, short, or
non-boolean verdict reads as not-met — an incomplete or garbled judgement can
never let a run COMPLETE on a criterion that was not actually proven.
"""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ravana.engine.expr import eval_condition, is_boolean_expression
from ravana.runtime.base import ProseJudgement
from ravana.schema.models import DefinitionOfDone

# Given (evaluated_by agent id, the prose criteria in order, final shared_state),
# returns a ProseJudgement whose `verdicts` are position-aligned to those
# criteria. Injected by the caller that has a runtime to ask; unset means prose
# isn't evaluated (advisory). Async because a real verdict runs an agent turn.
ProseVerdict = Callable[[str, list[str], dict[str, Any]], Awaitable[ProseJudgement]]


@dataclass
class CriterionResult:
    criterion: str
    kind: Literal["expression", "prose"]
    passed: bool | None  # None => not evaluated (prose with no verdict applied)


@dataclass
class DodResult:
    results: list[CriterionResult] = field(default_factory=list)

    @property
    def met(self) -> bool:
        """True iff every *evaluated* criterion passed. Unevaluated prose (no
        verdict applied) is advisory and does not gate. An empty criteria list
        is vacuously met."""
        return all(r.passed for r in self.results if r.passed is not None)

    @property
    def unmet(self) -> list[str]:
        return [r.criterion for r in self.results if r.passed is False]

    @property
    def unevaluated(self) -> list[str]:
        return [r.criterion for r in self.results if r.passed is None]

    @property
    def prose_criteria(self) -> list[str]:
        """The prose criteria, in declaration order — the exact list a prose
        verdict is asked to rule on and whose verdicts align positionally."""
        return [r.criterion for r in self.results if r.kind == "prose"]

    def apply_prose_verdict(self, verdicts: Sequence[bool]) -> None:
        """Apply a position-aligned prose verdict, FAIL-CLOSED: the i-th prose
        result takes `verdicts[i]` only when that entry is exactly `True`, else
        not-met; a missing entry (short list) is not-met. Verdicts are aligned
        to `prose_criteria` order, so two criteria with identical text keep
        independent rulings — nothing is keyed by (collidable) criterion text."""
        prose_results = [r for r in self.results if r.kind == "prose"]
        for i, result in enumerate(prose_results):
            result.passed = (verdicts[i] is True) if i < len(verdicts) else False

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


def evaluate_dod(dod: DefinitionOfDone, state: dict[str, Any]) -> DodResult:
    """Classify and deterministically evaluate a DoD against the final `state`.
    Expression criteria are resolved here; prose criteria are left unevaluated
    (`passed is None`) for the caller to judge via a ProseVerdict and
    `apply_prose_verdict`. Pure and synchronous — no I/O."""
    results: list[CriterionResult] = []
    for criterion in dod.criteria:
        if is_boolean_expression(criterion):
            results.append(CriterionResult(criterion, "expression", _evaluate_expression(criterion, state)))
        else:
            results.append(CriterionResult(criterion, "prose", None))
    return DodResult(results=results)
