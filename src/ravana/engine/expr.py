"""Sandboxed expression evaluation for `condition_expr` and `on_enter` (§6:
simpleeval was chosen over CEL specifically because its Python binding is a
second-class citizen — see ARCHITECTURE.md §6).

The YAML examples throughout ARCHITECTURE.md/EXAMPLES.md write conditions in
JS-ish syntax (`&&`, `||`, `!`, `null`) rather than Python's (`and`, `or`,
`not`, `None`) — e.g. "state.qa_status == 'FAIL' && state.iteration_count < 5"
and "state.lead_verdict == null". This module translates that surface syntax
before handing the expression to simpleeval, rather than requiring every
workflow author to write Python.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from simpleeval import EvalWithCompoundTypes

_NOT_RE = re.compile(r"!(?!=)")
_NULL_RE = re.compile(r"\bnull\b")
_TRUE_RE = re.compile(r"\btrue\b")
_FALSE_RE = re.compile(r"\bfalse\b")


class StateProxy:
    """Wraps shared_state so `state.qa_status` (attribute syntax, matching
    every example in ARCHITECTURE.md) reads from the underlying dict.
    Missing keys evaluate to None, matching "state.lead_verdict == null"
    being a meaningful, non-erroring check before that key is ever set."""

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)


def _translate(expr: str) -> str:
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = _NOT_RE.sub(" not ", expr)
    expr = _NULL_RE.sub("None", expr)
    expr = _TRUE_RE.sub("True", expr)
    expr = _FALSE_RE.sub("False", expr)
    return expr


def eval_condition(expr: str, shared_state: dict[str, Any]) -> bool:
    evaluator = EvalWithCompoundTypes(names={"state": StateProxy(shared_state)})
    result = evaluator.eval(_translate(expr))
    return bool(result)


def is_boolean_expression(expr: str) -> bool:
    """Whether `expr` PARSES as a Python expression (after surface-syntax
    translation) — used by the DoD evaluator (§3.1 step 7) to tell an
    *expression* criterion from a *prose* sentence. Classification is by parse,
    NOT by evaluation: a sentence like "All acceptance criteria are met" fails
    to parse (adjacent bare words) and is prose, whereas a parseable expression
    that merely ERRORS at evaluation (e.g. `state.count > 5` when the key is
    unset, so `None > 5` raises) is still an expression — a failing one — not
    prose. Evaluating-vs-parsing is the distinction that keeps a genuinely
    unmet ordering/`in` criterion from being silently downgraded to advisory
    prose."""
    try:
        ast.parse(_translate(expr), mode="eval")
        return True
    except SyntaxError:
        return False


_ON_ENTER_RE = re.compile(r"^\s*state\.(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>\+=|-=|=)\s*(?P<rhs>.+)$")


def apply_on_enter(expr: str, shared_state: dict[str, Any]) -> dict[str, Any]:
    """`on_enter` is a state *mutation*, not a boolean expression — simpleeval
    evaluates expressions, not assignment statements, so this supports the
    narrow set of forms actually used (`state.x = ...`, `state.x += ...`,
    `state.x -= ...`) rather than a full statement interpreter. Returns the
    delta to merge into shared_state (matching the same commit path as an
    agent's own state_delta, §3.1 step 4)."""
    match = _ON_ENTER_RE.match(expr)
    if not match:
        raise ValueError(f"unsupported on_enter expression: {expr!r} (expected 'state.<key> (=|+=|-=) <expr>')")
    key, op, rhs = match.group("key"), match.group("op"), match.group("rhs")
    evaluator = EvalWithCompoundTypes(names={"state": StateProxy(shared_state)})
    rhs_value = evaluator.eval(_translate(rhs))
    current = shared_state.get(key)
    if op == "=":
        new_value = rhs_value
    elif op == "+=":
        new_value = (current or 0) + rhs_value
    elif op == "-=":
        new_value = (current or 0) - rhs_value
    else:  # pragma: no cover - regex constrains op to the three above
        raise ValueError(f"unsupported operator: {op}")
    return {key: new_value}
