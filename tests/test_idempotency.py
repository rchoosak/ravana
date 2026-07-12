"""Logical tool-invocation identity: stable on retry, distinct on re-entry or
on a second intentional call with identical arguments."""

from __future__ import annotations

from ravana.runtime.idempotency import compute_idempotency_key


def _key(*, run="run-1", node="dev_code", visit="visit-1", ordinal=1, args=None):
    return compute_idempotency_key(
        run,
        node,
        visit,
        ordinal,
        "git_push",
        args if args is not None else {"branch": "ravana/run-1", "message": "fix bug"},
    )


def test_same_logical_call_produces_the_same_key_across_attempts():
    # Attempt is deliberately absent; retries retain visit + call ordinal.
    key_attempt_1 = _key()
    key_attempt_2 = _key()
    assert key_attempt_1 == key_attempt_2


def test_different_arguments_produce_a_different_key():
    key_a = _key(args={"message": "fix bug A"})
    key_b = _key(args={"message": "fix bug B"})
    assert key_a != key_b


def test_key_is_order_independent_in_argument_dict():
    key_a = _key(args={"a": 1, "b": 2})
    key_b = _key(args={"b": 2, "a": 1})
    assert key_a == key_b


def test_run_node_visit_and_ordinal_all_participate_in_identity():
    base = _key(node="node-a")
    assert base != _key(run="run-2", node="node-a")
    assert base != _key(node="node-b")
    assert base != _key(node="node-a", visit="visit-2")
    assert base != _key(node="node-a", ordinal=2)


def test_two_intentional_identical_calls_have_distinct_keys():
    assert _key(ordinal=1) != _key(ordinal=2)
