"""§3.6's P1 fix: the tool-call idempotency key must be stable across retries
(same tool + same arguments -> same key, regardless of attempt number) and
must change when the actual request changes."""

from __future__ import annotations

from ravana.runtime.idempotency import compute_idempotency_key


def test_same_call_produces_the_same_key_regardless_of_attempt():
    # The whole point of the fix: attempt number is NOT part of the key.
    key_attempt_1 = compute_idempotency_key("run-1", "dev_code", "git_push", {"branch": "ravana/run-1", "message": "fix bug"})
    key_attempt_2 = compute_idempotency_key("run-1", "dev_code", "git_push", {"branch": "ravana/run-1", "message": "fix bug"})
    assert key_attempt_1 == key_attempt_2


def test_different_arguments_produce_a_different_key():
    key_a = compute_idempotency_key("run-1", "dev_code", "git_push", {"message": "fix bug A"})
    key_b = compute_idempotency_key("run-1", "dev_code", "git_push", {"message": "fix bug B"})
    assert key_a != key_b


def test_key_is_order_independent_in_argument_dict():
    key_a = compute_idempotency_key("run-1", "n", "tool", {"a": 1, "b": 2})
    key_b = compute_idempotency_key("run-1", "n", "tool", {"b": 2, "a": 1})
    assert key_a == key_b


def test_different_node_or_run_changes_the_key():
    base = compute_idempotency_key("run-1", "node-a", "tool", {"x": 1})
    different_node = compute_idempotency_key("run-1", "node-b", "tool", {"x": 1})
    different_run = compute_idempotency_key("run-2", "node-a", "tool", {"x": 1})
    assert base != different_node
    assert base != different_run
