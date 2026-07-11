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

_MID = lambda a, b: (a + b) / 2  # noqa: E731 - rng stub returning the interval midpoint


def test_delay_doubles_per_attempt():
    # With rng pinned to the midpoint, equal jitter yields exactly 3/4 of the
    # exponential: attempt 1 -> 0.75*base, 2 -> 1.5*base, 3 -> 3*base.
    assert backoff_delay(1, base=1.0, cap=30.0, rng=_MID) == pytest.approx(0.75)
    assert backoff_delay(2, base=1.0, cap=30.0, rng=_MID) == pytest.approx(1.5)
    assert backoff_delay(3, base=1.0, cap=30.0, rng=_MID) == pytest.approx(3.0)


def test_delay_is_capped():
    # Attempt 10 of base=1 would be 512s uncapped; the cap bounds it.
    assert backoff_delay(10, base=1.0, cap=30.0, rng=_MID) == pytest.approx(22.5)  # 0.75 * cap


def test_jitter_stays_within_equal_jitter_bounds():
    # Real rng: delay must live in [exp/2, exp] — never below half (equal
    # jitter's deterministic floor), never above the full exponential.
    for attempt in (1, 2, 3, 4):
        exp = min(30.0, 1.0 * 2 ** (attempt - 1))
        for _ in range(50):
            d = backoff_delay(attempt, base=1.0, cap=30.0)
            assert exp / 2 <= d <= exp


def test_attempt_is_one_indexed():
    with pytest.raises(ValueError, match="1-indexed"):
        backoff_delay(0, base=1.0, cap=30.0)


def test_engine_delays_grow_exponentially_across_retries(con):
    # Two consecutive transient failures on one node: the recorded backoffs
    # must double (attempt 1 ~base, attempt 2 ~2*base), per §3.6.
    doc = WorkflowDoc.model_validate(
        {
            "apiVersion": "ravana/v1",
            "kind": "Workflow",
            "metadata": {"name": "backoff-test", "version": 1},
            "spec": {
                "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
                "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
            },
        }
    )
    graph = compile_workflow(doc)
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime(
        {"only": [{"transient_error": True}, {"transient_error": True}, {"structured_payload": {}}]}
    )
    delays: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        delays.append(seconds)

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, retry_sleep=recording_sleep)
    )
    assert con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()["status"] == "COMPLETED"
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 1.0  # attempt 1: exp=1s, equal jitter
    assert 1.0 <= delays[1] <= 2.0  # attempt 2: exp=2s — grew, not flat
