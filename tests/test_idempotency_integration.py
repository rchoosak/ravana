"""Idempotency across a real engine-driven retry (§3.6) — integration.

The executor-unit tests in test_tool_execution.py prove dedup when the SAME
idempotency key is handed to `execute()` twice. That leaves the load-bearing
question they can't answer: when the ENGINE retries a failed node, does the
re-dispatch actually recompute the *same* key — so the side effect fires once
across the retry — or does it mint a fresh one and double-fire?

This drives it end to end: a node calls a side-effecting tool, then fails
transiently AFTER the tool ran; the engine retries the node (§3.6); the second
dispatch recomputes the key from the engine-supplied `logical_visit_id` exactly
as the gateway does, and the executor's ledger must recognise the duplicate.
The assertion is on the real side effect — the handler's call count — not on a
key looking stable in isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.base import AgentTurnResult, TransientAgentError
from ravana.runtime.idempotency import compute_idempotency_key
from ravana.runtime.tool_executor import RavanaToolExecutor
from ravana.schema.models import WorkflowDoc

from tests.test_tool_execution import CountingHandler


def _single_node_graph():
    return compile_workflow(
        WorkflowDoc.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": "idem", "version": 1},
                "spec": {
                    "agents": [
                        {"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}
                    ],
                    "graph": {"entry": "n", "nodes": [{"id": "n", "agent": "a"}], "edges": []},
                    "guards": {"max_retries_per_node": 2},
                },
            }
        )
    )


class _ToolThenTransientRuntime:
    """Calls a side-effecting tool through the real executor, then fails
    transiently on the FIRST dispatch only — forcing exactly one engine retry.

    It computes the idempotency key the same way the gateway does
    (`compute_idempotency_key` over the engine-supplied `logical_visit_id` and
    tool ordinal), so the retry's key collides with the first only if the engine
    genuinely reuses the visit id — which is the property under test.
    """

    def __init__(self, con, handler: CountingHandler):
        self._executor = RavanaToolExecutor(con, {"git_connector": handler})
        self.turns = 0
        self.keys: list[str] = []

    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        logical_visit_id: str,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult:
        self.turns += 1
        key = compute_idempotency_key(
            run_id, node_id, logical_visit_id, 0, "git_connector", {"x": 1}
        )
        self.keys.append(key)
        await self._executor.execute(
            run_id=run_id,
            node_id=node_id,
            tool="git_connector",
            arguments={"x": 1},
            idempotency_key=key,
        )
        if self.turns == 1:
            # Transient failure AFTER the side effect fired — the exact window
            # §3.6's dedup exists for: the effect happened, the turn didn't.
            raise TransientAgentError("simulated transient failure after tool ran")
        return AgentTurnResult(structured_payload={"done": True})

    async def aclose(self) -> None:
        return None


def test_engine_retry_reuses_the_key_so_the_side_effect_fires_once(con):
    graph = _single_node_graph()
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    handler = CountingHandler()
    runtime = _ToolThenTransientRuntime(con, handler)

    run_id = asyncio.run(
        start_run(
            con, graph, runtime, org_id="test", workflow_id=workflow_id,
            retry_sleep=_no_sleep,
        )
    )

    status = con.execute("SELECT status FROM run WHERE id = ?", (run_id,)).fetchone()[0]
    assert status == "COMPLETED"
    assert runtime.turns == 2, "the transient failure must have forced exactly one retry"
    # The load-bearing assertion: the side effect fired ONCE despite two turns,
    # because the retry recomputed the SAME key and the ledger deduped it.
    assert handler.calls == 1
    assert runtime.keys[0] == runtime.keys[1], "engine reused the logical visit id on retry"
    rows = con.execute(
        "SELECT idempotency_key, status FROM tool_invocation"
    ).fetchall()
    assert [(r["idempotency_key"], r["status"]) for r in rows] == [(runtime.keys[0], "SUCCEEDED")]


async def _no_sleep(_seconds: float) -> None:
    return None
