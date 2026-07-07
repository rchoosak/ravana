"""Scripted mock agent backend for Phase 0a — exercises the engine loop with
zero real LLM calls. A fixture maps node_id to an ordered list of responses;
each dispatch of that node consumes the next entry (clamped to the last one
once exhausted, so a node without enough scripted turns just keeps repeating
its final answer rather than erroring out).

Fixture entries are dicts with a `structured_payload` key, or the literal
`{"transient_error": true}` to make that call raise TransientAgentError once
(for exercising §3.6's retry path) before moving on to the next entry.
Optional `tool_calls` (list of `{tool, arguments}` dicts), `tool_call_count`,
`repair_count`, `input_tokens`/`output_tokens` let a fixture entry exercise
the guards in §3.4/§9 that a bare structured_payload can't reach.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ravana.runtime.base import AgentTurnResult, TransientAgentError


class MockAgentRuntime:
    """Deliberately stateless: the call index comes from `attempt` (the
    DB-persisted node_execution counter), not an in-memory dict. An earlier
    version tracked call counts in memory, which broke the moment the CLI
    was used as intended — each CLI invocation (`run start`, then a separate
    `run hitl respond` process) is a fresh Python process, so an in-memory
    counter silently resets to 0 and replays response #0 forever instead of
    advancing. `attempt` survives across processes because it's a column,
    not a variable."""

    def __init__(self, fixture: dict[str, list[dict[str, Any]]]):
        self._fixture = fixture

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MockAgentRuntime":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw.get("responses", {}))

    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult:
        responses = self._fixture.get(node_id)
        if not responses:
            raise ValueError(f"no mock fixture responses configured for node '{node_id}'")

        call_index = attempt - 1
        entry = responses[min(call_index, len(responses) - 1)]

        if entry.get("transient_error"):
            raise TransientAgentError(f"mock transient error for node '{node_id}' (call {call_index})")

        return AgentTurnResult(
            structured_payload=entry.get("structured_payload", {}),
            content=entry.get("content", f"[mock turn for {node_id}, call {call_index}]"),
            tool_calls=entry.get("tool_calls", []),
            input_tokens=entry.get("input_tokens", 0),
            output_tokens=entry.get("output_tokens", 0),
            tool_call_count=entry.get("tool_call_count", len(entry.get("tool_calls", []))),
            repair_count=entry.get("repair_count", 0),
        )
