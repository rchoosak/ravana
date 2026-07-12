"""The interface the engine dispatches through (§1.2's Agent Runtime). Phase
0a implements only ravana.runtime.mock.MockAgentRuntime; Phase 0b adds a real
provider-backed implementation of the same protocol so the engine loop
doesn't change when the backend does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AgentTurnResult:
    """What one agent turn produces — becomes the state_delta (§3.4) plus
    bookkeeping fields that land on node_execution."""

    structured_payload: dict[str, Any]
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    repair_count: int = 0
    tool_call_count: int = 0


class TransientAgentError(Exception):
    """Raised by a runtime to simulate/represent a transient failure (LLM
    429/5xx, tool timeout) that §3.6 says should be retried with backoff,
    as opposed to a hard failure."""


class AgentOutputError(Exception):
    """The model exhausted its repair budget without producing valid
    structured output (or blew the bounded-turn ceiling without submitting).
    §3.6 classifies this as NON-transient — `node_execution.status = FAILED →
    run.status = FAILED`, immediately — because re-running the node just
    re-asks the same model the same question past a budget that already
    expired: it would spend `max_retries_per_node` and call the model beyond
    `max_output_repairs`. Deliberately NOT a TransientAgentError."""


class AgentRuntime(Protocol):
    async def run_turn(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        logical_visit_id: str,
        agent_id: str,
        shared_state: dict[str, Any],
    ) -> AgentTurnResult: ...

    async def aclose(self) -> None: ...
