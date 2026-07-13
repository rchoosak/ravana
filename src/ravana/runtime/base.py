"""The interface the engine dispatches through (§1.2's Agent Runtime). Phase
0a implements only ravana.runtime.mock.MockAgentRuntime; Phase 0b adds a real
provider-backed implementation of the same protocol so the engine loop
doesn't change when the backend does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMUsage:
    """Token usage accumulated over one or more provider calls. Non-negative
    **and immutable** by construction: a negative count (a buggy or hostile
    runtime reporting `input_tokens=-100` to duck a cost cap) is rejected at
    construction, and `frozen=True` blocks a post-hoc `usage.input_tokens = -100`
    that would otherwise slip past that check straight into the metered total.
    `add` returns a *new* LLMUsage (the frozen instance can't self-mutate), which
    is the accumulation point across retries/fallbacks so usage from an attempt
    that later failed is never lost."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError(f"token usage must be non-negative, got {self.input_tokens}/{self.output_tokens}")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> LLMUsage:
        """A new LLMUsage with these (non-negative) tokens folded in."""
        return LLMUsage(self.input_tokens + input_tokens, self.output_tokens + output_tokens)


@dataclass
class ProseJudgement:
    """A runtime's ruling on a run's *prose* Definition-of-Done criteria (§3.1
    step 7). `verdicts` is **position-aligned** to the prose criteria the judge
    was handed — `verdicts[i]` is the ruling on `prose_criteria[i]` — so no
    criterion identity is ever carried by (collidable) text. A short/empty list
    fails closed: any criterion without an explicit `True` is treated as not
    met. `usage` is the (non-negative) LLM token usage the judgement spent,
    which the engine records and meters against `guards.max_tokens_total`
    before it lets a run COMPLETE."""

    verdicts: list[bool] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)


class ProseJudgementError(Exception):
    """Raised by a prose judge when the whole judgement fails (repairs
    exhausted, every fallback down, a leaked-secret response). Carries the
    token `usage` spent up to the failure so the engine still accounts for it
    (a failed judgement's tokens were still billed), and chains the underlying
    cause via `__cause__` for the durable outcome's classification."""

    def __init__(self, usage: "LLMUsage") -> None:
        super().__init__("prose judgement failed")
        self.usage = usage


@runtime_checkable
class ProseJudge(Protocol):
    """A runtime that can judge prose DoD criteria. Detected structurally (not
    by concrete class) so any future runtime gaining this capability is wired
    at the DoD gate without a CLI change (§3.1 step 7)."""

    async def judge_prose(
        self, agent_id: str, criteria: list[str], state: dict[str, Any]
    ) -> ProseJudgement: ...


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
