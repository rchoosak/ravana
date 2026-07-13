"""The interface the engine dispatches through (§1.2's Agent Runtime). Phase
0a implements only ravana.runtime.mock.MockAgentRuntime; Phase 0b adds a real
provider-backed implementation of the same protocol so the engine loop
doesn't change when the backend does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def _as_token_count(value: int) -> int:
    """A token count is a non-negative **int** — nothing else. A bare `< 0`
    check is not enough: `bool` is an int subclass (so `True`/`False` would pass
    and mean 1/0), and `float`/`NaN` slip past every comparison (`NaN < 0`,
    `NaN >= 0`, and `NaN > cap` are all False — a NaN usage would make the cost
    cap un-triggerable and land NaN in the persisted event). Require an exact
    int that is `>= 0`."""
    if type(value) is not int or value < 0:
        raise ValueError(f"token usage must be a non-negative int, got {value!r} ({type(value).__name__})")
    return value


@dataclass(frozen=True)
class LLMUsage:
    """Token usage accumulated over one or more provider calls. Each field is a
    non-negative **int** (see `_as_token_count`) and the instance is **immutable**
    (`frozen=True`) — so neither a hostile value at construction (`bool`, `float`,
    `NaN`, negative) nor a post-hoc `usage.input_tokens = -100` can slip a bad
    count into the metered total. `add` validates each incoming *delta* (not just
    the running sum, so a `-100` can't be absorbed by a larger positive) and
    returns a *new* LLMUsage — the accumulation point across retries/fallbacks,
    so usage from an attempt that later failed is never lost."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        _as_token_count(self.input_tokens)
        _as_token_count(self.output_tokens)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> LLMUsage:
        """A new LLMUsage with these tokens folded in. Each delta must itself be
        a non-negative int — a negative delta is rejected here, not silently
        netted against the running total."""
        return LLMUsage(
            self.input_tokens + _as_token_count(input_tokens),
            self.output_tokens + _as_token_count(output_tokens),
        )


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

    def __init__(self, usage: LLMUsage) -> None:
        super().__init__("prose judgement failed")
        # Enforce the contract at the source: the engine meters/records this
        # usage on the failure path, so a non-LLMUsage payload (a runtime raising
        # `ProseJudgementError("oops")`) must fail HERE — not later, as an
        # AttributeError deep in the gate that strands the run at RUNNING.
        if not isinstance(usage, LLMUsage):
            raise TypeError(f"ProseJudgementError.usage must be an LLMUsage, got {type(usage).__name__}")
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
