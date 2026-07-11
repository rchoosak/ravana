"""Exponential backoff for transient-failure retries (§3.6). One pure delay
function shared by both retry sites — the engine's per-node retry (a
`TransientAgentError` re-queues the node, bounded by
`guards.max_retries_per_node`) and the gateway's per-fallback-entry retry (a
`ProviderError` retries the same entry once before moving down the chain).

Equal jitter (half the exponential delay fixed, half uniformly random) rather
than full jitter: it still de-synchronizes concurrent retriers, but keeps a
deterministic lower bound (delay >= exp/2), which makes "it actually backed
off" assertable in tests and predictable in logs.

The *sleep* itself is injected at each call site (an `asyncio.sleep`-shaped
callable), so tests record requested delays instead of actually waiting.
"""

from __future__ import annotations

import random
from typing import Awaitable, Callable

# Uniform-random source, injectable for deterministic tests: rng(a, b) -> float in [a, b].
Rng = Callable[[float, float], float]

# The asyncio.sleep-shaped waiter both retry sites inject (engine per-node,
# gateway per-entry). One alias so the concept is spelled once.
RetrySleep = Callable[[float], Awaitable[None]]


def backoff_delay(failure_number: int, *, base: float, cap: float, rng: Rng = random.uniform) -> float:
    """Delay in seconds before retrying after the `failure_number`-th
    CONSECUTIVE failure (1-indexed: the first failure backs off `~base`,
    doubling per failure, capped at `cap`). Named to be unmistakably the
    failure-streak ordinal — NOT the engine's lifetime `attempt` number, which
    counts successes too (a §3.7 loop revisit must not inflate the delay).
    Equal jitter: exp/2 fixed + uniform(0, exp/2)."""
    if failure_number < 1:
        raise ValueError(f"failure_number is 1-indexed, got {failure_number}")
    exp = min(cap, base * (2 ** (failure_number - 1)))
    return exp / 2 + rng(0.0, exp / 2)
