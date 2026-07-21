"""Toolkit handler interface (§1.2, §1.7). A handler is the executor behind a
Toolkit manifest — it takes JSON arguments and returns a result string the
agent turn feeds back to the model. The logical-invocation `idempotency_key`
(§3.6) is passed in so a side-effecting handler can forward it to the remote
(e.g. an `Idempotency-Key` header) in addition to Ravana's own ledger-level
dedup in RavanaToolExecutor.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Protocol


class ToolFailureKind(Enum):
    """§3.6's three-way classification of a tool failure — one enum, so an
    illegal combination (a failure both transient and fatal) cannot be
    constructed, and each kind names the route the gateway takes."""

    # Transport timeout, HTTP 5xx/429/408: §3.6 lists "tool timeout" as
    # TRANSIENT — the turn ends as a TransientAgentError so the engine retries
    # the node_execution attempt with backoff (side effects already fired are
    # deduped by the logical-invocation idempotency key).
    TRANSIENT = "transient"
    # Tool auth 401/403: §3.6 "tool auth failure" is NON-transient — the run
    # fails immediately; neither the model nor a retry can fix credentials.
    FATAL = "fatal"
    # HTTP 404/422, invalid arguments, bad path: fed back into the turn as an
    # error tool_result so the model can adjust its call or route around it.
    MODEL_ADDRESSABLE = "model_addressable"


class ToolkitError(Exception):
    """A toolkit failed to execute; `kind` (ToolFailureKind) decides how the
    gateway routes it. Defaults to MODEL_ADDRESSABLE — the safe middle: the
    model sees the error and adapts, nothing retries or dies silently."""

    def __init__(self, message: str, *, kind: ToolFailureKind = ToolFailureKind.MODEL_ADDRESSABLE):
        super().__init__(message)
        self.kind = kind


class ToolOutcomeUnknown(ToolkitError):
    """A side-effecting tool started, but its terminal outcome cannot be proven.

    The executor must leave this invocation STARTED so the same idempotency key
    fails closed on retry instead of being treated as a safely retryable FAILED
    call. This is distinct from an ordinary fatal error whose effect is known
    not to have happened.
    """

    def __init__(self, message: str):
        super().__init__(message, kind=ToolFailureKind.FATAL)


class ToolRetrySafeCancellation(asyncio.CancelledError):
    """Cancellation occurred before a non-repeatable tool effect could begin.

    The executor may mark a claimed side-effecting invocation FAILED so the
    same logical call can retry. Handlers must only raise this when they can
    prove no non-repeatable effect was possible.
    """


class ToolkitHandler(Protocol):
    # §8(a): every connector declares its input JSON schema. The result is a
    # plain string fed back to the model, so there is no separate output
    # schema. Both this and `description` are what the gateway needs to surface
    # the toolkit to the model as a callable tool (name = toolkit id).
    input_schema: dict[str, Any]

    # A human/model-readable line telling the model what this tool does and
    # when to reach for it. Author-provided descriptions in the manifest are a
    # later enhancement; for now each handler supplies a sensible default.
    description: str

    # Whether this handler can actually run in the current build. A deferred
    # toolkit type (currently web_search) is registered so a workflow still
    # compiles, but it is NOT executable and must not be surfaced to the model
    # as a callable tool. tools_for refuses to advertise such a handler.
    executable: bool

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        """Whether THIS call (given its arguments) has external side effects —
        method-aware, because the same handler can be read-only or mutating
        depending on args (api_connector GET vs POST). RavanaToolExecutor
        dedupes only side-effecting calls via the idempotency ledger (§3.6);
        a read-only call (GET/poll) is re-run so it returns live state, not a
        cached replay."""
        ...

    # `run_id` identifies the current run so a handler that needs per-run
    # resources can locate them — code_interpreter uses it to find `runs/<run_id>
    # /workspace` (§10.1). Handlers that don't need it ignore it. It is not the
    # idempotency key (that's the logical-invocation key above).
    async def call(self, *, arguments: dict[str, Any], idempotency_key: str, run_id: str) -> str: ...

    async def aclose(self) -> None: ...
