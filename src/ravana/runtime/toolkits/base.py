"""Toolkit handler interface (§1.2, §1.7). A handler is the executor behind a
Toolkit manifest — it takes JSON arguments and returns a result string the
agent turn feeds back to the model. The content-addressed `idempotency_key`
(§3.6) is passed in so a side-effecting handler can forward it to the remote
(e.g. an `Idempotency-Key` header) in addition to Ravana's own ledger-level
dedup in RavanaToolExecutor.
"""

from __future__ import annotations

from typing import Any, Protocol


class ToolkitError(Exception):
    """A toolkit failed to execute (bad config, remote error, unimplemented
    type). Surfaces to the gateway as a tool failure."""


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

    def is_side_effecting(self, arguments: dict[str, Any]) -> bool:
        """Whether THIS call (given its arguments) has external side effects —
        method-aware, because the same handler can be read-only or mutating
        depending on args (api_connector GET vs POST). RavanaToolExecutor
        dedupes only side-effecting calls via the idempotency ledger (§3.6);
        a read-only call (GET/poll) is re-run so it returns live state, not a
        cached replay."""
        ...

    async def call(self, *, arguments: dict[str, Any], idempotency_key: str) -> str: ...
