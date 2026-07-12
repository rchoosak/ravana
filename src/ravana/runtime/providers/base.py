"""Provider-agnostic normalization layer (ARCHITECTURE.md §1.4). The LLM
Gateway speaks these shapes; each provider adapter translates them to/from
its own wire format. This is the seam that lets a single run mix Anthropic,
OpenAI, and local Ollama agents without the engine (or the gateway's
strategy/loop logic) knowing which produced a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ravana.runtime.secrets import ResolvedSecret, redact_secrets


class Capability(str, Enum):
    """§3.4's capability flags — an adapter declares which structured-output
    mechanisms its model actually supports, and the Gateway picks the
    strongest available (guided > native > repair-loop)."""

    GUIDED_DECODING = "guided_decoding"
    NATIVE_STRUCTURED_OUTPUT = "native_structured_output"


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class NormalizedToolCall:
    """A tool the model asked to invoke. `id` threads back to the provider's
    own tool-call id so a result can be correlated on the next turn."""

    id: str
    tool: str
    arguments: dict[str, Any]


# --- Normalized transcript --------------------------------------------------
# The gateway builds the conversation in these provider-neutral shapes; each
# adapter translates them to its own wire format in complete(). This is the
# fix for multi-turn tool use: the OpenAI wire format needs the assistant's
# prior tool_calls message preserved and tool results as role:"tool", while
# Anthropic needs tool_use blocks in the assistant turn and tool_result
# content blocks inside a following user turn. Building one OpenAI-shaped list
# and passing it raw to both (the earlier approach) breaks real providers even
# though a fake adapter accepts it.


@dataclass
class UserMessage:
    text: str
    role: str = "user"


@dataclass
class AssistantMessage:
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    text: str | None = None
    role: str = "assistant"


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool: str
    content: str
    role: str = "tool_result"


Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass
class ProviderRequest:
    model: str
    system: str
    messages: list[Message]
    tools: list[Tool] = field(default_factory=list)
    force_tool: str | None = None  # force this tool (submit_result) via native forced tool-choice
    output_schema: dict[str, Any] | None = None  # for guided decoding / native json-schema
    temperature: float | None = None  # adapters MUST drop this for models that reject non-default sampling params
    max_tokens: int | None = None
    endpoint: str | None = None
    # The RESOLVED credential (§8c: the gateway resolves `llm.api_key_ref`
    # through the secret resolver at dispatch; adapters never see the pointer,
    # and ResolvedSecret self-redacts in repr so the plaintext cannot ride a
    # debug log or pytest assertion diff — §8: secrets never logged). None =
    # no per-agent key — the SDK falls back to its own env var.
    api_key: ResolvedSecret | None = None


@dataclass
class ProviderResponse:
    text: str | None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class ProviderError(Exception):
    """A provider call failed. `retryable` is the §3.6 taxonomy split the
    gateway acts on:

      - retryable=True (429/5xx/timeouts/connection blips): retrying the SAME
        entry after a backoff can plausibly succeed — the gateway spends its
        per-entry retry, then moves down the fallback chain.
      - retryable=False (auth 401/403, invalid request 400/422, model 404):
        retrying the same entry cannot succeed; the gateway skips the
        same-entry retry (and its backoff sleep) and goes straight to the next
        fallback entry — a different provider/model may well work.

    When EVERY entry in the chain failed permanently, the turn error is not
    transient at all: the gateway raises a hard error the engine fails the run
    on immediately, instead of TransientAgentError (which would burn
    max_retries_per_node re-running a hopeless chain)."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


# HTTP statuses where a retry of the same endpoint can plausibly succeed:
# timeout, contention, rate limit, server-side failure.
_RETRYABLE_STATUSES = frozenset({408, 409, 429})

# Builtin exception types that signal a programming/config error (a missing
# credential surfaces as the SDK's TypeError/ValueError, not an HTTP status) —
# a retry re-executes the same broken code/config and cannot succeed.
_PERMANENT_EXC_TYPES = (TypeError, ValueError, KeyError, AttributeError)


def classify_retryable(exc: Exception) -> bool:
    """Whether a provider SDK exception is worth retrying at the same entry.
    Both the anthropic and openai SDKs expose `status_code` on their API error
    types, so status decides when present. Without a status: a builtin
    programming/config error (e.g. the anthropic SDK's TypeError for a missing
    credential) is permanent — retrying re-runs the same broken config — while
    anything else (connection reset, timeout, unknown SDK error) is retryable,
    an availability bias for genuinely flaky networks."""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUSES or status >= 500
    return not isinstance(exc, _PERMANENT_EXC_TYPES)


# A per-adapter client-cache ceiling. Clients are keyed by resolved credential
# (§8c per-agent routing), so a genuinely-rotating resolver (Vault, Phase 2)
# would otherwise grow the cache without bound. Env-based keys are stable per
# process, so in Phase 0b this ceiling is never reached — it's a defensive cap.
# (Closing the evicted client's connection pool needs an async aclose(); that
# lands with Phase 1 connection management. The count is what's bounded here.)
_MAX_CACHED_CLIENTS = 32


def _bound_client_cache(clients: dict[Any, Any]) -> None:
    """Evict the oldest cached client (dicts preserve insertion order) before
    inserting a new one past the ceiling, so key rotation can't grow the cache
    without limit."""
    while len(clients) >= _MAX_CACHED_CLIENTS:
        oldest = next(iter(clients))
        del clients[oldest]


def to_provider_error(
    prefix: str, exc: Exception, *, retryable: bool | None = None, secret: ResolvedSecret | None = None
) -> ProviderError:
    """The one place an SDK exception becomes a normalized, classified
    ProviderError — both adapters call this instead of hand-rolling the wrap.
    The exception text is redacted (§8): an SDK error can echo the very
    credential the runtime injected into the client, and this message flows
    onward into node_execution.error and the log stream. `secret` is the
    credential this call had in scope — passed so it's scrubbed by exact value
    even if it doesn't match a known pattern; the pattern sweep is the backstop
    for anything the caller didn't have in hand."""
    values = (secret.value(),) if secret is not None else ()
    return ProviderError(
        f"{prefix}: {redact_secrets(str(exc), values=values)}",
        retryable=classify_retryable(exc) if retryable is None else retryable,
    )


class ProviderAdapter(Protocol):
    name: str

    def capabilities(self, model: str) -> set[Capability]: ...

    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...
