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
    api_key_ref: str | None = None


@dataclass
class ProviderResponse:
    text: str | None
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class ProviderError(Exception):
    """Non-transient provider failure (auth, invalid request). The gateway
    does not retry these — it moves to the next fallback entry (§3.6)."""


class ProviderAdapter(Protocol):
    name: str

    def capabilities(self, model: str) -> set[Capability]: ...

    async def complete(self, request: ProviderRequest) -> ProviderResponse: ...
