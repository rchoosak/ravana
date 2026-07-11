"""Anthropic provider adapter. Structured output uses **forced tool-calling**
(§3.4's "native structured output" tier): the Gateway's synthetic
`submit_result` tool is passed as the only forced choice, so the model's
final turn is guaranteed to be a `submit_result` tool_use whose `input`
conforms to the node's output_schema.

Two model-family facts drive this adapter (from the claude-api reference):
- Current models (claude-opus-4-8/4-7, claude-sonnet-5, claude-fable-5)
  **reject non-default `temperature`/`top_p`/`top_k` with a 400** — so the
  adapter drops `temperature` for those, exactly the normalization §1.4 says
  the gateway is responsible for. Ravana YAML can keep declaring temperature;
  the adapter decides whether the target model can actually accept it.
- Those same models don't take `budget_tokens`; we simply don't enable
  thinking on a forced-tool extraction call (thinking isn't needed to fill a
  schema, and keeping it off avoids forced-tool_choice interaction quirks).

The Anthropic client is injected (`client=`), so tests pass a fake and never
touch the network or need an API key.
"""

from __future__ import annotations

from typing import Any

from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    to_provider_error,
    ProviderRequest,
    ProviderResponse,
)

# Model families that 400 on non-default temperature/top_p/top_k (claude-api
# reference: removed on Fable 5 / Opus 4.8 / 4.7 / Sonnet 5). Matched by prefix
# so dated snapshots and aliases are both covered.
_NO_SAMPLING_PARAM_PREFIXES = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _accepts_temperature(model: str) -> bool:
    return not any(model.startswith(p) for p in _NO_SAMPLING_PARAM_PREFIXES)


class AnthropicAdapter:
    name = "anthropic"

    def __init__(self, client: Any | None = None):
        # Deferred import so the dependency is only needed when this adapter is
        # actually constructed with a real client (tests inject a fake).
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self._client = client

    def capabilities(self, model: str) -> set[Capability]:
        # Anthropic offers provider-guaranteed conformance via forced
        # tool-calling, not token-level guided decoding.
        return {Capability.NATIVE_STRUCTURED_OUTPUT}

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or 4096,
            "system": request.system,
            "messages": _to_anthropic_messages(request.messages),
        }
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in request.tools
            ]
        if request.force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": request.force_tool}
        if request.temperature is not None and _accepts_temperature(request.model):
            kwargs["temperature"] = request.temperature

        try:
            message = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize every provider failure to one type
            # §3.6 taxonomy: classified retryable/permanent in one shared place
            # (a missing credential arrives as the SDK's TypeError — no
            # status_code — and classifies permanent, not transient).
            raise to_provider_error("anthropic completion failed", exc) from exc

        text_parts: list[str] = []
        tool_calls: list[NormalizedToolCall] = []
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(NormalizedToolCall(id=block.id, tool=block.name, arguments=dict(block.input)))

        usage = getattr(message, "usage", None)
        return ProviderResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            stop_reason=getattr(message, "stop_reason", None),
        )


def _to_anthropic_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Translate the gateway's normalized transcript to Anthropic wire format:
    a tool call is a `tool_use` block inside the assistant turn, and its
    result is a `tool_result` content block inside a following user turn."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            content: list[dict[str, Any]] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.tool, "input": tc.arguments})
            out.append({"role": "assistant", "content": content})
        elif m.role == "tool_result":
            out.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}]}
            )
    return out
