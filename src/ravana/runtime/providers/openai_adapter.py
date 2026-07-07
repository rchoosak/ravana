"""OpenAI-compatible provider adapter — serves BOTH the hosted OpenAI API and
a local Ollama/vLLM runtime, since Ollama exposes an OpenAI-compatible
`/v1` surface (§1.4's `endpoint` override is exactly the `base_url` here).
This is why the §4 example's Dev agent (`provider: local`, Ollama endpoint)
and QA agent (`provider: openai`) can share one adapter.

Structured output uses forced tool-calling: the synthetic `submit_result`
tool is passed with `tool_choice` forcing it, so the model's final message
is a `submit_result` call whose arguments conform to output_schema —
the same internal contract the Anthropic adapter produces, so the gateway
treats both identically (§3.4).

Capability note: a *local* model behind Ollama/vLLM can additionally support
guided/grammar-constrained decoding (vLLM `guided_json`), which is stronger
than forced tool-calling — an adapter instance constructed with
`guided_decoding=True` declares that so the gateway prefers it. The hosted
OpenAI API declares only native structured output.
"""

from __future__ import annotations

import json
from typing import Any

from ravana.runtime.providers.base import (
    Capability,
    NormalizedToolCall,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)


class OpenAICompatibleAdapter:
    def __init__(self, name: str = "openai", client: Any | None = None, guided_decoding: bool = False):
        self.name = name
        self._guided = guided_decoding
        self._explicit_client = client  # an injected client (tests) is used verbatim
        # Real clients are cached per (endpoint, api_key_ref) so one adapter
        # instance can serve agents pointing at different local/hosted
        # endpoints — the per-agent routing §1.4 promises. Caching a single
        # first-seen client (the earlier bug) silently sent every agent's
        # traffic to whichever endpoint happened to be resolved first.
        self._clients: dict[tuple[str | None, str | None], Any] = {}

    def capabilities(self, model: str) -> set[Capability]:
        caps = {Capability.NATIVE_STRUCTURED_OUTPUT}
        if self._guided:
            caps.add(Capability.GUIDED_DECODING)
        return caps

    def _resolve_client(self, request: ProviderRequest) -> Any:
        if self._explicit_client is not None:
            return self._explicit_client
        key = (request.endpoint, request.api_key_ref)
        if key not in self._clients:
            from openai import AsyncOpenAI

            # For a local runtime the api_key is often a placeholder; the
            # endpoint (base_url) routes to Ollama/vLLM. Real secrets
            # resolution (§8) would turn api_key_ref into a key — stubbed as
            # the ref here until secrets-manager wiring lands.
            self._clients[key] = AsyncOpenAI(
                base_url=request.endpoint, api_key=request.api_key_ref or "not-needed-for-local"
            )
        return self._clients[key]

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        client = self._resolve_client(request)

        messages = [{"role": "system", "content": request.system}, *_to_openai_messages(request.messages)]
        kwargs: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        # Guided path (§3.4 strongest tier): constrain the whole response to the
        # schema via response_format json_schema (OpenAI structured outputs;
        # local vLLM/Ollama honor the same field). The model then returns the
        # JSON as message content — no tool call — so the gateway reads it
        # from `text`, not from a submit_result tool.
        if request.output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": request.output_schema, "strict": True},
            }
        if request.tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in request.tools
            ]
        if request.force_tool:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": request.force_tool}}

        try:
            completion = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"openai-compatible completion failed: {exc}") from exc

        choice = completion.choices[0]
        msg = choice.message
        tool_calls: list[NormalizedToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            raw_args = tc.function.arguments
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            tool_calls.append(NormalizedToolCall(id=tc.id, tool=tc.function.name, arguments=arguments))

        usage = getattr(completion, "usage", None)
        return ProviderResponse(
            text=getattr(msg, "content", None),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            stop_reason=getattr(choice, "finish_reason", None),
        )


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Translate the gateway's normalized transcript to OpenAI chat format.
    The assistant's prior `tool_calls` message must be preserved (a tool
    result with no preceding assistant tool_calls is rejected), and results
    are role:"tool" keyed by tool_call_id."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.text}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.tool, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif m.role == "tool_result":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out
