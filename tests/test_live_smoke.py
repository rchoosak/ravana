"""Live-API smoke tests (§3.4) — OPT-IN, real network, real credentials.

Everything else in this suite runs offline against fake adapters. Nothing here
has ever hit a real provider, which is the single largest untested assumption
in Phase 0b: the adapters are wired to the real Anthropic/OpenAI SDKs but the
round-trip — does the provider actually honour a forced `submit_result` tool
and return parseable structured output — is unverified.

These fill that gap, but they are **skipped unless you opt in**, so the normal
`uv run pytest` stays offline, free, and fast:

    RAVANA_LIVE_SMOKE=1 ANTHROPIC_API_KEY=sk-... uv run pytest tests/test_live_smoke.py

Each provider is independently gated on its own key, so you can run just the
one you have. Models default to the cheapest tier and are env-overridable
(RAVANA_LIVE_ANTHROPIC_MODEL, RAVANA_LIVE_OPENAI_MODEL). For an
OpenAI-compatible LOCAL endpoint (Ollama/vLLM), set RAVANA_LIVE_OPENAI_ENDPOINT.

They cost a few tokens per run and send a trivial prompt ("reply with the word
ping") to the provider — no repo content leaves the machine.
"""

from __future__ import annotations

import os

import pytest

from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
from ravana.runtime.providers.base import (
    Capability,
    ProviderRequest,
    ProviderTarget,
    Tool,
    UserMessage,
)
from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("RAVANA_LIVE_SMOKE") != "1",
    reason="live-API smoke test; set RAVANA_LIVE_SMOKE=1 (+ a provider key) to run",
)

# The minimal structured-output contract: one required string field, submitted
# through the forced `submit_result` tool — the same shape the gateway forces.
_SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_SUBMIT_TOOL = Tool(
    name="submit_result",
    description="Submit your final answer.",
    input_schema=_SUBMIT_SCHEMA,
)
_PROMPT = "Reply by calling submit_result with answer set to the single word: ping"


def _require(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        pytest.skip(f"{env} not set")
    return value


async def test_anthropic_native_structured_output_round_trip():
    """Anthropic's expected mechanism is native forced tool-calling (§3.4). Prove
    the live model actually returns a submit_result call with the schema's field
    — not that our fake said it would."""
    _require("ANTHROPIC_API_KEY")
    model = os.environ.get("RAVANA_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    adapter = AnthropicAdapter()

    caps = adapter.capabilities(ProviderTarget(provider="anthropic", model=model))
    assert Capability.NATIVE_STRUCTURED_OUTPUT in caps  # the strategy the gateway will pick

    response = await adapter.complete(
        ProviderRequest(
            model=model,
            system="You are a test harness. Follow the instruction exactly.",
            messages=[UserMessage(text=_PROMPT)],
            tools=[_SUBMIT_TOOL],
            force_tool="submit_result",
            max_tokens=64,
        )
    )
    calls = [c for c in response.tool_calls if c.tool == "submit_result"]
    assert calls, f"provider returned no submit_result call: {response!r}"
    assert isinstance(calls[0].arguments.get("answer"), str)
    assert response.input_tokens > 0 and response.output_tokens > 0  # usage is real


async def test_openai_compatible_structured_output_round_trip():
    """OpenAI-compatible endpoint (hosted OpenAI, or a local Ollama/vLLM via
    RAVANA_LIVE_OPENAI_ENDPOINT). Exercises whichever mechanism its capabilities
    advertise, through a forced submit_result."""
    endpoint = os.environ.get("RAVANA_LIVE_OPENAI_ENDPOINT")
    if endpoint is None:
        _require("OPENAI_API_KEY")  # hosted OpenAI
    model = os.environ.get("RAVANA_LIVE_OPENAI_MODEL", "gpt-4o-mini")
    adapter = OpenAICompatibleAdapter(name="openai")

    response = await adapter.complete(
        ProviderRequest(
            model=model,
            system="You are a test harness. Follow the instruction exactly.",
            messages=[UserMessage(text=_PROMPT)],
            tools=[_SUBMIT_TOOL],
            force_tool="submit_result",
            output_schema=_SUBMIT_SCHEMA,
            endpoint=endpoint,
            max_tokens=64,
        )
    )
    calls = [c for c in response.tool_calls if c.tool == "submit_result"]
    assert calls, f"provider returned no submit_result call: {response!r}"
    assert isinstance(calls[0].arguments.get("answer"), str)
