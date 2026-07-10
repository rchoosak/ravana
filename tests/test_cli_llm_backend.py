"""`ravana run ... --backend llm` wiring. These cover the pure selection/build
helpers — which providers become which adapters, and that the gateway is
constructed with the graph's toolkits — without any network call (constructing
an adapter builds a client object but makes no request). The end-to-end run
against real models/APIs is intentionally not exercised here (it needs real
credentials); that remains a manual smoke test, tracked in TASKS.md.
"""

from __future__ import annotations

import click
import pytest

from ravana.cli import (
    _adapters_for_graph,
    _build_llm_gateway,
    _build_runtime,
    _make_adapter,
    _providers_in_graph,
)
from ravana.runtime.gateway import LLMGateway
from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter


def test_providers_in_graph_collects_agent_and_fallback_providers(sdlc_graph):
    # SDLC: pm/sa anthropic, dev local (fallback anthropic), qa openai.
    assert _providers_in_graph(sdlc_graph) == {"anthropic", "local", "openai"}


def test_make_adapter_maps_non_anthropic_to_openai_compatible():
    local = _make_adapter("local")
    assert isinstance(local, OpenAICompatibleAdapter)
    assert local.name == "local"  # name preserved so the gateway keys match llm.provider
    assert isinstance(_make_adapter("openai"), OpenAICompatibleAdapter)


def test_make_adapter_maps_anthropic_to_anthropic_adapter(monkeypatch):
    # AnthropicAdapter() builds a real SDK client, which needs *a* key present
    # (no network call happens at construction).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-not-used")
    assert isinstance(_make_adapter("anthropic"), AnthropicAdapter)


def test_adapters_for_graph_covers_every_provider(sdlc_graph, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-not-used")
    adapters = _adapters_for_graph(sdlc_graph)
    assert set(adapters) == {"anthropic", "local", "openai"}


def test_build_llm_gateway_wires_graph_toolkits(sdlc_graph, con, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-not-used")
    gateway = _build_llm_gateway(con, sdlc_graph)
    assert isinstance(gateway, LLMGateway)
    # The gateway's executor surfaces an agent's declared toolkits as tools.
    specs = gateway._tools.tools_for(["git_connector"])
    assert [t.name for t in specs] == ["git_connector"]


def test_build_runtime_mock_requires_fixture(sdlc_graph, con):
    with pytest.raises(click.ClickException, match="requires --mock-fixture"):
        _build_runtime(con, sdlc_graph, "mock", None)
