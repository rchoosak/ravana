from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ravana.compiler.graph import CompiledGraph, compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.db import init_db
from ravana.schema.loader import load_workflow_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SDLC_WORKFLOW = REPO_ROOT / "examples" / "workflows" / "software-development-team.yaml"
SDLC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sdlc_mock_responses.yaml"


class RecordingSleep:
    """asyncio.sleep-shaped fake for §3.6 backoff: records each requested
    delay instead of actually waiting, so retry tests stay instant."""

    def __init__(self):
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@pytest.fixture
def con() -> sqlite3.Connection:
    return init_db(":memory:")


@pytest.fixture
def sdlc_graph() -> CompiledGraph:
    return compile_workflow(load_workflow_yaml(SDLC_WORKFLOW))


@pytest.fixture
def sdlc_workflow_id(con: sqlite3.Connection, sdlc_graph: CompiledGraph) -> str:
    return get_or_create_workflow(con, sdlc_graph, org_id="test-org", created_by="test")


@pytest.fixture
def sdlc_runtime() -> MockAgentRuntime:
    return MockAgentRuntime.from_yaml(SDLC_FIXTURE)
