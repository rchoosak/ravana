"""§1.5's CLI surface: `ravana init`, `ravana workflow validate`, `ravana run
start`, `ravana run watch`, `ravana run hitl respond`.

The run commands take `--backend [mock|llm]`. `mock` drives the Phase-0a
MockAgentRuntime from a scripted `--mock-fixture` (no LLM). `llm` drives the
real LLM Gateway (§1.1) with the compiled graph's toolkits wired through
RavanaToolExecutor and provider adapters selected per the agents' `llm.provider`
— a run against real models/APIs, so it reads credentials from the environment.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import click

from ravana.compiler.graph import CompiledGraph, compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.compiler.validate import validate
from ravana.engine.dod import ProseVerdict
from ravana.engine.loop import TERMINAL_STATUSES, resume_hitl, start_run
from ravana.runtime.base import AgentRuntime, ProseJudge
from ravana.runtime.gateway import LLMGateway
from ravana.runtime.mock import MockAgentRuntime
from ravana.runtime.providers.anthropic_adapter import AnthropicAdapter
from ravana.runtime.providers.base import ProviderAdapter
from ravana.runtime.providers.openai_adapter import OpenAICompatibleAdapter
from ravana.runtime.secrets import EnvSecretResolver
from ravana.runtime.tool_executor import RavanaToolExecutor
from ravana.runtime.toolkits.registry import build_registry
from ravana.schema.db import init_db
from ravana.schema.loader import load_workflow_yaml

RAVANA_DIR = ".ravana"


def find_ravana_dir(start: Path | None = None) -> Path:
    """Walks up from `start` (default: cwd) looking for `.ravana/`, the same
    way git locates `.git/` — matches §10.1's "install onto a project path"
    layout."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        ravana_dir = candidate / RAVANA_DIR
        if ravana_dir.is_dir():
            return ravana_dir
    raise click.ClickException(f"no {RAVANA_DIR}/ found in this directory or any parent — run 'ravana init' first")


def _connect() -> sqlite3.Connection:
    db_path = find_ravana_dir() / "state.db"
    return init_db(db_path)


def _find_workflow_file_for_run(con: sqlite3.Connection, run_row: sqlite3.Row) -> Path:
    workflow_row = con.execute("SELECT * FROM workflow WHERE id = ?", (run_row["workflow_id"],)).fetchone()
    for candidate in (find_ravana_dir() / "workflows").glob("*.yaml"):
        doc = load_workflow_yaml(candidate)
        if doc.metadata.name == workflow_row["name"] and doc.metadata.version == workflow_row["version"]:
            return candidate
    raise click.ClickException("could not locate the workflow YAML for this run under .ravana/workflows/")


def _compiled_graph_for_run(con: sqlite3.Connection, run_row: sqlite3.Row) -> CompiledGraph:
    return compile_workflow(load_workflow_yaml(_find_workflow_file_for_run(con, run_row)))


def _providers_in_graph(graph: CompiledGraph) -> set[str]:
    """Every distinct provider an agent (or one of its fallbacks) names — so we
    build exactly the adapters the run needs and no more (constructing an
    Anthropic client, say, only when an agent actually uses Anthropic)."""
    providers: set[str] = set()
    for agent in graph.agents_by_id.values():
        providers.add(agent.llm.provider)
        providers.update(fb.provider for fb in agent.llm.fallback)
    return providers


def _make_adapter(provider: str) -> ProviderAdapter:
    """Map a provider name to its adapter. Anthropic has its own SDK; every
    other provider (openai, and local OpenAI-compatible runtimes like Ollama /
    vLLM reached via `llm.endpoint`) goes through the OpenAI-compatible adapter.
    guided_decoding stays off by default — not every OpenAI-compatible runtime
    honors guided_json, so the safe default is the forced-tool path; enabling
    it per-provider is a follow-up."""
    if provider == "anthropic":
        return AnthropicAdapter()
    return OpenAICompatibleAdapter(name=provider)


def _adapters_for_graph(graph: CompiledGraph) -> dict[str, ProviderAdapter]:
    return {provider: _make_adapter(provider) for provider in _providers_in_graph(graph)}


def _build_llm_gateway(con: sqlite3.Connection, graph: CompiledGraph) -> LLMGateway:
    # §8c: ONE resolver serves both credential kinds. Toolkit auth_refs
    # resolve through the registry's lazy providers (a secret is only read if
    # its toolkit is actually called), and the gateway resolves each agent's
    # llm.api_key_ref at dispatch — adapters receive resolved keys, never the
    # pointers. An agent with no api_key_ref falls back to the provider SDK's
    # own env var (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    resolver = EnvSecretResolver()
    handlers = build_registry(graph, resolver)
    executor = RavanaToolExecutor(con, handlers)
    return LLMGateway(graph, _adapters_for_graph(graph), tool_executor=executor, secret_resolver=resolver)


def _build_runtime(
    con: sqlite3.Connection, graph: CompiledGraph, backend: str, mock_fixture: str | None
) -> AgentRuntime:
    if backend == "llm":
        return _build_llm_gateway(con, graph)
    if not mock_fixture:
        raise click.ClickException("--backend mock requires --mock-fixture")
    return MockAgentRuntime.from_yaml(mock_fixture)


def _prose_verdict_for(runtime: AgentRuntime) -> ProseVerdict | None:
    """§3.1 step 7: wire the DoD gate's prose judge for any runtime that can
    judge prose (the `ProseJudge` capability). Detected structurally, not by
    concrete class, so a future runtime that gains `judge_prose` is picked up
    here with no CLI change. A runtime without the capability (the mock backend)
    leaves prose criteria advisory (unevaluated, non-gating), as in Phase 0a."""
    return runtime.judge_prose if isinstance(runtime, ProseJudge) else None


async def _start_with_cleanup(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    runtime: AgentRuntime,
    *,
    org_id: str,
    workflow_id: str,
    input_payload: dict[str, Any],
) -> str:
    try:
        return await start_run(
            con,
            graph,
            runtime,
            org_id=org_id,
            workflow_id=workflow_id,
            triggered_by="cli-user",
            input_payload=input_payload,
            dod_prose_verdict=_prose_verdict_for(runtime),
        )
    finally:
        await runtime.aclose()


async def _resume_with_cleanup(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    runtime: AgentRuntime,
    run_id: str,
    hitl_id: str,
    response: dict[str, Any],
) -> None:
    try:
        await resume_hitl(
            con, graph, runtime, run_id, hitl_id, response,
            dod_prose_verdict=_prose_verdict_for(runtime),
        )
    finally:
        await runtime.aclose()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("path", type=click.Path(file_okay=False), default=".")
def init(path: str) -> None:
    """Scaffold .ravana/ in PATH (§10.1)."""
    root = Path(path).resolve()
    ravana_dir = root / RAVANA_DIR
    if ravana_dir.exists():
        raise click.ClickException(f"{ravana_dir} already exists")
    (ravana_dir / "workflows").mkdir(parents=True)
    (ravana_dir / "runs").mkdir(parents=True)
    init_db(ravana_dir / "state.db")
    gitignore = ravana_dir / ".gitignore"
    gitignore.write_text("state.db\nruns/\n")
    click.echo(f"Initialized {ravana_dir}")


@main.group()
def workflow() -> None:
    pass


@workflow.command("validate")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def workflow_validate(file: str) -> None:
    """Compile FILE and report §3.3 validation issues."""
    doc = load_workflow_yaml(file)
    graph = compile_workflow(doc)  # raises CompileError with a clear message on hard structural errors
    issues = validate(graph)
    if not issues:
        click.echo("OK: no issues found")
        return
    for issue in issues:
        click.echo(f"{issue.severity.upper()}: {issue.message}")
    if any(i.severity == "error" for i in issues):
        raise SystemExit(1)


@main.group()
def run() -> None:
    pass


@run.command("start")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "input_json", default="{}", help="JSON input payload")
@click.option("--org", default="local", help="org_id (Phase 0a: no real multi-tenancy)")
@click.option("--backend", type=click.Choice(["mock", "llm"]), default="mock", help="mock = scripted --mock-fixture; llm = real LLM Gateway (§1.1) with wired toolkits")
@click.option("--mock-fixture", type=click.Path(exists=True, dir_okay=False), help="required for --backend mock: a scripted response fixture")
def run_start(file: str, input_json: str, org: str, backend: str, mock_fixture: str | None) -> None:
    """Persist FILE's workflow and start a Run against it."""
    con = _connect()
    doc = load_workflow_yaml(file)
    graph = compile_workflow(doc)
    workflow_id = get_or_create_workflow(con, graph, org_id=org, created_by="cli-user")
    input_payload = json.loads(input_json)
    runtime = _build_runtime(con, graph, backend, mock_fixture)

    run_id = asyncio.run(
        _start_with_cleanup(
            con,
            graph,
            runtime,
            org_id=org,
            workflow_id=workflow_id,
            input_payload=input_payload,
        )
    )
    _print_run_status(con, run_id)


@run.command("watch")
@click.argument("run_id")
@click.option("--backend", type=click.Choice(["mock", "llm"]), default="mock", help="runtime used to resume HITL pauses: mock (--mock-fixture) or llm (real Gateway)")
@click.option(
    "--mock-fixture", type=click.Path(exists=True, dir_okay=False),
    help="with --backend mock, blocks and interactively resolves HITL pauses instead of just printing status once and exiting",
)
def run_watch(run_id: str, backend: str, mock_fixture: str | None) -> None:
    """Print RUN_ID's message trail. In a non-interactive setup (--backend mock
    with no --mock-fixture), prints once and exits. With --backend llm, or
    --backend mock plus a fixture, this blocks: it prompts interactively for
    each HITL pause, resumes via the chosen runtime, and keeps going until the
    run reaches a terminal status."""
    con = _connect()
    printed_message_ids: set[str] = set()
    interactive = backend == "llm" or bool(mock_fixture)
    runtime: AgentRuntime | None = None
    graph: CompiledGraph | None = None

    while True:
        row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise click.ClickException(f"run '{run_id}' not found")

        for msg in con.execute("SELECT * FROM message WHERE run_id = ? ORDER BY created_at", (run_id,)):
            if msg["id"] in printed_message_ids:
                continue
            printed_message_ids.add(msg["id"])
            click.echo(f"[{msg['node_id']}] {msg['role']}: {msg['content'] or msg['structured_payload']}")

        if row["status"] in TERMINAL_STATUSES:
            _print_run_status(con, run_id)
            return

        if row["status"] != "WAITING_HUMAN":
            # PENDING (blocked on §3.7 concurrency) or RUNNING with nothing
            # left to print — Phase 0a's engine is synchronous per dispatch
            # call, so there's nothing to poll for here without a real
            # scheduler (Phase 1); report status and stop rather than spin.
            _print_run_status(con, run_id)
            return

        pending = con.execute("SELECT * FROM hitl_request WHERE run_id = ? AND status = 'PENDING'", (run_id,)).fetchall()
        if not pending:  # pragma: no cover - WAITING_HUMAN with no PENDING row would itself be a bug
            _print_run_status(con, run_id)
            return

        for hitl in pending:
            click.echo(f"\nPENDING HITL [{hitl['id']}] on node '{hitl['node_id']}': {hitl['question']}")
            if not interactive:
                click.echo(f"  respond with: ravana run hitl respond {run_id} {hitl['id']} '<json response>'")
                click.echo("  (or re-run 'run watch' with --backend llm / --mock-fixture to answer interactively here)")
                return
            answer = click.prompt("  your answer")
            if graph is None:
                graph = _compiled_graph_for_run(con, row)
            if runtime is None:
                runtime = _build_runtime(con, graph, backend, mock_fixture)
            asyncio.run(
                _resume_with_cleanup(
                    con, graph, runtime, run_id, hitl["id"], {"answer": answer}
                )
            )
            runtime = None
        # loop again: the resume may have produced new messages, completed
        # the run, or raised another HITL pause elsewhere.


@run.group("hitl")
def run_hitl() -> None:
    pass


@run_hitl.command("respond")
@click.argument("run_id")
@click.argument("hitl_id")
@click.argument("response_json")
@click.option("--backend", type=click.Choice(["mock", "llm"]), default="mock", help="runtime used to resume the paused node")
@click.option("--mock-fixture", type=click.Path(exists=True, dir_okay=False), help="required for --backend mock")
def run_hitl_respond(run_id: str, hitl_id: str, response_json: str, backend: str, mock_fixture: str | None) -> None:
    """Answer a pending HITL request non-interactively, resuming that node
    (§3.1's corrected Resume). Kept alongside `run watch`'s interactive mode
    for scripting/automation."""
    con = _connect()
    run_row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if run_row is None:
        raise click.ClickException(f"run '{run_id}' not found")
    graph = _compiled_graph_for_run(con, run_row)
    response = json.loads(response_json)
    runtime = _build_runtime(con, graph, backend, mock_fixture)

    asyncio.run(
        _resume_with_cleanup(
            con, graph, runtime, run_id, hitl_id, response
        )
    )
    _print_run_status(con, run_id)


def _print_run_status(con: sqlite3.Connection, run_id: str) -> None:
    row = con.execute("SELECT status, shared_state FROM run WHERE id = ?", (run_id,)).fetchone()
    click.echo(f"run {run_id}: {row['status']}")
    click.echo(f"shared_state: {row['shared_state']}")


if __name__ == "__main__":
    main()
