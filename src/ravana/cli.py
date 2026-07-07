"""§1.5's CLI surface, Phase 0a scope: `ravana init`, `ravana workflow
validate`, `ravana run start`, `ravana run watch`, `ravana run hitl respond`.
No real LLM runtime exists yet (that's Phase 0b) — `run start`/`run watch`
take a `--mock-fixture` because MockAgentRuntime is the only backend Phase 0a
has.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import click

from ravana.compiler.graph import CompiledGraph, compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.compiler.validate import validate
from ravana.engine.loop import resume_hitl, start_run
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.db import init_db
from ravana.schema.loader import load_workflow_yaml

RAVANA_DIR = ".ravana"
TERMINAL_STATUSES = ("COMPLETED", "FAILED", "CANCELLED")


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
@click.option("--mock-fixture", required=True, type=click.Path(exists=True, dir_okay=False), help="Phase 0a has no real LLM runtime yet — a scripted response fixture is required")
def run_start(file: str, input_json: str, org: str, mock_fixture: str) -> None:
    """Persist FILE's workflow and start a Run against it."""
    con = _connect()
    doc = load_workflow_yaml(file)
    graph = compile_workflow(doc)
    workflow_id = get_or_create_workflow(con, graph, org_id=org, created_by="cli-user")
    runtime = MockAgentRuntime.from_yaml(mock_fixture)

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id=org, workflow_id=workflow_id, triggered_by="cli-user", input_payload=json.loads(input_json))
    )
    _print_run_status(con, run_id)


@run.command("watch")
@click.argument("run_id")
@click.option(
    "--mock-fixture", type=click.Path(exists=True, dir_okay=False),
    help="if given, blocks and interactively resolves HITL pauses instead of just printing status once and exiting",
)
def run_watch(run_id: str, mock_fixture: str | None) -> None:
    """Print RUN_ID's message trail. Without --mock-fixture, prints once and
    exits (the old behavior). With it, this is the "blocking terminal
    prompt" TASKS.md's Phase 0a asks for: it blocks, prompts interactively
    for each HITL pause, resumes, and keeps going until the run reaches a
    terminal status."""
    con = _connect()
    printed_message_ids: set[str] = set()
    runtime = MockAgentRuntime.from_yaml(mock_fixture) if mock_fixture else None
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
            if runtime is None:
                click.echo(f"  respond with: ravana run hitl respond {run_id} {hitl['id']} '<json response>'")
                click.echo("  (or re-run 'run watch' with --mock-fixture to answer interactively here)")
                return
            answer = click.prompt("  your answer")
            if graph is None:
                graph = _compiled_graph_for_run(con, row)
            asyncio.run(resume_hitl(con, graph, runtime, run_id, hitl["id"], {"answer": answer}))
        # loop again: the resume may have produced new messages, completed
        # the run, or raised another HITL pause elsewhere.


@run.group("hitl")
def run_hitl() -> None:
    pass


@run_hitl.command("respond")
@click.argument("run_id")
@click.argument("hitl_id")
@click.argument("response_json")
@click.option("--mock-fixture", required=True, type=click.Path(exists=True, dir_okay=False))
def run_hitl_respond(run_id: str, hitl_id: str, response_json: str, mock_fixture: str) -> None:
    """Answer a pending HITL request non-interactively, resuming that node
    (§3.1's corrected Resume). Kept alongside `run watch`'s interactive mode
    for scripting/automation."""
    con = _connect()
    run_row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if run_row is None:
        raise click.ClickException(f"run '{run_id}' not found")
    graph = _compiled_graph_for_run(con, run_row)
    runtime = MockAgentRuntime.from_yaml(mock_fixture)

    asyncio.run(resume_hitl(con, graph, runtime, run_id, hitl_id, json.loads(response_json)))
    _print_run_status(con, run_id)


def _print_run_status(con: sqlite3.Connection, run_id: str) -> None:
    row = con.execute("SELECT status, shared_state FROM run WHERE id = ?", (run_id,)).fetchone()
    click.echo(f"run {run_id}: {row['status']}")
    click.echo(f"shared_state: {row['shared_state']}")


if __name__ == "__main__":
    main()
