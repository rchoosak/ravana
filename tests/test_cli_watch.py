"""`ravana run watch --mock-fixture` must actually block and interactively
resolve HITL pauses (TASKS.md's "Blocking terminal prompt" checklist item),
not just print status once and exit. Uses Click's CliRunner with simulated
stdin for the interactive prompt."""

from __future__ import annotations

import json
import re

from click.testing import CliRunner

from ravana.cli import main
from tests.conftest import SDLC_FIXTURE, SDLC_WORKFLOW

_RUN_LINE_RE = re.compile(r"run ([0-9a-f-]{36}): \w+")


def _extract_run_id(output: str) -> str:
    # CliRunner concatenates stdout and stderr rather than truly interleaving
    # them by call order (our structured logs go to stderr, §9), so the
    # "run <id>: STATUS" line isn't reliably at a fixed position — search
    # for it instead of indexing into splitlines().
    match = _RUN_LINE_RE.search(output)
    assert match, f"no 'run <id>: STATUS' line found in output:\n{output}"
    return match.group(1)


def test_run_watch_blocks_and_resolves_hitl_interactively(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    assert runner.invoke(main, ["init", "."]).exit_code == 0

    workflows_dir = tmp_path / ".ravana" / "workflows"
    (workflows_dir / SDLC_WORKFLOW.name).write_text(SDLC_WORKFLOW.read_text())
    workflow_path = str(workflows_dir / SDLC_WORKFLOW.name)

    start_result = runner.invoke(
        main,
        [
            "run", "start", workflow_path,
            "--input", json.dumps({"requirement": "build X", "repository": "org/repo"}),
            "--mock-fixture", str(SDLC_FIXTURE),
        ],
    )
    assert start_result.exit_code == 0, start_result.output
    run_id = _extract_run_id(start_result.output)
    assert "WAITING_HUMAN" in start_result.output

    # Without --mock-fixture: must NOT block, just print instructions and return.
    passive_watch = runner.invoke(main, ["run", "watch", run_id])
    assert passive_watch.exit_code == 0
    assert "respond with: ravana run hitl respond" in passive_watch.output

    # With --mock-fixture and simulated stdin: must block, prompt, resume,
    # and keep going until the run reaches a terminal status.
    interactive_watch = runner.invoke(
        main, ["run", "watch", run_id, "--mock-fixture", str(SDLC_FIXTURE)], input="clarified, please proceed\n"
    )
    assert interactive_watch.exit_code == 0, interactive_watch.output
    assert "your answer" in interactive_watch.output
    assert "run " + run_id + ": COMPLETED" in interactive_watch.output


def test_run_watch_without_fixture_on_a_completed_run_just_prints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(main, ["init", "."]).exit_code == 0

    # A trivial single-node workflow with no HITL reaches COMPLETED in one shot.
    workflow_yaml = """
apiVersion: ravana/v1
kind: Workflow
metadata: {name: trivial, version: 1}
spec:
  agents: [{id: a, name: A, llm: {provider: anthropic, model: m}, system_prompt: p}]
  graph: {entry: only, nodes: [{id: only, agent: a}], edges: []}
"""
    workflow_path = tmp_path / ".ravana" / "workflows" / "trivial.yaml"
    workflow_path.write_text(workflow_yaml)
    fixture_path = tmp_path / "fixture.yaml"
    fixture_path.write_text("responses:\n  only:\n    - structured_payload: {}\n")

    start_result = runner.invoke(main, ["run", "start", str(workflow_path), "--mock-fixture", str(fixture_path)])
    assert start_result.exit_code == 0, start_result.output
    run_id = _extract_run_id(start_result.output)
    assert "COMPLETED" in start_result.output

    watch_result = runner.invoke(main, ["run", "watch", run_id])
    assert watch_result.exit_code == 0
    assert f"run {run_id}: COMPLETED" in watch_result.output
