"""Tests for the second review round's fixes: the implicit-terminal bug, the
three previously-unenforced guards (max_tool_calls_per_turn,
max_output_repairs, max_tokens_total), idempotency-key wiring into
message.tool_calls, and workflow.concurrency's queue/cancel_previous
strategies."""

from __future__ import annotations

import asyncio

from ravana.compiler.graph import compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.engine.loop import start_run
from ravana.runtime.base import AgentTurnResult
from ravana.runtime.mock import MockAgentRuntime
from ravana.schema.models import WorkflowDoc
from ravana.schema.util import dumps, loads


def _single_node_workflow(edges: list[dict] | None = None) -> WorkflowDoc:
    raw = {
        "apiVersion": "ravana/v1",
        "kind": "Workflow",
        "metadata": {"name": "single-node-test", "version": 1},
        "spec": {
            "agents": [{"id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p"}],
            "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": edges or []},
        },
    }
    return WorkflowDoc.model_validate(raw)


def test_implicit_terminal_node_completes_the_run(con):
    """A node with zero outgoing edges is an implicit terminal (same as an
    explicit `to: [__terminal__]`) — the run must reach COMPLETED, not get
    stuck at whatever status it had when the queue drained."""
    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}}]})

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "COMPLETED"
    terminate_events = con.execute(
        "SELECT * FROM state_transition_log WHERE run_id = ? AND event_type = 'TERMINATE'", (run_id,)
    ).fetchall()
    assert len(terminate_events) == 1


def test_successful_turn_commit_is_atomic(con):
    """If the COMMIT event write fails, message/status/state must all roll back."""
    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    con.executescript(
        """
        CREATE TRIGGER abort_turn_commit
        BEFORE INSERT ON state_transition_log
        WHEN NEW.event_type = 'COMMIT'
        BEGIN
            SELECT RAISE(ABORT, 'injected commit failure');
        END;
        """
    )
    runtime = MockAgentRuntime(
        {"only": [{"structured_payload": {"should_not_persist": True}}]}
    )

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id)
    )

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    execution = con.execute(
        "SELECT * FROM node_execution WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert run["status"] == "FAILED"
    assert loads(run["shared_state"]) == {}
    assert run["state_version"] == 0
    assert execution["status"] == "FAILED"
    assert con.execute(
        "SELECT COUNT(*) AS c FROM message WHERE run_id = ?", (run_id,)
    ).fetchone()["c"] == 0
    assert con.execute(
        """SELECT COUNT(*) AS c FROM state_transition_log
           WHERE run_id = ? AND event_type = 'COMMIT'""",
        (run_id,),
    ).fetchone()["c"] == 0


def test_successful_turn_merges_against_state_committed_while_agent_runs(con):
    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")

    class ConcurrentCommitRuntime:
        async def run_turn(
            self,
            *,
            run_id: str,
            node_id: str,
            attempt: int,
            logical_visit_id: str,
            agent_id: str,
            shared_state: dict,
        ) -> AgentTurnResult:
            latest = loads(
                con.execute(
                    "SELECT shared_state FROM run WHERE id = ?", (run_id,)
                ).fetchone()["shared_state"]
            )
            latest["parallel_branch"] = "committed"
            con.execute(
                """UPDATE run SET shared_state = ?, state_version = state_version + 1
                   WHERE id = ?""",
                (dumps(latest), run_id),
            )
            con.commit()
            return AgentTurnResult(structured_payload={"agent_branch": "committed"})

        async def aclose(self) -> None:
            return None

    run_id = asyncio.run(
        start_run(
            con,
            graph,
            ConcurrentCommitRuntime(),
            org_id="test",
            workflow_id=workflow_id,
        )
    )

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert loads(run["shared_state"]) == {
        "parallel_branch": "committed",
        "agent_branch": "committed",
    }
    assert run["state_version"] == 2


def test_engine_fails_closed_before_persisting_secret_bearing_output(con):
    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime(
        {
            "only": [
                {
                    "content": "provider echoed sk-DO-NOT-PERSIST",
                    "structured_payload": {"nested": ["sk-DO-NOT-PERSIST"]},
                }
            ]
        }
    )

    run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id)
    )
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    error = con.execute(
        "SELECT error FROM node_execution WHERE run_id = ?", (run_id,)
    ).fetchone()["error"]
    assert run["status"] == "FAILED"
    assert "sk-DO-NOT-PERSIST" not in error
    assert con.execute(
        "SELECT COUNT(*) AS c FROM message WHERE run_id = ?", (run_id,)
    ).fetchone()["c"] == 0
    assert "sk-DO-NOT-PERSIST" not in run["shared_state"]
    assert con.execute(
        """SELECT COUNT(*) AS c FROM state_transition_log
           WHERE run_id = ? AND event_type = 'COMMIT'""",
        (run_id,),
    ).fetchone()["c"] == 0


def test_non_transient_turn_error_fails_the_run_cleanly(con):
    """A non-transient error from run_turn (e.g. the gateway surfacing a
    deferred toolkit, a submit_result-id collision, or an unexpected bug) must
    land as a clean FAILED run + FAILED node_execution — never a crash that
    strands the node in RUNNING. Only TransientAgentError is retried; everything
    else is terminal."""
    from ravana.runtime.toolkits.base import ToolkitError

    class ExplodingRuntime:
        async def run_turn(
            self, *, run_id, node_id, attempt, logical_visit_id, agent_id, shared_state
        ):
            raise ToolkitError("toolkit 'code_interpreter' is not executable in this build")

    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")

    # Must NOT raise out of start_run.
    run_id = asyncio.run(start_run(con, graph, ExplodingRuntime(), org_id="test", workflow_id=workflow_id))

    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    ne = con.execute("SELECT * FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()
    assert ne["status"] == "FAILED"  # not stuck at RUNNING
    assert "not executable" in ne["error"]


def test_max_tool_calls_per_turn_guard_fails_the_run(con):
    graph = compile_workflow(_single_node_workflow())
    graph.doc.spec.graph.guards.max_tool_calls_per_turn = 2
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}, "tool_call_count": 5}]})

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"

    ne = con.execute("SELECT * FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()
    assert ne["status"] == "FAILED"
    assert "max_tool_calls_per_turn" in ne["error"]


def test_max_output_repairs_guard_fails_the_run(con):
    graph = compile_workflow(_single_node_workflow())
    graph.doc.spec.graph.guards.max_output_repairs = 1
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}, "repair_count": 3}]})

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    ne = con.execute("SELECT * FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()
    assert "max_output_repairs" in ne["error"]


def test_max_tokens_total_guard_fails_the_run(con):
    graph = compile_workflow(_single_node_workflow())
    graph.doc.spec.graph.guards.max_tokens_total = 100
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}, "input_tokens": 80, "output_tokens": 80}]})

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))
    run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    assert run["status"] == "FAILED"
    ne = con.execute("SELECT * FROM node_execution WHERE run_id = ?", (run_id,)).fetchone()
    assert "max_tokens_total" in ne["error"]
    # The turn's own tokens were still recorded even though it then failed —
    # the guard check happens after recording, not instead of it (§9 audit trail).
    assert ne["input_tokens"] == 80
    assert ne["output_tokens"] == 80


def test_tool_calls_get_a_stable_idempotency_key(con):
    """§3.6: the key is computed once, at persistence time, and attached to
    message.tool_calls — this is the fix for the finding that
    compute_idempotency_key existed but had no runtime call site."""
    graph = compile_workflow(_single_node_workflow())
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    tool_call = {"tool": "git_push", "arguments": {"branch": "b", "message": "m"}}
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}, "tool_calls": [dict(tool_call)]}]})

    run_id = asyncio.run(start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id))

    msg = con.execute("SELECT * FROM message WHERE run_id = ? AND role = 'agent'", (run_id,)).fetchone()
    tool_calls = loads(msg["tool_calls"])
    assert len(tool_calls) == 1
    assert "idempotency_key" in tool_calls[0]
    assert len(tool_calls[0]["idempotency_key"]) == 64  # sha256 hex digest

    # Same tool + arguments, called again independently, must reproduce the
    # identical key (that's the whole point of the fix).
    from ravana.runtime.idempotency import compute_idempotency_key

    execution = con.execute(
        "SELECT logical_visit_id FROM node_execution WHERE run_id = ?", (run_id,)
    ).fetchone()
    expected = compute_idempotency_key(
        run_id,
        "only",
        execution["logical_visit_id"],
        1,
        "git_push",
        tool_call["arguments"],
    )
    assert tool_calls[0]["idempotency_key"] == expected


def test_concurrency_queue_holds_a_second_run_pending(con):
    # The first run must never reach a terminal status on its own (blocked on
    # HITL forever, via a node with hitl configured) so the second run's
    # concurrency check provably sees it as still "active", not racing a run
    # that already finished.
    blocking_graph = compile_workflow(
        WorkflowDoc.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": "single-node-test", "version": 1},
                "spec": {
                    "concurrency": {"group": "repo:${input.repository}", "strategy": "queue"},
                    "agents": [
                        {
                            "id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p",
                            "hitl": {"enabled": True, "trigger_condition": "true"},
                        }
                    ],
                    "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
                },
            }
        )
    )
    blocking_workflow_id = get_or_create_workflow(con, blocking_graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}}]})
    first_run_id = asyncio.run(
        start_run(con, blocking_graph, runtime, org_id="test", workflow_id=blocking_workflow_id, input_payload={"repository": "org/repo"})
    )
    first_run = con.execute("SELECT * FROM run WHERE id = ?", (first_run_id,)).fetchone()
    assert first_run["status"] == "WAITING_HUMAN"

    second_run_id = asyncio.run(
        start_run(con, blocking_graph, runtime, org_id="test", workflow_id=blocking_workflow_id, input_payload={"repository": "org/repo"})
    )
    second_run = con.execute("SELECT * FROM run WHERE id = ?", (second_run_id,)).fetchone()
    assert second_run["status"] == "PENDING"


def test_concurrency_cancel_previous_cancels_the_active_run(con):
    from ravana.schema.models import WorkflowDoc as _WD

    graph = compile_workflow(
        _WD.model_validate(
            {
                "apiVersion": "ravana/v1",
                "kind": "Workflow",
                "metadata": {"name": "single-node-test", "version": 1},
                "spec": {
                    "concurrency": {"group": "repo:${input.repository}", "strategy": "cancel_previous"},
                    "agents": [
                        {
                            "id": "a", "name": "A", "llm": {"provider": "anthropic", "model": "m"}, "system_prompt": "p",
                            "hitl": {"enabled": True, "trigger_condition": "true"},
                        }
                    ],
                    "graph": {"entry": "only", "nodes": [{"id": "only", "agent": "a"}], "edges": []},
                },
            }
        )
    )
    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    runtime = MockAgentRuntime({"only": [{"structured_payload": {}}]})

    first_run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, input_payload={"repository": "org/repo"})
    )
    second_run_id = asyncio.run(
        start_run(con, graph, runtime, org_id="test", workflow_id=workflow_id, input_payload={"repository": "org/repo"})
    )

    first_run = con.execute("SELECT * FROM run WHERE id = ?", (first_run_id,)).fetchone()
    second_run = con.execute("SELECT * FROM run WHERE id = ?", (second_run_id,)).fetchone()
    assert first_run["status"] == "CANCELLED"
    assert second_run["status"] == "WAITING_HUMAN"
