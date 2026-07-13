"""The engine loop — ARCHITECTURE.md §3.1's corrected sequence:

    dispatch -> agent turn -> commit -> route-or-pause-or-fail -> resume -> terminate

Route-or-pause-or-fail (§3.1 step 5, §3.3) tries, in order: conditional
edges (respecting the §3.7 loop-iteration guard), then the node's HITL
trigger, then its `is_default` catch-all, then a hard fail-fast if none of
those applied to a non-terminal node. Resume (§3.1 step 6, corrected in
v0.14) dispatches a brand new `node_execution` attempt for the paused node,
not a bare re-route of stale output.

Phase 0a runs single-process (§10.1: no lease/CAS contention to speak of),
so this is a plain FIFO work queue rather than a distributed dispatch queue —
the DB rows and state machine are identical in shape to what Phase 1's
multi-worker version will use, only the "who claims the next item" mechanism
differs.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

from ravana.compiler.graph import TERMINAL, CompiledGraph
from ravana.engine.dod import DodResult, ProseVerdict, evaluate_dod
from ravana.engine.expr import apply_on_enter, eval_condition
from ravana.engine.state_merge import merge_delta
from ravana.observability.audit import write_audit
from ravana.observability.logging import log_event
from ravana.runtime.backoff import RetrySleep, backoff_delay
from ravana.runtime.base import (
    AgentRuntime,
    AgentTurnResult,
    LLMUsage,
    ProseJudgementError,
    TransientAgentError,
)
from ravana.runtime.idempotency import compute_idempotency_key
from ravana.runtime.secrets import ensure_secret_free, redact_secrets
from ravana.schema.models import DefinitionOfDone, HITLConfig
from ravana.schema.util import dumps, loads, new_id, now_iso

_GROUP_VAR_RE = re.compile(r"\$\{input\.([A-Za-z0-9_]+)\}")

# §3.6 backoff shape for the per-node transient retry. Module constants, not
# guards fields: §4's guards schema governs *budgets* (how many), not timing —
# making timing configurable is unwarranted surface until a real workflow
# needs it.
_NODE_RETRY_BASE_SECONDS = 1.0
_NODE_RETRY_CAP_SECONDS = 30.0


@dataclass
class _RunCtx:
    con: sqlite3.Connection
    graph: CompiledGraph
    run_id: str
    org_id: str
    workflow_id: str
    runtime: AgentRuntime
    queue: list[str] = field(default_factory=list)
    terminal_reached: bool = False
    failed: bool = False
    # §3.1 step 7: optional judge for *prose* DoD criteria. When set, prose
    # criteria are enforced at the Terminate gate; when None (the default), they
    # stay advisory. This is the engine-level injection point so a caller (CLI,
    # tests) can supply a real evaluated_by-agent verdict without the evaluator
    # itself living in the engine.
    dod_prose_verdict: ProseVerdict | None = None
    # §3.6: how a transient-retry backoff actually waits. Real runs sleep;
    # tests inject a recorder so the suite doesn't spend wall-clock time.
    retry_sleep: RetrySleep = asyncio.sleep

    def load_shared_state(self) -> dict[str, Any]:
        return loads(_get_run(self.con, self.run_id)["shared_state"], {})


def _agent_db_id(con: sqlite3.Connection, workflow_id: str, node_id: str) -> str | None:
    """workflow_node.agent_id is the *persisted* DB row id (§2.2) — the FK
    that message.sender_agent_id must actually reference, as opposed to
    AgentConfig.id, which is just the YAML-level string ('pm')."""
    row = con.execute(
        "SELECT agent_id FROM workflow_node WHERE workflow_id = ? AND id = ?", (workflow_id, node_id)
    ).fetchone()
    return row["agent_id"] if row else None


def _resolve_group(template: str, input_payload: dict[str, Any]) -> str:
    return _GROUP_VAR_RE.sub(lambda m: str(input_payload.get(m.group(1), "")), template)


def _get_run(con: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"run '{run_id}' not found")
    return row


def _next_sequence(con: sqlite3.Connection, run_id: str) -> int:
    row = con.execute(
        "SELECT COALESCE(MAX(sequence), 0) AS m FROM state_transition_log WHERE run_id = ?", (run_id,)
    ).fetchone()
    return row["m"] + 1


def _log_event(
    con: sqlite3.Connection,
    run_id: str,
    node_execution_id: str | None,
    event_type: str,
    *,
    from_node: str | None = None,
    to_node: str | None = None,
    condition_evaluated: str | None = None,
    result: bool | None = None,
    state_diff: dict[str, Any] | None = None,
    state_version_before: int | None = None,
    state_version_after: int | None = None,
) -> None:
    con.execute(
        """INSERT INTO state_transition_log
           (id, run_id, sequence, node_execution_id, event_type, from_node, to_node,
            condition_evaluated, result, state_diff, state_version_before, state_version_after, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            new_id(),
            run_id,
            _next_sequence(con, run_id),
            node_execution_id,
            event_type,
            from_node,
            to_node,
            condition_evaluated,
            None if result is None else int(result),
            dumps(state_diff),
            state_version_before,
            state_version_after,
            now_iso(),
        ),
    )


def _edge_fire_count(con: sqlite3.Connection, run_id: str, from_node: str, to_node: str) -> int:
    row = con.execute(
        """SELECT COUNT(*) AS c FROM state_transition_log
           WHERE run_id = ? AND event_type = 'ROUTE' AND from_node = ? AND to_node = ?""",
        (run_id, from_node, to_node),
    ).fetchone()
    return row["c"]


def _consecutive_failures(con: sqlite3.Connection, run_id: str, node_id: str) -> int:
    rows = con.execute(
        """SELECT status FROM node_execution WHERE run_id = ? AND node_id = ? ORDER BY attempt DESC""",
        (run_id, node_id),
    ).fetchall()
    count = 0
    for row in rows:
        if row["status"] == "FAILED":
            count += 1
        else:
            break
    return count


def _total_tokens(con: sqlite3.Connection, run_id: str) -> int:
    row = con.execute(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS t FROM node_execution WHERE run_id = ?", (run_id,)
    ).fetchone()
    return row["t"]


def _logical_visit_for_dispatch(con: sqlite3.Connection, run_id: str, node_id: str) -> str:
    """Stable identity across retries/HITL resume, fresh on graph re-entry."""
    latest = con.execute(
        """SELECT status, logical_visit_id FROM node_execution
           WHERE run_id = ? AND node_id = ? ORDER BY attempt DESC LIMIT 1""",
        (run_id, node_id),
    ).fetchone()
    if latest is not None and latest["status"] in ("FAILED", "WAITING_HUMAN"):
        existing = latest["logical_visit_id"]
        if existing:
            return existing
    return new_id()


async def start_run(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    runtime: AgentRuntime,
    org_id: str,
    workflow_id: str,
    triggered_by: str | None = None,
    input_payload: dict[str, Any] | None = None,
    dod_prose_verdict: ProseVerdict | None = None,
    retry_sleep: RetrySleep = asyncio.sleep,
) -> str:
    input_payload = input_payload or {}
    run_id = new_id()
    concurrency = graph.doc.spec.concurrency
    concurrency_group = _resolve_group(concurrency.group, input_payload) if concurrency else None

    status = "RUNNING"
    if concurrency and concurrency_group:
        active = con.execute(
            """SELECT id FROM run WHERE workflow_id = ? AND concurrency_group = ?
               AND status IN ('PENDING','RUNNING','WAITING_HUMAN')""",
            (workflow_id, concurrency_group),
        ).fetchall()
        if active:
            if concurrency.strategy == "queue":
                status = "PENDING"  # §3.7: held until the active run finishes; 0a has no scheduler to unblock it later
            elif concurrency.strategy == "cancel_previous":
                for row in active:
                    con.execute("UPDATE run SET status = 'CANCELLED', ended_at = ? WHERE id = ?", (now_iso(), row["id"]))
            # "allow": no restriction

    shared_state = dict(graph.doc.spec.state.initial)
    con.execute(
        """INSERT INTO run (id, org_id, workflow_id, workflow_version, status, current_nodes, shared_state,
                             state_version, concurrency_group, triggered_by, input_payload, started_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            org_id,
            workflow_id,
            graph.doc.metadata.version,
            status,
            dumps([]),
            dumps(shared_state),
            0,
            concurrency_group,
            triggered_by,
            dumps(input_payload),
            now_iso(),
        ),
    )
    con.commit()

    if status == "RUNNING":
        ctx = _RunCtx(
            con=con, graph=graph, run_id=run_id, org_id=org_id, workflow_id=workflow_id,
            runtime=runtime, queue=[graph.entry], dod_prose_verdict=dod_prose_verdict,
            retry_sleep=retry_sleep,
        )
        await _drain_queue(ctx)

    return run_id


def _join_arrivals(con: sqlite3.Connection, run_id: str, node_id: str) -> set[str]:
    """§3.8: the set of source nodes that have ROUTEd into `node_id` since its
    last dispatch. Derived entirely from state_transition_log — no in-memory
    bookkeeping — so it survives a HITL pause/resume across processes for
    free, the same way _edge_fire_count already does. "Since its last
    dispatch" is anchored on the node's own last COMMIT event: every
    successful dispatch commits at least once (§3.1 step 4), which is what
    consumes the arrivals that triggered it."""
    last_commit = con.execute(
        """SELECT COALESCE(MAX(sequence), 0) AS s FROM state_transition_log
           WHERE run_id = ? AND event_type = 'COMMIT' AND from_node = ?""",
        (run_id, node_id),
    ).fetchone()["s"]
    rows = con.execute(
        """SELECT DISTINCT from_node FROM state_transition_log
           WHERE run_id = ? AND event_type = 'ROUTE' AND to_node = ? AND sequence > ?""",
        (run_id, node_id, last_commit),
    ).fetchall()
    return {r["from_node"] for r in rows}


def _pending_joins(ctx: _RunCtx) -> list[tuple[str, set[str]]]:
    """Join nodes with at least one unconsumed arrival, excluding any already
    sitting in the queue."""
    pending = []
    for node_id in ctx.graph.join_all_nodes():
        if node_id in ctx.queue:
            continue
        arrivals = _join_arrivals(ctx.con, ctx.run_id, node_id)
        if arrivals:
            pending.append((node_id, arrivals))
    return pending


def _has_pending_hitl(ctx: _RunCtx) -> bool:
    return (
        ctx.con.execute(
            "SELECT 1 FROM hitl_request WHERE run_id = ? AND status = 'PENDING' LIMIT 1", (ctx.run_id,)
        ).fetchone()
        is not None
    )


async def _drain_queue(ctx: _RunCtx) -> None:
    while not ctx.failed:
        if ctx.queue:
            node_id = ctx.queue.pop(0)
            await _dispatch(ctx, node_id)
            continue
        # Queue is empty — quiescence check for held joins (§3.8): a join
        # node with partial arrivals fires now, because nothing is left that
        # could deliver the missing ones. But a PENDING HITL means answering
        # it may resume work that still delivers — so with HITL outstanding,
        # arrivals stay held and the resume's own drain re-evaluates them.
        if _has_pending_hitl(ctx):
            break
        stragglers = _pending_joins(ctx)
        if not stragglers:
            break
        for node_id, _arrivals in stragglers:
            ctx.queue.append(node_id)
    await _finalize_status(ctx)


async def _finalize_status(ctx: _RunCtx) -> None:
    if ctx.failed:
        return  # already set to FAILED at the point of failure
    run = _get_run(ctx.con, ctx.run_id)
    # Idempotent finalization: once a run has reached a terminal status, a
    # re-entry (a second drain, a stray resume) must NOT re-run the DoD gate.
    # Re-judging would call the evaluator again (double cost), append a second
    # DOD_EVALUATED event, and could flip a COMPLETED run to FAILED on a
    # non-deterministic verdict. A terminal state is final.
    if run["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
        return
    pending_hitl = ctx.con.execute(
        "SELECT 1 FROM hitl_request WHERE run_id = ? AND status = 'PENDING' LIMIT 1", (ctx.run_id,)
    ).fetchone()
    if pending_hitl:
        new_status = "WAITING_HUMAN"
    elif ctx.terminal_reached:
        # §3.1 step 7: reaching a terminal is necessary but not sufficient — the
        # run only COMPLETES if its definition_of_done is met, else it FAILs.
        # The gate fails closed on every KNOWN path; this outer net guarantees a
        # terminal status even on an UNFORESEEN raise, so a run can never strand
        # at RUNNING (the failure mode every DoD review round has probed for).
        try:
            new_status = await _dod_gate(ctx)
        except Exception as exc:  # noqa: BLE001 - last-resort: a terminal must resolve, never hang
            log_event("ERROR", f"run {ctx.run_id} DoD gate raised unexpectedly ({type(exc).__name__}); failing closed", run_id=ctx.run_id)
            new_status = "FAILED"
    else:
        # The queue drained with nothing pending and no __terminal__/implicit-terminal edge
        # ever fired — shouldn't happen given §3.1's fail-fast rule, but leave status as-is
        # rather than guess, so a genuine gap here is visible (stuck RUNNING) instead of
        # silently reported as either outcome.
        new_status = run["status"]
    ended_at = now_iso() if new_status in ("COMPLETED", "FAILED") else run["ended_at"]
    ctx.con.execute("UPDATE run SET status = ?, ended_at = ? WHERE id = ?", (new_status, ended_at, ctx.run_id))
    ctx.con.commit()


DodOutcome = Literal["met", "criteria_unmet", "evaluator_error", "cost_cap_exceeded"]


async def _dod_gate(ctx: _RunCtx) -> str:
    """§3.1 step 7: a run that reached a terminal COMPLETEs only if its
    definition_of_done is met, else FAILs. Expression criteria are enforced
    deterministically (pure, sync); prose criteria are judged by the injected
    `dod_prose_verdict`. The judgement's LLM usage is metered against
    guards.max_tokens_total (node tokens + this judgement's) before the run may
    COMPLETE, so a judgement can't slip the run past its hard cost cap. Records a
    DOD_EVALUATED event on every path with a durable `outcome` so a later reader
    can tell an unmet criterion from an evaluator that failed (provider down /
    repair exhausted). When no verdict is wired, prose stays advisory."""
    dod = ctx.graph.doc.spec.definition_of_done
    if dod is None:
        return "COMPLETED"
    guards = ctx.graph.doc.spec.graph.guards
    state = ctx.load_shared_state()
    result = evaluate_dod(dod, state)  # pure/sync — expressions only
    prose = result.prose_criteria
    if not (prose and ctx.dod_prose_verdict is not None):
        return _finish_dod(
            ctx, dod, result, outcome="met" if result.met else "criteria_unmet",
            status="COMPLETED" if result.met else "FAILED",
        )
    # A prose judgement runs an agent turn, and this gate executes in
    # _finalize_status — OUTSIDE _dispatch's failure boundary. The whole
    # obtain→meter→apply block is wrapped: a verdict that raises, OR returns
    # something malformed (None, a bad shape), must fail the run CLOSED with a
    # durable cause — never escape and strand it at RUNNING.
    try:
        judgement = await ctx.dod_prose_verdict(dod.evaluated_by, prose, state)
        # Engine-boundary revalidation: rebuild the usage THROUGH LLMUsage so a
        # judgement carrying a corrupted/duck-typed usage (float/NaN/negative/
        # bool) is rejected here too — the metered total is never trusted raw
        # from the runtime. A bad value raises → caught below → fail closed.
        usage = LLMUsage(judgement.usage.input_tokens, judgement.usage.output_tokens)
        # §3.6 cost cap: node tokens so far + this judgement's tokens. A
        # judgement that pushes the run past max_tokens_total FAILs it.
        over_cap = (
            guards.max_tokens_total is not None
            and _total_tokens(ctx.con, ctx.run_id) + usage.total > guards.max_tokens_total
        )
        if not over_cap:
            result.apply_prose_verdict(judgement.verdicts)
    except ProseJudgementError as exc:
        # The judgement failed AFTER spending tokens — record what it billed
        # (so the failure path isn't invisible to accounting) and fail closed,
        # persisting only the underlying cause's CLASS. Guard the usage: the
        # constructor enforces LLMUsage, but a post-construction mutation must
        # not AttributeError past this handler (which runs outside the broad
        # boundary) and strand the run — an unusable usage is simply not recorded.
        cause = type(exc.__cause__).__name__ if exc.__cause__ else "ProseJudgementError"
        spent = exc.usage if isinstance(exc.usage, LLMUsage) else None
        return _finish_dod(ctx, dod, result, outcome="evaluator_error", detail=cause, status="FAILED", usage=spent)
    except Exception as exc:  # noqa: BLE001 - fail closed; persist the error CLASS, never its (possibly secret-bearing) text
        return _finish_dod(ctx, dod, result, outcome="evaluator_error", detail=type(exc).__name__, status="FAILED")
    if over_cap:
        return _finish_dod(
            ctx, dod, result, outcome="cost_cap_exceeded", status="FAILED", usage=usage,
            detail=f"DoD judgement's {usage.total} tokens push the run over max_tokens_total ({guards.max_tokens_total})",
        )
    return _finish_dod(
        ctx, dod, result, outcome="met" if result.met else "criteria_unmet",
        status="COMPLETED" if result.met else "FAILED", usage=usage,
    )


def _finish_dod(
    ctx: _RunCtx,
    dod: DefinitionOfDone,
    result: DodResult,
    *,
    outcome: DodOutcome,
    status: str,
    detail: str | None = None,
    usage: LLMUsage | None = None,
) -> str:
    """Stage one DOD_EVALUATED event carrying the outcome (met / criteria_unmet
    / evaluator_error / cost_cap_exceeded), the per-criterion result, any
    judgement token usage, and — for a failure cause — a redacted detail, then
    return the run status. Deliberately does NOT commit: `_finalize_status`
    UPDATEs the run status and commits ONCE, so the terminal DoD event and the
    run's terminal status land atomically (a crash between them can't leave a
    RUNNING run carrying a terminal DoD event)."""
    state_diff: dict[str, Any] = {"outcome": outcome, **result.as_dict()}
    if usage is not None:
        state_diff["usage"] = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    if detail is not None:
        # §8 backstop: the cause is a fixed-shape error CLASS name (never the
        # exception text), but redact defensively before it lands in the event.
        state_diff["detail"] = redact_secrets(detail)
    _log_event(
        # `result` is the GATE outcome (did the run COMPLETE), not result.met —
        # an evaluator_error or cost_cap FAILs the run even though the criteria
        # it managed to evaluate were vacuously met.
        ctx.con, ctx.run_id, None, "DOD_EVALUATED",
        result=(status == "COMPLETED"), condition_evaluated="; ".join(dod.criteria), state_diff=state_diff,
    )
    if status == "FAILED":
        log_event(
            "ERROR",
            f"run {ctx.run_id} DoD gate FAILED ({outcome}): unmet={result.unmet}"
            + (f"; {redact_secrets(detail)}" if detail else ""),
            run_id=ctx.run_id,
        )
    return status


async def _dispatch(ctx: _RunCtx, node_id: str) -> None:
    con = ctx.con
    guards = ctx.graph.doc.spec.graph.guards

    consecutive_failures = _consecutive_failures(con, ctx.run_id, node_id)
    if consecutive_failures > guards.max_retries_per_node:
        _fail_run(ctx, node_id, f"node '{node_id}' exceeded max_retries_per_node ({guards.max_retries_per_node})")
        return

    total_dispatches = con.execute(
        "SELECT COUNT(*) AS c FROM node_execution WHERE run_id = ?", (ctx.run_id,)
    ).fetchone()["c"]
    if total_dispatches >= guards.max_total_steps:
        _fail_run(ctx, node_id, f"run exceeded guards.max_total_steps ({guards.max_total_steps})")
        return

    attempt = con.execute(
        "SELECT COALESCE(MAX(attempt), 0) + 1 AS n FROM node_execution WHERE run_id = ? AND node_id = ?",
        (ctx.run_id, node_id),
    ).fetchone()["n"]
    logical_visit_id = _logical_visit_for_dispatch(con, ctx.run_id, node_id)

    node_execution_id = new_id()
    con.execute(
        """INSERT INTO node_execution
           (id, run_id, node_id, attempt, logical_visit_id, status, started_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            node_execution_id,
            ctx.run_id,
            node_id,
            attempt,
            logical_visit_id,
            "RUNNING",
            now_iso(),
        ),
    )
    con.commit()

    agent = ctx.graph.agent_for_node(node_id)
    node = ctx.graph.node(node_id)
    shared_state = ctx.load_shared_state()

    if node.on_enter:
        on_enter_delta = apply_on_enter(node.on_enter, shared_state)
        shared_state = _commit_state(ctx, shared_state, node_execution_id, "COMMIT", from_node=node_id, delta=on_enter_delta)

    try:
        result = await ctx.runtime.run_turn(
            run_id=ctx.run_id,
            node_id=node_id,
            attempt=attempt,
            logical_visit_id=logical_visit_id,
            agent_id=agent.id,
            shared_state=shared_state,
        )
        ensure_secret_free(
            {
                "content": result.content,
                "structured_payload": result.structured_payload,
                "tool_calls": result.tool_calls,
            },
            context="agent turn output",
        )
    except TransientAgentError as exc:
        con.execute(
            "UPDATE node_execution SET status = 'FAILED', error = ?, finished_at = ? WHERE id = ?",
            (redact_secrets(str(exc)), now_iso(), node_execution_id),
        )
        con.commit()
        # §3.6: exponential backoff before the retry, keyed on the CONSECUTIVE
        # failure streak (1st failure ~base, doubling, capped) — the same
        # counter that bounds retries. NOT the node's lifetime `attempt`
        # number: a node re-entered by a §3.7 loop or a HITL resume has
        # SUCCEEDED rows inflating `attempt`, and its first transient failure
        # must back off ~base, not near the cap.
        failed_streak = consecutive_failures + 1  # +1 = the failure that just happened
        if failed_streak > guards.max_retries_per_node:
            # Budget already spent: the re-queued dispatch will trip the
            # max_retries_per_node guard without running a turn, so sleeping
            # here would only delay the inevitable FAILED verdict.
            ctx.queue.insert(0, node_id)
            return
        delay = backoff_delay(failed_streak, base=_NODE_RETRY_BASE_SECONDS, cap=_NODE_RETRY_CAP_SECONDS)
        log_event(
            "WARN",
            f"transient failure on node '{node_id}' ({failed_streak} consecutive), retrying in {delay:.1f}s: {exc}",
            run_id=ctx.run_id,
        )
        await ctx.retry_sleep(delay)
        ctx.queue.insert(0, node_id)
        return
    except Exception as exc:  # noqa: BLE001
        # Any NON-transient turn failure (a deferred/unknown toolkit surfaced by
        # the gateway, a submit_result-id collision, a genuinely unexpected bug)
        # is terminal for the run — but it must still land as a clean FAILED
        # state, never a process crash that leaves this node_execution stuck in
        # RUNNING. _fail_run flips the current attempt to FAILED and the run to
        # FAILED. (Recoverable tool failures don't reach here: the gateway feeds
        # those back into the turn as tool errors.)
        _fail_run(ctx, node_id, f"node '{node_id}' failed: {exc}")
        return

    # §3.4's within-turn guards and §9's cost cap are decided before the
    # durability transaction. A rejected turn is still recorded for audit,
    # but its delta never reaches shared state or routing.
    if result.tool_call_count > guards.max_tool_calls_per_turn:
        _fail_run(
            ctx,
            node_id,
            f"node '{node_id}' exceeded max_tool_calls_per_turn ({guards.max_tool_calls_per_turn})",
            result=result,
            logical_visit_id=logical_visit_id,
        )
        return
    if result.repair_count > guards.max_output_repairs:
        _fail_run(
            ctx,
            node_id,
            f"node '{node_id}' exceeded max_output_repairs ({guards.max_output_repairs})",
            result=result,
            logical_visit_id=logical_visit_id,
        )
        return
    projected_tokens = _total_tokens(con, ctx.run_id) + result.input_tokens + result.output_tokens
    if guards.max_tokens_total is not None and projected_tokens > guards.max_tokens_total:
        _fail_run(
            ctx,
            node_id,
            f"run exceeded guards.max_tokens_total ({guards.max_tokens_total})",
            result=result,
            logical_visit_id=logical_visit_id,
        )
        return

    try:
        shared_state = _commit_turn(
            ctx,
            node_execution_id,
            node_id,
            logical_visit_id,
            result,
        )
    except Exception as exc:  # noqa: BLE001 - transaction rolls back as a unit
        _fail_run(ctx, node_id, f"node '{node_id}' commit failed ({type(exc).__name__}): {exc}")
        return
    _route(ctx, node_id, node_execution_id, shared_state)


def _prepare_tool_calls(
    run_id: str,
    node_id: str,
    logical_visit_id: str,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fallback_ordinal, original in enumerate(tool_calls, start=1):
        tool_call = dict(original)
        visit_id = str(tool_call.setdefault("logical_visit_id", logical_visit_id))
        ordinal = int(tool_call.setdefault("tool_call_ordinal", fallback_ordinal))
        tool_call.setdefault(
            "idempotency_key",
            compute_idempotency_key(
                run_id,
                node_id,
                visit_id,
                ordinal,
                tool_call.get("tool", ""),
                tool_call.get("arguments", {}),
            ),
        )
        prepared.append(tool_call)
    return prepared


def _insert_turn_message(
    ctx: _RunCtx,
    node_id: str,
    result: AgentTurnResult,
    tool_calls: list[dict[str, Any]],
) -> None:
    ctx.con.execute(
        """INSERT INTO message
           (id, run_id, node_id, sender_agent_id, role, content,
            structured_payload, tool_calls, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            new_id(),
            ctx.run_id,
            node_id,
            _agent_db_id(ctx.con, ctx.workflow_id, node_id),
            "agent",
            result.content,
            dumps(result.structured_payload),
            dumps(tool_calls),
            now_iso(),
        ),
    )


def _commit_turn(
    ctx: _RunCtx,
    node_execution_id: str,
    node_id: str,
    logical_visit_id: str,
    result: AgentTurnResult,
) -> dict[str, Any]:
    """The successful-turn durability seam: all writes commit or none do."""
    con = ctx.con
    tool_calls = _prepare_tool_calls(
        ctx.run_id, node_id, logical_visit_id, result.tool_calls
    )
    for _ in range(3):
        run = _get_run(con, ctx.run_id)
        latest_state = loads(run["shared_state"])
        merged = merge_delta(
            latest_state, result.structured_payload, ctx.graph.doc.spec.state
        )
        version_before = run["state_version"]
        version_after = version_before + 1
        try:
            con.execute(
                """UPDATE node_execution
                   SET status = 'SUCCEEDED', finished_at = ?, input_tokens = ?,
                       output_tokens = ?, tool_call_count = ?, repair_count = ?
                   WHERE id = ?""",
                (
                    now_iso(),
                    result.input_tokens,
                    result.output_tokens,
                    result.tool_call_count,
                    result.repair_count,
                    node_execution_id,
                ),
            )
            _insert_turn_message(ctx, node_id, result, tool_calls)
            cursor = con.execute(
                """UPDATE run SET shared_state = ?, state_version = ?
                   WHERE id = ? AND state_version = ?""",
                (dumps(merged), version_after, ctx.run_id, version_before),
            )
            if cursor.rowcount == 0:
                con.rollback()
                continue
            _log_event(
                con,
                ctx.run_id,
                node_execution_id,
                "COMMIT",
                from_node=node_id,
                state_diff=result.structured_payload,
                state_version_before=version_before,
                state_version_after=version_after,
            )
            con.commit()
            return merged
        except Exception:
            con.rollback()
            raise
    raise RuntimeError("state_version CAS conflict after 3 retries")


def _commit_state(
    ctx: _RunCtx,
    shared_state: dict[str, Any],
    node_execution_id: str,
    event_type: str,
    *,
    from_node: str,
    delta: dict[str, Any],
) -> dict[str, Any]:
    """Applies `delta` via the merge policy and commits with the
    state_version CAS (§3.5) — single-writer here in 0a, so the CAS never
    actually conflicts, but the same code path Phase 1 uses under real
    contention."""
    con = ctx.con
    run = _get_run(con, ctx.run_id)
    merged = merge_delta(shared_state, delta, ctx.graph.doc.spec.state)
    version_before = run["state_version"]
    version_after = version_before + 1
    cursor = con.execute(
        "UPDATE run SET shared_state = ?, state_version = ? WHERE id = ? AND state_version = ?",
        (dumps(merged), version_after, ctx.run_id, version_before),
    )
    if cursor.rowcount == 0:  # pragma: no cover - unreachable single-writer in 0a, real CAS conflict path
        raise RuntimeError("state_version CAS conflict (unexpected in single-process Phase 0a)")
    _log_event(
        con, ctx.run_id, node_execution_id, event_type,
        from_node=from_node, state_diff=delta, state_version_before=version_before, state_version_after=version_after,
    )
    con.commit()
    return merged


def _edge_is_loop_capped(con: sqlite3.Connection, run_id: str, edge, guards) -> bool:
    """§3.7's loop guard is a hard cap independent of the edge's own
    condition — enforced by treating a capped-out edge as simply not
    matching, so it falls through to the next mechanism (another edge,
    HITL, default, fail-fast) exactly like a false condition would."""
    for target in edge.to:
        cap = guards.max_loop_iterations.get(f"{edge.from_}_to_{target}")
        if cap is not None and _edge_fire_count(con, run_id, edge.from_, target) >= cap:
            return True
    return False


def _route(ctx: _RunCtx, node_id: str, node_execution_id: str, shared_state: dict[str, Any]) -> None:
    con = ctx.con
    conditional, default = ctx.graph.outgoing(node_id)
    guards = ctx.graph.doc.spec.graph.guards

    for edge in conditional:
        if _edge_is_loop_capped(con, ctx.run_id, edge, guards):
            continue
        if edge.condition is None or eval_condition(edge.condition, shared_state):
            _fire_edge(ctx, edge.from_, edge, node_execution_id)
            return

    contract = ctx.graph.contract_for_node(node_id)
    if contract.hitl and contract.hitl.enabled and eval_condition(contract.hitl.trigger_condition, shared_state):
        _raise_hitl(ctx, node_id, node_execution_id, contract.hitl)
        return

    if default is not None:
        _fire_edge(ctx, node_id, default, node_execution_id)
        return

    if ctx.graph.has_outgoing_edges(node_id):
        _fail_run(ctx, node_id, f"no matching route from node '{node_id}', and no HITL or default edge configured")
        return

    # No outgoing edges at all: an implicit terminal, same as an explicit
    # `to: [__terminal__]` edge — this branch of the run is done. Previously
    # this fell through without setting terminal_reached, which left
    # run.status stuck at whatever it was (e.g. RUNNING) once the queue
    # drained instead of resolving to COMPLETED.
    _log_event(con, ctx.run_id, node_execution_id, "TERMINATE", from_node=node_id)
    con.commit()
    ctx.terminal_reached = True


def _fire_edge(ctx: _RunCtx, from_node: str, edge, node_execution_id: str) -> None:
    con = ctx.con
    for target in edge.to:
        _log_event(con, ctx.run_id, node_execution_id, "ROUTE", from_node=from_node, to_node=target, condition_evaluated=edge.condition, result=True)
        if target == TERMINAL:
            ctx.terminal_reached = True
        elif ctx.graph.node(target).join == "all":
            # §3.8: the ROUTE event above *is* the recorded arrival — dispatch
            # is deferred until arrivals cover every inbound source. Complete
            # joins promote immediately; partial ones wait for the remaining
            # sources or for quiescence (_drain_queue).
            con.commit()
            arrivals = _join_arrivals(con, ctx.run_id, target)
            required = ctx.graph.inbound_sources.get(target, set())
            if arrivals >= required and target not in ctx.queue:
                ctx.queue.append(target)
        else:
            ctx.queue.append(target)
    con.commit()


def _raise_hitl(ctx: _RunCtx, node_id: str, node_execution_id: str, hitl: HITLConfig) -> None:
    con = ctx.con
    con.execute("UPDATE node_execution SET status = 'WAITING_HUMAN', finished_at = ? WHERE id = ?", (now_iso(), node_execution_id))
    hitl_id = new_id()
    con.execute(
        """INSERT INTO hitl_request (id, run_id, node_id, question, assignee, status, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            hitl_id,
            ctx.run_id,
            node_id,
            hitl.prompt_template or "Human input required",
            hitl.assignee,
            "PENDING",
            now_iso(),
        ),
    )
    _log_event(con, ctx.run_id, node_execution_id, "HITL_RAISED", from_node=node_id)
    con.commit()
    log_event(
        "INFO",
        f"HITL raised on node '{node_id}'",
        run_id=ctx.run_id,
        node_execution_id=node_execution_id,
        assignee=hitl.assignee,
    )


def _fail_run(
    ctx: _RunCtx,
    node_id: str,
    error: str,
    *,
    result: AgentTurnResult | None = None,
    logical_visit_id: str | None = None,
) -> None:
    con = ctx.con
    # §8 backstop: error text may quote an SDK/HTTP exception that echoes an
    # injected credential — scrub known secret values before persisting.
    error = redact_secrets(error)
    node_execution_id_row = con.execute(
        """SELECT id, logical_visit_id FROM node_execution
           WHERE run_id = ? AND node_id = ? ORDER BY attempt DESC LIMIT 1""",
        (ctx.run_id, node_id),
    ).fetchone()
    ne_id = node_execution_id_row["id"] if node_execution_id_row else None
    if result is not None and ne_id is not None:
        con.execute(
            """UPDATE node_execution
               SET status = 'FAILED', error = ?, finished_at = ?, input_tokens = ?,
                   output_tokens = ?, tool_call_count = ?, repair_count = ?
               WHERE id = ?""",
            (
                error,
                now_iso(),
                result.input_tokens,
                result.output_tokens,
                result.tool_call_count,
                result.repair_count,
                ne_id,
            ),
        )
        visit_id = logical_visit_id or node_execution_id_row["logical_visit_id"] or new_id()
        tool_calls = _prepare_tool_calls(ctx.run_id, node_id, visit_id, result.tool_calls)
        _insert_turn_message(ctx, node_id, result, tool_calls)
    else:
        con.execute(
            """UPDATE node_execution SET status = 'FAILED', error = ?, finished_at = ?
               WHERE run_id = ? AND node_id = ? AND attempt = (
                   SELECT MAX(attempt) FROM node_execution WHERE run_id = ? AND node_id = ?
               )""",
            (error, now_iso(), ctx.run_id, node_id, ctx.run_id, node_id),
        )
    con.execute("UPDATE run SET status = 'FAILED', ended_at = ? WHERE id = ?", (now_iso(), ctx.run_id))
    _log_event(con, ctx.run_id, ne_id, "FAIL", from_node=node_id)
    con.commit()
    log_event("ERROR", error, run_id=ctx.run_id, node_execution_id=ne_id)
    ctx.failed = True
    ctx.queue.clear()


async def resume_hitl(
    con: sqlite3.Connection,
    graph: CompiledGraph,
    runtime: AgentRuntime,
    run_id: str,
    hitl_request_id: str,
    response: dict[str, Any],
    dod_prose_verdict: ProseVerdict | None = None,
    retry_sleep: RetrySleep = asyncio.sleep,
) -> None:
    """§3.1's corrected Resume: append the human's response to the message
    thread, then dispatch a brand-new node_execution attempt for the same
    node — NOT a bare re-route of stale output (that was the bug fixed in
    v0.14)."""
    hitl_row = con.execute("SELECT * FROM hitl_request WHERE id = ?", (hitl_request_id,)).fetchone()
    if hitl_row is None:
        raise KeyError(f"hitl_request '{hitl_request_id}' not found")
    if hitl_row["status"] != "PENDING":
        raise ValueError(f"hitl_request '{hitl_request_id}' is not PENDING (status={hitl_row['status']})")

    node_id = hitl_row["node_id"]
    con.execute(
        "UPDATE hitl_request SET status = 'ANSWERED', response = ?, responded_at = ? WHERE id = ?",
        (dumps(response), now_iso(), hitl_request_id),
    )
    con.execute(
        """INSERT INTO message (id, run_id, node_id, role, content, structured_payload, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (new_id(), run_id, node_id, "user", None, dumps(response), now_iso()),
    )
    _log_event(con, run_id, None, "HITL_RESOLVED", from_node=node_id)
    con.commit()
    write_audit(con, _get_run(con, run_id)["org_id"], "cli-user", "hitl.responded", "hitl_request", hitl_request_id, after=response)

    run_row = _get_run(con, run_id)
    ctx = _RunCtx(
        con=con, graph=graph, run_id=run_id, org_id=run_row["org_id"], workflow_id=run_row["workflow_id"],
        runtime=runtime, queue=[node_id], dod_prose_verdict=dod_prose_verdict, retry_sleep=retry_sleep,
    )
    await _drain_queue(ctx)
