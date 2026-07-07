"""`ravana workflow validate` (§7) — advisory checks beyond what compile_workflow
already hard-errors on. These are warnings: a workflow with them can still
run, but they're the kind of authoring mistake §3.3 calls out specifically
(the qa_test dead-end in ARCHITECTURE.md §4 was exactly this class of bug).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ravana.compiler.graph import TERMINAL, CompiledGraph

Severity = Literal["error", "warning"]


@dataclass
class Issue:
    severity: Severity
    message: str


def validate(graph: CompiledGraph) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(_unreachable_nodes(graph))
    issues.extend(_missing_safety_net(graph))
    issues.extend(_broadcast_merge_conflicts(graph))
    return issues


def _reachable(graph: CompiledGraph) -> set[str]:
    seen: set[str] = set()
    stack = [graph.entry]
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id == TERMINAL:
            continue
        seen.add(node_id)
        conditional, default = graph.outgoing(node_id)
        for edge in conditional:
            stack.extend(t for t in edge.to if t != TERMINAL)
        if default:
            stack.extend(t for t in default.to if t != TERMINAL)
    return seen


def _unreachable_nodes(graph: CompiledGraph) -> list[Issue]:
    reachable = _reachable(graph)
    unreachable = set(graph.nodes_by_id) - reachable
    return [
        Issue("warning", f"node '{node_id}' is unreachable from entry '{graph.entry}'")
        for node_id in sorted(unreachable)
    ]


def _missing_safety_net(graph: CompiledGraph) -> list[Issue]:
    """§3.3: a non-terminal node whose *only* outgoing edges are conditional,
    with no is_default catch-all and no HITL configured on its agent, can hit
    §3.1's fail-fast dead-end if condition coverage turns out to be
    incomplete. This is exactly the bug the qa_test example had.
    """
    issues = []
    for node_id, node in graph.nodes_by_id.items():
        conditional, default = graph.outgoing(node_id)
        if not conditional and default is None:
            continue  # no outgoing edges at all = intentionally terminal, not a warning case
        if default is not None:
            continue  # has a catch-all, covered
        has_unconditional = any(e.condition is None for e in conditional)
        if has_unconditional:
            continue  # an unconditional edge always fires, covered
        agent = graph.agents_by_id.get(node.agent) if node.agent else None
        if agent is not None and agent.hitl is not None and agent.hitl.enabled:
            continue  # HITL can catch the no-match case, covered
        issues.append(
            Issue(
                "warning",
                f"node '{node_id}' has only conditional edges and no is_default/HITL safety net — "
                "if none of its conditions match at runtime, the run fails fast (§3.1) with no diagnosis "
                "beyond 'no matching route'",
            )
        )
    return issues


def _broadcast_merge_conflicts(graph: CompiledGraph) -> list[Issue]:
    """§3.5: two nodes reachable via the same broadcast edge that both declare
    an 'overwrite' key in their output_schema risk one silently clobbering
    the other's write. This is a heuristic — output_schema is the only
    static signal we have for "what this node writes", and it's optional —
    so this under-reports rather than false-positives on nodes that don't
    declare output_schema at all.
    """
    issues = []
    state_fields = graph.doc.spec.state.fields
    all_edges = [e for edges in graph.conditional_edges_by_source.values() for e in edges]
    all_edges += list(graph.default_edge_by_source.values())

    for edge in all_edges:
        targets = [t for t in edge.to if t != TERMINAL]
        if len(targets) < 2:
            continue
        overwrite_keys: dict[str, list[str]] = {}
        for target in targets:
            node = graph.nodes_by_id.get(target)
            if node is None or node.agent is None:
                continue
            agent = graph.agents_by_id.get(node.agent)
            if agent is None or agent.output_schema is None:
                continue
            for key in agent.output_schema.get("properties", {}):
                field_schema = state_fields.get(key)
                if field_schema is not None and field_schema.merge == "overwrite":
                    overwrite_keys.setdefault(key, []).append(target)

        for key, writers in overwrite_keys.items():
            if len(writers) > 1:
                issues.append(
                    Issue(
                        "error",
                        f"broadcast from '{edge.from_}' fans out to {writers}, which both declare "
                        f"'{key}' in output_schema with merge policy 'overwrite' — one will silently "
                        f"clobber the other's write; use a non-overwrite merge policy for '{key}' (§3.5)",
                    )
                )
    return issues
