"""The compiled, resolved-by-id form of a WorkflowDoc — what the engine loop
(ravana.engine.loop) actually walks. Building this is separate from
validating it (ravana.compiler.validate) so the CLI can show validation
issues for a graph that still fails to compile cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ravana.schema.models import (
    AgentConfig,
    GraphEdge,
    GraphNode,
    HITLConfig,
    SkillConfig,
    ToolkitConfig,
    WorkflowDoc,
)

TERMINAL = "__terminal__"


class CompileError(Exception):
    """Raised when a workflow has structural errors severe enough that the
    engine cannot run it at all (as opposed to a `validate` warning, which is
    advisory)."""


@dataclass(frozen=True)
class NodeExecutionContract:
    """Task-specific behavior resolved at the node seam.

    AgentConfig remains the reusable persona and capability ceiling. A node
    can narrow tool grants and override output/HITL behavior without cloning
    the persona; legacy workflows inherit the agent defaults.
    """

    toolkits: tuple[str, ...]
    hitl: HITLConfig | None
    output_schema: dict[str, Any] | None


@dataclass
class CompiledGraph:
    doc: WorkflowDoc
    nodes_by_id: dict[str, GraphNode]
    agents_by_id: dict[str, AgentConfig]
    toolkits_by_id: dict[str, ToolkitConfig]
    skills_by_id: dict[str, SkillConfig]
    # Per source node: conditional edges (priority desc, is_default excluded)
    # and at most one default edge — this ordering IS the §3.1/§3.3 routing
    # priority: conditional edges are tried first, the default edge last.
    conditional_edges_by_source: dict[str, list[GraphEdge]] = field(default_factory=dict)
    default_edge_by_source: dict[str, GraphEdge] = field(default_factory=dict)
    # Per target node: the distinct set of source nodes with any edge into it.
    # This is what a `join: all` node's arrival set is checked against (§3.8).
    inbound_sources: dict[str, set[str]] = field(default_factory=dict)

    @property
    def entry(self) -> str:
        return self.doc.spec.graph.entry

    def join_all_nodes(self) -> list[str]:
        return [n.id for n in self.doc.spec.graph.nodes if n.join == "all"]

    def node(self, node_id: str) -> GraphNode:
        return self.nodes_by_id[node_id]

    def agent_for_node(self, node_id: str) -> AgentConfig:
        node = self.node(node_id)
        if node.agent is None:
            raise CompileError(f"node '{node_id}' has no agent (sub-workflow nodes aren't runnable in Phase 0a)")
        return self.agents_by_id[node.agent]

    def contract_for_node(self, node_id: str) -> NodeExecutionContract:
        node = self.node(node_id)
        agent = self.agent_for_node(node_id)
        explicit = node.model_fields_set
        return NodeExecutionContract(
            toolkits=tuple(agent.toolkits if node.toolkits is None else node.toolkits),
            hitl=agent.hitl if "hitl" not in explicit else node.hitl,
            output_schema=agent.output_schema if "output_schema" not in explicit else node.output_schema,
        )

    def outgoing(self, node_id: str) -> tuple[list[GraphEdge], GraphEdge | None]:
        """(conditional edges in priority order, the default edge or None)."""
        return self.conditional_edges_by_source.get(node_id, []), self.default_edge_by_source.get(node_id)

    def has_outgoing_edges(self, node_id: str) -> bool:
        conditional, default = self.outgoing(node_id)
        return bool(conditional) or default is not None


def compile_workflow(doc: WorkflowDoc) -> CompiledGraph:
    spec = doc.spec
    nodes_by_id = {n.id: n for n in spec.graph.nodes}
    if len(nodes_by_id) != len(spec.graph.nodes):
        raise CompileError("duplicate node id in graph.nodes")

    agents_by_id = {a.id: a for a in spec.agents}
    if len(agents_by_id) != len(spec.agents):
        raise CompileError("duplicate agent id")

    toolkits_by_id = {t.id: t for t in spec.toolkits}
    skills_by_id = {s.id: s for s in spec.skills}

    if doc.spec.graph.entry not in nodes_by_id:
        raise CompileError(f"entry node '{doc.spec.graph.entry}' is not defined in graph.nodes")

    conditional_edges_by_source: dict[str, list[GraphEdge]] = {}
    default_edge_by_source: dict[str, GraphEdge] = {}

    inbound_sources: dict[str, set[str]] = {}
    for edge in spec.graph.edges:
        if edge.from_ not in nodes_by_id:
            raise CompileError(f"edge references unknown source node '{edge.from_}'")
        for target in edge.to:
            if target != TERMINAL and target not in nodes_by_id:
                raise CompileError(f"edge from '{edge.from_}' targets unknown node '{target}'")
            if target != TERMINAL:
                inbound_sources.setdefault(target, set()).add(edge.from_)

        if edge.is_default:
            if edge.from_ in default_edge_by_source:
                raise CompileError(f"node '{edge.from_}' has more than one is_default edge")
            default_edge_by_source[edge.from_] = edge
        else:
            conditional_edges_by_source.setdefault(edge.from_, []).append(edge)

    for edges in conditional_edges_by_source.values():
        edges.sort(key=lambda e: e.priority, reverse=True)

    for node in spec.graph.nodes:
        if node.agent is not None and node.agent not in agents_by_id:
            raise CompileError(f"node '{node.id}' references unknown agent '{node.agent}'")
        if node.agent is not None and node.toolkits is not None:
            agent_toolkits = set(agents_by_id[node.agent].toolkits)
            unauthorized = sorted(set(node.toolkits) - agent_toolkits)
            if unauthorized:
                raise CompileError(
                    f"node '{node.id}' grants toolkits {unauthorized} outside agent "
                    f"'{node.agent}' allow-list"
                )
            unknown = sorted(set(node.toolkits) - set(toolkits_by_id))
            if unknown:
                raise CompileError(f"node '{node.id}' references unknown toolkit(s) {unknown}")

    for agent in spec.agents:
        for tk in agent.toolkits:
            if tk not in toolkits_by_id:
                raise CompileError(f"agent '{agent.id}' references unknown toolkit '{tk}'")
        for sk in agent.skills:
            if sk not in skills_by_id:
                raise CompileError(f"agent '{agent.id}' references unknown skill '{sk}'")

    return CompiledGraph(
        doc=doc,
        nodes_by_id=nodes_by_id,
        agents_by_id=agents_by_id,
        toolkits_by_id=toolkits_by_id,
        skills_by_id=skills_by_id,
        conditional_edges_by_source=conditional_edges_by_source,
        default_edge_by_source=default_edge_by_source,
        inbound_sources=inbound_sources,
    )
