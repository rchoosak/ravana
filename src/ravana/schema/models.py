"""Pydantic models for the workflow YAML schema described in ARCHITECTURE.md §4.

These mirror the persisted schema in ravana.schema.db (§2.2) closely enough
that compiling a WorkflowDoc into DB rows is close to a direct field mapping —
see ravana.compiler.compiler.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MergePolicy = Literal["overwrite", "merge-object", "append"]
ConcurrencyStrategy = Literal["queue", "cancel_previous", "allow"]
ToolkitType = Literal["web_search", "code_interpreter", "db", "api_connector", "mcp_server"]

TERMINAL = "__terminal__"

_SECRET_SCHEME = "secrets://"


def _require_secret_pointer(value: str | None) -> str | None:
    """§8: `auth_ref`/`api_key_ref` are POINTERS, never raw secrets. Enforced
    at the schema so a pasted raw key fails at compile/validate — BEFORE it is
    persisted into the workflow tables or echoed into any error message."""
    if value is not None and not value.startswith(_SECRET_SCHEME):
        raise ValueError(
            f"must be a {_SECRET_SCHEME} pointer, never a raw credential (§8) — got a non-pointer value (redacted)"
        )
    return value


class ConcurrencyConfig(BaseModel):
    group: str
    strategy: ConcurrencyStrategy = "queue"


class StateFieldSchema(BaseModel):
    type: Literal["string", "integer", "number", "boolean", "object", "array"]
    merge: MergePolicy = "overwrite"
    pii: bool = False


class StateConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fields: dict[str, StateFieldSchema] = Field(default_factory=dict, alias="schema")
    initial: dict[str, Any] = Field(default_factory=dict)


class ToolkitConfig(BaseModel):
    # hide_input_in_errors: a failed auth_ref validation must not echo the
    # pasted value back — it may BE the raw secret the validator exists to
    # keep out of persistence and logs (§8).
    model_config = ConfigDict(hide_input_in_errors=True)

    id: str
    type: ToolkitType
    config: dict[str, Any] = Field(default_factory=dict)
    auth_ref: str | None = None

    _pointer = field_validator("auth_ref")(_require_secret_pointer)


class SkillConfig(BaseModel):
    id: str
    description: str
    instructions: str
    resources: dict[str, Any] | None = None


class LLMFallbackEntry(BaseModel):
    # hide_input_in_errors: see ToolkitConfig — a rejected api_key_ref may BE
    # a pasted raw secret; never echo it in the validation error (§8).
    model_config = ConfigDict(hide_input_in_errors=True)

    provider: str
    model: str
    endpoint: str | None = None
    api_key_ref: str | None = None

    _pointer = field_validator("api_key_ref")(_require_secret_pointer)


class LLMConfig(BaseModel):
    # hide_input_in_errors: see ToolkitConfig (§8).
    model_config = ConfigDict(hide_input_in_errors=True)

    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int | None = None
    endpoint: str | None = None
    api_key_ref: str | None = None
    fallback: list[LLMFallbackEntry] = Field(default_factory=list)

    _pointer = field_validator("api_key_ref")(_require_secret_pointer)


class HITLConfig(BaseModel):
    enabled: bool = True
    trigger_condition: str
    prompt_template: str | None = None
    assignee: str | None = None


class AgentConfig(BaseModel):
    id: str
    name: str
    llm: LLMConfig
    system_prompt: str
    toolkits: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    hitl: HITLConfig | None = None
    output_schema: dict[str, Any] | None = None


class GraphNode(BaseModel):
    id: str
    agent: str | None = None
    sub_workflow: str | None = None
    on_enter: str | None = None
    # Per-node execution contract. Omitted fields inherit AgentConfig defaults;
    # an explicit null clears HITL/output_schema. `toolkits`, when set, is a
    # least-privilege subset of the agent's allow-list.
    toolkits: list[str] | None = None
    hitl: HITLConfig | None = None
    output_schema: dict[str, Any] | None = None
    # §3.8: "any" (default) — every arriving edge dispatches this node
    # independently; "all" — arrivals accumulate and the node dispatches once
    # per wave, when every inbound source has delivered (or at quiescence).
    join: Literal["any", "all"] = "any"

    @model_validator(mode="after")
    def _exactly_one_backing(self) -> "GraphNode":
        # Mirrors workflow_node_backing_xor (§2.2) at the YAML layer, before
        # it ever reaches the DB constraint of the same name.
        has_agent = self.agent is not None
        has_sub = self.sub_workflow is not None
        if has_agent == has_sub:
            raise ValueError(
                f"node '{self.id}' must set exactly one of agent/sub_workflow "
                f"(has_agent={has_agent}, has_sub_workflow={has_sub})"
            )
        return self


class GraphEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: list[str]
    condition: str | None = None
    is_default: bool = False
    mode: Literal["broadcast"] | None = None
    label: str | None = None
    priority: int = 0

    @field_validator("to")
    @classmethod
    def _non_empty_targets(cls, value: list[str]) -> list[str]:
        # Mirrors workflow_edge_has_targets (§2.2).
        if not value:
            raise ValueError("edge 'to' must list at least one target node")
        return value


class GraphGuards(BaseModel):
    max_total_steps: int = 100
    max_loop_iterations: dict[str, int] = Field(default_factory=dict)
    max_tool_calls_per_turn: int = 10
    max_output_repairs: int = 2
    max_retries_per_node: int = 3
    max_tokens_total: int | None = None


class GraphConfig(BaseModel):
    entry: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    guards: GraphGuards = Field(default_factory=GraphGuards)


class DefinitionOfDone(BaseModel):
    evaluated_by: str
    criteria: list[str] = Field(default_factory=list)


class WorkflowSpec(BaseModel):
    concurrency: ConcurrencyConfig | None = None
    state: StateConfig = Field(default_factory=StateConfig)
    toolkits: list[ToolkitConfig] = Field(default_factory=list)
    skills: list[SkillConfig] = Field(default_factory=list)
    agents: list[AgentConfig]
    graph: GraphConfig
    definition_of_done: DefinitionOfDone | None = None


class WorkflowMetadata(BaseModel):
    name: str
    description: str | None = None
    version: int = 1


class WorkflowDoc(BaseModel):
    # hide_input_in_errors applies from the OUTERMOST validating model, so it
    # must live here for `WorkflowDoc.model_validate(...)` errors to omit
    # input echoes. A workflow file can have a raw secret pasted anywhere
    # (that is exactly the mistake the §8 pointer validators reject), and the
    # validation error must not repeat it into terminals/logs. The error's
    # `loc` path still names the offending field.
    model_config = ConfigDict(hide_input_in_errors=True)

    apiVersion: str
    kind: Literal["Workflow"]
    metadata: WorkflowMetadata
    spec: WorkflowSpec
