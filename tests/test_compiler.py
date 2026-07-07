from __future__ import annotations

import yaml

from ravana.compiler.graph import compile_workflow
from ravana.compiler.validate import validate
from ravana.schema.models import WorkflowDoc
from tests.conftest import SDLC_WORKFLOW


def _load_raw() -> dict:
    with open(SDLC_WORKFLOW) as f:
        return yaml.safe_load(f)


def test_published_example_compiles_and_validates_clean(sdlc_graph):
    issues = validate(sdlc_graph)
    assert issues == []


def test_validator_catches_missing_safety_net_on_qa_test():
    raw = _load_raw()
    raw["spec"]["graph"]["edges"] = [e for e in raw["spec"]["graph"]["edges"] if not e.get("is_default")]
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    issues = validate(graph)
    messages = [i.message for i in issues if i.severity == "warning"]
    assert any("qa_test" in m and "safety net" in m for m in messages)


def test_validator_catches_unreachable_node():
    raw = _load_raw()
    raw["spec"]["graph"]["nodes"].append({"id": "orphan", "agent": "pm"})
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    issues = validate(graph)
    assert any(i.severity == "warning" and "orphan" in i.message and "unreachable" in i.message for i in issues)


def test_validator_catches_broadcast_overwrite_conflict():
    raw = _load_raw()
    # Give dev's output_schema a key that qa also writes with 'overwrite'
    # merge, both reachable from the same broadcast edge (sa_design's).
    for agent in raw["spec"]["agents"]:
        if agent["id"] == "dev":
            agent["output_schema"] = {"type": "object", "properties": {"qa_status": {"type": "string"}}}
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    issues = validate(graph)
    assert any(i.severity == "error" and "qa_status" in i.message and "overwrite" in i.message for i in issues)
