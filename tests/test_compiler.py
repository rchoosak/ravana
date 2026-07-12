from __future__ import annotations

import sqlite3

import yaml
import pytest

from ravana.compiler.graph import CompileError, compile_workflow
from ravana.compiler.persist import get_or_create_workflow
from ravana.compiler.validate import validate
from ravana.schema.db import init_db
from ravana.schema.models import WorkflowDoc
from ravana.schema.util import loads
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


def test_node_execution_contract_can_narrow_agent_defaults_and_is_persisted(con):
    raw = _load_raw()
    node = next(n for n in raw["spec"]["graph"]["nodes"] if n["id"] == "pm_intake")
    node["toolkits"] = []  # PM agent allows web_search; this task grants none.
    node["output_schema"] = {
        "type": "object",
        "properties": {"node_only": {"type": "boolean"}},
    }
    node["hitl"] = {
        "enabled": True,
        "trigger_condition": "false",
        "prompt_template": "Node-specific review",
    }
    graph = compile_workflow(WorkflowDoc.model_validate(raw))

    contract = graph.contract_for_node("pm_intake")
    assert contract.toolkits == ()
    assert contract.output_schema["properties"] == {"node_only": {"type": "boolean"}}
    assert contract.hitl.prompt_template == "Node-specific review"

    workflow_id = get_or_create_workflow(con, graph, org_id="test", created_by="test")
    row = con.execute(
        "SELECT * FROM workflow_node WHERE workflow_id = ? AND id = 'pm_intake'",
        (workflow_id,),
    ).fetchone()
    assert loads(row["toolkit_ids"]) == []
    assert loads(row["output_schema"])["properties"] == {
        "node_only": {"type": "boolean"}
    }
    assert loads(row["hitl_config"])["prompt_template"] == "Node-specific review"


def test_explicit_null_clears_inherited_node_policy():
    raw = _load_raw()
    node = next(n for n in raw["spec"]["graph"]["nodes"] if n["id"] == "pm_intake")
    node["hitl"] = None
    node["output_schema"] = None

    contract = compile_workflow(WorkflowDoc.model_validate(raw)).contract_for_node("pm_intake")
    assert contract.toolkits == ("web_search",)
    assert contract.hitl is None
    assert contract.output_schema is None


def test_node_cannot_grant_tool_outside_agent_allow_list():
    raw = _load_raw()
    node = next(n for n in raw["spec"]["graph"]["nodes"] if n["id"] == "pm_intake")
    node["toolkits"] = ["git_connector"]
    with pytest.raises(CompileError, match="outside agent 'pm' allow-list"):
        compile_workflow(WorkflowDoc.model_validate(raw))


def test_init_db_adds_execution_contract_columns_to_existing_sqlite(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE workflow_node (
            id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            hitl_config TEXT,
            PRIMARY KEY (workflow_id, id)
        );
        CREATE TABLE node_execution (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL
        );
        """
    )
    legacy.close()

    migrated = init_db(db_path)
    workflow_node_columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(workflow_node)")
    }
    execution_columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(node_execution)")
    }
    migrated.close()

    assert {"toolkit_ids", "output_schema"} <= workflow_node_columns
    assert "logical_visit_id" in execution_columns
