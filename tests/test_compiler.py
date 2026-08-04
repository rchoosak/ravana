from __future__ import annotations

import sqlite3

import yaml
import pytest

from ravana.compiler.graph import CompileError, compile_workflow
from ravana.compiler.persist import WorkflowPersistenceError, get_or_create_workflow
from ravana.compiler.validate import validate
from ravana.schema.db import init_db
from ravana.schema.models import WorkflowDoc
from ravana.schema.util import loads
from ravana.schema.workflow_snapshot import WorkflowSnapshot
from tests.conftest import SDLC_WORKFLOW


def _load_raw() -> dict:
    with open(SDLC_WORKFLOW) as f:
        return yaml.safe_load(f)


def _legacy_workflow_connection(tmp_path):
    db_path = tmp_path / "legacy_workflow.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE workflow (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            state_schema TEXT NOT NULL,
            entry_node_id TEXT NOT NULL,
            dod_criteria TEXT,
            guards TEXT,
            concurrency TEXT,
            status TEXT NOT NULL DEFAULT 'DRAFT',
            created_by TEXT NOT NULL,
            published_by TEXT,
            published_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (org_id, name, version)
        );
        """
    )
    legacy.close()
    return init_db(db_path)


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

    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="test")
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

    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    contract = graph.contract_for_node("pm_intake")
    assert contract.toolkits == ("web_search",)
    assert contract.hitl is None
    assert contract.output_schema is None
    assert WorkflowSnapshot.from_doc(graph.doc).compile().contract_for_node("pm_intake") == contract


def test_node_cannot_grant_tool_outside_agent_allow_list():
    raw = _load_raw()
    node = next(n for n in raw["spec"]["graph"]["nodes"] if n["id"] == "pm_intake")
    node["toolkits"] = ["git_connector"]
    with pytest.raises(CompileError, match="outside agent 'pm' allow-list"):
        compile_workflow(WorkflowDoc.model_validate(raw))


def test_unknown_dod_evaluated_by_is_rejected_at_compile(sdlc_graph):
    # §3.1 step 7: a DoD naming a non-existent evaluated_by agent must fail at
    # compile, not silently run the whole workflow and only surface at the gate
    # (or never, under a backend with no prose judge).
    raw = _load_raw()
    raw["spec"]["definition_of_done"] = {"evaluated_by": "ghost", "criteria": ["all done"]}
    with pytest.raises(CompileError, match="definition_of_done.evaluated_by references unknown agent 'ghost'"):
        compile_workflow(WorkflowDoc.model_validate(raw))


def test_author_description_on_mcp_server_is_rejected_at_compile():
    # §1.2/§8: an mcp_server surfaces many tools, each carrying the SERVER's own
    # provenance-tagged description — a single author line has no tool to land
    # on, so it is rejected rather than silently dropped (the author would
    # otherwise never learn their description had no effect).
    raw = _load_raw()
    tk = next(t for t in raw["spec"]["toolkits"] if t["id"] == "github_mcp")
    tk["description"] = "Talk to GitHub."
    with pytest.raises(CompileError, match="mcp_server.*may not set a description"):
        compile_workflow(WorkflowDoc.model_validate(raw))


def test_duplicate_toolkit_and_skill_ids_are_compile_errors():
    import copy

    raw = _load_raw()
    raw["spec"]["toolkits"].append(copy.deepcopy(raw["spec"]["toolkits"][0]))
    with pytest.raises(CompileError, match="duplicate toolkit id"):
        compile_workflow(WorkflowDoc.model_validate(raw))

    raw = _load_raw()
    raw["spec"]["skills"].append(copy.deepcopy(raw["spec"]["skills"][0]))
    with pytest.raises(CompileError, match="duplicate skill id"):
        compile_workflow(WorkflowDoc.model_validate(raw))


def test_blank_toolkit_description_is_an_authoring_error():
    # A description that is set but empty/whitespace is a mistake, not a request
    # to use the default — that is expressed by omitting the field entirely.
    from ravana.schema.models import ToolkitConfig

    with pytest.raises(Exception, match="must not be blank"):
        ToolkitConfig(id="t", type="api_connector", description="   ")


def test_toolkit_description_rejects_credential_material_without_echoing_it():
    from ravana.schema.models import ToolkitConfig

    secret = "ghp_PASTED_DESCRIPTION_SECRET"
    with pytest.raises(Exception, match="must not contain credential material") as exc_info:
        ToolkitConfig(id="t", type="api_connector", description=f"Use {secret}")
    assert secret not in str(exc_info.value)


def test_toolkit_description_revalidates_assignment_without_echoing_secret():
    from ravana.schema.models import ToolkitConfig

    toolkit = ToolkitConfig(id="t", type="api_connector", description="Safe description")
    secret = "ghp_ASSIGNED_DESCRIPTION_SECRET"
    with pytest.raises(Exception, match="must not contain credential material") as exc_info:
        toolkit.description = f"Use {secret}"
    assert secret not in str(exc_info.value)
    assert toolkit.description == "Safe description"


def test_toolkit_description_is_stripped_and_omission_is_none():
    from ravana.schema.models import ToolkitConfig

    assert ToolkitConfig(id="t", type="api_connector", description="  hi  ").description == "hi"
    assert ToolkitConfig(id="t", type="api_connector").description is None


def test_author_toolkit_description_is_persisted_and_updated_with_audit(con):
    raw = _load_raw()
    tk = next(t for t in raw["spec"]["toolkits"] if t["id"] == "git_connector")
    tk["description"] = "Query the GitHub REST API."
    graph = compile_workflow(WorkflowDoc.model_validate(raw))

    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="test")
    rows = {r["name"]: r["description"] for r in con.execute("SELECT name, description FROM toolkit")}
    assert rows["git_connector"] == "Query the GitHub REST API."
    # A toolkit that set no description persists NULL, not "" — the audit record
    # must not fabricate a description the author never wrote.
    assert rows["web_search"] is None

    tk["description"] = "Inspect GitHub repository state."
    raw["spec"]["agents"][0]["system_prompt"] = "Updated project-manager prompt."
    raw["spec"]["skills"][0]["instructions"] = "Updated reusable instructions."
    raw["spec"]["graph"]["guards"]["max_total_steps"] = 77
    raw["spec"]["graph"]["edges"][0]["priority"] = 42
    raw["metadata"]["description"] = "Updated workflow description"
    updated_graph = compile_workflow(WorkflowDoc.model_validate(raw))
    assert get_or_create_workflow(con, updated_graph, org_id="test", actor="editor") == workflow_id

    updated_rows = con.execute(
        "SELECT id, description FROM toolkit WHERE name = 'git_connector'"
    ).fetchall()
    assert len(updated_rows) == 1
    assert updated_rows[0]["description"] == "Inspect GitHub repository state."
    persisted_prompt = con.execute(
        """SELECT a.system_prompt FROM workflow_node AS n
           JOIN agent AS a ON a.id = n.agent_id
           WHERE n.workflow_id = ? AND n.id = 'pm_intake'""",
        (workflow_id,),
    ).fetchone()[0]
    assert persisted_prompt == "Updated project-manager prompt."
    assert con.execute(
        "SELECT description FROM workflow WHERE id = ?", (workflow_id,)
    ).fetchone()[0] == "Updated workflow description"
    assert loads(con.execute(
        "SELECT guards FROM workflow WHERE id = ?", (workflow_id,)
    ).fetchone()[0])["max_total_steps"] == 77
    assert con.execute(
        "SELECT instructions FROM skill WHERE org_id = 'test' AND name = 'conventional_commits'"
    ).fetchone()[0] == "Updated reusable instructions."
    assert con.execute(
        """SELECT priority FROM workflow_edge
           WHERE workflow_id = ? AND source_node_id = 'pm_intake'""",
        (workflow_id,),
    ).fetchone()[0] == 42

    audits = con.execute(
        """SELECT actor, before, after FROM audit_log
           WHERE entity_id = ? AND action = 'workflow.draft_saved' ORDER BY rowid""",
        (workflow_id,),
    ).fetchall()
    assert len(audits) == 2
    assert audits[1]["actor"] == "editor"
    before = loads(audits[1]["before"])
    after = loads(audits[1]["after"])
    before_git = next(t for t in before["spec"]["toolkits"] if t["id"] == "git_connector")
    after_git = next(t for t in after["spec"]["toolkits"] if t["id"] == "git_connector")
    assert before_git["description"] == "Query the GitHub REST API."
    assert after_git["description"] == "Inspect GitHub repository state."
    assert before["spec"]["agents"][0]["system_prompt"] != after["spec"]["agents"][0]["system_prompt"]

    # Re-saving identical content remains idempotent and does not fabricate an edit.
    get_or_create_workflow(con, updated_graph, org_id="test", actor="editor")
    audit_count = con.execute(
        "SELECT count(*) FROM audit_log WHERE entity_id = ? AND action = 'workflow.draft_saved'",
        (workflow_id,),
    ).fetchone()[0]
    assert audit_count == 2


def test_published_workflow_rejects_non_toolkit_mutation(con):
    raw = _load_raw()
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="author")
    con.execute("UPDATE workflow SET status = 'PUBLISHED' WHERE id = ?", (workflow_id,))
    con.commit()

    raw["spec"]["agents"][0]["system_prompt"] = "Changed after publication"
    changed = compile_workflow(WorkflowDoc.model_validate(raw))
    with pytest.raises(WorkflowPersistenceError, match="cannot be edited in place"):
        get_or_create_workflow(con, changed, org_id="test", actor="editor")

    persisted = con.execute(
        """SELECT a.system_prompt FROM workflow_node AS n
           JOIN agent AS a ON a.id = n.agent_id
           WHERE n.workflow_id = ? AND n.id = 'pm_intake'""",
        (workflow_id,),
    ).fetchone()[0]
    assert persisted != "Changed after publication"


def test_legacy_draft_claims_existing_toolkits_before_updating(tmp_path):
    con = _legacy_workflow_connection(tmp_path)
    raw = _load_raw()
    toolkit = next(t for t in raw["spec"]["toolkits"] if t["id"] == "git_connector")
    toolkit["description"] = "Legacy description"
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="author")
    original_id = con.execute(
        "SELECT id FROM toolkit WHERE name = 'git_connector'"
    ).fetchone()[0]

    con.execute(
        "UPDATE workflow SET toolkit_ids = '[]', definition_snapshot = NULL WHERE id = ?",
        (workflow_id,),
    )
    con.commit()
    toolkit["description"] = "Migrated description"
    changed = compile_workflow(WorkflowDoc.model_validate(raw))
    get_or_create_workflow(con, changed, org_id="test", actor="editor")

    rows = con.execute(
        "SELECT id, description FROM toolkit WHERE name = 'git_connector'"
    ).fetchall()
    assert [(row["id"], row["description"]) for row in rows] == [
        (original_id, "Migrated description")
    ]
    audit = con.execute(
        "SELECT before FROM audit_log WHERE entity_id = ? ORDER BY rowid DESC LIMIT 1",
        (workflow_id,),
    ).fetchone()
    assert loads(audit["before"])["toolkits"]["git_connector"]["description"] == (
        "Legacy description"
    )
    con.close()


def test_legacy_published_workflow_without_snapshot_fails_closed(tmp_path):
    con = _legacy_workflow_connection(tmp_path)
    graph = compile_workflow(WorkflowDoc.model_validate(_load_raw()))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="author")
    con.execute(
        """UPDATE workflow SET status = 'PUBLISHED', toolkit_ids = '[]',
           definition_snapshot = NULL WHERE id = ?""",
        (workflow_id,),
    )
    con.commit()
    with pytest.raises(WorkflowPersistenceError, match="no verifiable definition snapshot"):
        get_or_create_workflow(con, graph, org_id="test", actor="operator")
    con.close()


def test_draft_mutation_rolls_back_when_audit_write_fails(con, monkeypatch):
    raw = _load_raw()
    toolkit = next(t for t in raw["spec"]["toolkits"] if t["id"] == "git_connector")
    toolkit["description"] = "Before"
    graph = compile_workflow(WorkflowDoc.model_validate(raw))
    workflow_id = get_or_create_workflow(con, graph, org_id="test", actor="author")

    toolkit["description"] = "After"
    changed = compile_workflow(WorkflowDoc.model_validate(raw))

    def fail_audit(*args, **kwargs):
        raise sqlite3.OperationalError("audit unavailable")

    monkeypatch.setattr("ravana.compiler.persist.write_audit", fail_audit)
    with pytest.raises(sqlite3.OperationalError, match="audit unavailable"):
        get_or_create_workflow(con, changed, org_id="test", actor="editor")

    assert con.execute(
        "SELECT description FROM toolkit WHERE name = 'git_connector'"
    ).fetchone()[0] == "Before"
    stored_snapshot = con.execute(
        "SELECT definition_snapshot FROM workflow WHERE id = ?", (workflow_id,)
    ).fetchone()[0]
    stored_git = next(
        t for t in loads(stored_snapshot)["spec"]["toolkits"] if t["id"] == "git_connector"
    )
    assert stored_git["description"] == "Before"
    assert not con.in_transaction


def test_init_db_adds_toolkit_description_column(tmp_path):
    db_path = tmp_path / "legacy_toolkit.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE toolkit (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            config TEXT NOT NULL,
            auth_ref TEXT
        );
        """
    )
    legacy.close()

    migrated = init_db(db_path)
    columns = {row[1] for row in migrated.execute("PRAGMA table_info(toolkit)")}
    migrated.close()
    assert "description" in columns


def test_init_db_adds_workflow_and_run_snapshot_columns(tmp_path):
    db_path = tmp_path / "legacy_snapshots.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE workflow (id TEXT PRIMARY KEY);
        CREATE TABLE run (
            id TEXT PRIMARY KEY,
            workflow_id TEXT,
            concurrency_group TEXT,
            status TEXT NOT NULL
        );
        """
    )
    legacy.close()

    migrated = init_db(db_path)
    workflow_columns = {row[1] for row in migrated.execute("PRAGMA table_info(workflow)")}
    run_columns = {row[1] for row in migrated.execute("PRAGMA table_info(run)")}
    migrated.close()
    assert "toolkit_ids" in workflow_columns
    assert "definition_snapshot" in workflow_columns
    assert "workflow_snapshot" in run_columns
    assert "agent_db_ids" in run_columns


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
        CREATE TABLE mcp_tool_snapshot (
            run_id TEXT NOT NULL,
            toolkit_id TEXT NOT NULL,
            server_fingerprint TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            description TEXT NOT NULL,
            input_schema TEXT NOT NULL,
            PRIMARY KEY (run_id, toolkit_id, tool_name)
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
    snapshot_columns = {
        row[1] for row in migrated.execute("PRAGMA table_info(mcp_tool_snapshot)")
    }
    migrated.close()

    assert {"toolkit_ids", "output_schema"} <= workflow_node_columns
    assert "logical_visit_id" in execution_columns
    assert "created_at" in snapshot_columns
