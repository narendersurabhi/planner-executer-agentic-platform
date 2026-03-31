import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import redis
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app import main, memory_store  # noqa: E402
from services.api.app.database import Base, engine
from services.api.app.database import SessionLocal
from services.api.app.models import (
    EventOutboxRecord,
    InvocationRecord,
    JobRecord,
    PlanRecord,
    RunEventRecord,
    StepAttemptRecord,
    TaskRecord,
    TaskResultRecord,
    WorkflowDefinitionRecord,
    WorkflowRunRecord,
    WorkflowTriggerRecord,
    WorkflowVersionRecord,
)
from libs.core import events, execution_contracts, models, run_specs, workflow_contracts
from libs.core import capability_registry as cap_registry
from libs.core.llm_provider import LLMProvider, LLMRequest, LLMResponse


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _document_render_registry() -> cap_registry.CapabilityRegistry:
    return cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": cap_registry.CapabilitySpec(
                capability_id="document.spec.generate",
                description="Generate a document spec.",
                enabled=True,
                risk_tier="read_only",
                idempotency="read",
                group="document",
                subgroup="spec",
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="llm_generate_document_spec",
                    ),
                ),
                planner_hints={"task_intents": ["generate"]},
            ),
            "derive_output_filename": cap_registry.CapabilitySpec(
                capability_id="derive_output_filename",
                description="Derive a safe output path.",
                enabled=True,
                risk_tier="read_only",
                idempotency="read",
                group="document",
                subgroup="paths",
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="derive_output_filename",
                    ),
                ),
                planner_hints={"task_intents": ["transform"]},
            ),
            "document.docx.render": cap_registry.CapabilitySpec(
                capability_id="document.docx.render",
                description="Render a DOCX document.",
                enabled=True,
                risk_tier="bounded_write",
                idempotency="safe_write",
                group="document",
                subgroup="render",
                adapters=(
                    cap_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="docx_render_from_spec",
                    ),
                ),
                planner_hints={
                    "task_intents": ["render"],
                    "required_output_extension": "docx",
                },
            ),
        }
    )


def test_create_job():
    response = client.post(
        "/jobs",
        json={"goal": "demo", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "demo"
    assert data["metadata"]["normalized_intent_envelope"]["goal"] == "demo"
    assert data["metadata"]["render_path_mode"] == "explicit"


def test_intent_assessment_fallback_used_respects_mode(monkeypatch) -> None:
    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "hybrid")

    assert main._intent_assessment_fallback_used("heuristic") is True
    assert main._intent_assessment_fallback_used("llm") is False


def test_intent_decompose_fallback_used_respects_mode(monkeypatch) -> None:
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "hybrid")

    assert main._intent_decompose_fallback_used("heuristic") is True
    assert main._intent_decompose_fallback_used("llm") is False


def test_intent_segment_contract_reason_parses_prefixed_error() -> None:
    assert (
        main._intent_segment_contract_reason(
            "intent_segment_invalid:llm_generate:GenerateText:must_have_inputs_missing:path"
        )
        == "must_have_inputs_missing"
    )
    assert main._intent_segment_contract_reason("output_format_mismatch:pdf:docx") == (
        "output_format_mismatch"
    )


def test_workflow_definition_publish_and_run_bypasses_planner_job_event() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )

    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()
    assert version["version_number"] == 1
    assert version["compiled_plan"]["tasks"][0]["name"] == "ListWorkspace"
    assert version["run_spec"]["steps"][0]["name"] == "ListWorkspace"
    assert version["metadata"]["run_spec"]["steps"][0]["name"] == "ListWorkspace"

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 2},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    job_id = run_body["job"]["id"]
    assert run_body["workflow_run"]["definition_id"] == definition["id"]
    assert run_body["workflow_run"]["version_id"] == version["id"]
    assert run_body["workflow_run"]["job_id"] == job_id
    assert run_body["job"]["metadata"]["workflow_source"] == "studio"
    assert run_body["job"]["metadata"]["render_path_mode"] == "auto"
    assert run_body["job"]["metadata"]["workflow_definition_id"] == definition["id"]
    assert run_body["job"]["metadata"]["workflow_version_id"] == version["id"]
    assert run_body["job"]["metadata"]["workflow_run_id"] == run_body["workflow_run"]["id"]
    assert run_body["plan"]["job_id"] == job_id

    with SessionLocal() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        assert len(task_records) == 1
        event_records = db.query(EventOutboxRecord).all()
    matching_events = [
        row
        for row in event_records
        if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id
    ]
    event_types = {row.event_type for row in matching_events}
    assert "plan.created" in event_types
    assert "job.created" not in event_types

    runs_response = client.get(f"/workflows/definitions/{definition['id']}/runs")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) >= 1
    assert runs[0]["job_id"] == job_id
    assert runs[0]["job_status"] in {
        models.JobStatus.queued.value,
        models.JobStatus.running.value,
        models.JobStatus.planning.value,
    }


def test_workflow_run_uses_postgres_scheduler_when_flag_enabled(monkeypatch) -> None:
    monkeypatch.setattr(main, "STUDIO_RUN_SCHEDULER_ENABLED", True)
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Scheduled workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Scheduled workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 1},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    job_id = run_body["job"]["id"]
    assert (
        run_body["workflow_run"]["metadata"]["scheduler_mode"]
        == main.POSTGRES_RUN_SPEC_SCHEDULER_MODE
    )
    assert run_body["job"]["metadata"]["scheduler_mode"] == main.POSTGRES_RUN_SPEC_SCHEDULER_MODE

    with SessionLocal() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        assert len(task_records) == 1
        assert task_records[0].status == models.TaskStatus.ready.value
        assert task_records[0].attempts == 1
        event_records = db.query(EventOutboxRecord).all()
    matching_events = [
        row
        for row in event_records
        if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id
    ]
    event_types = {row.event_type for row in matching_events}
    assert "task.ready" in event_types
    assert "plan.created" not in event_types


def test_workflow_version_run_allows_derived_render_paths_in_studio_mode(monkeypatch) -> None:
    registry = _document_render_registry()
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main, "RUNTIME_CONFORMANCE_ENABLED", False)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", False)

    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Render DOCX workflow",
            "goal": "Render a DOCX deployment report",
            "context_json": {},
            "draft": {
                "summary": "Render DOCX workflow",
                "nodes": [
                    {
                        "id": "spec",
                        "taskName": "GenerateDocumentSpec",
                        "capabilityId": "document.spec.generate",
                        "bindings": {
                            "instruction": {
                                "kind": "literal",
                                "value": "Create a deployment report document.",
                            },
                            "topic": {"kind": "literal", "value": "Deployment report"},
                            "audience": {"kind": "literal", "value": "Engineers"},
                        },
                    },
                    {
                        "id": "path",
                        "taskName": "DeriveOutputPath",
                        "capabilityId": "derive_output_filename",
                        "bindings": {
                            "document_type": {"kind": "literal", "value": "document"},
                            "output_extension": {"kind": "literal", "value": "docx"},
                            "target_role_name": {
                                "kind": "literal",
                                "value": "deployment_report",
                            },
                        },
                    },
                    {
                        "id": "render",
                        "taskName": "RenderDocument",
                        "capabilityId": "document.docx.render",
                        "bindings": {
                            "document_spec": {
                                "kind": "step_output",
                                "sourceNodeId": "spec",
                                "sourcePath": "document_spec",
                            },
                            "path": {
                                "kind": "step_output",
                                "sourceNodeId": "path",
                                "sourcePath": "path",
                            },
                        },
                    },
                ],
                "edges": [
                    {"fromNodeId": "spec", "toNodeId": "render"},
                    {"fromNodeId": "path", "toNodeId": "render"},
                ],
            },
        },
    )

    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={},
    )

    assert run_response.status_code == 200
    body = run_response.json()
    assert body["job"]["metadata"]["workflow_source"] == "studio"
    assert body["job"]["metadata"]["render_path_mode"] == "auto"


def test_studio_scheduler_advances_dependency_from_durable_step_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "STUDIO_RUN_SCHEDULER_ENABLED", True)
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Durable dependency scheduling",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Durable dependency scheduling",
                "nodes": [
                    {
                        "id": "source",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    },
                    {
                        "id": "target",
                        "taskName": "CaptureWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    },
                ],
                "edges": [{"fromNodeId": "source", "toNodeId": "target"}],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 1},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    workflow_run_id = run_body["workflow_run"]["id"]
    job_id = run_body["job"]["id"]
    now = _utcnow()

    with SessionLocal() as db:
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        tasks_by_name = {task.name: task for task in tasks}
        source_task = tasks_by_name["ListWorkspace"]
        target_task = tasks_by_name["CaptureWorkspace"]
        assert source_task.status == models.TaskStatus.ready.value
        assert target_task.status == models.TaskStatus.pending.value
        source_task.status = models.TaskStatus.pending.value
        source_task.updated_at = now
        db.add(
            StepAttemptRecord(
                id=main._step_attempt_record_id(source_task.id, 1),
                run_id=workflow_run_id,
                job_id=job_id,
                step_id=source_task.id,
                attempt_number=1,
                status=models.TaskStatus.completed.value,
                worker_id="worker-a",
                started_at=now,
                finished_at=now,
                error_code=None,
                error_message=None,
                retry_classification="succeeded",
                result_summary_json={},
            )
        )
        for row in db.query(EventOutboxRecord).all():
            if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id:
                db.delete(row)
        db.commit()

    main._schedule_workflow_run(workflow_run_id, correlation_id="corr-durable")

    with SessionLocal() as db:
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        tasks_by_name = {task.name: task for task in tasks}
        assert tasks_by_name["ListWorkspace"].status == models.TaskStatus.completed.value
        assert tasks_by_name["CaptureWorkspace"].status == models.TaskStatus.ready.value
        assert tasks_by_name["CaptureWorkspace"].attempts == 1
        event_records = db.query(EventOutboxRecord).all()
    matching_events = [
        row
        for row in event_records
        if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id
    ]
    ready_payloads = [
        row.envelope_json.get("payload", {})
        for row in matching_events
        if row.event_type == "task.ready"
    ]
    assert len(ready_payloads) == 1
    assert ready_payloads[0]["name"] == "CaptureWorkspace"


def test_workflow_trigger_create_invoke_and_list() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Triggered workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Triggered workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    trigger_response = client.post(
        f"/workflows/definitions/{definition['id']}/triggers",
        json={
            "title": "Run latest workspace listing",
            "trigger_type": "manual",
            "enabled": True,
            "config": {"version_mode": "latest_published"},
            "metadata": {"source": "test"},
        },
    )
    assert trigger_response.status_code == 200
    trigger = trigger_response.json()
    assert trigger["definition_id"] == definition["id"]
    assert trigger["trigger_type"] == "manual"

    list_triggers_response = client.get(f"/workflows/definitions/{definition['id']}/triggers")
    assert list_triggers_response.status_code == 200
    trigger_items = list_triggers_response.json()
    assert len(trigger_items) >= 1
    assert trigger_items[0]["id"] == trigger["id"]

    invoke_response = client.post(
        f"/workflows/triggers/{trigger['id']}/invoke",
        json={"version_id": version["id"], "priority": 1},
    )
    assert invoke_response.status_code == 200
    invoke_body = invoke_response.json()
    assert invoke_body["workflow_run"]["trigger_id"] == trigger["id"]
    assert invoke_body["job"]["metadata"]["workflow_trigger_id"] == trigger["id"]
    assert invoke_body["workflow_run"]["job_id"] == invoke_body["job"]["id"]

    runs_response = client.get(f"/workflows/definitions/{definition['id']}/runs?limit=5")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) >= 1
    assert runs[0]["trigger_id"] == trigger["id"]


def test_task_result_persists_to_postgres_and_workflow_run_history_surfaces_latest_error() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Runtime failure visibility",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Runtime failure visibility",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 1},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    job_id = run_body["job"]["id"]

    tasks_response = client.get(f"/jobs/{job_id}/tasks")
    assert tasks_response.status_code == 200
    task = tasks_response.json()[0]
    task_id = task["id"]
    task_name = task["name"]
    now = _utcnow().isoformat()

    main._handle_task_failed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "status": models.TaskStatus.failed.value,
                "outputs": {"tool_error": {"error": "contract.input_missing:job"}},
                "artifacts": [],
                "tool_calls": [],
                "started_at": now,
                "finished_at": now,
                "error": "contract.input_missing:job",
            },
        }
    )

    with SessionLocal() as db:
        row = db.query(TaskResultRecord).filter(TaskResultRecord.task_id == task_id).first()
        assert row is not None
        assert row.latest_error == "contract.input_missing:job"

    loaded_result = main._load_task_result(task_id)
    assert loaded_result["error"] == "contract.input_missing:job"

    runs_response = client.get(f"/workflows/definitions/{definition['id']}/runs?limit=5")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert runs[0]["job_id"] == job_id
    assert runs[0]["job_status"] == models.JobStatus.failed.value
    assert runs[0]["latest_task_name"] == task_name
    assert runs[0]["latest_task_error"] == "contract.input_missing:job"


def test_publish_workflow_definition_rejects_runtime_conformance_when_secret_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "GitHub capability publish gate",
            "goal": "Check repository visibility",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "GitHub capability publish gate",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "CheckRepo",
                        "capabilityId": "github.repo.list",
                        "bindings": {
                            "query": {
                                "kind": "literal",
                                "value": "repo:planner-executer-agentic-platform owner:narendersurabhi",
                            }
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 400
    detail = publish_response.json()["detail"]
    assert detail["error"] == "workflow_preflight_failed"
    assert "CheckRepo" in detail["preflight_errors"]
    assert "runtime conformance failed" in detail["preflight_errors"]["CheckRepo"]
    assert "GITHUB_TOKEN" in detail["preflight_errors"]["CheckRepo"]


def test_run_workflow_version_rechecks_runtime_conformance(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "GitHub capability run gate",
            "goal": "Check repository visibility",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "GitHub capability run gate",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "CheckRepo",
                        "capabilityId": "github.repo.list",
                        "bindings": {
                            "query": {
                                "kind": "literal",
                                "value": "repo:planner-executer-agentic-platform owner:narendersurabhi",
                            }
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 1},
    )
    assert run_response.status_code == 400
    detail = run_response.json()["detail"]
    assert detail["error"] == "workflow_preflight_failed"
    assert "CheckRepo" in detail["preflight_errors"]
    assert "runtime conformance failed" in detail["preflight_errors"]["CheckRepo"]
    assert "GITHUB_TOKEN" in detail["preflight_errors"]["CheckRepo"]


def test_delete_workflow_definition_removes_related_records() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Delete me",
            "goal": "Delete workflow definition",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Delete me",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    trigger_response = client.post(
        f"/workflows/definitions/{definition['id']}/triggers",
        json={
            "title": "Delete trigger",
            "trigger_type": "manual",
            "enabled": True,
            "config": {"version_mode": "latest_published"},
        },
    )
    assert trigger_response.status_code == 200
    trigger = trigger_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 0},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["workflow_run"]["definition_id"] == definition["id"]

    delete_response = client.delete(f"/workflows/definitions/{definition['id']}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"ok": True}

    with SessionLocal() as db:
        assert (
            db.query(WorkflowDefinitionRecord)
            .filter(WorkflowDefinitionRecord.id == definition["id"])
            .count()
            == 0
        )
        assert (
            db.query(WorkflowVersionRecord)
            .filter(WorkflowVersionRecord.definition_id == definition["id"])
            .count()
            == 0
        )
        assert (
            db.query(WorkflowTriggerRecord)
            .filter(WorkflowTriggerRecord.definition_id == definition["id"])
            .count()
            == 0
        )
        assert (
            db.query(WorkflowRunRecord)
            .filter(WorkflowRunRecord.definition_id == definition["id"])
            .count()
            == 0
        )

    get_response = client.get(f"/workflows/definitions/{definition['id']}")
    assert get_response.status_code == 404

    list_response = client.get("/workflows/definitions?user_id=narendersurabhi")
    assert list_response.status_code == 200
    definition_ids = {item["id"] for item in list_response.json()}
    assert definition["id"] not in definition_ids


def test_build_workflow_interface_runtime_context_resolves_bindings() -> None:
    with SessionLocal() as db:
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="user_profile",
                scope=models.MemoryScope.user,
                user_id="narendersurabhi",
                key="profile",
                payload={"full_name": "Narender Rao Surabhi"},
                metadata={},
            ),
        )
        workflow_interface, errors, _warnings = main._coerce_workflow_interface(
            {
                "inputs": [
                    {
                        "key": "topic",
                        "valueType": "string",
                        "required": True,
                        "binding": {"kind": "context", "path": "topic"},
                    },
                    {
                        "key": "profile",
                        "valueType": "object",
                        "binding": {
                            "kind": "memory",
                            "scope": "user",
                            "name": "user_profile",
                            "key": "profile",
                        },
                    },
                    {
                        "key": "api_key",
                        "valueType": "string",
                        "binding": {"kind": "secret", "secretName": "TEST_WORKFLOW_SECRET"},
                    },
                ],
                "variables": [
                    {
                        "key": "topic_alias",
                        "binding": {"kind": "workflow_input", "inputKey": "topic"},
                    }
                ],
            }
        )
        assert errors == []
        context, diagnostics = main._build_workflow_interface_runtime_context(
            workflow_interface,
            base_context={"topic": "Release notes", "user_id": "narendersurabhi"},
            db=db,
            preview=False,
        )

    assert diagnostics == []
    assert context["workflow"]["inputs"]["topic"] == "Release notes"
    assert context["workflow"]["inputs"]["profile"] == {"full_name": "Narender Rao Surabhi"}
    assert context["workflow"]["inputs"]["api_key"] == execution_contracts.build_secret_ref(
        "TEST_WORKFLOW_SECRET"
    )
    assert context["workflow"]["variables"]["topic_alias"] == "Release notes"


def test_intent_clarify_endpoint_validates_goal():
    response = client.post("/intent/clarify", json={})
    assert response.status_code == 400
    assert response.json()["detail"] == "goal_required"


def test_intent_clarify_endpoint_returns_assessment():
    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})
    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    envelope = body["normalized_intent_envelope"]
    assert body["goal"] == "Render a PDF status report"
    assert assessment["intent"] == "render"
    assert assessment["needs_clarification"] is True
    assert assessment["missing_slots"] == ["path"]
    assert assessment["source"] in {"goal_text", "task_text", "explicit", "default"}
    assert envelope["goal"] == "Render a PDF status report"
    assert envelope["profile"]["intent"] == assessment["intent"]
    assert envelope["clarification"]["missing_inputs"] == ["path"]


def test_intent_clarify_endpoint_skips_vector_search_for_strong_heuristic(monkeypatch):
    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del path, method, body, query, timeout_s
        raise AssertionError("vector retriever should not run for strong heuristic intent matches")

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "heuristic")
    monkeypatch.setattr(main, "INTENT_VECTOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_intent_vector_synced_namespace", None)

    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})

    assert response.status_code == 200
    assert response.json()["assessment"]["intent"] == "render"


def test_intent_clarify_endpoint_skips_llm_for_strong_hybrid_inference(monkeypatch):
    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            raise AssertionError(f"LLM assessment should be skipped for strong hybrid inference: {request}")

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del path, method, body, query, timeout_s
        raise AssertionError("vector retriever should not run for strong heuristic intent matches")

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "hybrid")
    monkeypatch.setattr(main, "INTENT_VECTOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(main, "_intent_assess_provider", _Provider())
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_intent_vector_synced_namespace", None)

    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})

    assert response.status_code == 200
    body = response.json()
    assert body["assessment"]["intent"] == "render"
    assert body["assessment"]["source"] in {"goal_text", "task_text", "explicit"}


def test_intent_clarify_endpoint_uses_llm_assessment_with_capability_catalog(monkeypatch):
    requests: list[LLMRequest] = []

    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            requests.append(request)
            return LLMResponse(
                content=(
                    '{"intent":"render","confidence":0.94,'
                    '"suggested_capabilities":["document.pdf.render"],'
                    '"reason":"PDF rendering capability best matches the goal"}'
                )
            )

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "llm")
    monkeypatch.setattr(main, "INTENT_CAPABILITY_TOP_K", 3)
    monkeypatch.setattr(main, "_intent_assess_provider", _Provider())
    monkeypatch.setattr(
        main,
        "_intent_catalog_capability_entries",
        lambda: [
            {
                "id": "document.pdf.render",
                "group": "document",
                "subgroup": "render",
                "description": "Render a PDF from a document spec",
            },
            {
                "id": "llm.text.generate",
                "group": "llm",
                "subgroup": "generate",
                "description": "Generate text with an LLM",
            },
        ],
    )
    monkeypatch.setattr(
        main,
        "_intent_catalog_capability_ids",
        lambda: {"document.pdf.render", "llm.text.generate"},
    )
    monkeypatch.setattr(
        main,
        "_semantic_goal_capability_hints",
        lambda **_kwargs: [
            {
                "id": "document.pdf.render",
                "score": 0.99,
                "reason": "semantic match",
                "source": "semantic_search",
            }
        ],
    )

    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})
    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert assessment["intent"] == "render"
    assert assessment["source"] == "llm"
    assert requests
    assert requests[0].metadata is not None
    assert requests[0].metadata["operation"] == "intent_assess"
    assert "document.pdf.render" in requests[0].prompt


def test_intent_clarify_endpoint_falls_back_when_llm_assessment_fails(monkeypatch):
    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            del request
            return LLMResponse(content="not-json")

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "hybrid")
    monkeypatch.setattr(main, "_intent_assess_provider", _Provider())
    monkeypatch.setattr(main, "_intent_catalog_capability_entries", lambda: [])
    monkeypatch.setattr(main, "_intent_catalog_capability_ids", lambda: set())
    monkeypatch.setattr(main, "_semantic_goal_capability_hints", lambda **_kwargs: [])

    response = client.post("/intent/clarify", json={"goal": "Render a PDF status report"})
    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert assessment["intent"] == "render"
    assert assessment["source"] in {"goal_text", "task_text", "explicit", "default"}


def test_intent_clarify_endpoint_uses_vector_intent_for_fuzzy_goal(monkeypatch):
    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 5, "chunk_ids": ["a", "b", "c"]}
        if path == "/retrieve":
            assert body is not None
            assert body["query"] == "Polish this document spec for release readiness"
            return {
                "matches": [
                    {
                        "document_id": "transform",
                        "score": 0.88,
                        "metadata": {"intent_id": "transform"},
                    },
                    {
                        "document_id": "generate",
                        "score": 0.53,
                        "metadata": {"intent_id": "generate"},
                    },
                ]
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "heuristic")
    monkeypatch.setattr(main, "INTENT_VECTOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_intent_vector_synced_namespace", None)

    response = client.post(
        "/intent/clarify",
        json={"goal": "Polish this document spec for release readiness"},
    )

    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert assessment["intent"] == "transform"
    assert assessment["source"] == "vector"
    assert assessment["needs_clarification"] is False


def test_intent_clarify_endpoint_uses_vector_fallback_when_llm_assessment_fails(monkeypatch):
    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            del request
            return LLMResponse(content="not-json")

    def _rag_request(path, *, method="GET", body=None, query=None, timeout_s=20.0):
        del method, query, timeout_s
        if path == "/index/upsert_texts":
            return {"collection_name": "test", "upserted_count": 5, "chunk_ids": ["a", "b", "c"]}
        if path == "/retrieve":
            assert body is not None
            assert body["query"] == "Polish this document spec for release readiness"
            return {
                "matches": [
                    {
                        "document_id": "transform",
                        "score": 0.86,
                        "metadata": {"intent_id": "transform"},
                    },
                    {
                        "document_id": "generate",
                        "score": 0.55,
                        "metadata": {"intent_id": "generate"},
                    },
                ]
            }
        raise AssertionError(f"unexpected RAG path: {path}")

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "hybrid")
    monkeypatch.setattr(main, "INTENT_VECTOR_SEARCH_ENABLED", True)
    monkeypatch.setattr(main, "_intent_assess_provider", _Provider())
    monkeypatch.setattr(main, "_intent_catalog_capability_entries", lambda: [])
    monkeypatch.setattr(main, "_intent_catalog_capability_ids", lambda: set())
    monkeypatch.setattr(main, "_semantic_goal_capability_hints", lambda **_kwargs: [])
    monkeypatch.setattr(main, "_rag_retriever_request_json", _rag_request)
    monkeypatch.setattr(main, "_intent_vector_synced_namespace", None)

    response = client.post(
        "/intent/clarify",
        json={"goal": "Polish this document spec for release readiness"},
    )

    assert response.status_code == 200
    body = response.json()
    assessment = body["assessment"]
    assert assessment["intent"] == "transform"
    assert assessment["source"] == "vector"
    assert assessment["needs_clarification"] is False


def test_intent_decompose_endpoint_returns_graph():
    response = client.post(
        "/intent/decompose",
        json={"goal": "Fetch repositories, summarize insights, and render a PDF report."},
    )
    assert response.status_code == 200
    body = response.json()
    graph = body["intent_graph"]
    envelope = body["normalized_intent_envelope"]
    assessment = body["assessment"]
    assert graph["summary"]["segment_count"] >= 2
    assert graph["segments"][0]["id"] == "s1"
    assert graph["summary"]["schema_version"] == "intent_v2"
    assert isinstance(graph["segments"][0].get("slots"), dict)
    assert "must_have_inputs" in graph["segments"][0]["slots"]
    assert envelope["graph"] == graph
    assert envelope["profile"] == assessment


def test_intent_decompose_endpoint_uses_normalized_envelope_without_assessment_llm(monkeypatch):
    class _AssessProvider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            raise AssertionError(f"assessment provider should not be called: {request.metadata}")

    class _DecomposeProvider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.92,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate"],'
                    '"slots":{"entity":"summary","artifact_type":"content","output_format":"txt",'
                    '"risk_level":"read_only","must_have_inputs":["instruction"]}}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_ASSESS_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_ASSESS_MODE", "hybrid")
    monkeypatch.setattr(main, "_intent_assess_provider", _AssessProvider())
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _DecomposeProvider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    body = response.json()
    assert body["assessment"]["source"] in {"goal_text", "task_text", "explicit", "default"}
    assert body["normalized_intent_envelope"]["profile"] == body["assessment"]
    assert body["normalized_intent_envelope"]["graph"] == body["intent_graph"]


def test_intent_clarify_endpoint_uses_intent_context_projection_for_slots(monkeypatch):
    monkeypatch.setattr(
        main,
        "_assess_goal_intent",
        lambda goal_text, mode_override=None: workflow_contracts.GoalIntentProfile(
            intent="generate",
            source="test",
            confidence=0.93,
            risk_level="read_only",
            threshold=0.7,
            low_confidence=False,
            needs_clarification=True,
            requires_blocking_clarification=True,
            questions=["What tone should I use?"],
            blocking_slots=["tone"],
            missing_slots=["tone"],
            slot_values={"intent_action": "generate", "topic": "Deployment report"},
        ),
    )

    response = client.post(
        "/intent/clarify",
        json={
            "goal": "Generate a deployment report",
            "context_json": {
                "tone": "practical",
                "clarification_normalization": {"fields": ["tone"]},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assessment"]["needs_clarification"] is False
    assert body["assessment"]["missing_slots"] == []
    assert body["assessment"]["slot_values"]["tone"] == "practical"
    assert body["normalized_intent_envelope"]["trace"]["context_projection"] == "intent"
    assert (
        body["normalized_intent_envelope"]["trace"]["context_slot_provenance"]["tone"]
        == "clarification_normalized"
    )


def test_capabilities_search_returns_ranked_matches() -> None:
    response = client.post(
        "/capabilities/search",
        json={"query": "render a pdf report from a document spec", "intent": "render", "limit": 5},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "render a pdf report from a document spec"
    assert body["intent"] == "render"
    assert len(body["items"]) >= 1
    ids = [item["id"] for item in body["items"]]
    assert "document.pdf.render" in ids
    first = body["items"][0]
    assert isinstance(first["score"], float)
    assert isinstance(first["reason"], str) and first["reason"]


def test_capabilities_search_emits_capability_search_event(monkeypatch) -> None:
    events_seen = []
    monkeypatch.setattr(
        main,
        "_emit_event",
        lambda event_type, payload: events_seen.append((event_type, payload)),
    )

    response = client.post(
        "/capabilities/search",
        json={
            "query": "render a pdf report from a document spec",
            "intent": "render",
            "limit": 5,
            "request_source": "composer",
            "correlation_id": "corr-search-1",
            "job_id": "job-search-1",
        },
    )
    assert response.status_code == 200
    assert events_seen
    event_type, payload = events_seen[-1]
    assert event_type == "plan.capability_search"
    assert payload["request_source"] == "composer"
    assert payload["correlation_id"] == "corr-search-1"
    assert payload["job_id"] == "job-search-1"
    assert payload["result_count"] >= 1
    assert any(item["id"] == "document.pdf.render" for item in payload["results"])


def test_capabilities_search_requires_query() -> None:
    response = client.post("/capabilities/search", json={"query": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "query_required"


def test_capabilities_search_validates_limit() -> None:
    response = client.post("/capabilities/search", json={"query": "pdf", "limit": 0})
    assert response.status_code == 400
    assert response.json()["detail"] == "limit_out_of_range"


def test_intent_decompose_endpoint_uses_llm_when_enabled(monkeypatch):
    requests: list[LLMRequest] = []

    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            requests.append(request)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.92,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate"],'
                    '"slots":{"entity":"summary","artifact_type":"content","output_format":"txt",'
                    '"risk_level":"read_only","must_have_inputs":["instruction"]}}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["source"] == "llm"
    assert graph["summary"]["segment_count"] == 1
    assert graph["segments"][0]["intent"] == "generate"
    assert graph["segments"][0]["source"] == "llm"
    assert graph["segments"][0]["slots"]["entity"] == "summary"
    assert graph["segments"][0]["slots"]["must_have_inputs"] == ["instruction"]
    assert requests
    assert requests[0].metadata is not None
    assert requests[0].metadata["operation"] == "intent_decompose"


def test_intent_decompose_endpoint_falls_back_to_heuristic_on_llm_failure(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(content="not-json")

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "hybrid")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Validate schema and render PDF"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["source"] == "heuristic"
    assert graph["summary"]["segment_count"] >= 1


def test_intent_decompose_endpoint_filters_unknown_llm_capabilities(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["cap.web_fetch","cap.unknown"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    suggested = segment["suggested_capabilities"]
    assert "cap.web_fetch" not in suggested
    assert "cap.unknown" not in suggested
    assert len(suggested) >= 1
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == len(suggested)
    assert all(isinstance(entry.get("reason"), str) and entry.get("reason") for entry in rankings)
    assert all(entry.get("source") in {"llm", "fallback_segment", "heuristic"} for entry in rankings)
    catalog_ids = main._intent_catalog_capability_ids()
    assert all(capability_id in catalog_ids for capability_id in suggested)


def test_intent_decompose_endpoint_reconciles_legacy_render_capability_intent(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Render PDF",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["document_spec","path"],'
                    '"suggested_capabilities":["document.pdf.generate"],'
                    '"slots":{"entity":"artifact","artifact_type":"document","output_format":"pdf",'
                    '"risk_level":"bounded_write","must_have_inputs":["document_spec","path"]}}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Render a PDF report"})

    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    assert segment["intent"] == "render"
    assert segment["suggested_capabilities"] == ["document.pdf.render"]
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert rankings[0]["id"] == "document.pdf.render"


def test_intent_decompose_endpoint_normalizes_capability_id_casing(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["LLM.TEXT.GENERATE"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    assert segment["suggested_capabilities"] == ["llm.text.generate"]
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == 1
    assert rankings[0]["id"] == "llm.text.generate"
    assert rankings[0]["source"] == "llm"
    assert isinstance(rankings[0]["reason"], str) and rankings[0]["reason"]


def test_intent_decompose_endpoint_limits_capability_rankings_to_top_k(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create summary",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate","document.spec.generate",'
                    '"document.spec.validate","document.pdf.render"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "INTENT_CAPABILITY_TOP_K", 3)
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post("/intent/decompose", json={"goal": "Create summary"})
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    rankings = segment.get("suggested_capability_rankings")
    assert isinstance(rankings, list)
    assert len(rankings) == 3
    assert len(segment["suggested_capabilities"]) == 3
    assert segment["suggested_capabilities"] == [entry["id"] for entry in rankings]
    assert graph["summary"]["capability_top_k"] == 3


def test_intent_decompose_endpoint_validates_interaction_summaries_contract():
    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [{"facts": ["captured fact only"]}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "interaction_summaries[0].action_required"


def test_intent_decompose_endpoint_filters_unsupported_facts_with_summaries(monkeypatch):
    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Build report",'
                    '"objective_facts":["collect relevant sources","include pricing analysis"],'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["llm.text.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [
                {
                    "id": "i1",
                    "facts": ["collect relevant sources"],
                    "action": "open search",
                }
            ],
        },
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    segment = graph["segments"][0]
    assert segment["objective"] == "collect relevant sources"
    assert "include pricing analysis" in segment.get("unsupported_facts", [])
    summary = graph["summary"]
    assert summary["has_interaction_summaries"] is True
    assert summary["fact_candidates"] == 2
    assert summary["fact_supported"] == 1
    assert summary["fact_stripped"] == 1


def test_intent_decompose_endpoint_reports_compaction_metadata(monkeypatch):
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_MAX_ITEMS", 1)
    response = client.post(
        "/intent/decompose",
        json={
            "goal": "Create summary",
            "interaction_summaries": [
                {"id": "i1", "facts": ["first fact"], "action": "open tab"},
                {"id": "i2", "facts": ["second fact"], "action": "click button"},
            ],
        },
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    summary = graph["summary"]
    compaction = summary["interaction_summary_compaction"]
    assert compaction["input_count"] == 2
    assert compaction["output_count"] == 1
    assert compaction["applied"] is True


def test_create_job_requires_intent_clarification_when_confidence_is_low():
    response = client.post(
        "/jobs?require_clarification=true",
        json={"goal": "help", "context_json": {}, "priority": 0},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "intent_clarification_required"
    assert detail["goal_intent_profile"]["needs_clarification"] is True


def test_create_job_clarification_gate_blocks_capability_required_path(monkeypatch):
    monkeypatch.setattr(main, "INTENT_MIN_CONFIDENCE", 0.99)
    monkeypatch.setattr(main, "INTENT_CLARIFICATION_BLOCKING_SLOTS", {"output_format"})
    response = client.post(
        "/jobs?require_clarification=true",
        json={"goal": "Render a PDF deployment report", "context_json": {}, "priority": 0},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    profile = detail["goal_intent_profile"]
    assert profile["low_confidence"] is True
    assert profile["missing_slots"] == ["path"]
    assert profile["requires_blocking_clarification"] is True


def test_create_job_rejects_invalid_interaction_summaries_in_context():
    response = client.post(
        "/jobs",
        json={
            "goal": "Create report",
            "context_json": {"interaction_summaries": [{"action": "open page"}]},
            "priority": 0,
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "interaction_summaries[0].facts_required"


def test_create_job_persists_raw_and_compact_interaction_summaries(monkeypatch):
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(main, "INTERACTION_SUMMARY_MAX_ITEMS", 2)
    response = client.post(
        "/jobs",
        json={
            "goal": "Create report",
            "context_json": {
                "interaction_summaries": [
                    {"id": "i1", "facts": ["first fact"], "action": "open"},
                    {"id": "i2", "facts": ["second fact"], "action": "click"},
                    {"id": "i3", "facts": ["third fact"], "action": "submit"},
                ]
            },
            "priority": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    job_id = body["id"]
    compacted = body["context_json"]["interaction_summaries"]
    assert len(compacted) <= 2
    assert body["context_json"]["interaction_summaries_ref"]["memory_name"] == "interaction_summaries_compact"

    raw_read = client.get("/memory/read", params={"name": "interaction_summaries", "job_id": job_id})
    assert raw_read.status_code == 200
    raw_entries = raw_read.json()
    assert len(raw_entries) >= 1
    assert len(raw_entries[0]["payload"]["interaction_summaries"]) == 3

    compact_read = client.get(
        "/memory/read",
        params={"name": "interaction_summaries_compact", "job_id": job_id},
    )
    assert compact_read.status_code == 200
    compact_entries = compact_read.json()
    assert len(compact_entries) >= 1
    assert len(compact_entries[0]["payload"]["interaction_summaries"]) <= 2


def test_create_plan_persists_task_intent_profiles_and_surfaces_on_tasks():
    job_response = client.post(
        "/jobs",
        json={"goal": "render a deployment document", "context_json": {}, "priority": 1},
    )
    assert job_response.status_code == 200
    job_id = job_response.json()["id"]

    plan_payload = {
        "planner_version": "test",
        "tasks_summary": "render one file",
        "dag_edges": [],
        "tasks": [
            {
                "name": "RenderDoc",
                "description": "Render output to PDF",
                "instruction": "Render the final document",
                "acceptance_criteria": ["returns output path"],
                "expected_output_schema_ref": "schemas/output_path",
                "deps": [],
                "tool_requests": [],
                "tool_inputs": {},
                "critic_required": False,
            }
        ],
    }
    plan_response = client.post(f"/plans?job_id={job_id}", json=plan_payload)
    assert plan_response.status_code == 200

    with SessionLocal() as db:
        job_record = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert job_record is not None
        metadata = job_record.metadata_json or {}
        profiles = metadata.get("task_intent_profiles")
        assert isinstance(profiles, dict)
        assert len(profiles) == 1
        only_profile = next(iter(profiles.values()))
        assert only_profile["intent"] == "render"
        assert only_profile["source"] in {"task_text", "goal_text", "explicit", "default"}
        assert 0.0 <= float(only_profile["confidence"]) <= 1.0

    tasks_response = client.get(f"/jobs/{job_id}/tasks")
    assert tasks_response.status_code == 200
    tasks = tasks_response.json()
    assert len(tasks) == 1
    assert tasks[0]["intent_source"] is not None
    assert isinstance(tasks[0]["intent_confidence"], (int, float))


def test_memory_specs_endpoint_lists_user_scoped_specs() -> None:
    response = client.get("/memory/specs")
    assert response.status_code == 200
    specs = response.json()
    names = {entry["name"] for entry in specs}
    assert "user_profile" in names
    assert "semantic_memory" in names


def test_delete_user_memory_entry() -> None:
    write_response = client.post(
        "/memory/write",
        json={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
            "payload": {"theme": "light"},
            "metadata": {"source": "test"},
        },
    )
    assert write_response.status_code == 200

    delete_response = client.delete(
        "/memory/delete",
        params={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
        },
    )
    assert delete_response.status_code == 200
    deleted = delete_response.json()
    assert deleted["name"] == "user_profile"
    assert deleted["user_id"] == "demo-user"
    assert deleted["key"] == "preferences"

    read_response = client.get(
        "/memory/read",
        params={
            "name": "user_profile",
            "scope": "user",
            "user_id": "demo-user",
            "key": "preferences",
        },
    )
    assert read_response.status_code == 200
    assert read_response.json() == []


def test_create_job_persists_goal_intent_graph():
    response = client.post(
        "/jobs",
        json={"goal": "Fetch data then validate and render PDF", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert "goal_intent_graph" in metadata
    assert "normalized_intent_envelope" in metadata
    graph = metadata["goal_intent_graph"]
    envelope = metadata["normalized_intent_envelope"]
    assert graph["summary"]["segment_count"] >= 2
    assert envelope["graph"]["summary"]["segment_count"] == graph["summary"]["segment_count"]


def test_intent_decompose_uses_semantic_workflow_hints_in_llm_prompt(monkeypatch):
    user_id = f"intent-hints-{uuid.uuid4()}"
    write = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "intent_workflows",
            "subject": "pdf_report_flow",
            "fact": "Successful workflow for PDF report generation.",
            "confidence": 0.95,
            "metadata": {"from_test": True},
            "source": "test",
            "key": f"intent_workflow:{uuid.uuid4()}",
            "reasoning": "stored from prior successful run",
            "keywords": ["pdf", "report", "workflow"],
            "aliases": ["render pdf report"],
        },
    )
    assert write.status_code == 200
    with SessionLocal() as db:
        entry = memory_store.read_memory(
            db,
            models.MemoryQuery(
                name="semantic_memory",
                scope=models.MemoryScope.user,
                user_id=user_id,
                key=write.json()["entry"]["key"],
                limit=1,
            ),
        )[0]
        payload = dict(entry.payload)
        payload["intent_workflow"] = {
            "job_id": "job-prev",
            "goal": "Render monthly pdf report",
            "outcome": "succeeded",
            "intent_order": ["generate", "render"],
            "capabilities": ["document.spec.generate", "document.pdf.render"],
            "confidence": 0.9,
            "threshold": 0.7,
        }
        memory_store.write_memory(
            db,
            models.MemoryWrite(
                name="semantic_memory",
                scope=models.MemoryScope.user,
                user_id=user_id,
                key=entry.key,
                payload=payload,
                metadata=entry.metadata,
            ),
        )

    prompts: list[str] = []

    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            prompts.append(prompt)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"generate","objective":"Create report",'
                    '"confidence":0.91,"depends_on":[],"required_inputs":["instruction"],'
                    '"suggested_capabilities":["document.spec.generate"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())
    monkeypatch.setattr(main, "INTENT_MEMORY_RETRIEVAL_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_MEMORY_RETRIEVAL_LIMIT", 3)

    response = client.post(
        "/intent/decompose",
        json={"goal": "Create a PDF report", "user_id": user_id},
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["summary"]["memory_hints_used"] >= 1
    assert prompts
    assert "Similar successful workflow hints from semantic memory" in prompts[0]


def test_intent_decompose_uses_semantic_capability_hints_in_llm_prompt(monkeypatch):
    prompts: list[str] = []

    class _Provider(LLMProvider):
        def generate(self, prompt: str):  # noqa: ARG002
            prompts.append(prompt)
            return LLMResponse(
                content=(
                    '{"segments":[{"id":"s1","intent":"render","objective":"Render pdf report",'
                    '"confidence":0.9,"depends_on":[],"required_inputs":["document_spec","path"],'
                    '"suggested_capabilities":["document.pdf.render"]}]}'
                )
            )

    monkeypatch.setattr(main, "INTENT_DECOMPOSE_ENABLED", True)
    monkeypatch.setattr(main, "INTENT_DECOMPOSE_MODE", "llm")
    monkeypatch.setattr(main, "_intent_decompose_provider", _Provider())

    response = client.post(
        "/intent/decompose",
        json={"goal": "Render a PDF report from a document spec"},
    )
    assert response.status_code == 200
    graph = response.json()["intent_graph"]
    assert graph["summary"]["semantic_capability_hints_used"] >= 1
    assert prompts
    assert "Most relevant capabilities for this goal from local semantic search" in prompts[0]
    assert "document.pdf.render" in prompts[0]


def test_intent_decompose_emits_capability_search_event_for_semantic_hints(monkeypatch):
    events_seen = []
    monkeypatch.setattr(
        main,
        "_emit_event",
        lambda event_type, payload: events_seen.append((event_type, payload)),
    )

    graph = main._decompose_goal_intent("Search semantic memory for user preferences")

    assert graph.summary.semantic_capability_hints_used is not None
    assert graph.summary.semantic_capability_hints_used >= 1
    assert events_seen
    event_type, payload = events_seen[-1]
    assert event_type == "plan.capability_search"
    assert payload["request_source"] == "intent_decompose"
    assert payload["query"] == "Search semantic memory for user preferences"
    assert payload["result_count"] >= 1


def test_refresh_job_status_persists_intent_outcome_memory_and_calibration():
    create = client.post(
        "/jobs",
        json={
            "goal": "Generate and render a PDF report",
            "context_json": {"user_id": f"intent-loop-{uuid.uuid4()}"},
            "priority": 0,
        },
    )
    assert create.status_code == 200
    job = create.json()
    job_id = job["id"]
    user_id = job["context_json"]["user_id"]

    plan = client.post(
        f"/plans?job_id={job_id}",
        json={
            "planner_version": "test",
            "tasks_summary": "single step",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "GenerateSpec",
                    "description": "create spec",
                    "instruction": "create document spec",
                    "acceptance_criteria": ["spec exists"],
                    "expected_output_schema_ref": "schemas/document_spec",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["document.spec.generate"],
                    "tool_inputs": {"document.spec.generate": {"instruction": "Create report spec"}},
                    "critic_required": False,
                }
            ],
        },
    )
    assert plan.status_code == 200
    tasks = client.get(f"/jobs/{job_id}/tasks")
    assert tasks.status_code == 200
    task_id = tasks.json()[0]["id"]

    main._handle_task_completed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "outputs": {"document.spec.generate": {"document_spec": {"blocks": []}}},
            },
        }
    )

    job_after = client.get(f"/jobs/{job_id}")
    assert job_after.status_code == 200
    metadata = job_after.json()["metadata"]
    assert metadata.get("intent_confidence_outcome_recorded") is True
    assert metadata.get("intent_memory_persisted") is True

    semantic = client.get(
        "/memory/read",
        params={"name": "semantic_memory", "user_id": user_id, "key": f"intent_workflow:{job_id}"},
    )
    assert semantic.status_code == 200
    entries = semantic.json()
    assert len(entries) == 1
    payload = entries[0]["payload"]
    assert payload["namespace"] == "intent_workflows"
    assert payload["intent_workflow"]["outcome"] == "succeeded"


def test_contract_intent_mismatch_triggers_auto_replan_with_recovery_metadata():
    now = _utcnow()
    job_id = str(uuid.uuid4())
    plan_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="List github repos",
                context_json={},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="one task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="ListRepos",
                description="list repos",
                instruction="list repos from github",
                acceptance_criteria=["repos listed"],
                expected_output_schema_ref="schemas/unknown",
                status=models.TaskStatus.running.value,
                deps=[],
                attempts=0,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                intent="generate",
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "octocat"}},
                created_at=now,
                updated_at=now,
                critic_required=False,
            )
        )
        db.commit()

    main._handle_task_failed(
        {
            "task_id": task_id,
            "payload": {
                "task_id": task_id,
                "error": "contract.intent_mismatch:task_intent_mismatch:github.repo.list:generate:allowed=io",
            },
        }
    )

    job_after = client.get(f"/jobs/{job_id}")
    assert job_after.status_code == 200
    body = job_after.json()
    assert body["status"] == "planning"
    metadata = body["metadata"]
    assert metadata.get("replan_reason") == "intent_mismatch_auto_repair"
    recovery = metadata.get("intent_mismatch_recovery")
    assert isinstance(recovery, dict)
    assert recovery.get("failing_capability") == "github.repo.list"
    assert recovery.get("allowed_task_intents") == ["io"]

def test_semantic_memory_write_and_read_round_trip():
    user_id = f"semantic-user-{uuid.uuid4()}"
    response = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "user_prefs",
            "subject": "response_style",
            "fact": "User prefers concise answers with concrete next steps.",
            "keywords": ["concise", "actionable"],
            "confidence": 0.92,
            "source": "ui_manual",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["entry"]["name"] == "semantic_memory"
    assert body["entry"]["scope"] == "user"
    assert body["entry"]["user_id"] == user_id
    assert body["semantic_record"]["subject"] == "response_style"

    read_response = client.get(
        "/memory/read",
        params={"name": "semantic_memory", "user_id": user_id, "limit": 10},
    )
    assert read_response.status_code == 200
    entries = read_response.json()
    assert len(entries) >= 1
    assert any(
        isinstance(entry.get("payload"), dict)
        and "concise answers" in str(entry["payload"].get("fact", "")).lower()
        for entry in entries
    )


def test_semantic_memory_search_returns_ranked_match():
    user_id = f"semantic-search-{uuid.uuid4()}"
    write_a = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "runtime",
            "subject": "mcp_reliability",
            "fact": "Use retries with bounded backoff when MCP calls timeout.",
            "keywords": ["mcp", "retries", "timeout", "backoff"],
            "confidence": 0.95,
            "source": "api_test",
        },
    )
    assert write_a.status_code == 200
    write_b = client.post(
        "/memory/semantic/write",
        json={
            "user_id": user_id,
            "namespace": "ui",
            "subject": "dashboard_theme",
            "fact": "Use neutral color palettes for operational dashboards.",
            "keywords": ["ui", "color"],
            "confidence": 0.8,
            "source": "api_test",
        },
    )
    assert write_b.status_code == 200

    search = client.post(
        "/memory/semantic/search",
        json={
            "user_id": user_id,
            "query": "mcp timeout retries with backoff",
            "limit": 5,
            "min_score": 0.05,
        },
    )
    assert search.status_code == 200
    body = search.json()
    assert body["count"] >= 1
    top = body["matches"][0]
    assert "retries" in str(top.get("fact", "")).lower()
    assert float(top.get("score", 0)) > 0


def test_emit_event_persists_outbox_when_redis_is_unavailable(monkeypatch):
    job_id = f"job-outbox-down-{uuid.uuid4()}"

    class _RedisDown:
        def xadd(self, stream, payload):
            raise redis.RedisError("redis down")

    monkeypatch.setattr(main, "redis_client", _RedisDown())
    monkeypatch.setattr(main, "EVENT_OUTBOX_ENABLED", True)
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.commit()

    main._emit_event(
        "job.created",
        {
            "id": job_id,
            "job_id": job_id,
            "goal": "outbox fallback",
            "context_json": {},
            "status": models.JobStatus.queued.value,
            "priority": 0,
            "metadata": {},
            "created_at": _utcnow().isoformat(),
            "updated_at": _utcnow().isoformat(),
        },
    )

    with SessionLocal() as db:
        rows = db.query(EventOutboxRecord).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.stream == events.JOB_STREAM
        assert row.event_type == "job.created"
        assert row.published_at is None
        assert (row.attempts or 0) >= 1
        assert row.last_error is not None


def test_dispatch_event_outbox_once_publishes_pending_rows(monkeypatch):
    outbox_id = f"outbox-{uuid.uuid4()}"
    now = _utcnow()
    sent = []

    class _RedisOk:
        def xadd(self, stream, payload):
            sent.append((stream, payload))
            return "1-0"

    monkeypatch.setattr(main, "redis_client", _RedisOk())
    monkeypatch.setattr(main, "EVENT_OUTBOX_ENABLED", True)
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.add(
            EventOutboxRecord(
                id=outbox_id,
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "version": "1",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "job_id": "job-x",
                    "task_id": "task-x",
                    "payload": {"task_id": "task-x"},
                },
                attempts=0,
                last_error=None,
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.commit()

    dispatched = main._dispatch_event_outbox_once()
    assert dispatched == 1
    assert len(sent) == 1
    assert sent[0][0] == events.TASK_STREAM

    with SessionLocal() as db:
        row = db.query(EventOutboxRecord).filter(EventOutboxRecord.id == outbox_id).first()
        assert row is not None
        assert row.published_at is not None
        assert row.last_error is None
        assert (row.attempts or 0) >= 1


def test_event_stream():
    response = client.get("/events/stream?once=true")
    assert response.status_code == 200


def test_job_event_outbox_filters_by_job_and_pending_state():
    job_id = f"job-outbox-view-{uuid.uuid4()}"
    other_job_id = f"job-outbox-other-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.query(EventOutboxRecord).delete()
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": job_id,
                    "task_id": "task-1",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-1"},
                },
                attempts=2,
                last_error="redis down",
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": other_job_id,
                    "task_id": "task-2",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-2"},
                },
                attempts=1,
                last_error="redis down",
                created_at=now,
                updated_at=now,
                published_at=None,
            )
        )
        db.add(
            EventOutboxRecord(
                id=str(uuid.uuid4()),
                stream=events.TASK_STREAM,
                event_type="task.ready",
                envelope_json={
                    "type": "task.ready",
                    "job_id": job_id,
                    "task_id": "task-3",
                    "occurred_at": now.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "payload": {"task_id": "task-3"},
                },
                attempts=1,
                last_error=None,
                created_at=now,
                updated_at=now,
                published_at=now,
            )
        )
        db.commit()

    response = client.get(f"/jobs/{job_id}/events/outbox?pending_only=true&limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["pending_only"] is True
    assert body["count"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["job_id"] == job_id
    assert body["items"][0]["published_at"] is None


def test_job_details():
    job_id = f"job-details-{uuid.uuid4()}"
    plan_id = f"plan-details-{uuid.uuid4()}"
    task_id = f"task-details-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="Render the detail report",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={
                    "normalized_intent_envelope": {
                        "goal": "Render the detail report",
                        "profile": {
                            "intent": "render",
                            "source": "llm",
                            "needs_clarification": True,
                            "requires_blocking_clarification": True,
                            "missing_slots": ["path"],
                            "blocking_slots": ["path"],
                            "questions": ["What output path or filename should be used?"],
                            "slot_values": {"intent_action": "render", "output_format": "pdf"},
                            "clarification_mode": "capability_required_inputs",
                        },
                        "graph": {
                            "segments": [
                                {
                                    "id": "s1",
                                    "intent": "render",
                                    "objective": "Render the final PDF",
                                    "suggested_capabilities": ["document.pdf.render"],
                                }
                            ]
                        },
                        "candidate_capabilities": {"s1": ["document.pdf.render"]},
                        "clarification": {
                            "needs_clarification": True,
                            "requires_blocking_clarification": True,
                            "missing_inputs": ["path"],
                            "questions": ["What output path or filename should be used?"],
                            "blocking_slots": ["path"],
                            "slot_values": {"intent_action": "render", "output_format": "pdf"},
                            "clarification_mode": "capability_required_inputs",
                        },
                        "trace": {
                            "assessment_source": "llm",
                            "assessment_mode": "hybrid",
                            "decomposition_source": "llm",
                            "decomposition_mode": "llm",
                        },
                    }
                },
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="one task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="t1",
                description="desc",
                instruction="do it",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.pending.value,
                deps=[],
                attempts=0,
                max_attempts=1,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=1,
            )
        )
        db.commit()

    response = client.get(f"/jobs/{job_id}/details")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["plan"]["id"] == plan_id
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["id"] == task_id
    assert task_id in data["task_results"]
    assert data["normalized_intent_envelope"]["goal"] == "Render the detail report"
    assert data["goal_intent_profile"]["intent"] == "render"
    assert data["goal_intent_graph"]["segments"][0]["id"] == "s1"
    assert data["normalization_trace"]["assessment_source"] == "llm"
    assert data["normalization_clarification"]["missing_inputs"] == ["path"]
    assert data["normalization_candidate_capabilities"] == {"s1": ["document.pdf.render"]}


def test_job_debugger_returns_timeline_and_error_classification():
    job_id = f"job-debug-{uuid.uuid4()}"
    plan_id = f"plan-debug-{uuid.uuid4()}"
    task_id = f"task-debug-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="debug",
                context_json={"job": {"topic": "latency"}},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={
                    "goal_intent_profile": {
                        "intent": "io",
                        "source": "heuristic",
                        "needs_clarification": True,
                        "requires_blocking_clarification": True,
                        "missing_slots": ["target_system"],
                        "blocking_slots": ["target_system"],
                        "questions": ["Which target system should this use?"],
                        "slot_values": {"intent_action": "io"},
                        "clarification_mode": "targeted_slot_filling",
                    },
                    "goal_intent_graph": {
                        "segments": [
                            {
                                "id": "s1",
                                "intent": "io",
                                "objective": "List repositories",
                                "suggested_capabilities": ["github.repo.list"],
                            }
                        ]
                    },
                },
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single debug task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="GenerateSpec",
                description="desc",
                instruction="do it",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.failed.value,
                deps=[],
                attempts=2,
                max_attempts=3,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=["document.spec.generate"],
                tool_inputs={"document.spec.generate": {"job": {"topic": "latency"}}},
                created_at=now,
                updated_at=now,
                critic_required=1,
            )
        )
        db.commit()

    started_envelope = models.EventEnvelope(
        type="task.started",
        version="1",
        occurred_at=now,
        correlation_id=f"corr-start-{uuid.uuid4()}",
        job_id=job_id,
        task_id=task_id,
        payload={
            "task_id": task_id,
            "attempts": 2,
            "max_attempts": 3,
            "worker_consumer": "worker-a",
            "run_id": "run-1",
        },
    ).model_dump(mode="json")
    failed_envelope = models.EventEnvelope(
        type="task.failed",
        version="1",
        occurred_at=now + timedelta(seconds=1),
        correlation_id=f"corr-fail-{uuid.uuid4()}",
        job_id=job_id,
        task_id=task_id,
        payload={
            "task_id": task_id,
            "status": models.TaskStatus.failed.value,
            "outputs": {
                "tool_error": {
                    "error": "contract.input_missing:job",
                    "error_code": "contract.input_missing",
                }
            },
            "artifacts": [
                {
                    "type": "run_scorecard",
                    "summary": {"total_latency_ms": 125, "failure_stage": "task_execution"},
                }
            ],
            "tool_calls": [
                models.ToolCall(
                    tool_name="document.spec.generate",
                    input={"job": {"topic": "latency"}},
                    idempotency_key="debug-call-1",
                    trace_id="trace-debug-1",
                    request_id="document.spec.generate",
                    capability_id="document.spec.generate",
                    adapter_id="local_tool:document.spec.generate",
                    started_at=now,
                    finished_at=now + timedelta(milliseconds=125),
                    status="failed",
                    output_or_error={
                        "error": "contract.input_missing:job",
                        "error_code": "contract.input_missing",
                    },
                ).model_dump(mode="json")
            ],
            "started_at": now.isoformat(),
            "finished_at": (now + timedelta(seconds=1)).isoformat(),
            "error": "contract.input_missing:job",
            "attempts": 2,
            "max_attempts": 3,
            "worker_consumer": "worker-a",
            "run_id": "run-1",
        },
    ).model_dump(mode="json")

    main._handle_event(events.TASK_STREAM, {"data": json.dumps(started_envelope)})
    main._handle_event(events.TASK_STREAM, {"data": json.dumps(failed_envelope)})

    with SessionLocal() as db:
        step_attempt_rows = db.query(StepAttemptRecord).filter(StepAttemptRecord.job_id == job_id).all()
        invocation_rows = db.query(InvocationRecord).filter(InvocationRecord.job_id == job_id).all()
        run_event_rows = (
            db.query(RunEventRecord)
            .filter(RunEventRecord.job_id == job_id)
            .order_by(RunEventRecord.occurred_at.asc())
            .all()
        )
    assert len(step_attempt_rows) == 1
    assert step_attempt_rows[0].attempt_number == 2
    assert step_attempt_rows[0].status == models.TaskStatus.failed.value
    assert step_attempt_rows[0].retry_classification == "terminal_contract"
    assert len(invocation_rows) == 1
    assert invocation_rows[0].capability_id == "document.spec.generate"
    assert len(run_event_rows) == 2
    assert [row.event_type for row in run_event_rows] == ["task.started", "task.failed"]

    attempts_response = client.get(f"/jobs/{job_id}/debugger/attempts")
    assert attempts_response.status_code == 200
    attempts_payload = attempts_response.json()
    assert len(attempts_payload) == 1
    assert attempts_payload[0]["attempt_number"] == 2

    invocations_response = client.get(f"/jobs/{job_id}/debugger/invocations")
    assert invocations_response.status_code == 200
    invocations_payload = invocations_response.json()
    assert len(invocations_payload) == 1
    assert invocations_payload[0]["request_id"] == "document.spec.generate"

    events_response = client.get(f"/jobs/{job_id}/debugger/events")
    assert events_response.status_code == 200
    events_payload = events_response.json()
    assert len(events_payload) == 2
    assert [entry["event_type"] for entry in events_payload] == ["task.started", "task.failed"]

    response = client.get(f"/jobs/{job_id}/debugger")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job_id
    assert payload["job_status"] == "failed"
    assert payload["plan_id"] == plan_id
    assert payload["timeline_events_scanned"] == 2
    assert payload["goal_intent_profile"]["intent"] == "io"
    assert payload["goal_intent_graph"]["segments"][0]["id"] == "s1"
    assert payload["normalized_intent_envelope"]["goal"] == "debug"
    assert payload["normalization_trace"] == {}
    assert payload["normalization_clarification"]["missing_inputs"] == ["target_system"]
    assert payload["normalization_candidate_capabilities"] == {"s1": ["github.repo.list"]}
    assert len(payload["tasks"]) == 1
    task_payload = payload["tasks"][0]
    assert task_payload["task"]["id"] == task_id
    assert task_payload["tool_inputs_resolved"] is True
    assert task_payload["error"]["category"] == "contract"
    assert task_payload["error"]["retryable"] is False
    assert len(task_payload["timeline"]) == 2
    assert len(task_payload["attempts"]) == 1
    assert task_payload["attempts"][0]["attempt_number"] == 2
    assert task_payload["attempts"][0]["status"] == "failed"
    assert len(task_payload["attempts"][0]["invocations"]) == 1
    assert task_payload["attempts"][0]["invocations"][0]["capability_id"] == "document.spec.generate"


def test_plan_created_enqueues_ready_tasks():
    job_id = f"job-test-plan-{uuid.uuid4()}"
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="demo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=_utcnow(),
                updated_at=_utcnow(),
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="t1 then t2",
        dag_edges=[["t1", "t2"]],
        tasks=[
            models.TaskCreate(
                name="t1",
                description="first",
                instruction="do first",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
            models.TaskCreate(
                name="t2",
                description="second",
                instruction="do second",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=["t1"],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
        ],
    )
    payload = plan.model_dump()
    payload["job_id"] = job_id
    envelope = {
        "type": "plan.created",
        "payload": payload,
        "job_id": job_id,
        "correlation_id": "corr",
    }
    events: list[tuple[str, dict]] = []
    original_emit = main._emit_event
    try:
        main._emit_event = lambda event_type, event_payload: events.append(
            (event_type, event_payload)
        )
        main._handle_plan_created(envelope)
    finally:
        main._emit_event = original_emit
    with SessionLocal() as db:
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        by_name = {task.name: task for task in tasks}
        assert by_name["t1"].status == models.TaskStatus.ready.value
        assert by_name["t2"].status == models.TaskStatus.pending.value
    assert any(event_type == "task.ready" for event_type, _ in events)


def test_plan_created_with_run_spec_uses_postgres_scheduler_for_planner_jobs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(main, "PLANNER_RUN_SCHEDULER_ENABLED", True)
    job_id = f"job-test-planner-run-spec-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="demo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="t1 then t2",
        dag_edges=[["t1", "t2"]],
        tasks=[
            models.TaskCreate(
                name="t1",
                description="first",
                instruction="do first",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
            models.TaskCreate(
                name="t2",
                description="second",
                instruction="do second",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=["t1"],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
        ],
    )
    run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.planner)
    envelope = {
        "type": "plan.created",
        "payload": {"job_id": job_id, "run_spec": run_spec.model_dump(mode="json")},
        "job_id": job_id,
        "correlation_id": "corr-planner",
    }
    events: list[tuple[str, dict]] = []
    original_emit = main._emit_event
    try:
        main._emit_event = lambda event_type, event_payload: events.append(
            (event_type, event_payload)
        )
        main._handle_plan_created(envelope)
    finally:
        main._emit_event = original_emit

    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).one()
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        by_name = {task.name: task for task in tasks}
        assert by_name["t1"].status == models.TaskStatus.ready.value
        assert by_name["t2"].status == models.TaskStatus.pending.value
        assert job.metadata_json["scheduler_mode"] == main.POSTGRES_RUN_SPEC_SCHEDULER_MODE
        stored_run_spec = run_specs.parse_run_spec(job.metadata_json.get("run_spec"))
        assert stored_run_spec is not None
        assert stored_run_spec.kind == models.RunKind.planner
    assert any(event_type == "task.ready" for event_type, _ in events)


def test_planner_scheduler_advances_dependency_from_durable_step_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "PLANNER_RUN_SCHEDULER_ENABLED", True)
    job_id = f"job-test-planner-durable-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="demo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="t1 then t2",
        dag_edges=[["t1", "t2"]],
        tasks=[
            models.TaskCreate(
                name="t1",
                description="first",
                instruction="do first",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
            models.TaskCreate(
                name="t2",
                description="second",
                instruction="do second",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=["t1"],
                tool_requests=["filesystem.workspace.list"],
                critic_required=False,
            ),
        ],
    )
    run_spec = run_specs.plan_to_run_spec(plan, kind=models.RunKind.planner)
    main._handle_plan_created(
        {
            "type": "plan.created",
            "payload": {"job_id": job_id, "run_spec": run_spec.model_dump(mode="json")},
            "job_id": job_id,
            "correlation_id": "corr-initial",
        }
    )

    with SessionLocal() as db:
        plan_record = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).one()
        plan_id = plan_record.id
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        tasks_by_name = {task.name: task for task in tasks}
        source_task = tasks_by_name["t1"]
        target_task = tasks_by_name["t2"]
        assert source_task.status == models.TaskStatus.ready.value
        assert target_task.status == models.TaskStatus.pending.value
        source_task.status = models.TaskStatus.pending.value
        source_task.updated_at = now
        db.add(
            StepAttemptRecord(
                id=main._step_attempt_record_id(source_task.id, 1),
                run_id=job_id,
                job_id=job_id,
                step_id=source_task.id,
                attempt_number=1,
                status=models.TaskStatus.completed.value,
                worker_id="worker-a",
                started_at=now,
                finished_at=now,
                error_code=None,
                error_message=None,
                retry_classification="succeeded",
                result_summary_json={},
            )
        )
        for row in db.query(EventOutboxRecord).all():
            if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id:
                db.delete(row)
        db.commit()

    main._dispatch_ready_work_for_job(job_id, plan_id, "corr-durable-planner")

    with SessionLocal() as db:
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        tasks_by_name = {task.name: task for task in tasks}
        assert tasks_by_name["t1"].status == models.TaskStatus.completed.value
        assert tasks_by_name["t2"].status == models.TaskStatus.ready.value
        assert tasks_by_name["t2"].attempts == 1
        event_records = db.query(EventOutboxRecord).all()
    matching_events = [
        row
        for row in event_records
        if isinstance(row.envelope_json, dict) and row.envelope_json.get("job_id") == job_id
    ]
    ready_payloads = [
        row.envelope_json.get("payload", {})
        for row in matching_events
        if row.event_type == "task.ready"
    ]
    assert len(ready_payloads) == 1
    assert ready_payloads[0]["name"] == "t2"


def test_handle_plan_created_persists_embedded_capability_bindings(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    job_id = f"job-test-bindings-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="check repo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="check repo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepo",
                description="Check repo",
                instruction="Check repo",
                acceptance_criteria=["checked"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}},
                capability_bindings={
                    "github.repo.list": {
                        "request_id": "github.repo.list",
                        "capability_id": "github.repo.list",
                        "tool_name": "github.repo.list",
                        "adapter_type": "mcp",
                    }
                },
                critic_required=False,
            )
        ],
    )
    envelope = {
        "type": "plan.created",
        "payload": {"job_id": job_id, **plan.model_dump(mode="json")},
        "job_id": job_id,
        "correlation_id": "corr",
    }

    main._handle_plan_created(envelope)

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).one()
        assert task.tool_inputs["github.repo.list"]["repo"] == "demo"
        assert (
            task.tool_inputs[main.execution_contracts.EXECUTION_BINDINGS_KEY]["github.repo.list"][
                "capability_id"
            ]
            == "github.repo.list"
        )


def test_handle_plan_failed_marks_queued_job_failed_and_exposes_details_error():
    job_id = f"job-plan-failed-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="planner fail test",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.commit()

    main._handle_plan_failed(
        {
            "type": "plan.failed",
            "job_id": job_id,
            "payload": {
                "job_id": job_id,
                "error": "Invalid plan generated: tool_intent_mismatch:test",
            },
        }
    )

    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert job is not None
        assert job.status == models.JobStatus.failed.value
        assert isinstance(job.metadata_json, dict)
        assert "tool_intent_mismatch" in str(job.metadata_json.get("plan_error", ""))

    response = client.get(f"/jobs/{job_id}/details")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_status"] == models.JobStatus.failed.value
    assert "tool_intent_mismatch" in str(payload.get("job_error", ""))


def test_handle_task_started_sets_task_running_and_job_running():
    job_id = f"job-task-started-{uuid.uuid4()}"
    plan_id = f"plan-task-started-{uuid.uuid4()}"
    task_id = f"task-task-started-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="task started",
                context_json={},
                status=models.JobStatus.planning.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.ready.value,
                deps=[],
                attempts=1,
                max_attempts=3,
                rework_count=0,
                max_reworks=0,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    envelope = {
        "type": "task.started",
        "job_id": job_id,
        "task_id": task_id,
        "payload": {
            "task_id": task_id,
            "attempts": 1,
            "max_attempts": 3,
            "worker_consumer": "worker-a",
        },
    }
    main._handle_task_started(envelope)

    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert job is not None
        assert task is not None
        assert job.status == models.JobStatus.running.value
        assert task.status == models.TaskStatus.running.value
        assert task.assigned_to == "worker-a"


def test_read_task_dlq_filters_by_job_and_respects_limit(monkeypatch):
    class _RedisStub:
        def xrevrange(self, stream, max_id, min_id, count=0):
            return [
                (
                    "11-0",
                    {
                        "data": '{"message_id":"m-1","job_id":"job-a","task_id":"t-1","error":"timed out","failed_at":"2026-02-14T00:00:00Z"}'
                    },
                ),
                (
                    "10-0",
                    {
                        "data": '{"message_id":"m-2","job_id":"job-b","task_id":"t-2","error":"hard failure"}'
                    },
                ),
                (
                    "9-0",
                    {
                        "data": '{"message_id":"m-3","job_id":"job-a","task_id":"t-3","error":"fatal"}'
                    },
                ),
            ]

    monkeypatch.setattr(main, "redis_client", _RedisStub())
    response = client.get("/jobs/job-a/tasks/dlq?limit=1")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["job_id"] == "job-a"
    assert payload[0]["message_id"] == "m-1"


def test_read_task_dlq_returns_503_on_redis_error(monkeypatch):
    class _RedisStub:
        def xrevrange(self, stream, max_id, min_id, count=0):
            raise redis.RedisError("down")

    monkeypatch.setattr(main, "redis_client", _RedisStub())
    response = client.get("/jobs/job-a/tasks/dlq?limit=5")
    assert response.status_code == 503
    assert response.json()["detail"].startswith("redis_error:")


def test_list_capabilities_returns_required_inputs(monkeypatch):
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
        group="github",
        subgroup="repositories",
        input_schema_ref="github_repo_list_capability_input",
        output_schema_ref=None,
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_local",
                tool_name="search_repositories",
            ),
        ),
        tags=("github",),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(
        main,
        "_load_schema_from_ref",
        lambda schema_ref: {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        },
    )
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "enabled"
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["id"] == "github.repo.list"
    assert item["group"] == "github"
    assert item["subgroup"] == "repositories"
    assert item["required_inputs"] == ["query"]


def test_capability_required_inputs_uses_repo_local_schema_fallback(monkeypatch):
    monkeypatch.setattr(main, "SCHEMA_REGISTRY_PATH", "/tmp/missing-schemas")
    required = main._capability_required_inputs_for_intent_normalization("github.issue.search")
    assert required == ["query"]


def test_composer_recommend_capabilities_heuristic(monkeypatch):
    spec_generate = cap_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_generate",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    spec_render = cap_registry.CapabilitySpec(
        capability_id="document.pdf.render",
        description="Render a PDF",
        risk_tier="bounded_write",
        idempotency="read",
        input_schema_ref="schema_render",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": spec_generate,
            "document.pdf.render": spec_render,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    def _schema_loader(schema_ref):
        if schema_ref == "schema_generate":
            return {
                "type": "object",
                "required": ["topic"],
                "properties": {"topic": {"type": "string"}},
            }
        if schema_ref == "schema_render":
            return {
                "type": "object",
                "required": ["document_spec", "path"],
                "properties": {
                    "document_spec": {"type": "object"},
                    "path": {"type": "string"},
                },
            }
        return None

    monkeypatch.setattr(main, "_load_schema_from_ref", _schema_loader)
    response = client.post(
        "/composer/recommend_capabilities",
        json={
            "goal": "Render the generated document to a PDF",
            "context_json": {"topic": "Latency", "path": "documents/latency.pdf"},
            "draft": {"nodes": [{"id": "n1", "capabilityId": "document.spec.generate"}]},
            "use_llm": False,
            "max_results": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "heuristic"
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) >= 1
    assert body["recommendations"][0]["id"] == "document.pdf.render"


def test_composer_recommend_capabilities_uses_llm_when_available(monkeypatch):
    spec_generate = cap_registry.CapabilitySpec(
        capability_id="document.spec.generate",
        description="Generate a document spec",
        risk_tier="read_only",
        idempotency="read",
        input_schema_ref="schema_generate",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    spec_render = cap_registry.CapabilitySpec(
        capability_id="document.pdf.render",
        description="Render a PDF",
        risk_tier="bounded_write",
        idempotency="read",
        input_schema_ref="schema_render",
        output_schema_ref=None,
        adapters=(),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(
        capabilities={
            "document.spec.generate": spec_generate,
            "document.pdf.render": spec_render,
        }
    )
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main, "_load_schema_from_ref", lambda _ref: {"type": "object"})

    requests: list[LLMRequest] = []

    class _Provider(LLMProvider):
        def generate_request(self, request: LLMRequest):
            requests.append(request)
            return LLMResponse(
                content='{"recommendations":[{"id":"document.pdf.render","reason":"next step","confidence":0.91}]}'
            )

    monkeypatch.setattr(main, "_composer_recommender_provider", _Provider())

    response = client.post(
        "/composer/recommend_capabilities",
        json={
            "goal": "Generate and render document",
            "context_json": {"topic": "Latency"},
            "draft": {"nodes": [{"id": "n1", "capabilityId": "document.spec.generate"}]},
            "use_llm": True,
            "max_results": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["recommendations"][0]["id"] == "document.pdf.render"
    assert requests
    assert requests[0].metadata is not None
    assert requests[0].metadata["operation"] == "capability_recommendations"


def test_preflight_plan_endpoint_returns_valid_true_for_simple_plan():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "simple",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "TransformData",
                    "description": "transform",
                    "instruction": "transform",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": [],
                    "tool_requests": ["utility.json.transform"],
                    "tool_inputs": {"utility.json.transform": {"input": {"name": "demo"}}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_plan_endpoint_rejects_raw_runtime_tool_name() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "simple",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "TransformData",
                    "description": "transform",
                    "instruction": "transform",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": [],
                    "tool_requests": ["json_transform"],
                    "tool_inputs": {"json_transform": {"input": {"name": "demo"}}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }

    response = client.post("/plans/preflight", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["TransformData"] == (
        "planner_request_language_invalid:json_transform:"
        "use_capability_id:utility.json.transform"
    )
    assert body["diagnostics"][0]["code"] == "planner_request_language_invalid"


def test_preflight_plan_endpoint_rejects_render_task_without_explicit_path() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "render",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "RenderDocx",
                    "description": "Render document",
                    "instruction": "Render the document",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "render",
                    "deps": [],
                    "tool_requests": ["document.docx.render"],
                    "tool_inputs": {"document.docx.render": {"document_spec": {"blocks": []}}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }

    response = client.post("/plans/preflight", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["RenderDocx"] == "render_path_explicit_required:document.docx.render"
    assert body["diagnostics"][0]["code"] == "render_path_explicit_required"


def test_preflight_plan_endpoint_rejects_dependency_derived_render_path() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "render",
            "dag_edges": [["DeriveOutputPath", "RenderDocx"]],
            "tasks": [
                {
                    "name": "DeriveOutputPath",
                    "description": "Derive output path",
                    "instruction": "derive",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "transform",
                    "deps": [],
                    "tool_requests": ["derive_output_filename"],
                    "tool_inputs": {
                        "derive_output_filename": {
                            "document_type": "document",
                            "output_extension": "docx",
                            "target_role_name": "deployment_report",
                        }
                    },
                    "critic_required": False,
                },
                {
                    "name": "RenderDocx",
                    "description": "Render document",
                    "instruction": "Render the document",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "render",
                    "deps": ["DeriveOutputPath"],
                    "tool_requests": ["document.docx.render"],
                    "tool_inputs": {
                        "document.docx.render": {
                            "document_spec": {"blocks": []},
                            "path": {
                                "$from": "dependencies_by_name.DeriveOutputPath.derive_output_filename.path"
                            },
                        }
                    },
                    "critic_required": False,
                },
            ],
        },
        "job_context": {},
    }

    response = client.post("/plans/preflight", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["RenderDocx"] == "render_path_derived_not_allowed:document.docx.render"
    assert any(
        diagnostic["code"] == "render_path_derived_not_allowed"
        and diagnostic["field"] == "RenderDocx"
        for diagnostic in body["diagnostics"]
    )


def test_preflight_plan_endpoint_accepts_render_output_path_alias_literal() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "render",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "RenderDocx",
                    "description": "Render document",
                    "instruction": "Render the document",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "render",
                    "deps": [],
                    "tool_requests": ["document.docx.render"],
                    "tool_inputs": {
                        "document.docx.render": {
                            "document_spec": {"blocks": []},
                            "output_path": "documents/report.docx",
                        }
                    },
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }

    response = client.post("/plans/preflight", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_plan_endpoint_accepts_render_path_from_job_context_reference() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "render",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "RenderDocx",
                    "description": "Render document",
                    "instruction": "Render the document",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "render",
                    "deps": [],
                    "tool_requests": ["document.docx.render"],
                    "tool_inputs": {
                        "document.docx.render": {
                            "document_spec": {"blocks": []},
                            "path": {"$from": "job_context.path"},
                        }
                    },
                    "critic_required": False,
                }
            ],
        },
        "job_context": {"path": "documents/report.docx"},
    }

    response = client.post("/plans/preflight", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_plan_endpoint_returns_reference_error_for_broken_dependency_tool():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "broken",
            "dag_edges": [["GenerateSpec", "ValidateSpec"]],
            "tasks": [
                {
                    "name": "GenerateSpec",
                    "description": "generate",
                    "instruction": "generate",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": [],
                    "tool_requests": ["document.spec.generate"],
                    "tool_inputs": {"document.spec.generate": {"job": {"topic": "demo"}}},
                    "critic_required": False,
                },
                {
                    "name": "ValidateSpec",
                    "description": "validate",
                    "instruction": "validate",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "deps": ["GenerateSpec"],
                    "tool_requests": ["document.spec.validate"],
                    "tool_inputs": {
                        "document.spec.validate": {
                            "document_spec": {
                                "$from": [
                                    "dependencies_by_name",
                                    "GenerateSpec",
                                    "unknown.tool",
                                    "document_spec",
                                ]
                            }
                        }
                    },
                    "critic_required": False,
                },
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "ValidateSpec" in body["errors"]
    assert "input reference resolution failed" in body["errors"]["ValidateSpec"]


def test_preflight_plan_endpoint_rejects_tool_intent_mismatch():
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "mismatch",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "IoTask",
                    "description": "Read data",
                    "instruction": "Fetch data only",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "io",
                    "deps": [],
                    "tool_requests": ["llm.text.generate"],
                    "tool_inputs": {"llm.text.generate": {"text": "hello"}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "IoTask" in body["errors"]
    assert body["errors"]["IoTask"].startswith("task_intent_mismatch:llm.text.generate")


def test_preflight_plan_endpoint_rejects_unknown_memory_read_name() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v2",
            "tasks_summary": "Workflow Studio draft",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "MemoryRead",
                    "description": "Read memory entries for pointer resolution and context hydration.",
                    "instruction": "Use capability memory.read.",
                    "acceptance_criteria": ["Completed capability memory.read"],
                    "expected_output_schema_ref": "",
                    "intent": "io",
                    "deps": [],
                    "tool_requests": ["memory.read"],
                    "tool_inputs": {
                        "memory.read": {
                            "name": "profile",
                            "scope": "user",
                            "key": "profile",
                            "user_id": "narendersurabhi",
                        }
                    },
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }

    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["MemoryRead"].startswith("memory.read:unknown_memory:profile")
    assert "user_profile" in body["errors"]["MemoryRead"]


def test_preflight_plan_endpoint_rejects_capability_intent_mismatch(monkeypatch):
    capability = cap_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={"task_intents": ["io"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_local",
                tool_name="search_repositories",
            ),
        ),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"github.repo.list": capability})
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "capability mismatch",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "BadCapabilityIntent",
                    "description": "Generate report",
                    "instruction": "Generate data",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["github.repo.list"],
                    "tool_inputs": {"github.repo.list": {"query": "user:octocat"}},
                    "critic_required": False,
                }
            ],
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert "BadCapabilityIntent" in body["errors"]
    assert body["errors"]["BadCapabilityIntent"].startswith(
        "task_intent_mismatch:github.repo.list:generate"
    )


def test_preflight_plan_endpoint_returns_intent_segment_slot_diagnostics() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "segment contract",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "GenerateText",
                    "description": "Generate text",
                    "instruction": "Generate the content",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["llm.text.generate"],
                    "tool_inputs": {"llm.text.generate": {"text": "hello"}},
                    "critic_required": False,
                }
            ],
        },
        "goal_intent_graph": {
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Generate report text",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["llm.text.generate"],
                    "slots": {
                        "entity": "report",
                        "artifact_type": "content",
                        "output_format": "txt",
                        "risk_level": "read_only",
                        "must_have_inputs": ["path"],
                    },
                }
            ]
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["GenerateText"].startswith("intent_segment_invalid:llm.text.generate")
    diagnostics = body.get("diagnostics", [])
    assert isinstance(diagnostics, list) and diagnostics
    assert diagnostics[0]["code"] == "intent_segment.must_have_inputs_missing"
    assert diagnostics[0]["field"] == "GenerateText"


def test_preflight_plan_endpoint_accepts_normalized_envelope_without_legacy_graph() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "segment contract",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "GenerateText",
                    "description": "Generate text",
                    "instruction": "Generate the content",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["llm.text.generate"],
                    "tool_inputs": {"llm.text.generate": {"text": "hello"}},
                    "critic_required": False,
                }
            ],
        },
        "normalized_intent_envelope": {
            "goal": "Generate report text",
            "profile": {"intent": "generate", "source": "llm"},
            "graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "generate",
                        "objective": "Generate report text",
                        "required_inputs": ["instruction"],
                        "suggested_capabilities": ["llm.text.generate"],
                        "slots": {
                            "entity": "report",
                            "artifact_type": "content",
                            "output_format": "txt",
                            "risk_level": "read_only",
                            "must_have_inputs": ["path"],
                        },
                    }
                ]
            },
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["errors"]["GenerateText"].startswith("intent_segment_invalid:llm.text.generate")
    diagnostics = body.get("diagnostics", [])
    assert isinstance(diagnostics, list) and diagnostics
    assert diagnostics[0]["code"] == "intent_segment.must_have_inputs_missing"
    assert diagnostics[0]["field"] == "GenerateText"


def test_preflight_plan_endpoint_accepts_intent_segment_tool_inputs_requirement() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "segment contract",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "ImplementRepository",
                    "description": "Implement repository changes",
                    "instruction": "Use codegen.autonomous to implement the goal.",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["llm.text.generate"],
                    "tool_inputs": {"llm.text.generate": {"text": "implement it"}},
                    "critic_required": False,
                }
            ],
        },
        "goal_intent_graph": {
            "segments": [
                {
                    "id": "s1",
                    "intent": "generate",
                    "objective": "Implement repository",
                    "required_inputs": ["instruction"],
                    "suggested_capabilities": ["llm.text.generate"],
                    "slots": {
                        "entity": "repository",
                        "artifact_type": "code",
                        "output_format": "txt",
                        "risk_level": "read_only",
                        "must_have_inputs": ["tool_inputs"],
                    },
                }
            ]
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_plan_endpoint_accepts_document_segment_main_topic_alias() -> None:
    payload = {
        "plan": {
            "planner_version": "ui_chaining_composer_v1",
            "tasks_summary": "document generation",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "Generate DocumentSpec",
                    "description": "Generate a DocumentSpec",
                    "instruction": "Generate a 2-page DOCX cheatsheet for advanced AI engineers.",
                    "acceptance_criteria": ["done"],
                    "expected_output_schema_ref": "",
                    "intent": "generate",
                    "deps": [],
                    "tool_requests": ["document.spec.generate"],
                    "tool_inputs": {
                        "document.spec.generate": {
                            "instruction": "Generate a 2-page DOCX cheatsheet for advanced AI engineers.",
                            "topic": "AI lifecycle using Kubernetes, Kubeflow, RAG retrieval, and RAG indexing",
                            "audience": "Advanced AI engineers",
                            "tone": "concise",
                        }
                    },
                    "critic_required": False,
                }
            ],
        },
        "normalized_intent_envelope": {
            "goal": "Create a word cheatsheet document",
            "profile": {"intent": "generate", "source": "llm"},
            "graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "generate",
                        "objective": "Generate a DocumentSpec for AI lifecycle cheat sheet",
                        "required_inputs": ["main_topic", "length", "audience"],
                        "suggested_capabilities": ["document.spec.generate"],
                        "slots": {
                            "entity": "document_spec",
                            "artifact_type": "document_spec",
                            "output_format": "docx",
                            "risk_level": "read_only",
                            "must_have_inputs": ["main_topic", "length", "audience"],
                        },
                    }
                ]
            },
        },
        "job_context": {},
    }
    response = client.post("/plans/preflight", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["errors"] == {}


def test_preflight_task_intent_uses_goal_text_when_task_text_generic() -> None:
    task = models.TaskCreate(
        name="ValidateSpec",
        description="Step one",
        instruction="Handle this task.",
        acceptance_criteria=["Done"],
        expected_output_schema_ref="",
        deps=[],
        tool_requests=[],
        tool_inputs={},
        critic_required=False,
    )
    inferred = main._preflight_task_intent(
        task,
        goal_text="Validate the generated document against schema constraints.",
    )
    assert inferred == "validate"


def test_build_plan_from_composer_draft_derives_intent_from_goal(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="custom.step",
        description="Custom step",
        risk_tier="read_only",
        idempotency="read",
        planner_hints={},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
        enabled=True,
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"custom.step": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {"nodes": [{"id": "n1", "capabilityId": "custom.step", "taskName": "CustomStep"}]},
        goal_text="Render the final document as PDF.",
    )
    assert not errors
    assert plan is not None
    assert plan.tasks[0].intent == models.ToolIntent.render


def test_composer_compile_emits_run_spec(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    response = client.post(
        "/composer/compile",
        json={
            "draft": {
                "summary": "Compile a gated chain",
                "nodes": [
                    {"id": "source", "capabilityId": "json_transform", "taskName": "LoadData"},
                    {
                        "id": "gate",
                        "taskName": "OnlyIfApproved",
                        "nodeKind": "control",
                        "controlKind": "if",
                        "capabilityId": "studio.control.if",
                        "controlConfig": {"expression": "context.approved == true"},
                    },
                    {
                        "id": "target",
                        "capabilityId": "json_transform",
                        "taskName": "TransformData",
                        "bindings": {
                            "source": {
                                "kind": "step_output",
                                "sourceNodeId": "source",
                                "sourcePath": "items",
                            }
                        },
                    },
                ],
                "edges": [
                    {"fromNodeId": "source", "toNodeId": "gate"},
                    {"fromNodeId": "gate", "toNodeId": "target"},
                ],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert body["plan"] is not None
    assert "LoadData" in body["preflight_errors"]
    assert body["run_spec"]["kind"] == "studio"
    assert [step["name"] for step in body["run_spec"]["steps"]] == ["LoadData", "TransformData"]
    assert body["run_spec"]["steps"][1]["execution_gate"] == {
        "expression": "context.approved == true"
    }
    assert body["run_spec"]["steps"][1]["input_bindings"] == {
        "source": {
            "$from": ["dependencies_by_name", "LoadData", "json_transform", "items"]
        }
    }
    assert body["run_spec"]["dag_edges"] == [
        [
            body["run_spec"]["steps"][0]["step_id"],
            body["run_spec"]["steps"][1]["step_id"],
        ]
    ]


def test_composer_compile_allows_derived_render_paths_in_studio_mode(monkeypatch) -> None:
    registry = _document_render_registry()
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)
    monkeypatch.setattr(main.capability_registry, "resolve_capability_mode", lambda: "enabled")

    response = client.post(
        "/composer/compile",
        json={
            "draft": {
                "summary": "Render a DOCX from a generated spec",
                "nodes": [
                    {
                        "id": "spec",
                        "taskName": "GenerateDocumentSpec",
                        "capabilityId": "document.spec.generate",
                        "bindings": {
                            "instruction": {
                                "kind": "literal",
                                "value": "Create a deployment report document.",
                            },
                            "topic": {"kind": "literal", "value": "Deployment report"},
                            "audience": {"kind": "literal", "value": "Engineers"},
                        },
                    },
                    {
                        "id": "path",
                        "taskName": "DeriveOutputPath",
                        "capabilityId": "derive_output_filename",
                        "bindings": {
                            "document_type": {"kind": "literal", "value": "document"},
                            "output_extension": {"kind": "literal", "value": "docx"},
                            "target_role_name": {
                                "kind": "literal",
                                "value": "deployment_report",
                            },
                        },
                    },
                    {
                        "id": "render",
                        "taskName": "RenderDocument",
                        "capabilityId": "document.docx.render",
                        "bindings": {
                            "document_spec": {
                                "kind": "step_output",
                                "sourceNodeId": "spec",
                                "sourcePath": "document_spec",
                            },
                            "path": {
                                "kind": "step_output",
                                "sourceNodeId": "path",
                                "sourcePath": "path",
                            },
                        },
                    },
                ],
                "edges": [
                    {"fromNodeId": "spec", "toNodeId": "render"},
                    {"fromNodeId": "path", "toNodeId": "render"},
                ],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["preflight_errors"] == {}


def test_build_plan_from_composer_draft_flags_control_flow_nodes() -> None:
    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "taskName": "ApprovalGate",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {"expression": "context.approved == true"},
                }
            ]
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_if_else_true_branch_missing" in codes
    assert "draft.control_if_else_false_branch_missing" in codes


def test_build_plan_from_composer_draft_validates_switch_control_cases() -> None:
    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "taskName": "RouteByStatus",
                    "nodeKind": "control",
                    "controlKind": "switch",
                    "capabilityId": "studio.control.switch",
                    "controlConfig": {
                        "expression": "context.status",
                        "switchCases": [{"id": "c1", "label": "", "match": ""}],
                    },
                }
            ]
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_switch_case_label_missing" in codes
    assert "draft.control_switch_case_match_missing" in codes


def test_build_plan_from_composer_draft_lowers_if_control_to_execution_gate(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "source",
                    "capabilityId": "json_transform",
                    "taskName": "LoadData",
                },
                {
                    "id": "gate",
                    "taskName": "OnlyIfApproved",
                    "nodeKind": "control",
                    "controlKind": "if",
                    "capabilityId": "studio.control.if",
                    "controlConfig": {"expression": "context.approved == true"},
                },
                {
                    "id": "target",
                    "capabilityId": "json_transform",
                    "taskName": "TransformData",
                },
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    assert [task.name for task in plan.tasks] == ["LoadData", "TransformData"]
    assert plan.tasks[1].deps == ["LoadData"]
    assert plan.tasks[1].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true"}
    }


def test_build_plan_from_composer_draft_allows_workflow_variable_expression(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "workflowInterface": {
                "variables": [
                    {
                        "key": "should_publish",
                        "valueType": "boolean",
                        "binding": {"kind": "literal", "value": True},
                    }
                ]
            },
            "nodes": [
                {
                    "id": "source",
                    "capabilityId": "json_transform",
                    "taskName": "LoadData",
                },
                {
                    "id": "gate",
                    "taskName": "OnlyIfPublishRequested",
                    "nodeKind": "control",
                    "controlKind": "if",
                    "capabilityId": "studio.control.if",
                    "controlConfig": {"expression": "workflow.variable.should_publish == true"},
                },
                {
                    "id": "target",
                    "capabilityId": "json_transform",
                    "taskName": "PublishData",
                },
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    assert plan.tasks[1].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "workflow.variable.should_publish == true"}
    }


def test_build_plan_from_composer_draft_rejects_undefined_workflow_expression_key(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "workflowInterface": {
                "variables": [
                    {
                        "key": "should_publish",
                        "valueType": "boolean",
                        "binding": {"kind": "literal", "value": True},
                    }
                ]
            },
            "nodes": [
                {
                    "id": "source",
                    "capabilityId": "json_transform",
                    "taskName": "LoadData",
                },
                {
                    "id": "gate",
                    "taskName": "OnlyIfApproved",
                    "nodeKind": "control",
                    "controlKind": "if",
                    "capabilityId": "studio.control.if",
                    "controlConfig": {"expression": "workflow.variable.missing_flag == true"},
                },
                {
                    "id": "target",
                    "capabilityId": "json_transform",
                    "taskName": "PublishData",
                },
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_expression_unsupported" in codes


def test_build_plan_from_composer_draft_lowers_if_else_to_branch_gates(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "source", "capabilityId": "json_transform", "taskName": "LoadData"},
                {
                    "id": "gate",
                    "taskName": "ApprovalBranch",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {
                        "expression": "context.approved == true",
                        "trueLabel": "approved",
                        "falseLabel": "rejected",
                    },
                },
                {"id": "true-node", "capabilityId": "json_transform", "taskName": "ApprovedPath"},
                {"id": "false-node", "capabilityId": "json_transform", "taskName": "RejectedPath"},
                {"id": "join-node", "capabilityId": "json_transform", "taskName": "JoinedPath"},
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "gate"},
                {"fromNodeId": "gate", "toNodeId": "true-node", "branchLabel": "approved"},
                {"fromNodeId": "gate", "toNodeId": "false-node", "branchLabel": "rejected"},
                {"fromNodeId": "true-node", "toNodeId": "join-node"},
                {"fromNodeId": "false-node", "toNodeId": "join-node"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["ApprovedPath"].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true"}
    }
    assert task_by_name["RejectedPath"].tool_inputs[main.execution_contracts.EXECUTION_GATE_KEY] == {
        "json_transform": {"expression": "context.approved == true", "negate": True}
    }
    assert main.execution_contracts.EXECUTION_GATE_KEY not in task_by_name["JoinedPath"].tool_inputs


def test_build_plan_from_composer_draft_requires_if_else_branch_labels(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "gate",
                    "taskName": "ApprovalBranch",
                    "nodeKind": "control",
                    "controlKind": "if_else",
                    "capabilityId": "studio.control.if_else",
                    "controlConfig": {
                        "expression": "context.approved == true",
                    },
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "TargetPath"},
            ],
            "edges": [
                {"fromNodeId": "gate", "toNodeId": "target"},
            ],
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_if_else_true_branch_missing" in codes
    assert "draft.control_if_else_false_branch_missing" in codes


def test_build_plan_from_composer_draft_lowers_parallel_fan_out(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "source", "capabilityId": "json_transform", "taskName": "LoadData"},
                {
                    "id": "parallel",
                    "taskName": "FanOut",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_out"},
                },
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {"id": "right", "capabilityId": "json_transform", "taskName": "RightBranch"},
            ],
            "edges": [
                {"fromNodeId": "source", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "left"},
                {"fromNodeId": "parallel", "toNodeId": "right"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["LeftBranch"].deps == ["LoadData"]
    assert task_by_name["RightBranch"].deps == ["LoadData"]


def test_build_plan_from_composer_draft_lowers_parallel_fan_in(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {"id": "right", "capabilityId": "json_transform", "taskName": "RightBranch"},
                {
                    "id": "parallel",
                    "taskName": "JoinBranches",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_in"},
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "AfterJoin"},
            ],
            "edges": [
                {"fromNodeId": "left", "toNodeId": "parallel"},
                {"fromNodeId": "right", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "target"},
            ],
        }
    )

    assert errors == []
    assert plan is not None
    task_by_name = {task.name: task for task in plan.tasks}
    assert task_by_name["AfterJoin"].deps == ["LeftBranch", "RightBranch"]


def test_build_plan_from_composer_draft_requires_parallel_fan_in_sources(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {"id": "left", "capabilityId": "json_transform", "taskName": "LeftBranch"},
                {
                    "id": "parallel",
                    "taskName": "JoinBranches",
                    "nodeKind": "control",
                    "controlKind": "parallel",
                    "capabilityId": "studio.control.parallel",
                    "controlConfig": {"parallelMode": "fan_in"},
                },
                {"id": "target", "capabilityId": "json_transform", "taskName": "AfterJoin"},
            ],
            "edges": [
                {"fromNodeId": "left", "toNodeId": "parallel"},
                {"fromNodeId": "parallel", "toNodeId": "target"},
            ],
        }
    )

    assert plan is None
    codes = {entry["code"] for entry in errors}
    assert "draft.control_parallel_fan_in_sources_missing" in codes


def test_build_plan_from_composer_draft_injects_user_id_for_user_memory_bindings(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "nodes": [
                {
                    "id": "n1",
                    "capabilityId": "json_transform",
                    "taskName": "LoadProfile",
                    "bindings": {
                        "profile": {"kind": "memory", "scope": "user", "name": "user_profile", "key": "profile"}
                    },
                }
            ]
        },
        job_context={"user_id": "narendersurabhi"},
    )

    assert errors == []
    assert plan is not None
    assert plan.tasks[0].tool_inputs["json_transform"]["profile"] == {
        "scope": "user",
        "name": "user_profile",
        "key": "profile",
        "user_id": "narendersurabhi",
    }


def test_build_plan_from_composer_draft_lowers_workflow_inputs_and_variables(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    plan, errors, _warnings = main._build_plan_from_composer_draft(
        {
            "workflowInterface": {
                "inputs": [
                    {
                        "key": "topic",
                        "valueType": "string",
                        "required": True,
                    }
                ],
                "variables": [
                    {
                        "key": "topic_alias",
                        "binding": {"kind": "workflow_input", "inputKey": "topic"},
                    }
                ],
                "outputs": [],
            },
            "nodes": [
                {
                    "id": "n1",
                    "capabilityId": "json_transform",
                    "taskName": "UseInterface",
                    "bindings": {
                        "topic": {"kind": "workflow_input", "inputKey": "topic"},
                        "alias": {
                            "kind": "workflow_variable",
                            "variableKey": "topic_alias",
                        },
                    },
                }
            ],
        }
    )

    assert errors == []
    assert plan is not None
    assert plan.tasks[0].tool_inputs["json_transform"]["topic"] == {
        "$from": ["job_context", "workflow", "inputs", "topic"]
    }
    assert plan.tasks[0].tool_inputs["json_transform"]["alias"] == {
        "$from": ["job_context", "workflow", "variables", "topic_alias"]
    }


def test_workflow_version_run_accepts_explicit_inputs(monkeypatch) -> None:
    capability = cap_registry.CapabilitySpec(
        capability_id="json_transform",
        description="Transform JSON",
        enabled=True,
        risk_tier="read_only",
        idempotency="read",
        output_schema_ref="schemas/json_object",
        planner_hints={"task_intents": ["transform"]},
        adapters=(
            cap_registry.CapabilityAdapterSpec(
                type="local_tool",
                server_id="local_worker",
                tool_name="json_transform",
            ),
        ),
    )
    registry = cap_registry.CapabilityRegistry(capabilities={"json_transform": capability})
    monkeypatch.setattr(main.capability_registry, "load_capability_registry", lambda: registry)

    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Workflow inputs demo",
            "goal": "Use workflow inputs",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Workflow inputs demo",
                "workflowInterface": {
                    "inputs": [
                        {
                            "key": "topic",
                            "valueType": "string",
                            "required": True,
                        }
                    ],
                    "variables": [
                        {
                            "key": "topic_alias",
                            "binding": {"kind": "workflow_input", "inputKey": "topic"},
                        }
                    ],
                    "outputs": [],
                },
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "Transform",
                        "capabilityId": "json_transform",
                        "bindings": {
                            "input": {
                                "kind": "workflow_variable",
                                "variableKey": "topic_alias",
                            },
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"inputs": {"topic": "Quarterly update"}},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["job"]["context_json"]["workflow"]["inputs"]["topic"] == "Quarterly update"
    assert (
        run_body["job"]["context_json"]["workflow"]["variables"]["topic_alias"]
        == "Quarterly update"
    )
    assert run_body["workflow_run"]["metadata"]["workflow_input_keys"] == ["topic"]

    with SessionLocal() as db:
        task_record = db.query(TaskRecord).filter(TaskRecord.job_id == run_body["job"]["id"]).first()
        assert task_record is not None
    assert task_record.tool_inputs["json_transform"]["input"] == {
        "$from": ["job_context", "workflow", "variables", "topic_alias"]
    }


def test_publish_workflow_definition_rejects_unknown_memory_read_name() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Broken profile read",
            "goal": "Read profile memory",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Broken profile read",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ReadProfile",
                        "capabilityId": "memory.read",
                        "bindings": {
                            "name": {"kind": "literal", "value": "profile"},
                            "scope": {"kind": "literal", "value": "user"},
                            "key": {"kind": "literal", "value": "profile"},
                            "user_id": {"kind": "literal", "value": "narendersurabhi"},
                        },
                    }
                ],
                "edges": [],
            },
        },
    )
    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 400
    detail = publish_response.json()["detail"]
    assert detail["error"] == "workflow_preflight_failed"
    assert detail["preflight_errors"]["ReadProfile"].startswith("memory.read:unknown_memory:profile")
    assert "user_profile" in detail["preflight_errors"]["ReadProfile"]


def test_workflow_run_uses_published_run_spec_when_compiled_plan_missing() -> None:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": "Workspace listing",
            "goal": "List workspace files",
            "user_id": "narendersurabhi",
            "context_json": {"user_id": "narendersurabhi"},
            "draft": {
                "summary": "Workspace listing",
                "nodes": [
                    {
                        "id": "n1",
                        "taskName": "ListWorkspace",
                        "capabilityId": "filesystem.workspace.list",
                        "bindings": {},
                    }
                ],
                "edges": [],
            },
        },
    )

    assert create_response.status_code == 200
    definition = create_response.json()

    publish_response = client.post(f"/workflows/definitions/{definition['id']}/publish", json={})
    assert publish_response.status_code == 200
    version = publish_response.json()
    parsed_run_spec = run_specs.parse_run_spec(version.get("run_spec"))
    assert parsed_run_spec is not None

    with SessionLocal() as db:
        version_record = (
            db.query(WorkflowVersionRecord)
            .filter(WorkflowVersionRecord.id == version["id"])
            .first()
        )
        assert version_record is not None
        version_record.compiled_plan_json = {}
        db.commit()

    run_response = client.post(
        f"/workflows/versions/{version['id']}/run",
        json={"priority": 2},
    )
    assert run_response.status_code == 200
    run_body = run_response.json()
    assert run_body["workflow_run"]["version_id"] == version["id"]
    assert run_body["plan"]["job_id"] == run_body["job"]["id"]


def test_task_payload_from_record_flags_unresolved_reference_inputs() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-ref-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="RenderDocument",
        description="Render document",
        instruction="Render",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/docx_output",
        status=models.TaskStatus.ready.value,
        deps=["GenerateDocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_render_from_spec"],
        tool_inputs={
            "docx_render_from_spec": {
                "path": "documents/out.docx",
                "document_spec": {
                    "$from": "dependencies_by_name.GenerateDocumentSpec.llm_generate_document_spec.document_spec"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    payload = main._task_payload_from_record(record, correlation_id="corr", context={})
    validation = payload.get("tool_inputs_validation", {})
    assert "docx_render_from_spec" in validation
    assert "input reference resolution failed" in validation["docx_render_from_spec"]


def test_task_payload_from_record_resolves_validation_report_alias_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-improve-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Improve_DocumentSpec",
        description="Improve doc spec",
        instruction="Improve",
        acceptance_criteria=["improved"],
        expected_output_schema_ref="schemas/document_spec",
        status=models.TaskStatus.ready.value,
        deps=["Generate_DocumentSpec", "Validate_DocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="transform",
        tool_requests=["llm_improve_document_spec"],
        tool_inputs={
            "llm_improve_document_spec": {
                "document_spec": {
                    "$from": "dependencies_by_name.Generate_DocumentSpec.llm_generate_document_spec.document_spec"
                },
                "validation_report": {
                    "$from": "dependencies_by_name.Validate_DocumentSpec.document_spec_validate.validation_report"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Generate_DocumentSpec": {
                "llm_generate_document_spec": {
                    "document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}
                }
            },
            "Validate_DocumentSpec": {
                "document_spec_validate": {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "stats": {"block_count": 1},
                }
            },
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["llm_improve_document_spec"]
    assert resolved["document_spec"]["blocks"][0]["text"] == "hello"
    assert resolved["validation_report"]["valid"] is True


def test_task_payload_from_record_resolves_dotted_request_id_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-validate-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Validate DocumentSpec",
        description="Validate doc spec",
        instruction="Validate",
        acceptance_criteria=["validated"],
        expected_output_schema_ref="schemas/validation_report",
        status=models.TaskStatus.ready.value,
        deps=["Generate DocumentSpec"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="validate",
        tool_requests=["document_spec_validate"],
        tool_inputs={
            "document_spec_validate": {
                "strict": True,
                "document_spec": {
                    "$from": "dependencies_by_name.Generate DocumentSpec.document.spec.generate.document_spec"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Generate DocumentSpec": {
                "document.spec.generate": {
                    "document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}
                }
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["document_spec_validate"]
    assert resolved["document_spec"]["blocks"][0]["text"] == "hello"


def test_build_task_context_aliases_compiled_request_outputs_for_capability_refs(
    monkeypatch,
) -> None:
    now = _utcnow()
    generate_task = TaskRecord(
        id=f"task-generate-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Generate DocumentSpec",
        description="Generate doc spec",
        instruction="Generate",
        acceptance_criteria=["generated"],
        expected_output_schema_ref="schemas/document_spec",
        status=models.TaskStatus.completed.value,
        deps=[],
        attempts=1,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="generate",
        tool_requests=["llm_generate_document_spec"],
        tool_inputs=execution_contracts.embed_capability_bindings(
            {"llm_generate_document_spec": {"instruction": "Generate a document"}},
            {
                "llm_generate_document_spec": {
                    "request_id": "llm_generate_document_spec",
                    "capability_id": "document.spec.generate",
                    "tool_name": "llm_generate_document_spec",
                    "adapter_type": "tool",
                }
            },
            request_ids=["llm_generate_document_spec"],
        ),
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    validate_task = TaskRecord(
        id=f"task-validate-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Validate DocumentSpec",
        description="Validate doc spec",
        instruction="Validate",
        acceptance_criteria=["validated"],
        expected_output_schema_ref="schemas/validation_report",
        status=models.TaskStatus.ready.value,
        deps=[generate_task.id],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="validate",
        tool_requests=["document_spec_validate"],
        tool_inputs={
            "document_spec_validate": {
                "strict": True,
                "document_spec": {
                    "$from": "dependencies_by_name.Generate DocumentSpec.document.spec.generate.document_spec"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )

    monkeypatch.setattr(
        main,
        "_load_task_output",
        lambda task_id: {
            "llm_generate_document_spec": {
                "document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}
            }
        }
        if task_id == generate_task.id
        else {},
    )

    context = main._build_task_context(
        validate_task.id,
        {generate_task.id: generate_task, validate_task.id: validate_task},
        {
            generate_task.id: generate_task.name,
            validate_task.id: validate_task.name,
        },
    )

    aliased = context["dependencies_by_name"]["Generate DocumentSpec"]
    assert "document.spec.generate" in aliased

    payload = main._task_payload_from_record(validate_task, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["document_spec_validate"]
    assert resolved["document_spec"]["blocks"][0]["text"] == "hello"


def test_task_payload_from_record_resolves_output_path_alias_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-render-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Render DOCX",
        description="Render DOCX",
        instruction="Render DOCX",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/output_path",
        status=models.TaskStatus.ready.value,
        deps=["Derive Output Filename"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_render_from_spec"],
        tool_inputs={
            "docx_render_from_spec": {
                "document_spec": {"blocks": []},
                "output_path": {
                    "$from": "dependencies_by_name.Derive Output Filename.derive_output_filename.output_path"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "Derive Output Filename": {
                "derive_output_filename": {"path": "artifacts/output.docx"}
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["docx_render_from_spec"]
    assert resolved["output_path"] == "artifacts/output.docx"


def test_task_payload_from_record_resolves_task_level_path_reference() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-render-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="Render DOCX",
        description="Render DOCX",
        instruction="Render DOCX",
        acceptance_criteria=["docx created"],
        expected_output_schema_ref="schemas/output_path",
        status=models.TaskStatus.ready.value,
        deps=["DeriveResumeOutputPath"],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent="render",
        tool_requests=["docx_render_from_spec"],
        tool_inputs={
            "docx_render_from_spec": {
                "document_spec": {"blocks": []},
                "output_path": {
                    "$from": "dependencies_by_name.DeriveResumeOutputPath.path"
                },
            }
        },
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    context = {
        "dependencies_by_name": {
            "DeriveResumeOutputPath": {
                "derive_output_filename": {"path": "artifacts/output.docx"}
            }
        },
        "dependencies": {},
    }

    payload = main._task_payload_from_record(record, correlation_id="corr", context=context)
    assert "tool_inputs_validation" not in payload
    resolved = payload["tool_inputs"]["docx_render_from_spec"]
    assert resolved["output_path"] == "artifacts/output.docx"


def test_task_payload_from_record_includes_intent_segment_profile() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-segment-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="RenderReport",
        description="Render report",
        instruction="Render",
        acceptance_criteria=["done"],
        expected_output_schema_ref="schemas/pdf_output",
        status=models.TaskStatus.ready.value,
        deps=[],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent=None,
        tool_requests=["document.pdf.render"],
        tool_inputs={"document.pdf.render": {"path": "artifacts/report.pdf"}},
        created_at=now,
        updated_at=now,
        critic_required=0,
    )
    intent_profile = {
        "intent": "render",
        "source": "goal_graph",
        "confidence": 0.91,
        "segment": {
            "id": "s3",
            "intent": "render",
            "objective": "Render final PDF report",
            "required_inputs": ["document_spec", "path"],
            "suggested_capabilities": ["document.pdf.render"],
            "slots": {
                "entity": "report",
                "artifact_type": "document",
                "output_format": "pdf",
                "risk_level": "bounded_write",
                "must_have_inputs": ["document_spec", "path"],
            },
        },
    }

    payload = main._task_payload_from_record(
        record,
        correlation_id="corr",
        context={},
        intent_profile=intent_profile,
    )
    assert payload["intent"] == "render"
    assert payload["intent_source"] == "goal_graph"
    assert payload["intent_confidence"] == 0.91
    assert payload["intent_segment"]["id"] == "s3"
    assert payload["intent_segment"]["slots"]["output_format"] == "pdf"
    assert payload["trace_id"] == "corr"
    parsed = main.execution_contracts.build_task_dispatch_payload(payload)
    assert parsed.correlation_id == "corr"
    assert parsed.trace_id == "corr"


def test_task_payload_from_record_uses_typed_dispatch_contract() -> None:
    now = _utcnow()
    record = TaskRecord(
        id=f"task-dispatch-{uuid.uuid4()}",
        job_id="job-ref",
        plan_id="plan-ref",
        name="GenerateSpec",
        description="Generate doc spec",
        instruction="Generate",
        acceptance_criteria=["done"],
        expected_output_schema_ref="schemas/document_spec",
        status=models.TaskStatus.ready.value,
        deps=[],
        attempts=0,
        max_attempts=0,
        rework_count=0,
        max_reworks=0,
        assigned_to=None,
        intent=None,
        tool_requests=["document.spec.generate"],
        tool_inputs=main.execution_contracts.embed_capability_bindings(
            {"document.spec.generate": {"job": {"topic": "latency"}}},
            {
                "document.spec.generate": {
                    "request_id": "document.spec.generate",
                    "capability_id": "document.spec.generate",
                    "tool_name": "document.spec.generate",
                    "adapter_type": "capability",
                }
            },
            request_ids=["document.spec.generate"],
        ),
        created_at=now,
        updated_at=now,
        critic_required=0,
    )

    payload = main._task_payload_from_record(record, correlation_id="corr", context={})

    dispatch = main.execution_contracts.build_task_dispatch_payload(payload)
    assert dispatch.task_id == record.id
    assert dispatch.plan_id == "plan-ref"
    assert dispatch.correlation_id == "corr"
    assert dispatch.trace_id == "corr"
    assert dispatch.attempts == 1
    assert dispatch.max_attempts == 1
    assert dispatch.capability_requests == ["document.spec.generate"]
    assert dispatch.tool_requests == ["llm_generate_document_spec"]
    assert main.execution_contracts.EXECUTION_BINDINGS_KEY not in payload["tool_inputs"]
    assert dispatch.capability_bindings["llm_generate_document_spec"].capability_id == (
        "document.spec.generate"
    )
    assert dispatch.capability_bindings["llm_generate_document_spec"].tool_name == (
        "llm_generate_document_spec"
    )
    assert dispatch.tool_inputs_resolved is True


def test_plan_preflight_compiler_accepts_valid_dependency_chain() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="json chain",
        dag_edges=[["MakeJson", "ReuseJson"]],
        tasks=[
            models.TaskCreate(
                name="MakeJson",
                description="Build json",
                instruction="Build",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["utility.json.transform"],
                tool_inputs={"utility.json.transform": {"input": {"name": "demo"}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="ReuseJson",
                description="Reuse json",
                instruction="Reuse",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.transform,
                deps=["MakeJson"],
                tool_requests=["utility.json.transform"],
                tool_inputs={
                    "utility.json.transform": {
                        "input": {
                            "$from": "dependencies_by_name.MakeJson.utility.json.transform.result"
                        }
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert errors == {}


def test_plan_preflight_compiler_flags_broken_reference_path() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="broken ref",
        dag_edges=[["MakeJson", "ReuseJson"]],
        tasks=[
            models.TaskCreate(
                name="MakeJson",
                description="Build json",
                instruction="Build",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["utility.json.transform"],
                tool_inputs={"utility.json.transform": {"input": {"name": "demo"}}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="ReuseJson",
                description="Reuse json",
                instruction="Reuse",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/json_object",
                intent=models.ToolIntent.transform,
                deps=["MakeJson"],
                tool_requests=["utility.json.transform"],
                tool_inputs={
                    "utility.json.transform": {
                        "input": {"$from": "dependencies_by_name.MakeJson.missing_tool.result"}
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert "ReuseJson" in errors
    assert "input reference resolution failed" in errors["ReuseJson"]


def test_plan_preflight_accepts_reference_path_with_dotted_tool_name() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="github repo ref",
        dag_edges=[["Verify repository exists", "Decide whether to proceed"]],
        tasks=[
            models.TaskCreate(
                name="Verify repository exists",
                description="Verify repository exists",
                instruction="Check repo",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {"query": "repo:demo owner:octocat"}},
                critic_required=False,
            ),
            models.TaskCreate(
                name="Decide whether to proceed",
                description="Decide whether to proceed",
                instruction="Use repository check output",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/validation_report",
                intent=models.ToolIntent.transform,
                deps=["Verify repository exists"],
                tool_requests=["utility.json.transform"],
                tool_inputs={
                    "utility.json.transform": {
                        "input": {
                            "$from": "dependencies_by_name.Verify repository exists.github.repo.list"
                        }
                    }
                },
                critic_required=False,
            ),
        ],
    )
    errors = main._compile_plan_preflight(plan, job_context={})
    assert errors == {}


def test_plan_preflight_ignores_non_matching_intent_segments_when_suggested_capabilities_present() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="derive path",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="DeriveOutput",
                description="Derive filename",
                instruction="Derive output path",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/output_path",
                intent=models.ToolIntent.transform,
                deps=[],
                tool_requests=["derive_output_filename"],
                tool_inputs={
                    "derive_output_filename": {
                        "candidate_name": "Anjali Surabhi",
                        "target_role_name": "Associate Analyst",
                        "company_name": "Molina Healthcare",
                        "document_type": "document",
                        "output_extension": "docx",
                        "output_dir": "documents",
                        "today": "2026-02-28",
                    }
                },
                critic_required=False,
            )
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "render",
                "objective": "Render docx",
                "required_inputs": ["document_spec", "output_path"],
                "suggested_capabilities": ["document.docx.render"],
            }
        ]
    }
    errors = main._compile_plan_preflight(
        plan,
        job_context={},
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_plan_preflight_uses_task_instruction_for_intent_segment_contract() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="abort if missing",
        dag_edges=[["VerifyRepoExists", "AbortIfRepoMissing"]],
        tasks=[
            models.TaskCreate(
                name="VerifyRepoExists",
                description="Verify repository exists",
                instruction="Call github.repo.list to verify the repository exists.",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={
                    "github.repo.list": {
                        "query": "repo:scientific-agent-lab owner:narendersurabhhi"
                    }
                },
                critic_required=False,
            ),
            models.TaskCreate(
                name="AbortIfRepoMissing",
                description="Stop if repository missing",
                instruction=(
                    "Inspect the VerifyRepoExists output; if the repository is not found, "
                    "mark the plan aborted and stop execution."
                ),
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/validation_report",
                intent=models.ToolIntent.generate,
                deps=["VerifyRepoExists"],
                tool_requests=["llm.text.generate"],
                tool_inputs={},
                critic_required=True,
            ),
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "generate",
                "objective": "Stop if repository missing",
                "required_inputs": ["instruction"],
                "suggested_capabilities": ["llm.text.generate"],
                "slots": {
                    "entity": "repository",
                    "artifact_type": "validation_report",
                    "output_format": "txt",
                    "risk_level": "read_only",
                    "must_have_inputs": ["instruction"],
                },
            }
        ]
    }

    errors = main._compile_plan_preflight(
        plan,
        job_context={},
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_plan_preflight_uses_synthesized_github_query_for_intent_segment_contract() -> None:
    plan = models.PlanCreate(
        planner_version="test",
        tasks_summary="repo check",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="CheckRepoExists",
                description="Verify repository exists",
                instruction="Call github.repo.list to verify the repository exists.",
                acceptance_criteria=["done"],
                expected_output_schema_ref="schemas/github_repo_list_result",
                intent=models.ToolIntent.validate,
                deps=[],
                tool_requests=["github.repo.list"],
                tool_inputs={"github.repo.list": {}},
                critic_required=False,
            )
        ],
    )
    goal_intent_graph = {
        "segments": [
            {
                "id": "s1",
                "intent": "validate",
                "objective": "Verify repository exists",
                "required_inputs": ["query"],
                "suggested_capabilities": ["github.repo.list"],
                "slots": {
                    "entity": "repository",
                    "artifact_type": "validation_report",
                    "output_format": None,
                    "risk_level": "read_only",
                    "must_have_inputs": ["query"],
                },
            }
        ]
    }

    errors = main._compile_plan_preflight(
        plan,
        job_context={
            "repo_owner": "narendersurabhi",
            "repo_name": "scientific-agent-lab",
        },
        goal_intent_graph=goal_intent_graph,
    )
    assert errors == {}


def test_retry_task_from_dlq_resets_task_and_deletes_stream_entry(monkeypatch):
    job_id = f"job-retry-task-{uuid.uuid4()}"
    plan_id = f"plan-retry-task-{uuid.uuid4()}"
    task_id = f"task-retry-task-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="retry one task",
                context_json={},
                status=models.JobStatus.failed.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.failed.value,
                deps=[],
                attempts=2,
                max_attempts=3,
                rework_count=1,
                max_reworks=2,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    class _RedisStub:
        def __init__(self):
            self.deleted = []

        def xdel(self, stream, stream_id):
            self.deleted.append((stream, stream_id))
            return 1

    redis_stub = _RedisStub()
    monkeypatch.setattr(main, "redis_client", redis_stub)
    captured = {"called": False}
    monkeypatch.setattr(
        main,
        "_enqueue_ready_tasks",
        lambda *args, **kwargs: captured.__setitem__("called", True),
    )

    response = client.post(
        f"/jobs/{job_id}/tasks/{task_id}/retry",
        json={"stream_id": "99-0"},
    )
    assert response.status_code == 200
    assert captured["called"] is True
    assert redis_stub.deleted == [(events.TASK_DLQ_STREAM, "99-0")]

    with SessionLocal() as db:
        refreshed = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert refreshed is not None
        assert refreshed.status == models.TaskStatus.pending.value
        assert refreshed.attempts == 0
        assert refreshed.rework_count == 0


def test_retry_task_from_dlq_requires_failed_status():
    job_id = f"job-retry-task-state-{uuid.uuid4()}"
    plan_id = f"plan-retry-task-state-{uuid.uuid4()}"
    task_id = f"task-retry-task-state-{uuid.uuid4()}"
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="retry one task state",
                context_json={},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            PlanRecord(
                id=plan_id,
                job_id=job_id,
                planner_version="test",
                created_at=now,
                tasks_summary="single task",
                dag_edges=[],
                policy_decision={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="only-task",
                description="desc",
                instruction="do",
                acceptance_criteria=[],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.completed.value,
                deps=[],
                attempts=1,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                intent=None,
                tool_requests=[],
                tool_inputs={},
                created_at=now,
                updated_at=now,
                critic_required=0,
            )
        )
        db.commit()

    response = client.post(f"/jobs/{job_id}/tasks/{task_id}/retry", json={"stream_id": "100-0"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Task is not failed"


def test_list_rag_documents_proxies_to_retriever(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_rag_request(path: str, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {
            "collection_name": "rag_default",
            "truncated": False,
            "scanned_point_count": 2,
            "documents": [
                {
                    "document_id": "docs/user-guide.md",
                    "source_uri": "docs/user-guide.md",
                    "namespace": "docs",
                    "user_id": "narendersurabhi",
                    "chunk_count": 4,
                    "metadata": {},
                }
            ],
        }

    monkeypatch.setattr(main, "_rag_retriever_request_json", _fake_rag_request)

    response = client.get(
        "/rag/documents",
        params={
            "namespace": "docs",
            "user_id": "narendersurabhi",
            "query": "guide",
            "limit": 25,
        },
    )

    assert response.status_code == 200
    assert response.json()["documents"][0]["document_id"] == "docs/user-guide.md"
    assert captured["path"] == "/documents/list"
    assert captured["method"] == "POST"
    assert captured["body"] == {
        "collection_name": None,
        "namespace": "docs",
        "tenant_id": None,
        "user_id": "narendersurabhi",
        "workspace_id": None,
        "query": "guide",
        "limit": 25,
    }


def test_index_rag_text_mode_builds_single_entry_upsert(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_rag_request(path: str, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {
            "collection_name": "rag_default",
            "upserted_count": 1,
            "chunk_ids": ["chunk-1"],
        }

    monkeypatch.setattr(main, "_rag_retriever_request_json", _fake_rag_request)

    response = client.post(
        "/rag/index",
        json={
            "mode": "text",
            "collection_name": "rag_default",
            "namespace": "docs",
            "user_id": "narendersurabhi",
            "document_id": "manual/doc-1",
            "source_uri": "manual/doc-1",
            "text": "Agentic Workflow Studio ships a visual DAG editor.",
            "metadata": {"doc_type": "note"},
        },
    )

    assert response.status_code == 200
    assert response.json()["upserted_count"] == 1
    assert captured["path"] == "/index/upsert_texts"
    assert captured["method"] == "POST"
    assert captured["body"] == {
        "collection_name": "rag_default",
        "ensure_collection": True,
        "namespace": "docs",
        "tenant_id": None,
        "user_id": "narendersurabhi",
        "workspace_id": None,
        "entries": [
            {
                "document_id": "manual/doc-1",
                "text": "Agentic Workflow Studio ships a visual DAG editor.",
                "source_uri": "manual/doc-1",
                "metadata": {"doc_type": "note"},
            }
        ],
    }


def test_replace_rag_document_deletes_then_reindexes(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_rag_request(path: str, **kwargs: object) -> dict[str, object]:
        calls.append((path, dict(kwargs)))
        if path == "/documents/delete":
            return {
                "collection_name": "rag_default",
                "document_id": "docs/user-guide.md",
                "deleted_chunk_count": 3,
            }
        if path == "/index/markdown":
            return {
                "collection_name": "rag_default",
                "document_id": "docs/user-guide.md",
                "source_uri": "docs/user-guide.md",
                "section_count": 2,
                "chunk_count": 3,
                "upserted_count": 3,
                "chunk_ids": ["c1", "c2", "c3"],
            }
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(main, "_rag_retriever_request_json", _fake_rag_request)

    response = client.put(
        "/rag/documents",
        params={"document_id": "docs/user-guide.md"},
        json={
            "mode": "markdown",
            "collection_name": "rag_default",
            "namespace": "docs",
            "user_id": "narendersurabhi",
            "markdown_text": "# User Guide\n\nUpdated content.",
            "metadata": {"doc_type": "markdown"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"]["deleted_chunk_count"] == 3
    assert body["indexed"]["upserted_count"] == 3
    assert [path for path, _kwargs in calls] == ["/documents/delete", "/index/markdown"]
    assert calls[0][1]["body"] == {
        "document_id": "docs/user-guide.md",
        "collection_name": "rag_default",
        "namespace": "docs",
        "tenant_id": None,
        "user_id": "narendersurabhi",
        "workspace_id": None,
    }
    assert calls[1][1]["body"] == {
        "markdown_text": "# User Guide\n\nUpdated content.",
        "collection_name": "rag_default",
        "ensure_collection": True,
        "document_id": "docs/user-guide.md",
        "source_uri": "docs/user-guide.md",
        "namespace": "docs",
        "tenant_id": None,
        "user_id": "narendersurabhi",
        "workspace_id": None,
        "chunk_size_chars": None,
        "chunk_overlap_chars": None,
        "max_chunks": None,
        "metadata": {"doc_type": "markdown"},
    }
