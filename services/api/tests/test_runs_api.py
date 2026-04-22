import json
import os
import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"
os.environ["INTENT_VECTOR_SEARCH_ENABLED"] = "false"
os.environ["CHAT_INTENT_VECTOR_SEARCH_ENABLED"] = "false"

from libs.core import models  # noqa: E402
from services.api.app import main  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from services.api.app.models import (  # noqa: E402
    ExecutionRequestRecord,
    JobRecord,
    RunRecord,
    RunStepRecord,
    StepAttemptRecord,
    StepCheckpointRecord,
    TaskRecord,
)


Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _create_job(goal: str | None = None) -> dict[str, str]:
    response = client.post(
        "/jobs",
        json={
            "goal": goal or f"run-shadow-{uuid.uuid4()}",
            "context_json": {},
            "priority": 1,
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_plan(job_id: str) -> dict[str, str]:
    response = client.post(
        f"/plans?job_id={job_id}",
        json={
            "planner_version": "test",
            "tasks_summary": "List workspace files",
            "dag_edges": [],
            "tasks": [
                {
                    "name": "ListWorkspace",
                    "description": "List workspace files",
                    "instruction": "List the files in the workspace",
                    "acceptance_criteria": ["returns files"],
                    "expected_output_schema_ref": "schemas/workspace_listing",
                    "deps": [],
                    "tool_requests": ["filesystem.workspace.list"],
                    "tool_inputs": {"filesystem.workspace.list": {}},
                    "critic_required": False,
                }
            ],
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_workflow_run() -> dict[str, dict]:
    create_response = client.post(
        "/workflows/definitions",
        json={
            "title": f"Workspace listing {uuid.uuid4()}",
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

    run_response = client.post(f"/workflows/versions/{version['id']}/run", json={"priority": 2})
    assert run_response.status_code == 200
    return run_response.json()


def test_create_job_surfaces_run_id_and_creates_shadow_run() -> None:
    job = _create_job()

    assert job["run_id"] == job["id"]

    with SessionLocal() as db:
        record = db.query(RunRecord).filter(RunRecord.id == job["run_id"]).first()
        assert record is not None
        assert record.job_id == job["id"]
        assert record.kind == models.RunKind.planner.value


def test_workflow_run_creates_shadow_run_with_workflow_run_id() -> None:
    run_body = _create_workflow_run()
    workflow_run = run_body["workflow_run"]
    job = run_body["job"]

    assert workflow_run["run_id"] == workflow_run["id"]
    assert job["run_id"] == workflow_run["id"]

    with SessionLocal() as db:
        record = db.query(RunRecord).filter(RunRecord.id == workflow_run["id"]).first()
        assert record is not None
        assert record.workflow_run_id == workflow_run["id"]
        assert record.job_id == job["id"]
        assert record.kind == models.RunKind.studio.value


def test_runs_list_and_get_return_shadow_run() -> None:
    job = _create_job()
    run_id = job["run_id"]

    list_response = client.get("/runs", params={"kind": models.RunKind.planner.value, "limit": 50})
    assert list_response.status_code == 200
    runs = list_response.json()
    matching = [run for run in runs if run["id"] == run_id]
    assert matching

    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    run = get_response.json()
    assert run["id"] == run_id
    assert run["job_id"] == job["id"]
    assert run["kind"] == models.RunKind.planner.value


def test_runs_steps_returns_shadow_steps_after_plan_creation() -> None:
    job = _create_job()
    _create_plan(job["id"])

    response = client.get(f"/runs/{job['run_id']}/steps")
    assert response.status_code == 200
    steps = response.json()

    assert len(steps) == 1
    assert steps[0]["run_id"] == job["run_id"]
    assert steps[0]["task_id"] == steps[0]["id"]
    assert steps[0]["spec_step_id"]
    assert steps[0]["capability_request_id"] == "filesystem.workspace.list"
    assert steps[0]["capability_id"] == "filesystem.workspace.list"


def test_run_debugger_includes_execution_requests() -> None:
    job = _create_job()
    _create_plan(job["id"])

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        payload = main._task_payload_from_record(task, correlation_id=f"corr-{uuid.uuid4()}", context={})
        assert payload["task_id"] == task.id

    response = client.get(f"/runs/{job['run_id']}/debugger")
    assert response.status_code == 200
    debugger = response.json()

    assert debugger["run"]["id"] == job["run_id"]
    assert debugger["execution_requests"]
    assert debugger["steps"][0]["execution_requests"]
    assert debugger["execution_requests"][0]["run_id"] == job["run_id"]


def test_execution_request_snapshot_captures_retry_policy_and_context_provenance() -> None:
    job = _create_job()
    _create_plan(job["id"])

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        payload = main._task_payload_from_record(
            task,
            correlation_id=f"corr-{uuid.uuid4()}",
            context={
                "job_context": {"workspace_id": "demo"},
                "dependencies": {"dep-1": {"status": "completed"}},
                "dependencies_by_name": {"ListWorkspace": {"files": []}},
            },
        )
        assert payload["task_id"] == task.id
        record = (
            db.query(ExecutionRequestRecord)
            .filter(ExecutionRequestRecord.run_id == job["run_id"])
            .first()
        )
        assert record is not None
        assert record.request_id == "workspace_list_files"
        assert record.capability_id == "filesystem.workspace.list"
        assert record.retry_policy_json["max_attempts"] == 3
        assert record.retry_policy_json["max_reworks"] == 2
        assert record.policy_snapshot_json["critic_required"] is False
        assert "job_context_keys" in record.context_provenance_json
        assert record.context_provenance_json["job_context_keys"] == ["workspace_id"]


def test_task_started_and_heartbeat_update_attempt_leases_and_execution_request() -> None:
    job = _create_job()
    _create_plan(job["id"])
    correlation_id = f"corr-{uuid.uuid4()}"

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        main._task_payload_from_record(task, correlation_id=correlation_id, context={})
        task_id = task.id

    started_at = _utcnow()
    main._handle_event(
        "tasks.events",
        {
            "data": json.dumps(
                {
                    "type": "task.started",
                    "job_id": job["id"],
                    "task_id": task_id,
                    "occurred_at": started_at.isoformat(),
                    "payload": {
                        "task_id": task_id,
                        "attempts": 1,
                        "max_attempts": 3,
                        "worker_consumer": "worker-phase2",
                    },
                    "correlation_id": correlation_id,
                }
            )
        },
    )

    heartbeat_at = _utcnow()
    main._handle_event(
        "tasks.events",
        {
            "data": json.dumps(
                {
                    "type": "task.heartbeat",
                    "job_id": job["id"],
                    "task_id": task_id,
                    "occurred_at": heartbeat_at.isoformat(),
                    "payload": {
                        "task_id": task_id,
                        "attempts": 1,
                        "status": "heartbeat",
                        "worker_consumer": "worker-phase2",
                    },
                    "correlation_id": correlation_id,
                }
            )
        },
    )

    with SessionLocal() as db:
        attempt = db.query(StepAttemptRecord).filter(StepAttemptRecord.step_id == task_id).first()
        request = (
            db.query(ExecutionRequestRecord)
            .filter(ExecutionRequestRecord.run_id == job["run_id"])
            .first()
        )
        assert attempt is not None
        assert request is not None
        assert attempt.lease_owner == "worker-phase2"
        assert attempt.last_heartbeat_at is not None
        assert attempt.lease_expires_at is not None
        assert attempt.heartbeat_count == 1
        assert request.status == "heartbeat"
        assert request.lease_owner == "worker-phase2"
        assert request.last_heartbeat_at is not None
        assert request.lease_expires_at is not None


def test_task_started_accepts_existing_naive_step_attempt_timestamp() -> None:
    job = _create_job()
    _create_plan(job["id"])
    correlation_id = f"corr-{uuid.uuid4()}"

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        main._task_payload_from_record(task, correlation_id=correlation_id, context={})
        task_id = task.id

    started_at = _utcnow()
    main._handle_event(
        "tasks.events",
        {
            "data": json.dumps(
                {
                    "type": "task.started",
                    "job_id": job["id"],
                    "task_id": task_id,
                    "occurred_at": started_at.isoformat(),
                    "payload": {
                        "task_id": task_id,
                        "attempts": 1,
                        "worker_consumer": "worker-phase2",
                    },
                    "correlation_id": correlation_id,
                }
            )
        },
    )

    with SessionLocal() as db:
        attempt = db.query(StepAttemptRecord).filter(StepAttemptRecord.step_id == task_id).first()
        assert attempt is not None
        attempt.started_at = started_at.replace(tzinfo=None)
        db.commit()

    main._handle_event(
        "tasks.events",
        {
            "data": json.dumps(
                {
                    "type": "task.started",
                    "job_id": job["id"],
                    "task_id": task_id,
                    "occurred_at": started_at.isoformat(),
                    "payload": {
                        "task_id": task_id,
                        "attempts": 1,
                        "worker_consumer": "worker-phase2",
                    },
                    "correlation_id": correlation_id,
                }
            )
        },
    )

    with SessionLocal() as db:
        attempt = db.query(StepAttemptRecord).filter(StepAttemptRecord.step_id == task_id).first()
        assert attempt is not None
        assert attempt.status == models.TaskStatus.running.value


def test_task_completed_ignores_missing_step_attempt_on_execution_request(monkeypatch) -> None:
    job = _create_job()
    _create_plan(job["id"])
    correlation_id = f"corr-{uuid.uuid4()}"

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        main._task_payload_from_record(task, correlation_id=correlation_id, context={})
        task_id = task.id

    def _fake_finished_attempt(*_args, **_kwargs) -> StepAttemptRecord:
        return StepAttemptRecord(
            id="missing-step-attempt-id",
            run_id=job["run_id"],
            job_id=job["id"],
            step_id=task_id,
            attempt_number=1,
            status=models.TaskStatus.completed.value,
            worker_id="worker-phase2",
            started_at=_utcnow(),
            finished_at=_utcnow(),
            error_code=None,
            error_message=None,
            retry_classification="succeeded",
            lease_owner="worker-phase2",
            lease_expires_at=_utcnow(),
            last_heartbeat_at=_utcnow(),
            heartbeat_count=0,
            result_summary_json={},
        )

    monkeypatch.setattr(main, "_upsert_step_attempt_finished", _fake_finished_attempt)

    main._handle_event(
        "tasks.events",
        {
            "data": json.dumps(
                {
                    "type": "task.completed",
                    "job_id": job["id"],
                    "task_id": task_id,
                    "occurred_at": _utcnow().isoformat(),
                    "payload": {
                        "task_id": task_id,
                        "attempts": 1,
                        "worker_consumer": "worker-phase2",
                        "outputs": {"ok": True},
                    },
                    "correlation_id": correlation_id,
                }
            )
        },
    )

    with SessionLocal() as db:
        request = (
            db.query(ExecutionRequestRecord)
            .filter(ExecutionRequestRecord.run_id == job["run_id"])
            .first()
        )
        assert request is not None
        assert request.status == models.TaskStatus.completed.value
        assert request.step_attempt_id is None


def test_run_control_endpoints_delegate_to_job_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(main, "_dispatch_ready_work_for_job", lambda *_args, **_kwargs: None)

    job = _create_job()
    _create_plan(job["id"])
    run_id = job["run_id"]

    cancel_response = client.post(f"/runs/{run_id}/cancel")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == models.JobStatus.canceled.value

    with SessionLocal() as db:
        job_record = db.query(JobRecord).filter(JobRecord.id == job["id"]).first()
        assert job_record is not None
        assert job_record.status == models.JobStatus.canceled.value

    resume_response = client.post(f"/runs/{run_id}/resume")
    assert resume_response.status_code == 200
    assert resume_response.json()["status"] == models.JobStatus.planning.value

    with SessionLocal() as db:
        job_record = db.query(JobRecord).filter(JobRecord.id == job["id"]).first()
        task_record = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert job_record is not None
        assert task_record is not None
        job_record.status = models.JobStatus.failed.value
        job_record.updated_at = _utcnow()
        task_record.status = models.TaskStatus.failed.value
        task_record.attempts = 2
        task_record.rework_count = 1
        task_record.updated_at = _utcnow()
        db.commit()

    retry_response = client.post(f"/runs/{run_id}/retry")
    assert retry_response.status_code == 200
    assert retry_response.json()["status"] == models.JobStatus.planning.value

    with SessionLocal() as db:
        job_record = db.query(JobRecord).filter(JobRecord.id == job["id"]).first()
        task_record = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert job_record is not None
        assert task_record is not None
        assert job_record.status == models.JobStatus.planning.value
        assert task_record.status == models.TaskStatus.pending.value
        assert task_record.attempts == 0
        assert task_record.rework_count == 0


def test_clear_job_deletes_shadow_run_records() -> None:
    job = _create_job()
    _create_plan(job["id"])
    run_id = job["run_id"]

    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.job_id == job["id"]).first()
        assert task is not None
        main._task_payload_from_record(task, correlation_id=f"corr-{uuid.uuid4()}", context={})
        checkpoint = StepCheckpointRecord(
            id=f"checkpoint-{uuid.uuid4()}",
            run_id=run_id,
            job_id=job["id"],
            step_id=task.id,
            step_attempt_id=None,
            checkpoint_key="initial",
            payload_json={"stage": "prepared"},
            input_digest="digest",
            replay_count=0,
            source="test",
            outcome="pending",
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        db.add(checkpoint)
        db.commit()

    response = client.post(f"/jobs/{job['id']}/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

    with SessionLocal() as db:
        assert db.query(RunRecord).filter(RunRecord.id == run_id).first() is None
        assert db.query(RunStepRecord).filter(RunStepRecord.run_id == run_id).first() is None
        assert (
            db.query(ExecutionRequestRecord)
            .filter(ExecutionRequestRecord.run_id == run_id)
            .first()
            is None
        )
        assert db.query(StepCheckpointRecord).filter(StepCheckpointRecord.run_id == run_id).first() is None
