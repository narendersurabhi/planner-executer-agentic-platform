import os
import uuid
from datetime import datetime

import redis
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, engine
from services.api.app.database import SessionLocal
from services.api.app.models import JobRecord, PlanRecord, TaskRecord
from libs.core import events, models


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def test_create_job():
    response = client.post(
        "/jobs",
        json={"goal": "demo", "context_json": {}, "priority": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "demo"


def test_event_stream():
    response = client.get("/events/stream?once=true")
    assert response.status_code == 200


def test_job_details():
    job_id = f"job-details-{uuid.uuid4()}"
    plan_id = f"plan-details-{uuid.uuid4()}"
    task_id = f"task-details-{uuid.uuid4()}"
    now = datetime.utcnow()
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="details",
                context_json={},
                status=models.JobStatus.queued.value,
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


def test_plan_created_enqueues_ready_tasks():
    job_id = f"job-test-plan-{uuid.uuid4()}"
    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="demo",
                context_json={},
                status=models.JobStatus.queued.value,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
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
                tool_requests=[],
                critic_required=False,
            ),
            models.TaskCreate(
                name="t2",
                description="second",
                instruction="do second",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=["t1"],
                tool_requests=[],
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


def test_handle_task_started_sets_task_running_and_job_running():
    job_id = f"job-task-started-{uuid.uuid4()}"
    plan_id = f"plan-task-started-{uuid.uuid4()}"
    task_id = f"task-task-started-{uuid.uuid4()}"
    now = datetime.utcnow()
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


def test_retry_task_from_dlq_resets_task_and_deletes_stream_entry(monkeypatch):
    job_id = f"job-retry-task-{uuid.uuid4()}"
    plan_id = f"plan-retry-task-{uuid.uuid4()}"
    task_id = f"task-retry-task-{uuid.uuid4()}"
    now = datetime.utcnow()
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
    now = datetime.utcnow()
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
