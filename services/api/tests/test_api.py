import os
import uuid
from datetime import datetime

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, engine
from services.api.app.database import SessionLocal
from services.api.app.models import JobRecord, PlanRecord, TaskRecord
from libs.core import models


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
