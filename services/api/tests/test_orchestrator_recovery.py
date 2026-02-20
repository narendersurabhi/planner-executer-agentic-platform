import os
import uuid
from datetime import datetime

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from services.api.app.models import JobRecord, PlanRecord  # noqa: E402
from libs.core import models  # noqa: E402


Base.metadata.create_all(bind=engine)


class _RedisStub:
    def __init__(self, response):
        self._response = response
        self.acked = []

    def xautoclaim(self, *args, **kwargs):
        return self._response

    def xack(self, stream, group, message_id):
        self.acked.append((stream, group, message_id))


def test_recover_pending_events_accepts_two_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _RedisStub(("0-1", [("msg-1", {"data": "{}"})]))
    called = {"count": 0}

    def _handle_event(stream, data):
        called["count"] += 1

    monkeypatch.setattr(main, "_handle_event", _handle_event)
    main._recover_pending_events(stub, "group", "consumer", ["stream"])
    assert called["count"] == 1
    assert stub.acked == [("stream", "group", "msg-1")]


def test_recover_pending_events_accepts_three_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _RedisStub(("0-1", [("msg-2", {"data": "{}"})], ["msg-x"]))
    called = {"count": 0}

    def _handle_event(stream, data):
        called["count"] += 1

    monkeypatch.setattr(main, "_handle_event", _handle_event)
    main._recover_pending_events(stub, "group", "consumer", ["stream"])
    assert called["count"] == 1
    assert stub.acked == [("stream", "group", "msg-2")]


def test_handle_plan_created_uses_plan_id_after_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = f"job-plan-recover-{uuid.uuid4()}"
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

    payload = models.PlanCreate(
        planner_version="test",
        tasks_summary="single",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="task-a",
                description="desc",
                instruction="do",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="TaskResult",
                deps=[],
                tool_requests=[],
                critic_required=False,
            )
        ],
    ).model_dump()
    payload["job_id"] = job_id

    captured = {"plan_id": None}

    def _enqueue_ready_tasks(job_id_arg, plan_id_arg, correlation_id):
        captured["plan_id"] = plan_id_arg

    monkeypatch.setattr(main, "_enqueue_ready_tasks", _enqueue_ready_tasks)
    monkeypatch.setattr(main, "_refresh_job_status", lambda job_id_arg: None)

    envelope = {"type": "plan.created", "payload": payload, "job_id": job_id}
    main._handle_plan_created(envelope)

    assert captured["plan_id"] is not None
    with SessionLocal() as db:
        record = db.query(PlanRecord).filter(PlanRecord.id == captured["plan_id"]).first()
        assert record is not None
