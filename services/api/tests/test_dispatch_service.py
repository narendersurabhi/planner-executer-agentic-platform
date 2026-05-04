import os
import uuid
from datetime import UTC, datetime

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"

from libs.core import models
from services.api.app import dispatch_service, main
from services.api.app.database import Base, SessionLocal, engine
from services.api.app.models import JobRecord, TaskRecord


Base.metadata.create_all(bind=engine)


def test_enqueue_ready_tasks_emits_via_callback() -> None:
    job_id = f"job-dispatch-{uuid.uuid4()}"
    plan_id = f"plan-dispatch-{uuid.uuid4()}"
    task_id = f"task-dispatch-{uuid.uuid4()}"
    now = datetime.now(UTC)

    with SessionLocal() as db:
        db.add(
            JobRecord(
                id=job_id,
                goal="Dispatch ready work",
                context_json={},
                status=models.JobStatus.running.value,
                created_at=now,
                updated_at=now,
                priority=0,
                metadata_json={},
            )
        )
        db.add(
            TaskRecord(
                id=task_id,
                job_id=job_id,
                plan_id=plan_id,
                name="Dispatch task",
                description="ready task",
                instruction="do work",
                acceptance_criteria=["done"],
                expected_output_schema_ref="TaskResult",
                status=models.TaskStatus.pending.value,
                deps=[],
                attempts=0,
                max_attempts=3,
                rework_count=0,
                max_reworks=0,
                assigned_to="worker",
                tool_requests=[],
                tool_inputs={},
                critic_required=False,
                created_at=now,
                updated_at=now,
            )
        )
        db.commit()

    emitted: list[tuple[str, dict]] = []
    refreshed: list[str] = []
    runtime = dispatch_service.ApiDispatchRuntime(
        redis_client=None,
        session_factory=SessionLocal,
        logger=main.logger,
        config=dispatch_service.ApiDispatchConfig(
            event_outbox_enabled=False,
            event_outbox_batch_size=10,
            event_outbox_poll_s=0.1,
            event_outbox_redis_retries=1,
            event_outbox_redis_retry_sleep_s=0.0,
            policy_gate_enabled=False,
            tool_input_validation_enabled=False,
            tool_input_schemas={},
        ),
    )
    callbacks = dispatch_service.ApiDispatchCallbacks(
        stream_for_event=main._stream_for_event,
        resolve_task_deps=main._resolve_task_deps,
        build_task_context=lambda _task_id, _task_map, _id_to_name, job_context: {
            "job_context": dict(job_context),
        },
        project_execution_context=lambda _goal, _context, _metadata: {"projected": True},
        coerce_task_intent_profiles=lambda _metadata: {},
        normalize_task_intent_profile_segment=lambda segment: segment,
        refresh_job_status=refreshed.append,
        emit_event=lambda event_type, payload: emitted.append((event_type, payload)),
    )

    dispatch_service.enqueue_ready_tasks(
        job_id,
        plan_id,
        "corr-dispatch",
        runtime=runtime,
        callbacks=callbacks,
    )

    with SessionLocal() as db:
        record = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert record is not None
        assert record.status == models.TaskStatus.ready.value
        assert record.attempts == 1

    assert emitted
    assert emitted[0][0] == "task.ready"
    assert emitted[0][1]["task_id"] == task_id
    assert emitted[0][1]["context"]["job_context"] == {"projected": True}
    assert refreshed == [job_id]
