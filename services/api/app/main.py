from __future__ import annotations

import json
import os
import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Generator, List

import redis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from sqlalchemy.orm import Session

from libs.core import events, logging as core_logging, models, orchestrator, payload_resolver, state_machine
from .database import Base, SessionLocal, engine
from .models import JobRecord, PlanRecord, TaskRecord

core_logging.configure_logging("api")
logger = logging.getLogger("api.orchestrator")

app = FastAPI(title="Agentic Planner Executor API")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3002,http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
ORCHESTRATOR_ENABLED = os.getenv("ORCHESTRATOR_ENABLED", "true").lower() == "true"
POLICY_GATE_ENABLED = os.getenv("POLICY_GATE_ENABLED", "false").lower() == "true"
JOB_RECOVERY_ENABLED = os.getenv("JOB_RECOVERY_ENABLED", "true").lower() == "true"
ORCHESTRATOR_RECOVER_PENDING = (
    os.getenv("ORCHESTRATOR_RECOVER_PENDING", "true").lower() == "true"
)
ORCHESTRATOR_RECOVER_IDLE_MS = int(os.getenv("ORCHESTRATOR_RECOVER_IDLE_MS", "60000"))
REPLAN_MAX = int(os.getenv("REPLAN_MAX", "1"))
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
TASK_OUTPUT_KEY_PREFIX = "task_output:"
TASK_RESULT_KEY_PREFIX = "task_result:"


@app.on_event("startup")
def _init_db() -> None:
    Base.metadata.create_all(bind=engine)
    if ORCHESTRATOR_ENABLED:
        _start_orchestrator()
    if JOB_RECOVERY_ENABLED:
        _recover_jobs()

jobs_created_total = Counter("jobs_created_total", "Jobs created")
orchestrator_loop_errors_total = Counter(
    "orchestrator_loop_errors_total", "Orchestrator loop errors"
)
orchestrator_handle_errors_total = Counter(
    "orchestrator_handle_errors_total", "Orchestrator event handling errors", ["stream"]
)
orchestrator_recovered_events_total = Counter(
    "orchestrator_recovered_events_total", "Orchestrator recovered pending events", ["stream"]
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _job_from_record(record: JobRecord) -> models.Job:
    return models.Job(
        id=record.id,
        goal=record.goal,
        context_json=record.context_json or {},
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        priority=record.priority or 0,
        metadata=record.metadata_json or {},
    )


def _plan_from_record(record: PlanRecord) -> models.Plan:
    return models.Plan(
        id=record.id,
        job_id=record.job_id,
        planner_version=record.planner_version,
        created_at=record.created_at,
        tasks_summary=record.tasks_summary,
        dag_edges=record.dag_edges or [],
        policy_decision=record.policy_decision or None,
    )


def _task_from_record(record: TaskRecord) -> models.Task:
    return models.Task(
        id=record.id,
        job_id=record.job_id,
        plan_id=record.plan_id,
        name=record.name,
        description=record.description,
        instruction=record.instruction,
        acceptance_criteria=record.acceptance_criteria or [],
        expected_output_schema_ref=record.expected_output_schema_ref,
        status=record.status,
        intent=record.intent,
        deps=record.deps or [],
        attempts=record.attempts or 0,
        max_attempts=record.max_attempts or 0,
        rework_count=record.rework_count or 0,
        max_reworks=record.max_reworks or 0,
        assigned_to=record.assigned_to,
        tool_requests=record.tool_requests or [],
        tool_inputs=record.tool_inputs or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
        critic_required=bool(record.critic_required),
    )


def _emit_event(event_type: str, payload: dict[str, Any]) -> None:
    envelope = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.utcnow(),
        correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
        job_id=payload.get("job_id") or payload.get("id"),
        task_id=payload.get("task_id"),
        payload=payload,
    )
    stream = _stream_for_event(event_type)
    redis_client.xadd(stream, {"data": envelope.model_dump_json()})


def _plan_created_payload(plan: models.PlanCreate, job_id: str) -> dict[str, Any]:
    payload = plan.model_dump()
    payload["job_id"] = job_id
    return payload


def _stream_for_event(event_type: str) -> str:
    if event_type.startswith("job"):
        return events.JOB_STREAM
    if event_type.startswith("plan"):
        return events.PLAN_STREAM
    if event_type.startswith("task"):
        return events.TASK_STREAM
    if event_type.startswith("policy"):
        return events.POLICY_STREAM
    return events.TASK_STREAM


def _start_orchestrator() -> None:
    thread = threading.Thread(target=_orchestrator_loop, daemon=True)
    thread.start()


def _orchestrator_loop() -> None:
    consumer = str(uuid.uuid4())
    group = "api-orchestrator"
    stream_keys = [events.PLAN_STREAM, events.TASK_STREAM, events.CRITIC_STREAM, events.POLICY_STREAM]
    local_redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    last_recovery = 0.0
    for stream in stream_keys:
        try:
            local_redis.xgroup_create(stream, group, id="0-0", mkstream=True)
        except redis.ResponseError:
            pass
    while True:
        try:
            now = time.time()
            if ORCHESTRATOR_RECOVER_PENDING and now - last_recovery > 30:
                _recover_pending_events(local_redis, group, consumer, stream_keys)
                last_recovery = now
            messages = local_redis.xreadgroup(
                group, consumer, {stream: ">" for stream in stream_keys}, count=10, block=1000
            )
            for stream_name, entries in messages:
                for message_id, data in entries:
                    try:
                        _handle_event(stream_name, data)
                        local_redis.xack(stream_name, group, message_id)
                    except Exception:
                        logger.exception(
                            "orchestrator_handle_event_error",
                            extra={"stream": stream_name, "message_id": message_id},
                        )
                        orchestrator_handle_errors_total.labels(stream=stream_name).inc()
        except Exception:
            logger.exception("orchestrator_loop_error")
            orchestrator_loop_errors_total.inc()
            time.sleep(1)


def _recover_pending_events(
    local_redis: redis.Redis,
    group: str,
    consumer: str,
    stream_keys: list[str],
) -> None:
    for stream in stream_keys:
        try:
            res = local_redis.xautoclaim(
                stream,
                group,
                consumer,
                min_idle_time=ORCHESTRATOR_RECOVER_IDLE_MS,
                start_id="0-0",
                count=50,
            )
            next_id, messages, *rest = res
            if rest:
                logger.info(
                    "orchestrator_recover_deleted_ids",
                    extra={"stream": stream, "deleted_ids": rest[0]},
                )
        except Exception:
            logger.exception("orchestrator_recover_error", extra={"stream": stream})
            continue
        if not messages:
            continue
        for message_id, data in messages:
            try:
                _handle_event(stream, data)
                local_redis.xack(stream, group, message_id)
                orchestrator_recovered_events_total.labels(stream=stream).inc()
            except Exception:
                logger.exception(
                    "orchestrator_recover_handle_error",
                    extra={"stream": stream, "message_id": message_id},
                )


def _handle_event(stream_name: str, data: dict[str, str]) -> None:
    try:
        envelope = json.loads(data.get("data", "{}"))
    except json.JSONDecodeError:
        return
    event_type = envelope.get("type")
    if event_type == "plan.created":
        _handle_plan_created(envelope)
    elif event_type == "plan.failed":
        _handle_plan_failed(envelope)
    elif event_type == "task.completed":
        _handle_task_completed(envelope)
    elif event_type == "task.failed":
        _handle_task_failed(envelope)
    elif event_type == "task.accepted":
        _handle_task_accepted(envelope)
    elif event_type == "task.rework_requested":
        _handle_task_rework(envelope)
    elif event_type == "policy.decision_made":
        _handle_policy_decision(envelope)


def _handle_plan_created(envelope: dict) -> None:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return
    job_id = envelope.get("job_id") or payload.get("job_id")
    if not job_id:
        return
    plan = _parse_plan_payload(payload)
    if plan is None:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        existing = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
        if existing:
            plan_id = existing.id
            db.commit()
            _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
            _refresh_job_status(job_id)
            return
        record = PlanRecord(
            id=str(uuid.uuid4()),
            job_id=job_id,
            planner_version=plan.planner_version,
            created_at=now,
            tasks_summary=plan.tasks_summary,
            dag_edges=plan.dag_edges,
            policy_decision={},
        )
        db.add(record)
        db.flush()
        plan_record_id = record.id
        for task in plan.tasks:
            task_record = TaskRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                plan_id=plan_record_id,
                name=task.name,
                description=task.description,
                instruction=task.instruction,
                acceptance_criteria=task.acceptance_criteria,
                expected_output_schema_ref=task.expected_output_schema_ref,
                status=models.TaskStatus.pending.value,
                intent=task.intent.value if isinstance(task.intent, models.ToolIntent) else task.intent,
                deps=task.deps,
                attempts=0,
                max_attempts=3,
                rework_count=0,
                max_reworks=2,
                assigned_to=None,
                tool_requests=task.tool_requests,
                tool_inputs=task.tool_inputs,
                created_at=now,
                updated_at=now,
                critic_required=1 if task.critic_required else 0,
            )
            db.add(task_record)
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            _set_job_status(job, models.JobStatus.planning)
            job.updated_at = now
        db.commit()
    _enqueue_ready_tasks(job_id, plan_record_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_plan_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    job_id = envelope.get("job_id") or payload.get("job_id") or payload.get("id")
    if not job_id:
        return
    error_message = payload.get("error")
    now = datetime.utcnow()
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not job:
            return
        _set_job_status(job, models.JobStatus.failed)
        if error_message:
            metadata = job.metadata_json or {}
            metadata["plan_error"] = error_message
            job.metadata_json = metadata
        job.updated_at = now
        db.commit()


def _handle_task_completed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    _store_task_output(task_id, payload.get("outputs", {}))
    _store_task_result(task_id, payload)
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.completed.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_failed(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    _store_task_result(task_id, payload)
    error = payload.get("error")
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        if isinstance(error, str) and error.startswith("tool_intent_mismatch"):
            replan_done = _replan_job_for_intent_mismatch(db, task.job_id)
            if replan_done:
                return
        task.status = models.TaskStatus.failed.value
        task.updated_at = now
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
        if job:
            _set_job_status(job, models.JobStatus.failed)
            job.updated_at = now
        db.commit()


def _replan_job_for_intent_mismatch(db: Session, job_id: str) -> bool:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        return False
    metadata = job.metadata_json or {}
    count = int(metadata.get("replan_count", 0))
    if count >= REPLAN_MAX:
        return False
    metadata["replan_count"] = count + 1
    job.metadata_json = metadata
    job.status = models.JobStatus.planning.value
    job.updated_at = datetime.utcnow()
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete(synchronize_session=False)
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete(synchronize_session=False)
    db.commit()
    _emit_event("job.created", _job_from_record(job).model_dump())
    return True


def _handle_task_accepted(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.accepted.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))
    _refresh_job_status(job_id)


def _handle_task_rework(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.rework_count = (task.rework_count or 0) + 1
        if _limit_exceeded(task.rework_count, task.max_reworks):
            task.status = models.TaskStatus.failed.value
            task.updated_at = now
            db.commit()
            _emit_event(
                "task.failed",
                _task_payload_with_error(
                    task, envelope.get("correlation_id"), "max_reworks_exceeded"
                ),
            )
            _refresh_job_status(task.job_id)
            return
        task.status = models.TaskStatus.pending.value
        task.updated_at = now
        db.commit()
        job_id = task.job_id
        plan_id = task.plan_id
    _enqueue_ready_tasks(job_id, plan_id, envelope.get("correlation_id"))


def _parse_plan_payload(payload: dict) -> models.PlanCreate | None:
    try:
        return models.PlanCreate.model_validate(payload)
    except Exception:
        return None


def _limit_exceeded(count: int, limit: int | None) -> bool:
    if not limit or limit <= 0:
        return False
    return count > limit


def _enqueue_ready_tasks(job_id: str, plan_id: str, correlation_id: str | None) -> None:
    now = datetime.utcnow()
    events: list[tuple[str, dict[str, Any]]] = []
    with SessionLocal() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == plan_id).all()
        if not task_records:
            return
        tasks = _resolve_task_deps(task_records)
        task_map = {task.id: task for task in tasks}
        id_to_name = {record.id: record.name for record in task_records}
        ready_ids = set(orchestrator.ready_tasks(tasks))
        if not ready_ids:
            return
        for record in task_records:
            if record.id in ready_ids and record.status == models.TaskStatus.pending.value:
                if POLICY_GATE_ENABLED:
                    record.status = models.TaskStatus.blocked.value
                    record.updated_at = now
                    context = _build_task_context(record.id, task_map, id_to_name)
                    payload = _task_payload_from_record(record, correlation_id, context)
                    events.append(("task.policy_check", payload))
                    continue
                next_attempt = (record.attempts or 0) + 1
                if _limit_exceeded(next_attempt, record.max_attempts):
                    record.status = models.TaskStatus.failed.value
                    record.updated_at = now
                    events.append(
                        (
                            "task.failed",
                            _task_payload_with_error(
                                record, correlation_id, "max_attempts_exceeded"
                            ),
                        )
                    )
                    continue
                record.attempts = next_attempt
                record.status = models.TaskStatus.ready.value
                record.updated_at = now
                context = _build_task_context(record.id, task_map, id_to_name)
                payload = _task_payload_from_record(record, correlation_id, context)
                events.append(("task.ready", payload))
        db.commit()
    for event_type, payload in events:
        _emit_event(event_type, payload)
    _refresh_job_status(job_id)


def _task_payload_from_record(
    record: TaskRecord, correlation_id: str | None, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "task_id": record.id,
        "id": record.id,
        "job_id": record.job_id,
        "plan_id": record.plan_id,
        "name": record.name,
        "description": record.description,
        "instruction": record.instruction,
        "acceptance_criteria": record.acceptance_criteria or [],
        "expected_output_schema_ref": record.expected_output_schema_ref,
        "status": record.status,
        "deps": record.deps or [],
        "attempts": record.attempts or 0,
        "max_attempts": record.max_attempts or 0,
        "rework_count": record.rework_count or 0,
        "max_reworks": record.max_reworks or 0,
        "assigned_to": record.assigned_to,
        "tool_requests": record.tool_requests or [],
        "tool_inputs": record.tool_inputs or {},
        "critic_required": bool(record.critic_required),
        "intent": record.intent,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }
    ctx = context or {}
    if context:
        payload["context"] = context
    resolved_inputs = payload_resolver.resolve_tool_inputs(
        payload["tool_requests"],
        payload["instruction"],
        ctx,
        payload,
        payload.get("tool_inputs", {}),
    )
    if resolved_inputs:
        payload["tool_inputs"] = resolved_inputs
        payload["tool_inputs_resolved"] = True
    return payload


def _task_payload_with_error(
    record: TaskRecord, correlation_id: str | None, error: str
) -> dict[str, Any]:
    payload = _task_payload_from_record(record, correlation_id)
    payload["error"] = error
    return payload


def _handle_policy_decision(envelope: dict) -> None:
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return
    task_id = envelope.get("task_id") or payload.get("task_id")
    if not task_id:
        return
    decision = payload.get("decision")
    reasons = payload.get("reasons") or []
    rewrites = payload.get("rewrites")
    correlation_id = envelope.get("correlation_id")
    now = datetime.utcnow()
    events_to_emit: list[tuple[str, dict[str, Any]]] = []
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        if task.status not in {
            models.TaskStatus.pending.value,
            models.TaskStatus.blocked.value,
        }:
            return
        if decision == "deny":
            task.status = models.TaskStatus.failed.value
            task.updated_at = now
            db.commit()
            reason_text = "policy_denied"
            if reasons:
                reason_text = f"policy_denied:{'; '.join(reasons)}"
            events_to_emit.append(
                ("task.failed", _task_payload_with_error(task, correlation_id, reason_text))
            )
        elif decision in {"allow", "rewrite"}:
            if isinstance(rewrites, dict):
                _apply_task_rewrites(task, rewrites)
            next_attempt = (task.attempts or 0) + 1
            if _limit_exceeded(next_attempt, task.max_attempts):
                task.status = models.TaskStatus.failed.value
                task.updated_at = now
                db.commit()
                events_to_emit.append(
                    (
                        "task.failed",
                        _task_payload_with_error(task, correlation_id, "max_attempts_exceeded"),
                    )
                )
            else:
                task.attempts = next_attempt
                task.status = models.TaskStatus.ready.value
                task.updated_at = now
                db.commit()
                task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == task.plan_id).all()
                tasks = _resolve_task_deps(task_records)
                task_map = {entry.id: entry for entry in tasks}
                id_to_name = {record.id: record.name for record in task_records}
                context = _build_task_context(task.id, task_map, id_to_name)
                payload = _task_payload_from_record(task, correlation_id, context)
                events_to_emit.append(("task.ready", payload))
        db.commit()
    for event_type, event_payload in events_to_emit:
        _emit_event(event_type, event_payload)
    job_id = envelope.get("job_id") or payload.get("job_id")
    if job_id:
        _refresh_job_status(job_id)


def _recover_jobs() -> None:
    with SessionLocal() as db:
        jobs = (
            db.query(JobRecord)
            .filter(
                JobRecord.status.in_(
                    [
                        models.JobStatus.queued.value,
                        models.JobStatus.planning.value,
                        models.JobStatus.running.value,
                    ]
                )
            )
            .all()
        )
        for job in jobs:
            plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
            if plan:
                _enqueue_ready_tasks(job.id, plan.id, None)
                continue
            _emit_event("job.created", _job_from_record(job).model_dump())


def _apply_task_rewrites(task: TaskRecord, rewrites: dict[str, Any]) -> None:
    if "instruction" in rewrites and isinstance(rewrites["instruction"], str):
        task.instruction = rewrites["instruction"]
    if "description" in rewrites and isinstance(rewrites["description"], str):
        task.description = rewrites["description"]
    if "acceptance_criteria" in rewrites and isinstance(rewrites["acceptance_criteria"], list):
        task.acceptance_criteria = rewrites["acceptance_criteria"]
    if "expected_output_schema_ref" in rewrites and isinstance(
        rewrites["expected_output_schema_ref"], str
    ):
        task.expected_output_schema_ref = rewrites["expected_output_schema_ref"]
    if "tool_requests" in rewrites and isinstance(rewrites["tool_requests"], list):
        task.tool_requests = rewrites["tool_requests"]
    if "tool_inputs" in rewrites and isinstance(rewrites["tool_inputs"], dict):
        task.tool_inputs = rewrites["tool_inputs"]


def _resolve_task_deps(task_records: list[TaskRecord]) -> list[models.Task]:
    name_to_id = {record.name: record.id for record in task_records}
    tasks: list[models.Task] = []
    for record in task_records:
        task = _task_from_record(record)
        resolved = []
        for dep in task.deps:
            if dep in name_to_id:
                resolved.append(name_to_id[dep])
            else:
                resolved.append(dep)
        tasks.append(task.model_copy(update={"deps": resolved}))
    return tasks


def _store_task_output(task_id: str, outputs: dict[str, Any]) -> None:
    try:
        redis_client.set(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}", json.dumps(outputs))
    except Exception:
        return


def _load_task_output(task_id: str) -> dict[str, Any]:
    try:
        raw = redis_client.get(f"{TASK_OUTPUT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _store_task_result(task_id: str, result: dict[str, Any]) -> None:
    try:
        redis_client.set(f"{TASK_RESULT_KEY_PREFIX}{task_id}", json.dumps(result))
    except Exception:
        return


def _load_task_result(task_id: str) -> dict[str, Any]:
    try:
        raw = redis_client.get(f"{TASK_RESULT_KEY_PREFIX}{task_id}")
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _build_task_context(
    task_id: str, task_map: dict[str, models.Task], id_to_name: dict[str, str]
) -> dict[str, Any]:
    task = task_map.get(task_id)
    if not task:
        return {}
    deps = task.deps or []
    visited: set[str] = set()
    stack: list[str] = list(deps)
    while stack:
        dep_id = stack.pop()
        if dep_id in visited:
            continue
        visited.add(dep_id)
        dep_task = task_map.get(dep_id)
        if dep_task and dep_task.deps:
            for child in dep_task.deps:
                if child not in visited:
                    stack.append(child)
    outputs_by_id = {dep_id: _load_task_output(dep_id) for dep_id in visited}
    outputs_by_name = {id_to_name.get(dep_id, dep_id): output for dep_id, output in outputs_by_id.items()}
    return {"dependencies": outputs_by_id, "dependencies_by_name": outputs_by_name}


def _refresh_job_status(job_id: str) -> None:
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not job:
            return
        tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
        if not tasks:
            return
        statuses = {task.status for task in tasks}
        now = datetime.utcnow()
        if models.TaskStatus.failed.value in statuses:
            _set_job_status(job, models.JobStatus.failed)
        elif statuses.issubset(
            {models.TaskStatus.completed.value, models.TaskStatus.accepted.value}
        ):
            _set_job_status(job, models.JobStatus.succeeded)
        elif models.TaskStatus.ready.value in statuses or models.TaskStatus.running.value in statuses:
            _set_job_status(job, models.JobStatus.running)
        else:
            _set_job_status(job, models.JobStatus.planning)
        job.updated_at = now
        db.commit()


def _set_job_status(job: JobRecord, status: models.JobStatus) -> None:
    current = models.JobStatus(job.status)
    if state_machine.validate_job_transition(current, status):
        job.status = status.value


@app.post("/jobs", response_model=models.Job)
def create_job(job: models.JobCreate, db: Session = Depends(get_db)) -> models.Job:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    record = JobRecord(
        id=job_id,
        goal=job.goal,
        context_json=job.context_json,
        status=models.JobStatus.queued.value,
        created_at=now,
        updated_at=now,
        priority=job.priority,
        metadata_json={"idempotency_key": job.idempotency_key} if job.idempotency_key else {},
    )
    db.add(record)
    db.commit()
    jobs_created_total.inc()
    _emit_event("job.created", _job_from_record(record).model_dump())
    return _job_from_record(record)


@app.get("/jobs", response_model=List[models.Job])
def list_jobs(db: Session = Depends(get_db)) -> List[models.Job]:
    jobs = db.query(JobRecord).all()
    return [_job_from_record(job) for job in jobs]


@app.get("/jobs/{job_id}", response_model=models.Job)
def get_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_from_record(job)


@app.get("/jobs/{job_id}/plan", response_model=models.Plan)
def get_plan(job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return _plan_from_record(plan)


@app.get("/jobs/{job_id}/tasks", response_model=List[models.Task])
def get_tasks(job_id: str, db: Session = Depends(get_db)) -> List[models.Task]:
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    return [_task_from_record(task) for task in tasks]


@app.get("/jobs/{job_id}/task_results")
def get_task_results(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    return {task.id: _load_task_result(task.id) for task in tasks}


@app.get("/jobs/{job_id}/details", response_model=models.JobDetails)
def get_job_details(job_id: str, db: Session = Depends(get_db)) -> models.JobDetails:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    plan_record = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    return models.JobDetails(
        job_id=job_id,
        plan=_plan_from_record(plan_record) if plan_record else None,
        tasks=[_task_from_record(task) for task in tasks],
        task_results={task.id: _load_task_result(task.id) for task in tasks},
    )


@app.get("/tasks/{task_id}", response_model=models.Task)
def get_task(task_id: str, db: Session = Depends(get_db)) -> models.Task:
    task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_from_record(task)


@app.post("/jobs/{job_id}/cancel", response_model=models.Job)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not state_machine.validate_job_transition(models.JobStatus(job.status), models.JobStatus.canceled):
        raise HTTPException(status_code=400, detail="Invalid state transition")
    job.status = models.JobStatus.canceled.value
    job.updated_at = datetime.utcnow()
    db.commit()
    _emit_event("job.canceled", {"job_id": job_id, "correlation_id": str(uuid.uuid4())})
    return _job_from_record(job)


@app.post("/jobs/{job_id}/resume", response_model=models.Job)
def resume_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if models.JobStatus(job.status) != models.JobStatus.canceled:
        raise HTTPException(status_code=400, detail="Job is not canceled")
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = datetime.utcnow()
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry", response_model=models.Job)
def retry_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if models.JobStatus(job.status) not in {models.JobStatus.failed, models.JobStatus.canceled}:
        raise HTTPException(status_code=400, detail="Job is not retryable")
    now = datetime.utcnow()
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    for task in tasks:
        task.status = models.TaskStatus.pending.value
        task.attempts = 0
        task.rework_count = 0
        task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/retry_failed", response_model=models.Job)
def retry_failed_tasks(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    now = datetime.utcnow()
    failed_tasks = (
        db.query(TaskRecord)
        .filter(
            TaskRecord.job_id == job_id,
            TaskRecord.status == models.TaskStatus.failed.value,
        )
        .all()
    )
    if not failed_tasks:
        raise HTTPException(status_code=400, detail="No failed tasks to retry")
    for task in failed_tasks:
        task.status = models.TaskStatus.pending.value
        task.attempts = 0
        task.rework_count = 0
        task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/replan", response_model=models.Job)
def replan_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    metadata = job.metadata_json or {}
    metadata["replan_count"] = int(metadata.get("replan_count", 0)) + 1
    metadata["replan_reason"] = "manual"
    job.metadata_json = metadata
    job.status = models.JobStatus.planning.value
    job.updated_at = datetime.utcnow()
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete(synchronize_session=False)
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete(synchronize_session=False)
    db.commit()
    _emit_event("job.created", _job_from_record(job).model_dump())
    return _job_from_record(job)


@app.post("/jobs/{job_id}/clear")
def clear_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete()
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete()
    db.query(JobRecord).filter(JobRecord.id == job_id).delete()
    db.commit()
    return {"status": "cleared"}


@app.get("/events/stream")
def stream_events(request: Request, once: bool = False):
    def event_generator():
        if once:
            yield "data: {}\n\n"
            return
        last_ids = {
            events.JOB_STREAM: "0-0",
            events.PLAN_STREAM: "0-0",
            events.TASK_STREAM: "0-0",
            events.CRITIC_STREAM: "0-0",
            events.POLICY_STREAM: "0-0",
        }
        while True:
            if request.client is None:
                break
            results = redis_client.xread(last_ids, block=1000, count=10)
            for stream_name, messages in results:
                for message_id, data in messages:
                    last_ids[stream_name] = message_id
                    payload = data.get("data")
                    yield f"data: {payload}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/plans", response_model=models.Plan)
def create_plan(plan: models.PlanCreate, job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    plan_id = str(uuid.uuid4())
    now = datetime.utcnow()
    record = PlanRecord(
        id=plan_id,
        job_id=job_id,
        planner_version=plan.planner_version,
        created_at=now,
        tasks_summary=plan.tasks_summary,
        dag_edges=plan.dag_edges,
        policy_decision={},
    )
    db.add(record)
    for task in plan.tasks:
        task_id = str(uuid.uuid4())
        task_record = TaskRecord(
            id=task_id,
            job_id=job_id,
            plan_id=plan_id,
            name=task.name,
            description=task.description,
            instruction=task.instruction,
            acceptance_criteria=task.acceptance_criteria,
            expected_output_schema_ref=task.expected_output_schema_ref,
            status=models.TaskStatus.pending.value,
            deps=task.deps,
            attempts=0,
            max_attempts=3,
            rework_count=0,
            max_reworks=2,
            assigned_to=None,
            tool_requests=task.tool_requests,
            tool_inputs=task.tool_inputs,
            created_at=now,
            updated_at=now,
            critic_required=1 if task.critic_required else 0,
        )
        db.add(task_record)
    db.commit()
    payload = _plan_created_payload(plan, job_id)
    payload["correlation_id"] = str(uuid.uuid4())
    _emit_event("plan.created", payload)
    return _plan_from_record(record)
