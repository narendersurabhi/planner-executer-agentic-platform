from __future__ import annotations

import json
import os
import logging
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping

import redis
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import Counter, make_asgi_app
from sqlalchemy.orm import Session

from libs.core import (
    document_store,
    events,
    logging as core_logging,
    models,
    orchestrator,
    payload_resolver,
    state_machine,
    tool_registry,
)
from libs.core.llm_provider import MockLLMProvider
from .database import Base, SessionLocal, engine
from .models import JobRecord, PlanRecord, TaskRecord
from . import memory_store

core_logging.configure_logging("api")
logger = logging.getLogger("api.orchestrator")

app = FastAPI(title="Agentic Planner Executor API")

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3002,http://localhost:3000").split(
        ","
    )
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
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/shared/artifacts")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "/shared/workspace")
ORCHESTRATOR_ENABLED = os.getenv("ORCHESTRATOR_ENABLED", "true").lower() == "true"
POLICY_GATE_ENABLED = os.getenv("POLICY_GATE_ENABLED", "false").lower() == "true"
JOB_RECOVERY_ENABLED = os.getenv("JOB_RECOVERY_ENABLED", "true").lower() == "true"
DEV_RESUME_RENDER_ENABLED = os.getenv("DEV_RESUME_RENDER_ENABLED", "false").lower() == "true"
LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "").strip()
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "").strip()
ORCHESTRATOR_RECOVER_PENDING = os.getenv("ORCHESTRATOR_RECOVER_PENDING", "true").lower() == "true"
ORCHESTRATOR_RECOVER_IDLE_MS = int(os.getenv("ORCHESTRATOR_RECOVER_IDLE_MS", "60000"))
REPLAN_MAX = int(os.getenv("REPLAN_MAX", "1"))
TOOL_INPUT_VALIDATION_ENABLED = os.getenv("TOOL_INPUT_VALIDATION_ENABLED", "true").lower() == "true"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
TASK_OUTPUT_KEY_PREFIX = "task_output:"
TASK_RESULT_KEY_PREFIX = "task_result:"

_tool_spec_registry = tool_registry.default_registry(
    http_fetch_enabled=False,
    llm_enabled=True,
    llm_provider=MockLLMProvider(),
)
TOOL_INPUT_SCHEMAS = {spec.name: spec.input_schema for spec in _tool_spec_registry.list_specs()}


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
    try:
        redis_client.xadd(stream, {"data": envelope.model_dump_json()})
    except redis.RedisError:
        logger.warning(
            "event_emit_failed",
            extra={"event_type": event_type, "stream": stream},
        )


def _resolve_download_path(path: str, root_dir: str, label: str) -> str:
    candidate = path.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="path is required")
    if candidate.startswith("/"):
        raise HTTPException(status_code=400, detail="path must be relative")
    root = Path(root_dir).resolve()
    target = (root / candidate).resolve()
    if not str(target).startswith(str(root)):
        raise HTTPException(status_code=400, detail=f"Invalid {label} path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"{label.title()} file not found")
    return str(target)


def _resolve_artifact_path(path: str) -> str:
    return _resolve_download_path(path, ARTIFACTS_DIR, "artifact")


def _resolve_workspace_path(path: str) -> str:
    return _resolve_download_path(path, WORKSPACE_DIR, "workspace")


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
    stream_keys = [
        events.PLAN_STREAM,
        events.TASK_STREAM,
        events.CRITIC_STREAM,
        events.POLICY_STREAM,
    ]
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
    elif event_type == "task.started":
        _handle_task_started(envelope)
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
                intent=task.intent.value
                if isinstance(task.intent, models.ToolIntent)
                else task.intent,
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


def _handle_task_started(envelope: dict) -> None:
    payload = envelope.get("payload") or {}
    task_id = payload.get("task_id") or envelope.get("task_id")
    if not task_id:
        return
    now = datetime.utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        if not task:
            return
        task.status = models.TaskStatus.running.value
        worker_consumer = payload.get("worker_consumer")
        if isinstance(worker_consumer, str) and worker_consumer.strip():
            task.assigned_to = worker_consumer.strip()
        attempts = payload.get("attempts")
        if isinstance(attempts, int):
            task.attempts = max(0, attempts)
        task.updated_at = now
        job = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
        if job:
            _set_job_status(job, models.JobStatus.running)
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
        job_record = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        job_context = (
            job_record.context_json
            if job_record and isinstance(job_record.context_json, dict)
            else {}
        )
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
                    context = _build_task_context(record.id, task_map, id_to_name, job_context)
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
                context = _build_task_context(record.id, task_map, id_to_name, job_context)
                payload = _task_payload_from_record(record, correlation_id, context)
                if TOOL_INPUT_VALIDATION_ENABLED and payload.get("tool_inputs_validation"):
                    record.status = models.TaskStatus.failed.value
                    record.updated_at = now
                    events.append(
                        (
                            "task.failed",
                            _task_payload_with_error(record, correlation_id, "tool_inputs_invalid"),
                        )
                    )
                    continue
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
        validation_errors = payload_resolver.validate_tool_inputs(
            resolved_inputs, TOOL_INPUT_SCHEMAS
        )
        if validation_errors:
            payload["tool_inputs_validation"] = validation_errors
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
                job_record = db.query(JobRecord).filter(JobRecord.id == task.job_id).first()
                job_context = (
                    job_record.context_json
                    if job_record and isinstance(job_record.context_json, dict)
                    else {}
                )
                context = _build_task_context(task.id, task_map, id_to_name, job_context)
                payload = _task_payload_from_record(task, correlation_id, context)
                if TOOL_INPUT_VALIDATION_ENABLED and payload.get("tool_inputs_validation"):
                    task.status = models.TaskStatus.failed.value
                    task.updated_at = now
                    db.commit()
                    events_to_emit.append(
                        (
                            "task.failed",
                            _task_payload_with_error(task, correlation_id, "tool_inputs_invalid"),
                        )
                    )
                else:
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
    task_id: str,
    task_map: dict[str, models.Task],
    id_to_name: dict[str, str],
    job_context: dict[str, Any] | None = None,
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
    outputs_by_name = {
        id_to_name.get(dep_id, dep_id): output for dep_id, output in outputs_by_id.items()
    }
    context: dict[str, Any] = {
        "dependencies": outputs_by_id,
        "dependencies_by_name": outputs_by_name,
    }
    if isinstance(job_context, dict) and job_context:
        context["job_context"] = job_context
    return context


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
        elif (
            models.TaskStatus.ready.value in statuses or models.TaskStatus.running.value in statuses
        ):
            _set_job_status(job, models.JobStatus.running)
        else:
            _set_job_status(job, models.JobStatus.planning)
        job.updated_at = now
        db.commit()


def _set_job_status(job: JobRecord, status: models.JobStatus) -> None:
    current = models.JobStatus(job.status)
    if state_machine.validate_job_transition(current, status):
        job.status = status.value


def _read_resume_doc_spec_from_memory(db: Session, job_id: str) -> dict[str, Any] | None:
    try:
        entries = memory_store.read_memory(
            db,
            models.MemoryQuery(
                name="task_outputs",
                job_id=job_id,
                key="resume_doc_spec:latest",
                limit=1,
            ),
        )
    except Exception:
        entries = []
    for entry in entries:
        if not isinstance(entry, models.MemoryEntry):
            continue
        payload = entry.payload or {}
        if isinstance(payload, dict) and payload:
            spec = payload.get("resume_doc_spec") if "resume_doc_spec" in payload else payload
            if isinstance(spec, dict):
                return spec
    try:
        fallback = memory_store.read_memory(
            db,
            models.MemoryQuery(name="task_outputs", job_id=job_id, limit=50),
        )
    except Exception:
        fallback = []
    for entry in fallback:
        if not isinstance(entry, models.MemoryEntry):
            continue
        payload = entry.payload or {}
        if isinstance(payload, dict) and "resume_doc_spec" in payload:
            spec = payload.get("resume_doc_spec")
            if isinstance(spec, dict):
                return spec
    return None


def _parse_task_dlq_entry(stream_id: str, record: Mapping[str, str]) -> models.TaskDlqEntry | None:
    raw = record.get("data")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    envelope = payload.get("envelope")
    envelope_dict = envelope if isinstance(envelope, dict) else {}
    task_payload = payload.get("task_payload")
    task_payload_dict = task_payload if isinstance(task_payload, dict) else {}
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        maybe_job = envelope_dict.get("job_id")
        job_id = maybe_job if isinstance(maybe_job, str) else None
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        maybe_task = task_payload_dict.get("task_id")
        task_id = maybe_task if isinstance(maybe_task, str) else None
    error = payload.get("error")
    return models.TaskDlqEntry(
        stream_id=stream_id,
        message_id=str(payload.get("message_id") or stream_id),
        failed_at=payload.get("failed_at") if isinstance(payload.get("failed_at"), str) else None,
        error=error if isinstance(error, str) and error else "unknown_error",
        worker_consumer=payload.get("worker_consumer")
        if isinstance(payload.get("worker_consumer"), str)
        else None,
        job_id=job_id,
        task_id=task_id,
        envelope=envelope_dict,
        task_payload=task_payload_dict,
    )


def _read_task_dlq(job_id: str, limit: int) -> list[models.TaskDlqEntry]:
    scan_count = min(max(limit * 5, limit), 500)
    rows = redis_client.xrevrange(events.TASK_DLQ_STREAM, "+", "-", count=scan_count)
    entries: list[models.TaskDlqEntry] = []
    for stream_id, record in rows:
        parsed = _parse_task_dlq_entry(stream_id, record)
        if parsed is None:
            continue
        if parsed.job_id != job_id:
            continue
        entries.append(parsed)
        if len(entries) >= limit:
            break
    return entries


@app.post("/jobs", response_model=models.Job)
def create_job(job: models.JobCreate, db: Session = Depends(get_db)) -> models.Job:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    metadata: Dict[str, Any] = {}
    if job.idempotency_key:
        metadata["idempotency_key"] = job.idempotency_key
    if LLM_PROVIDER_NAME:
        metadata["llm_provider"] = LLM_PROVIDER_NAME
    if LLM_MODEL_NAME:
        metadata["llm_model"] = LLM_MODEL_NAME
    record = JobRecord(
        id=job_id,
        goal=job.goal,
        context_json=job.context_json,
        status=models.JobStatus.queued.value,
        created_at=now,
        updated_at=now,
        priority=job.priority,
        metadata_json=metadata,
    )
    db.add(record)
    db.commit()
    jobs_created_total.inc()
    if isinstance(job.context_json, dict) and job.context_json:
        try:
            memory_store.write_memory(
                db,
                models.MemoryWrite(
                    name="job_context",
                    job_id=job_id,
                    payload=job.context_json,
                    metadata={"source": "job_create"},
                ),
            )
        except (KeyError, ValueError):
            pass
        _seed_task_output_memory(db, job_id, job.context_json)
    _emit_event("job.created", _job_from_record(record).model_dump())
    return _job_from_record(record)


def _seed_task_output_memory(db: Session, job_id: str, context_json: Dict[str, Any]) -> None:
    seed_map = {
        "tailored_text": "tailored_text:latest",
        "tailored_resume": "tailored_resume:latest",
        "resume_doc_spec": "resume_doc_spec:latest",
        "document_spec": "document_spec:latest",
        "path": "docx_path:latest",
    }
    for field, key in seed_map.items():
        value = context_json.get(field)
        if isinstance(value, str):
            if not value.strip():
                continue
        elif isinstance(value, dict):
            if not value:
                continue
        elif value is None:
            continue
        try:
            memory_store.write_memory(
                db,
                models.MemoryWrite(
                    name="task_outputs",
                    job_id=job_id,
                    key=key,
                    payload={field: value},
                    metadata={"source": "job_create"},
                ),
            )
        except (KeyError, ValueError):
            continue


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
    if not state_machine.validate_job_transition(
        models.JobStatus(job.status), models.JobStatus.canceled
    ):
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


@app.post("/jobs/{job_id}/tasks/{task_id}/retry", response_model=models.Job)
def retry_task(
    job_id: str,
    task_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    task = (
        db.query(TaskRecord).filter(TaskRecord.id == task_id, TaskRecord.job_id == job_id).first()
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != models.TaskStatus.failed.value:
        raise HTTPException(status_code=400, detail="Task is not failed")
    now = datetime.utcnow()
    task.status = models.TaskStatus.pending.value
    task.attempts = 0
    task.rework_count = 0
    task.updated_at = now
    _set_job_status(job, models.JobStatus.planning)
    job.updated_at = now
    db.commit()
    stream_id = payload.get("stream_id")
    if isinstance(stream_id, str) and stream_id:
        try:
            redis_client.xdel(events.TASK_DLQ_STREAM, stream_id)
        except redis.RedisError:
            logger.warning(
                "task_dlq_delete_failed",
                extra={"job_id": job_id, "task_id": task_id, "stream_id": stream_id},
            )
    plan = db.query(PlanRecord).filter(PlanRecord.id == task.plan_id).first()
    if not plan:
        plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if plan:
        _enqueue_ready_tasks(job_id, plan.id, None)
    return _job_from_record(job)


@app.post("/jobs/{job_id}/dev_resume_render", response_model=models.Job)
def dev_resume_render_job(
    job_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> models.Job:
    if not DEV_RESUME_RENDER_ENABLED:
        raise HTTPException(status_code=404, detail="Dev endpoint disabled")
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        output_dir = None
        if isinstance(job.context_json, dict):
            output_dir = job.context_json.get("output_dir")
        safe_dir = output_dir if isinstance(output_dir, str) and output_dir.strip() else "resumes"
        path = f"{safe_dir}/dev_render_{job_id}.docx"
    if not isinstance(path, str) or not path.endswith(".docx"):
        raise HTTPException(status_code=400, detail="path must be a .docx filename")

    resume_doc_spec = _read_resume_doc_spec_from_memory(db, job_id)
    if not isinstance(resume_doc_spec, dict) or not resume_doc_spec:
        raise HTTPException(
            status_code=400,
            detail="resume_doc_spec not found in memory for this job",
        )

    now = datetime.utcnow()
    db.query(TaskRecord).filter(TaskRecord.job_id == job_id).delete(synchronize_session=False)
    db.query(PlanRecord).filter(PlanRecord.job_id == job_id).delete(synchronize_session=False)

    plan_id = str(uuid.uuid4())
    tasks_summary = "Dev render: ResumeDocSpec -> DocumentSpec -> DOCX"
    dag_edges = [["ConvertResumeDocSpec", "RenderResumeDocx"]]
    plan_record = PlanRecord(
        id=plan_id,
        job_id=job_id,
        planner_version="dev_resume_render_v1",
        created_at=now,
        tasks_summary=tasks_summary,
        dag_edges=dag_edges,
        policy_decision={},
    )
    db.add(plan_record)

    tasks: list[models.TaskCreate] = []
    tasks.append(
        models.TaskCreate(
            name="ConvertResumeDocSpec",
            description="Convert ResumeDocSpec to DocumentSpec",
            instruction="Convert resume_doc_spec from memory into DocumentSpec JSON.",
            acceptance_criteria=["document_spec converted"],
            expected_output_schema_ref="schemas/DocumentSpec",
            intent=models.ToolIntent.transform,
            deps=[],
            tool_requests=["resume_doc_spec_to_document_spec"],
            tool_inputs={"resume_doc_spec_to_document_spec": {"resume_doc_spec": resume_doc_spec}},
            critic_required=False,
        )
    )
    tasks.append(
        models.TaskCreate(
            name="RenderResumeDocx",
            description="Render DOCX from DocumentSpec",
            instruction=f"Render DOCX with docx_generate_from_spec to {path}.",
            acceptance_criteria=["docx artifact written"],
            expected_output_schema_ref="artifacts/docx",
            intent=models.ToolIntent.render,
            deps=["ConvertResumeDocSpec"],
            tool_requests=["docx_generate_from_spec"],
            tool_inputs={"docx_generate_from_spec": {"path": path, "strict": True}},
            critic_required=False,
        )
    )

    for task in tasks:
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
            intent=task.intent.value if task.intent else None,
            tool_requests=task.tool_requests,
            tool_inputs=task.tool_inputs,
            created_at=now,
            updated_at=now,
            critic_required=1 if task.critic_required else 0,
        )
        db.add(task_record)

    job.status = models.JobStatus.planning.value
    job.updated_at = now
    db.commit()

    plan_payload = models.PlanCreate(
        planner_version="dev_resume_render_v1",
        tasks_summary=tasks_summary,
        dag_edges=dag_edges,
        tasks=tasks,
    )
    payload_out = _plan_created_payload(plan_payload, job_id)
    payload_out["correlation_id"] = str(uuid.uuid4())
    _emit_event("plan.created", payload_out)
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


@app.get("/artifacts/download")
def download_artifact(path: str = Query(..., description="Path relative to /shared/artifacts")):
    filename = Path(path).name
    try:
        resolved = _resolve_artifact_path(path)
        filename = Path(resolved).name
        return FileResponse(resolved, filename=filename, media_type="application/octet-stream")
    except HTTPException as exc:
        if exc.status_code != 404 or not document_store.is_s3_enabled():
            raise
    try:
        payload = document_store.download_artifact_bytes(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404, detail=f"Artifact not found in object store: {exc}"
        ) from exc
    return StreamingResponse(
        iter([payload]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/workspace/download")
def download_workspace_file(
    path: str = Query(..., description="Path relative to /shared/workspace"),
):
    resolved = _resolve_workspace_path(path)
    filename = Path(resolved).name
    return FileResponse(resolved, filename=filename, media_type="application/octet-stream")


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


@app.post("/memory/write", response_model=models.MemoryEntry)
def write_memory(entry: models.MemoryWrite, db: Session = Depends(get_db)) -> models.MemoryEntry:
    try:
        return memory_store.write_memory(db, entry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        detail = str(exc)
        if "memory_conflict" in detail:
            raise HTTPException(status_code=409, detail=detail) from exc
        raise HTTPException(status_code=400, detail=detail) from exc


@app.get("/memory/read", response_model=List[models.MemoryEntry])
def read_memory(
    name: str = Query(...),
    scope: models.MemoryScope | None = Query(None),
    key: str | None = Query(None),
    job_id: str | None = Query(None),
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    include_expired: bool = Query(False),
    db: Session = Depends(get_db),
) -> List[models.MemoryEntry]:
    query = models.MemoryQuery(
        name=name,
        scope=scope,
        key=key,
        job_id=job_id,
        user_id=user_id,
        project_id=project_id,
        limit=limit,
        include_expired=include_expired,
    )
    try:
        return memory_store.read_memory(db, query)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/tasks/dlq", response_model=List[models.TaskDlqEntry])
def read_task_dlq(job_id: str, limit: int = Query(25, ge=1, le=200)) -> List[models.TaskDlqEntry]:
    try:
        return _read_task_dlq(job_id, limit)
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail=f"redis_error:{exc}") from exc


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
