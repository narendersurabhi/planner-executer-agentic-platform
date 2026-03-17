from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping

import redis
from sqlalchemy.orm import Session

from libs.core import (
    capability_registry,
    execution_contracts,
    intent_contract,
    models,
    orchestrator,
    payload_resolver,
)
from .models import EventOutboxRecord, JobRecord, TaskRecord


@dataclass(frozen=True)
class ApiDispatchConfig:
    event_outbox_enabled: bool
    event_outbox_batch_size: int
    event_outbox_poll_s: float
    event_outbox_redis_retries: int
    event_outbox_redis_retry_sleep_s: float
    policy_gate_enabled: bool
    tool_input_validation_enabled: bool
    tool_input_schemas: dict[str, Any]


@dataclass(frozen=True)
class ApiDispatchRuntime:
    redis_client: Any
    session_factory: Callable[[], Session]
    logger: Any
    config: ApiDispatchConfig


@dataclass(frozen=True)
class ApiDispatchCallbacks:
    stream_for_event: Callable[[str], str]
    resolve_task_deps: Callable[[list[TaskRecord]], list[models.Task]]
    build_task_context: Callable[[str, Mapping[str, Any], Mapping[str, str], dict[str, Any]], dict[str, Any]]
    coerce_task_intent_profiles: Callable[[Mapping[str, Any]], dict[str, dict[str, Any]]]
    normalize_task_intent_profile_segment: Callable[[Any], Mapping[str, Any] | None]
    refresh_job_status: Callable[[str], None]
    emit_event: Callable[[str, dict[str, Any]], None]


def _enabled_capabilities() -> Mapping[str, Any]:
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return {}
    return registry.enabled_capabilities()


def publish_envelope_to_redis(
    runtime: ApiDispatchRuntime,
    stream: str,
    envelope_json: str,
) -> tuple[bool, str | None]:
    retries = max(1, runtime.config.event_outbox_redis_retries)
    sleep_s = max(0.0, runtime.config.event_outbox_redis_retry_sleep_s)
    for attempt in range(1, retries + 1):
        try:
            runtime.redis_client.xadd(stream, {"data": envelope_json})
            return True, None
        except redis.RedisError as exc:
            if attempt >= retries:
                return False, str(exc)
            if sleep_s > 0:
                time.sleep(sleep_s)
    return False, "unknown_redis_publish_error"


def insert_outbox_event(
    runtime: ApiDispatchRuntime,
    stream: str,
    event_type: str,
    envelope_json: str,
) -> str | None:
    outbox_id = str(uuid.uuid4())
    now = datetime.now(UTC)
    try:
        with runtime.session_factory() as db:
            db.add(
                EventOutboxRecord(
                    id=outbox_id,
                    stream=stream,
                    event_type=event_type,
                    envelope_json=json.loads(envelope_json),
                    attempts=0,
                    last_error=None,
                    created_at=now,
                    updated_at=now,
                    published_at=None,
                )
            )
            db.commit()
        return outbox_id
    except Exception:
        runtime.logger.exception(
            "event_outbox_insert_failed",
            extra={"event_type": event_type, "stream": stream},
        )
        return None


def update_outbox_publish_state(
    runtime: ApiDispatchRuntime,
    outbox_id: str | None,
    published: bool,
    error: str | None = None,
) -> None:
    if not outbox_id:
        return
    now = datetime.now(UTC)
    try:
        with runtime.session_factory() as db:
            row = db.query(EventOutboxRecord).filter(EventOutboxRecord.id == outbox_id).first()
            if not row:
                return
            row.attempts = (row.attempts or 0) + 1
            row.updated_at = now
            row.last_error = None if published else (error or "redis_publish_failed")
            if published:
                row.published_at = now
            db.commit()
    except Exception:
        runtime.logger.exception(
            "event_outbox_update_failed",
            extra={"outbox_id": outbox_id, "published": published},
        )


def dispatch_event_outbox_once(runtime: ApiDispatchRuntime) -> int:
    if not runtime.config.event_outbox_enabled:
        return 0
    dispatched = 0
    try:
        with runtime.session_factory() as db:
            pending = (
                db.query(EventOutboxRecord)
                .filter(EventOutboxRecord.published_at.is_(None))
                .order_by(EventOutboxRecord.created_at.asc())
                .limit(max(1, runtime.config.event_outbox_batch_size))
                .all()
            )
            if not pending:
                return 0
            for row in pending:
                envelope_json = json.dumps(row.envelope_json)
                published, error = publish_envelope_to_redis(runtime, row.stream, envelope_json)
                row.attempts = (row.attempts or 0) + 1
                row.updated_at = datetime.now(UTC)
                if published:
                    row.published_at = row.updated_at
                    row.last_error = None
                    dispatched += 1
                else:
                    row.last_error = error or "redis_publish_failed"
            db.commit()
    except Exception:
        runtime.logger.exception("event_outbox_dispatch_failed")
    return dispatched


def start_event_outbox_dispatcher(runtime: ApiDispatchRuntime) -> None:
    def _loop() -> None:
        while True:
            dispatch_event_outbox_once(runtime)
            time.sleep(max(0.1, runtime.config.event_outbox_poll_s))

    thread = threading.Thread(target=_loop, daemon=True, name="event-outbox-dispatcher")
    thread.start()


def emit_event(
    runtime: ApiDispatchRuntime,
    callbacks: ApiDispatchCallbacks,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    envelope = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.now(UTC),
        correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
        job_id=payload.get("job_id") or payload.get("id"),
        task_id=payload.get("task_id"),
        payload=payload,
    )
    stream = callbacks.stream_for_event(event_type)
    envelope_json = envelope.model_dump_json()
    if not runtime.config.event_outbox_enabled:
        published, error = publish_envelope_to_redis(runtime, stream, envelope_json)
        if not published:
            runtime.logger.warning(
                "event_emit_failed",
                extra={
                    "event_type": event_type,
                    "stream": stream,
                    "error": error or "redis_publish_failed",
                },
            )
        return
    outbox_id = insert_outbox_event(runtime, stream, event_type, envelope_json)
    published, error = publish_envelope_to_redis(runtime, stream, envelope_json)
    if outbox_id is None and not published:
        outbox_id = insert_outbox_event(runtime, stream, event_type, envelope_json)
    update_outbox_publish_state(runtime, outbox_id, published, error)
    if not published:
        runtime.logger.warning(
            "event_emit_deferred_to_outbox",
            extra={
                "event_type": event_type,
                "stream": stream,
                "outbox_id": outbox_id,
                "error": error or "redis_publish_failed",
            },
        )


def task_payload_from_record(
    record: TaskRecord,
    correlation_id: str | None,
    *,
    context: dict[str, Any] | None = None,
    goal_text: str = "",
    intent_profile: Mapping[str, Any] | None = None,
    config: ApiDispatchConfig,
    callbacks: ApiDispatchCallbacks,
) -> dict[str, Any]:
    raw_tool_inputs = record.tool_inputs if isinstance(record.tool_inputs, dict) else {}
    capability_bindings = execution_contracts.normalize_capability_bindings(
        {"tool_inputs": raw_tool_inputs},
        request_ids=record.tool_requests or [],
        capabilities=_enabled_capabilities(),
    )
    execution_gates = execution_contracts.normalize_execution_gates(
        {"tool_inputs": raw_tool_inputs},
        request_ids=record.tool_requests or [],
    )
    payload: dict[str, Any] = {
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
        "tool_inputs": execution_contracts.strip_execution_metadata_from_tool_inputs(
            raw_tool_inputs
        ),
        "capability_bindings": capability_bindings,
        "execution_gates": execution_gates,
        "critic_required": bool(record.critic_required),
        "intent": record.intent,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "correlation_id": correlation_id or str(uuid.uuid4()),
    }
    normalized_profile_segment = callbacks.normalize_task_intent_profile_segment(
        intent_profile.get("segment") if isinstance(intent_profile, Mapping) else None
    )
    if normalized_profile_segment is not None:
        payload["intent_segment"] = normalized_profile_segment
    profile_intent = (
        intent_contract.normalize_task_intent(intent_profile.get("intent"))
        if isinstance(intent_profile, Mapping)
        else None
    )
    profile_source = (
        str(intent_profile.get("source") or "").strip()
        if isinstance(intent_profile, Mapping)
        else ""
    )
    profile_confidence = None
    if isinstance(intent_profile, Mapping):
        raw_confidence = intent_profile.get("confidence")
        if isinstance(raw_confidence, (int, float)):
            profile_confidence = max(0.0, min(1.0, float(raw_confidence)))
    if profile_intent and not payload.get("intent"):
        payload["intent"] = profile_intent
    if profile_intent and profile_source and profile_confidence is not None:
        payload["intent_source"] = profile_source
        payload["intent_confidence"] = round(profile_confidence, 3)
    else:
        inference_payload = dict(payload)
        if goal_text:
            inference_payload["goal"] = goal_text
        intent_inference = intent_contract.infer_task_intent_for_payload_with_metadata(
            inference_payload
        )
        payload["intent_source"] = intent_inference.source
        payload["intent_confidence"] = round(float(intent_inference.confidence), 3)
        if not payload.get("intent"):
            payload["intent"] = intent_inference.intent
    ctx = context or {}
    if context:
        payload["context"] = context
    resolved_inputs, resolution_errors = payload_resolver.resolve_tool_inputs_with_errors(
        payload["tool_requests"],
        payload["instruction"],
        ctx,
        payload,
        payload.get("tool_inputs", {}),
    )
    validation_errors: dict[str, str] = {}
    if resolved_inputs:
        payload["tool_inputs"] = resolved_inputs
        payload["tool_inputs_resolved"] = True
        validation_errors.update(
            payload_resolver.validate_tool_inputs(resolved_inputs, config.tool_input_schemas)
        )
    if resolution_errors:
        validation_errors.update(resolution_errors)
    if validation_errors:
        payload["tool_inputs_validation"] = validation_errors
    return execution_contracts.dump_task_dispatch_payload(payload)


def task_payload_with_error(
    record: TaskRecord,
    correlation_id: str | None,
    error: str,
    *,
    config: ApiDispatchConfig,
    callbacks: ApiDispatchCallbacks,
) -> dict[str, Any]:
    payload = task_payload_from_record(
        record,
        correlation_id,
        config=config,
        callbacks=callbacks,
    )
    payload["error"] = error
    return payload


def enqueue_ready_tasks(
    job_id: str,
    plan_id: str,
    correlation_id: str | None,
    *,
    runtime: ApiDispatchRuntime,
    callbacks: ApiDispatchCallbacks,
) -> None:
    now = datetime.now(UTC)
    events_to_emit: list[tuple[str, dict[str, Any]]] = []
    with runtime.session_factory() as db:
        task_records = db.query(TaskRecord).filter(TaskRecord.plan_id == plan_id).all()
        if not task_records:
            return
        job_record = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        job_context = (
            job_record.context_json
            if job_record and isinstance(job_record.context_json, dict)
            else {}
        )
        job_goal = job_record.goal if job_record and isinstance(job_record.goal, str) else ""
        task_intent_profiles = callbacks.coerce_task_intent_profiles(
            job_record.metadata_json if job_record and isinstance(job_record.metadata_json, dict) else {}
        )
        tasks = callbacks.resolve_task_deps(task_records)
        task_map = {task.id: task for task in tasks}
        id_to_name = {record.id: record.name for record in task_records}
        ready_ids = set(orchestrator.ready_tasks(tasks))
        if not ready_ids:
            return
        for record in task_records:
            if record.id in ready_ids and record.status == models.TaskStatus.pending.value:
                if record.attempts is None:
                    record.attempts = 0
                next_attempt = (record.attempts or 0) + 1
                if _limit_exceeded(next_attempt, record.max_attempts):
                    record.status = models.TaskStatus.failed.value
                    record.updated_at = now
                    events_to_emit.append(
                        (
                            "task.failed",
                            task_payload_with_error(
                                record,
                                correlation_id,
                                "max_attempts_exceeded",
                                config=runtime.config,
                                callbacks=callbacks,
                            ),
                        )
                    )
                    continue
                if runtime.config.policy_gate_enabled:
                    record.status = models.TaskStatus.blocked.value
                    record.updated_at = now
                    context = callbacks.build_task_context(record.id, task_map, id_to_name, job_context)
                    payload = task_payload_from_record(
                        record,
                        correlation_id,
                        context=context,
                        goal_text=job_goal,
                        intent_profile=task_intent_profiles.get(record.id),
                        config=runtime.config,
                        callbacks=callbacks,
                    )
                    events_to_emit.append(("task.policy_check", payload))
                    continue
                record.attempts = next_attempt
                record.status = models.TaskStatus.ready.value
                record.updated_at = now
                context = callbacks.build_task_context(record.id, task_map, id_to_name, job_context)
                payload = task_payload_from_record(
                    record,
                    correlation_id,
                    context=context,
                    goal_text=job_goal,
                    intent_profile=task_intent_profiles.get(record.id),
                    config=runtime.config,
                    callbacks=callbacks,
                )
                if runtime.config.tool_input_validation_enabled and payload.get("tool_inputs_validation"):
                    record.status = models.TaskStatus.failed.value
                    record.updated_at = now
                    failed_payload = dict(payload)
                    failed_payload["error"] = "tool_inputs_invalid"
                    events_to_emit.append(("task.failed", failed_payload))
                    continue
                events_to_emit.append(("task.ready", payload))
        db.commit()
    for event_type, event_payload in events_to_emit:
        callbacks.emit_event(event_type, event_payload)
    callbacks.refresh_job_status(job_id)
def _limit_exceeded(count: int, limit: int | None) -> bool:
    if not limit or limit <= 0:
        return False
    return count > limit
