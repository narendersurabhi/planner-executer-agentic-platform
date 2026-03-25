from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import uuid
from typing import Any

from sqlalchemy.orm import Session

from libs.core import models, workflow_contracts

from .models import (
    ChatMessageRecord,
    FeedbackRecord,
    JobRecord,
    PlanRecord,
    TaskRecord,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _non_empty_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in value:
        item = _non_empty_string(entry)
        if item is None or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _normalize_metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _candidate_capabilities_from_graph(
    graph: workflow_contracts.IntentGraph | None,
) -> dict[str, list[str]]:
    if graph is None:
        return {}
    candidates: dict[str, list[str]] = {}
    for segment in graph.segments:
        deduped: list[str] = []
        for capability_id in segment.suggested_capabilities:
            normalized = _non_empty_string(capability_id)
            if normalized is None or normalized in deduped:
                continue
            deduped.append(normalized)
        if deduped:
            candidates[segment.id] = deduped
    return candidates


def _normalized_intent_snapshot(
    metadata: Mapping[str, Any] | None,
    *,
    goal: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    payload = dict(metadata) if isinstance(metadata, Mapping) else {}
    envelope = workflow_contracts.parse_normalized_intent_envelope(
        payload.get("normalized_intent_envelope")
    )
    if envelope is None:
        profile = workflow_contracts.parse_goal_intent_profile(payload.get("goal_intent_profile"))
        graph = workflow_contracts.parse_intent_graph(payload.get("goal_intent_graph"))
        if profile is not None or graph is not None:
            profile = profile or workflow_contracts.GoalIntentProfile()
            graph = graph or workflow_contracts.IntentGraph()
            envelope = workflow_contracts.NormalizedIntentEnvelope(
                goal=goal,
                profile=profile,
                graph=graph,
                candidate_capabilities=_candidate_capabilities_from_graph(graph),
                clarification=workflow_contracts.ClarificationState(
                    needs_clarification=bool(profile.needs_clarification),
                    requires_blocking_clarification=bool(profile.requires_blocking_clarification),
                    missing_inputs=list(profile.missing_slots),
                    questions=list(profile.questions),
                    blocking_slots=list(profile.blocking_slots),
                    slot_values=dict(profile.slot_values),
                    clarification_mode=profile.clarification_mode,
                ),
            )
    if envelope is None:
        return {}, None, {}
    envelope_json = workflow_contracts.dump_normalized_intent_envelope(envelope) or {}
    return (
        workflow_contracts.dump_goal_intent_profile(envelope.profile) or {},
        workflow_contracts.dump_intent_graph(envelope.graph),
        envelope_json,
    )


def _job_error_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    plan_error = _non_empty_string(metadata.get("plan_error"))
    if plan_error is not None:
        return plan_error
    if metadata.get("plan_preflight_errors") is not None:
        return f"plan_preflight_failed: {metadata.get('plan_preflight_errors')}"
    return None


def build_chat_message_snapshot(record: ChatMessageRecord) -> dict[str, Any]:
    return {
        "message_id": record.id,
        "session_id": record.session_id,
        "role": record.role,
        "content": record.content,
        "metadata": dict(record.metadata_json or {}),
        "action": dict(record.action_json or {}) if isinstance(record.action_json, Mapping) else {},
        "job_id": record.job_id,
        "created_at": record.created_at.isoformat() if isinstance(record.created_at, datetime) else None,
    }


def build_intent_snapshot(job: JobRecord) -> dict[str, Any]:
    metadata = dict(job.metadata_json or {})
    profile, graph, envelope = _normalized_intent_snapshot(metadata, goal=str(job.goal or "").strip())
    return {
        "job_id": job.id,
        "goal": job.goal,
        "job_status": job.status,
        "goal_intent_profile": profile,
        "goal_intent_graph": graph,
        "normalized_intent_envelope": envelope,
        "created_at": job.created_at.isoformat() if isinstance(job.created_at, datetime) else None,
        "updated_at": job.updated_at.isoformat() if isinstance(job.updated_at, datetime) else None,
    }


def build_plan_snapshot(db: Session, plan: PlanRecord) -> dict[str, Any]:
    tasks = (
        db.query(TaskRecord)
        .filter(TaskRecord.plan_id == plan.id)
        .order_by(TaskRecord.created_at.asc(), TaskRecord.name.asc())
        .all()
    )
    return {
        "plan_id": plan.id,
        "job_id": plan.job_id,
        "planner_version": plan.planner_version,
        "tasks_summary": plan.tasks_summary,
        "dag_edges": plan.dag_edges or [],
        "policy_decision": dict(plan.policy_decision or {}),
        "task_count": len(tasks),
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "instruction": task.instruction,
                "intent": task.intent,
                "deps": list(task.deps or []),
                "tool_requests": list(task.tool_requests or []),
                "tool_inputs": dict(task.tool_inputs or {}),
                "status": task.status,
            }
            for task in tasks
        ],
        "created_at": plan.created_at.isoformat() if isinstance(plan.created_at, datetime) else None,
    }


def build_job_outcome_snapshot(db: Session, job: JobRecord) -> dict[str, Any]:
    metadata = dict(job.metadata_json or {})
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
    task_count = 0
    if plan is not None:
        task_count = db.query(TaskRecord).filter(TaskRecord.plan_id == plan.id).count()
    return {
        "job_id": job.id,
        "goal": job.goal,
        "status": job.status,
        "priority": job.priority,
        "job_error": _job_error_from_metadata(metadata),
        "plan_id": plan.id if plan is not None else None,
        "planner_version": plan.planner_version if plan is not None else None,
        "tasks_summary": plan.tasks_summary if plan is not None else None,
        "task_count": task_count,
        "created_at": job.created_at.isoformat() if isinstance(job.created_at, datetime) else None,
        "updated_at": job.updated_at.isoformat() if isinstance(job.updated_at, datetime) else None,
    }


def _feedback_from_record(record: FeedbackRecord) -> models.Feedback:
    return models.Feedback(
        id=record.id,
        target_type=models.FeedbackTargetType(record.target_type),
        target_id=record.target_id,
        session_id=record.session_id,
        job_id=record.job_id,
        plan_id=record.plan_id,
        message_id=record.message_id,
        user_id=record.user_id,
        actor_key=record.actor_key,
        sentiment=models.FeedbackSentiment(record.sentiment),
        score=record.score,
        reason_codes=_normalize_reason_codes(record.reason_codes_json),
        comment=_non_empty_string(record.comment),
        snapshot=dict(record.snapshot_json or {}),
        metadata=dict(record.metadata_json or {}),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _resolve_feedback_target(
    db: Session,
    target_type: models.FeedbackTargetType,
    target_id: str,
) -> dict[str, Any]:
    if target_type == models.FeedbackTargetType.chat_message:
        message = db.query(ChatMessageRecord).filter(ChatMessageRecord.id == target_id).first()
        if message is None:
            raise ValueError("feedback_target_not_found")
        plan = None
        if _non_empty_string(message.job_id):
            plan = db.query(PlanRecord).filter(PlanRecord.job_id == message.job_id).first()
        return {
            "session_id": message.session_id,
            "job_id": message.job_id,
            "plan_id": plan.id if plan is not None else None,
            "message_id": message.id,
            "snapshot": build_chat_message_snapshot(message),
        }
    if target_type == models.FeedbackTargetType.intent_assessment:
        job = db.query(JobRecord).filter(JobRecord.id == target_id).first()
        if job is None:
            raise ValueError("feedback_target_not_found")
        session = (
            db.query(ChatMessageRecord.session_id)
            .filter(ChatMessageRecord.job_id == job.id)
            .order_by(ChatMessageRecord.created_at.asc())
            .first()
        )
        plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
        return {
            "session_id": session[0] if session is not None else None,
            "job_id": job.id,
            "plan_id": plan.id if plan is not None else None,
            "message_id": None,
            "snapshot": build_intent_snapshot(job),
        }
    if target_type == models.FeedbackTargetType.plan:
        plan = db.query(PlanRecord).filter(PlanRecord.id == target_id).first()
        if plan is None:
            raise ValueError("feedback_target_not_found")
        session = (
            db.query(ChatMessageRecord.session_id)
            .filter(ChatMessageRecord.job_id == plan.job_id)
            .order_by(ChatMessageRecord.created_at.asc())
            .first()
        )
        return {
            "session_id": session[0] if session is not None else None,
            "job_id": plan.job_id,
            "plan_id": plan.id,
            "message_id": None,
            "snapshot": build_plan_snapshot(db, plan),
        }
    if target_type == models.FeedbackTargetType.job_outcome:
        job = db.query(JobRecord).filter(JobRecord.id == target_id).first()
        if job is None:
            raise ValueError("feedback_target_not_found")
        session = (
            db.query(ChatMessageRecord.session_id)
            .filter(ChatMessageRecord.job_id == job.id)
            .order_by(ChatMessageRecord.created_at.asc())
            .first()
        )
        plan = db.query(PlanRecord).filter(PlanRecord.job_id == job.id).first()
        return {
            "session_id": session[0] if session is not None else None,
            "job_id": job.id,
            "plan_id": plan.id if plan is not None else None,
            "message_id": None,
            "snapshot": build_job_outcome_snapshot(db, job),
        }
    raise ValueError("feedback_target_type_invalid")


def submit_feedback(
    db: Session,
    request: models.FeedbackCreate,
    *,
    actor_key: str | None = None,
    user_id: str | None = None,
    now: datetime | None = None,
) -> models.Feedback:
    target_id = _non_empty_string(request.target_id)
    if target_id is None:
        raise ValueError("feedback_target_required")
    timestamp = now or _utcnow()
    normalized_actor_key = _non_empty_string(actor_key)
    normalized_user_id = _non_empty_string(user_id)
    resolution = _resolve_feedback_target(db, request.target_type, target_id)

    record: FeedbackRecord | None = None
    if normalized_actor_key is not None:
        record = (
            db.query(FeedbackRecord)
            .filter(
                FeedbackRecord.target_type == request.target_type.value,
                FeedbackRecord.target_id == target_id,
                FeedbackRecord.actor_key == normalized_actor_key,
            )
            .first()
        )
    if record is None:
        record = FeedbackRecord(
            id=str(uuid.uuid4()),
            target_type=request.target_type.value,
            target_id=target_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        db.add(record)

    record.session_id = resolution.get("session_id")
    record.job_id = resolution.get("job_id")
    record.plan_id = resolution.get("plan_id")
    record.message_id = resolution.get("message_id")
    record.user_id = normalized_user_id
    record.actor_key = normalized_actor_key
    record.sentiment = request.sentiment.value
    record.score = request.score
    record.reason_codes_json = _normalize_reason_codes(request.reason_codes)
    record.comment = _non_empty_string(request.comment)
    record.snapshot_json = dict(resolution.get("snapshot") or {})
    record.metadata_json = _normalize_metadata(request.metadata)
    record.updated_at = timestamp
    db.commit()
    db.refresh(record)
    return _feedback_from_record(record)


def list_feedback(
    db: Session,
    *,
    target_type: models.FeedbackTargetType | None = None,
    target_id: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
    plan_id: str | None = None,
    message_id: str | None = None,
    actor_key: str | None = None,
    limit: int = 200,
) -> list[models.Feedback]:
    query = db.query(FeedbackRecord)
    if target_type is not None:
        query = query.filter(FeedbackRecord.target_type == target_type.value)
    normalized_target_id = _non_empty_string(target_id)
    if normalized_target_id is not None:
        query = query.filter(FeedbackRecord.target_id == normalized_target_id)
    normalized_session_id = _non_empty_string(session_id)
    if normalized_session_id is not None:
        query = query.filter(FeedbackRecord.session_id == normalized_session_id)
    normalized_job_id = _non_empty_string(job_id)
    if normalized_job_id is not None:
        query = query.filter(FeedbackRecord.job_id == normalized_job_id)
    normalized_plan_id = _non_empty_string(plan_id)
    if normalized_plan_id is not None:
        query = query.filter(FeedbackRecord.plan_id == normalized_plan_id)
    normalized_message_id = _non_empty_string(message_id)
    if normalized_message_id is not None:
        query = query.filter(FeedbackRecord.message_id == normalized_message_id)
    normalized_actor_key = _non_empty_string(actor_key)
    if normalized_actor_key is not None:
        query = query.filter(FeedbackRecord.actor_key == normalized_actor_key)
    rows = (
        query.order_by(FeedbackRecord.updated_at.desc(), FeedbackRecord.created_at.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    return [_feedback_from_record(row) for row in rows]


def summarize_feedback(items: list[models.Feedback]) -> models.FeedbackSummary:
    summary = models.FeedbackSummary(total=len(items))
    for item in items:
        if item.sentiment == models.FeedbackSentiment.positive:
            summary.positive += 1
        elif item.sentiment == models.FeedbackSentiment.negative:
            summary.negative += 1
        elif item.sentiment == models.FeedbackSentiment.neutral:
            summary.neutral += 1
        elif item.sentiment == models.FeedbackSentiment.partial:
            summary.partial += 1
    return summary


def list_feedback_response(
    db: Session,
    *,
    target_type: models.FeedbackTargetType | None = None,
    target_id: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
    plan_id: str | None = None,
    message_id: str | None = None,
    actor_key: str | None = None,
    limit: int = 200,
) -> models.FeedbackListResponse:
    items = list_feedback(
        db,
        target_type=target_type,
        target_id=target_id,
        session_id=session_id,
        job_id=job_id,
        plan_id=plan_id,
        message_id=message_id,
        actor_key=actor_key,
        limit=limit,
    )
    return models.FeedbackListResponse(items=items, summary=summarize_feedback(items))
