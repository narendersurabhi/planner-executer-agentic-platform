from __future__ import annotations

from collections import Counter
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
    StepAttemptRecord,
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


def _boundary_decision_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _boundary_decision_fields(boundary_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(boundary_payload) if isinstance(boundary_payload, Mapping) else {}
    evidence = payload.get("evidence")
    normalized_evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    top_families = normalized_evidence.get("top_families")
    top_family = None
    if isinstance(top_families, list):
        for item in top_families:
            if not isinstance(item, Mapping):
                continue
            top_family = _non_empty_string(item.get("family"))
            if top_family is not None:
                break
    return {
        "boundary_decision": _non_empty_string(payload.get("decision")),
        "boundary_reason_code": _non_empty_string(payload.get("reason_code")),
        "boundary_conversation_mode_hint": _non_empty_string(
            normalized_evidence.get("conversation_mode_hint")
        ),
        "boundary_top_family": top_family,
    }


def _boundary_fields_from_message(
    message: ChatMessageRecord | None,
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    message_metadata = (
        dict(message.metadata_json or {}) if message is not None and isinstance(message.metadata_json, Mapping) else {}
    )
    boundary_payload = _boundary_decision_payload(message_metadata.get("boundary_decision"))
    if not boundary_payload and isinstance(snapshot, Mapping):
        snapshot_metadata = snapshot.get("metadata")
        if isinstance(snapshot_metadata, Mapping):
            boundary_payload = _boundary_decision_payload(snapshot_metadata.get("boundary_decision"))
    return _boundary_decision_fields(boundary_payload)


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


def derive_feedback_dimensions(
    db: Session,
    request: models.FeedbackCreate,
    *,
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = dict(resolution.get("snapshot") or {})
    job_id = _non_empty_string(resolution.get("job_id"))
    plan_id = _non_empty_string(resolution.get("plan_id"))
    message_id = _non_empty_string(resolution.get("message_id"))
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first() if job_id else None
    plan = db.query(PlanRecord).filter(PlanRecord.id == plan_id).first() if plan_id else None
    message = (
        db.query(ChatMessageRecord).filter(ChatMessageRecord.id == message_id).first()
        if message_id
        else None
    )
    job_metadata = dict(job.metadata_json or {}) if job is not None else {}
    message_action = (
        dict(message.action_json or {})
        if message is not None and isinstance(message.action_json, Mapping)
        else {}
    )
    comment = _non_empty_string(request.comment)
    reason_codes = _normalize_reason_codes(request.reason_codes)
    workflow_source = _non_empty_string(job_metadata.get("workflow_source"))
    llm_provider = _non_empty_string(job_metadata.get("llm_provider"))
    llm_model = _non_empty_string(job_metadata.get("llm_model"))
    planner_version = (
        _non_empty_string(plan.planner_version if plan is not None else None)
        or _non_empty_string(snapshot.get("planner_version"))
    )
    job_status = (
        _non_empty_string(job.status if job is not None else None)
        or _non_empty_string(snapshot.get("job_status"))
        or _non_empty_string(snapshot.get("status"))
    )
    assistant_action_type = _non_empty_string(message_action.get("type"))
    boundary_fields = _boundary_fields_from_message(message, snapshot)
    return {
        "target_type": request.target_type.value,
        "target_id": _non_empty_string(request.target_id),
        "workflow_source": workflow_source,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "planner_version": planner_version,
        "job_status_at_feedback": job_status,
        "assistant_action_type": assistant_action_type,
        "boundary_decision": boundary_fields.get("boundary_decision"),
        "boundary_reason_code": boundary_fields.get("boundary_reason_code"),
        "boundary_conversation_mode_hint": boundary_fields.get("boundary_conversation_mode_hint"),
        "boundary_top_family": boundary_fields.get("boundary_top_family"),
        "has_comment": bool(comment),
        "reason_count": len(reason_codes),
    }


def _feedback_dimensions(item: models.Feedback) -> dict[str, Any]:
    metadata = dict(item.metadata or {})
    dimensions = metadata.get("dimensions")
    normalized = dict(dimensions) if isinstance(dimensions, Mapping) else {}
    snapshot = dict(item.snapshot or {})
    if "target_type" not in normalized:
        normalized["target_type"] = item.target_type.value
    if "target_id" not in normalized:
        normalized["target_id"] = item.target_id
    if "has_comment" not in normalized:
        normalized["has_comment"] = bool(_non_empty_string(item.comment))
    if "reason_count" not in normalized:
        normalized["reason_count"] = len(_normalize_reason_codes(item.reason_codes))
    if not _non_empty_string(normalized.get("planner_version")):
        normalized["planner_version"] = _non_empty_string(snapshot.get("planner_version"))
    if not _non_empty_string(normalized.get("job_status_at_feedback")):
        normalized["job_status_at_feedback"] = (
            _non_empty_string(snapshot.get("job_status")) or _non_empty_string(snapshot.get("status"))
        )
    if not _non_empty_string(normalized.get("assistant_action_type")):
        action = snapshot.get("action")
        if isinstance(action, Mapping):
            normalized["assistant_action_type"] = _non_empty_string(action.get("type"))
    boundary_fields = _boundary_fields_from_message(None, snapshot)
    for key, value in boundary_fields.items():
        if not _non_empty_string(normalized.get(key)) and value is not None:
            normalized[key] = value
    return normalized


def _filter_feedback_items(
    items: list[models.Feedback],
    *,
    workflow_source: str | None = None,
    llm_model: str | None = None,
    planner_version: str | None = None,
    reason_code: str | None = None,
    sentiments: set[models.FeedbackSentiment] | None = None,
) -> list[models.Feedback]:
    normalized_workflow_source = _non_empty_string(workflow_source)
    normalized_llm_model = _non_empty_string(llm_model)
    normalized_planner_version = _non_empty_string(planner_version)
    normalized_reason_code = _non_empty_string(reason_code)
    filtered = items
    if sentiments:
        filtered = [item for item in filtered if item.sentiment in sentiments]
    if normalized_workflow_source is not None:
        filtered = [
            item
            for item in filtered
            if _non_empty_string(_feedback_dimensions(item).get("workflow_source"))
            == normalized_workflow_source
        ]
    if normalized_llm_model is not None:
        filtered = [
            item
            for item in filtered
            if _non_empty_string(_feedback_dimensions(item).get("llm_model")) == normalized_llm_model
        ]
    if normalized_planner_version is not None:
        filtered = [
            item
            for item in filtered
            if _non_empty_string(_feedback_dimensions(item).get("planner_version"))
            == normalized_planner_version
        ]
    if normalized_reason_code is not None:
        filtered = [
            item
            for item in filtered
            if normalized_reason_code in _normalize_reason_codes(item.reason_codes)
        ]
    return filtered


def _increment_sentiment_bucket(
    bucket: models.FeedbackSummary | models.FeedbackBreakdownBucket,
    sentiment: models.FeedbackSentiment,
) -> None:
    bucket.total += 1
    if sentiment == models.FeedbackSentiment.positive:
        bucket.positive += 1
    elif sentiment == models.FeedbackSentiment.negative:
        bucket.negative += 1
    elif sentiment == models.FeedbackSentiment.neutral:
        bucket.neutral += 1
    elif sentiment == models.FeedbackSentiment.partial:
        bucket.partial += 1


def dimension_breakdown(
    items: list[models.Feedback],
    dimension_key: str,
    *,
    fallback: str | None = None,
) -> list[models.FeedbackBreakdownBucket]:
    buckets: dict[str, models.FeedbackBreakdownBucket] = {}
    for item in items:
        dimensions = _feedback_dimensions(item)
        value = _non_empty_string(dimensions.get(dimension_key))
        if value is None and fallback == "target_type":
            value = item.target_type.value
        if value is None:
            continue
        bucket = buckets.get(value)
        if bucket is None:
            bucket = models.FeedbackBreakdownBucket(key=value)
            buckets[value] = bucket
        _increment_sentiment_bucket(bucket, item.sentiment)
    return sorted(buckets.values(), key=lambda bucket: (-bucket.total, bucket.key))


def reason_code_breakdown(items: list[models.Feedback]) -> list[models.FeedbackReasonBucket]:
    counts: Counter[str] = Counter()
    for item in items:
        if item.sentiment not in {
            models.FeedbackSentiment.negative,
            models.FeedbackSentiment.partial,
        }:
            continue
        for reason in _normalize_reason_codes(item.reason_codes):
            counts[reason] += 1
    return [
        models.FeedbackReasonBucket(reason_code=reason_code, count=count)
        for reason_code, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
    ]


def collect_job_feedback_correlates(
    db: Session,
    items: list[models.Feedback],
) -> models.FeedbackCorrelationSummary:
    job_ids = sorted(
        {
            normalized
            for normalized in (_non_empty_string(item.job_id) for item in items)
            if normalized is not None
        }
    )
    if not job_ids:
        return models.FeedbackCorrelationSummary()

    jobs = {
        record.id: record
        for record in db.query(JobRecord).filter(JobRecord.id.in_(job_ids)).all()
    }
    tasks_by_job: dict[str, list[TaskRecord]] = {job_id: [] for job_id in job_ids}
    for task in db.query(TaskRecord).filter(TaskRecord.job_id.in_(job_ids)).all():
        tasks_by_job.setdefault(task.job_id, []).append(task)
    attempts_by_job: dict[str, list[StepAttemptRecord]] = {job_id: [] for job_id in job_ids}
    for attempt in db.query(StepAttemptRecord).filter(StepAttemptRecord.job_id.in_(job_ids)).all():
        attempts_by_job.setdefault(attempt.job_id, []).append(attempt)

    session_ids = sorted(
        {
            normalized
            for item in items
            for normalized in [_non_empty_string(item.session_id)]
            if normalized is not None and _non_empty_string(item.job_id) in jobs
        }
    )
    clarification_counts_by_session: dict[str, int] = {session_id: 0 for session_id in session_ids}
    if session_ids:
        messages = (
            db.query(ChatMessageRecord)
            .filter(ChatMessageRecord.session_id.in_(session_ids))
            .all()
        )
        for message in messages:
            if message.role != "assistant" or not isinstance(message.action_json, Mapping):
                continue
            if _non_empty_string(message.action_json.get("type")) != "ask_clarification":
                continue
            clarification_counts_by_session[message.session_id] = (
                clarification_counts_by_session.get(message.session_id, 0) + 1
            )

    terminal_status_counts: dict[str, models.FeedbackBreakdownBucket] = {}
    replan_count = 0
    retry_count = 0
    failed_task_count = 0
    plan_failure_count = 0
    clarification_turn_count = 0

    first_session_by_job: dict[str, str | None] = {}
    for item in items:
        job_id = _non_empty_string(item.job_id)
        if job_id is None or job_id in first_session_by_job:
            continue
        first_session_by_job[job_id] = _non_empty_string(item.session_id)

    for job_id in job_ids:
        job = jobs.get(job_id)
        if job is None:
            continue
        status = _non_empty_string(job.status) or "unknown"
        status_bucket = terminal_status_counts.get(status)
        if status_bucket is None:
            status_bucket = models.FeedbackBreakdownBucket(key=status, total=0)
            terminal_status_counts[status] = status_bucket
        status_bucket.total += 1
        metadata = dict(job.metadata_json or {})
        replan_count += max(0, int(metadata.get("replan_count", 0) or 0))
        if _job_error_from_metadata(metadata):
            plan_failure_count += 1
        task_rows = tasks_by_job.get(job_id, [])
        failed_task_count += sum(1 for task in task_rows if task.status == models.TaskStatus.failed.value)
        retry_count += sum(max(0, int(task.rework_count or 0)) for task in task_rows)
        step_attempts = attempts_by_job.get(job_id, [])
        attempts_by_step: dict[str, int] = {}
        for attempt in step_attempts:
            attempts_by_step[attempt.step_id] = max(
                attempts_by_step.get(attempt.step_id, 0),
                max(0, int(attempt.attempt_number or 0)),
            )
        retry_count += sum(max(0, count - 1) for count in attempts_by_step.values())
        session_id = first_session_by_job.get(job_id)
        if session_id is not None:
            clarification_turn_count += clarification_counts_by_session.get(session_id, 0)

    return models.FeedbackCorrelationSummary(
        job_count=len(jobs),
        replan_count=replan_count,
        retry_count=retry_count,
        failed_task_count=failed_task_count,
        plan_failure_count=plan_failure_count,
        clarification_turn_count=clarification_turn_count,
        terminal_statuses=sorted(
            terminal_status_counts.values(),
            key=lambda bucket: (-bucket.total, bucket.key),
        ),
    )


def _positive_rate(
    items: list[models.Feedback],
    target_type: models.FeedbackTargetType,
) -> float:
    scoped = [item for item in items if item.target_type == target_type]
    if not scoped:
        return 0.0
    positive = sum(1 for item in scoped if item.sentiment == models.FeedbackSentiment.positive)
    return round(positive / len(scoped), 4)


def summarize_feedback_rows(
    db: Session,
    items: list[models.Feedback],
) -> models.FeedbackSummaryResponse:
    return models.FeedbackSummaryResponse(
        total=len(items),
        sentiment_counts=summarize_feedback(items),
        target_type_counts=dimension_breakdown(items, "target_type", fallback="target_type"),
        negative_reasons=reason_code_breakdown(items),
        workflow_sources=dimension_breakdown(items, "workflow_source"),
        llm_models=dimension_breakdown(items, "llm_model"),
        planner_versions=dimension_breakdown(items, "planner_version"),
        job_statuses=dimension_breakdown(items, "job_status_at_feedback"),
        assistant_action_types=dimension_breakdown(items, "assistant_action_type"),
        boundary_decisions=dimension_breakdown(items, "boundary_decision"),
        boundary_reason_codes=dimension_breakdown(items, "boundary_reason_code"),
        boundary_top_families=dimension_breakdown(items, "boundary_top_family"),
        metrics={
            "chat_helpfulness_rate": _positive_rate(
                items,
                models.FeedbackTargetType.chat_message,
            ),
            "intent_agreement_rate": _positive_rate(
                items,
                models.FeedbackTargetType.intent_assessment,
            ),
            "plan_approval_rate": _positive_rate(
                items,
                models.FeedbackTargetType.plan,
            ),
            "job_outcome_positive_rate": _positive_rate(
                items,
                models.FeedbackTargetType.job_outcome,
            ),
        },
        correlates=collect_job_feedback_correlates(db, items),
    )


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
    metadata = _normalize_metadata(request.metadata)
    dimensions = dict(metadata.get("dimensions") or {}) if isinstance(metadata.get("dimensions"), Mapping) else {}
    dimensions.update(
        derive_feedback_dimensions(
            db,
            request,
            resolution=resolution,
        )
    )
    metadata["dimensions"] = dimensions
    record.metadata_json = metadata
    record.updated_at = timestamp
    db.commit()
    db.refresh(record)
    return _feedback_from_record(record)


def _feedback_records(
    db: Session,
    *,
    target_type: models.FeedbackTargetType | None = None,
    target_id: str | None = None,
    session_id: str | None = None,
    job_id: str | None = None,
    plan_id: str | None = None,
    message_id: str | None = None,
    actor_key: str | None = None,
    sentiment: models.FeedbackSentiment | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 200,
) -> list[FeedbackRecord]:
    query = db.query(FeedbackRecord)
    if target_type is not None:
        query = query.filter(FeedbackRecord.target_type == target_type.value)
    if sentiment is not None:
        query = query.filter(FeedbackRecord.sentiment == sentiment.value)
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
    if since is not None:
        query = query.filter(FeedbackRecord.created_at >= since)
    if until is not None:
        query = query.filter(FeedbackRecord.created_at <= until)
    return (
        query.order_by(FeedbackRecord.updated_at.desc(), FeedbackRecord.created_at.desc())
        .limit(max(1, min(limit, 5000)))
        .all()
    )


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
    rows = _feedback_records(
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


def export_feedback_examples(
    db: Session,
    *,
    target_type: models.FeedbackTargetType | None = None,
    sentiments: list[models.FeedbackSentiment] | None = None,
    reason_code: str | None = None,
    workflow_source: str | None = None,
    llm_model: str | None = None,
    planner_version: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 200,
) -> models.FeedbackExampleExportResponse:
    rows = _feedback_records(
        db,
        target_type=target_type,
        since=since,
        until=until,
        limit=max(limit, 5000),
    )
    items = [_feedback_from_record(row) for row in rows]
    allowed_sentiments = set(sentiments or [])
    if not allowed_sentiments:
        allowed_sentiments = {
            models.FeedbackSentiment.negative,
            models.FeedbackSentiment.partial,
        }
    filtered = _filter_feedback_items(
        items,
        workflow_source=workflow_source,
        llm_model=llm_model,
        planner_version=planner_version,
        reason_code=reason_code,
        sentiments=allowed_sentiments,
    )[: max(1, min(limit, 5000))]
    examples = [
        models.FeedbackExample(
            feedback=item,
            snapshot=dict(item.snapshot or {}),
            dimensions=_feedback_dimensions(item),
            linked_ids={
                "session_id": item.session_id,
                "job_id": item.job_id,
                "plan_id": item.plan_id,
                "message_id": item.message_id,
                "target_id": item.target_id,
            },
        )
        for item in filtered
    ]
    return models.FeedbackExampleExportResponse(total=len(examples), items=examples)


def _chat_boundary_review_label(item: models.Feedback) -> str | None:
    if item.target_type != models.FeedbackTargetType.chat_message:
        return None
    if item.sentiment not in {
        models.FeedbackSentiment.negative,
        models.FeedbackSentiment.partial,
    }:
        return None
    dimensions = _feedback_dimensions(item)
    boundary_decision = _non_empty_string(dimensions.get("boundary_decision"))
    assistant_action_type = _non_empty_string(dimensions.get("assistant_action_type"))
    if boundary_decision == "chat_reply" and assistant_action_type == "respond":
        return "likely_false_chat_reply"
    if boundary_decision == "execution_request" and assistant_action_type in {
        "submit_job",
        "ask_clarification",
        "tool_call",
        "run_workflow",
    }:
        return "likely_false_execution_request"
    if boundary_decision == "continue_pending":
        return "likely_pending_misroute"
    return None


def _chat_boundary_review_score(item: models.Feedback, review_label: str) -> int:
    score = 0
    if item.sentiment == models.FeedbackSentiment.negative:
        score += 100
    elif item.sentiment == models.FeedbackSentiment.partial:
        score += 70
    if review_label == "likely_false_chat_reply":
        score += 40
    elif review_label == "likely_false_execution_request":
        score += 30
    elif review_label == "likely_pending_misroute":
        score += 20
    reasons = set(_normalize_reason_codes(item.reason_codes))
    if "missed_request" in reasons:
        score += 25
    if "incorrect" in reasons or "wrong_scope" in reasons:
        score += 15
    if _non_empty_string(item.comment):
        score += 5
    return score


def chat_boundary_review_queue(
    db: Session,
    *,
    limit: int = 100,
    review_label: str | None = None,
    boundary_decision: str | None = None,
) -> models.ChatBoundaryReviewQueueResponse:
    rows = _feedback_records(
        db,
        target_type=models.FeedbackTargetType.chat_message,
        limit=max(limit * 4, 200),
    )
    normalized_review_label = _non_empty_string(review_label)
    normalized_boundary_decision = _non_empty_string(boundary_decision)
    items: list[models.ChatBoundaryReviewQueueItem] = []
    for row in rows:
        feedback = _feedback_from_record(row)
        dimensions = _feedback_dimensions(feedback)
        label = _chat_boundary_review_label(feedback)
        if label is None:
            continue
        if normalized_review_label is not None and label != normalized_review_label:
            continue
        if (
            normalized_boundary_decision is not None
            and _non_empty_string(dimensions.get("boundary_decision")) != normalized_boundary_decision
        ):
            continue
        excerpt = _non_empty_string(feedback.snapshot.get("content")) if isinstance(feedback.snapshot, Mapping) else None
        items.append(
            models.ChatBoundaryReviewQueueItem(
                feedback=feedback,
                dimensions=dimensions,
                linked_ids={
                    "session_id": feedback.session_id,
                    "job_id": feedback.job_id,
                    "plan_id": feedback.plan_id,
                    "message_id": feedback.message_id,
                    "target_id": feedback.target_id,
                },
                review_label=label,
                review_score=_chat_boundary_review_score(feedback, label),
                excerpt=excerpt,
            )
        )
    ordered = sorted(
        items,
        key=lambda item: (-item.review_score, item.feedback.updated_at),
    )
    return models.ChatBoundaryReviewQueueResponse(
        total=len(ordered),
        items=ordered[: max(1, min(limit, 500))],
    )


def summary_feedback_response(
    db: Session,
    request: models.FeedbackSummaryRequest,
) -> models.FeedbackSummaryResponse:
    requires_dimension_filter = any(
        [
            _non_empty_string(request.workflow_source),
            _non_empty_string(request.llm_model),
            _non_empty_string(request.planner_version),
        ]
    )
    scan_limit = 5000 if requires_dimension_filter else request.limit
    rows = _feedback_records(
        db,
        target_type=request.target_type,
        sentiment=request.sentiment,
        since=request.since,
        until=request.until,
        limit=scan_limit,
    )
    items = [_feedback_from_record(row) for row in rows]
    items = _filter_feedback_items(
        items,
        workflow_source=request.workflow_source,
        llm_model=request.llm_model,
        planner_version=request.planner_version,
    )
    items = items[: max(1, min(request.limit, 5000))]
    return summarize_feedback_rows(db, items)
