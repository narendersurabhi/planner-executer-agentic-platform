from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from sqlalchemy.orm import Session

from libs.core import chat_contracts, models, workflow_contracts

from .models import ChatMessageRecord, ChatSessionRecord


@dataclass(frozen=True)
class ChatServiceRuntime:
    route_turn: Callable[..., dict[str, Any]]
    execute_direct_capability: Callable[..., dict[str, Any]]
    create_job: Callable[[models.JobCreate, Session], models.Job]
    utcnow: Callable[[], datetime]
    make_id: Callable[[], str]


def create_session(
    db: Session,
    request: chat_contracts.ChatSessionCreate,
    *,
    runtime: ChatServiceRuntime,
) -> chat_contracts.ChatSession:
    now = runtime.utcnow()
    title = str(request.title or "").strip() or "New chat"
    record = ChatSessionRecord(
        id=runtime.make_id(),
        title=title,
        metadata_json=dict(request.metadata) if isinstance(request.metadata, dict) else {},
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    db.commit()
    return get_session(db, record.id)


def get_session(db: Session, session_id: str) -> chat_contracts.ChatSession | None:
    record = db.query(ChatSessionRecord).filter(ChatSessionRecord.id == session_id).first()
    if record is None:
        return None
    messages = (
        db.query(ChatMessageRecord)
        .filter(ChatMessageRecord.session_id == session_id)
        .order_by(ChatMessageRecord.created_at.asc(), ChatMessageRecord.id.asc())
        .all()
    )
    return _session_from_record(record, messages)


def handle_turn(
    db: Session,
    session_id: str,
    request: chat_contracts.ChatTurnRequest,
    *,
    runtime: ChatServiceRuntime,
) -> chat_contracts.ChatTurnResponse:
    record = db.query(ChatSessionRecord).filter(ChatSessionRecord.id == session_id).first()
    if record is None:
        raise KeyError(session_id)

    content = str(request.content or "").strip()
    if not content:
        raise ValueError("content_required")

    now = runtime.utcnow()
    session_metadata = dict(record.metadata_json or {})
    session_context = _coerce_context_json(session_metadata.get("context_json"))
    turn_context = dict(request.context_json) if isinstance(request.context_json, dict) else {}
    merged_context = {**session_context, **turn_context}

    user_message = ChatMessageRecord(
        id=runtime.make_id(),
        session_id=record.id,
        role=chat_contracts.ChatRole.user.value,
        content=content,
        metadata_json={"context_json": turn_context} if turn_context else {},
        action_json=None,
        job_id=None,
        created_at=now,
    )
    db.add(user_message)

    candidate_goal = _candidate_goal(content, session_metadata)
    messages = _message_records_for_session(db, record.id)
    turn_plan = runtime.route_turn(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=merged_context,
        messages=[_message_from_record(message) for message in messages],
    )
    assessment = workflow_contracts.dump_goal_intent_profile(
        turn_plan.get("goal_intent_profile")
    ) or {}
    route_type = str(turn_plan.get("type") or "").strip().lower() or "respond"
    assistant_content = str(turn_plan.get("assistant_content") or "").strip()
    if not assistant_content:
        assistant_content = "What should I do next?"

    assistant_action: chat_contracts.AssistantAction
    created_job: models.Job | None = None
    direct_output: dict[str, Any] | None = None

    if route_type == "ask_clarification":
        questions = [
            str(question).strip()
            for question in turn_plan.get("clarification_questions", assessment.get("questions", []))
            if isinstance(question, str) and question.strip()
        ]
        if questions and not assistant_content:
            assistant_content = "\n".join(questions)
        assistant_action = chat_contracts.AssistantAction(
            type=chat_contracts.AssistantActionType.ask_clarification,
            goal=candidate_goal,
            clarification_questions=questions,
            goal_intent_profile=dict(assessment),
            context_json=merged_context,
        )
        session_metadata["draft_goal"] = candidate_goal
        session_metadata["pending_clarification"] = {
            "goal_intent_profile": dict(assessment),
            "questions": questions,
        }
    elif route_type == "submit_job":
        created_job = runtime.create_job(
            models.JobCreate(
                goal=candidate_goal,
                context_json=merged_context,
                priority=request.priority,
            ),
            db,
        )
        if not assistant_content:
            assistant_content = (
                f"Started job {created_job.id}. "
                "I submitted it to the normal planner and worker pipeline."
            )
        assistant_action = chat_contracts.AssistantAction(
            type=chat_contracts.AssistantActionType.submit_job,
            goal=candidate_goal,
            job_id=created_job.id,
            goal_intent_profile=dict(assessment),
            context_json=merged_context,
        )
        session_metadata["active_job_id"] = created_job.id
        session_metadata.pop("draft_goal", None)
        session_metadata.pop("pending_clarification", None)
    elif route_type == "tool_call":
        capability_id = str(turn_plan.get("capability_id") or "").strip()
        arguments = (
            dict(turn_plan.get("arguments"))
            if isinstance(turn_plan.get("arguments"), Mapping)
            else {}
        )
        arguments = _enrich_memory_arguments(capability_id, arguments, merged_context)
        try:
            direct_result = runtime.execute_direct_capability(
                capability_id=capability_id,
                arguments=arguments,
                trace_id=record.id,
            )
            direct_output = (
                dict(direct_result.get("output"))
                if isinstance(direct_result.get("output"), Mapping)
                else None
            )
            assistant_content = str(direct_result.get("assistant_response") or assistant_content).strip()
            assistant_action = chat_contracts.AssistantAction(
                type=chat_contracts.AssistantActionType.tool_call,
                goal=candidate_goal,
                capability_id=str(direct_result.get("capability_id") or capability_id or "") or None,
                tool_name=str(direct_result.get("tool_name") or "") or None,
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
        except Exception as exc:  # noqa: BLE001
            assistant_content = (
                "I could not complete that directly in chat. "
                f"Direct capability call failed: {exc}"
            )
            assistant_action = chat_contracts.AssistantAction(
                type=chat_contracts.AssistantActionType.respond,
                goal=candidate_goal,
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
    else:
        assistant_action = chat_contracts.AssistantAction(
            type=chat_contracts.AssistantActionType.respond,
            goal=candidate_goal,
            goal_intent_profile=dict(assessment),
            context_json=merged_context,
        )

    session_metadata["context_json"] = merged_context
    record.metadata_json = session_metadata
    record.updated_at = runtime.utcnow()
    if record.title == "New chat":
        record.title = _default_session_title(candidate_goal)

    assistant_message = ChatMessageRecord(
        id=runtime.make_id(),
        session_id=record.id,
        role=chat_contracts.ChatRole.assistant.value,
        content=assistant_content,
        metadata_json=_assistant_metadata(assessment, direct_output),
        action_json=assistant_action.model_dump(mode="json", exclude_none=True),
        job_id=created_job.id if created_job is not None else None,
        created_at=runtime.utcnow(),
    )
    db.add(assistant_message)
    db.commit()

    session = get_session(db, record.id)
    if session is None:
        raise KeyError(record.id)
    return chat_contracts.ChatTurnResponse(
        session=session,
        user_message=_message_from_record(user_message),
        assistant_message=_message_from_record(assistant_message),
        job=created_job,
    )


def _candidate_goal(content: str, session_metadata: Mapping[str, Any]) -> str:
    draft_goal = session_metadata.get("draft_goal")
    pending = session_metadata.get("pending_clarification")
    if isinstance(draft_goal, str) and draft_goal.strip() and isinstance(pending, Mapping):
        return f"{draft_goal.strip()}\n\nUser clarification: {content.strip()}"
    return content.strip()


def _message_records_for_session(db: Session, session_id: str) -> list[ChatMessageRecord]:
    return (
        db.query(ChatMessageRecord)
        .filter(ChatMessageRecord.session_id == session_id)
        .order_by(ChatMessageRecord.created_at.asc(), ChatMessageRecord.id.asc())
        .all()
    )


def _coerce_context_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _default_session_title(goal: str) -> str:
    title = " ".join(str(goal or "").split()).strip()
    if not title:
        return "New chat"
    return title[:80]


def _enrich_memory_arguments(
    capability_id: str,
    arguments: dict[str, Any],
    merged_context: Mapping[str, Any],
) -> dict[str, Any]:
    if capability_id not in {"memory.read", "memory.semantic.search"}:
        return arguments
    normalized_user_id = str(
        merged_context.get("user_id") or merged_context.get("semantic_user_id") or ""
    ).strip()
    if not normalized_user_id or str(arguments.get("user_id") or "").strip():
        return arguments
    enriched = dict(arguments)
    enriched["user_id"] = normalized_user_id
    if capability_id == "memory.read" and not str(enriched.get("scope") or "").strip():
        enriched["scope"] = "user"
    return enriched


def _session_from_record(
    record: ChatSessionRecord,
    messages: list[ChatMessageRecord],
) -> chat_contracts.ChatSession:
    metadata = dict(record.metadata_json or {})
    return chat_contracts.ChatSession(
        id=record.id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=metadata,
        active_job_id=_active_job_id(metadata),
        messages=[_message_from_record(message) for message in messages],
    )


def _message_from_record(record: ChatMessageRecord) -> chat_contracts.ChatMessage:
    action = None
    if isinstance(record.action_json, dict):
        action = chat_contracts.AssistantAction.model_validate(record.action_json)
    return chat_contracts.ChatMessage(
        id=record.id,
        session_id=record.session_id,
        role=record.role,
        content=record.content,
        created_at=record.created_at,
        metadata=dict(record.metadata_json or {}),
        action=action,
        job_id=record.job_id,
    )


def _active_job_id(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("active_job_id")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _assistant_metadata(
    assessment: Mapping[str, Any],
    direct_output: Any = None,
) -> dict[str, Any]:
    metadata = {
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(assessment) or {}
    }
    if isinstance(direct_output, Mapping):
        metadata["tool_output"] = dict(direct_output)
    return metadata
