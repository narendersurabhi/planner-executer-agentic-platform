from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from typing import Any, Callable

from sqlalchemy.orm import Session

from libs.core import chat_contracts, models, workflow_contracts

from . import context_service, memory_profile_service
from .models import ChatMessageRecord, ChatSessionRecord

logger = logging.getLogger("api.chat_service")
_INTERNAL_CHAT_USER_ID_KEY = "_chat_user_id"


@dataclass(frozen=True)
class ChatServiceRuntime:
    route_turn: Callable[..., dict[str, Any]]
    run_direct_capability: Callable[..., "ChatDirectRunResult"]
    create_job: Callable[[models.JobCreate, Session], models.Job]
    run_workflow: Callable[..., models.WorkflowRunResult]
    inspect_workflow: Callable[..., "ChatWorkflowInspection"]
    utcnow: Callable[[], datetime]
    make_id: Callable[[], str]
    normalize_submit_context: Callable[..., "ChatSubmitNormalizationResult | None"] | None = None
    is_chat_only_correction: Callable[[str], bool] | None = None


@dataclass(frozen=True)
class ChatWorkflowInvocation:
    trigger_id: str | None = None
    version_id: str | None = None
    definition_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    context_json: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None

    def has_target(self) -> bool:
        return bool(self.trigger_id or self.version_id or self.definition_id)


@dataclass(frozen=True)
class ChatWorkflowInspection:
    trigger_id: str | None = None
    version_id: str | None = None
    definition_id: str | None = None
    missing_inputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ChatDirectRunResult:
    job: models.Job
    capability_id: str
    tool_name: str | None = None
    output: dict[str, Any] = field(default_factory=dict)
    assistant_response: str = ""
    error: str | None = None


@dataclass(frozen=True)
class ChatSubmitNormalizationResult:
    goal: str | None = None
    context_json: dict[str, Any] = field(default_factory=dict)
    clarification_questions: list[str] = field(default_factory=list)


def looks_like_chat_only_correction(content: str) -> bool:
    lowered = str(content or "").strip().lower()
    if not lowered:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if not tokens:
        return False

    negation_tokens = {"no", "not", "dont", "don't", "do", "without", "skip", "cancel", "stop", "instead", "rather"}
    execution_tokens = {
        "document",
        "doc",
        "docx",
        "pdf",
        "markdown",
        "report",
        "file",
        "files",
        "workflow",
        "job",
        "artifact",
        "artifacts",
        "render",
        "generate",
        "create",
        "submit",
    }
    chat_tokens = {
        "chat",
        "response",
        "respond",
        "answer",
        "reply",
        "explain",
        "thoughts",
        "here",
    }
    redirect_tokens = {"just", "only", "instead", "rather"}

    has_execution_reference = bool(tokens & execution_tokens)
    has_chat_reference = bool(tokens & chat_tokens)
    has_negation_or_redirect = bool(tokens & negation_tokens)
    has_redirect_reference = bool(tokens & redirect_tokens)

    if has_chat_reference and (has_negation_or_redirect or has_redirect_reference):
        return True
    if has_execution_reference and has_chat_reference:
        return True
    if has_execution_reference and has_negation_or_redirect:
        return True

    compact = re.sub(r"[^a-z0-9]+", " ", lowered)
    semantic_patterns = (
        r"\b(?:answer|reply|respond|explain)\b.{0,24}\b(?:here|in chat|in the chat)\b",
        r"\b(?:just|only)\b.{0,24}\b(?:chat|response|reply|answer)\b",
        r"\b(?:not|no|dont|don't|skip|cancel|stop)\b.{0,24}\b(?:document|workflow|job|pdf|docx|report|artifact)\b",
        r"\binstead\b.{0,24}\b(?:chat|answer|reply|respond|here)\b",
    )
    return any(re.search(pattern, compact) for pattern in semantic_patterns)


def create_session(
    db: Session,
    request: chat_contracts.ChatSessionCreate,
    *,
    runtime: ChatServiceRuntime,
    user_id: str | None = None,
) -> chat_contracts.ChatSession:
    now = runtime.utcnow()
    title = str(request.title or "").strip() or "New chat"
    metadata = _coerce_context_json(request.metadata)
    metadata.pop(_INTERNAL_CHAT_USER_ID_KEY, None)
    metadata.pop("user_id", None)
    metadata.pop("semantic_user_id", None)
    if isinstance(metadata.get("context_json"), Mapping):
        metadata["context_json"] = _sanitize_chat_context(metadata.get("context_json"))
    if _normalized_user_id(user_id):
        metadata[_INTERNAL_CHAT_USER_ID_KEY] = _normalized_user_id(user_id)
    record = ChatSessionRecord(
        id=runtime.make_id(),
        title=title,
        metadata_json=metadata,
        created_at=now,
        updated_at=now,
    )
    db.add(record)
    db.commit()
    return get_session(db, record.id, user_id=user_id)


def get_session(
    db: Session,
    session_id: str,
    *,
    user_id: str | None = None,
) -> chat_contracts.ChatSession | None:
    record = db.query(ChatSessionRecord).filter(ChatSessionRecord.id == session_id).first()
    if record is None:
        return None
    if not _chat_session_access_allowed(record, user_id):
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
    user_id: str | None = None,
) -> chat_contracts.ChatTurnResponse:
    record = db.query(ChatSessionRecord).filter(ChatSessionRecord.id == session_id).first()
    if record is None:
        raise KeyError(session_id)
    if not _chat_session_access_allowed(record, user_id):
        raise KeyError(session_id)

    content = str(request.content or "").strip()
    if not content:
        raise ValueError("content_required")

    now = runtime.utcnow()
    session_metadata = dict(record.metadata_json or {})
    bound_user_id = _normalized_user_id(user_id) or _chat_session_user_id(session_metadata)
    if bound_user_id and not _chat_session_user_id(session_metadata):
        session_metadata[_INTERNAL_CHAT_USER_ID_KEY] = bound_user_id
    session_context = _sanitize_chat_context(_coerce_context_json(session_metadata.get("context_json")))
    turn_context = _prepare_turn_context(
        request.context_json,
        session_metadata=session_metadata,
        content=content,
    )
    turn_context = _sanitize_chat_context(turn_context)
    candidate_goal = _candidate_goal(
        content,
        session_metadata,
        is_chat_only_correction=runtime.is_chat_only_correction,
    )
    context_envelope = context_service.build_chat_context_envelope(
        db=db,
        goal=candidate_goal,
        session_metadata=session_metadata,
        session_context=session_context,
        turn_context=turn_context,
        user_id=bound_user_id,
    )
    merged_context = context_service.chat_submit_context_view(context_envelope)
    route_context = context_service.chat_route_context_view(context_envelope)

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

    messages = _message_records_for_session(db, record.id)
    turn_plan = runtime.route_turn(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=route_context,
        messages=[_message_from_record(message) for message in messages],
    )
    assessment = workflow_contracts.dump_goal_intent_profile(
        turn_plan.get("goal_intent_profile")
    ) or {}
    boundary_decision = (
        dict(turn_plan.get("boundary_decision"))
        if isinstance(turn_plan.get("boundary_decision"), Mapping)
        else None
    )
    route_type = str(turn_plan.get("type") or "").strip().lower() or "respond"
    resolved_goal = str(turn_plan.get("resolved_goal") or candidate_goal or "").strip()
    if not resolved_goal:
        resolved_goal = content
    assistant_content = str(turn_plan.get("assistant_content") or "").strip()

    assistant_action: chat_contracts.AssistantAction
    created_job: models.Job | None = None
    direct_output: dict[str, Any] | None = None
    workflow_run: models.WorkflowRun | None = None

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
            goal=resolved_goal,
            clarification_questions=questions,
            goal_intent_profile=dict(assessment),
            context_json=merged_context,
        )
        session_metadata["draft_goal"] = resolved_goal
        session_metadata["pending_clarification"] = {
            "goal_intent_profile": dict(assessment),
            "questions": questions,
        }
    elif route_type == "submit_job":
        normalization = None
        if runtime.normalize_submit_context is not None:
            try:
                normalization = runtime.normalize_submit_context(
                    db=db,
                    goal=resolved_goal,
                    content=content,
                    session_metadata=session_metadata,
                    merged_context=merged_context,
                    context_envelope=context_envelope,
                    user_id=bound_user_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "chat_submit_normalization_failed",
                    extra={"session_id": record.id},
                )
                normalization = None
        if normalization is not None:
            if isinstance(normalization.context_json, Mapping) and normalization.context_json:
                context_envelope = context_service.update_chat_context_envelope(
                    context_envelope,
                    goal=resolved_goal,
                    context_json=normalization.context_json,
                )
                merged_context = context_service.chat_submit_context_view(context_envelope)
            if isinstance(normalization.goal, str) and normalization.goal.strip():
                resolved_goal = normalization.goal.strip()
                context_envelope = context_service.update_chat_context_envelope(
                    context_envelope,
                    goal=resolved_goal,
                )
            clarification_questions = [
                str(question).strip()
                for question in normalization.clarification_questions
                if isinstance(question, str) and question.strip()
            ]
            if clarification_questions:
                if not assistant_content:
                    assistant_content = "\n".join(clarification_questions)
                assistant_action = chat_contracts.AssistantAction(
                    type=chat_contracts.AssistantActionType.ask_clarification,
                    goal=resolved_goal,
                    clarification_questions=clarification_questions,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
                session_metadata["draft_goal"] = resolved_goal
                session_metadata["pending_clarification"] = {
                    "goal_intent_profile": dict(assessment),
                    "questions": clarification_questions,
                }
            else:
                created_job = runtime.create_job(
                    models.JobCreate(
                        goal=resolved_goal,
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
                    goal=resolved_goal,
                    job_id=created_job.id,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
                session_metadata["active_job_id"] = created_job.id
                session_metadata.pop("draft_goal", None)
                session_metadata.pop("pending_clarification", None)
        else:
            created_job = runtime.create_job(
                models.JobCreate(
                    goal=resolved_goal,
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
                goal=resolved_goal,
                job_id=created_job.id,
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
            session_metadata["active_job_id"] = created_job.id
            session_metadata.pop("draft_goal", None)
            session_metadata.pop("pending_clarification", None)
    elif route_type == "run_workflow":
        workflow_invocation = workflow_invocation_from_context(merged_context)
        if workflow_invocation is None or not workflow_invocation.has_target():
            question = (
                "Which published workflow should I run? Provide workflow_trigger_id, "
                "workflow_version_id, or workflow_definition_id in context_json."
            )
            if not assistant_content:
                assistant_content = question
            assistant_action = chat_contracts.AssistantAction(
                type=chat_contracts.AssistantActionType.ask_clarification,
                goal=resolved_goal,
                clarification_questions=[question],
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
            session_metadata["draft_goal"] = resolved_goal
            session_metadata["pending_clarification"] = {
                "goal_intent_profile": dict(assessment),
                "questions": [question],
            }
        else:
            try:
                workflow_inspection = runtime.inspect_workflow(
                    db=db,
                    workflow_trigger_id=workflow_invocation.trigger_id,
                    workflow_version_id=workflow_invocation.version_id,
                    workflow_definition_id=workflow_invocation.definition_id,
                    inputs=workflow_invocation.inputs,
                    context_json=workflow_invocation.context_json,
                )
                if workflow_inspection.missing_inputs:
                    next_input = workflow_inspection.missing_inputs[0]
                    question = _workflow_input_question(next_input)
                    if not assistant_content:
                        assistant_content = question
                    assistant_action = chat_contracts.AssistantAction(
                        type=chat_contracts.AssistantActionType.ask_clarification,
                        goal=resolved_goal,
                        clarification_questions=[question],
                        goal_intent_profile=dict(assessment),
                        context_json=merged_context,
                    )
                    session_metadata["draft_goal"] = resolved_goal
                    session_metadata["pending_clarification"] = {
                        "goal_intent_profile": dict(assessment),
                        "questions": [question],
                    }
                    session_metadata["pending_workflow_input"] = dict(next_input)
                else:
                    workflow_result = runtime.run_workflow(
                        db=db,
                        workflow_trigger_id=workflow_invocation.trigger_id,
                        workflow_version_id=workflow_invocation.version_id,
                        workflow_definition_id=workflow_invocation.definition_id,
                        inputs=workflow_invocation.inputs,
                        context_json=workflow_invocation.context_json,
                        metadata={
                            **workflow_invocation.metadata,
                            "chat_session_id": record.id,
                        },
                        idempotency_key=workflow_invocation.idempotency_key,
                        priority=request.priority,
                    )
                    created_job = workflow_result.job
                    workflow_run = workflow_result.workflow_run
                    if not assistant_content:
                        assistant_content = (
                            f"Started workflow run {workflow_run.id}. "
                            f"Job {created_job.id} is queued."
                        )
                    assistant_action = chat_contracts.AssistantAction(
                        type=chat_contracts.AssistantActionType.run_workflow,
                        goal=resolved_goal,
                        job_id=created_job.id,
                        workflow_run_id=workflow_run.id,
                        workflow_definition_id=workflow_run.definition_id,
                        workflow_version_id=workflow_run.version_id,
                        workflow_trigger_id=workflow_run.trigger_id,
                        goal_intent_profile=dict(assessment),
                        context_json=merged_context,
                    )
                    session_metadata["active_job_id"] = created_job.id
                    session_metadata["active_workflow_run_id"] = workflow_run.id
                    session_metadata["active_workflow_definition_id"] = workflow_run.definition_id
                    session_metadata["active_workflow_version_id"] = workflow_run.version_id
                    if workflow_run.trigger_id:
                        session_metadata["active_workflow_trigger_id"] = workflow_run.trigger_id
                    else:
                        session_metadata.pop("active_workflow_trigger_id", None)
                    session_metadata.pop("draft_goal", None)
                    session_metadata.pop("pending_clarification", None)
                    session_metadata.pop("pending_workflow_input", None)
            except Exception as exc:  # noqa: BLE001
                assistant_content = (
                    "I could not start that published workflow from chat. "
                    f"Workflow invocation failed: {exc}"
                )
                assistant_action = chat_contracts.AssistantAction(
                    type=chat_contracts.AssistantActionType.respond,
                    goal=resolved_goal,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
    elif route_type == "tool_call":
        capability_id = str(turn_plan.get("capability_id") or "").strip()
        arguments = (
            dict(turn_plan.get("arguments"))
            if isinstance(turn_plan.get("arguments"), Mapping)
            else {}
        )
        arguments = _enrich_memory_arguments(capability_id, arguments, merged_context)
        try:
            direct_result = runtime.run_direct_capability(
                db=db,
                chat_session_id=record.id,
                goal=resolved_goal,
                capability_id=capability_id,
                arguments=arguments,
                context_json=merged_context,
                priority=request.priority,
            )
            created_job = direct_result.job
            if direct_result.error:
                assistant_content = (
                    "I could not complete that directly in chat. "
                    f"One-step run failed: {direct_result.error}"
                )
                assistant_action = chat_contracts.AssistantAction(
                    type=chat_contracts.AssistantActionType.respond,
                    goal=resolved_goal,
                    job_id=created_job.id,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
            else:
                direct_output = (
                    dict(direct_result.output)
                    if isinstance(direct_result.output, Mapping)
                    else None
                )
                assistant_content = str(
                    direct_result.assistant_response or assistant_content
                ).strip()
                assistant_action = chat_contracts.AssistantAction(
                    type=chat_contracts.AssistantActionType.tool_call,
                    goal=resolved_goal,
                    job_id=created_job.id,
                    capability_id=direct_result.capability_id or capability_id or None,
                    tool_name=direct_result.tool_name,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
                session_metadata.pop("draft_goal", None)
                session_metadata.pop("pending_clarification", None)
                session_metadata.pop("pending_workflow_input", None)
        except Exception as exc:  # noqa: BLE001
            assistant_content = (
                "I could not complete that directly in chat. "
                f"One-step run failed: {exc}"
            )
            assistant_action = chat_contracts.AssistantAction(
                type=chat_contracts.AssistantActionType.respond,
                goal=resolved_goal,
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
    else:
        assistant_action = chat_contracts.AssistantAction(
            type=chat_contracts.AssistantActionType.respond,
            goal=resolved_goal,
            goal_intent_profile=dict(assessment),
            context_json=merged_context,
        )
        if bool(turn_plan.get("clear_pending_clarification")):
            session_metadata.pop("draft_goal", None)
            session_metadata.pop("pending_clarification", None)
            session_metadata.pop("pending_workflow_input", None)
    if not assistant_content:
        assistant_content = "What should I do next?"

    context_envelope = context_service.update_chat_context_envelope(
        context_envelope,
        goal=resolved_goal,
        context_json=merged_context,
    )
    merged_context = context_service.chat_submit_context_view(context_envelope)
    session_metadata["context_json"] = merged_context
    record.metadata_json = session_metadata
    record.updated_at = runtime.utcnow()
    if record.title == "New chat":
        record.title = _default_session_title(resolved_goal)

    assistant_message = ChatMessageRecord(
        id=runtime.make_id(),
        session_id=record.id,
        role=chat_contracts.ChatRole.assistant.value,
        content=assistant_content,
        metadata_json=_assistant_metadata(
            assessment,
            direct_output,
            workflow_run,
            boundary_decision=boundary_decision,
        ),
        action_json=assistant_action.model_dump(mode="json", exclude_none=True),
        job_id=created_job.id if created_job is not None else None,
        created_at=runtime.utcnow(),
    )
    db.add(assistant_message)
    db.commit()
    if bound_user_id:
        try:
            memory_profile_service.apply_user_profile_updates_from_text(
                db,
                user_id=bound_user_id,
                content=content,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "chat_profile_memory_persist_failed",
                extra={"session_id": record.id},
            )

    session = get_session(db, record.id, user_id=bound_user_id or user_id)
    if session is None:
        raise KeyError(record.id)
    return chat_contracts.ChatTurnResponse(
        session=session,
        user_message=_message_from_record(user_message),
        assistant_message=_message_from_record(assistant_message),
        job=created_job,
        workflow_run=workflow_run,
    )


def _candidate_goal(
    content: str,
    session_metadata: Mapping[str, Any],
    *,
    is_chat_only_correction: Callable[[str], bool] | None = None,
) -> str:
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


def _prepare_turn_context(
    value: Any,
    *,
    session_metadata: Mapping[str, Any],
    content: str,
) -> dict[str, Any]:
    turn_context = _coerce_context_json(value)
    pending_input = session_metadata.get("pending_workflow_input")
    if not isinstance(pending_input, Mapping):
        return turn_context
    input_key = _normalized_optional_str(pending_input.get("key"))
    if not input_key:
        return turn_context

    workflow_inputs = (
        dict(turn_context.get("workflow_inputs"))
        if isinstance(turn_context.get("workflow_inputs"), Mapping)
        else {}
    )
    if input_key in workflow_inputs:
        if _workflow_input_omission_requested(workflow_inputs.get(input_key)):
            workflow_inputs[input_key] = None
        turn_context["workflow_inputs"] = workflow_inputs
        return turn_context

    if input_key in turn_context:
        raw_value = turn_context[input_key]
        workflow_inputs[input_key] = (
            None if _workflow_input_omission_requested(raw_value) else raw_value
        )
    elif str(content or "").strip():
        workflow_inputs[input_key] = (
            None if _workflow_input_omission_requested(content) else str(content).strip()
        )

    if workflow_inputs:
        turn_context["workflow_inputs"] = workflow_inputs
    return turn_context


def _merge_chat_context(
    session_context: Mapping[str, Any],
    turn_context: Mapping[str, Any],
) -> dict[str, Any]:
    merged = {**dict(session_context), **dict(turn_context)}
    for key in ("workflow_inputs", "workflow_context_json", "workflow_run_metadata"):
        session_value = session_context.get(key)
        turn_value = turn_context.get(key)
        if isinstance(session_value, Mapping) or isinstance(turn_value, Mapping):
            merged[key] = {
                **(dict(session_value) if isinstance(session_value, Mapping) else {}),
                **(dict(turn_value) if isinstance(turn_value, Mapping) else {}),
            }
    session_ref = session_context.get("workflow_ref")
    turn_ref = turn_context.get("workflow_ref")
    if isinstance(session_ref, Mapping) or isinstance(turn_ref, Mapping):
        merged["workflow_ref"] = {
            **(dict(session_ref) if isinstance(session_ref, Mapping) else {}),
            **(dict(turn_ref) if isinstance(turn_ref, Mapping) else {}),
        }
    return merged


def workflow_invocation_from_context(
    value: Mapping[str, Any] | None,
) -> ChatWorkflowInvocation | None:
    context = dict(value) if isinstance(value, Mapping) else {}
    nested = context.get("workflow_ref")
    nested_ref = dict(nested) if isinstance(nested, Mapping) else {}

    control_keys = {
        "workflow_ref",
        "workflow_trigger_id",
        "workflow_version_id",
        "workflow_definition_id",
        "workflow_inputs",
        "workflow_context_json",
        "workflow_run_metadata",
        "workflow_idempotency_key",
    }
    passthrough_context = {
        key: raw_value for key, raw_value in context.items() if key not in control_keys
    }

    trigger_id = _normalized_optional_str(
        context.get("workflow_trigger_id") or nested_ref.get("trigger_id")
    )
    version_id = _normalized_optional_str(
        context.get("workflow_version_id") or nested_ref.get("version_id")
    )
    definition_id = _normalized_optional_str(
        context.get("workflow_definition_id") or nested_ref.get("definition_id")
    )

    raw_inputs = context.get("workflow_inputs")
    if not isinstance(raw_inputs, Mapping):
        raw_inputs = nested_ref.get("inputs")
    inputs = dict(raw_inputs) if isinstance(raw_inputs, Mapping) else {}

    raw_context_override = context.get("workflow_context_json")
    if not isinstance(raw_context_override, Mapping):
        raw_context_override = nested_ref.get("context_json")
    context_override = (
        dict(raw_context_override) if isinstance(raw_context_override, Mapping) else {}
    )

    raw_metadata = context.get("workflow_run_metadata")
    if not isinstance(raw_metadata, Mapping):
        raw_metadata = nested_ref.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}

    idempotency_key = _normalized_optional_str(
        context.get("workflow_idempotency_key") or nested_ref.get("idempotency_key")
    )

    invocation = ChatWorkflowInvocation(
        trigger_id=trigger_id,
        version_id=version_id,
        definition_id=definition_id,
        inputs=inputs,
        context_json={**passthrough_context, **context_override},
        metadata=metadata,
        idempotency_key=idempotency_key,
    )
    if invocation.has_target():
        return invocation
    return None


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
    metadata = _public_session_metadata(record.metadata_json)
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
        action_payload = dict(record.action_json)
        if isinstance(action_payload.get("context_json"), Mapping):
            action_payload["context_json"] = _sanitize_chat_context(action_payload.get("context_json"))
        action = chat_contracts.AssistantAction.model_validate(action_payload)
    return chat_contracts.ChatMessage(
        id=record.id,
        session_id=record.session_id,
        role=record.role,
        content=record.content,
        created_at=record.created_at,
        metadata=_public_session_metadata(record.metadata_json),
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
    workflow_run: models.WorkflowRun | None = None,
    *,
    boundary_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(assessment) or {}
    }
    if isinstance(direct_output, Mapping):
        metadata["tool_output"] = dict(direct_output)
    if isinstance(workflow_run, models.WorkflowRun):
        metadata["workflow_run"] = workflow_run.model_dump(mode="json", exclude_none=True)
    if isinstance(boundary_decision, Mapping):
        metadata["boundary_decision"] = dict(boundary_decision)
    return metadata


def _sanitize_chat_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    context = dict(value) if isinstance(value, Mapping) else {}
    context.pop("user_id", None)
    context.pop("semantic_user_id", None)
    return context


def _public_session_metadata(value: Any) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, Mapping) else {}
    metadata.pop(_INTERNAL_CHAT_USER_ID_KEY, None)
    if isinstance(metadata.get("context_json"), Mapping):
        metadata["context_json"] = _sanitize_chat_context(metadata.get("context_json"))
    return metadata


def _chat_session_user_id(metadata: Mapping[str, Any]) -> str:
    return _normalized_user_id(metadata.get(_INTERNAL_CHAT_USER_ID_KEY))


def _normalized_user_id(value: Any) -> str:
    return str(value or "").strip()


def _chat_session_access_allowed(record: ChatSessionRecord, user_id: str | None) -> bool:
    bound_user_id = _chat_session_user_id(record.metadata_json or {})
    normalized_user_id = _normalized_user_id(user_id)
    if not bound_user_id or not normalized_user_id:
        return True
    return bound_user_id == normalized_user_id


def _workflow_input_question(definition: Mapping[str, Any]) -> str:
    key = str(definition.get("key") or "value").strip() or "value"
    label = str(definition.get("label") or key).strip() or key
    value_type = str(definition.get("value_type") or "string").strip().lower() or "string"
    description = str(definition.get("description") or "").strip()
    message = str(definition.get("message") or "").strip()

    if value_type in {"object", "array"}:
        question = f"I need `{label}` before I can run this workflow. Reply with valid JSON for `{key}`."
    elif value_type == "boolean":
        question = f"I need `{label}` before I can run this workflow. What value should I use for `{key}`? Reply with `true` or `false`."
    elif value_type in {"number", "integer"}:
        question = f"I need `{label}` before I can run this workflow. What numeric value should I use for `{key}`?"
    else:
        question = f"I need `{label}` before I can run this workflow. What value should I use for `{key}`?"

    if description:
        question = f"{question} {description}"
    if message:
        question = f"{question} ({message})"
    question = (
        f"{question} Reply `use default` if you want me to leave it unset and let the workflow "
        "use its own fallback behavior when possible."
    )
    return question


def _workflow_input_omission_requested(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()
    if not normalized:
        return False
    return normalized in {
        "default",
        "use default",
        "use the default",
        "default is fine",
        "the default is fine",
        "leave blank",
        "leave it blank",
        "leave it empty",
        "skip",
        "skip it",
        "omit it",
        "no preference",
        "whatever works",
        "anything is fine",
        "anything works",
        "i dont care",
        "i do not care",
    }


def _normalized_optional_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
