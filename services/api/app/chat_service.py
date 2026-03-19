from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Callable

from sqlalchemy.orm import Session

from libs.core import chat_contracts, models, workflow_contracts

from .models import ChatMessageRecord, ChatSessionRecord


@dataclass(frozen=True)
class ChatServiceRuntime:
    route_turn: Callable[..., dict[str, Any]]
    execute_direct_capability: Callable[..., dict[str, Any]]
    create_job: Callable[[models.JobCreate, Session], models.Job]
    run_workflow: Callable[..., models.WorkflowRunResult]
    inspect_workflow: Callable[..., "ChatWorkflowInspection"]
    utcnow: Callable[[], datetime]
    make_id: Callable[[], str]


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
    turn_context = _prepare_turn_context(
        request.context_json,
        session_metadata=session_metadata,
        content=content,
    )
    merged_context = _merge_chat_context(session_context, turn_context)

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
                goal=candidate_goal,
                clarification_questions=[question],
                goal_intent_profile=dict(assessment),
                context_json=merged_context,
            )
            session_metadata["draft_goal"] = candidate_goal
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
                        goal=candidate_goal,
                        clarification_questions=[question],
                        goal_intent_profile=dict(assessment),
                        context_json=merged_context,
                    )
                    session_metadata["draft_goal"] = candidate_goal
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
                        goal=candidate_goal,
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
                    goal=candidate_goal,
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
    if not assistant_content:
        assistant_content = "What should I do next?"

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
        metadata_json=_assistant_metadata(assessment, direct_output, workflow_run),
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
        workflow_run=workflow_run,
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
    workflow_run: models.WorkflowRun | None = None,
) -> dict[str, Any]:
    metadata = {
        "goal_intent_profile": workflow_contracts.dump_goal_intent_profile(assessment) or {}
    }
    if isinstance(direct_output, Mapping):
        metadata["tool_output"] = dict(direct_output)
    if isinstance(workflow_run, models.WorkflowRun):
        metadata["workflow_run"] = workflow_run.model_dump(mode="json", exclude_none=True)
    return metadata


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
