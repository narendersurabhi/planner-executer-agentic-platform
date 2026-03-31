from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from typing import Any, Callable, Sequence

from sqlalchemy.orm import Session

from libs.core import capability_registry, chat_contracts, intent_contract, models, workflow_contracts

from . import context_service, memory_profile_service
from .models import ChatMessageRecord, ChatSessionRecord

logger = logging.getLogger("api.chat_service")
_INTERNAL_CHAT_USER_ID_KEY = "_chat_user_id"
_CHAT_STATE_VERSION_KEY = "_chat_state_version"
_CHAT_STATE_CONFLICT_COUNT_KEY = "_chat_state_conflict_count"
_PENDING_CLARIFICATION_SLOT_KEYS = (
    "instruction",
    "topic",
    "audience",
    "tone",
    "query",
    "path",
    "output_path",
    "filename",
    "file_name",
    "output_filename",
    "output_format",
    "target_system",
    "safety_constraints",
    "intent_action",
    "risk_level",
    "length",
    "markdown_text",
    "target_repo",
)
_CLARIFICATION_INTENT_CHANGE_ARTIFACT_TOKENS = {
    "document",
    "doc",
    "docx",
    "pdf",
    "markdown",
    "json",
    "report",
    "guide",
    "checklist",
    "workflow",
    "job",
    "issue",
    "repo",
    "repository",
    "file",
    "files",
    "summary",
    "runbook",
    "query",
}
_CLARIFICATION_INTENT_CHANGE_ACTION_TOKENS = {
    "create",
    "generate",
    "write",
    "make",
    "render",
    "build",
    "draft",
    "search",
    "list",
    "read",
    "inspect",
    "fetch",
    "produce",
}
_OUTPUT_FORMAT_TOKENS = {"pdf", "docx", "markdown", "json", "word"}
_TONE_TOKENS = {"practical", "formal", "conversational", "executive", "technical", "concise"}


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
    requires_blocking_clarification: bool = False
    goal_intent_profile: dict[str, Any] = field(default_factory=dict)


def _submit_normalization_failure_questions(
    *,
    session_metadata: Mapping[str, Any] | None,
    assessment: Mapping[str, Any] | None,
) -> list[str]:
    questions: list[str] = []

    def _append(raw_question: Any) -> None:
        if isinstance(raw_question, str):
            question = raw_question.strip()
            if question and question not in questions:
                questions.append(question)

    pending = _parse_pending_clarification_state(session_metadata)
    for raw_question in (pending.questions if pending is not None else []):
        _append(raw_question)

    if isinstance(assessment, Mapping):
        for raw_question in assessment.get("questions") or []:
            _append(raw_question)

    if not questions:
        questions.append(
            "I still need the remaining required details before I can submit this request."
        )
    return questions


def _parse_pending_clarification_state(
    metadata: Mapping[str, Any] | None,
) -> chat_contracts.ClarificationState | None:
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("pending_clarification")
    if not isinstance(raw, Mapping):
        return None
    try:
        return chat_contracts.ClarificationState.model_validate(dict(raw))
    except Exception:  # noqa: BLE001
        return None


def _coerce_clarification_state(
    value: Mapping[str, Any] | None,
) -> chat_contracts.ClarificationState | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return chat_contracts.ClarificationState.model_validate(dict(value))
    except Exception:  # noqa: BLE001
        return None


def _session_state_version(metadata: Mapping[str, Any] | None) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    raw = metadata.get(_CHAT_STATE_VERSION_KEY)
    if isinstance(raw, int):
        return max(0, raw)
    if isinstance(raw, str):
        try:
            return max(0, int(raw.strip()))
        except ValueError:
            return 0
    return 0


def _unique_string_list(*values: Sequence[Any] | None) -> list[str]:
    normalized: list[str] = []
    for collection in values:
        if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
            continue
        for item in collection:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in normalized:
                normalized.append(value)
    return normalized


def _merge_clarification_state_for_persistence(
    latest: Mapping[str, Any] | None,
    desired: Mapping[str, Any] | None,
) -> dict[str, Any]:
    latest_state = _coerce_clarification_state(latest)
    desired_state = _coerce_clarification_state(desired)
    if latest_state is None and desired_state is None:
        return {}
    if latest_state is None:
        return desired_state.model_dump(mode="json", exclude_none=True)
    if desired_state is None:
        return latest_state.model_dump(mode="json", exclude_none=True)

    known_slot_values = dict(latest_state.known_slot_values or {})
    known_slot_values.update(dict(desired_state.known_slot_values or {}))
    resolved_slots = dict(latest_state.resolved_slots or {})
    resolved_slots.update(dict(desired_state.resolved_slots or {}))
    slot_provenance = dict(latest_state.slot_provenance or {})
    slot_provenance.update(dict(desired_state.slot_provenance or {}))
    workflow_target = dict(latest_state.execution_frame.workflow_target or {})
    workflow_target.update(dict(desired_state.execution_frame.workflow_target or {}))

    execution_frame = latest_state.execution_frame.model_copy(deep=True)
    execution_frame.original_goal = (
        str(desired_state.execution_frame.original_goal or "").strip()
        or str(latest_state.execution_frame.original_goal or "").strip()
    )
    execution_frame.mode = desired_state.execution_frame.mode
    execution_frame.active_family = desired_state.execution_frame.active_family or execution_frame.active_family
    execution_frame.active_segment_id = (
        desired_state.execution_frame.active_segment_id or execution_frame.active_segment_id
    )
    execution_frame.active_capability_id = (
        desired_state.execution_frame.active_capability_id or execution_frame.active_capability_id
    )
    execution_frame.workflow_target = workflow_target
    execution_frame.state_version = max(
        int(latest_state.execution_frame.state_version or 0),
        int(desired_state.execution_frame.state_version or 0),
    ) + 1

    merged = desired_state.model_copy(deep=True)
    merged.schema_version = desired_state.schema_version or latest_state.schema_version
    merged.state_version = max(
        int(latest_state.state_version or 0),
        int(desired_state.state_version or 0),
    ) + 1
    merged.execution_frame = execution_frame
    merged.original_goal = str(desired_state.original_goal or "").strip() or str(
        latest_state.original_goal or ""
    ).strip()
    merged.active_family = desired_state.active_family or latest_state.active_family
    merged.active_segment_id = desired_state.active_segment_id or latest_state.active_segment_id
    merged.active_capability_id = (
        desired_state.active_capability_id or latest_state.active_capability_id
    )
    merged.goal_intent_profile = {
        **dict(latest_state.goal_intent_profile or {}),
        **dict(desired_state.goal_intent_profile or {}),
    }
    merged.current_question = (
        str(desired_state.current_question or "").strip()
        or str(latest_state.current_question or "").strip()
        or None
    )
    merged.current_question_field = (
        str(desired_state.current_question_field or "").strip()
        or str(latest_state.current_question_field or "").strip()
        or None
    )
    merged.pending_questions = _unique_string_list(
        desired_state.pending_questions or desired_state.questions,
        latest_state.pending_questions if not desired_state.pending_questions else (),
        latest_state.questions if not desired_state.pending_questions else (),
    )
    merged.pending_fields = _unique_string_list(
        desired_state.pending_fields if desired_state.pending_fields else latest_state.pending_fields,
    )
    merged.required_fields = _unique_string_list(
        desired_state.required_fields,
        latest_state.required_fields,
    )
    merged.known_slot_values = known_slot_values
    merged.resolved_slots = resolved_slots
    merged.slot_provenance = slot_provenance
    merged.pending_fields = [
        field
        for field in merged.pending_fields
        if str(field).strip() and str(field).strip() not in known_slot_values
    ]
    merged.answered_fields = sorted(
        {
            *[str(value).strip() for value in latest_state.answered_fields if str(value).strip()],
            *[str(value).strip() for value in desired_state.answered_fields if str(value).strip()],
            *[str(key).strip() for key in known_slot_values.keys() if str(key).strip()],
        }
    )
    if not merged.current_question_field and merged.pending_fields:
        merged.current_question_field = str(merged.pending_fields[0]).strip() or None
    if not merged.current_question and merged.pending_questions:
        merged.current_question = str(merged.pending_questions[0]).strip() or None
    merged.questions = [merged.current_question] if merged.current_question else []
    merged.question_history = _unique_string_list(
        latest_state.question_history,
        desired_state.question_history,
        merged.pending_questions,
    )
    merged.answer_history = _unique_string_list(
        latest_state.answer_history,
        desired_state.answer_history,
    )
    merged.candidate_capabilities = _unique_string_list(
        desired_state.candidate_capabilities,
        latest_state.candidate_capabilities,
    )
    merged.auto_path_allowed = bool(
        latest_state.auto_path_allowed or desired_state.auto_path_allowed
    )
    return merged.model_dump(mode="json", exclude_none=True)


def _clarification_question_queue(questions: Sequence[Any] | None) -> list[str]:
    return _unique_string_list(questions)


def _active_clarification_questions(questions: Sequence[Any] | None) -> list[str]:
    queue = _clarification_question_queue(questions)
    return queue[:1]


def _merge_session_metadata_for_persistence(
    *,
    latest_metadata: Mapping[str, Any] | None,
    desired_metadata: Mapping[str, Any] | None,
    cleared_keys: set[str],
    loaded_state_version: int,
) -> tuple[dict[str, Any], bool]:
    latest = dict(latest_metadata or {})
    desired = dict(desired_metadata or {})
    latest_state_version = _session_state_version(latest)
    conflict_detected = latest_state_version > loaded_state_version

    merged = dict(latest)
    for key, value in desired.items():
        if key in {"context_json", "pending_clarification"}:
            continue
        merged[key] = value

    latest_context = latest.get("context_json")
    desired_context = desired.get("context_json")
    if isinstance(latest_context, Mapping) or isinstance(desired_context, Mapping):
        merged["context_json"] = _merge_chat_context(
            dict(latest_context) if isinstance(latest_context, Mapping) else {},
            dict(desired_context) if isinstance(desired_context, Mapping) else {},
        )

    if "pending_clarification" in cleared_keys:
        merged.pop("pending_clarification", None)
    else:
        merged_pending = _merge_clarification_state_for_persistence(
            latest.get("pending_clarification")
            if isinstance(latest.get("pending_clarification"), Mapping)
            else None,
            desired.get("pending_clarification")
            if isinstance(desired.get("pending_clarification"), Mapping)
            else None,
        )
        if merged_pending:
            merged["pending_clarification"] = merged_pending

    for key in cleared_keys:
        if key == "pending_clarification":
            continue
        merged.pop(key, None)

    merged[_CHAT_STATE_VERSION_KEY] = max(latest_state_version, loaded_state_version) + 1
    if conflict_detected:
        merged[_CHAT_STATE_CONFLICT_COUNT_KEY] = int(
            latest.get(_CHAT_STATE_CONFLICT_COUNT_KEY) or 0
        ) + 1
    elif _CHAT_STATE_CONFLICT_COUNT_KEY in latest:
        merged[_CHAT_STATE_CONFLICT_COUNT_KEY] = int(latest.get(_CHAT_STATE_CONFLICT_COUNT_KEY) or 0)
    return merged, conflict_detected


def _resolved_session_title(
    *,
    latest_title: str,
    desired_title: str,
) -> str:
    normalized_latest = str(latest_title or "").strip() or "New chat"
    normalized_desired = str(desired_title or "").strip() or normalized_latest
    if normalized_latest != "New chat":
        return normalized_latest
    return normalized_desired


def _persist_chat_session_state(
    *,
    db: Session,
    record: ChatSessionRecord,
    desired_metadata: Mapping[str, Any] | None,
    desired_title: str,
    loaded_updated_at: datetime,
    loaded_state_version: int,
    cleared_keys: set[str],
    runtime: ChatServiceRuntime,
) -> dict[str, Any]:
    expected_updated_at = loaded_updated_at
    for _attempt in range(3):
        db.expire_all()
        latest_record = (
            db.query(ChatSessionRecord).filter(ChatSessionRecord.id == record.id).first()
        )
        if latest_record is None:
            raise KeyError(record.id)
        merged_metadata, _conflict_detected = _merge_session_metadata_for_persistence(
            latest_metadata=latest_record.metadata_json,
            desired_metadata=desired_metadata,
            cleared_keys=cleared_keys,
            loaded_state_version=loaded_state_version,
        )
        resolved_title = _resolved_session_title(
            latest_title=latest_record.title,
            desired_title=desired_title,
        )
        updated_at = runtime.utcnow()
        updated_rows = (
            db.query(ChatSessionRecord)
            .filter(
                ChatSessionRecord.id == record.id,
                ChatSessionRecord.updated_at == expected_updated_at,
            )
            .update(
                {
                    ChatSessionRecord.metadata_json: merged_metadata,
                    ChatSessionRecord.updated_at: updated_at,
                    ChatSessionRecord.title: resolved_title,
                },
                synchronize_session=False,
            )
        )
        if updated_rows:
            record.metadata_json = dict(merged_metadata)
            record.updated_at = updated_at
            record.title = resolved_title
            return dict(merged_metadata)
        expected_updated_at = latest_record.updated_at
    raise RuntimeError("chat_session_state_conflict")


def _normalized_clarification_fields(context_json: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(context_json, Mapping):
        return set()
    raw = context_json.get("clarification_normalization")
    if not isinstance(raw, Mapping):
        return set()
    normalized: set[str] = set()
    for value in raw.get("fields", []):
        key = str(value or "").strip()
        if key:
            normalized.add(key)
    return normalized


def _workflow_target_state(context_json: Mapping[str, Any] | None) -> dict[str, Any]:
    invocation = workflow_invocation_from_context(context_json)
    if invocation is None or not invocation.has_target():
        return {}
    workflow_target: dict[str, Any] = {}
    if invocation.trigger_id:
        workflow_target["trigger_id"] = invocation.trigger_id
    if invocation.version_id:
        workflow_target["version_id"] = invocation.version_id
    if invocation.definition_id:
        workflow_target["definition_id"] = invocation.definition_id
    return workflow_target


def _execution_family_for_capability(capability_id: str | None) -> str | None:
    normalized = capability_registry.canonicalize_capability_id(str(capability_id or "").strip())
    if not normalized:
        return None
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        registry = None
    spec = registry.get(normalized) if registry is not None else None
    family = str(spec.group or spec.subgroup or "").strip() if spec is not None else ""
    if family:
        return family
    prefix = normalized.split(".", 1)[0].strip()
    return prefix or None


def _active_execution_target(
    *,
    normalized_intent_envelope: Mapping[str, Any] | None,
    candidate_capabilities: list[str],
    context_json: Mapping[str, Any] | None,
    existing_state: chat_contracts.ClarificationState | None,
    known_slot_values: Mapping[str, Any] | None,
    pending_fields: Sequence[str] | None,
) -> tuple[str | None, str | None, str | None, dict[str, Any]]:
    active_family = existing_state.active_family if existing_state is not None else None
    active_segment_id = existing_state.active_segment_id if existing_state is not None else None
    active_capability_id = existing_state.active_capability_id if existing_state is not None else None
    workflow_target = (
        dict(existing_state.execution_frame.workflow_target)
        if existing_state is not None
        and isinstance(existing_state.execution_frame.workflow_target, Mapping)
        else {}
    )

    envelope = workflow_contracts.parse_normalized_intent_envelope(normalized_intent_envelope)
    if envelope is not None:
        active_target = intent_contract.select_active_execution_target(
            graph=workflow_contracts.dump_intent_graph(envelope.graph) or {},
            candidate_capabilities=envelope.candidate_capabilities,
            known_slot_values=known_slot_values,
            pending_fields=pending_fields or (),
            preferred_segment_id=active_segment_id,
            preferred_capability_id=active_capability_id,
        )
        if active_target is not None:
            active_segment_id = active_target.segment_id or active_segment_id
            active_capability_id = active_target.capability_id or active_capability_id
    if not active_capability_id and candidate_capabilities:
        active_capability_id = next(
            (str(capability_id).strip() for capability_id in candidate_capabilities if str(capability_id).strip()),
            None,
        )
    if active_capability_id:
        active_capability_id = capability_registry.canonicalize_capability_id(active_capability_id) or active_capability_id
        active_family = _execution_family_for_capability(active_capability_id) or active_family
    latest_workflow_target = _workflow_target_state(context_json)
    if latest_workflow_target:
        workflow_target = latest_workflow_target
        active_family = active_family or "workflow"
    return active_family, active_segment_id, active_capability_id, workflow_target


def _pending_clarification_state(
    *,
    resolved_goal: str,
    questions: list[str],
    assessment: Mapping[str, Any] | None,
    session_metadata: Mapping[str, Any] | None,
    context_json: Mapping[str, Any] | None,
    normalized_intent_envelope: Mapping[str, Any] | None = None,
    latest_user_answer: str | None = None,
) -> dict[str, Any]:
    existing_state = _parse_pending_clarification_state(session_metadata)
    question_queue = _clarification_question_queue(questions)
    active_questions = question_queue[:1]
    original_goal = str(
        (existing_state.original_goal if existing_state is not None else "")
        or (session_metadata.get("draft_goal") if isinstance(session_metadata, Mapping) else "")
        or resolved_goal
        or ""
    ).strip()
    known_slot_values = (
        dict(existing_state.known_slot_values)
        if existing_state is not None and isinstance(existing_state.known_slot_values, Mapping)
        else {}
    )
    slot_provenance = (
        dict(existing_state.slot_provenance)
        if existing_state is not None and isinstance(existing_state.slot_provenance, Mapping)
        else {}
    )
    normalized_fields = _normalized_clarification_fields(context_json)
    if isinstance(context_json, Mapping):
        for key in _PENDING_CLARIFICATION_SLOT_KEYS:
            value = context_json.get(key)
            if isinstance(value, str):
                if value.strip():
                    known_slot_values[key] = value.strip()
                    slot_provenance[key] = (
                        workflow_contracts.SlotProvenance.clarification_normalized.value
                        if key in normalized_fields
                        else workflow_contracts.SlotProvenance.explicit_user.value
                    )
                continue
            if value is not None:
                known_slot_values[key] = value
                slot_provenance[key] = (
                    workflow_contracts.SlotProvenance.clarification_normalized.value
                    if key in normalized_fields
                    else workflow_contracts.SlotProvenance.explicit_user.value
                )
    pending_fields: list[str] = []
    required_fields: list[str] = []
    if isinstance(assessment, Mapping):
        for raw_field in (assessment.get("missing_slots") or []) + (assessment.get("blocking_slots") or []):
            if isinstance(raw_field, str):
                field = raw_field.strip()
                if field and field not in required_fields:
                    required_fields.append(field)
                if field and field not in pending_fields:
                    pending_fields.append(field)
    for raw_field in (existing_state.pending_fields if existing_state is not None else []):
        if isinstance(raw_field, str):
            field = raw_field.strip()
            if field and field not in pending_fields and field not in known_slot_values:
                pending_fields.append(field)
    if existing_state is not None:
        for raw_field in existing_state.required_fields:
            if isinstance(raw_field, str):
                field = raw_field.strip()
                if field and field not in required_fields:
                    required_fields.append(field)
    question_history: list[str] = []
    for raw_question in (existing_state.question_history if existing_state is not None else []):
        if isinstance(raw_question, str):
            question = raw_question.strip()
            if question and question not in question_history:
                question_history.append(question)
    for question in questions:
        normalized = str(question or "").strip()
        if normalized and normalized not in question_history:
            question_history.append(normalized)
    answer_history: list[str] = []
    for raw_answer in (existing_state.answer_history if existing_state is not None else []):
        if isinstance(raw_answer, str):
            answer = raw_answer.strip()
            if answer and answer not in answer_history:
                answer_history.append(answer)
    if existing_state is not None and isinstance(latest_user_answer, str):
        latest_answer = latest_user_answer.strip()
        if latest_answer and latest_answer not in answer_history:
            answer_history.append(latest_answer)
    candidate_capabilities: list[str] = []
    raw_candidates = context_json.get("capability_candidates") if isinstance(context_json, Mapping) else None
    if isinstance(raw_candidates, list):
        for raw_value in raw_candidates:
            if isinstance(raw_value, str):
                value = raw_value.strip()
                if value and value not in candidate_capabilities:
                    candidate_capabilities.append(value)
    if not candidate_capabilities:
        for raw_value in (existing_state.candidate_capabilities if existing_state is not None else []):
            if isinstance(raw_value, str):
                value = raw_value.strip()
                if value and value not in candidate_capabilities:
                    candidate_capabilities.append(value)
    active_family, active_segment_id, active_capability_id, workflow_target = _active_execution_target(
        normalized_intent_envelope=normalized_intent_envelope,
        candidate_capabilities=candidate_capabilities,
        context_json=context_json,
        existing_state=existing_state,
        known_slot_values=known_slot_values,
        pending_fields=pending_fields,
    )
    execution_frame = (
        existing_state.execution_frame.model_copy(deep=True)
        if existing_state is not None
        else workflow_contracts.ExecutionFrame()
    )
    execution_frame.original_goal = original_goal
    execution_frame.mode = workflow_contracts.ExecutionFrameMode.clarification
    execution_frame.active_family = active_family
    execution_frame.active_segment_id = active_segment_id
    execution_frame.active_capability_id = active_capability_id
    execution_frame.workflow_target = workflow_target
    execution_frame.state_version = (
        existing_state.execution_frame.state_version + 1
        if existing_state is not None
        else 1
    )
    if not str(execution_frame.frame_id or "").strip():
        execution_frame.frame_id = (
            active_segment_id or active_capability_id or active_family or "clarification"
        )

    state = chat_contracts.ClarificationState(
        schema_version="clarification_state_v1",
        state_version=(existing_state.state_version + 1) if existing_state is not None else 1,
        execution_frame=execution_frame,
        original_goal=original_goal,
        active_family=active_family,
        active_segment_id=active_segment_id,
        active_capability_id=active_capability_id,
        goal_intent_profile=dict(assessment or {}),
        questions=active_questions,
        pending_questions=question_queue,
        current_question=active_questions[0] if active_questions else None,
        current_question_field=pending_fields[0] if pending_fields else None,
        pending_fields=pending_fields,
        required_fields=required_fields,
        known_slot_values=known_slot_values,
        resolved_slots=dict(known_slot_values),
        slot_provenance=slot_provenance,
        answered_fields=sorted(str(key) for key in known_slot_values.keys()),
        question_history=question_history,
        answer_history=answer_history,
        candidate_capabilities=candidate_capabilities,
        auto_path_allowed=bool(existing_state.auto_path_allowed) if existing_state is not None else False,
    )
    return state.model_dump(mode="json", exclude_none=True)


def _apply_pending_clarification_mapping(
    *,
    db: Session,
    runtime: ChatServiceRuntime,
    goal: str,
    content: str,
    session_metadata: Mapping[str, Any],
    merged_context: Mapping[str, Any] | None,
    context_envelope: workflow_contracts.ContextEnvelope,
    user_id: str | None,
    messages: Sequence[chat_contracts.ChatMessage] | None = None,
) -> tuple[
    str,
    workflow_contracts.ContextEnvelope,
    dict[str, Any],
    dict[str, Any],
    ChatSubmitNormalizationResult | None,
]:
    pending_state = _parse_pending_clarification_state(session_metadata)
    if pending_state is None or runtime.normalize_submit_context is None:
        return (
            goal,
            context_envelope,
            dict(merged_context or {}),
            dict(session_metadata),
            None,
        )

    try:
        normalization = runtime.normalize_submit_context(
            db=db,
            goal=goal,
            content=content,
            session_metadata=session_metadata,
            merged_context=merged_context,
            context_envelope=context_envelope,
            user_id=user_id,
            messages=messages,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "chat_pending_clarification_mapping_failed",
            extra={"session_id": session_metadata.get("id")},
        )
        return (
            goal,
            context_envelope,
            dict(merged_context or {}),
            dict(session_metadata),
            None,
        )

    if normalization is None:
        return (
            goal,
            context_envelope,
            dict(merged_context or {}),
            dict(session_metadata),
            None,
        )

    updated_goal = str(normalization.goal or "").strip() or goal
    updated_envelope = context_envelope
    if updated_goal != goal or (
        isinstance(normalization.context_json, Mapping) and normalization.context_json
    ):
        updated_envelope = context_service.update_chat_context_envelope(
            context_envelope,
            goal=updated_goal,
            context_json=normalization.context_json,
        )
    updated_context = context_service.chat_submit_context_view(updated_envelope)
    updated_metadata = dict(session_metadata)
    updated_metadata["draft_goal"] = updated_goal
    updated_metadata["context_json"] = updated_context

    clarification_question_queue = [
        str(question).strip()
        for question in normalization.clarification_questions
        if isinstance(question, str) and question.strip()
    ]
    assessment = (
        dict(normalization.goal_intent_profile)
        if isinstance(normalization.goal_intent_profile, Mapping)
        and normalization.goal_intent_profile
        else dict(pending_state.goal_intent_profile or {})
    )
    updated_metadata["pending_clarification"] = _pending_clarification_state(
        resolved_goal=updated_goal,
        questions=clarification_question_queue,
        assessment=assessment,
        session_metadata=session_metadata,
        context_json=updated_context,
        latest_user_answer=content,
    )
    return updated_goal, updated_envelope, updated_context, updated_metadata, normalization


def _resolved_clarification_mapping_fields(
    normalization: ChatSubmitNormalizationResult | None,
) -> list[str]:
    if normalization is None or not isinstance(normalization.context_json, Mapping):
        return []
    payload = normalization.context_json.get("clarification_normalization")
    if not isinstance(payload, Mapping):
        return []
    resolved: list[str] = []
    for raw_field in payload.get("fields") or []:
        field = intent_contract.normalize_required_input_key(raw_field)
        if field and field not in resolved:
            resolved.append(field)
    return resolved


def _clarification_mapping_metadata(
    *,
    before_state: chat_contracts.ClarificationState | None,
    after_state: chat_contracts.ClarificationState | None,
    normalization: ChatSubmitNormalizationResult | None,
    restarted: bool,
) -> dict[str, Any] | None:
    if before_state is None and not restarted:
        return None
    active_field_before = (
        intent_contract.normalize_required_input_key(before_state.current_question_field)
        if before_state is not None and before_state.current_question_field
        else None
    )
    active_field_after = (
        intent_contract.normalize_required_input_key(after_state.current_question_field)
        if after_state is not None and after_state.current_question_field
        else None
    )
    resolved_fields = _resolved_clarification_mapping_fields(normalization)
    resolved_active_field = bool(
        active_field_before and active_field_before in set(resolved_fields)
    )
    queue_advanced = bool(
        resolved_active_field
        and (active_field_after is None or active_field_after != active_field_before)
    )
    return {
        "active_field_before": active_field_before,
        "active_field_after": active_field_after,
        "resolved_fields": resolved_fields,
        "resolved_active_field": resolved_active_field,
        "queue_advanced": queue_advanced,
        "restarted": bool(restarted),
    }


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


def _looks_like_local_clarification_field_answer(
    content: str,
    *,
    current_field: str | None,
) -> bool:
    normalized_field = intent_contract.normalize_required_input_key(current_field)
    lowered = str(content or "").strip().lower()
    if not lowered or not normalized_field:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if not tokens:
        return False

    if normalized_field == "output_format":
        return bool(tokens & _OUTPUT_FORMAT_TOKENS)
    if normalized_field == "tone":
        return bool(tokens & _TONE_TOKENS) and len(tokens) <= 8
    if normalized_field == "audience":
        return not bool(tokens & _CLARIFICATION_INTENT_CHANGE_ACTION_TOKENS) and len(tokens) <= 12
    if normalized_field in {"path", "output_path"}:
        return bool(
            re.search(
                r"(\.(?:docx|pdf|md|markdown|json)\b|\b(?:workspace|folder|download|filename|path|save)\b)",
                lowered,
            )
        )
    return False


def _looks_like_pending_clarification_intent_change(
    content: str,
    *,
    pending_state: chat_contracts.ClarificationState | None,
) -> bool:
    if pending_state is None:
        return False
    lowered = str(content or "").strip().lower()
    if not lowered:
        return False
    if re.search(r"\b(?:chat|answer|reply|respond|explain|here)\b", lowered) and re.search(
        r"\b(?:instead|only|just|not|no|skip|cancel|stop|rather)\b",
        lowered,
    ):
        return False
    if _looks_like_local_clarification_field_answer(
        content,
        current_field=pending_state.current_question_field,
    ):
        return False
    redirect_patterns = (
        r"\bactually\b",
        r"\binstead\b",
        r"\brather\b",
        r"\bstart over\b",
        r"\brestart\b",
        r"\bnew request\b",
        r"\bdifferent request\b",
        r"\bmake this\b",
        r"\bturn this\b",
        r"\bconvert this\b",
    )
    if not any(re.search(pattern, lowered) for pattern in redirect_patterns):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    return bool(tokens & _CLARIFICATION_INTENT_CHANGE_ARTIFACT_TOKENS) and bool(
        tokens & _CLARIFICATION_INTENT_CHANGE_ACTION_TOKENS
    )


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
    metadata[_CHAT_STATE_VERSION_KEY] = max(1, _session_state_version(metadata) or 1)
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
    loaded_updated_at = record.updated_at
    loaded_state_version = _session_state_version(session_metadata)
    cleared_session_keys: set[str] = set()
    bound_user_id = _normalized_user_id(user_id) or _chat_session_user_id(session_metadata)
    if bound_user_id and not _chat_session_user_id(session_metadata):
        session_metadata[_INTERNAL_CHAT_USER_ID_KEY] = bound_user_id
    restarted_pending_clarification = False
    pending_state = _parse_pending_clarification_state(session_metadata)
    pending_state_before_mapping = pending_state
    if _looks_like_pending_clarification_intent_change(
        content,
        pending_state=pending_state,
    ):
        restarted_pending_clarification = True
        session_metadata.pop("draft_goal", None)
        session_metadata.pop("pending_clarification", None)
        session_metadata.pop("pending_workflow_input", None)
    session_context = _sanitize_chat_context(_coerce_context_json(session_metadata.get("context_json")))
    turn_context = _prepare_turn_context(
        request.context_json,
        session_metadata=session_metadata,
        content=content,
    )
    turn_context = _sanitize_chat_context(turn_context)
    candidate_goal = (
        content.strip()
        if restarted_pending_clarification
        else _candidate_goal(
            content,
            session_metadata,
            is_chat_only_correction=runtime.is_chat_only_correction,
        )
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
    messages = _message_records_for_session(db, record.id)
    chat_messages = [_message_from_record(message) for message in messages]
    pre_route_normalization: ChatSubmitNormalizationResult | None = None
    (
        candidate_goal,
        context_envelope,
        merged_context,
        session_metadata,
        pre_route_normalization,
    ) = _apply_pending_clarification_mapping(
        db=db,
        runtime=runtime,
        goal=candidate_goal,
        content=content,
        session_metadata=session_metadata,
        merged_context=merged_context,
        context_envelope=context_envelope,
        user_id=bound_user_id,
        messages=chat_messages,
    )
    pending_state_after_mapping = _parse_pending_clarification_state(session_metadata)
    clarification_mapping = _clarification_mapping_metadata(
        before_state=pending_state_before_mapping,
        after_state=pending_state_after_mapping,
        normalization=pre_route_normalization,
        restarted=restarted_pending_clarification,
    )
    route_context = context_service.chat_route_context_view(context_envelope)
    turn_plan = runtime.route_turn(
        content=content,
        candidate_goal=candidate_goal,
        session_metadata=session_metadata,
        merged_context=route_context,
        messages=chat_messages,
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
        question_queue = [
            str(question).strip()
            for question in turn_plan.get("clarification_questions", assessment.get("questions", []))
            if isinstance(question, str) and question.strip()
        ]
        questions = _active_clarification_questions(question_queue)
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
        session_metadata["pending_clarification"] = _pending_clarification_state(
            resolved_goal=resolved_goal,
            questions=question_queue,
            assessment=assessment,
            session_metadata=session_metadata,
            context_json=merged_context,
            normalized_intent_envelope=turn_plan.get("normalized_intent_envelope"),
            latest_user_answer=content,
        )
    elif route_type == "submit_job":
        normalization = pre_route_normalization
        if normalization is None and runtime.normalize_submit_context is not None:
            try:
                normalization = runtime.normalize_submit_context(
                    db=db,
                    goal=resolved_goal,
                    content=content,
                    session_metadata=session_metadata,
                    merged_context=merged_context,
                    context_envelope=context_envelope,
                    user_id=bound_user_id,
                    messages=chat_messages,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "chat_submit_normalization_failed",
                    extra={"session_id": record.id},
                )
                normalization = ChatSubmitNormalizationResult(
                    goal=resolved_goal,
                    clarification_questions=_submit_normalization_failure_questions(
                        session_metadata=session_metadata,
                        assessment=assessment,
                    ),
                )
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
            if isinstance(normalization.goal_intent_profile, Mapping) and normalization.goal_intent_profile:
                assessment = dict(normalization.goal_intent_profile)
            clarification_question_queue = [
                str(question).strip()
                for question in normalization.clarification_questions
                if isinstance(question, str) and question.strip()
            ]
            if normalization.requires_blocking_clarification and not clarification_question_queue:
                clarification_question_queue = _submit_normalization_failure_questions(
                    session_metadata=session_metadata,
                    assessment=assessment,
                )
            clarification_questions = _active_clarification_questions(clarification_question_queue)
            if clarification_questions or normalization.requires_blocking_clarification:
                assistant_content = "\n".join(clarification_questions)
                assistant_action = chat_contracts.AssistantAction(
                    type=chat_contracts.AssistantActionType.ask_clarification,
                    goal=resolved_goal,
                    clarification_questions=clarification_questions,
                    goal_intent_profile=dict(assessment),
                    context_json=merged_context,
                )
                session_metadata["draft_goal"] = resolved_goal
                session_metadata["pending_clarification"] = _pending_clarification_state(
                    resolved_goal=resolved_goal,
                    questions=clarification_question_queue,
                    assessment=assessment,
                    session_metadata=session_metadata,
                    context_json=merged_context,
                    normalized_intent_envelope=turn_plan.get("normalized_intent_envelope"),
                    latest_user_answer=content,
                )
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
                cleared_session_keys.update({"draft_goal", "pending_clarification"})
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
            cleared_session_keys.update({"draft_goal", "pending_clarification"})
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
            session_metadata["pending_clarification"] = _pending_clarification_state(
                resolved_goal=resolved_goal,
                questions=[question],
                assessment=assessment,
                session_metadata=session_metadata,
                context_json=merged_context,
                normalized_intent_envelope=turn_plan.get("normalized_intent_envelope"),
                latest_user_answer=content,
            )
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
                    session_metadata["pending_clarification"] = _pending_clarification_state(
                        resolved_goal=resolved_goal,
                        questions=[question],
                        assessment=assessment,
                        session_metadata=session_metadata,
                        context_json=merged_context,
                        normalized_intent_envelope=turn_plan.get("normalized_intent_envelope"),
                        latest_user_answer=content,
                    )
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
                        cleared_session_keys.add("active_workflow_trigger_id")
                    session_metadata.pop("draft_goal", None)
                    session_metadata.pop("pending_clarification", None)
                    session_metadata.pop("pending_workflow_input", None)
                    cleared_session_keys.update(
                        {"draft_goal", "pending_clarification", "pending_workflow_input"}
                    )
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
                cleared_session_keys.update(
                    {"draft_goal", "pending_clarification", "pending_workflow_input"}
                )
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
            cleared_session_keys.update(
                {"draft_goal", "pending_clarification", "pending_workflow_input"}
            )
    if restarted_pending_clarification:
        for key in ("draft_goal", "pending_clarification", "pending_workflow_input"):
            if key not in session_metadata:
                cleared_session_keys.add(key)
    if not assistant_content:
        assistant_content = "What should I do next?"

    context_envelope = context_service.update_chat_context_envelope(
        context_envelope,
        goal=resolved_goal,
        context_json=merged_context,
    )
    merged_context = context_service.chat_submit_context_view(context_envelope)
    session_metadata["context_json"] = merged_context
    desired_title = record.title
    if desired_title == "New chat":
        desired_title = _default_session_title(resolved_goal)
    session_metadata = _persist_chat_session_state(
        db=db,
        record=record,
        desired_metadata=session_metadata,
        desired_title=desired_title,
        loaded_updated_at=loaded_updated_at,
        loaded_state_version=loaded_state_version,
        cleared_keys=cleared_session_keys,
        runtime=runtime,
    )

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
            clarification_mapping=clarification_mapping,
        ),
        action_json=assistant_action.model_dump(mode="json", exclude_none=True),
        job_id=created_job.id if created_job is not None else None,
        created_at=runtime.utcnow(),
    )
    db.add(user_message)
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
    pending_state = _parse_pending_clarification_state(session_metadata)
    original_goal = str(pending_state.original_goal or "").strip() if pending_state is not None else ""
    if original_goal:
        return f"{original_goal}\n\nUser clarification: {content.strip()}"
    if isinstance(draft_goal, str) and draft_goal.strip() and pending_state is not None:
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
    clarification_mapping: Mapping[str, Any] | None = None,
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
    if isinstance(clarification_mapping, Mapping):
        metadata["clarification_mapping"] = dict(clarification_mapping)
    return metadata


def _sanitize_chat_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    context = dict(value) if isinstance(value, Mapping) else {}
    context.pop("user_id", None)
    context.pop("semantic_user_id", None)
    return context


def _public_session_metadata(value: Any) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, Mapping) else {}
    metadata.pop(_INTERNAL_CHAT_USER_ID_KEY, None)
    metadata.pop(_CHAT_STATE_VERSION_KEY, None)
    metadata.pop(_CHAT_STATE_CONFLICT_COUNT_KEY, None)
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
