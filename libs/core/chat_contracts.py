from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libs.core import models, workflow_contracts


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatBoundaryDecisionType(str, Enum):
    chat_reply = "chat_reply"
    execution_request = "execution_request"
    continue_pending = "continue_pending"
    exit_pending_to_chat = "exit_pending_to_chat"
    meta_clarification = "meta_clarification"


class ChatBoundaryCapabilityEvidence(BaseModel):
    capability_id: str
    group: str | None = None
    subgroup: str | None = None
    score: float | None = None
    source: str | None = None
    reason: str | None = None


class ChatBoundaryFamilyEvidence(BaseModel):
    family: str
    score: float = 0.0
    capability_ids: list[str] = Field(default_factory=list)


class ChatBoundaryEvidence(BaseModel):
    goal: str = ""
    conversation_mode_hint: str = ""
    pending_clarification: bool = False
    workflow_target_available: bool = False
    likely_clarification_answer: bool = False
    intent: str = ""
    risk_level: str = ""
    needs_clarification: bool = False
    missing_inputs: list[str] = Field(default_factory=list)
    active_family: str = ""
    active_capability_id: str = ""
    clarification_resolved_slot_count: int = 0
    clarification_pending_field_count: int = 0
    clarification_answer_count: int = 0
    top_capability_score: float = 0.0
    top_family_score: float = 0.0
    family_concentration: float = 0.0
    execution_signal_strength: str = ""
    top_capabilities: list[ChatBoundaryCapabilityEvidence] = Field(default_factory=list)
    top_families: list[ChatBoundaryFamilyEvidence] = Field(default_factory=list)


class ClarificationResolvedField(BaseModel):
    field: str
    value: Any
    confidence: float | None = None
    source: str | None = None


class ClarificationState(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "clarification_state_v1"
    state_version: int = 1
    execution_frame: workflow_contracts.ExecutionFrame = Field(
        default_factory=workflow_contracts.ExecutionFrame
    )
    original_goal: str = ""
    active_family: str | None = None
    active_segment_id: str | None = None
    active_capability_id: str | None = None
    goal_intent_profile: dict[str, Any] = Field(default_factory=dict)
    questions: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    current_question: str | None = None
    current_question_field: str | None = None
    pending_fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    known_slot_values: dict[str, Any] = Field(default_factory=dict)
    resolved_slots: dict[str, Any] = Field(default_factory=dict)
    slot_provenance: dict[str, str] = Field(default_factory=dict)
    answered_fields: list[str] = Field(default_factory=list)
    question_history: list[str] = Field(default_factory=list)
    answer_history: list[str] = Field(default_factory=list)
    candidate_capabilities: list[str] = Field(default_factory=list)
    auto_path_allowed: bool = False

    @field_validator("candidate_capabilities", mode="before")
    @classmethod
    def _normalize_candidate_capabilities(cls, value: Any) -> list[str] | Any:
        if not isinstance(value, list):
            return value
        return workflow_contracts._canonicalize_capability_id_list(value)

    @field_validator("active_capability_id", mode="before")
    @classmethod
    def _normalize_active_capability_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        candidate = workflow_contracts._canonicalize_capability_id(value)
        return candidate or None

    @field_validator("slot_provenance", mode="before")
    @classmethod
    def _normalize_slot_provenance(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            candidate = str(raw_value or "").strip()
            if not candidate:
                continue
            try:
                normalized[key] = workflow_contracts.SlotProvenance(candidate).value
            except ValueError:
                normalized[key] = candidate
        return normalized

    @model_validator(mode="after")
    def _synchronize_state(self) -> "ClarificationState":
        if not self.pending_questions and self.questions:
            self.pending_questions = list(self.questions)
        if not self.current_question:
            if self.questions:
                self.current_question = str(self.questions[0]).strip() or None
            elif self.pending_questions:
                self.current_question = str(self.pending_questions[0]).strip() or None
        if not self.questions and self.current_question:
            self.questions = [self.current_question]
        elif self.questions and self.current_question:
            self.questions = [self.current_question]
        elif self.pending_questions:
            first = str(self.pending_questions[0]).strip()
            self.questions = [first] if first else []
        if not self.current_question_field:
            candidates = list(self.pending_fields or self.required_fields)
            if candidates:
                first_field = str(candidates[0]).strip()
                self.current_question_field = first_field or None
        if not self.known_slot_values and self.resolved_slots:
            self.known_slot_values = dict(self.resolved_slots)
        if not self.resolved_slots and self.known_slot_values:
            self.resolved_slots = dict(self.known_slot_values)
        if not self.answered_fields and self.known_slot_values:
            self.answered_fields = sorted(str(key) for key in self.known_slot_values.keys())
        if not self.required_fields and self.pending_fields:
            self.required_fields = list(self.pending_fields)
        if self.execution_frame.mode == workflow_contracts.ExecutionFrameMode.chat and (
            self.pending_questions or self.pending_fields
        ):
            self.execution_frame.mode = workflow_contracts.ExecutionFrameMode.clarification
        if not self.original_goal and self.execution_frame.original_goal:
            self.original_goal = self.execution_frame.original_goal
        if self.original_goal and not self.execution_frame.original_goal:
            self.execution_frame.original_goal = self.original_goal
        for attr in ("active_family", "active_segment_id", "active_capability_id"):
            current = getattr(self, attr)
            frame_value = getattr(self.execution_frame, attr)
            if not current and frame_value:
                setattr(self, attr, frame_value)
            elif current and not frame_value:
                setattr(self.execution_frame, attr, current)
        return self


class ClarificationPendingState(ClarificationState):
    pass


class ClarificationMappingRequest(BaseModel):
    original_goal: str = ""
    latest_answer: str = ""
    pending_state: ClarificationPendingState = Field(default_factory=ClarificationPendingState)
    context_json: dict[str, Any] = Field(default_factory=dict)


class ClarificationMappingResult(BaseModel):
    resolved_fields: list[ClarificationResolvedField] = Field(default_factory=list)
    remaining_fields: list[str] = Field(default_factory=list)
    confidence_by_field: dict[str, float] = Field(default_factory=dict)
    question_answer_map: dict[str, str] = Field(default_factory=dict)
    user_intent_changed: bool = False
    revised_goal_summary: str | None = None
    auto_path_allowed: bool = False


class AssistantActionType(str, Enum):
    respond = "respond"
    tool_call = "tool_call"
    ask_clarification = "ask_clarification"
    submit_job = "submit_job"
    run_workflow = "run_workflow"
    attach_to_job = "attach_to_job"
    summarize_job = "summarize_job"


class AssistantAction(BaseModel):
    type: AssistantActionType
    goal: str | None = None
    job_id: str | None = None
    workflow_run_id: str | None = None
    workflow_definition_id: str | None = None
    workflow_version_id: str | None = None
    workflow_trigger_id: str | None = None
    capability_id: str | None = None
    tool_name: str | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    goal_intent_profile: workflow_contracts.GoalIntentProfile = Field(
        default_factory=workflow_contracts.GoalIntentProfile
    )
    context_json: dict[str, Any] = Field(default_factory=dict)


class ChatBoundaryDecision(BaseModel):
    decision: ChatBoundaryDecisionType
    confidence: float | None = None
    assistant_response: str = ""
    reason_code: str | None = None
    evidence: ChatBoundaryEvidence | None = None


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: ChatRole
    content: str
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    action: AssistantAction | None = None
    job_id: str | None = None


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    active_job_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatSessionCreate(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatTurnRequest(BaseModel):
    content: str
    context_json: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0


class ChatTurnResponse(BaseModel):
    session: ChatSession
    user_message: ChatMessage
    assistant_message: ChatMessage
    job: models.Job | None = None
    workflow_run: models.WorkflowRun | None = None
