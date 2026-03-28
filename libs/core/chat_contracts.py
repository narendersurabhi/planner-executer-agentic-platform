from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    top_capability_score: float = 0.0
    top_family_score: float = 0.0
    family_concentration: float = 0.0
    execution_signal_strength: str = ""
    top_capabilities: list[ChatBoundaryCapabilityEvidence] = Field(default_factory=list)
    top_families: list[ChatBoundaryFamilyEvidence] = Field(default_factory=list)


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
