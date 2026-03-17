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


class AssistantActionType(str, Enum):
    respond = "respond"
    tool_call = "tool_call"
    ask_clarification = "ask_clarification"
    submit_job = "submit_job"
    attach_to_job = "attach_to_job"
    summarize_job = "summarize_job"


class AssistantAction(BaseModel):
    type: AssistantActionType
    goal: str | None = None
    job_id: str | None = None
    capability_id: str | None = None
    tool_name: str | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    goal_intent_profile: workflow_contracts.GoalIntentProfile = Field(
        default_factory=workflow_contracts.GoalIntentProfile
    )
    context_json: dict[str, Any] = Field(default_factory=dict)


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
