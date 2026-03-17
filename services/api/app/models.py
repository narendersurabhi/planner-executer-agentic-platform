from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    goal: Mapped[str] = mapped_column(String)
    context_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    plan: Mapped["PlanRecord"] = relationship("PlanRecord", back_populates="job", uselist=False)


class ChatSessionRecord(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    messages: Mapped[List["ChatMessageRecord"]] = relationship(
        "ChatMessageRecord",
        back_populates="session",
    )


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String, index=True)
    content: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    action_json: Mapped[Dict[str, Any] | None] = mapped_column("action", JSON, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    session: Mapped[ChatSessionRecord] = relationship("ChatSessionRecord", back_populates="messages")


class WorkflowDefinitionRecord(Base):
    __tablename__ = "workflow_definitions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    goal: Mapped[str] = mapped_column(String, default="")
    context_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    draft_json: Mapped[Dict[str, Any]] = mapped_column("draft", JSON, default=dict)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    versions: Mapped[List["WorkflowVersionRecord"]] = relationship(
        "WorkflowVersionRecord",
        back_populates="definition",
        cascade="all, delete-orphan",
    )


class WorkflowVersionRecord(Base):
    __tablename__ = "workflow_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    definition_id: Mapped[str] = mapped_column(ForeignKey("workflow_definitions.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String)
    goal: Mapped[str] = mapped_column(String, default="")
    context_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    draft_json: Mapped[Dict[str, Any]] = mapped_column("draft", JSON, default=dict)
    compiled_plan_json: Mapped[Dict[str, Any]] = mapped_column("compiled_plan", JSON, default=dict)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    definition: Mapped[WorkflowDefinitionRecord] = relationship(
        "WorkflowDefinitionRecord",
        back_populates="versions",
    )


class WorkflowTriggerRecord(Base):
    __tablename__ = "workflow_triggers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    definition_id: Mapped[str] = mapped_column(ForeignKey("workflow_definitions.id"), index=True)
    title: Mapped[str] = mapped_column(String)
    trigger_type: Mapped[str] = mapped_column(String)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config_json: Mapped[Dict[str, Any]] = mapped_column("config", JSON, default=dict)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class WorkflowRunRecord(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    definition_id: Mapped[str] = mapped_column(ForeignKey("workflow_definitions.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("workflow_versions.id"), index=True)
    trigger_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_triggers.id"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String)
    goal: Mapped[str] = mapped_column(String, default="")
    requested_context_json: Mapped[Dict[str, Any]] = mapped_column(
        "requested_context", JSON, default=dict
    )
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"), index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class PlanRecord(Base):
    __tablename__ = "plans"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    planner_version: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    tasks_summary: Mapped[str] = mapped_column(String)
    dag_edges: Mapped[List[List[str]]] = mapped_column(JSON, default=list)
    policy_decision: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    job: Mapped[JobRecord] = relationship("JobRecord", back_populates="plan")
    tasks: Mapped[List["TaskRecord"]] = relationship("TaskRecord", back_populates="plan")


class TaskRecord(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    instruction: Mapped[str] = mapped_column(String)
    acceptance_criteria: Mapped[List[str]] = mapped_column(JSON, default=list)
    expected_output_schema_ref: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    deps: Mapped[List[str]] = mapped_column(JSON, default=list)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    rework_count: Mapped[int] = mapped_column(Integer, default=0)
    max_reworks: Mapped[int] = mapped_column(Integer, default=2)
    assigned_to: Mapped[str | None] = mapped_column(String, nullable=True)
    intent: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_requests: Mapped[List[str]] = mapped_column(JSON, default=list)
    tool_inputs: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    critic_required: Mapped[bool] = mapped_column(Integer, default=1)

    plan: Mapped[PlanRecord] = relationship("PlanRecord", back_populates="tasks")


class EventOutboxRecord(Base):
    __tablename__ = "event_outbox"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    stream: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    envelope_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)


class MemoryRecord(Base):
    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    scope: Mapped[str] = mapped_column(String, index=True)
    key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    project_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    version: Mapped[str] = mapped_column(String, default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
