from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
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


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    target_type: Mapped[str] = mapped_column(String, index=True)
    target_id: Mapped[str] = mapped_column(String, index=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    plan_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    message_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    actor_key: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    sentiment: Mapped[str] = mapped_column(String, index=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason_codes_json: Mapped[List[str]] = mapped_column("reason_codes", JSON, default=list)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    snapshot_json: Mapped[Dict[str, Any]] = mapped_column("snapshot", JSON, default=dict)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


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


class AgentDefinitionRecord(Base):
    __tablename__ = "agent_definitions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_capability_id: Mapped[str] = mapped_column(String, index=True)
    instructions: Mapped[str] = mapped_column(Text)
    default_goal: Mapped[str] = mapped_column(String, default="")
    default_workspace_path: Mapped[str | None] = mapped_column(String, nullable=True)
    default_constraints_json: Mapped[List[str]] = mapped_column(
        "default_constraints", JSON, default=list
    )
    default_max_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_config_json: Mapped[Dict[str, Any]] = mapped_column(
        "model_config", JSON, default=dict
    )
    allowed_capability_ids_json: Mapped[List[str]] = mapped_column(
        "allowed_capability_ids", JSON, default=list
    )
    memory_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "memory_policy", JSON, default=dict
    )
    guardrail_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "guardrail_policy", JSON, default=dict
    )
    workspace_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "workspace_policy", JSON, default=dict
    )
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class AgentDefinitionVersionRecord(Base):
    __tablename__ = "agent_definition_versions"
    __table_args__ = (
        UniqueConstraint(
            "agent_definition_id",
            "version_number",
            name="uq_agent_definition_versions_definition_version",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_definition_id: Mapped[str] = mapped_column(
        ForeignKey("agent_definitions.id"),
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_capability_id: Mapped[str] = mapped_column(String, index=True)
    instructions: Mapped[str] = mapped_column(Text)
    default_goal: Mapped[str] = mapped_column(String, default="")
    default_workspace_path: Mapped[str | None] = mapped_column(String, nullable=True)
    default_constraints_json: Mapped[List[str]] = mapped_column(
        "default_constraints", JSON, default=list
    )
    default_max_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_config_json: Mapped[Dict[str, Any]] = mapped_column(
        "model_config", JSON, default=dict
    )
    allowed_capability_ids_json: Mapped[List[str]] = mapped_column(
        "allowed_capability_ids", JSON, default=list
    )
    memory_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "memory_policy", JSON, default=dict
    )
    guardrail_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "guardrail_policy", JSON, default=dict
    )
    workspace_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "workspace_policy", JSON, default=dict
    )
    definition_metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "definition_metadata", JSON, default=dict
    )
    version_metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        "version_metadata", JSON, default=dict
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    published_by: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    version_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    definition_created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    definition_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    goal: Mapped[str] = mapped_column(String, default="")
    requested_context_json: Mapped[Dict[str, Any]] = mapped_column(
        "requested_context", JSON, default=dict
    )
    status: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    workflow_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_runs.id"), nullable=True, index=True
    )
    plan_id: Mapped[str | None] = mapped_column(ForeignKey("plans.id"), nullable=True, index=True)
    source_definition_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_definitions.id"), nullable=True, index=True
    )
    source_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_versions.id"), nullable=True, index=True
    )
    source_trigger_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_triggers.id"), nullable=True, index=True
    )
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    run_spec_json: Mapped[Dict[str, Any]] = mapped_column("run_spec", JSON, default=dict)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


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


class RunStepRecord(Base):
    __tablename__ = "run_steps"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    plan_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    spec_step_id: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    instruction: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, index=True)
    intent: Mapped[str | None] = mapped_column(String, nullable=True)
    capability_request_id: Mapped[str] = mapped_column(String)
    execution_request_id: Mapped[str | None] = mapped_column(String, nullable=True)
    capability_id: Mapped[str] = mapped_column(String, index=True)
    input_bindings_json: Mapped[Dict[str, Any]] = mapped_column("input_bindings", JSON, default=dict)
    execution_gate_json: Mapped[Dict[str, Any] | None] = mapped_column(
        "execution_gate", JSON, nullable=True
    )
    retry_policy_json: Mapped[Dict[str, Any]] = mapped_column("retry_policy", JSON, default=dict)
    acceptance_policy_json: Mapped[Dict[str, Any]] = mapped_column(
        "acceptance_policy", JSON, default=dict
    )
    depends_on_json: Mapped[List[str]] = mapped_column("depends_on", JSON, default=list)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class TaskResultRecord(Base):
    __tablename__ = "task_results"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    plan_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, default="")
    result_json: Mapped[Dict[str, Any]] = mapped_column("result", JSON, default=dict)
    latest_error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class StepAttemptRecord(Base):
    __tablename__ = "step_attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    step_id: Mapped[str] = mapped_column(String, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String, default="")
    worker_id: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    retry_classification: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    heartbeat_count: Mapped[int] = mapped_column(Integer, default=0)
    result_summary_json: Mapped[Dict[str, Any]] = mapped_column("result_summary", JSON, default=dict)


class ExecutionRequestRecord(Base):
    __tablename__ = "execution_requests"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    step_id: Mapped[str] = mapped_column(String, index=True)
    request_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    capability_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    step_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("step_attempts.id"),
        nullable=True,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String, default="", index=True)
    request_json: Mapped[Dict[str, Any]] = mapped_column("request", JSON, default=dict)
    retry_policy_json: Mapped[Dict[str, Any]] = mapped_column("retry_policy", JSON, default=dict)
    policy_snapshot_json: Mapped[Dict[str, Any]] = mapped_column("policy_snapshot", JSON, default=dict)
    context_provenance_json: Mapped[Dict[str, Any]] = mapped_column(
        "context_provenance", JSON, default=dict
    )
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    retry_classification: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class StepCheckpointRecord(Base):
    __tablename__ = "step_checkpoints"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    step_id: Mapped[str] = mapped_column(String, index=True)
    step_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("step_attempts.id"),
        nullable=True,
        index=True,
    )
    checkpoint_key: Mapped[str] = mapped_column(String, index=True)
    payload_json: Mapped[Dict[str, Any]] = mapped_column("payload", JSON, default=dict)
    input_digest: Mapped[str | None] = mapped_column(String, nullable=True)
    replay_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class InvocationRecord(Base):
    __tablename__ = "invocations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    step_id: Mapped[str] = mapped_column(String, index=True)
    step_attempt_id: Mapped[str] = mapped_column(
        ForeignKey("step_attempts.id"),
        index=True,
    )
    request_id: Mapped[str | None] = mapped_column(String, nullable=True)
    capability_id: Mapped[str] = mapped_column(String)
    adapter_id: Mapped[str | None] = mapped_column(String, nullable=True)
    request_json: Mapped[Dict[str, Any]] = mapped_column("request", JSON, default=dict)
    response_json: Mapped[Dict[str, Any]] = mapped_column("response", JSON, default=dict)
    status: Mapped[str] = mapped_column(String, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)


class RunEventRecord(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    step_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    step_attempt_id: Mapped[str | None] = mapped_column(
        ForeignKey("step_attempts.id"),
        nullable=True,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String, index=True)
    payload_json: Mapped[Dict[str, Any]] = mapped_column("payload", JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)


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
