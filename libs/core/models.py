from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    planning = "planning"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


class TaskStatus(str, Enum):
    pending = "pending"
    ready = "ready"
    running = "running"
    blocked = "blocked"
    completed = "completed"
    accepted = "accepted"
    rework_requested = "rework_requested"
    failed = "failed"
    canceled = "canceled"


class PolicyDecisionType(str, Enum):
    allow = "allow"
    deny = "deny"
    rewrite = "rewrite"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ToolIntent(str, Enum):
    transform = "transform"
    generate = "generate"
    validate = "validate"
    render = "render"
    io = "io"


class MemoryScope(str, Enum):
    request = "request"
    session = "session"
    user = "user"
    project = "project"
    global_ = "global"


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    usage_guidance: Optional[str] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    auth_required: bool = False
    timeout_s: int = 30
    risk_level: RiskLevel = RiskLevel.low
    tool_intent: ToolIntent = ToolIntent.transform
    memory_reads: List[str] = Field(default_factory=list)
    memory_writes: List[str] = Field(default_factory=list)


class MemorySpec(BaseModel):
    name: str
    description: str
    scope: MemoryScope
    schema_def: Dict[str, Any] = Field(default_factory=dict)
    ttl_seconds: Optional[int] = None
    version: str = "1.0"
    read_roles: List[str] = Field(default_factory=list)
    write_roles: List[str] = Field(default_factory=list)


class MemoryWrite(BaseModel):
    name: str
    payload: Dict[str, Any]
    scope: Optional[MemoryScope] = None
    key: Optional[str] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    if_match_updated_at: Optional[datetime] = None


class MemoryEntry(BaseModel):
    id: str
    name: str
    scope: MemoryScope
    payload: Dict[str, Any]
    key: Optional[str] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None


class MemoryQuery(BaseModel):
    name: str
    scope: Optional[MemoryScope] = None
    key: Optional[str] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    limit: int = 50
    include_expired: bool = False


class TaskDlqEntry(BaseModel):
    stream_id: str
    message_id: str
    failed_at: Optional[str] = None
    error: str
    worker_consumer: Optional[str] = None
    job_id: Optional[str] = None
    task_id: Optional[str] = None
    envelope: Dict[str, Any] = Field(default_factory=dict)
    task_payload: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool_name: str
    input: Dict[str, Any]
    idempotency_key: str
    trace_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    status: str
    output_or_error: Dict[str, Any]


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    outputs: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    tool_calls: List[ToolCall]
    started_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class CriticResult(BaseModel):
    task_id: str
    decision: str
    reasons: List[str]
    feedback: Optional[str] = None
    checked_at: datetime


class PolicyDecision(BaseModel):
    scope: str
    decision: PolicyDecisionType
    reasons: List[str]
    rewrites: Optional[Dict[str, Any]] = None
    decided_at: datetime


class EventEnvelope(BaseModel):
    type: str
    version: str
    occurred_at: datetime
    correlation_id: str
    job_id: Optional[str] = None
    task_id: Optional[str] = None
    payload: Dict[str, Any]


class Job(BaseModel):
    id: str
    goal: str
    context_json: Dict[str, Any] = Field(default_factory=dict)
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    id: str
    job_id: str
    planner_version: str
    created_at: datetime
    tasks_summary: str
    dag_edges: List[List[str]]
    policy_decision: Optional[PolicyDecision] = None


class Task(BaseModel):
    id: str
    job_id: str
    plan_id: str
    name: str
    description: str
    instruction: str
    acceptance_criteria: List[str]
    expected_output_schema_ref: str
    status: TaskStatus
    intent: Optional[ToolIntent] = None
    deps: List[str]
    attempts: int
    max_attempts: int
    rework_count: int
    max_reworks: int
    assigned_to: Optional[str] = None
    tool_requests: List[str]
    tool_inputs: Dict[str, Any] = Field(default_factory=dict)
    tool_inputs_resolved: bool = False
    created_at: datetime
    updated_at: datetime
    critic_required: bool = True


class JobDetails(BaseModel):
    job_id: str
    plan: Optional[Plan] = None
    tasks: List[Task] = Field(default_factory=list)
    task_results: Dict[str, Any] = Field(default_factory=dict)


class TaskCreate(BaseModel):
    name: str
    description: str
    instruction: str
    acceptance_criteria: List[str]
    expected_output_schema_ref: str
    intent: Optional[ToolIntent] = None
    deps: List[str]
    tool_requests: List[str]
    tool_inputs: Dict[str, Any] = Field(default_factory=dict)
    critic_required: bool = True


class PlanCreate(BaseModel):
    planner_version: str
    tasks_summary: str
    dag_edges: List[List[str]]
    tasks: List[TaskCreate]


class JobCreate(BaseModel):
    goal: str
    context_json: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    idempotency_key: Optional[str] = None


class JobUpdate(BaseModel):
    status: JobStatus


class TaskUpdate(BaseModel):
    status: TaskStatus
    assigned_to: Optional[str] = None
    attempts: Optional[int] = None
    rework_count: Optional[int] = None
