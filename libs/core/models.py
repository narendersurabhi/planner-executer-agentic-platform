from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class MemoryCandidateType(str, Enum):
    user_profile_update = "user_profile_update"
    semantic_fact = "semantic_fact"
    interaction_summary = "interaction_summary"
    task_pattern = "task_pattern"


class MemorySensitivity(str, Enum):
    low = "low"
    restricted = "restricted"


class WorkflowTriggerType(str, Enum):
    manual = "manual"
    api = "api"
    webhook = "webhook"
    schedule = "schedule"


class RunKind(str, Enum):
    planner = "planner"
    studio = "studio"
    chat_direct = "chat_direct"
    trigger = "trigger"
    api = "api"


class FeedbackTargetType(str, Enum):
    chat_message = "chat_message"
    intent_assessment = "intent_assessment"
    plan = "plan"
    job_outcome = "job_outcome"


class FeedbackSentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    partial = "partial"


class FeedbackExampleFormat(str, Enum):
    json = "json"
    jsonl = "jsonl"


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    aliases: List[str] = Field(default_factory=list)
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


class UserProfilePreferences(BaseModel):
    preferred_output_format: Optional[str] = None
    response_verbosity: Optional[str] = None


class UserProfilePayload(BaseModel):
    preferences: UserProfilePreferences = Field(default_factory=UserProfilePreferences)
    updated_at: Optional[str] = None


class MemoryPromotionDecision(BaseModel):
    candidate_type: MemoryCandidateType
    accepted: bool
    reason: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    sensitivity: MemorySensitivity = MemorySensitivity.low
    indexable: bool = True


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
    request_id: Optional[str] = None
    capability_id: Optional[str] = None
    adapter_id: Optional[str] = None
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
    intent_source: Optional[str] = None
    intent_confidence: Optional[float] = None
    deps: List[str]
    attempts: int
    max_attempts: int
    rework_count: int
    max_reworks: int
    assigned_to: Optional[str] = None
    tool_requests: List[str]
    tool_inputs: Dict[str, Any] = Field(default_factory=dict)
    capability_bindings: Dict[str, Any] = Field(default_factory=dict)
    tool_inputs_resolved: bool = False
    created_at: datetime
    updated_at: datetime
    critic_required: bool = True


class JobDetails(BaseModel):
    job_id: str
    job_status: Optional[JobStatus] = None
    job_error: Optional[str] = None
    plan: Optional[Plan] = None
    tasks: List[Task] = Field(default_factory=list)
    task_results: Dict[str, Any] = Field(default_factory=dict)
    goal_intent_profile: Dict[str, Any] = Field(default_factory=dict)
    goal_intent_graph: Optional[Dict[str, Any]] = None
    normalized_intent_envelope: Dict[str, Any] = Field(default_factory=dict)
    normalization_trace: Dict[str, Any] = Field(default_factory=dict)
    normalization_clarification: Dict[str, Any] = Field(default_factory=dict)
    normalization_candidate_capabilities: Dict[str, List[str]] = Field(default_factory=dict)


class TaskCreate(BaseModel):
    name: str
    description: str
    instruction: str
    acceptance_criteria: List[str]
    expected_output_schema_ref: str
    intent: Optional[ToolIntent] = None
    deps: List[str]
    capability_requests: List[str] = Field(default_factory=list)
    tool_requests: List[str] = Field(default_factory=list)
    tool_inputs: Dict[str, Any] = Field(default_factory=dict)
    capability_bindings: Dict[str, Any] = Field(default_factory=dict)
    critic_required: bool = True


class PlanCreate(BaseModel):
    planner_version: str
    tasks_summary: str
    dag_edges: List[List[str]]
    tasks: List[TaskCreate]


class CapabilityRequestSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    capability_id: str
    execution_request_id: Optional[str] = None

    @model_validator(mode="after")
    def _normalize_request_ids(self) -> "CapabilityRequestSpec":
        request_id = str(self.request_id or "").strip()
        capability_id = str(self.capability_id or "").strip()
        execution_request_id = str(self.execution_request_id or "").strip()

        if not capability_id:
            capability_id = request_id or execution_request_id
        if not request_id:
            request_id = capability_id or execution_request_id

        if not execution_request_id:
            if capability_id and request_id and capability_id != request_id:
                execution_request_id = request_id
                request_id = capability_id
            else:
                execution_request_id = request_id or capability_id

        if not capability_id:
            capability_id = request_id or execution_request_id
        if not request_id:
            request_id = capability_id or execution_request_id

        self.request_id = request_id
        self.capability_id = capability_id
        self.execution_request_id = execution_request_id or None
        return self


class StepRetryPolicy(BaseModel):
    max_attempts: int = 1


class StepAcceptancePolicy(BaseModel):
    acceptance_criteria: List[str] = Field(default_factory=list)
    critic_required: bool = True


class StepSpec(BaseModel):
    step_id: str
    name: str
    description: str
    instruction: str = ""
    intent: Optional[ToolIntent] = None
    capability_request: CapabilityRequestSpec
    input_bindings: Dict[str, Any] = Field(default_factory=dict)
    execution_gate: Optional[Dict[str, Any]] = None
    expected_output_schema_ref: str = ""
    retry_policy: StepRetryPolicy = Field(default_factory=StepRetryPolicy)
    acceptance_policy: StepAcceptancePolicy = Field(default_factory=StepAcceptancePolicy)
    depends_on: List[str] = Field(default_factory=list)
    routing_hints: Dict[str, Any] = Field(default_factory=dict)


class RunSpec(BaseModel):
    version: str = "1"
    kind: RunKind = RunKind.studio
    planner_version: str = ""
    tasks_summary: str = ""
    steps: List[StepSpec] = Field(default_factory=list)
    dag_edges: List[List[str]] = Field(default_factory=list)
    capability_requests: List[CapabilityRequestSpec] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StepAttempt(BaseModel):
    id: str
    run_id: str
    job_id: str
    step_id: str
    attempt_number: int
    status: str
    worker_id: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_classification: Optional[str] = None
    result_summary: Dict[str, Any] = Field(default_factory=dict)


class Invocation(BaseModel):
    id: str
    run_id: str
    job_id: str
    step_id: str
    step_attempt_id: str
    request_id: Optional[str] = None
    capability_id: str
    adapter_id: Optional[str] = None
    request: Dict[str, Any] = Field(default_factory=dict)
    response: Dict[str, Any] = Field(default_factory=dict)
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class RunEvent(BaseModel):
    id: str
    run_id: str
    job_id: str
    step_id: Optional[str] = None
    step_attempt_id: Optional[str] = None
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime


class FeedbackCreate(BaseModel):
    target_type: FeedbackTargetType
    target_id: str
    sentiment: FeedbackSentiment
    score: Optional[int] = Field(default=None, ge=0, le=5)
    reason_codes: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Feedback(BaseModel):
    id: str
    target_type: FeedbackTargetType
    target_id: str
    session_id: Optional[str] = None
    job_id: Optional[str] = None
    plan_id: Optional[str] = None
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    actor_key: Optional[str] = None
    sentiment: FeedbackSentiment
    score: Optional[int] = None
    reason_codes: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    snapshot: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class FeedbackSummary(BaseModel):
    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    partial: int = 0


class FeedbackListResponse(BaseModel):
    items: List[Feedback] = Field(default_factory=list)
    summary: FeedbackSummary = Field(default_factory=FeedbackSummary)


class FeedbackBreakdownBucket(BaseModel):
    key: str
    total: int = 0
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    partial: int = 0


class FeedbackReasonBucket(BaseModel):
    reason_code: str
    count: int = 0


class FeedbackCorrelationSummary(BaseModel):
    job_count: int = 0
    replan_count: int = 0
    retry_count: int = 0
    failed_task_count: int = 0
    plan_failure_count: int = 0
    clarification_turn_count: int = 0
    terminal_statuses: List[FeedbackBreakdownBucket] = Field(default_factory=list)


class FeedbackSummaryRequest(BaseModel):
    target_type: Optional[FeedbackTargetType] = None
    sentiment: Optional[FeedbackSentiment] = None
    workflow_source: Optional[str] = None
    llm_model: Optional[str] = None
    planner_version: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: int = Field(default=500, ge=1, le=5000)


class FeedbackSummaryResponse(BaseModel):
    total: int = 0
    sentiment_counts: FeedbackSummary = Field(default_factory=FeedbackSummary)
    target_type_counts: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    negative_reasons: List[FeedbackReasonBucket] = Field(default_factory=list)
    workflow_sources: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    llm_models: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    planner_versions: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    job_statuses: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    assistant_action_types: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_assessment_intents: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_assessment_sources: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_clarification_modes: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_disagreement_reasons: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    boundary_decisions: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    boundary_reason_codes: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    boundary_top_families: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_active_families: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_slot_loss_states: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_family_alignments: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_mapping_resolved_active_field_states: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_mapping_queue_advancement_states: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    clarification_mapping_restart_states: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    correlates: FeedbackCorrelationSummary = Field(default_factory=FeedbackCorrelationSummary)


class FeedbackExample(BaseModel):
    feedback: Feedback
    snapshot: Dict[str, Any] = Field(default_factory=dict)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    linked_ids: Dict[str, Optional[str]] = Field(default_factory=dict)


class FeedbackExampleExportResponse(BaseModel):
    total: int = 0
    items: List[FeedbackExample] = Field(default_factory=list)


class ChatBoundaryReviewQueueItem(BaseModel):
    feedback: Feedback
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    linked_ids: Dict[str, Optional[str]] = Field(default_factory=dict)
    review_label: str
    review_score: int = 0
    excerpt: Optional[str] = None


class ChatBoundaryReviewQueueResponse(BaseModel):
    total: int = 0
    items: List[ChatBoundaryReviewQueueItem] = Field(default_factory=list)


class ChatClarificationReviewQueueItem(BaseModel):
    feedback: Feedback
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    linked_ids: Dict[str, Optional[str]] = Field(default_factory=dict)
    review_label: str
    review_score: int = 0
    excerpt: Optional[str] = None


class ChatClarificationReviewQueueResponse(BaseModel):
    total: int = 0
    items: List[ChatClarificationReviewQueueItem] = Field(default_factory=list)


class IntentReviewQueueItem(BaseModel):
    feedback: Feedback
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    linked_ids: Dict[str, Optional[str]] = Field(default_factory=dict)
    review_label: str
    review_score: int = 0
    excerpt: Optional[str] = None


class IntentReviewQueueResponse(BaseModel):
    total: int = 0
    items: List[IntentReviewQueueItem] = Field(default_factory=list)


class IntentTuningReportResponse(BaseModel):
    total: int = 0
    review_labels: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    tuning_focuses: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_assessment_intents: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_assessment_sources: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_clarification_modes: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_disagreement_reasons: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_top_capabilities: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    intent_top_families: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    missing_input_counts: List[FeedbackBreakdownBucket] = Field(default_factory=list)
    negative_reasons: List[FeedbackReasonBucket] = Field(default_factory=list)


class IntentTuningCandidate(BaseModel):
    feedback: Feedback
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    linked_ids: Dict[str, Optional[str]] = Field(default_factory=dict)
    review_label: str
    review_score: int = 0
    excerpt: Optional[str] = None
    tuning_focus: str
    suggested_case_id: str
    observed_case: Dict[str, Any] = Field(default_factory=dict)
    gold_case_stub: Dict[str, Any] = Field(default_factory=dict)


class IntentTuningCandidateExportResponse(BaseModel):
    total: int = 0
    items: List[IntentTuningCandidate] = Field(default_factory=list)


class JobCreate(BaseModel):
    goal: str
    context_json: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    idempotency_key: Optional[str] = None


class WorkflowDefinitionCreate(BaseModel):
    title: str
    goal: str = ""
    context_json: Dict[str, Any] = Field(default_factory=dict)
    draft: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinitionUpdate(BaseModel):
    title: Optional[str] = None
    goal: Optional[str] = None
    context_json: Optional[Dict[str, Any]] = None
    draft: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowDefinition(BaseModel):
    id: str
    title: str
    goal: str = ""
    context_json: Dict[str, Any] = Field(default_factory=dict)
    draft: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class WorkflowVersion(BaseModel):
    id: str
    definition_id: str
    version_number: int
    title: str
    goal: str = ""
    context_json: Dict[str, Any] = Field(default_factory=dict)
    draft: Dict[str, Any] = Field(default_factory=dict)
    compiled_plan: Dict[str, Any] = Field(default_factory=dict)
    run_spec: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class WorkflowTriggerCreate(BaseModel):
    title: str
    trigger_type: WorkflowTriggerType = WorkflowTriggerType.manual
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowTriggerUpdate(BaseModel):
    title: Optional[str] = None
    enabled: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowTrigger(BaseModel):
    id: str
    definition_id: str
    title: str
    trigger_type: WorkflowTriggerType
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class WorkflowRun(BaseModel):
    id: str
    definition_id: str
    version_id: str
    trigger_id: Optional[str] = None
    title: str
    goal: str = ""
    requested_context_json: Dict[str, Any] = Field(default_factory=dict)
    job_id: str
    plan_id: str
    job_status: Optional[JobStatus] = None
    job_error: Optional[str] = None
    latest_task_id: Optional[str] = None
    latest_task_name: Optional[str] = None
    latest_task_error: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class WorkflowRunResult(BaseModel):
    workflow_definition: WorkflowDefinition
    workflow_version: WorkflowVersion
    workflow_run: WorkflowRun
    job: Job
    plan: Plan


class JobUpdate(BaseModel):
    status: JobStatus


class TaskUpdate(BaseModel):
    status: TaskStatus
    assigned_to: Optional[str] = None
    attempts: Optional[int] = None
    rework_count: Optional[int] = None
