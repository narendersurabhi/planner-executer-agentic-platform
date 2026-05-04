"use client";

export type CapabilityAdapter = {
  type: string;
  server_id: string;
  tool_name: string;
};

export type CapabilitySchemaField = {
  path: string;
  type: string;
  required: boolean;
  description?: string | null;
};

export type CapabilityItem = {
  id: string;
  description: string;
  enabled: boolean;
  risk_tier: string;
  idempotency: string;
  group?: string | null;
  subgroup?: string | null;
  tags: string[];
  input_schema_ref?: string | null;
  input_schema?: Record<string, unknown> | null;
  output_schema_ref?: string | null;
  output_schema?: Record<string, unknown> | null;
  input_fields?: CapabilitySchemaField[];
  output_fields?: CapabilitySchemaField[];
  required_inputs?: string[];
  adapters?: CapabilityAdapter[];
  planner_hints?: Record<string, unknown> | null;
};

export type CapabilityCatalog = {
  mode: string;
  items: CapabilityItem[];
};

export type AgentDefinition = {
  id: string;
  name: string;
  description?: string | null;
  agent_capability_id: string;
  instructions: string;
  default_goal: string;
  default_workspace_path?: string | null;
  default_constraints: string[];
  default_max_steps?: number | null;
  model_config: Record<string, unknown>;
  allowed_capability_ids: string[];
  memory_policy: Record<string, unknown>;
  guardrail_policy: Record<string, unknown>;
  workspace_policy: Record<string, unknown>;
  enabled: boolean;
  user_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type AgentDefinitionCreateRequest = {
  name: string;
  description?: string | null;
  agent_capability_id: string;
  instructions: string;
  default_goal?: string;
  default_workspace_path?: string | null;
  default_constraints?: string[];
  default_max_steps?: number | null;
  model_config?: Record<string, unknown>;
  allowed_capability_ids?: string[];
  memory_policy?: Record<string, unknown>;
  guardrail_policy?: Record<string, unknown>;
  workspace_policy?: Record<string, unknown>;
  user_id?: string | null;
  metadata?: Record<string, unknown>;
};

export type AgentDefinitionUpdateRequest = Partial<
  AgentDefinitionCreateRequest & {
    enabled: boolean;
  }
>;

export type AgentDefinitionVersionPublishRequest = {
  version_note?: string | null;
  published_by?: string | null;
  metadata?: Record<string, unknown>;
};

export type AgentDefinitionVersion = {
  id: string;
  agent_definition_id: string;
  version_number: number;
  name: string;
  description?: string | null;
  agent_capability_id: string;
  instructions: string;
  default_goal: string;
  default_workspace_path?: string | null;
  default_constraints: string[];
  default_max_steps?: number | null;
  model_config: Record<string, unknown>;
  allowed_capability_ids: string[];
  memory_policy: Record<string, unknown>;
  guardrail_policy: Record<string, unknown>;
  workspace_policy: Record<string, unknown>;
  definition_metadata: Record<string, unknown>;
  version_metadata: Record<string, unknown>;
  enabled: boolean;
  user_id?: string | null;
  published_by?: string | null;
  version_note?: string | null;
  definition_created_at?: string | null;
  definition_updated_at?: string | null;
  created_at: string;
};

export type StudioSurface = "workflow" | "workbench";

export type ComposerInputBinding =
  | {
      kind: "step_output";
      sourceNodeId: string;
      sourcePath: string;
      defaultValue?: string;
    }
  | {
      kind: "workflow_input";
      inputKey: string;
      defaultValue?: string;
    }
  | {
      kind: "workflow_variable";
      variableKey: string;
      defaultValue?: string;
    }
  | {
      kind: "literal";
      value: string;
    }
  | {
      kind: "context";
      path: string;
    }
  | {
      kind: "memory";
      scope: "job" | "user" | "global";
      name: string;
      key?: string;
    };

export type StudioControlKind = "if" | "if_else" | "switch" | "parallel";

export type StudioControlCase = {
  id: string;
  label: string;
  match: string;
};

export type StudioControlConfig = {
  expression: string;
  trueLabel?: string;
  falseLabel?: string;
  parallelMode?: "fan_out" | "fan_in";
  switchCases?: StudioControlCase[];
};

export type ComposerDraftNode = {
  id: string;
  taskName: string;
  capabilityId: string;
  outputPath: string;
  nodeKind?: "capability" | "control";
  controlKind?: StudioControlKind | null;
  controlConfig?: StudioControlConfig | null;
  inputBindings: Record<string, ComposerInputBinding>;
  outputs: StudioNodeOutput[];
  variables: StudioNodeVariable[];
};

export type StudioNodeOutput = {
  id: string;
  name: string;
  path: string;
  description?: string;
};

export type StudioNodeVariable = {
  id: string;
  key: string;
  value: string;
  description?: string;
};

export type ComposerDraftEdge = {
  fromNodeId: string;
  toNodeId: string;
  branchLabel?: string;
};

export type ComposerDraft = {
  summary: string;
  nodes: ComposerDraftNode[];
  edges: ComposerDraftEdge[];
};

export type CanvasPoint = {
  x: number;
  y: number;
};

export type WorkflowBinding =
  | {
      kind: "literal";
      value: string;
    }
  | {
      kind: "context";
      path: string;
    }
  | {
      kind: "memory";
      scope: "job" | "user" | "global";
      name: string;
      key?: string;
    }
  | {
      kind: "secret";
      secretName: string;
    }
  | {
      kind: "workflow_input";
      inputKey: string;
    }
  | {
      kind: "workflow_variable";
      variableKey: string;
    }
  | {
      kind: "step_output";
      sourceNodeId: string;
      sourcePath: string;
    };

export type WorkflowInputDefinition = {
  id: string;
  key: string;
  label: string;
  valueType: "string" | "number" | "boolean" | "object" | "array";
  required: boolean;
  description?: string;
  defaultValue?: string;
  binding?: WorkflowBinding | null;
};

export type WorkflowVariableDefinition = {
  id: string;
  key: string;
  description?: string;
  binding?: WorkflowBinding | null;
};

export type WorkflowOutputDefinition = {
  id: string;
  key: string;
  label: string;
  description?: string;
  binding?: WorkflowBinding | null;
};

export type WorkflowInterface = {
  inputs: WorkflowInputDefinition[];
  variables: WorkflowVariableDefinition[];
  outputs: WorkflowOutputDefinition[];
};

export type WorkflowRuntimeSettings = {
  executionMode?: "static" | "adaptive";
  adaptivePolicy?: {
    maxReplans?: number;
  };
};

export type AdaptiveReplanStatus = {
  active_plan_id?: string | null;
  pending_replan?: boolean;
  pending_replan_reason?: string | null;
  max_replans?: number;
  replans_used?: number;
  replans_remaining?: number;
  can_manual_replan?: boolean;
  replan_block_reason?: string | null;
};

export type StudioPersistedWorkflowDraft = {
  summary?: string;
  goal?: string;
  contextJsonText?: string;
  nodePositions?: Record<string, CanvasPoint>;
  nodes?: ComposerDraftNode[];
  edges?: ComposerDraftEdge[];
  workflowInterface?: WorkflowInterface;
  runtimeSettings?: WorkflowRuntimeSettings;
};

export type WorkbenchConversionDiagnostic = {
  code: string;
  message: string;
  stepId?: string;
  field?: string;
};

export type ReplayableCapabilityDraft = {
  sourceRunId: string;
  sourceStepId: string;
  title: string;
  goal: string;
  userId: string;
  contextJson: Record<string, unknown>;
  capabilityId: string;
  inputs: Record<string, unknown>;
  retryPolicy: Record<string, unknown>;
  notice: string;
};

export type WorkbenchReplayResult =
  | {
      replayable: true;
      draft: ReplayableCapabilityDraft;
    }
  | {
      replayable: false;
      reason: string;
      diagnostics: WorkbenchConversionDiagnostic[];
    };

export type WorkbenchForkTargetMode =
  | "capability"
  | "agent_structured"
  | "agent_raw";

export type WorkbenchAgentStructuredStepDraft = {
  stepId: string;
  name: string;
  description: string;
  instruction: string;
  capabilityId: string;
  dependsOn: string[];
  inputBindings: Record<string, unknown>;
  retryPolicy: Record<string, unknown>;
};

export type WorkbenchAgentStructuredDraft = {
  sourceRunId: string;
  title: string;
  goal: string;
  userId: string;
  contextJson: Record<string, unknown>;
  steps: WorkbenchAgentStructuredStepDraft[];
  notice: string;
};

export type WorkbenchAgentRawDraft = {
  sourceRunId: string;
  title: string;
  goal: string;
  userId: string;
  contextJson: Record<string, unknown>;
  runSpec: Record<string, unknown>;
  reason: string;
  notice: string;
};

export type WorkbenchForkResult =
  | {
      mode: "capability";
      draft: ReplayableCapabilityDraft;
      diagnostics: WorkbenchConversionDiagnostic[];
    }
  | {
      mode: "agent_structured";
      draft: WorkbenchAgentStructuredDraft;
      diagnostics: WorkbenchConversionDiagnostic[];
    }
  | {
      mode: "agent_raw";
      draft: WorkbenchAgentRawDraft;
      diagnostics: WorkbenchConversionDiagnostic[];
    };

export type WorkbenchWorkflowPromotionDraft = StudioPersistedWorkflowDraft & {
  summary: string;
  goal: string;
  contextJsonText: string;
  nodePositions: Record<string, CanvasPoint>;
  nodes: ComposerDraftNode[];
  edges: ComposerDraftEdge[];
  workflowInterface: WorkflowInterface;
  sourceRunId: string;
  sourceTitle: string;
  notice: string;
};

export type WorkbenchWorkflowPromotionResult =
  | {
      promotable: true;
      draft: WorkbenchWorkflowPromotionDraft;
      diagnostics: WorkbenchConversionDiagnostic[];
    }
  | {
      promotable: false;
      reason: string;
      diagnostics: WorkbenchConversionDiagnostic[];
    };

export type ComposerCompileDiagnostic = {
  code: string;
  message: string;
  field?: string;
  node_id?: string;
};

export type ComposerCompileResponse = {
  valid: boolean;
  diagnostics: {
    valid: boolean;
    errors: ComposerCompileDiagnostic[];
    warnings: ComposerCompileDiagnostic[];
  };
  plan: Record<string, unknown> | null;
  run_spec?: Record<string, unknown> | null;
  preflight_errors: Record<string, string>;
};

export type ChainPreflightResult = {
  valid: boolean;
  localErrors: string[];
  serverErrors: Record<string, string>;
  serverDiagnostics?: {
    severity?: "error" | "warning";
    code: string;
    field?: string;
    message: string;
    slot_fields?: string[];
  }[];
  checkedAt: string;
};

export type WorkflowDefinition = {
  id: string;
  title: string;
  goal: string;
  context_json: Record<string, unknown>;
  draft: StudioPersistedWorkflowDraft;
  user_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkflowVersion = {
  id: string;
  definition_id: string;
  version_number: number;
  title: string;
  goal: string;
  context_json: Record<string, unknown>;
  draft: StudioPersistedWorkflowDraft;
  compiled_plan: Record<string, unknown>;
  run_spec?: Record<string, unknown>;
  user_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type WorkflowTrigger = {
  id: string;
  definition_id: string;
  title: string;
  trigger_type: "manual" | "api" | "webhook" | "schedule";
  enabled: boolean;
  config: Record<string, unknown>;
  user_id?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkflowRun = {
  id: string;
  definition_id: string;
  version_id: string;
  trigger_id?: string | null;
  title: string;
  goal: string;
  requested_context_json: Record<string, unknown>;
  job_id: string;
  plan_id: string;
  job_status?: string | null;
  job_error?: string | null;
  latest_task_id?: string | null;
  latest_task_name?: string | null;
  latest_task_error?: string | null;
  user_id?: string | null;
  planning_mode?: "static" | "adaptive" | string;
  current_revision_number?: number;
  adaptive_status?: AdaptiveReplanStatus;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkflowRunResult = {
  workflow_definition: WorkflowDefinition;
  workflow_version: WorkflowVersion;
  workflow_run: WorkflowRun;
  job: {
    id: string;
    goal: string;
    status: string;
    planning_mode?: "static" | "adaptive" | string;
    current_revision_number?: number;
    adaptive_status?: AdaptiveReplanStatus;
    metadata?: Record<string, unknown>;
  };
  plan: {
    id: string;
    job_id: string;
    planner_version: string;
  };
};

export type ComposerValidationIssue = {
  severity: "error" | "warning";
  source: "local" | "compile" | "preflight";
  code: string;
  message: string;
  field?: string;
  nodeId?: string;
};

export type ComposerIssueFocus = {
  nodeId: string;
  field?: string;
};

export type NodeRequiredStatus = {
  field: string;
  status: "missing" | "from_chain" | "from_context" | "provided";
  detail: string;
  schemaType: string;
  schemaDescription: string;
};
