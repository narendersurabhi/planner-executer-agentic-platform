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

export type StudioPersistedWorkflowDraft = {
  summary?: string;
  goal?: string;
  contextJsonText?: string;
  nodePositions?: Record<string, CanvasPoint>;
  nodes?: ComposerDraftNode[];
  edges?: ComposerDraftEdge[];
  workflowInterface?: WorkflowInterface;
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
  user_id?: string | null;
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
