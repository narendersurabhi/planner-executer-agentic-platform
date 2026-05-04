"use client";

import type { CanonicalRun, WorkbenchDebuggerData, WorkbenchDebuggerStep } from "./studioApi";
import type {
  ComposerDraftEdge,
  ComposerDraftNode,
  ComposerInputBinding,
  ReplayableCapabilityDraft,
  WorkbenchAgentRawDraft,
  WorkbenchAgentStructuredDraft,
  WorkbenchConversionDiagnostic,
  WorkbenchForkResult,
  WorkbenchReplayResult,
  WorkbenchWorkflowPromotionDraft,
  WorkbenchWorkflowPromotionResult,
  WorkflowInterface,
} from "./types";
import {
  inferCapabilityOutputPath,
  normalizeComposerEdges,
  taskNameFromCapability,
  uniqueTaskName,
} from "./utils";

type NormalizedRunSpecStep = {
  stepId: string;
  name: string;
  description: string;
  instruction: string;
  capabilityId: string;
  inputBindings: Record<string, unknown>;
  retryPolicy: Record<string, unknown>;
  dependsOn: string[];
  executionGate: Record<string, unknown> | null;
};

type NormalizedRunSpec = {
  raw: Record<string, unknown>;
  metadata: Record<string, unknown>;
  tasksSummary: string;
  steps: NormalizedRunSpecStep[];
};

const EMPTY_WORKFLOW_INTERFACE: WorkflowInterface = {
  inputs: [],
  variables: [],
  outputs: [],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => stringValue(item)).filter(Boolean);
}

function cloneJsonValue<T>(value: T): T {
  try {
    return JSON.parse(JSON.stringify(value)) as T;
  } catch {
    return value;
  }
}

function cloneRecord(value: Record<string, unknown> | null | undefined): Record<string, unknown> {
  return value ? cloneJsonValue(value) : {};
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return "{}";
  }
}

function normalizeRunSpec(run: CanonicalRun): NormalizedRunSpec {
  const raw = isRecord(run.run_spec) ? cloneRecord(run.run_spec) : {};
  const metadata = isRecord(raw.metadata) ? cloneRecord(raw.metadata) : {};
  const rawSteps = Array.isArray(raw.steps) ? raw.steps : [];
  const steps = rawSteps.reduce<NormalizedRunSpecStep[]>((items, rawStep, index) => {
    if (!isRecord(rawStep)) {
      return items;
    }
    const capabilityRequest = isRecord(rawStep.capability_request)
      ? rawStep.capability_request
      : {};
    const stepId =
      stringValue(rawStep.step_id) ||
      stringValue(rawStep.id) ||
      `step_${index + 1}`;
    const capabilityId =
      stringValue(capabilityRequest.capability_id) ||
      stringValue(rawStep.capability_id);
    items.push({
      stepId,
      name: stringValue(rawStep.name) || stepId,
      description: stringValue(rawStep.description),
      instruction: stringValue(rawStep.instruction),
      capabilityId,
      inputBindings: isRecord(rawStep.input_bindings)
        ? cloneRecord(rawStep.input_bindings)
        : {},
      retryPolicy: isRecord(rawStep.retry_policy)
        ? cloneRecord(rawStep.retry_policy)
        : {},
      dependsOn: stringArray(rawStep.depends_on),
      executionGate: isRecord(rawStep.execution_gate)
        ? cloneRecord(rawStep.execution_gate)
        : null,
    });
    return items;
  }, []);
  return {
    raw,
    metadata,
    tasksSummary: stringValue(raw.tasks_summary),
    steps,
  };
}

function workbenchModeForRun(run: CanonicalRun, normalizedRunSpec: NormalizedRunSpec): "capability" | "agent" {
  const explicitMode =
    stringValue(run.metadata?.workbench_mode) ||
    stringValue(normalizedRunSpec.metadata.workbench_mode);
  if (explicitMode === "capability" || explicitMode === "agent") {
    return explicitMode;
  }
  return normalizedRunSpec.steps.length <= 1 ? "capability" : "agent";
}

function runTitle(run: CanonicalRun, normalizedRunSpec: NormalizedRunSpec): string {
  return run.title.trim() || run.goal.trim() || normalizedRunSpec.tasksSummary || "Workbench run";
}

function runGoal(run: CanonicalRun, normalizedRunSpec: NormalizedRunSpec): string {
  return run.goal.trim() || normalizedRunSpec.tasksSummary || runTitle(run, normalizedRunSpec);
}

function runContext(run: CanonicalRun): Record<string, unknown> {
  return isRecord(run.requested_context_json) ? cloneRecord(run.requested_context_json) : {};
}

function runUserId(run: CanonicalRun): string {
  return stringValue(run.user_id);
}

function isExecutableCapabilityStep(step: NormalizedRunSpecStep): boolean {
  const capabilityId = step.capabilityId.toLowerCase();
  if (!capabilityId) {
    return false;
  }
  return !capabilityId.startsWith("studio.control.") && !capabilityId.startsWith("workflow.control");
}

function matchNormalizedStep(
  debuggerData: WorkbenchDebuggerData,
  stepPayload: WorkbenchDebuggerStep
): NormalizedRunSpecStep | null {
  const normalizedRunSpec = normalizeRunSpec(debuggerData.run);
  const specStepId = stringValue(stepPayload.step.spec_step_id);
  const taskStepId = stringValue(stepPayload.step.id);
  return (
    normalizedRunSpec.steps.find((step) => step.stepId === specStepId) ||
    normalizedRunSpec.steps.find((step) => step.stepId === taskStepId) ||
    normalizedRunSpec.steps.find((step) => step.capabilityId === stepPayload.step.capability_id) ||
    null
  );
}

function firstResolvedInputs(requests: Record<string, unknown>[]): Record<string, unknown> | null {
  for (const request of requests) {
    const requestPayload = isRecord(request.request) ? request.request : null;
    if (!requestPayload) {
      continue;
    }
    const payloadRequests = Array.isArray(requestPayload.requests) ? requestPayload.requests : [];
    for (const payloadStep of payloadRequests) {
      if (!isRecord(payloadStep)) {
        continue;
      }
      if (isRecord(payloadStep.resolved_inputs)) {
        return cloneRecord(payloadStep.resolved_inputs);
      }
    }
    if (isRecord(requestPayload.tool_inputs)) {
      const preferredRequestId = stringValue(request.request_id);
      if (preferredRequestId && isRecord(requestPayload.tool_inputs[preferredRequestId])) {
        return cloneRecord(requestPayload.tool_inputs[preferredRequestId] as Record<string, unknown>);
      }
      const firstInput = Object.values(requestPayload.tool_inputs).find((value) => isRecord(value));
      if (isRecord(firstInput)) {
        return cloneRecord(firstInput);
      }
    }
  }
  return null;
}

function replayDraftFromSources(
  run: CanonicalRun,
  normalizedRunSpec: NormalizedRunSpec,
  sourceStep: {
    stepId: string;
    stepName: string;
    capabilityId: string;
    inputBindings: Record<string, unknown>;
    retryPolicy: Record<string, unknown>;
  }
): ReplayableCapabilityDraft {
  return {
    sourceRunId: run.id,
    sourceStepId: sourceStep.stepId,
    title: runTitle(run, normalizedRunSpec),
    goal: runGoal(run, normalizedRunSpec),
    userId: runUserId(run),
    contextJson: runContext(run),
    capabilityId: sourceStep.capabilityId,
    inputs: cloneRecord(sourceStep.inputBindings),
    retryPolicy: cloneRecord(sourceStep.retryPolicy),
    notice: `Loaded replay draft from ${sourceStep.stepName || sourceStep.stepId} in run ${run.id}.`,
  };
}

function firstDiagnosticReason(
  diagnostics: WorkbenchConversionDiagnostic[],
  fallback: string
): string {
  return diagnostics[0]?.message || fallback;
}

function jsonLiteralValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value) || isRecord(value)) {
    return formatJson(value);
  }
  return JSON.stringify(value ?? null);
}

function mapPromotionBinding(
  field: string,
  value: unknown,
  step: NormalizedRunSpecStep,
  validStepIds: Set<string>
): { binding: ComposerInputBinding | null; diagnostic: WorkbenchConversionDiagnostic | null } {
  if (isRecord(value) && typeof value.kind === "string") {
    const kind = value.kind.trim();
    if (kind === "literal") {
      return {
        binding: {
          kind: "literal",
          value: jsonLiteralValue(value.value),
        },
        diagnostic: null,
      };
    }
    if (kind === "context") {
      const path = stringValue(value.path);
      if (!path) {
        return {
          binding: null,
          diagnostic: {
            code: "invalid_context_binding",
            message: `Step '${step.stepId}' field '${field}' is missing a context path.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      return {
        binding: { kind: "context", path },
        diagnostic: null,
      };
    }
    if (kind === "workflow_input") {
      const inputKey = stringValue(value.inputKey);
      if (!inputKey) {
        return {
          binding: null,
          diagnostic: {
            code: "invalid_workflow_input_binding",
            message: `Step '${step.stepId}' field '${field}' is missing a workflow input key.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      return {
        binding: { kind: "workflow_input", inputKey },
        diagnostic: null,
      };
    }
    if (kind === "workflow_variable") {
      const variableKey = stringValue(value.variableKey);
      if (!variableKey) {
        return {
          binding: null,
          diagnostic: {
            code: "invalid_workflow_variable_binding",
            message: `Step '${step.stepId}' field '${field}' is missing a workflow variable key.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      return {
        binding: { kind: "workflow_variable", variableKey },
        diagnostic: null,
      };
    }
    if (kind === "memory") {
      const scope = stringValue(value.scope);
      const name = stringValue(value.name);
      if (!name || (scope !== "job" && scope !== "user" && scope !== "global")) {
        return {
          binding: null,
          diagnostic: {
            code: "invalid_memory_binding",
            message: `Step '${step.stepId}' field '${field}' uses an unsupported memory binding.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      return {
        binding: {
          kind: "memory",
          scope,
          name,
          ...(stringValue(value.key) ? { key: stringValue(value.key) } : {}),
        },
        diagnostic: null,
      };
    }
    if (kind === "step_output") {
      const sourceNodeId = stringValue(value.sourceNodeId);
      const sourcePath = stringValue(value.sourcePath);
      if (!sourceNodeId || !sourcePath) {
        return {
          binding: null,
          diagnostic: {
            code: "invalid_step_output_binding",
            message: `Step '${step.stepId}' field '${field}' is missing a step output source.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      if (!validStepIds.has(sourceNodeId)) {
        return {
          binding: null,
          diagnostic: {
            code: "missing_step_output_source",
            message: `Step '${step.stepId}' field '${field}' references unknown source step '${sourceNodeId}'.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      if (!step.dependsOn.includes(sourceNodeId)) {
        return {
          binding: null,
          diagnostic: {
            code: "missing_step_dependency",
            message: `Step '${step.stepId}' field '${field}' references '${sourceNodeId}' without a matching depends_on entry.`,
            stepId: step.stepId,
            field,
          },
        };
      }
      return {
        binding: { kind: "step_output", sourceNodeId, sourcePath },
        diagnostic: null,
      };
    }
    return {
      binding: null,
      diagnostic: {
        code: "unsupported_binding_kind",
        message: `Step '${step.stepId}' field '${field}' uses unsupported binding kind '${kind}'.`,
        stepId: step.stepId,
        field,
      },
    };
  }
  if (isRecord(value) && ("$from" in value || "from" in value)) {
    return {
      binding: null,
      diagnostic: {
        code: "unsupported_runtime_binding",
        message: `Step '${step.stepId}' field '${field}' uses a runtime binding shape that Workflow Studio cannot import.`,
        stepId: step.stepId,
        field,
      },
    };
  }
  return {
    binding: {
      kind: "literal",
      value: jsonLiteralValue(value),
    },
    diagnostic: null,
  };
}

export function mapDebuggerStepToReplayDraft(
  debuggerData: WorkbenchDebuggerData,
  stepId: string
): WorkbenchReplayResult {
  const stepPayload = debuggerData.steps.find((step) => step.step.id === stepId);
  if (!stepPayload) {
    return {
      replayable: false,
      reason: "The selected step is no longer available in the debugger payload.",
      diagnostics: [
        {
          code: "missing_debugger_step",
          message: "The selected step is no longer available in the debugger payload.",
          stepId,
        },
      ],
    };
  }
  const normalizedRunSpec = normalizeRunSpec(debuggerData.run);
  const normalizedStep = matchNormalizedStep(debuggerData, stepPayload);
  const capabilityId =
    stringValue(stepPayload.step.capability_id) ||
    normalizedStep?.capabilityId ||
    "";
  if (!capabilityId) {
    return {
      replayable: false,
      reason: `Step '${stepPayload.step.name}' does not map to an executable capability.`,
      diagnostics: [
        {
          code: "missing_capability_id",
          message: `Step '${stepPayload.step.name}' does not map to an executable capability.`,
          stepId: stepPayload.step.id,
        },
      ],
    };
  }
  if (
    normalizedStep &&
    !isExecutableCapabilityStep(normalizedStep)
  ) {
    return {
      replayable: false,
      reason: `Step '${stepPayload.step.name}' is not a replayable capability step.`,
      diagnostics: [
        {
          code: "non_executable_step",
          message: `Step '${stepPayload.step.name}' is not a replayable capability step.`,
          stepId: stepPayload.step.id,
        },
      ],
    };
  }
  const replayInputs =
    firstResolvedInputs(stepPayload.execution_requests) ||
    (isRecord(stepPayload.step.input_bindings)
      ? cloneRecord(stepPayload.step.input_bindings)
      : normalizedStep?.inputBindings || {});
  const retryPolicy =
    (stepPayload.execution_requests.find((request) => isRecord(request.retry_policy))
      ?.retry_policy as Record<string, unknown> | undefined) ||
    normalizedStep?.retryPolicy ||
    {};
  return {
    replayable: true,
    draft: replayDraftFromSources(debuggerData.run, normalizedRunSpec, {
      stepId: stepPayload.step.id,
      stepName: stepPayload.step.name,
      capabilityId,
      inputBindings: replayInputs,
      retryPolicy: isRecord(retryPolicy) ? retryPolicy : {},
    }),
  };
}

export function mapRunToWorkbenchFork(debuggerData: WorkbenchDebuggerData): WorkbenchForkResult {
  const normalizedRunSpec = normalizeRunSpec(debuggerData.run);
  const mode = workbenchModeForRun(debuggerData.run, normalizedRunSpec);

  if (mode === "capability") {
    const primaryStep = normalizedRunSpec.steps[0] || null;
    if (!primaryStep || !isExecutableCapabilityStep(primaryStep)) {
      const diagnostics = [
        {
          code: "invalid_capability_run",
          message: "This run does not contain a single executable capability step.",
          stepId: primaryStep?.stepId,
        },
      ];
      const rawDraft: WorkbenchAgentRawDraft = {
        sourceRunId: debuggerData.run.id,
        title: runTitle(debuggerData.run, normalizedRunSpec),
        goal: runGoal(debuggerData.run, normalizedRunSpec),
        userId: runUserId(debuggerData.run),
        contextJson: runContext(debuggerData.run),
        runSpec: cloneRecord(normalizedRunSpec.raw),
        reason: diagnostics[0].message,
        notice: `Forked run ${debuggerData.run.id} into the raw RunSpec editor because the capability run shape was not replayable.`,
      };
      return {
        mode: "agent_raw",
        draft: rawDraft,
        diagnostics,
      };
    }
    const replayResult =
      debuggerData.steps[0]
        ? mapDebuggerStepToReplayDraft(debuggerData, debuggerData.steps[0].step.id)
        : {
            replayable: true as const,
            draft: replayDraftFromSources(debuggerData.run, normalizedRunSpec, {
              stepId: primaryStep.stepId,
              stepName: primaryStep.name,
              capabilityId: primaryStep.capabilityId,
              inputBindings: primaryStep.inputBindings,
              retryPolicy: primaryStep.retryPolicy,
            }),
          };
    if (!replayResult.replayable) {
      const rawDraft: WorkbenchAgentRawDraft = {
        sourceRunId: debuggerData.run.id,
        title: runTitle(debuggerData.run, normalizedRunSpec),
        goal: runGoal(debuggerData.run, normalizedRunSpec),
        userId: runUserId(debuggerData.run),
        contextJson: runContext(debuggerData.run),
        runSpec: cloneRecord(normalizedRunSpec.raw),
        reason: replayResult.reason,
        notice: `Forked run ${debuggerData.run.id} into the raw RunSpec editor because the capability replay payload could not be reconstructed.`,
      };
      return {
        mode: "agent_raw",
        draft: rawDraft,
        diagnostics: replayResult.diagnostics,
      };
    }
    return {
      mode: "capability",
      draft: replayResult.draft,
      diagnostics: [],
    };
  }

  const diagnostics: WorkbenchConversionDiagnostic[] = [];
  const seenStepIds = new Set<string>();
  const structuredSteps = normalizedRunSpec.steps.reduce<WorkbenchAgentStructuredDraft["steps"]>(
    (items, step) => {
      if (!isExecutableCapabilityStep(step)) {
        diagnostics.push({
          code: "non_executable_step",
          message: `Step '${step.stepId}' is not an executable capability step and can only be forked as raw RunSpec.`,
          stepId: step.stepId,
        });
        return items;
      }
      if (seenStepIds.has(step.stepId)) {
        diagnostics.push({
          code: "duplicate_step_id",
          message: `Step id '${step.stepId}' is duplicated in this run.`,
          stepId: step.stepId,
        });
        return items;
      }
      if (step.executionGate && Object.keys(step.executionGate).length > 0) {
        diagnostics.push({
          code: "unsupported_execution_gate",
          message: `Step '${step.stepId}' uses execution_gate and must be forked as raw RunSpec.`,
          stepId: step.stepId,
        });
        return items;
      }
      const missingDependency = step.dependsOn.find((dependencyId) => !normalizedRunSpec.steps.some((candidate) => candidate.stepId === dependencyId));
      if (missingDependency) {
        diagnostics.push({
          code: "missing_dependency",
          message: `Step '${step.stepId}' depends on unknown step '${missingDependency}'.`,
          stepId: step.stepId,
        });
        return items;
      }
      seenStepIds.add(step.stepId);
      items.push({
        stepId: step.stepId,
        name: step.name,
        description: step.description,
        instruction: step.instruction,
        capabilityId: step.capabilityId,
        dependsOn: [...step.dependsOn],
        inputBindings: cloneRecord(step.inputBindings),
        retryPolicy: cloneRecord(step.retryPolicy),
      });
      return items;
    },
    []
  );

  if (diagnostics.length > 0 || structuredSteps.length === 0) {
    const reason = firstDiagnosticReason(
      diagnostics,
      "This run uses structures that the structured agent builder cannot represent."
    );
    return {
      mode: "agent_raw",
      draft: {
        sourceRunId: debuggerData.run.id,
        title: runTitle(debuggerData.run, normalizedRunSpec),
        goal: runGoal(debuggerData.run, normalizedRunSpec),
        userId: runUserId(debuggerData.run),
        contextJson: runContext(debuggerData.run),
        runSpec: cloneRecord(normalizedRunSpec.raw),
        reason,
        notice: `Forked run ${debuggerData.run.id} into the raw RunSpec editor because the run shape is not representable in the structured builder.`,
      },
      diagnostics,
    };
  }

  return {
    mode: "agent_structured",
    draft: {
      sourceRunId: debuggerData.run.id,
      title: runTitle(debuggerData.run, normalizedRunSpec),
      goal: runGoal(debuggerData.run, normalizedRunSpec),
      userId: runUserId(debuggerData.run),
      contextJson: runContext(debuggerData.run),
      steps: structuredSteps,
      notice: `Forked run ${debuggerData.run.id} into the structured agent builder.`,
    },
    diagnostics: [],
  };
}

export function mapRunToWorkflowPromotion(
  debuggerData: WorkbenchDebuggerData
): WorkbenchWorkflowPromotionResult {
  const normalizedRunSpec = normalizeRunSpec(debuggerData.run);
  if (workbenchModeForRun(debuggerData.run, normalizedRunSpec) !== "agent") {
    return {
      promotable: false,
      reason: "Only agent runs can be promoted to Workflow Builder drafts.",
      diagnostics: [
        {
          code: "unsupported_run_mode",
          message: "Only agent runs can be promoted to Workflow Builder drafts.",
        },
      ],
    };
  }
  if (normalizedRunSpec.steps.length === 0) {
    return {
      promotable: false,
      reason: "The selected run does not contain any steps to promote.",
      diagnostics: [
        {
          code: "empty_run_spec",
          message: "The selected run does not contain any steps to promote.",
        },
      ],
    };
  }

  const diagnostics: WorkbenchConversionDiagnostic[] = [];
  const nodes: ComposerDraftNode[] = [];
  const stepIds = new Set(normalizedRunSpec.steps.map((step) => step.stepId).filter(Boolean));

  normalizedRunSpec.steps.forEach((step) => {
    if (!step.stepId) {
      diagnostics.push({
        code: "missing_step_id",
        message: "A run step is missing step_id and cannot be promoted.",
      });
      return;
    }
    if (!isExecutableCapabilityStep(step)) {
      diagnostics.push({
        code: "non_executable_step",
        message: `Step '${step.stepId}' is not a promotable capability step.`,
        stepId: step.stepId,
      });
      return;
    }
    if (step.executionGate && Object.keys(step.executionGate).length > 0) {
      diagnostics.push({
        code: "unsupported_execution_gate",
        message: `Step '${step.stepId}' uses execution_gate and cannot be promoted safely.`,
        stepId: step.stepId,
      });
      return;
    }
    const mappedBindings: Record<string, ComposerInputBinding> = {};
    Object.entries(step.inputBindings).forEach(([field, value]) => {
      const mapped = mapPromotionBinding(field, value, step, stepIds);
      if (mapped.diagnostic) {
        diagnostics.push(mapped.diagnostic);
        return;
      }
      if (mapped.binding) {
        mappedBindings[field] = mapped.binding;
      }
    });
    const taskName = uniqueTaskName(
      step.name || taskNameFromCapability(step.capabilityId),
      nodes
    );
    nodes.push({
      id: step.stepId,
      taskName,
      capabilityId: step.capabilityId,
      outputPath: inferCapabilityOutputPath(step.capabilityId),
      inputBindings: mappedBindings,
      outputs: [],
      variables: [],
    });
  });

  const edges = normalizedRunSpec.steps.reduce<ComposerDraftEdge[]>((items, step) => {
    step.dependsOn.forEach((dependencyId) => {
      if (!stepIds.has(dependencyId)) {
        diagnostics.push({
          code: "missing_dependency",
          message: `Step '${step.stepId}' depends on unknown step '${dependencyId}'.`,
          stepId: step.stepId,
        });
        return;
      }
      items.push({
        fromNodeId: dependencyId,
        toNodeId: step.stepId,
      });
    });
    return items;
  }, []);

  if (diagnostics.length > 0) {
    return {
      promotable: false,
      reason: firstDiagnosticReason(
        diagnostics,
        "This run cannot be promoted into a Workflow Builder draft."
      ),
      diagnostics,
    };
  }

  const draft: WorkbenchWorkflowPromotionDraft = {
    summary: runTitle(debuggerData.run, normalizedRunSpec),
    goal: runGoal(debuggerData.run, normalizedRunSpec),
    contextJsonText: formatJson(runContext(debuggerData.run)),
    nodePositions: {},
    nodes,
    edges: normalizeComposerEdges(nodes, edges),
    workflowInterface: cloneJsonValue(EMPTY_WORKFLOW_INTERFACE),
    sourceRunId: debuggerData.run.id,
    sourceTitle: runTitle(debuggerData.run, normalizedRunSpec),
    notice: `Imported from workbench run ${debuggerData.run.id}.`,
  };

  return {
    promotable: true,
    draft,
    diagnostics: [],
  };
}
