"use client";

import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import ComposerDagCanvas from "../../components/composer/ComposerDagCanvas";
import ComposerValidationPanel from "../../components/composer/ComposerValidationPanel";
import ScreenHeader, {
  screenHeaderPrimaryActionClassName,
  screenHeaderSecondaryActionClassName
} from "../../components/ScreenHeader";
import StudioCapabilityPalette from "./StudioCapabilityPalette";
import StudioCompilePanel from "./StudioCompilePanel";
import StudioWorkflowInterfacePanel from "./StudioWorkflowInterfacePanel";
import StudioNodeInspector from "./StudioNodeInspector";
import StudioWorkflowLibrary from "./StudioWorkflowLibrary";
import type {
  CanvasPoint,
  CapabilityCatalog,
  CapabilityItem,
  CapabilitySchemaField,
  ChainPreflightResult,
  ComposerCompileResponse,
  ComposerDraft,
  ComposerDraftEdge,
  ComposerDraftNode,
  ComposerInputBinding,
  ComposerIssueFocus,
  StudioPersistedWorkflowDraft,
  StudioControlCase,
  StudioControlConfig,
  StudioControlKind,
  WorkflowBinding,
  WorkflowDefinition,
  WorkflowInputDefinition,
  WorkflowInterface,
  WorkflowOutputDefinition,
  WorkflowRun,
  WorkflowRunResult,
  WorkflowTrigger,
  WorkflowVariableDefinition,
  WorkflowVersion,
} from "./types";
import {
  CHAINABLE_REQUIRED_FIELDS,
  DAG_CANVAS_MIN_HEIGHT,
  DAG_CANVAS_MIN_WIDTH,
  DAG_CANVAS_NODE_HEIGHT,
  DAG_CANVAS_NODE_WIDTH,
  DAG_CANVAS_PADDING,
  DAG_CANVAS_SNAP,
  capabilityInputSchemaProperties,
  capabilityOutputSchemaFields,
  collectContextPathSuggestions,
  collectComposerValidationIssues,
  defaultDagNodePosition,
  detectDagCycle,
  formatTimestamp,
  getCapabilityRequiredInputs,
  inferCapabilityOutputPath,
  isContextInputPresent,
  isInteractiveCanvasTarget,
  isPathOutputReference,
  normalizeComposerEdges,
  outputPathSuggestionsForNodeWithCapability,
  readContextObject,
  schemaPropertyTypeLabel,
  taskNameFromCapability,
  uniqueTaskName,
} from "./utils";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "/api";
const MEMORY_USER_ID_KEY = "ape.memory.user_id.v1";
const DEFAULT_WORKSPACE_USER_ID = "narendersurabhi";

const initialStudioDraft = (): ComposerDraft => ({
  summary: "Workflow Studio draft",
  nodes: [],
  edges: [],
});

const initialWorkflowInterface = (): WorkflowInterface => ({
  inputs: [],
  variables: [],
  outputs: [],
});

const createStudioOutput = () => ({
  id: `output-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  name: "",
  path: "",
  description: "",
});

const createStudioVariable = () => ({
  id: `variable-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  key: "",
  value: "",
  description: "",
});

const createStudioControlCase = (): StudioControlCase => ({
  id: `case-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  label: "",
  match: "",
});

const createWorkflowInputDefinition = (): WorkflowInputDefinition => ({
  id: `workflow-input-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  key: "",
  label: "",
  valueType: "string",
  required: false,
  description: "",
  defaultValue: "",
  binding: null,
});

const createWorkflowVariableDefinition = (): WorkflowVariableDefinition => ({
  id: `workflow-variable-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  key: "",
  description: "",
  binding: { kind: "literal", value: "" },
});

const createWorkflowOutputDefinition = (): WorkflowOutputDefinition => ({
  id: `workflow-output-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  key: "",
  label: "",
  description: "",
  binding: null,
});

const defaultControlConfig = (kind: StudioControlKind): StudioControlConfig => {
  if (kind === "if") {
    return { expression: "", trueLabel: "true" };
  }
  if (kind === "if_else") {
    return { expression: "", trueLabel: "true", falseLabel: "false" };
  }
  if (kind === "switch") {
    return { expression: "", switchCases: [createStudioControlCase()] };
  }
  return { expression: "", parallelMode: "fan_out" };
};

const defaultBranchLabelForSourceNode = (
  sourceNode: ComposerDraftNode | undefined,
  existingEdges: ComposerDraftEdge[]
): string => {
  if (!sourceNode || sourceNode.nodeKind !== "control") {
    return "";
  }
  const config = sourceNode.controlConfig || defaultControlConfig(sourceNode.controlKind || "if");
  if (sourceNode.controlKind === "if") {
    return String(config.trueLabel || "true").trim();
  }
  if (sourceNode.controlKind === "if_else") {
    const outgoing = existingEdges.filter((edge) => edge.fromNodeId === sourceNode.id);
    const used = new Set(outgoing.map((edge) => String(edge.branchLabel || "").trim().toLowerCase()));
    const trueLabel = String(config.trueLabel || "true").trim();
    const falseLabel = String(config.falseLabel || "false").trim();
    if (!used.has(trueLabel.toLowerCase())) {
      return trueLabel;
    }
    if (!used.has(falseLabel.toLowerCase())) {
      return falseLabel;
    }
  }
  return "";
};

const initialContextJson = () =>
  JSON.stringify(
    {
      today: new Date().toISOString().slice(0, 10),
      output_dir: "documents",
    },
    null,
    2
  );

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const normalizeCanvasPointMap = (value: unknown): Record<string, CanvasPoint> => {
  if (!isRecord(value)) {
    return {};
  }
  const next: Record<string, CanvasPoint> = {};
  Object.entries(value).forEach(([key, rawPoint]) => {
    if (!isRecord(rawPoint)) {
      return;
    }
    const x = typeof rawPoint.x === "number" ? rawPoint.x : Number(rawPoint.x);
    const y = typeof rawPoint.y === "number" ? rawPoint.y : Number(rawPoint.y);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      next[key] = { x, y };
    }
  });
  return next;
};

const normalizeInputBinding = (value: unknown): ComposerInputBinding | null => {
  if (!isRecord(value)) {
    return null;
  }
  if (value.kind === "step_output") {
    return {
      kind: "step_output",
      sourceNodeId: typeof value.sourceNodeId === "string" ? value.sourceNodeId : "",
      sourcePath: typeof value.sourcePath === "string" ? value.sourcePath : "",
      ...(typeof value.defaultValue === "string" ? { defaultValue: value.defaultValue } : {}),
    };
  }
  if (value.kind === "workflow_input") {
    return {
      kind: "workflow_input",
      inputKey: typeof value.inputKey === "string" ? value.inputKey : "",
      ...(typeof value.defaultValue === "string" ? { defaultValue: value.defaultValue } : {}),
    };
  }
  if (value.kind === "workflow_variable") {
    return {
      kind: "workflow_variable",
      variableKey: typeof value.variableKey === "string" ? value.variableKey : "",
      ...(typeof value.defaultValue === "string" ? { defaultValue: value.defaultValue } : {}),
    };
  }
  if (value.kind === "literal") {
    return {
      kind: "literal",
      value: typeof value.value === "string" ? value.value : "",
    };
  }
  if (value.kind === "context") {
    return {
      kind: "context",
      path: typeof value.path === "string" ? value.path : "",
    };
  }
  if (value.kind === "memory") {
    return {
      kind: "memory",
      scope: value.scope === "user" || value.scope === "global" ? value.scope : "job",
      name: typeof value.name === "string" ? value.name : "",
      ...(typeof value.key === "string" ? { key: value.key } : {}),
    };
  }
  return null;
};

const normalizeWorkflowBinding = (value: unknown): WorkflowBinding | null => {
  if (!isRecord(value) || typeof value.kind !== "string") {
    return null;
  }
  if (value.kind === "literal") {
    return { kind: "literal", value: typeof value.value === "string" ? value.value : "" };
  }
  if (value.kind === "context") {
    return { kind: "context", path: typeof value.path === "string" ? value.path : "" };
  }
  if (value.kind === "memory") {
    return {
      kind: "memory",
      scope: value.scope === "user" || value.scope === "global" ? value.scope : "job",
      name: typeof value.name === "string" ? value.name : "",
      ...(typeof value.key === "string" ? { key: value.key } : {}),
    };
  }
  if (value.kind === "secret") {
    return {
      kind: "secret",
      secretName: typeof value.secretName === "string" ? value.secretName : "",
    };
  }
  if (value.kind === "workflow_input") {
    return {
      kind: "workflow_input",
      inputKey: typeof value.inputKey === "string" ? value.inputKey : "",
    };
  }
  if (value.kind === "workflow_variable") {
    return {
      kind: "workflow_variable",
      variableKey: typeof value.variableKey === "string" ? value.variableKey : "",
    };
  }
  if (value.kind === "step_output") {
    return {
      kind: "step_output",
      sourceNodeId: typeof value.sourceNodeId === "string" ? value.sourceNodeId : "",
      sourcePath: typeof value.sourcePath === "string" ? value.sourcePath : "",
    };
  }
  return null;
};

const normalizeInputBindings = (value: unknown): Record<string, ComposerInputBinding> => {
  if (!isRecord(value)) {
    return {};
  }
  const next: Record<string, ComposerInputBinding> = {};
  Object.entries(value).forEach(([field, rawBinding]) => {
    const binding = normalizeInputBinding(rawBinding);
    if (binding) {
      next[field] = binding;
    }
  });
  return next;
};

const normalizeControlKind = (value: unknown): StudioControlKind | null => {
  if (value === "if" || value === "if_else" || value === "switch" || value === "parallel") {
    return value;
  }
  return null;
};

const normalizeControlCases = (value: unknown): StudioControlCase[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.reduce<StudioControlCase[]>((cases, rawCase, index) => {
    if (!isRecord(rawCase)) {
      return cases;
    }
    cases.push({
      id:
        typeof rawCase.id === "string" && rawCase.id.trim()
          ? rawCase.id
          : `restored-case-${index + 1}`,
      label: typeof rawCase.label === "string" ? rawCase.label : "",
      match: typeof rawCase.match === "string" ? rawCase.match : "",
    });
    return cases;
  }, []);
};

const normalizeControlConfig = (
  kind: StudioControlKind | null,
  value: unknown
): StudioControlConfig | null => {
  if (!kind) {
    return null;
  }
  const fallback = defaultControlConfig(kind);
  if (!isRecord(value)) {
    return fallback;
  }
  return {
    ...fallback,
    ...(typeof value.expression === "string" ? { expression: value.expression } : {}),
    ...(typeof value.trueLabel === "string" ? { trueLabel: value.trueLabel } : {}),
    ...(typeof value.falseLabel === "string" ? { falseLabel: value.falseLabel } : {}),
    ...(value.parallelMode === "fan_in" || value.parallelMode === "fan_out"
      ? { parallelMode: value.parallelMode }
      : {}),
    ...(Array.isArray(value.switchCases)
      ? { switchCases: normalizeControlCases(value.switchCases) }
      : {}),
  };
};

const normalizeNodeOutputs = (value: unknown): ComposerDraftNode["outputs"] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.reduce<ComposerDraftNode["outputs"]>((outputs, rawOutput, index) => {
    if (!isRecord(rawOutput)) {
      return outputs;
    }
    outputs.push({
      id:
        typeof rawOutput.id === "string" && rawOutput.id.trim()
          ? rawOutput.id
          : `restored-output-${index + 1}`,
      name: typeof rawOutput.name === "string" ? rawOutput.name : "",
      path: typeof rawOutput.path === "string" ? rawOutput.path : "",
      description: typeof rawOutput.description === "string" ? rawOutput.description : "",
    });
    return outputs;
  }, []);
};

const normalizeNodeVariables = (value: unknown): ComposerDraftNode["variables"] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.reduce<ComposerDraftNode["variables"]>((variables, rawVariable, index) => {
    if (!isRecord(rawVariable)) {
      return variables;
    }
    variables.push({
      id:
        typeof rawVariable.id === "string" && rawVariable.id.trim()
          ? rawVariable.id
          : `restored-variable-${index + 1}`,
      key: typeof rawVariable.key === "string" ? rawVariable.key : "",
      value: typeof rawVariable.value === "string" ? rawVariable.value : "",
      description: typeof rawVariable.description === "string" ? rawVariable.description : "",
    });
    return variables;
  }, []);
};

const normalizePersistedNodes = (value: unknown): ComposerDraftNode[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.reduce<ComposerDraftNode[]>((nodes, rawNode, index) => {
    if (!isRecord(rawNode)) {
      return nodes;
    }
    const nodeKind = rawNode.nodeKind === "control" ? "control" : "capability";
    const controlKind = nodeKind === "control" ? normalizeControlKind(rawNode.controlKind) : null;
    nodes.push({
      id:
        typeof rawNode.id === "string" && rawNode.id.trim()
          ? rawNode.id
          : `restored-node-${index + 1}`,
      taskName:
        typeof rawNode.taskName === "string" && rawNode.taskName.trim()
          ? rawNode.taskName
          : `Restored Step ${index + 1}`,
      capabilityId:
        typeof rawNode.capabilityId === "string" && rawNode.capabilityId.trim()
          ? rawNode.capabilityId
          : "unknown.capability",
      outputPath: typeof rawNode.outputPath === "string" ? rawNode.outputPath : "result",
      nodeKind,
      controlKind,
      controlConfig: normalizeControlConfig(controlKind, rawNode.controlConfig),
      inputBindings: normalizeInputBindings(rawNode.inputBindings),
      outputs: normalizeNodeOutputs(rawNode.outputs),
      variables: normalizeNodeVariables(rawNode.variables),
    });
    return nodes;
  }, []);
};

const normalizePersistedEdges = (value: unknown): ComposerDraftEdge[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.reduce<ComposerDraftEdge[]>((edges, rawEdge) => {
    if (!isRecord(rawEdge)) {
      return edges;
    }
    if (typeof rawEdge.fromNodeId !== "string" || typeof rawEdge.toNodeId !== "string") {
      return edges;
    }
    edges.push({
      fromNodeId: rawEdge.fromNodeId,
      toNodeId: rawEdge.toNodeId,
      ...(typeof rawEdge.branchLabel === "string" ? { branchLabel: rawEdge.branchLabel } : {}),
    });
    return edges;
  }, []);
};

const normalizeWorkflowInterface = (value: unknown): WorkflowInterface => {
  if (!isRecord(value)) {
    return initialWorkflowInterface();
  }
  const inputs = Array.isArray(value.inputs)
    ? value.inputs.reduce<WorkflowInputDefinition[]>((items, rawInput, index) => {
        if (!isRecord(rawInput)) {
          return items;
        }
        items.push({
          id:
            typeof rawInput.id === "string" && rawInput.id.trim()
              ? rawInput.id
              : `restored-workflow-input-${index + 1}`,
          key: typeof rawInput.key === "string" ? rawInput.key : "",
          label: typeof rawInput.label === "string" ? rawInput.label : "",
          valueType:
            rawInput.valueType === "number" ||
            rawInput.valueType === "boolean" ||
            rawInput.valueType === "object" ||
            rawInput.valueType === "array"
              ? rawInput.valueType
              : "string",
          required: Boolean(rawInput.required),
          description: typeof rawInput.description === "string" ? rawInput.description : "",
          defaultValue:
            typeof rawInput.defaultValue === "string" ? rawInput.defaultValue : "",
          binding: normalizeWorkflowBinding(rawInput.binding),
        });
        return items;
      }, [])
    : [];
  const variables = Array.isArray(value.variables)
    ? value.variables.reduce<WorkflowVariableDefinition[]>((items, rawVariable, index) => {
        if (!isRecord(rawVariable)) {
          return items;
        }
        items.push({
          id:
            typeof rawVariable.id === "string" && rawVariable.id.trim()
              ? rawVariable.id
              : `restored-workflow-variable-${index + 1}`,
          key: typeof rawVariable.key === "string" ? rawVariable.key : "",
          description:
            typeof rawVariable.description === "string" ? rawVariable.description : "",
          binding: normalizeWorkflowBinding(rawVariable.binding),
        });
        return items;
      }, [])
    : [];
  const outputs = Array.isArray(value.outputs)
    ? value.outputs.reduce<WorkflowOutputDefinition[]>((items, rawOutput, index) => {
        if (!isRecord(rawOutput)) {
          return items;
        }
        items.push({
          id:
            typeof rawOutput.id === "string" && rawOutput.id.trim()
              ? rawOutput.id
              : `restored-workflow-output-${index + 1}`,
          key: typeof rawOutput.key === "string" ? rawOutput.key : "",
          label: typeof rawOutput.label === "string" ? rawOutput.label : "",
          description: typeof rawOutput.description === "string" ? rawOutput.description : "",
          binding: normalizeWorkflowBinding(rawOutput.binding),
        });
        return items;
      }, [])
    : [];
  return { inputs, variables, outputs };
};

const restorePersistedWorkflowDraft = (
  draft: StudioPersistedWorkflowDraft | null | undefined,
  fallbackGoal: string,
  fallbackContext: Record<string, unknown>
) => {
  const nodes = normalizePersistedNodes(draft?.nodes);
  return {
    goal: typeof draft?.goal === "string" ? draft.goal : fallbackGoal,
    contextJsonText:
      typeof draft?.contextJsonText === "string"
        ? draft.contextJsonText
        : JSON.stringify(fallbackContext, null, 2),
    nodePositions: normalizeCanvasPointMap(draft?.nodePositions),
    workflowInterface: normalizeWorkflowInterface(
      draft?.workflowInterface ||
        (isRecord(draft) && "workflow_interface" in draft ? draft.workflow_interface : null)
    ),
    composerDraft: {
      summary:
        typeof draft?.summary === "string" && draft.summary.trim()
          ? draft.summary
          : fallbackGoal || "Workflow Studio draft",
      nodes,
      edges: normalizeComposerEdges(nodes, normalizePersistedEdges(draft?.edges)),
    },
  };
};

export default function WorkflowStudio() {
  const [goal, setGoal] = useState("");
  const [contextJson, setContextJson] = useState(initialContextJson);
  const [workspaceUserId, setWorkspaceUserId] = useState(DEFAULT_WORKSPACE_USER_ID);
  const [composerDraft, setComposerDraft] = useState<ComposerDraft>(initialStudioDraft);
  const [workflowInterface, setWorkflowInterface] = useState<WorkflowInterface>(
    initialWorkflowInterface
  );
  const [capabilityCatalog, setCapabilityCatalog] = useState<CapabilityCatalog | null>(null);
  const [capabilityLoading, setCapabilityLoading] = useState(true);
  const [capabilityError, setCapabilityError] = useState<string | null>(null);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteGroup, setPaletteGroup] = useState("all");
  const [selectedDagNodeId, setSelectedDagNodeId] = useState<string | null>(null);
  const [studioNotice, setStudioNotice] = useState<string | null>(null);
  const [chainPreflightLoading, setChainPreflightLoading] = useState(false);
  const [composerCompileLoading, setComposerCompileLoading] = useState(false);
  const [chainPreflightResult, setChainPreflightResult] = useState<ChainPreflightResult | null>(null);
  const [composerCompileResult, setComposerCompileResult] = useState<ComposerCompileResponse | null>(null);
  const [savedWorkflowDefinition, setSavedWorkflowDefinition] = useState<WorkflowDefinition | null>(null);
  const [publishedWorkflowVersion, setPublishedWorkflowVersion] = useState<WorkflowVersion | null>(null);
  const [loadedWorkflowVersionId, setLoadedWorkflowVersionId] = useState<string | null>(null);
  const [workflowDefinitions, setWorkflowDefinitions] = useState<WorkflowDefinition[]>([]);
  const [workflowDefinitionsLoading, setWorkflowDefinitionsLoading] = useState(true);
  const [workflowDefinitionsError, setWorkflowDefinitionsError] = useState<string | null>(null);
  const [workflowVersions, setWorkflowVersions] = useState<WorkflowVersion[]>([]);
  const [workflowVersionsLoading, setWorkflowVersionsLoading] = useState(false);
  const [workflowVersionsError, setWorkflowVersionsError] = useState<string | null>(null);
  const [workflowTriggers, setWorkflowTriggers] = useState<WorkflowTrigger[]>([]);
  const [workflowTriggersLoading, setWorkflowTriggersLoading] = useState(false);
  const [workflowTriggersError, setWorkflowTriggersError] = useState<string | null>(null);
  const [workflowRuns, setWorkflowRuns] = useState<WorkflowRun[]>([]);
  const [workflowRunsLoading, setWorkflowRunsLoading] = useState(false);
  const [workflowRunsError, setWorkflowRunsError] = useState<string | null>(null);
  const [workflowActionLoading, setWorkflowActionLoading] = useState<
    "save" | "publish" | "run" | "delete" | null
  >(null);
  const [workflowDefinitionDeleteId, setWorkflowDefinitionDeleteId] = useState<string | null>(null);
  const [activeComposerIssueFocus, setActiveComposerIssueFocus] = useState<ComposerIssueFocus | null>(
    null
  );
  const [composerNodePositions, setComposerNodePositions] = useState<Record<string, CanvasPoint>>({});
  const [hoveredDagEdgeKey, setHoveredDagEdgeKey] = useState<string | null>(null);
  const [dagEdgeDraftSourceNodeId, setDagEdgeDraftSourceNodeId] = useState<string | null>(null);
  const [dagConnectorDrag, setDagConnectorDrag] = useState<{
    sourceNodeId: string;
    x: number;
    y: number;
  } | null>(null);
  const [dagCanvasDraggingNodeId, setDagCanvasDraggingNodeId] = useState<string | null>(null);
  const [dagConnectorHoverTargetNodeId, setDagConnectorHoverTargetNodeId] = useState<string | null>(
    null
  );

  const dagCanvasDragOffsetRef = useRef<CanvasPoint>({ x: 0, y: 0 });
  const dagCanvasViewportRef = useRef<HTMLDivElement | null>(null);
  const dagCanvasRef = useRef<HTMLDivElement | null>(null);
  const inspectorBindingRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const deferredPaletteQuery = useDeferredValue(paletteQuery);
  const visualChainNodes = composerDraft.nodes;
  const composerDraftEdges = composerDraft.edges;
  const contextState = useMemo(() => readContextObject(contextJson), [contextJson]);
  const withWorkspaceUserContext = (value: Record<string, unknown>) => {
    const normalizedUserId = workspaceUserId.trim();
    if (!normalizedUserId) {
      return value;
    }
    const existing = value.user_id;
    if (typeof existing === "string" && existing.trim()) {
      return value;
    }
    return { ...value, user_id: normalizedUserId };
  };

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(MEMORY_USER_ID_KEY);
    if (stored && stored.trim()) {
      setWorkspaceUserId(stored.trim());
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(MEMORY_USER_ID_KEY, workspaceUserId);
  }, [workspaceUserId]);
  const contextPathSuggestions = useMemo(
    () => collectContextPathSuggestions(contextState.context),
    [contextState.context]
  );
  const activeWorkflowDefinitionId = savedWorkflowDefinition?.id || null;
  const activeWorkflowVersionId = loadedWorkflowVersionId || publishedWorkflowVersion?.id || null;

  useEffect(() => {
    let cancelled = false;
    const loadCapabilities = async () => {
      setCapabilityLoading(true);
      setCapabilityError(null);
      try {
        const response = await fetch(`${apiUrl}/capabilities?with_schemas=true`);
        if (!response.ok) {
          throw new Error(`Capability catalog request failed (${response.status}).`);
        }
        const data = (await response.json()) as CapabilityCatalog;
        if (!cancelled) {
          setCapabilityCatalog(data);
        }
      } catch (error) {
        if (!cancelled) {
          setCapabilityCatalog(null);
          setCapabilityError(
            error instanceof Error
              ? error.message
              : `Network error while loading capabilities from ${apiUrl}/capabilities?with_schemas=true.`
          );
        }
      } finally {
        if (!cancelled) {
          setCapabilityLoading(false);
        }
      }
    };
    void loadCapabilities();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshWorkflowDefinitions = async (nextUserId?: string) => {
    setWorkflowDefinitionsLoading(true);
    setWorkflowDefinitionsError(null);
    try {
      const params = new URLSearchParams();
      const normalizedUserId = (nextUserId ?? workspaceUserId).trim();
      if (normalizedUserId) {
        params.set("user_id", normalizedUserId);
      }
      const response = await fetch(
        `${apiUrl}/workflows/definitions${params.size > 0 ? `?${params.toString()}` : ""}`
      );
      const body = (await response.json()) as WorkflowDefinition[] | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Workflow library request failed (${response.status}).`
        );
      }
      setWorkflowDefinitions(Array.isArray(body) ? body : []);
    } catch (error) {
      setWorkflowDefinitionsError(
        error instanceof Error ? error.message : "Failed to load saved workflows."
      );
      setWorkflowDefinitions([]);
    } finally {
      setWorkflowDefinitionsLoading(false);
    }
  };

  const refreshWorkflowVersions = async (definitionId: string) => {
    if (!definitionId.trim()) {
      setWorkflowVersions([]);
      setWorkflowVersionsError(null);
      setWorkflowVersionsLoading(false);
      return;
    }
    setWorkflowVersionsLoading(true);
    setWorkflowVersionsError(null);
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definitionId)}/versions`
      );
      const body = (await response.json()) as WorkflowVersion[] | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Workflow version history request failed (${response.status}).`
        );
      }
      setWorkflowVersions(Array.isArray(body) ? body : []);
    } catch (error) {
      setWorkflowVersionsError(
        error instanceof Error ? error.message : "Failed to load workflow versions."
      );
      setWorkflowVersions([]);
    } finally {
      setWorkflowVersionsLoading(false);
    }
  };

  const refreshWorkflowTriggers = async (definitionId: string) => {
    if (!definitionId.trim()) {
      setWorkflowTriggers([]);
      setWorkflowTriggersError(null);
      setWorkflowTriggersLoading(false);
      return;
    }
    setWorkflowTriggersLoading(true);
    setWorkflowTriggersError(null);
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definitionId)}/triggers`
      );
      const body = (await response.json()) as WorkflowTrigger[] | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Workflow trigger request failed (${response.status}).`
        );
      }
      setWorkflowTriggers(Array.isArray(body) ? body : []);
    } catch (error) {
      setWorkflowTriggersError(
        error instanceof Error ? error.message : "Failed to load workflow triggers."
      );
      setWorkflowTriggers([]);
    } finally {
      setWorkflowTriggersLoading(false);
    }
  };

  const refreshWorkflowRuns = async (definitionId: string) => {
    if (!definitionId.trim()) {
      setWorkflowRuns([]);
      setWorkflowRunsError(null);
      setWorkflowRunsLoading(false);
      return;
    }
    setWorkflowRunsLoading(true);
    setWorkflowRunsError(null);
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definitionId)}/runs?limit=12`
      );
      const body = (await response.json()) as WorkflowRun[] | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Workflow run history request failed (${response.status}).`
        );
      }
      setWorkflowRuns(Array.isArray(body) ? body : []);
    } catch (error) {
      setWorkflowRunsError(
        error instanceof Error ? error.message : "Failed to load workflow run history."
      );
      setWorkflowRuns([]);
    } finally {
      setWorkflowRunsLoading(false);
    }
  };

  useEffect(() => {
    void refreshWorkflowDefinitions();
  }, [workspaceUserId]);

  useEffect(() => {
    if (!activeWorkflowDefinitionId) {
      setWorkflowVersions([]);
      setWorkflowVersionsError(null);
      setWorkflowVersionsLoading(false);
      setWorkflowTriggers([]);
      setWorkflowTriggersError(null);
      setWorkflowTriggersLoading(false);
      setWorkflowRuns([]);
      setWorkflowRunsError(null);
      setWorkflowRunsLoading(false);
      return;
    }
    void refreshWorkflowVersions(activeWorkflowDefinitionId);
    void refreshWorkflowTriggers(activeWorkflowDefinitionId);
    void refreshWorkflowRuns(activeWorkflowDefinitionId);
  }, [activeWorkflowDefinitionId]);

  const availableCapabilities = useMemo(() => {
    const items = capabilityCatalog?.items || [];
    return [...items].sort((left, right) => left.id.localeCompare(right.id));
  }, [capabilityCatalog]);

  const capabilityById = useMemo(
    () => new Map(availableCapabilities.map((item) => [item.id, item])),
    [availableCapabilities]
  );

  const paletteGroups = useMemo(
    () =>
      Array.from(
        new Set(
          availableCapabilities
            .map((item) => (item.group || "").trim())
            .filter(Boolean)
        )
      ).sort((left, right) => left.localeCompare(right)),
    [availableCapabilities]
  );

  const paletteCapabilities = useMemo(() => {
    const query = deferredPaletteQuery.trim().toLowerCase();
    return availableCapabilities.filter((item) => {
      if (paletteGroup !== "all" && (item.group || "").trim() !== paletteGroup) {
        return false;
      }
      if (!query) {
        return true;
      }
      const haystack = [
        item.id,
        item.description,
        item.group,
        item.subgroup,
        ...(item.tags || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [availableCapabilities, deferredPaletteQuery, paletteGroup]);

  useEffect(() => {
    setComposerNodePositions((prev) => {
      const next = { ...prev };
      let changed = false;
      visualChainNodes.forEach((node, index) => {
        if (!next[node.id]) {
          next[node.id] = defaultDagNodePosition(index);
          changed = true;
        }
      });
      Object.keys(next).forEach((nodeId) => {
        if (!visualChainNodes.some((node) => node.id === nodeId)) {
          delete next[nodeId];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [visualChainNodes]);

  useEffect(() => {
    if (!selectedDagNodeId) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === selectedDagNodeId)) {
      setSelectedDagNodeId(null);
    }
  }, [selectedDagNodeId, visualChainNodes]);

  useEffect(() => {
    if (!dagEdgeDraftSourceNodeId) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === dagEdgeDraftSourceNodeId)) {
      setDagEdgeDraftSourceNodeId(null);
    }
  }, [dagEdgeDraftSourceNodeId, visualChainNodes]);

  useEffect(() => {
    if (!dagConnectorDrag) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === dagConnectorDrag.sourceNodeId)) {
      setDagConnectorDrag(null);
      setDagConnectorHoverTargetNodeId(null);
    }
  }, [dagConnectorDrag, visualChainNodes]);

  useEffect(() => {
    setChainPreflightResult(null);
    setComposerCompileResult(null);
    setActiveComposerIssueFocus(null);
  }, [goal, contextJson, visualChainNodes, composerDraftEdges, workflowInterface]);

  useEffect(() => {
    if (!dagCanvasDraggingNodeId && !dagConnectorDrag) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const canvas = dagCanvasRef.current;
      if (!canvas) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (dagCanvasDraggingNodeId) {
        const rawX = event.clientX - rect.left - dagCanvasDragOffsetRef.current.x;
        const rawY = event.clientY - rect.top - dagCanvasDragOffsetRef.current.y;
        const x = Math.max(0, Math.round(rawX / DAG_CANVAS_SNAP) * DAG_CANVAS_SNAP);
        const y = Math.max(0, Math.round(rawY / DAG_CANVAS_SNAP) * DAG_CANVAS_SNAP);
        setComposerNodePositions((prev) => ({
          ...prev,
          [dagCanvasDraggingNodeId]: { x, y },
        }));
      }
      if (dagConnectorDrag) {
        setDagConnectorDrag((prev) =>
          prev
            ? {
                ...prev,
                x: Math.max(0, event.clientX - rect.left),
                y: Math.max(0, event.clientY - rect.top),
              }
            : prev
        );
      }
    };

    const handleUp = () => {
      if (dagCanvasDraggingNodeId) {
        setDagCanvasDraggingNodeId(null);
      }
      if (dagConnectorDrag) {
        setDagConnectorDrag(null);
        setDagConnectorHoverTargetNodeId(null);
        setDagEdgeDraftSourceNodeId(null);
      }
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [dagCanvasDraggingNodeId, dagConnectorDrag]);

  const setVisualChainNodes = (
    next: ComposerDraftNode[] | ((prev: ComposerDraftNode[]) => ComposerDraftNode[])
  ) => {
    setComposerDraft((prev) => {
      const nextNodes =
        typeof next === "function"
          ? (next as (nodes: ComposerDraftNode[]) => ComposerDraftNode[])(prev.nodes)
          : next;
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, prev.edges),
      };
    });
  };

  const centerDagNodeInView = (nodeId: string) => {
    const viewport = dagCanvasViewportRef.current;
    if (!viewport) {
      return;
    }
    const fallbackIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    const position =
      composerNodePositions[nodeId] ||
      (fallbackIndex >= 0 ? defaultDagNodePosition(fallbackIndex) : null);
    if (!position) {
      return;
    }
    const left = Math.max(0, position.x - Math.max(0, viewport.clientWidth - DAG_CANVAS_NODE_WIDTH) / 2);
    const top = Math.max(0, position.y - Math.max(0, viewport.clientHeight - DAG_CANVAS_NODE_HEIGHT) / 2);
    viewport.scrollTo({ left, top, behavior: "smooth" });
  };

  const setVisualBindingFromSource = (
    nodeId: string,
    field: string,
    sourceNodeId: string,
    preferredPath?: string
  ) => {
    setVisualChainNodes((prev) => {
      const sourceNode = prev.find((node) => node.id === sourceNodeId);
      if (!sourceNode) {
        return prev;
      }
      return prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const existing = node.inputBindings[field];
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              kind: "step_output",
              sourceNodeId: sourceNode.id,
              sourcePath:
                preferredPath ||
                (existing?.kind === "step_output" ? existing.sourcePath : "") ||
                sourceNode.outputPath ||
                "result",
            },
          },
        };
      });
    });
  };

  const setVisualBindingFromPrevious = (nodeId: string, field: string) => {
    const targetIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    if (targetIndex <= 0) {
      return;
    }
    const previousNode = visualChainNodes[targetIndex - 1];
    if (!previousNode) {
      return;
    }
    setVisualBindingFromSource(nodeId, field, previousNode.id, previousNode.outputPath || "result");
    setStudioNotice(`Wired ${field} from previous step.`);
  };

  const clearVisualBinding = (nodeId: string, field: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const nextBindings = { ...node.inputBindings };
        delete nextBindings[field];
        return { ...node, inputBindings: nextBindings };
      })
    );
  };

  const setVisualBindingLiteral = (nodeId: string, field: string, value: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "literal", value },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingContext = (nodeId: string, field: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "context", path: field },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingMemory = (
    nodeId: string,
    field: string,
    scope: "job" | "user" | "global" = "job"
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: {
                  kind: "memory",
                  scope,
                  name: "task_outputs",
                },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingMode = (
    nodeId: string,
    field: string,
    mode:
      | "context"
      | "from"
      | "literal"
      | "memory"
      | "workflow_input"
      | "workflow_variable"
  ) => {
    if (mode === "context") {
      setVisualBindingContext(nodeId, field);
      return;
    }
    if (mode === "literal") {
      setVisualBindingLiteral(nodeId, field, "");
      return;
    }
    if (mode === "memory") {
      setVisualBindingMemory(nodeId, field, "job");
      return;
    }
    if (mode === "workflow_input") {
      setVisualBindingWorkflowInput(nodeId, field);
      return;
    }
    if (mode === "workflow_variable") {
      setVisualBindingWorkflowVariable(nodeId, field);
      return;
    }
    const sourceNodes = visualChainNodes.filter((node) => node.id !== nodeId);
    if (sourceNodes.length === 0) {
      setStudioNotice(`No source step is available for ${field}.`);
      return;
    }
    const sourceNode = sourceNodes[sourceNodes.length - 1];
    setVisualBindingFromSource(nodeId, field, sourceNode.id, sourceNode.outputPath || "result");
  };

  const updateVisualBindingSourceNode = (nodeId: string, field: string, sourceNodeId: string) => {
    const sourceNode = visualChainNodes.find((node) => node.id === sourceNodeId);
    if (!sourceNode) {
      return;
    }
    setVisualBindingFromSource(nodeId, field, sourceNodeId, sourceNode.outputPath || "result");
  };

  const updateVisualBindingPath = (nodeId: string, field: string, sourcePath: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "step_output") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              sourcePath,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingLiteral = (nodeId: string, field: string, value: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "literal") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              value,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingContextPath = (nodeId: string, field: string, path: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "context") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              path,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingMemory = (
    nodeId: string,
    field: string,
    patch: { scope?: "job" | "user" | "global"; name?: string; key?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "memory") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              ...patch,
            },
          },
        };
      })
    );
  };

  const setVisualBindingWorkflowInput = (nodeId: string, field: string) => {
    const fallbackKey = workflowInterface.inputs[0]?.key || "";
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "workflow_input", inputKey: fallbackKey },
              },
            }
          : node
      )
    );
  };

  const updateVisualBindingWorkflowInput = (
    nodeId: string,
    field: string,
    inputKey: string
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "workflow_input") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              inputKey,
            },
          },
        };
      })
    );
  };

  const setVisualBindingWorkflowVariable = (nodeId: string, field: string) => {
    const fallbackKey = workflowInterface.variables[0]?.key || "";
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "workflow_variable", variableKey: fallbackKey },
              },
            }
          : node
      )
    );
  };

  const updateVisualBindingWorkflowVariable = (
    nodeId: string,
    field: string,
    variableKey: string
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "workflow_variable") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              variableKey,
            },
          },
        };
      })
    );
  };

  const addCustomInputField = (nodeId: string, field: string) => {
    const trimmed = field.trim();
    if (!trimmed) {
      return;
    }
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId || node.inputBindings[trimmed]) {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [trimmed]: {
              kind: "context",
              path: trimmed,
            },
          },
        };
      })
    );
  };

  const removeCustomInputField = (nodeId: string, field: string) => {
    clearVisualBinding(nodeId, field);
  };

  const addNodeOutput = (nodeId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, outputs: [...node.outputs, createStudioOutput()] }
          : node
      )
    );
  };

  const updateNodeOutput = (
    nodeId: string,
    outputId: string,
    patch: { name?: string; path?: string; description?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        return {
          ...node,
          outputs: node.outputs.map((output) =>
            output.id === outputId ? { ...output, ...patch } : output
          ),
        };
      })
    );
  };

  const removeNodeOutput = (nodeId: string, outputId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, outputs: node.outputs.filter((output) => output.id !== outputId) }
          : node
      )
    );
  };

  const upsertNodeOutputFromSchema = (nodeId: string, field: CapabilitySchemaField) => {
    const normalizedPath = String(field.path || "").trim();
    if (!normalizedPath) {
      return;
    }
    const normalizedName = normalizedPath.split(/[.[\]]/)[0]?.trim() || normalizedPath;
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const alreadyPresent = node.outputs.some(
          (output) =>
            output.path.trim() === normalizedPath ||
            output.name.trim() === normalizedName
        );
        if (alreadyPresent) {
          return node;
        }
        return {
          ...node,
          outputs: [
            ...node.outputs,
            {
              ...createStudioOutput(),
              name: normalizedName,
              path: normalizedPath,
              description: field.description || "",
            },
          ],
        };
      })
    );
  };

  const addNodeVariable = (nodeId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, variables: [...node.variables, createStudioVariable()] }
          : node
      )
    );
  };

  const updateNodeVariable = (
    nodeId: string,
    variableId: string,
    patch: { key?: string; value?: string; description?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        return {
          ...node,
          variables: node.variables.map((variable) =>
            variable.id === variableId ? { ...variable, ...patch } : variable
          ),
        };
      })
    );
  };

  const removeNodeVariable = (nodeId: string, variableId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, variables: node.variables.filter((variable) => variable.id !== variableId) }
          : node
      )
    );
  };

  const addWorkflowInputDefinition = () => {
    setWorkflowInterface((prev) => ({
      ...prev,
      inputs: [...prev.inputs, createWorkflowInputDefinition()],
    }));
  };

  const updateWorkflowInputDefinition = (
    inputId: string,
    patch: Partial<WorkflowInputDefinition>
  ) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      inputs: prev.inputs.map((item) => (item.id === inputId ? { ...item, ...patch } : item)),
    }));
  };

  const removeWorkflowInputDefinition = (inputId: string) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      inputs: prev.inputs.filter((item) => item.id !== inputId),
    }));
  };

  const addWorkflowVariableDefinition = () => {
    setWorkflowInterface((prev) => ({
      ...prev,
      variables: [...prev.variables, createWorkflowVariableDefinition()],
    }));
  };

  const updateWorkflowVariableDefinition = (
    variableId: string,
    patch: Partial<WorkflowVariableDefinition>
  ) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      variables: prev.variables.map((item) =>
        item.id === variableId ? { ...item, ...patch } : item
      ),
    }));
  };

  const removeWorkflowVariableDefinition = (variableId: string) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      variables: prev.variables.filter((item) => item.id !== variableId),
    }));
  };

  const addWorkflowOutputDefinition = () => {
    setWorkflowInterface((prev) => ({
      ...prev,
      outputs: [...prev.outputs, createWorkflowOutputDefinition()],
    }));
  };

  const updateWorkflowOutputDefinition = (
    outputId: string,
    patch: Partial<WorkflowOutputDefinition>
  ) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      outputs: prev.outputs.map((item) => (item.id === outputId ? { ...item, ...patch } : item)),
    }));
  };

  const removeWorkflowOutputDefinition = (outputId: string) => {
    setWorkflowInterface((prev) => ({
      ...prev,
      outputs: prev.outputs.filter((item) => item.id !== outputId),
    }));
  };

  const addCapabilityNodeToStudio = (capabilityId: string) => {
    const capability = capabilityById.get(capabilityId);
    const context = contextState.context;
    const nodeId = `studio-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    setComposerDraft((prev) => {
      const anchorNode =
        prev.nodes.find((node) => node.id === selectedDagNodeId) || prev.nodes[prev.nodes.length - 1] || null;
      const inputBindings: Record<string, ComposerInputBinding> = {};
      const requiredInputs = getCapabilityRequiredInputs(capability);
      requiredInputs.forEach((field) => {
        if (!CHAINABLE_REQUIRED_FIELDS.has(field)) {
          return;
        }
        if (isContextInputPresent(context[field])) {
          return;
        }
        if (!anchorNode) {
          return;
        }
        inputBindings[field] = {
          kind: "step_output",
          sourceNodeId: anchorNode.id,
          sourcePath: anchorNode.outputPath || "result",
        };
      });

      const newNode: ComposerDraftNode = {
        id: nodeId,
        taskName: uniqueTaskName(taskNameFromCapability(capabilityId), prev.nodes),
        capabilityId,
        outputPath: inferCapabilityOutputPath(capabilityId),
        nodeKind: "capability",
        controlKind: null,
        controlConfig: null,
        inputBindings,
        outputs: [],
        variables: [],
      };
      const nextNodes = [
        ...prev.nodes,
        newNode,
      ];
      const nextEdges = anchorNode ? [...prev.edges, { fromNodeId: anchorNode.id, toNodeId: nodeId }] : prev.edges;
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, nextEdges),
      };
    });

    setComposerNodePositions((prev) => {
      const anchorPosition = selectedDagNodeId ? prev[selectedDagNodeId] : null;
      return {
        ...prev,
        [nodeId]: anchorPosition
          ? {
              x: anchorPosition.x + DAG_CANVAS_NODE_WIDTH + 64,
              y: anchorPosition.y,
            }
          : defaultDagNodePosition(Object.keys(prev).length),
      };
    });
    setSelectedDagNodeId(nodeId);
    setStudioNotice(`Added ${capabilityId} to the workflow.`);
  };

  const addControlNodeToStudio = (kind: StudioControlKind) => {
    const nodeId = `studio-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const controlId = `studio.control.${kind}`;
    setComposerDraft((prev) => {
      const anchorNode =
        prev.nodes.find((node) => node.id === selectedDagNodeId) || prev.nodes[prev.nodes.length - 1] || null;
      const newNode: ComposerDraftNode = {
        id: nodeId,
        taskName: uniqueTaskName(taskNameFromCapability(controlId), prev.nodes),
        capabilityId: controlId,
        outputPath: "result",
        nodeKind: "control",
        controlKind: kind,
        controlConfig: defaultControlConfig(kind),
        inputBindings: {},
        outputs: [],
        variables: [],
      };
      const nextNodes = [
        ...prev.nodes,
        newNode,
      ];
      const nextEdges = anchorNode ? [...prev.edges, { fromNodeId: anchorNode.id, toNodeId: nodeId }] : prev.edges;
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, nextEdges),
      };
    });
    setComposerNodePositions((prev) => {
      const anchorPosition = selectedDagNodeId ? prev[selectedDagNodeId] : null;
      return {
        ...prev,
        [nodeId]: anchorPosition
          ? { x: anchorPosition.x + DAG_CANVAS_NODE_WIDTH + 64, y: anchorPosition.y }
          : defaultDagNodePosition(Object.keys(prev).length),
      };
    });
    setSelectedDagNodeId(nodeId);
    setStudioNotice(`Added ${kind.replace("_", " ")} control node.`);
  };

  const updateVisualChainNode = (
    nodeId: string,
    patch: Partial<Pick<ComposerDraftNode, "taskName" | "capabilityId" | "outputPath">>
  ) => {
    setVisualChainNodes((prev) => {
      const current = prev.find((node) => node.id === nodeId);
      if (!current) {
        return prev;
      }
      return prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const nextTaskName =
          patch.taskName !== undefined ? uniqueTaskName(patch.taskName, prev, nodeId) : node.taskName;
        const nextCapabilityId = patch.capabilityId ?? node.capabilityId;
        let nextOutputPath = patch.outputPath ?? node.outputPath;
        if (
          patch.capabilityId &&
          patch.capabilityId !== node.capabilityId &&
          !patch.outputPath &&
          node.nodeKind !== "control"
        ) {
          nextOutputPath = inferCapabilityOutputPath(patch.capabilityId);
        }
        return {
          ...node,
          taskName: nextTaskName,
          capabilityId: nextCapabilityId,
          outputPath: nextOutputPath,
        };
      });
    });
  };

  const updateNodeControlConfig = (nodeId: string, patch: Partial<StudioControlConfig>) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              controlConfig: {
                ...(node.controlConfig || defaultControlConfig(node.controlKind || "if")),
                ...patch,
              },
            }
          : node
      )
    );
  };

  const addSwitchCase = (nodeId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              controlConfig: {
                ...(node.controlConfig || defaultControlConfig("switch")),
                switchCases: [...(node.controlConfig?.switchCases || []), createStudioControlCase()],
              },
            }
          : node
      )
    );
  };

  const updateSwitchCase = (
    nodeId: string,
    caseId: string,
    patch: Partial<Pick<StudioControlCase, "label" | "match">>
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              controlConfig: {
                ...(node.controlConfig || defaultControlConfig("switch")),
                switchCases: (node.controlConfig?.switchCases || []).map((item) =>
                  item.id === caseId ? { ...item, ...patch } : item
                ),
              },
            }
          : node
      )
    );
  };

  const removeSwitchCase = (nodeId: string, caseId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              controlConfig: {
                ...(node.controlConfig || defaultControlConfig("switch")),
                switchCases: (node.controlConfig?.switchCases || []).filter((item) => item.id !== caseId),
              },
            }
          : node
      )
    );
  };

  const removeVisualChainNode = (nodeId: string) => {
    setComposerDraft((prev) => {
      const nextNodes = prev.nodes
        .filter((node) => node.id !== nodeId)
        .map((node) => {
          const nextBindings: Record<string, ComposerInputBinding> = {};
          Object.entries(node.inputBindings).forEach(([field, binding]) => {
            if (binding.kind === "step_output" && binding.sourceNodeId === nodeId) {
              return;
            }
            nextBindings[field] = binding;
          });
          return { ...node, inputBindings: nextBindings };
        });
      const nextEdges = prev.edges.filter(
        (edge) => edge.fromNodeId !== nodeId && edge.toNodeId !== nodeId
      );
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, nextEdges),
      };
    });
    setComposerNodePositions((prev) => {
      const next = { ...prev };
      delete next[nodeId];
      return next;
    });
    setSelectedDagNodeId((prev) => (prev === nodeId ? null : prev));
    setStudioNotice("Removed step from workflow.");
  };

  const autoBindTargetFromSource = (sourceNodeId: string, targetNodeId: string) => {
    const sourceNode = visualChainNodes.find((node) => node.id === sourceNodeId);
    const targetNode = visualChainNodes.find((node) => node.id === targetNodeId);
    if (!sourceNode || !targetNode) {
      return false;
    }
    const targetStatus = visualChainNodeStatusById.get(targetNodeId);
    if (!targetStatus || targetStatus.requiredStatus.length === 0) {
      return false;
    }
    const missingFields = targetStatus.requiredStatus
      .filter((status) => status.status === "missing")
      .map((status) => status.field);
    const candidateField = missingFields[0] || targetStatus.requiredStatus[0]?.field || "";
    if (!candidateField) {
      return false;
    }
    setVisualBindingFromSource(targetNodeId, candidateField, sourceNodeId, sourceNode.outputPath || "result");
    return true;
  };

  const addDagEdge = (fromNodeId: string, toNodeId: string) => {
    if (!fromNodeId || !toNodeId || fromNodeId === toNodeId) {
      return;
    }
    setComposerDraft((prev) => {
      const sourceNode = prev.nodes.find((node) => node.id === fromNodeId);
      const branchLabel = defaultBranchLabelForSourceNode(sourceNode, prev.edges);
      return {
        ...prev,
        edges: normalizeComposerEdges(prev.nodes, [
          ...prev.edges,
          { fromNodeId, toNodeId, ...(branchLabel ? { branchLabel } : {}) },
        ]),
      };
    });
    const mapped = autoBindTargetFromSource(fromNodeId, toNodeId);
    setSelectedDagNodeId(toNodeId);
    centerDagNodeInView(toNodeId);
    setStudioNotice(mapped ? "Connected edge and wired a missing input." : "Connected DAG edge.");
  };

  const removeDagEdge = (fromNodeId: string, toNodeId: string) => {
    setHoveredDagEdgeKey(null);
    setComposerDraft((prev) => ({
      ...prev,
      edges: prev.edges.filter(
        (edge) => !(edge.fromNodeId === fromNodeId && edge.toNodeId === toNodeId)
      ),
    }));
  };

  const beginDagNodeDrag = (event: React.MouseEvent<HTMLDivElement>, nodeId: string) => {
    const canvas = dagCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const current = composerNodePositions[nodeId] || defaultDagNodePosition(0);
    dagCanvasDragOffsetRef.current = {
      x: event.clientX - rect.left - current.x,
      y: event.clientY - rect.top - current.y,
    };
    setDagCanvasDraggingNodeId(nodeId);
  };

  const beginDagConnectorDrag = (
    event: React.MouseEvent<HTMLButtonElement>,
    sourceNodeId: string
  ) => {
    event.preventDefault();
    const canvas = dagCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    setDagConnectorDrag({
      sourceNodeId,
      x: Math.max(0, event.clientX - rect.left),
      y: Math.max(0, event.clientY - rect.top),
    });
    setDagConnectorHoverTargetNodeId(null);
    setDagEdgeDraftSourceNodeId(sourceNodeId);
  };

  const autoLayoutDagCanvas = () => {
    setComposerNodePositions(() => {
      const next: Record<string, CanvasPoint> = {};
      visualChainNodes.forEach((node, index) => {
        next[node.id] = defaultDagNodePosition(index);
      });
      return next;
    });
    setStudioNotice("Auto-layout applied to canvas.");
  };

  const visualChainNodesWithStatus = useMemo(() => {
    const context = contextState.context;
    return visualChainNodes.map((node, index) => {
      if (node.nodeKind === "control") {
        return {
          node,
          index,
          requiredStatus: [],
        };
      }
      const capability = capabilityById.get(node.capabilityId);
      const schemaProperties = capabilityInputSchemaProperties(capability);
      const required = getCapabilityRequiredInputs(capability);
      const requiredStatus = required.map((field) => {
        const property = schemaProperties[field];
        const schemaType = schemaPropertyTypeLabel(property);
        const schemaDescription =
          property && typeof property.description === "string" ? property.description : "";
        const binding = node.inputBindings[field];
        if (binding?.kind === "step_output") {
          const source = visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId);
          return {
            field,
            status: "from_chain" as const,
            detail: source ? `${source.taskName}.${binding.sourcePath}` : binding.sourcePath,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "literal" && binding.value.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: "literal value",
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "memory" && binding.name.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: `memory:${binding.scope}/${binding.name}${binding.key ? `/${binding.key}` : ""}`,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "workflow_input" && binding.inputKey.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: `workflow.input.${binding.inputKey}`,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "workflow_variable" && binding.variableKey.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: `workflow.variable.${binding.variableKey}`,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "context" && binding.path.trim()) {
          return {
            field,
            status: "from_context" as const,
            detail: binding.path,
            schemaType,
            schemaDescription,
          };
        }
        if (isContextInputPresent(context[field])) {
          return {
            field,
            status: "from_context" as const,
            detail: "context_json",
            schemaType,
            schemaDescription,
          };
        }
        return {
          field,
          status: "missing" as const,
          detail: "missing",
          schemaType,
          schemaDescription,
        };
      });
      return {
        node,
        index,
        requiredStatus,
      };
    });
  }, [capabilityById, contextState.context, visualChainNodes]);

  const visualChainNodeStatusById = useMemo(() => {
    const next = new Map<
      string,
      {
        missingCount: number;
        resolvedCount: number;
        requiredCount: number;
        requiredStatus: {
          field: string;
          status: "missing" | "from_chain" | "from_context" | "provided";
          detail: string;
          schemaType: string;
          schemaDescription: string;
        }[];
      }
    >();
    visualChainNodesWithStatus.forEach(({ node, requiredStatus }) => {
      const missingCount = requiredStatus.filter((status) => status.status === "missing").length;
      next.set(node.id, {
        missingCount,
        resolvedCount: requiredStatus.length - missingCount,
        requiredCount: requiredStatus.length,
        requiredStatus,
      });
    });
    return next;
  }, [visualChainNodesWithStatus]);

  const selectedDagNode = useMemo(
    () => visualChainNodes.find((node) => node.id === selectedDagNodeId) || null,
    [selectedDagNodeId, visualChainNodes]
  );

  const selectedCapability = useMemo<CapabilityItem | null>(() => {
    if (!selectedDagNode || selectedDagNode.nodeKind === "control") {
      return null;
    }
    return capabilityById.get(selectedDagNode.capabilityId) || null;
  }, [capabilityById, selectedDagNode]);

  const selectedDagNodeStatus = useMemo(() => {
    if (!selectedDagNodeId) {
      return null;
    }
    return visualChainNodeStatusById.get(selectedDagNodeId) || null;
  }, [selectedDagNodeId, visualChainNodeStatusById]);

  const selectedDagNodeInspectorFields = useMemo(() => {
    if (!selectedDagNode) {
      return [];
    }
    if (selectedDagNode.nodeKind === "control") {
      return [];
    }

    const context = contextState.context;
    const schemaProperties = capabilityInputSchemaProperties(selectedCapability);
    const requiredFieldSet = new Set(getCapabilityRequiredInputs(selectedCapability));
    const schemaFields = Array.from(
      new Set([...Object.keys(schemaProperties), ...Array.from(requiredFieldSet)])
    );
    const customFields = Object.keys(selectedDagNode.inputBindings).filter(
      (field) => !requiredFieldSet.has(field) && !schemaFields.includes(field)
    );

    const buildFieldStatus = (field: string, required: boolean, custom: boolean) => {
      const property = schemaProperties[field] || null;
      const schemaType = schemaPropertyTypeLabel(property || undefined);
      const schemaDescription =
        property && typeof property.description === "string"
          ? property.description
          : custom
            ? "Custom Studio binding"
            : "";
      const binding = selectedDagNode.inputBindings[field];
      if (binding?.kind === "step_output") {
        const source = visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId);
        return {
          field,
          required,
          custom,
          status: "from_chain" as const,
          detail: source ? `${source.taskName}.${binding.sourcePath}` : binding.sourcePath,
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (binding?.kind === "literal") {
        return {
          field,
          required,
          custom,
          status: binding.value.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.value.trim() ? "literal value" : "literal value missing",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (binding?.kind === "memory") {
        return {
          field,
          required,
          custom,
          status: binding.name.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.name.trim()
            ? `memory:${binding.scope}/${binding.name}${binding.key ? `/${binding.key}` : ""}`
            : "memory name missing",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (binding?.kind === "workflow_input") {
        return {
          field,
          required,
          custom,
          status: binding.inputKey.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.inputKey.trim()
            ? `workflow.input.${binding.inputKey}`
            : "workflow input key missing",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (binding?.kind === "workflow_variable") {
        return {
          field,
          required,
          custom,
          status: binding.variableKey.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.variableKey.trim()
            ? `workflow.variable.${binding.variableKey}`
            : "workflow variable key missing",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (binding?.kind === "context") {
        return {
          field,
          required,
          custom,
          status: binding.path.trim() ? ("from_context" as const) : ("missing" as const),
          detail: binding.path.trim() || "context path missing",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      if (isContextInputPresent(context[field])) {
        return {
          field,
          required,
          custom,
          status: "from_context" as const,
          detail: "context_json",
          schemaType,
          schemaDescription,
          schemaProperty: property,
        };
      }
      return {
        field,
        required,
        custom,
        status: "missing" as const,
        detail: "missing",
        schemaType,
        schemaDescription,
        schemaProperty: property,
      };
    };

    const schemaFieldStatus = schemaFields.map((field) =>
      buildFieldStatus(field, requiredFieldSet.has(field), false)
    );
    const customFieldStatus = customFields.map((field) => buildFieldStatus(field, false, true));

    return [
      ...schemaFieldStatus.filter((field) => field.required),
      ...schemaFieldStatus.filter((field) => !field.required),
      ...customFieldStatus,
    ];
  }, [contextState.context, selectedCapability, selectedDagNode, visualChainNodes]);

  const selectedDagNodeOutputSchemaFields = useMemo(
    () => capabilityOutputSchemaFields(selectedCapability),
    [selectedCapability]
  );

  const studioOutputPathSuggestionsForNode = (node: ComposerDraftNode | undefined) =>
    outputPathSuggestionsForNodeWithCapability(
      node,
      node ? capabilityById.get(node.capabilityId) : undefined
    );

  const quickFixNodeBindings = (nodeId: string) => {
    const targetNode = visualChainNodes.find((node) => node.id === nodeId);
    if (!targetNode) {
      return;
    }
    if (targetNode.nodeKind === "control") {
      setStudioNotice("Quick Fix is only available for capability nodes.");
      return;
    }
    const context = contextState.context;
    const requiredFields = getCapabilityRequiredInputs(capabilityById.get(targetNode.capabilityId));
    if (requiredFields.length === 0) {
      setStudioNotice("Selected node has no required inputs.");
      return;
    }
    const sourceNodeIds = composerDraftEdges
      .filter((edge) => edge.toNodeId === nodeId)
      .map((edge) => edge.fromNodeId);
    const sourceNodes = sourceNodeIds
      .map((id) => visualChainNodes.find((node) => node.id === id))
      .filter((node): node is ComposerDraftNode => Boolean(node));
    let updatedCount = 0;
    requiredFields.forEach((field) => {
      const existingBinding = targetNode.inputBindings[field];
      if (existingBinding) {
        return;
      }
      if (isContextInputPresent(context[field])) {
        return;
      }
      const sourceNode = sourceNodes[sourceNodes.length - 1];
      if (!sourceNode) {
        return;
      }
      setVisualBindingFromSource(nodeId, field, sourceNode.id, sourceNode.outputPath || "result");
      updatedCount += 1;
    });
    setStudioNotice(
      updatedCount > 0
        ? `Quick-fixed ${updatedCount} missing input(s) for ${targetNode.taskName}.`
        : "No missing inputs could be auto-fixed for selected node."
    );
  };

  const autoWireNodeBindings = (nodeId: string) => {
    const targetIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    if (targetIndex <= 0) {
      setStudioNotice("Auto-wire requires a previous step.");
      return;
    }
    const targetNode = visualChainNodes[targetIndex];
    const previousNode = visualChainNodes[targetIndex - 1];
    if (!targetNode || !previousNode) {
      setStudioNotice("Unable to auto-wire selected node.");
      return;
    }
    if (targetNode.nodeKind === "control") {
      setStudioNotice("Auto-wire is only available for capability nodes.");
      return;
    }
    const context = contextState.context;
    const requiredInputs = getCapabilityRequiredInputs(capabilityById.get(targetNode.capabilityId));
    let updatedCount = 0;
    requiredInputs.forEach((field) => {
      if (!CHAINABLE_REQUIRED_FIELDS.has(field)) {
        return;
      }
      if (targetNode.inputBindings[field]) {
        return;
      }
      if (isContextInputPresent(context[field])) {
        return;
      }
      setVisualBindingFromSource(nodeId, field, previousNode.id, previousNode.outputPath || "result");
      updatedCount += 1;
    });
    setStudioNotice(
      updatedCount > 0
        ? `Auto-wired ${updatedCount} input(s) from previous step.`
        : "No missing inputs were eligible for auto-wire."
    );
  };

  const focusComposerValidationIssue = (issue: {
    nodeId?: string;
    field?: string;
  }) => {
    const nodeId = issue.nodeId;
    if (!nodeId) {
      return;
    }
    const nodeStatus = visualChainNodeStatusById.get(nodeId);
    const resolvedField =
      issue.field && nodeStatus?.requiredStatus.some((entry) => entry.field === issue.field)
        ? issue.field
        : undefined;
    setSelectedDagNodeId(nodeId);
    centerDagNodeInView(nodeId);
    setActiveComposerIssueFocus({ nodeId, field: resolvedField });
    if (resolvedField) {
      const refKey = `${nodeId}::${resolvedField}`;
      window.setTimeout(() => {
        inspectorBindingRefs.current[refKey]?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }, 180);
    }
  };

  const visualChainSummary = useMemo(() => {
    const summary = {
      steps: visualChainNodesWithStatus.length,
      dagEdges: composerDraftEdges.length,
      requiredInputs: 0,
      missingInputs: 0,
      contextInputs: 0,
      chainedInputs: 0,
      literalInputs: 0,
    };
    visualChainNodesWithStatus.forEach(({ requiredStatus }) => {
      summary.requiredInputs += requiredStatus.length;
      requiredStatus.forEach((item) => {
        if (item.status === "missing") {
          summary.missingInputs += 1;
        } else if (item.status === "from_context") {
          summary.contextInputs += 1;
        } else if (item.status === "from_chain") {
          summary.chainedInputs += 1;
        } else {
          summary.literalInputs += 1;
        }
      });
    });
    return summary;
  }, [composerDraftEdges.length, visualChainNodesWithStatus]);

  const dagCanvasNodes = useMemo(
    () =>
      visualChainNodes.map((node, index) => ({
        node,
        position: composerNodePositions[node.id] || defaultDagNodePosition(index),
      })),
    [composerNodePositions, visualChainNodes]
  );

  const dagCanvasNodeById = useMemo(
    () => new Map(dagCanvasNodes.map((entry) => [entry.node.id, entry])),
    [dagCanvasNodes]
  );

  const dagCanvasEdges = useMemo(
    () =>
      composerDraftEdges
        .map((edge) => {
          const fromEntry = dagCanvasNodeById.get(edge.fromNodeId);
          const toEntry = dagCanvasNodeById.get(edge.toNodeId);
          if (!fromEntry || !toEntry) {
            return null;
          }
          const edgeKey = `${edge.fromNodeId}->${edge.toNodeId}`;
          const startX = fromEntry.position.x + DAG_CANVAS_NODE_WIDTH;
          const startY = fromEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
          const endX = toEntry.position.x;
          const endY = toEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
          const controlX = (startX + endX) / 2;
          const midX = controlX;
          const midY = (startY + endY) / 2;
          return {
            ...edge,
            edgeKey,
            fromTaskName: fromEntry.node.taskName,
            toTaskName: toEntry.node.taskName,
            path: `M ${startX} ${startY} C ${controlX} ${startY}, ${controlX} ${endY}, ${endX} ${endY}`,
            midX,
            midY,
          };
        })
        .filter(
          (
            item
          ): item is {
            fromNodeId: string;
            toNodeId: string;
            edgeKey: string;
            fromTaskName: string;
            toTaskName: string;
            path: string;
            midX: number;
            midY: number;
            branchLabel?: string;
          } => item !== null
        ),
    [composerDraftEdges, dagCanvasNodeById]
  );

  const dagConnectorPreview = useMemo(() => {
    if (!dagConnectorDrag) {
      return null;
    }
    const sourceEntry = dagCanvasNodeById.get(dagConnectorDrag.sourceNodeId);
    if (!sourceEntry) {
      return null;
    }
    const startX = sourceEntry.position.x + DAG_CANVAS_NODE_WIDTH;
    const startY = sourceEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
    const endX = dagConnectorDrag.x;
    const endY = dagConnectorDrag.y;
    const controlX = (startX + endX) / 2;
    return {
      path: `M ${startX} ${startY} C ${controlX} ${startY}, ${controlX} ${endY}, ${endX} ${endY}`,
    };
  }, [dagCanvasNodeById, dagConnectorDrag]);

  const dagCanvasSurface = useMemo(() => {
    let maxX = DAG_CANVAS_MIN_WIDTH;
    let maxY = DAG_CANVAS_MIN_HEIGHT;
    dagCanvasNodes.forEach((entry) => {
      maxX = Math.max(maxX, entry.position.x + DAG_CANVAS_NODE_WIDTH + DAG_CANVAS_PADDING);
      maxY = Math.max(maxY, entry.position.y + DAG_CANVAS_NODE_HEIGHT + DAG_CANVAS_PADDING);
    });
    return { width: maxX, height: maxY };
  }, [dagCanvasNodes]);

  const dagNodeAdjacency = useMemo(() => {
    const incoming: Record<string, number> = {};
    const outgoing: Record<string, number> = {};
    visualChainNodes.forEach((node) => {
      incoming[node.id] = 0;
      outgoing[node.id] = 0;
    });
    composerDraftEdges.forEach((edge) => {
      if (incoming[edge.toNodeId] !== undefined) {
        incoming[edge.toNodeId] += 1;
      }
      if (outgoing[edge.fromNodeId] !== undefined) {
        outgoing[edge.fromNodeId] += 1;
      }
    });
    return { incoming, outgoing };
  }, [composerDraftEdges, visualChainNodes]);

  const compileRequestPayload = useMemo(() => {
    const parsedContext = contextState.invalid
      ? { __invalid_context_json: true }
      : withWorkspaceUserContext(contextState.context);
    return {
      draft: {
        summary: composerDraft.summary || "Workflow Studio draft",
        nodes: visualChainNodes.map((node) => ({
          id: node.id,
          taskName: node.taskName,
          capabilityId: node.capabilityId,
          nodeKind: node.nodeKind || "capability",
          controlKind: node.controlKind || undefined,
          controlConfig: node.controlConfig || undefined,
          bindings: node.inputBindings,
        })),
        edges: composerDraftEdges,
        workflowInterface,
      },
      job_context: parsedContext,
      goal: goal.trim() || undefined,
    };
  }, [
    composerDraft.summary,
    composerDraftEdges,
    contextState.context,
    contextState.invalid,
    goal,
    visualChainNodes,
    workflowInterface,
    workspaceUserId,
  ]);

  const draftPayloadPreview = useMemo(() => {
    const parsedContext = contextState.invalid
      ? { __invalid_context_json: true }
      : withWorkspaceUserContext(contextState.context);
    return {
      draft: {
        summary: composerDraft.summary || "Workflow Studio draft",
        nodes: visualChainNodes.map((node) => ({
          id: node.id,
          taskName: node.taskName,
          capabilityId: node.capabilityId,
          nodeKind: node.nodeKind || "capability",
          controlKind: node.controlKind || undefined,
          controlConfig: node.controlConfig || undefined,
          bindings: node.inputBindings,
          outputPath: node.outputPath,
          outputs: node.outputs,
          variables: node.variables,
        })),
        edges: composerDraftEdges,
        workflowInterface,
      },
      job_context: parsedContext,
      goal: goal.trim() || undefined,
    };
  }, [
    composerDraft.summary,
    composerDraftEdges,
    contextState.context,
    contextState.invalid,
    goal,
    visualChainNodes,
    workflowInterface,
    workspaceUserId,
  ]);

  const persistedWorkflowDraft = useMemo(
    () => ({
      summary: composerDraft.summary || "Workflow Studio draft",
      goal: goal.trim() || undefined,
      contextJsonText: contextJson,
      nodePositions: composerNodePositions,
      nodes: visualChainNodes.map((node) => ({
        id: node.id,
        taskName: node.taskName,
        capabilityId: node.capabilityId,
        outputPath: node.outputPath,
        nodeKind: node.nodeKind || "capability",
        controlKind: node.controlKind || undefined,
        controlConfig: node.controlConfig || undefined,
        inputBindings: node.inputBindings,
        outputs: node.outputs,
        variables: node.variables,
      })),
      edges: composerDraftEdges,
      workflowInterface,
    }),
    [
      composerDraft.summary,
      composerDraftEdges,
      composerNodePositions,
      contextJson,
      goal,
      visualChainNodes,
      workflowInterface,
    ]
  );

  const resetStudioTransientState = () => {
    setSelectedDagNodeId(null);
    setChainPreflightResult(null);
    setComposerCompileResult(null);
    setActiveComposerIssueFocus(null);
    setHoveredDagEdgeKey(null);
    setDagEdgeDraftSourceNodeId(null);
    setDagConnectorDrag(null);
    setDagCanvasDraggingNodeId(null);
    setDagConnectorHoverTargetNodeId(null);
  };

  const startFreshStudioDraft = () => {
    setGoal("");
    setContextJson(initialContextJson());
    setComposerDraft(initialStudioDraft());
    setWorkflowInterface(initialWorkflowInterface());
    setComposerNodePositions({});
    setSavedWorkflowDefinition(null);
    setPublishedWorkflowVersion(null);
    setLoadedWorkflowVersionId(null);
    setWorkflowVersions([]);
    setWorkflowTriggers([]);
    setWorkflowRuns([]);
    resetStudioTransientState();
    setStudioNotice("Started a fresh studio draft.");
  };

  const restoreWorkflowDefinition = (definition: WorkflowDefinition) => {
    const restored = restorePersistedWorkflowDraft(
      definition.draft,
      definition.goal,
      definition.context_json
    );
    setGoal(restored.goal);
    setContextJson(restored.contextJsonText);
    setComposerDraft(restored.composerDraft);
    setWorkflowInterface(restored.workflowInterface);
    setComposerNodePositions(restored.nodePositions);
    setSavedWorkflowDefinition(definition);
    setPublishedWorkflowVersion(null);
    setLoadedWorkflowVersionId(null);
    resetStudioTransientState();
    setStudioNotice(`Opened saved draft ${definition.title}.`);
  };

  const deleteWorkflowDefinition = async (definition: WorkflowDefinition) => {
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        `Delete saved draft "${definition.title}" and its published versions, triggers, and run history?`
      );
      if (!confirmed) {
        return;
      }
    }
    setWorkflowActionLoading("delete");
    setWorkflowDefinitionDeleteId(definition.id);
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definition.id)}`,
        { method: "DELETE" }
      );
      const body = response.status === 204 ? null : ((await response.json()) as { detail?: unknown } | null);
      if (!response.ok) {
        const detail = body && typeof body.detail === "string" ? body.detail : null;
        throw new Error(detail || `Delete draft failed (${response.status}).`);
      }
      setWorkflowDefinitions((prev) => prev.filter((item) => item.id !== definition.id));
      if (savedWorkflowDefinition?.id === definition.id) {
        setSavedWorkflowDefinition(null);
        setPublishedWorkflowVersion(null);
        setLoadedWorkflowVersionId(null);
        setWorkflowVersions([]);
        setWorkflowTriggers([]);
        setWorkflowRuns([]);
      }
      void refreshWorkflowDefinitions();
      setStudioNotice(`Deleted saved draft ${definition.title}.`);
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to delete saved draft.");
    } finally {
      setWorkflowDefinitionDeleteId(null);
      setWorkflowActionLoading(null);
    }
  };

  const restoreWorkflowVersion = async (version: WorkflowVersion) => {
    let definition = savedWorkflowDefinition;
    if (!definition || definition.id !== version.definition_id) {
      definition =
        workflowDefinitions.find((item) => item.id === version.definition_id) || null;
    }
    if (!definition) {
      try {
        const response = await fetch(
          `${apiUrl}/workflows/definitions/${encodeURIComponent(version.definition_id)}`
        );
        const body = (await response.json()) as WorkflowDefinition | { detail?: unknown };
        if (!response.ok) {
          const detail = (body as { detail?: unknown }).detail;
          throw new Error(
            typeof detail === "string"
              ? detail
              : `Workflow definition request failed (${response.status}).`
          );
        }
        definition = body as WorkflowDefinition;
      } catch (error) {
        setStudioNotice(
          error instanceof Error ? error.message : "Failed to load workflow definition."
        );
        return;
      }
    }
    const restored = restorePersistedWorkflowDraft(
      version.draft,
      version.goal || definition.goal,
      version.context_json
    );
    setGoal(restored.goal);
    setContextJson(restored.contextJsonText);
    setComposerDraft(restored.composerDraft);
    setWorkflowInterface(restored.workflowInterface);
    setComposerNodePositions(restored.nodePositions);
    setSavedWorkflowDefinition(definition);
    setPublishedWorkflowVersion(version);
    setLoadedWorkflowVersionId(version.id);
    resetStudioTransientState();
    setStudioNotice(`Restored workflow version v${version.version_number}.`);
  };

  const saveWorkflowDefinition = async () => {
    if (contextState.invalid) {
      setStudioNotice("Workflow drafts can only be saved when Context JSON is valid.");
      return null;
    }
    setWorkflowActionLoading("save");
    try {
      const response = await fetch(
        savedWorkflowDefinition
          ? `${apiUrl}/workflows/definitions/${encodeURIComponent(savedWorkflowDefinition.id)}`
          : `${apiUrl}/workflows/definitions`,
        {
          method: savedWorkflowDefinition ? "PUT" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: composerDraft.summary || goal.trim() || "Workflow Studio draft",
            goal: goal.trim(),
            context_json: withWorkspaceUserContext(contextState.context),
            draft: persistedWorkflowDraft,
            user_id: workspaceUserId.trim() || undefined,
            metadata: { source: "workflow_studio" },
          }),
        }
      );
      const body = (await response.json()) as WorkflowDefinition | { detail?: unknown };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: unknown }).detail === "string"
            ? (body as { detail: string }).detail
            : `Save draft failed (${response.status}).`
        );
      }
      const definition = body as WorkflowDefinition;
      setSavedWorkflowDefinition(definition);
      void refreshWorkflowDefinitions();
      void refreshWorkflowTriggers(definition.id);
      void refreshWorkflowRuns(definition.id);
      setStudioNotice(`Saved draft ${definition.title}.`);
      return definition;
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to save workflow draft.");
      return null;
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const publishWorkflowVersion = async () => {
    const definition = await saveWorkflowDefinition();
    if (!definition) {
      return null;
    }
    setWorkflowActionLoading("publish");
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definition.id)}/publish`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ metadata: { source: "workflow_studio" } }),
        }
      );
      const body = (await response.json()) as WorkflowVersion | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string" ? detail : `Publish version failed (${response.status}).`
        );
      }
      const version = body as WorkflowVersion;
      setPublishedWorkflowVersion(version);
      setLoadedWorkflowVersionId(version.id);
      void refreshWorkflowDefinitions();
      void refreshWorkflowVersions(definition.id);
      void refreshWorkflowRuns(definition.id);
      setStudioNotice(`Published workflow version v${version.version_number}.`);
      return version;
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to publish workflow version.");
      return null;
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const runWorkflowVersion = async () => {
    const version = await publishWorkflowVersion();
    if (!version) {
      return;
    }
    setWorkflowActionLoading("run");
    try {
      const response = await fetch(
        `${apiUrl}/workflows/versions/${encodeURIComponent(version.id)}/run`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ priority: 0 }),
        }
      );
      const body = (await response.json()) as WorkflowRunResult | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string" ? detail : `Run workflow failed (${response.status}).`
        );
      }
      const result = body as WorkflowRunResult;
      setPublishedWorkflowVersion(result.workflow_version);
      setLoadedWorkflowVersionId(result.workflow_version.id);
      void refreshWorkflowRuns(result.workflow_definition.id);
      setStudioNotice(
        `Started job ${result.job.id} from workflow version v${result.workflow_version.version_number}.`
      );
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to run workflow version.");
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const createManualWorkflowTrigger = async () => {
    if (!activeWorkflowDefinitionId || !savedWorkflowDefinition) {
      setStudioNotice("Save or open a workflow definition before creating a trigger.");
      return;
    }
    setWorkflowActionLoading("save");
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(activeWorkflowDefinitionId)}/triggers`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: `${savedWorkflowDefinition.title} manual trigger`,
            trigger_type: "manual",
            enabled: true,
            config: { version_mode: "latest_published" },
            user_id: workspaceUserId.trim() || undefined,
            metadata: { source: "workflow_studio" },
          }),
        }
      );
      const body = (await response.json()) as WorkflowTrigger | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Create trigger failed (${response.status}).`
        );
      }
      void refreshWorkflowTriggers(activeWorkflowDefinitionId);
      setStudioNotice(`Created manual trigger ${(body as WorkflowTrigger).title}.`);
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to create workflow trigger.");
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const invokeWorkflowTrigger = async (trigger: WorkflowTrigger) => {
    setWorkflowActionLoading("run");
    try {
      const response = await fetch(
        `${apiUrl}/workflows/triggers/${encodeURIComponent(trigger.id)}/invoke`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ priority: 0 }),
        }
      );
      const body = (await response.json()) as WorkflowRunResult | { detail?: unknown };
      if (!response.ok) {
        const detail = (body as { detail?: unknown }).detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : `Trigger invoke failed (${response.status}).`
        );
      }
      const result = body as WorkflowRunResult;
      setPublishedWorkflowVersion(result.workflow_version);
      setLoadedWorkflowVersionId(result.workflow_version.id);
      void refreshWorkflowRuns(result.workflow_definition.id);
      setStudioNotice(`Triggered job ${result.job.id} via ${trigger.title}.`);
    } catch (error) {
      setStudioNotice(error instanceof Error ? error.message : "Failed to invoke workflow trigger.");
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const composerIssues = useMemo(
    () => collectComposerValidationIssues(chainPreflightResult, composerCompileResult, visualChainNodes),
    [chainPreflightResult, composerCompileResult, visualChainNodes]
  );

  const runChainPreflight = async () => {
    const localErrors: string[] = [];
    if (visualChainNodes.length === 0) {
      localErrors.push("No chain steps configured.");
    }

    const seenTaskNames = new Set<string>();
    visualChainNodes.forEach((node, index) => {
      const taskName = node.taskName.trim();
      if (!taskName) {
        localErrors.push(`Step ${index + 1}: task name is required.`);
      } else if (seenTaskNames.has(taskName)) {
        localErrors.push(`Duplicate task name: ${taskName}`);
      } else {
        seenTaskNames.add(taskName);
      }
      if (node.nodeKind === "control") {
        const config = node.controlConfig || defaultControlConfig(node.controlKind || "if");
        if ((node.controlKind === "if" || node.controlKind === "if_else" || node.controlKind === "switch") && !config.expression.trim()) {
          localErrors.push(`Step ${index + 1} (${taskName || "unnamed"}): control expression is required.`);
        }
        if (node.controlKind === "switch") {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): control-flow node 'switch' is not compiled yet.`
          );
        }
        if (node.controlKind === "parallel") {
          const parallelMode = config.parallelMode || "fan_out";
          const incomingEdges = composerDraftEdges.filter((edge) => edge.toNodeId === node.id);
          const outgoingEdges = composerDraftEdges.filter((edge) => edge.fromNodeId === node.id);
          if (parallelMode === "fan_out" && outgoingEdges.length < 2) {
            localErrors.push(
              `Step ${index + 1} (${taskName || "unnamed"}): parallel fan_out needs at least two outgoing edges.`
            );
          }
          if (parallelMode === "fan_in" && incomingEdges.length < 2) {
            localErrors.push(
              `Step ${index + 1} (${taskName || "unnamed"}): parallel fan_in needs at least two incoming edges.`
            );
          }
          if (parallelMode === "fan_in" && outgoingEdges.length < 1) {
            localErrors.push(
              `Step ${index + 1} (${taskName || "unnamed"}): parallel fan_in needs at least one outgoing edge.`
            );
          }
        }
        if (node.controlKind === "switch") {
          const cases = config.switchCases || [];
          if (cases.length === 0) {
            localErrors.push(`Step ${index + 1} (${taskName || "unnamed"}): switch needs at least one case.`);
          }
          const seenLabels = new Set<string>();
          cases.forEach((item, caseIndex) => {
            const label = item.label.trim();
            const match = item.match.trim();
            if (!label) {
              localErrors.push(
                `Step ${index + 1} (${taskName || "unnamed"}): switch case ${caseIndex + 1} needs a label.`
              );
            } else if (seenLabels.has(label)) {
              localErrors.push(
                `Step ${index + 1} (${taskName || "unnamed"}): duplicate switch case label '${label}'.`
              );
            } else {
              seenLabels.add(label);
            }
            if (!match) {
              localErrors.push(
                `Step ${index + 1} (${taskName || "unnamed"}): switch case ${caseIndex + 1} needs a match value.`
              );
            }
          });
        }
        if (node.controlKind === "if_else") {
          const outgoingEdges = composerDraftEdges.filter((edge) => edge.fromNodeId === node.id);
          const trueLabel = (config.trueLabel || "true").trim().toLowerCase();
          const falseLabel = (config.falseLabel || "false").trim().toLowerCase();
          const labels = new Set(
            outgoingEdges.map((edge) => (edge.branchLabel || "").trim().toLowerCase()).filter(Boolean)
          );
          if (!labels.has(trueLabel)) {
            localErrors.push(
              `Step ${index + 1} (${taskName || "unnamed"}): missing '${config.trueLabel || "true"}' branch edge.`
            );
          }
          if (!labels.has(falseLabel)) {
            localErrors.push(
              `Step ${index + 1} (${taskName || "unnamed"}): missing '${config.falseLabel || "false"}' branch edge.`
            );
          }
        }
      } else if (!capabilityById.has(node.capabilityId)) {
        localErrors.push(
          `Step ${index + 1} (${taskName || "unnamed"}): capability ${node.capabilityId} not found in catalog.`
        );
      }
      if (!node.outputPath.trim()) {
        localErrors.push(`Step ${index + 1} (${taskName || "unnamed"}): output path is required.`);
      }
      const seenOutputNames = new Set<string>();
      const seenOutputPaths = new Set<string>();
      node.outputs.forEach((output, outputIndex) => {
        const name = output.name.trim();
        const path = output.path.trim();
        if (!name) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): extra output ${outputIndex + 1} needs a name.`
          );
        } else if (seenOutputNames.has(name)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate extra output name '${name}'.`
          );
        } else {
          seenOutputNames.add(name);
        }
        if (!path) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): extra output ${outputIndex + 1} needs a path.`
          );
        } else if (seenOutputPaths.has(path)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate extra output path '${path}'.`
          );
        } else {
          seenOutputPaths.add(path);
        }
      });
      const seenVariableKeys = new Set<string>();
      node.variables.forEach((variable, variableIndex) => {
        const key = variable.key.trim();
        if (!key) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): variable ${variableIndex + 1} needs a name.`
          );
        } else if (seenVariableKeys.has(key)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate variable name '${key}'.`
          );
        } else {
          seenVariableKeys.add(key);
        }
      });
    });

    const seenWorkflowInputKeys = new Set<string>();
    workflowInterface.inputs.forEach((input, index) => {
      const key = input.key.trim();
      if (!key) {
        localErrors.push(`Workflow input ${index + 1}: key is required.`);
      } else if (seenWorkflowInputKeys.has(key)) {
        localErrors.push(`Duplicate workflow input key '${key}'.`);
      } else {
        seenWorkflowInputKeys.add(key);
      }
      if (input.binding?.kind === "context" && !input.binding.path.trim()) {
        localErrors.push(`Workflow input '${key || index + 1}': context path is required.`);
      }
      if (input.binding?.kind === "memory" && !input.binding.name.trim()) {
        localErrors.push(`Workflow input '${key || index + 1}': memory name is required.`);
      }
      if (input.binding?.kind === "secret" && !input.binding.secretName.trim()) {
        localErrors.push(`Workflow input '${key || index + 1}': secret name is required.`);
      }
    });

    const seenWorkflowVariableKeys = new Set<string>();
    workflowInterface.variables.forEach((variable, index) => {
      const key = variable.key.trim();
      if (!key) {
        localErrors.push(`Workflow variable ${index + 1}: key is required.`);
      } else if (seenWorkflowVariableKeys.has(key)) {
        localErrors.push(`Duplicate workflow variable key '${key}'.`);
      } else {
        seenWorkflowVariableKeys.add(key);
      }
      if (variable.binding?.kind === "context" && !variable.binding.path.trim()) {
        localErrors.push(`Workflow variable '${key || index + 1}': context path is required.`);
      }
      if (variable.binding?.kind === "memory" && !variable.binding.name.trim()) {
        localErrors.push(`Workflow variable '${key || index + 1}': memory name is required.`);
      }
      if (variable.binding?.kind === "secret" && !variable.binding.secretName.trim()) {
        localErrors.push(`Workflow variable '${key || index + 1}': secret name is required.`);
      }
      if (variable.binding?.kind === "workflow_input" && !variable.binding.inputKey.trim()) {
        localErrors.push(`Workflow variable '${key || index + 1}': workflow input key is required.`);
      }
    });

    const seenWorkflowOutputKeys = new Set<string>();
    workflowInterface.outputs.forEach((output, index) => {
      const key = output.key.trim();
      if (!key) {
        localErrors.push(`Workflow output ${index + 1}: key is required.`);
      } else if (seenWorkflowOutputKeys.has(key)) {
        localErrors.push(`Duplicate workflow output key '${key}'.`);
      } else {
        seenWorkflowOutputKeys.add(key);
      }
      if (output.binding?.kind === "context" && !output.binding.path.trim()) {
        localErrors.push(`Workflow output '${key || index + 1}': context path is required.`);
      }
      if (output.binding?.kind === "workflow_input" && !output.binding.inputKey.trim()) {
        localErrors.push(`Workflow output '${key || index + 1}': workflow input key is required.`);
      }
      if (
        output.binding?.kind === "workflow_variable" &&
        !output.binding.variableKey.trim()
      ) {
        localErrors.push(
          `Workflow output '${key || index + 1}': workflow variable key is required.`
        );
      }
      if (output.binding?.kind === "step_output") {
        if (!output.binding.sourceNodeId.trim()) {
          localErrors.push(`Workflow output '${key || index + 1}': source step is required.`);
        }
        if (!output.binding.sourcePath.trim()) {
          localErrors.push(`Workflow output '${key || index + 1}': source path is required.`);
        }
      }
    });

    visualChainNodesWithStatus.forEach(({ node, requiredStatus }) => {
      if (node.nodeKind === "control") {
        return;
      }
      requiredStatus
        .filter((entry) => entry.status === "missing")
        .forEach((entry) => {
          localErrors.push(
            `Step ${node.taskName}: required input '${entry.field}' is missing (not in chain or context).`
          );
        });
      Object.entries(node.inputBindings).forEach(([field, binding]) => {
        if (binding.kind !== "step_output") {
          if (binding.kind === "context" && !binding.path.trim()) {
            localErrors.push(`Step ${node.taskName}: context path for '${field}' is empty.`);
          }
          if (binding.kind === "memory" && !binding.name.trim()) {
            localErrors.push(`Step ${node.taskName}: memory name for '${field}' is required.`);
          }
          if (binding.kind === "workflow_input" && !binding.inputKey.trim()) {
            localErrors.push(`Step ${node.taskName}: workflow input for '${field}' is required.`);
          }
          if (binding.kind === "workflow_variable" && !binding.variableKey.trim()) {
            localErrors.push(`Step ${node.taskName}: workflow variable for '${field}' is required.`);
          }
          return;
        }
        if (!binding.sourcePath.trim()) {
          localErrors.push(`Step ${node.taskName}: binding for '${field}' has empty source path.`);
        }
        if (field === "path" && binding.sourcePath.trim() && !isPathOutputReference(binding.sourcePath)) {
          localErrors.push(
            `Step ${node.taskName}: binding for 'path' should reference a path output (for example 'path').`
          );
        }
        const sourceIndex = visualChainNodes.findIndex((candidate) => candidate.id === binding.sourceNodeId);
        if (sourceIndex < 0) {
          localErrors.push(
            `Step ${node.taskName}: binding for '${field}' references a removed source step.`
          );
        }
      });
    });

    const nodeIds = visualChainNodes.map((node) => node.id);
    const nodeIdSet = new Set(nodeIds);
    const explicitEdges = normalizeComposerEdges(visualChainNodes, composerDraftEdges);
    explicitEdges.forEach((edge) => {
      if (!nodeIdSet.has(edge.fromNodeId) || !nodeIdSet.has(edge.toNodeId)) {
        localErrors.push(`DAG edge '${edge.fromNodeId} -> ${edge.toNodeId}' references missing node(s).`);
      }
      if (edge.fromNodeId === edge.toNodeId) {
        localErrors.push(`DAG edge '${edge.fromNodeId} -> ${edge.toNodeId}' is a self-cycle.`);
      }
    });

    const implicitEdges: ComposerDraftEdge[] = [];
    visualChainNodes.forEach((node) => {
      Object.values(node.inputBindings).forEach((binding) => {
        if (binding.kind === "step_output") {
          implicitEdges.push({ fromNodeId: binding.sourceNodeId, toNodeId: node.id });
        }
      });
    });

    const combinedEdges = normalizeComposerEdges(visualChainNodes, [...explicitEdges, ...implicitEdges]);
    if (detectDagCycle(nodeIds, combinedEdges)) {
      localErrors.push("DAG contains a cycle (including step-output dependencies).");
    }
    if (contextState.invalid) {
      localErrors.push("Context JSON is invalid JSON.");
    }

    let serverErrors: Record<string, string> = {};
    let serverDiagnostics: {
      severity?: "error" | "warning";
      code: string;
      field?: string;
      message: string;
      slot_fields?: string[];
    }[] = [];
    let compiledPlan: Record<string, unknown> | null = null;

    if (localErrors.length === 0) {
      setComposerCompileLoading(true);
      setChainPreflightLoading(true);
      try {
        const compileResponse = await fetch(`${apiUrl}/composer/compile`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(compileRequestPayload),
        });
        const compileBody = (await compileResponse.json()) as ComposerCompileResponse | { detail?: unknown };
        if (!compileResponse.ok) {
          localErrors.push(
            typeof (compileBody as { detail?: unknown }).detail === "string"
              ? (compileBody as { detail: string }).detail
              : `Compile request failed (${compileResponse.status}).`
          );
        } else {
          const typedCompile = compileBody as ComposerCompileResponse;
          setComposerCompileResult(typedCompile);
          if (!typedCompile.valid) {
            typedCompile.diagnostics.errors.forEach((diag) => {
              serverErrors[diag.code] = diag.message;
            });
            serverErrors = { ...serverErrors, ...(typedCompile.preflight_errors || {}) };
          } else {
            compiledPlan = typedCompile.plan;
          }
        }

        if (compiledPlan) {
          const response = await fetch(`${apiUrl}/plans/preflight`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              plan: compiledPlan,
              job_context: withWorkspaceUserContext(contextState.context),
              goal: goal.trim() || undefined,
            }),
          });
          const body = (await response.json()) as
            | {
                valid: boolean;
                errors: Record<string, string>;
                diagnostics?: {
                  severity?: "error" | "warning";
                  code: string;
                  field?: string;
                  message: string;
                  slot_fields?: string[];
                }[];
              }
            | { detail?: unknown };
          if (!response.ok) {
            localErrors.push(
              typeof (body as { detail?: unknown }).detail === "string"
                ? (body as { detail: string }).detail
                : `Preflight request failed (${response.status}).`
            );
          } else if (body && typeof body === "object" && "errors" in body && body.errors) {
            serverErrors = { ...serverErrors, ...body.errors };
            if (Array.isArray(body.diagnostics)) {
              serverDiagnostics = body.diagnostics;
            }
          }
        } else if (Object.keys(serverErrors).length === 0 && localErrors.length === 0) {
          localErrors.push("Compile succeeded but returned no executable plan.");
        }
      } catch (error) {
        localErrors.push(error instanceof Error ? error.message : "Compile/preflight request failed.");
      } finally {
        setComposerCompileLoading(false);
        setChainPreflightLoading(false);
      }
    } else {
      setComposerCompileResult(null);
    }

    setChainPreflightResult({
      valid: localErrors.length === 0 && Object.keys(serverErrors).length === 0,
      localErrors,
      serverErrors,
      serverDiagnostics,
      checkedAt: new Date().toISOString(),
    });

    setStudioNotice(
      localErrors.length === 0 && Object.keys(serverErrors).length === 0
        ? "Compile + preflight passed."
        : "Compile or preflight found issues."
    );
  };

  return (
    <div className="space-y-6">
      <ScreenHeader
        eyebrow="Agentic Workflow Studio"
        title="Design DAGs before you run them."
        description="Build visual workflows with a capability palette on the left, graph canvas in the middle, and node inspector plus compile preview on the right."
        activeScreen="studio"
        actions={
          <>
            <button
              className={screenHeaderSecondaryActionClassName}
              onClick={startFreshStudioDraft}
            >
              New Draft
            </button>
            <button
              className={screenHeaderSecondaryActionClassName}
              onClick={saveWorkflowDefinition}
              disabled={workflowActionLoading !== null}
            >
              {workflowActionLoading === "save" ? "Saving..." : "Save Draft"}
            </button>
            <button
              className={screenHeaderSecondaryActionClassName}
              onClick={publishWorkflowVersion}
              disabled={workflowActionLoading !== null}
            >
              {workflowActionLoading === "publish" ? "Publishing..." : "Publish Version"}
            </button>
            <button
              className={screenHeaderSecondaryActionClassName}
              onClick={runWorkflowVersion}
              disabled={workflowActionLoading !== null}
            >
              {workflowActionLoading === "run" ? "Starting..." : "Run Workflow"}
            </button>
            <button
              className={screenHeaderPrimaryActionClassName}
              onClick={runChainPreflight}
              disabled={composerCompileLoading || chainPreflightLoading || workflowActionLoading !== null}
            >
              {composerCompileLoading || chainPreflightLoading ? "Compiling..." : "Compile Preview"}
            </button>
          </>
        }
      >
        <div className="mt-6 rounded-2xl border border-white/15 bg-white/10 px-4 py-4 text-white/95">
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-[220px] flex-1">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-sky-100">
                Memory User ID
              </div>
              <input
                className="mt-2 w-full rounded-2xl border border-white/15 bg-white/10 px-3 py-2 text-sm text-white outline-none transition placeholder:text-white/45 focus:border-white/40 focus:bg-white/15"
                value={workspaceUserId}
                onChange={(event) => setWorkspaceUserId(event.target.value)}
                placeholder="narendersurabhi"
              />
            </label>
            <div className="max-w-xl text-xs leading-5 text-slate-200">
              User-scoped memory bindings inherit this id automatically unless a node overrides
              it explicitly.
            </div>
          </div>
        </div>
      </ScreenHeader>

      {studioNotice ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800">
          {studioNotice}
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)_360px]">
            <StudioCapabilityPalette
              capabilities={paletteCapabilities}
              groups={paletteGroups}
          loading={capabilityLoading}
          error={capabilityError}
          query={paletteQuery}
          selectedGroup={paletteGroup}
              onQueryChange={setPaletteQuery}
              onGroupChange={setPaletteGroup}
              onAddCapability={addCapabilityNodeToStudio}
              onAddControl={addControlNodeToStudio}
            />

        <div className="space-y-6">
          <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
              <label className="block">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Goal
                </div>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                  value={goal}
                  onChange={(event) => setGoal(event.target.value)}
                  placeholder="Generate a document pipeline with validation and render output"
                />
              </label>
              <label className="block">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Draft Summary
                </div>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                  value={composerDraft.summary}
                  onChange={(event) =>
                    setComposerDraft((prev) => ({ ...prev, summary: event.target.value }))
                  }
                  placeholder="Workflow Studio draft"
                />
              </label>
            </div>
            <label className="mt-4 block">
              <div className="flex items-center justify-between gap-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Context JSON
                </div>
                <div className="text-xs text-slate-500">
                  {contextState.invalid ? "Invalid JSON" : "Object ready"}
                </div>
              </div>
              <textarea
                className="mt-1 min-h-[180px] w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 font-mono text-xs text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                value={contextJson}
                onChange={(event) => setContextJson(event.target.value)}
              />
            </label>
            <div className="mt-4 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                Steps <span className="font-semibold">{visualChainSummary.steps}</span>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                DAG edges <span className="font-semibold">{visualChainSummary.dagEdges}</span>
              </div>
              <div
                className={`rounded-2xl border px-3 py-2 ${
                  visualChainSummary.missingInputs > 0
                    ? "border-rose-200 bg-rose-50 text-rose-700"
                    : "border-emerald-200 bg-emerald-50 text-emerald-700"
                }`}
              >
                Missing <span className="font-semibold">{visualChainSummary.missingInputs}</span>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                Context hits <span className="font-semibold">{visualChainSummary.contextInputs}</span>
              </div>
            </div>
            <ComposerValidationPanel
              preflightResult={chainPreflightResult}
              compileLoading={composerCompileLoading || chainPreflightLoading}
              issues={composerIssues}
              needsValidation={visualChainNodes.length > 0}
              onIssueClick={focusComposerValidationIssue}
              activeIssue={activeComposerIssueFocus}
              formatTimestamp={formatTimestamp}
            />
          </section>

          <StudioWorkflowInterfacePanel
            workflowInterface={workflowInterface}
            contextPathSuggestions={contextPathSuggestions}
            visualChainNodes={visualChainNodes}
            outputPathSuggestionsForNode={studioOutputPathSuggestionsForNode}
            onAddInput={addWorkflowInputDefinition}
            onUpdateInput={updateWorkflowInputDefinition}
            onRemoveInput={removeWorkflowInputDefinition}
            onAddVariable={addWorkflowVariableDefinition}
            onUpdateVariable={updateWorkflowVariableDefinition}
            onRemoveVariable={removeWorkflowVariableDefinition}
            onAddOutput={addWorkflowOutputDefinition}
            onUpdateOutput={updateWorkflowOutputDefinition}
            onRemoveOutput={removeWorkflowOutputDefinition}
          />

          <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Canvas
                </div>
                <h2 className="mt-1 font-display text-2xl text-slate-900">Workflow Graph</h2>
              </div>
              <button
                className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900 disabled:opacity-50"
                onClick={autoLayoutDagCanvas}
                disabled={visualChainNodes.length === 0}
              >
                Auto Layout
              </button>
            </div>

            <ComposerDagCanvas
              visualChainNodes={visualChainNodes}
              dagEdgeDraftSourceNodeId={dagEdgeDraftSourceNodeId}
              setDagEdgeDraftSourceNodeId={setDagEdgeDraftSourceNodeId}
              setDagConnectorDrag={setDagConnectorDrag}
              setDagConnectorHoverTargetNodeId={setDagConnectorHoverTargetNodeId}
              autoLayoutDagCanvas={autoLayoutDagCanvas}
              dagCanvasViewportRef={dagCanvasViewportRef}
              dagCanvasRef={dagCanvasRef}
              dagCanvasSurface={dagCanvasSurface}
              dagCanvasEdges={dagCanvasEdges}
              hoveredDagEdgeKey={hoveredDagEdgeKey}
              setHoveredDagEdgeKey={setHoveredDagEdgeKey}
              removeDagEdge={removeDagEdge}
              dagConnectorPreview={dagConnectorPreview}
              dagCanvasNodes={dagCanvasNodes}
              composerDraftEdges={composerDraftEdges}
              dagNodeAdjacency={dagNodeAdjacency}
              visualChainNodeStatusById={visualChainNodeStatusById as Map<
                string,
                { missingCount: number; requiredCount: number }
              >}
              selectedDagNodeId={selectedDagNodeId}
              setSelectedDagNodeId={setSelectedDagNodeId}
              dagConnectorDrag={dagConnectorDrag}
              dagCanvasDraggingNodeId={dagCanvasDraggingNodeId}
              dagConnectorHoverTargetNodeId={dagConnectorHoverTargetNodeId}
              addDagEdge={addDagEdge}
              beginDagNodeDrag={beginDagNodeDrag}
              isInteractiveCanvasTarget={isInteractiveCanvasTarget}
              beginDagConnectorDrag={beginDagConnectorDrag}
              centerDagNodeInView={centerDagNodeInView}
              nodeWidth={DAG_CANVAS_NODE_WIDTH}
              nodeHeight={DAG_CANVAS_NODE_HEIGHT}
            />

            <div className="mt-4 flex flex-wrap gap-2">
              {visualChainNodes.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
                  Add capabilities from the palette to start shaping a DAG.
                </div>
              ) : (
                visualChainNodes.map((node, index) => {
                  const isSelected = selectedDagNodeId === node.id;
                  return (
                    <div
                      key={`studio-node-chip-${node.id}`}
                      className={`flex items-center gap-2 rounded-full border px-3 py-2 text-sm ${
                        isSelected
                          ? "border-sky-300 bg-sky-50 text-sky-900"
                          : "border-slate-200 bg-slate-50 text-slate-700"
                      }`}
                    >
                      <button onClick={() => setSelectedDagNodeId(node.id)}>
                        {index + 1}. {node.taskName}
                      </button>
                      <button
                        className="rounded-full border border-current px-1.5 py-0 text-[11px]"
                        onClick={() => removeVisualChainNode(node.id)}
                        title="Remove step"
                      >
                        ×
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </div>

        <div className="space-y-6">
          <StudioWorkflowLibrary
            workflowDefinitions={workflowDefinitions}
            workflowDefinitionsLoading={workflowDefinitionsLoading}
            workflowDefinitionsError={workflowDefinitionsError}
            workflowVersions={workflowVersions}
            workflowVersionsLoading={workflowVersionsLoading}
            workflowVersionsError={workflowVersionsError}
            workflowTriggers={workflowTriggers}
            workflowTriggersLoading={workflowTriggersLoading}
            workflowTriggersError={workflowTriggersError}
            workflowRuns={workflowRuns}
            workflowRunsLoading={workflowRunsLoading}
            workflowRunsError={workflowRunsError}
            activeWorkflowDefinitionId={activeWorkflowDefinitionId}
            activeWorkflowVersionId={activeWorkflowVersionId}
            deletingWorkflowDefinitionId={workflowDefinitionDeleteId}
            onRefresh={() => {
              void refreshWorkflowDefinitions();
              if (activeWorkflowDefinitionId) {
                void refreshWorkflowVersions(activeWorkflowDefinitionId);
                void refreshWorkflowTriggers(activeWorkflowDefinitionId);
                void refreshWorkflowRuns(activeWorkflowDefinitionId);
              }
            }}
            onOpenDefinition={restoreWorkflowDefinition}
            onDeleteDefinition={(definition) => {
              void deleteWorkflowDefinition(definition);
            }}
            onOpenVersion={(version) => {
              void restoreWorkflowVersion(version);
            }}
            onCreateManualTrigger={createManualWorkflowTrigger}
            onInvokeTrigger={(trigger) => {
              void invokeWorkflowTrigger(trigger);
            }}
          />

          <StudioNodeInspector
            selectedDagNode={selectedDagNode}
            selectedDagNodeStatus={selectedDagNodeStatus}
            inputFields={selectedDagNodeInspectorFields}
            selectedCapability={selectedCapability}
            outputSchemaFields={selectedDagNodeOutputSchemaFields}
            activeComposerIssueFocus={activeComposerIssueFocus}
            inspectorBindingRefs={inspectorBindingRefs}
            visualChainNodes={visualChainNodes}
            outputPathSuggestionsForNode={studioOutputPathSuggestionsForNode}
            contextPathSuggestions={contextPathSuggestions}
            workflowInterface={workflowInterface}
            autoWireNodeBindings={autoWireNodeBindings}
            quickFixNodeBindings={quickFixNodeBindings}
            setSelectedDagNodeId={setSelectedDagNodeId}
            capabilityIdOptionsId="studio-capability-id-options"
            onDeleteNode={removeVisualChainNode}
            updateNodeBasics={updateVisualChainNode}
            setVisualBindingMode={setVisualBindingMode}
            clearVisualBinding={clearVisualBinding}
            removeCustomInputField={removeCustomInputField}
            addCustomInputField={addCustomInputField}
            updateVisualBindingSourceNode={updateVisualBindingSourceNode}
            updateVisualBindingPath={updateVisualBindingPath}
            updateVisualBindingLiteral={updateVisualBindingLiteral}
            updateVisualBindingContextPath={updateVisualBindingContextPath}
            updateVisualBindingMemory={updateVisualBindingMemory}
            updateVisualBindingWorkflowInput={updateVisualBindingWorkflowInput}
            updateVisualBindingWorkflowVariable={updateVisualBindingWorkflowVariable}
            setVisualBindingFromPrevious={setVisualBindingFromPrevious}
            addNodeOutput={addNodeOutput}
            upsertNodeOutputFromSchema={upsertNodeOutputFromSchema}
            updateNodeOutput={updateNodeOutput}
            removeNodeOutput={removeNodeOutput}
            addNodeVariable={addNodeVariable}
            updateNodeVariable={updateNodeVariable}
            removeNodeVariable={removeNodeVariable}
            updateNodeControlConfig={updateNodeControlConfig}
            addSwitchCase={addSwitchCase}
            updateSwitchCase={updateSwitchCase}
            removeSwitchCase={removeSwitchCase}
          />

          <StudioCompilePanel
            compileLoading={composerCompileLoading || chainPreflightLoading}
            compileResult={composerCompileResult}
            preflightResult={chainPreflightResult}
            issues={composerIssues}
            draftPayloadPreview={draftPayloadPreview}
            onCompile={runChainPreflight}
          />
        </div>
      </div>

      <datalist id="studio-capability-id-options">
        {availableCapabilities.map((item) => (
          <option key={`studio-capability-id-option-${item.id}`} value={item.id} />
        ))}
      </datalist>
    </div>
  );
}
