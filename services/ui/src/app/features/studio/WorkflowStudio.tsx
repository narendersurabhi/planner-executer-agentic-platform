"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useDeferredValue, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import AppShell from "../../components/AppShell";
import ComposerDagCanvas from "../../components/composer/ComposerDagCanvas";
import ComposerValidationPanel from "../../components/composer/ComposerValidationPanel";
import {
  WorkflowNodeIcon,
  resolveWorkflowNodeVisual,
} from "../../components/workflow/WorkflowNodeIcon";
import StudioCapabilityPalette from "./StudioCapabilityPalette";
import StudioCompilePanel from "./StudioCompilePanel";
import StudioWorkbenchSurface from "./StudioWorkbenchSurface";
import StudioWorkbenchIcon from "./StudioWorkbenchIcon";
import StudioWorkflowInterfacePanel from "./StudioWorkflowInterfacePanel";
import StudioNodeInspector from "./StudioNodeInspector";
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
  StudioSurface,
  WorkflowBinding,
  WorkflowDefinition,
  WorkflowInputDefinition,
  WorkflowInterface,
  WorkflowOutputDefinition,
  WorkflowRun,
  WorkflowRunResult,
  WorkflowRuntimeSettings,
  WorkflowTrigger,
  WorkflowVariableDefinition,
  WorkflowVersion,
  WorkbenchWorkflowPromotionDraft,
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
const DEMO_DATA_ENABLED = process.env.NEXT_PUBLIC_DEMO_DATA === "true";
const MEMORY_USER_ID_KEY = "ape.memory.user_id.v1";
const STUDIO_WORKSPACE_LAYOUT_VERSION = 3;
const STUDIO_WORKSPACE_LAYOUT_STORAGE_KEY = `ape.studio.workspace_layout.v${STUDIO_WORKSPACE_LAYOUT_VERSION}`;
const LEGACY_FLOATING_STUDIO_LAYOUT_STORAGE_KEY = "ape.studio.workspace_layout.v2";
const DEFAULT_WORKSPACE_USER_ID = "narendersurabhi";
const DAG_CANVAS_ZOOM_MIN = 0.7;
const DAG_CANVAS_ZOOM_MAX = 1.5;
const DAG_CANVAS_ZOOM_STEP = 0.1;

const initialStudioDraft = (): ComposerDraft => ({
  summary: "Workflow Builder draft",
  nodes: [],
  edges: [],
});

const initialWorkflowInterface = (): WorkflowInterface => ({
  inputs: [],
  variables: [],
  outputs: [],
});

const initialWorkflowRuntimeSettings = (): WorkflowRuntimeSettings => ({
  executionMode: "static",
  adaptivePolicy: {
    maxReplans: 2,
  },
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

const initialStudioBottomTrayState = (): StudioBottomTrayState => ({
  activePanelId: "interface",
  height: 300,
  collapsed: false,
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

type DagConnectorDragState = {
  sourceNodeId: string;
  x: number;
  y: number;
  branchLabel?: string;
  sourcePortY?: number;
};

const dagNodeOutputAnchorY = (
  node: ComposerDraftNode,
  branchLabel?: string
) => {
  if (node.nodeKind !== "control") {
    return DAG_CANVAS_NODE_HEIGHT / 2;
  }
  if (node.controlKind === "if_else") {
    const normalized = String(branchLabel || "").trim().toLowerCase();
    if (normalized.includes("false") || normalized.includes("else")) {
      return 64;
    }
    if (normalized) {
      return 40;
    }
  }
  return DAG_CANVAS_NODE_HEIGHT / 2;
};

const normalizeDagBranchLabel = (branchLabel?: string) =>
  String(branchLabel || "").trim().toLowerCase();

const buildDagEdgeRoute = ({
  startX,
  startY,
  endX,
  endY,
  branchLabel,
}: {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  branchLabel?: string;
}) => {
  const normalizedBranchLabel = normalizeDagBranchLabel(branchLabel);
  const deltaX = endX - startX;
  const deltaY = endY - startY;
  const horizontalDistance = Math.max(Math.abs(deltaX), 1);
  const branchDirection =
    normalizedBranchLabel.includes("false") || normalizedBranchLabel.includes("else")
      ? 1
      : normalizedBranchLabel
        ? -1
        : 0;

  if (deltaX <= 72) {
    const controlPull = Math.max(42, Math.min(112, horizontalDistance * 0.45));
    const labelOffsetY = branchDirection < 0 ? -18 : 20;
    return {
      path: `M ${startX} ${startY} C ${startX + controlPull} ${startY}, ${endX - controlPull} ${endY}, ${endX} ${endY}`,
      midX: startX + deltaX / 2,
      midY: startY + deltaY / 2,
      labelX: startX + deltaX / 2,
      labelY: startY + deltaY / 2 + labelOffsetY,
    };
  }

  if (!normalizedBranchLabel) {
    const startLead = Math.min(96, Math.max(58, horizontalDistance * 0.24));
    const endLead = Math.min(118, Math.max(70, horizontalDistance * 0.3));
    return {
      path: `M ${startX} ${startY} C ${startX + startLead} ${startY}, ${endX - endLead} ${endY}, ${endX} ${endY}`,
      midX: startX + deltaX / 2,
      midY: startY + deltaY / 2,
      labelX: startX + deltaX / 2,
      labelY: startY + deltaY / 2,
    };
  }

  const branchOffset = Math.min(54, Math.max(28, Math.abs(deltaY) * 0.16 + 18));
  const branchY = startY + branchDirection * branchOffset;
  const splitX = startX + Math.min(78, Math.max(48, horizontalDistance * 0.16));
  const settleX = startX + Math.min(168, Math.max(112, horizontalDistance * 0.34));
  const arriveX = endX - Math.min(116, Math.max(62, horizontalDistance * 0.22));
  const labelOffsetY = branchDirection < 0 ? -18 : 20;

  return {
    path: [
      `M ${startX} ${startY}`,
      `C ${startX + 26} ${startY}, ${splitX - 16} ${startY}, ${splitX} ${branchY}`,
      `S ${settleX - 22} ${branchY}, ${settleX} ${branchY}`,
      `C ${settleX + 44} ${branchY}, ${arriveX} ${endY}, ${endX} ${endY}`,
    ].join(" "),
    midX: settleX + (endX - settleX) * 0.44,
    midY: branchY + (endY - branchY) * 0.56,
    labelX: splitX + (settleX - splitX) * 0.42,
    labelY: branchY + labelOffsetY,
  };
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

const clampDagCanvasZoom = (value: number) =>
  Math.min(DAG_CANVAS_ZOOM_MAX, Math.max(DAG_CANVAS_ZOOM_MIN, Math.round(value * 100) / 100));

const clampNumber = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const buildAutoLayoutPositions = (
  nodes: ComposerDraftNode[],
  edges: ComposerDraftEdge[]
): Record<string, CanvasPoint> => {
  if (nodes.length === 0) {
    return {};
  }

  const nodeIds = nodes.map((node) => node.id);
  const indegree = new Map<string, number>(nodeIds.map((id) => [id, 0]));
  const outgoing = new Map<string, string[]>(nodeIds.map((id) => [id, []]));
  const level = new Map<string, number>(nodeIds.map((id) => [id, 0]));

  edges.forEach((edge) => {
    if (!indegree.has(edge.fromNodeId) || !indegree.has(edge.toNodeId)) {
      return;
    }
    outgoing.get(edge.fromNodeId)?.push(edge.toNodeId);
    indegree.set(edge.toNodeId, (indegree.get(edge.toNodeId) || 0) + 1);
  });

  const queue = nodes
    .filter((node) => (indegree.get(node.id) || 0) === 0)
    .map((node) => node.id);
  const visited: string[] = [];

  while (queue.length > 0) {
    const current = queue.shift()!;
    visited.push(current);
    const currentLevel = level.get(current) || 0;
    (outgoing.get(current) || []).forEach((nextId) => {
      level.set(nextId, Math.max(level.get(nextId) || 0, currentLevel + 1));
      indegree.set(nextId, (indegree.get(nextId) || 0) - 1);
      if ((indegree.get(nextId) || 0) === 0) {
        queue.push(nextId);
      }
    });
  }

  if (visited.length !== nodes.length) {
    return Object.fromEntries(nodes.map((node, index) => [node.id, defaultDagNodePosition(index)]));
  }

  const columns = new Map<number, ComposerDraftNode[]>();
  nodes.forEach((node) => {
    const columnIndex = level.get(node.id) || 0;
    const columnNodes = columns.get(columnIndex) || [];
    columnNodes.push(node);
    columns.set(columnIndex, columnNodes);
  });

  const sortedColumns = Array.from(columns.entries()).sort(([left], [right]) => left - right);
  const columnGap = 128;
  const rowGap = 92;
  const layoutWidth =
    sortedColumns.length * DAG_CANVAS_NODE_WIDTH + Math.max(0, sortedColumns.length - 1) * columnGap;
  const startX = Math.max(DAG_CANVAS_PADDING + 36, Math.round((DAG_CANVAS_MIN_WIDTH - layoutWidth) / 2));

  const maxRows = Math.max(...sortedColumns.map(([, columnNodes]) => columnNodes.length));
  const layoutHeight = maxRows * DAG_CANVAS_NODE_HEIGHT + Math.max(0, maxRows - 1) * rowGap;
  const startY = Math.max(DAG_CANVAS_PADDING + 22, Math.round((DAG_CANVAS_MIN_HEIGHT - layoutHeight) / 2));

  const positions: Record<string, CanvasPoint> = {};
  sortedColumns.forEach(([columnIndex, columnNodes]) => {
    const columnHeight =
      columnNodes.length * DAG_CANVAS_NODE_HEIGHT +
      Math.max(0, columnNodes.length - 1) * rowGap;
    const columnStartY = Math.max(startY, Math.round((DAG_CANVAS_MIN_HEIGHT - columnHeight) / 2));
    columnNodes.forEach((node, rowIndex) => {
      positions[node.id] = {
        x: startX + columnIndex * (DAG_CANVAS_NODE_WIDTH + columnGap),
        y: columnStartY + rowIndex * (DAG_CANVAS_NODE_HEIGHT + rowGap),
      };
    });
  });

  return positions;
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
    runtimeSettings: (() => {
      const runtimeSettings =
        draft?.runtimeSettings ||
        (isRecord(draft) && "runtime_settings" in draft ? draft.runtime_settings : null);
      const adaptivePolicy =
        runtimeSettings &&
        typeof runtimeSettings === "object" &&
        !Array.isArray(runtimeSettings) &&
        "adaptivePolicy" in runtimeSettings
          ? (runtimeSettings as { adaptivePolicy?: { maxReplans?: unknown } }).adaptivePolicy
          : runtimeSettings &&
              typeof runtimeSettings === "object" &&
              !Array.isArray(runtimeSettings) &&
              "adaptive_policy" in runtimeSettings
            ? (runtimeSettings as { adaptive_policy?: { maxReplans?: unknown; max_replans?: unknown } })
                .adaptive_policy
            : null;
      const maxReplansValue =
        adaptivePolicy && typeof adaptivePolicy === "object" && !Array.isArray(adaptivePolicy)
          ? Number(
              (adaptivePolicy as { maxReplans?: unknown; max_replans?: unknown }).maxReplans ??
                (adaptivePolicy as { maxReplans?: unknown; max_replans?: unknown }).max_replans ??
                initialWorkflowRuntimeSettings().adaptivePolicy?.maxReplans ??
                2
            )
          : Number(initialWorkflowRuntimeSettings().adaptivePolicy?.maxReplans ?? 2);
      const normalizedMaxReplans = Number.isFinite(maxReplansValue)
        ? maxReplansValue
        : Number(initialWorkflowRuntimeSettings().adaptivePolicy?.maxReplans ?? 2);
      const executionModeValue =
        runtimeSettings &&
        typeof runtimeSettings === "object" &&
        !Array.isArray(runtimeSettings) &&
        ("executionMode" in runtimeSettings || "execution_mode" in runtimeSettings)
          ? ((runtimeSettings as { executionMode?: unknown; execution_mode?: unknown }).executionMode ??
              (runtimeSettings as { executionMode?: unknown; execution_mode?: unknown }).execution_mode)
          : "static";
      return {
        executionMode: executionModeValue === "adaptive" ? "adaptive" : "static",
        adaptivePolicy: {
          maxReplans: Math.max(0, Math.min(10, normalizedMaxReplans)),
        },
      } satisfies WorkflowRuntimeSettings;
    })(),
    composerDraft: {
      summary:
        typeof draft?.summary === "string" && draft.summary.trim()
          ? draft.summary
          : fallbackGoal || "Workflow Builder draft",
      nodes,
      edges: normalizeComposerEdges(nodes, normalizePersistedEdges(draft?.edges)),
    },
  };
};

type FloatingStudioPanelId =
  | "palette"
  | "compile"
  | "setup"
  | "interface"
  | "library"
  | "inspector";

type StudioPanelMode = "docked" | "floating";
type StudioDockZone = "left" | "right" | "bottom" | "overlay" | "none";
type StudioWorkspaceMode = "default" | "focus_graph";

type FloatingStudioPanelLayout = {
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  minimized: boolean;
  mode: StudioPanelMode;
  dockZone: StudioDockZone;
};

type FloatingStudioPanelDragState = {
  id: FloatingStudioPanelId;
  offsetX: number;
  offsetY: number;
};

type FloatingStudioPanelResizeDirection =
  | "n"
  | "e"
  | "s"
  | "w"
  | "ne"
  | "nw"
  | "se"
  | "sw";

type FloatingStudioPanelResizeState = {
  id: FloatingStudioPanelId;
  direction: FloatingStudioPanelResizeDirection;
  mode: StudioPanelMode;
  startX: number;
  startY: number;
  startLeft: number;
  startTop: number;
  startWidth: number;
  startHeight: number;
};

type StudioBottomTrayState = {
  activePanelId: FloatingStudioPanelId;
  height: number;
  collapsed: boolean;
};

type StudioBottomTrayResizeState = {
  startY: number;
  startHeight: number;
};

type StudioWorkspaceRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type ResolvedStudioWorkspacePanelRects = {
  graph: StudioWorkspaceRect;
  leftDock: StudioWorkspaceRect | null;
  rightDock: StudioWorkspaceRect | null;
  bottomDock: StudioWorkspaceRect | null;
  minimizedShelf: StudioWorkspaceRect | null;
};

const FLOATING_STUDIO_PANEL_IDS: FloatingStudioPanelId[] = [
  "palette",
  "compile",
  "setup",
  "interface",
  "library",
  "inspector",
];

const FLOATING_STUDIO_PANEL_HEADER_HEIGHT = 44;
const FLOATING_STUDIO_PANEL_MIN_WIDTH = 280;
const FLOATING_STUDIO_PANEL_MIN_HEIGHT = 180;
const FLOATING_STUDIO_PANEL_STAGE_PADDING = 18;
const FLOATING_STUDIO_PANEL_SNAP_DISTANCE = 28;
const STUDIO_MINIMIZED_SHELF_HEIGHT = 58;
const STUDIO_BOTTOM_TRAY_MIN_HEIGHT = 240;
const STUDIO_BOTTOM_TRAY_MAX_HEIGHT = 420;
const STUDIO_BOTTOM_TRAY_COLLAPSED_HEIGHT = 56;
const STUDIO_DOCK_PANEL_GAP = 14;

const STUDIO_DOCK_ZONE_DEFAULTS: Record<
  FloatingStudioPanelId,
  { mode: StudioPanelMode; dockZone: StudioDockZone }
> = {
  palette: { mode: "docked", dockZone: "left" },
  compile: { mode: "docked", dockZone: "right" },
  setup: { mode: "docked", dockZone: "right" },
  interface: { mode: "docked", dockZone: "bottom" },
  library: { mode: "docked", dockZone: "bottom" },
  inspector: { mode: "docked", dockZone: "right" },
};

const STUDIO_PANEL_DISPLAY_ORDER: FloatingStudioPanelId[] = [
  "palette",
  "inspector",
  "compile",
  "setup",
  "interface",
  "library",
];
const STUDIO_PANEL_TITLES: Record<FloatingStudioPanelId, string> = {
  palette: "Step Palette",
  compile: "Readiness Check",
  setup: "Workflow Settings",
  interface: "Request Contract",
  library: "Saved Workflows",
  inspector: "Step Inspector",
};
const STUDIO_DOCK_ZONE_LABELS: Record<Exclude<StudioDockZone, "overlay" | "none">, string> = {
  left: "Dock Left",
  right: "Dock Right",
  bottom: "Dock Bottom",
};
const STUDIO_PANEL_ALLOWED_DOCK_ZONES: Record<
  FloatingStudioPanelId,
  Exclude<StudioDockZone, "overlay" | "none">[]
> = {
  palette: ["left", "bottom"],
  compile: ["left", "right", "bottom"],
  setup: ["left", "right", "bottom"],
  interface: ["left", "right", "bottom"],
  library: ["left", "right", "bottom"],
  inspector: ["right", "bottom"],
};

const FLOATING_STUDIO_PANEL_RESIZE_HANDLES: {
  direction: FloatingStudioPanelResizeDirection;
  className: string;
}[] = [
  {
    direction: "n",
    className: "absolute inset-x-3 top-0 z-20 h-2 cursor-ns-resize",
  },
  {
    direction: "s",
    className: "absolute inset-x-3 bottom-0 z-20 h-2 cursor-ns-resize",
  },
  {
    direction: "w",
    className: "absolute inset-y-3 left-0 z-20 w-2 cursor-ew-resize",
  },
  {
    direction: "e",
    className: "absolute inset-y-3 right-0 z-20 w-2 cursor-ew-resize",
  },
  {
    direction: "nw",
    className: "absolute left-0 top-0 z-20 h-3 w-3 cursor-nwse-resize",
  },
  {
    direction: "ne",
    className: "absolute right-0 top-0 z-20 h-3 w-3 cursor-nesw-resize",
  },
  {
    direction: "sw",
    className: "absolute bottom-0 left-0 z-20 h-3 w-3 cursor-nesw-resize",
  },
  {
    direction: "se",
    className: "absolute bottom-0 right-0 z-20 h-3 w-3 cursor-nwse-resize",
  },
];

const getFloatingStudioPanelActiveHeight = (layout: FloatingStudioPanelLayout) =>
  layout.minimized ? FLOATING_STUDIO_PANEL_HEADER_HEIGHT : layout.height;

const buildDockedPanelLayouts = (
  stageWidth: number,
  stageHeight: number
): Record<FloatingStudioPanelId, FloatingStudioPanelLayout> => {
  const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const columnGap = STUDIO_DOCK_PANEL_GAP;
  const paletteWidth = Math.round(clampNumber(stageWidth * 0.2, 280, 320));
  const rightRailWidth = Math.round(clampNumber(stageWidth * 0.235, 320, 364));
  const compileHeight = 248;
  const setupHeight = Math.round(clampNumber(stageHeight * 0.36, 320, 420));
  const inspectorHeight = Math.round(clampNumber(stageHeight * 0.34, 280, 396));
  const bottomHeight = Math.round(clampNumber(stageHeight * 0.3, 250, 320));
  const bottomAvailableWidth = Math.max(
    FLOATING_STUDIO_PANEL_MIN_WIDTH * 2 + columnGap,
    stageWidth - padding * 2
  );
  const libraryWidth = Math.round(
    clampNumber(
      bottomAvailableWidth * 0.3,
      FLOATING_STUDIO_PANEL_MIN_WIDTH,
      Math.max(FLOATING_STUDIO_PANEL_MIN_WIDTH, bottomAvailableWidth - 420 - columnGap)
    )
  );
  const interfaceWidth = Math.max(
    420,
    Math.min(
      bottomAvailableWidth - libraryWidth - columnGap,
      bottomAvailableWidth - FLOATING_STUDIO_PANEL_MIN_WIDTH - columnGap
    )
  );
  const bottomY = Math.max(padding, stageHeight - bottomHeight - padding);
  const rightX = Math.max(padding, stageWidth - rightRailWidth - padding);

  return {
    palette: {
      x: padding,
      y: padding,
      width: paletteWidth,
      height: Math.max(520, stageHeight - bottomHeight - columnGap - padding * 2),
      zIndex: 2,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.palette,
    },
    compile: {
      x: rightX,
      y: padding,
      width: rightRailWidth,
      height: compileHeight,
      zIndex: 4,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.compile,
    },
    setup: {
      x: rightX,
      y: padding + compileHeight + columnGap,
      width: rightRailWidth,
      height: setupHeight,
      zIndex: 3,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.setup,
    },
    interface: {
      x: padding,
      y: bottomY,
      width: interfaceWidth,
      height: bottomHeight,
      zIndex: 2,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.interface,
    },
    library: {
      x: padding + interfaceWidth + columnGap,
      y: bottomY,
      width: libraryWidth,
      height: bottomHeight,
      zIndex: 2,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.library,
    },
    inspector: {
      x: rightX,
      y: padding,
      width: rightRailWidth,
      height: inspectorHeight,
      zIndex: 5,
      minimized: false,
      ...STUDIO_DOCK_ZONE_DEFAULTS.inspector,
    },
  };
};

const createInitialFloatingStudioPanelLayouts = (
  stageWidth: number,
  stageHeight: number
) => buildDockedPanelLayouts(stageWidth, stageHeight);

const clampDockedStudioPanelLayout = (
  layout: FloatingStudioPanelLayout,
  defaultLayout: FloatingStudioPanelLayout,
  stageWidth: number,
  stageHeight: number
): FloatingStudioPanelLayout => {
  const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const dockZone = layout.dockZone === "none" ? defaultLayout.dockZone : layout.dockZone;
  const maxSideWidth = Math.max(
    FLOATING_STUDIO_PANEL_MIN_WIDTH,
    stageWidth - padding * 2 - 220
  );
  const maxBottomWidth = Math.max(
    FLOATING_STUDIO_PANEL_MIN_WIDTH,
    stageWidth - padding * 2
  );
  const width =
    dockZone === "bottom"
      ? Math.min(
          maxBottomWidth,
          Math.max(FLOATING_STUDIO_PANEL_MIN_WIDTH, layout.width || defaultLayout.width)
        )
      : Math.min(
          maxSideWidth,
          Math.max(FLOATING_STUDIO_PANEL_MIN_WIDTH, layout.width || defaultLayout.width)
        );
  const minHeight = dockZone === "bottom" ? 220 : FLOATING_STUDIO_PANEL_HEADER_HEIGHT;
  const maxHeight = Math.max(minHeight, stageHeight - padding * 2);
  const height = Math.min(
    maxHeight,
    Math.max(minHeight, layout.height || defaultLayout.height)
  );

  return {
    ...defaultLayout,
    width,
    height,
    minimized: layout.minimized,
    mode: "docked",
    dockZone,
    zIndex: layout.zIndex,
  };
};

const clampFloatingStudioPanelLayout = (
  layout: FloatingStudioPanelLayout,
  stageWidth: number,
  stageHeight: number,
  measuredWidth?: number,
  measuredHeight?: number
): FloatingStudioPanelLayout => {
  const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const maxPanelWidth = Math.max(FLOATING_STUDIO_PANEL_MIN_WIDTH, stageWidth - padding * 2);
  const maxPanelHeight = Math.max(FLOATING_STUDIO_PANEL_MIN_HEIGHT, stageHeight - padding * 2);
  const panelWidth = Math.min(
    maxPanelWidth,
    Math.max(FLOATING_STUDIO_PANEL_MIN_WIDTH, layout.width)
  );
  const panelHeight = Math.min(
    maxPanelHeight,
    Math.max(FLOATING_STUDIO_PANEL_MIN_HEIGHT, layout.height)
  );
  const activeHeight = layout.minimized ? FLOATING_STUDIO_PANEL_HEADER_HEIGHT : panelHeight;
  const maxX = Math.max(padding, stageWidth - panelWidth - padding);
  const maxY = Math.max(padding, stageHeight - activeHeight - padding);
  return {
    ...layout,
    width: panelWidth,
    height: panelHeight,
    x: Math.min(maxX, Math.max(padding, layout.x)),
    y: Math.min(maxY, Math.max(padding, layout.y)),
  };
};

const syncFloatingStudioPanelLayoutToStage = (
  layout: FloatingStudioPanelLayout,
  defaultLayout: FloatingStudioPanelLayout,
  stageWidth: number,
  stageHeight: number,
  measuredWidth?: number,
  measuredHeight?: number
) => {
  if (layout.mode === "floating") {
    return clampFloatingStudioPanelLayout(
      layout,
      stageWidth,
      stageHeight,
      measuredWidth,
      measuredHeight
    );
  }
  return clampDockedStudioPanelLayout(layout, defaultLayout, stageWidth, stageHeight);
};

const pickFloatingStudioPanelSnapTarget = (
  value: number,
  targets: number[],
  threshold: number
) => {
  let closest = value;
  let closestDistance = threshold + 1;
  targets.forEach((target) => {
    const distance = Math.abs(value - target);
    if (distance <= threshold && distance < closestDistance) {
      closest = target;
      closestDistance = distance;
    }
  });
  return closest;
};

const resizeFloatingStudioPanelLayout = (
  layout: FloatingStudioPanelLayout,
  panelId: FloatingStudioPanelId,
  resizeState: FloatingStudioPanelResizeState,
  pointerX: number,
  pointerY: number,
  stageWidth: number,
  stageHeight: number
): FloatingStudioPanelLayout => {
  if (layout.mode === "docked") {
    if (layout.dockZone !== "left" && layout.dockZone !== "right") {
      return layout;
    }
    const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
    const maxSideWidth = Math.max(
      FLOATING_STUDIO_PANEL_MIN_WIDTH,
      stageWidth - padding * 2 - 220
    );
    const deltaX = pointerX - resizeState.startX;
    const width = clampNumber(
      resizeState.startWidth + (layout.dockZone === "left" ? deltaX : -deltaX),
      FLOATING_STUDIO_PANEL_MIN_WIDTH,
      maxSideWidth
    );
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    return clampDockedStudioPanelLayout(
      {
        ...layout,
        width,
      },
      layout.dockZone === defaults[panelId].dockZone
        ? defaults[panelId]
        : { ...defaults[panelId], dockZone: layout.dockZone },
      stageWidth,
      stageHeight
    );
  }

  const stageMinX = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const stageMinY = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const stageMaxX = Math.max(stageMinX, stageWidth - FLOATING_STUDIO_PANEL_STAGE_PADDING);
  const stageMaxY = Math.max(stageMinY, stageHeight - FLOATING_STUDIO_PANEL_STAGE_PADDING);
  const deltaX = pointerX - resizeState.startX;
  const deltaY = pointerY - resizeState.startY;
  const startRight = resizeState.startLeft + resizeState.startWidth;
  const startBottom = resizeState.startTop + resizeState.startHeight;
  let left = resizeState.startLeft;
  let top = resizeState.startTop;
  let right = startRight;
  let bottom = startBottom;

  if (resizeState.direction.includes("w")) {
    left = Math.max(
      stageMinX,
      Math.min(startRight - FLOATING_STUDIO_PANEL_MIN_WIDTH, resizeState.startLeft + deltaX)
    );
  }

  if (resizeState.direction.includes("e")) {
    right = Math.min(
      stageMaxX,
      Math.max(resizeState.startLeft + FLOATING_STUDIO_PANEL_MIN_WIDTH, startRight + deltaX)
    );
  }

  if (resizeState.direction.includes("n")) {
    top = Math.max(
      stageMinY,
      Math.min(startBottom - FLOATING_STUDIO_PANEL_MIN_HEIGHT, resizeState.startTop + deltaY)
    );
  }

  if (resizeState.direction.includes("s")) {
    bottom = Math.min(
      stageMaxY,
      Math.max(resizeState.startTop + FLOATING_STUDIO_PANEL_MIN_HEIGHT, startBottom + deltaY)
    );
  }

  return clampFloatingStudioPanelLayout(
    {
      ...layout,
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      minimized: false,
      mode: "floating",
    },
    stageWidth,
    stageHeight
  );
};

const snapFloatingStudioPanelLayout = (
  panelId: FloatingStudioPanelId,
  layout: FloatingStudioPanelLayout,
  stageWidth: number,
  stageHeight: number
) => {
  const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
  const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const activeHeight = getFloatingStudioPanelActiveHeight(layout);
  const rightDockX = Math.max(padding, stageWidth - layout.width - padding);
  const bottomDockY = Math.max(padding, stageHeight - activeHeight - padding);
  const centeredDockX = Math.max(
    padding,
    Math.min((stageWidth - layout.width) / 2, stageWidth - layout.width - padding)
  );
  const snappedX = pickFloatingStudioPanelSnapTarget(
    layout.x,
    [padding, defaults[panelId].x, centeredDockX, rightDockX],
    FLOATING_STUDIO_PANEL_SNAP_DISTANCE
  );
  const snappedY = pickFloatingStudioPanelSnapTarget(
    layout.y,
    [padding, defaults[panelId].y, bottomDockY],
    FLOATING_STUDIO_PANEL_SNAP_DISTANCE
  );
  return clampFloatingStudioPanelLayout(
    {
      ...layout,
      x: snappedX,
      y: snappedY,
    },
    stageWidth,
    stageHeight
  );
};

type PersistedStudioWorkspaceLayout = {
  version: number;
  workspaceMode?: StudioWorkspaceMode;
  bottomTray?: {
    activePanelId?: string;
    height?: number;
    collapsed?: boolean;
  };
  panels: Record<string, Partial<FloatingStudioPanelLayout>>;
};

const parsePersistedStudioPanelLayouts = (
  parsed: unknown
): Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null => {
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  const candidatePanels = parsed as Record<string, Partial<FloatingStudioPanelLayout>>;
  return FLOATING_STUDIO_PANEL_IDS.reduce<Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null>(
    (acc, panelId) => {
      if (!acc) {
        return null;
      }
      const candidate = candidatePanels[panelId];
      if (!candidate) {
        return null;
      }
      const defaults = STUDIO_DOCK_ZONE_DEFAULTS[panelId];
      const { x, y, width, height, zIndex, minimized, mode, dockZone } = candidate;
      if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(width) ||
        !Number.isFinite(height) ||
        !Number.isFinite(zIndex) ||
        typeof minimized !== "boolean"
      ) {
        return null;
      }
      acc[panelId] = {
        x: Number(x),
        y: Number(y),
        width: Number(width),
        height: Number(height),
        zIndex: Number(zIndex),
        minimized,
        mode: mode === "floating" || mode === "docked" ? mode : defaults.mode,
        dockZone:
          dockZone === "left" ||
          dockZone === "right" ||
          dockZone === "bottom" ||
          dockZone === "overlay" ||
          dockZone === "none"
            ? dockZone
            : defaults.dockZone,
      };
      return acc;
    },
    {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
  );
};

const readPersistedStudioWorkspaceLayout = (raw: string | null) => {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as PersistedStudioWorkspaceLayout | Record<string, unknown>;
    if (
      parsed &&
      typeof parsed === "object" &&
      "version" in parsed &&
      Number((parsed as PersistedStudioWorkspaceLayout).version) === STUDIO_WORKSPACE_LAYOUT_VERSION &&
      "panels" in parsed
    ) {
      const layout = parsed as PersistedStudioWorkspaceLayout;
      const panels = parsePersistedStudioPanelLayouts(layout.panels);
      if (!panels) {
        return null;
      }
      const traySource = layout.bottomTray;
      const trayHeight = Number(traySource?.height);
      return {
        panels,
        workspaceMode: layout.workspaceMode === "focus_graph" ? "focus_graph" : "default",
        bottomTray: {
          activePanelId:
            traySource?.activePanelId &&
            FLOATING_STUDIO_PANEL_IDS.includes(traySource.activePanelId as FloatingStudioPanelId)
              ? (traySource.activePanelId as FloatingStudioPanelId)
              : initialStudioBottomTrayState().activePanelId,
          height: Number.isFinite(trayHeight)
            ? clampNumber(trayHeight, STUDIO_BOTTOM_TRAY_MIN_HEIGHT, STUDIO_BOTTOM_TRAY_MAX_HEIGHT)
            : initialStudioBottomTrayState().height,
          collapsed: Boolean(traySource?.collapsed),
        } satisfies StudioBottomTrayState,
      };
    }
    const panels = parsePersistedStudioPanelLayouts(parsed);
    if (!panels) {
      return null;
    }
    return {
      panels,
      workspaceMode: "default" as const,
      bottomTray: initialStudioBottomTrayState(),
    };
  } catch {
    return null;
  }
};

const createPersistedStudioWorkspaceLayout = ({
  panels,
  bottomTray,
}: {
  panels: Record<FloatingStudioPanelId, FloatingStudioPanelLayout>;
  bottomTray: StudioBottomTrayState;
}): PersistedStudioWorkspaceLayout => ({
  version: STUDIO_WORKSPACE_LAYOUT_VERSION,
  workspaceMode: "default",
  bottomTray: {
    activePanelId: bottomTray.activePanelId,
    height: bottomTray.height,
    collapsed: bottomTray.collapsed,
  },
  panels,
});

const readPersistedFloatingStudioPanelLayouts = (
  raw: string | null
): Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null => {
  const restored = readPersistedStudioWorkspaceLayout(raw);
  return restored?.panels || null;
};

const readLegacyFloatingStudioPanelLayouts = (
  raw: string | null
): Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null => {
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, Partial<FloatingStudioPanelLayout>>;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return FLOATING_STUDIO_PANEL_IDS.reduce<Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null>(
      (acc, panelId) => {
        if (!acc) {
          return null;
        }
        const candidate = parsed[panelId];
        if (!candidate) {
          return null;
        }
        const defaults = STUDIO_DOCK_ZONE_DEFAULTS[panelId];
        const { x, y, width, height, zIndex, minimized, mode, dockZone } = candidate;
        if (
          !Number.isFinite(x) ||
          !Number.isFinite(y) ||
          !Number.isFinite(width) ||
          !Number.isFinite(height) ||
          !Number.isFinite(zIndex) ||
          typeof minimized !== "boolean"
        ) {
          return null;
        }
        acc[panelId] = {
          x: Number(x),
          y: Number(y),
          width: Number(width),
          height: Number(height),
          zIndex: Number(zIndex),
          minimized,
          mode: mode === "floating" || mode === "docked" ? mode : defaults.mode,
          dockZone:
            dockZone === "left" ||
            dockZone === "right" ||
            dockZone === "bottom" ||
            dockZone === "overlay" ||
            dockZone === "none"
              ? dockZone
              : defaults.dockZone,
        };
        return acc;
      },
      {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
    );
  } catch {
    return null;
  }
};

const resolveWorkspacePanelRects = (
  layouts: Record<FloatingStudioPanelId, FloatingStudioPanelLayout>,
  stageWidth: number,
  stageHeight: number,
  options: {
    showInspector: boolean;
    showMinimizedShelf: boolean;
    showBottomTray: boolean;
    bottomTrayHeight: number;
  }
): ResolvedStudioWorkspacePanelRects => {
  const padding = FLOATING_STUDIO_PANEL_STAGE_PADDING;
  const gap = STUDIO_DOCK_PANEL_GAP;
  const shouldShowDockedPanel = (panelId: FloatingStudioPanelId) =>
    layouts[panelId].mode === "docked" &&
    !layouts[panelId].minimized &&
    (panelId !== "inspector" || options.showInspector);

  const dockedLeftPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) => shouldShowDockedPanel(panelId) && layouts[panelId].dockZone === "left"
  );
  const dockedRightPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) => shouldShowDockedPanel(panelId) && layouts[panelId].dockZone === "right"
  );
  const dockedBottomPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) => shouldShowDockedPanel(panelId) && layouts[panelId].dockZone === "bottom"
  );

  const leftDockWidth = dockedLeftPanelIds.reduce(
    (maxWidth, panelId) => Math.max(maxWidth, layouts[panelId].width),
    0
  );
  const rightDockWidth = dockedRightPanelIds.reduce(
    (maxWidth, panelId) => Math.max(maxWidth, layouts[panelId].width),
    0
  );
  const bottomDockHeight = options.showBottomTray ? options.bottomTrayHeight : 0;
  const minimizedShelfHeight = options.showMinimizedShelf ? STUDIO_MINIMIZED_SHELF_HEIGHT : 0;

  const graphX = padding + (leftDockWidth ? leftDockWidth + gap : 0);
  const graphY = padding;
  const graphBottom =
    stageHeight -
    padding -
    (minimizedShelfHeight ? minimizedShelfHeight + gap : 0) -
    (bottomDockHeight ? bottomDockHeight + gap : 0);
  const graphWidth = Math.max(
    420,
    stageWidth - graphX - padding - (rightDockWidth ? rightDockWidth + gap : 0)
  );
  const graphHeight = Math.max(360, graphBottom - graphY);
  const bottomDockY = graphY + graphHeight + gap;
  const minimizedShelfY =
    stageHeight - padding - (minimizedShelfHeight || 0);

  return {
    graph: {
      x: graphX,
      y: graphY,
      width: graphWidth,
      height: graphHeight,
    },
    leftDock: leftDockWidth
      ? {
          x: padding,
          y: graphY,
          width: leftDockWidth,
          height: graphHeight,
        }
      : null,
    rightDock: rightDockWidth
      ? {
          x: stageWidth - padding - rightDockWidth,
          y: graphY,
          width: rightDockWidth,
          height: graphHeight,
        }
      : null,
    bottomDock: bottomDockHeight
      ? {
          x: padding,
          y: bottomDockY,
          width: stageWidth - padding * 2,
          height: bottomDockHeight,
        }
      : null,
    minimizedShelf: minimizedShelfHeight
      ? {
          x: padding,
          y: minimizedShelfY,
          width: stageWidth - padding * 2,
          height: minimizedShelfHeight,
        }
      : null,
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
  const [workflowRuntimeSettings, setWorkflowRuntimeSettings] = useState<WorkflowRuntimeSettings>(
    initialWorkflowRuntimeSettings
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
  const [demoDataDetected, setDemoDataDetected] = useState(DEMO_DATA_ENABLED);
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
    "save" | "publish" | "run" | null
  >(null);
  const [pendingWorkbenchWorkflowDraft, setPendingWorkbenchWorkflowDraft] =
    useState<WorkbenchWorkflowPromotionDraft | null>(null);
  const [activeComposerIssueFocus, setActiveComposerIssueFocus] = useState<ComposerIssueFocus | null>(
    null
  );
  const [composerNodePositions, setComposerNodePositions] = useState<Record<string, CanvasPoint>>({});
  const [hoveredDagEdgeKey, setHoveredDagEdgeKey] = useState<string | null>(null);
  const [dagEdgeDraftSourceNodeId, setDagEdgeDraftSourceNodeId] = useState<string | null>(null);
  const [dagConnectorDrag, setDagConnectorDrag] = useState<DagConnectorDragState | null>(null);
  const [dagCanvasDraggingNodeId, setDagCanvasDraggingNodeId] = useState<string | null>(null);
  const [dagConnectorHoverTargetNodeId, setDagConnectorHoverTargetNodeId] = useState<string | null>(
    null
  );
  const [dagCanvasZoom, setDagCanvasZoom] = useState(1);
  const [studioWorkspaceMode, setStudioWorkspaceMode] =
    useState<StudioWorkspaceMode>("default");
  const [workflowSetupExpanded, setWorkflowSetupExpanded] = useState(false);
  const [studioBottomTray, setStudioBottomTray] = useState<StudioBottomTrayState>(
    initialStudioBottomTrayState
  );
  const [floatingStudioPanelDrag, setFloatingStudioPanelDrag] =
    useState<FloatingStudioPanelDragState | null>(null);
  const [floatingStudioPanelResize, setFloatingStudioPanelResize] =
    useState<FloatingStudioPanelResizeState | null>(null);
  const [studioBottomTrayResize, setStudioBottomTrayResize] =
    useState<StudioBottomTrayResizeState | null>(null);
  const [studioWorkspaceStageSize, setStudioWorkspaceStageSize] = useState({
    width: 0,
    height: 0,
  });
  const [floatingStudioPanels, setFloatingStudioPanels] = useState<
    Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
  >(() => createInitialFloatingStudioPanelLayouts(1440, 1040));
  const [activeStudioPanelMenuId, setActiveStudioPanelMenuId] =
    useState<FloatingStudioPanelId | null>(null);

  const dagCanvasDragOffsetRef = useRef<CanvasPoint>({ x: 0, y: 0 });
  const dagCanvasViewportRef = useRef<HTMLDivElement | null>(null);
  const dagCanvasRef = useRef<HTMLDivElement | null>(null);
  const inspectorBindingRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const studioWorkspaceStageRef = useRef<HTMLDivElement | null>(null);
  const floatingStudioPanelRefs = useRef<Record<FloatingStudioPanelId, HTMLDivElement | null>>({
    palette: null,
    compile: null,
    setup: null,
    interface: null,
    library: null,
    inspector: null,
  });
  const floatingStudioPanelsInitializedRef = useRef(false);
  const persistedFloatingStudioPanelsRef = useRef<
    Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null
  >(null);
  const persistedStudioBottomTrayRef = useRef<StudioBottomTrayState | null>(null);
  const focusGraphPanelSnapshotRef = useRef<
    Record<FloatingStudioPanelId, FloatingStudioPanelLayout> | null
  >(null);
  const router = useRouter();
  const pathname = usePathname() || "/studio";
  const searchParams = useSearchParams();
  const requestedStudioMode = String(searchParams.get("mode") || "").trim();
  const requestedStudioSurface = String(searchParams.get("surface") || "")
    .trim()
    .toLowerCase();
  const activeStudioSurface: StudioSurface =
    requestedStudioSurface === "workbench" ? "workbench" : "workflow";
  const requestedWorkflowDefinitionId = String(searchParams.get("definition") || "").trim();
  const requestedWorkflowVersionId = String(searchParams.get("version") || "").trim();
  const handledStudioRouteSelectionRef = useRef("");
  const handledDemoWorkflowSelectionRef = useRef(false);
  const [workbenchSurfaceMounted, setWorkbenchSurfaceMounted] = useState(
    activeStudioSurface === "workbench"
  );

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
    if (activeStudioSurface === "workbench") {
      setWorkbenchSurfaceMounted(true);
    }
  }, [activeStudioSurface]);

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

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const restored =
      readPersistedStudioWorkspaceLayout(
        window.localStorage.getItem(STUDIO_WORKSPACE_LAYOUT_STORAGE_KEY)
      ) ||
      readPersistedStudioWorkspaceLayout(
        window.localStorage.getItem(LEGACY_FLOATING_STUDIO_LAYOUT_STORAGE_KEY)
      ) || {
        panels: readLegacyFloatingStudioPanelLayouts(
          window.localStorage.getItem(LEGACY_FLOATING_STUDIO_LAYOUT_STORAGE_KEY)
        ),
        workspaceMode: "default" as const,
        bottomTray: initialStudioBottomTrayState(),
      };
    persistedFloatingStudioPanelsRef.current = restored?.panels || null;
    persistedStudioBottomTrayRef.current = restored?.bottomTray || initialStudioBottomTrayState();
    setStudioWorkspaceMode(restored?.workspaceMode === "focus_graph" ? "focus_graph" : "default");
  }, []);
  const contextPathSuggestions = useMemo(
    () => collectContextPathSuggestions(contextState.context),
    [contextState.context]
  );
  const activeWorkflowDefinitionId = savedWorkflowDefinition?.id || null;
  const activeWorkflowVersionId = loadedWorkflowVersionId || publishedWorkflowVersion?.id || null;
  const isTypingTarget = (target: EventTarget | null) => {
    const element = target instanceof HTMLElement ? target : null;
    if (!element) {
      return false;
    }
    const tagName = element.tagName.toLowerCase();
    return (
      element.isContentEditable ||
      tagName === "input" ||
      tagName === "textarea" ||
      tagName === "select" ||
      tagName === "button" ||
      Boolean(element.closest("[contenteditable='true']"))
    );
  };

  useEffect(() => {
    if (contextState.invalid) {
      setWorkflowSetupExpanded(true);
    }
  }, [contextState.invalid]);

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
      if (response.headers.get("x-demo-data") === "true") {
        setDemoDataDetected(true);
      }
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
      const missingNodeIds = visualChainNodes.filter((node) => !next[node.id]).map((node) => node.id);
      const staleNodeIds = Object.keys(next).filter(
        (nodeId) => !visualChainNodes.some((node) => node.id === nodeId)
      );

      if (missingNodeIds.length === 0 && staleNodeIds.length === 0) {
        return prev;
      }

      if (Object.keys(prev).length === 0 && visualChainNodes.length > 0) {
        return buildAutoLayoutPositions(visualChainNodes, composerDraftEdges);
      }

      visualChainNodes.forEach((node, index) => {
        if (!next[node.id]) {
          next[node.id] = defaultDagNodePosition(index);
        }
      });
      staleNodeIds.forEach((nodeId) => {
        delete next[nodeId];
      });
      return next;
    });
  }, [composerDraftEdges, visualChainNodes]);

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
      const pointerX = Math.max(0, (event.clientX - rect.left) / dagCanvasZoom);
      const pointerY = Math.max(0, (event.clientY - rect.top) / dagCanvasZoom);
      if (dagCanvasDraggingNodeId) {
        const rawX = pointerX - dagCanvasDragOffsetRef.current.x;
        const rawY = pointerY - dagCanvasDragOffsetRef.current.y;
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
                x: pointerX,
                y: pointerY,
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
  }, [dagCanvasDraggingNodeId, dagCanvasZoom, dagConnectorDrag]);

  useEffect(() => {
    const stage = studioWorkspaceStageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") {
      return;
    }
    const updateSize = () => {
      const rect = stage.getBoundingClientRect();
      setStudioWorkspaceStageSize({
        width: rect.width,
        height: rect.height,
      });
    };
    updateSize();
    const observer = new ResizeObserver(() => {
      updateSize();
    });
    observer.observe(stage);
    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!studioWorkspaceStageSize.width || !studioWorkspaceStageSize.height) {
      return;
    }
    setFloatingStudioPanels((prev) => {
      const defaults = createInitialFloatingStudioPanelLayouts(
        studioWorkspaceStageSize.width,
        studioWorkspaceStageSize.height
      );
      const seeded = floatingStudioPanelsInitializedRef.current
        ? prev
        : persistedFloatingStudioPanelsRef.current || defaults;
      floatingStudioPanelsInitializedRef.current = true;
      return FLOATING_STUDIO_PANEL_IDS.reduce<
        Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
      >((acc, panelId) => {
        const measured = floatingStudioPanelRefs.current[panelId];
        acc[panelId] = syncFloatingStudioPanelLayoutToStage(
          seeded[panelId],
          defaults[panelId],
          studioWorkspaceStageSize.width,
          studioWorkspaceStageSize.height,
          measured?.offsetWidth,
          measured?.offsetHeight
        );
        return acc;
      }, {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>);
    });
    if (persistedStudioBottomTrayRef.current) {
      setStudioBottomTray((prev) => ({
        activePanelId: persistedStudioBottomTrayRef.current?.activePanelId || prev.activePanelId,
        height: clampNumber(
          persistedStudioBottomTrayRef.current?.height || prev.height,
          STUDIO_BOTTOM_TRAY_MIN_HEIGHT,
          Math.min(STUDIO_BOTTOM_TRAY_MAX_HEIGHT, studioWorkspaceStageSize.height - 220)
        ),
        collapsed: Boolean(persistedStudioBottomTrayRef.current?.collapsed),
      }));
      persistedStudioBottomTrayRef.current = null;
    } else {
      setStudioBottomTray((prev) => ({
        ...prev,
        height: clampNumber(
          prev.height,
          STUDIO_BOTTOM_TRAY_MIN_HEIGHT,
          Math.min(STUDIO_BOTTOM_TRAY_MAX_HEIGHT, studioWorkspaceStageSize.height - 220)
        ),
      }));
    }
  }, [studioWorkspaceStageSize.height, studioWorkspaceStageSize.width]);

  useEffect(() => {
    if (typeof window === "undefined" || !floatingStudioPanelsInitializedRef.current) {
      return;
    }
    const saveTimer = window.setTimeout(() => {
      const panelsToPersist =
        studioWorkspaceMode === "focus_graph" && focusGraphPanelSnapshotRef.current
          ? focusGraphPanelSnapshotRef.current
          : floatingStudioPanels;
      window.localStorage.setItem(
        STUDIO_WORKSPACE_LAYOUT_STORAGE_KEY,
        JSON.stringify(
          createPersistedStudioWorkspaceLayout({
            panels: panelsToPersist,
            bottomTray: studioBottomTray,
          })
        )
      );
    }, 120);
    return () => {
      window.clearTimeout(saveTimer);
    };
  }, [floatingStudioPanels, studioBottomTray, studioWorkspaceMode]);

  useEffect(() => {
    if (!activeStudioPanelMenuId) {
      return;
    }
    const handleMouseDown = (event: MouseEvent) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (target?.closest(`[data-studio-panel-menu-root="${activeStudioPanelMenuId}"]`)) {
        return;
      }
      setActiveStudioPanelMenuId(null);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setActiveStudioPanelMenuId(null);
      }
    };
    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeStudioPanelMenuId]);

  useEffect(() => {
    if (!floatingStudioPanelDrag) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const stage = studioWorkspaceStageRef.current;
      if (!stage) {
        return;
      }
      const stageRect = stage.getBoundingClientRect();
      setFloatingStudioPanels((prev) => {
        const current = prev[floatingStudioPanelDrag.id];
        const measured = floatingStudioPanelRefs.current[floatingStudioPanelDrag.id];
        return {
          ...prev,
          [floatingStudioPanelDrag.id]: clampFloatingStudioPanelLayout(
            {
              ...current,
              x: event.clientX - stageRect.left - floatingStudioPanelDrag.offsetX,
              y: event.clientY - stageRect.top - floatingStudioPanelDrag.offsetY,
            },
            stageRect.width,
            stageRect.height,
            measured?.offsetWidth,
            measured?.offsetHeight
          ),
        };
      });
    };

    const handleUp = () => {
      const stage = studioWorkspaceStageRef.current;
      if (stage) {
        const stageRect = stage.getBoundingClientRect();
        setFloatingStudioPanels((prev) => ({
          ...prev,
          [floatingStudioPanelDrag.id]: snapFloatingStudioPanelLayout(
            floatingStudioPanelDrag.id,
            prev[floatingStudioPanelDrag.id],
            stageRect.width,
            stageRect.height
          ),
        }));
      }
      setFloatingStudioPanelDrag(null);
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [floatingStudioPanelDrag]);

  useEffect(() => {
    if (!floatingStudioPanelResize) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const stage = studioWorkspaceStageRef.current;
      if (!stage) {
        return;
      }
      const stageRect = stage.getBoundingClientRect();
      setFloatingStudioPanels((prev) => {
        const current = prev[floatingStudioPanelResize.id];
        if (
          floatingStudioPanelResize.mode === "docked" &&
          current.mode === "docked" &&
          (current.dockZone === "left" || current.dockZone === "right")
        ) {
          const resizedLayout = resizeFloatingStudioPanelLayout(
            current,
            floatingStudioPanelResize.id,
            floatingStudioPanelResize,
            event.clientX,
            event.clientY,
            stageRect.width,
            stageRect.height
          );
          return FLOATING_STUDIO_PANEL_IDS.reduce<
            Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
          >((acc, panelId) => {
            const panel = prev[panelId];
            acc[panelId] =
              panel.mode === "docked" && panel.dockZone === current.dockZone
                ? {
                    ...panel,
                    width: resizedLayout.width,
                  }
                : panel;
            return acc;
          }, {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>);
        }
        return {
          ...prev,
          [floatingStudioPanelResize.id]: resizeFloatingStudioPanelLayout(
            current,
            floatingStudioPanelResize.id,
            floatingStudioPanelResize,
            event.clientX,
            event.clientY,
            stageRect.width,
            stageRect.height
          ),
        };
      });
    };

    const handleUp = () => {
      setFloatingStudioPanelResize(null);
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [floatingStudioPanelResize]);

  useEffect(() => {
    if (!studioBottomTrayResize) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const stage = studioWorkspaceStageRef.current;
      if (!stage) {
        return;
      }
      const stageRect = stage.getBoundingClientRect();
      setStudioBottomTray((prev) => ({
        ...prev,
        collapsed: false,
        height: clampNumber(
          studioBottomTrayResize.startHeight + (studioBottomTrayResize.startY - event.clientY),
          STUDIO_BOTTOM_TRAY_MIN_HEIGHT,
          Math.min(STUDIO_BOTTOM_TRAY_MAX_HEIGHT, stageRect.height - 220)
        ),
      }));
    };

    const handleUp = () => {
      setStudioBottomTrayResize(null);
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [studioBottomTrayResize]);

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
    const scaledNodeWidth = DAG_CANVAS_NODE_WIDTH * dagCanvasZoom;
    const scaledNodeHeight = DAG_CANVAS_NODE_HEIGHT * dagCanvasZoom;
    const left = Math.max(
      0,
      position.x * dagCanvasZoom - Math.max(0, viewport.clientWidth - scaledNodeWidth) / 2
    );
    const top = Math.max(
      0,
      position.y * dagCanvasZoom - Math.max(0, viewport.clientHeight - scaledNodeHeight) / 2
    );
    viewport.scrollTo({ left, top, behavior: "smooth" });
  };

  const setDagCanvasZoomLevel = (nextZoom: number) => {
    const normalizedZoom = clampDagCanvasZoom(nextZoom);
    if (normalizedZoom === dagCanvasZoom) {
      return;
    }
    const viewport = dagCanvasViewportRef.current;
    const focusX = viewport ? (viewport.scrollLeft + viewport.clientWidth / 2) / dagCanvasZoom : null;
    const focusY = viewport ? (viewport.scrollTop + viewport.clientHeight / 2) / dagCanvasZoom : null;
    setDagCanvasZoom(normalizedZoom);
    if (focusX !== null && focusY !== null) {
      requestAnimationFrame(() => {
        const nextViewport = dagCanvasViewportRef.current;
        if (!nextViewport) {
          return;
        }
        nextViewport.scrollTo({
          left: Math.max(0, focusX * normalizedZoom - nextViewport.clientWidth / 2),
          top: Math.max(0, focusY * normalizedZoom - nextViewport.clientHeight / 2),
        });
      });
    }
  };

  const zoomInDagCanvas = () => {
    const nextZoom = clampDagCanvasZoom(dagCanvasZoom + DAG_CANVAS_ZOOM_STEP);
    setDagCanvasZoomLevel(nextZoom);
    if (nextZoom !== dagCanvasZoom) {
      setStudioNotice(`Canvas zoom set to ${Math.round(nextZoom * 100)}%.`);
    }
  };

  const zoomOutDagCanvas = () => {
    const nextZoom = clampDagCanvasZoom(dagCanvasZoom - DAG_CANVAS_ZOOM_STEP);
    setDagCanvasZoomLevel(nextZoom);
    if (nextZoom !== dagCanvasZoom) {
      setStudioNotice(`Canvas zoom set to ${Math.round(nextZoom * 100)}%.`);
    }
  };

  const resetDagCanvasZoom = () => {
    setDagCanvasZoomLevel(1);
    setStudioNotice("Canvas zoom reset to 100%.");
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

  const addDagEdge = (fromNodeId: string, toNodeId: string, branchLabel?: string) => {
    if (!fromNodeId || !toNodeId || fromNodeId === toNodeId) {
      return;
    }
    setComposerDraft((prev) => {
      const sourceNode = prev.nodes.find((node) => node.id === fromNodeId);
      const nextBranchLabel =
        String(branchLabel || "").trim() || defaultBranchLabelForSourceNode(sourceNode, prev.edges);
      return {
        ...prev,
        edges: normalizeComposerEdges(prev.nodes, [
          ...prev.edges,
          { fromNodeId, toNodeId, ...(nextBranchLabel ? { branchLabel: nextBranchLabel } : {}) },
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
      x: Math.max(0, (event.clientX - rect.left) / dagCanvasZoom) - current.x,
      y: Math.max(0, (event.clientY - rect.top) / dagCanvasZoom) - current.y,
    };
    setDagCanvasDraggingNodeId(nodeId);
  };

  const beginDagConnectorDrag = (
    event: React.MouseEvent<HTMLButtonElement>,
    sourceNodeId: string,
    options: { branchLabel?: string; sourcePortY?: number } = {}
  ) => {
    event.preventDefault();
    const canvas = dagCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    setDagConnectorDrag({
      sourceNodeId,
      x: Math.max(0, (event.clientX - rect.left) / dagCanvasZoom),
      y: Math.max(0, (event.clientY - rect.top) / dagCanvasZoom),
      ...(options.branchLabel ? { branchLabel: options.branchLabel } : {}),
      sourcePortY: options.sourcePortY ?? DAG_CANVAS_NODE_HEIGHT / 2,
    });
    setDagConnectorHoverTargetNodeId(null);
    setDagEdgeDraftSourceNodeId(sourceNodeId);
  };

  const autoLayoutDagCanvas = () => {
    setComposerNodePositions(() => {
      return buildAutoLayoutPositions(visualChainNodes, composerDraftEdges);
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
          const startY = fromEntry.position.y + dagNodeOutputAnchorY(fromEntry.node, edge.branchLabel);
          const endX = toEntry.position.x;
          const endY = toEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
          const route = buildDagEdgeRoute({
            startX,
            startY,
            endX,
            endY,
            branchLabel: edge.branchLabel,
          });
          return {
            ...edge,
            edgeKey,
            fromTaskName: fromEntry.node.taskName,
            toTaskName: toEntry.node.taskName,
            path: route.path,
            midX: route.midX,
            midY: route.midY,
            labelX: route.labelX,
            labelY: route.labelY,
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
            labelX: number;
            labelY: number;
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
    const startY =
      sourceEntry.position.y +
      (dagConnectorDrag.sourcePortY ??
        dagNodeOutputAnchorY(sourceEntry.node, dagConnectorDrag.branchLabel));
    const endX = dagConnectorDrag.x;
    const endY = dagConnectorDrag.y;
    const route = buildDagEdgeRoute({
      startX,
      startY,
      endX,
      endY,
      branchLabel: dagConnectorDrag.branchLabel,
    });
    return {
      path: route.path,
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
        summary: composerDraft.summary || "Workflow Builder draft",
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
        summary: composerDraft.summary || "Workflow Builder draft",
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
      summary: composerDraft.summary || "Workflow Builder draft",
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
      runtimeSettings: workflowRuntimeSettings,
    }),
    [
      composerDraft.summary,
      composerDraftEdges,
      composerNodePositions,
      contextJson,
      goal,
      visualChainNodes,
      workflowInterface,
      workflowRuntimeSettings,
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
    setWorkflowRuntimeSettings(initialWorkflowRuntimeSettings());
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

  useEffect(() => {
    if (requestedStudioMode !== "new") {
      return;
    }
    if (handledStudioRouteSelectionRef.current === "mode:new") {
      return;
    }
    handledStudioRouteSelectionRef.current = "mode:new";
    startFreshStudioDraft();
  }, [requestedStudioMode]);

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
    setWorkflowRuntimeSettings(restored.runtimeSettings);
    setComposerNodePositions(restored.nodePositions);
    setSavedWorkflowDefinition(definition);
    setPublishedWorkflowVersion(null);
    setLoadedWorkflowVersionId(null);
    resetStudioTransientState();
    setStudioNotice(`Opened saved draft ${definition.title}.`);
  };

  useEffect(() => {
    if (
      !demoDataDetected ||
      handledDemoWorkflowSelectionRef.current ||
      workflowDefinitionsLoading ||
      requestedWorkflowDefinitionId ||
      requestedWorkflowVersionId ||
      requestedStudioMode ||
      savedWorkflowDefinition
    ) {
      return;
    }
    const demoDefinition =
      workflowDefinitions.find((definition) => definition.metadata?.source === "demo_data") ||
      (workflowDefinitions.length === 1 ? workflowDefinitions[0] : null);
    if (!demoDefinition) {
      return;
    }
    handledDemoWorkflowSelectionRef.current = true;
    restoreWorkflowDefinition(demoDefinition);
  }, [
    demoDataDetected,
    requestedStudioMode,
    requestedWorkflowDefinitionId,
    requestedWorkflowVersionId,
    restoreWorkflowDefinition,
    savedWorkflowDefinition,
    workflowDefinitions,
    workflowDefinitionsLoading,
  ]);

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
    setWorkflowRuntimeSettings(restored.runtimeSettings);
    setComposerNodePositions(restored.nodePositions);
    setSavedWorkflowDefinition(definition);
    setPublishedWorkflowVersion(version);
    setLoadedWorkflowVersionId(version.id);
    resetStudioTransientState();
    setStudioNotice(`Restored workflow version v${version.version_number}.`);
  };

  useEffect(() => {
    const routeKey = `${requestedWorkflowDefinitionId}::${requestedWorkflowVersionId}`;
    if (!requestedWorkflowDefinitionId && !requestedWorkflowVersionId) {
      handledStudioRouteSelectionRef.current = "";
      return;
    }
    if (handledStudioRouteSelectionRef.current === routeKey) {
      return;
    }

    let cancelled = false;
    const loadRouteSelection = async () => {
      try {
        if (requestedWorkflowVersionId) {
          const definitionId = requestedWorkflowDefinitionId;
          if (!definitionId) {
            throw new Error("Workflow version links must include a definition id.");
          }

          let definition =
            workflowDefinitions.find((item) => item.id === definitionId) || null;
          if (!definition) {
            const definitionResponse = await fetch(
              `${apiUrl}/workflows/definitions/${encodeURIComponent(definitionId)}`
            );
            const definitionBody = (await definitionResponse.json()) as
              | WorkflowDefinition
              | { detail?: unknown };
            if (!definitionResponse.ok) {
              const detail = (definitionBody as { detail?: unknown }).detail;
              throw new Error(
                typeof detail === "string"
                  ? detail
                  : `Workflow definition request failed (${definitionResponse.status}).`
              );
            }
            definition = definitionBody as WorkflowDefinition;
          }

          const versionsResponse = await fetch(
            `${apiUrl}/workflows/definitions/${encodeURIComponent(definition.id)}/versions`
          );
          const versionsBody = (await versionsResponse.json()) as
            | WorkflowVersion[]
            | { detail?: unknown };
          if (!versionsResponse.ok) {
            const detail = (versionsBody as { detail?: unknown }).detail;
            throw new Error(
              typeof detail === "string"
                ? detail
                : `Workflow version history request failed (${versionsResponse.status}).`
            );
          }
          const matchingVersion = (Array.isArray(versionsBody) ? versionsBody : []).find(
            (item) => item.id === requestedWorkflowVersionId
          );
          if (!matchingVersion) {
            throw new Error("Requested workflow version could not be found.");
          }
          if (!cancelled) {
            handledStudioRouteSelectionRef.current = routeKey;
            await restoreWorkflowVersion(matchingVersion);
          }
          return;
        }

        const definitionId = requestedWorkflowDefinitionId;
        if (!definitionId) {
          return;
        }
        let definition = workflowDefinitions.find((item) => item.id === definitionId) || null;
        if (!definition) {
          const response = await fetch(
            `${apiUrl}/workflows/definitions/${encodeURIComponent(definitionId)}`
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
        }
        if (!cancelled) {
          handledStudioRouteSelectionRef.current = routeKey;
          restoreWorkflowDefinition(definition);
        }
      } catch (error) {
        if (!cancelled) {
          handledStudioRouteSelectionRef.current = routeKey;
          setStudioNotice(
            error instanceof Error ? error.message : "Failed to load workflow route selection."
          );
        }
      }
    };

    void loadRouteSelection();
    return () => {
      cancelled = true;
    };
  }, [
    requestedWorkflowDefinitionId,
    requestedWorkflowVersionId,
    restoreWorkflowDefinition,
    restoreWorkflowVersion,
    workflowDefinitions,
  ]);

  useEffect(() => {
    const routeKey = `${requestedWorkflowDefinitionId}::${requestedWorkflowVersionId}`;
    const definitionOrVersionPending =
      (requestedWorkflowDefinitionId || requestedWorkflowVersionId) &&
      handledStudioRouteSelectionRef.current !== routeKey;
    const newModePending =
      requestedStudioMode === "new" && handledStudioRouteSelectionRef.current !== "mode:new";
    if (definitionOrVersionPending || newModePending) {
      return;
    }

    const nextDefinitionId = activeWorkflowDefinitionId || "";
    const nextVersionId = activeWorkflowVersionId || "";
    if (
      nextDefinitionId === requestedWorkflowDefinitionId &&
      nextVersionId === requestedWorkflowVersionId &&
      !requestedStudioMode
    ) {
      return;
    }

    const params = new URLSearchParams();
    if (activeStudioSurface === "workbench") {
      params.set("surface", "workbench");
    }
    if (nextDefinitionId) {
      params.set("definition", nextDefinitionId);
    }
    if (nextVersionId) {
      params.set("version", nextVersionId);
    }
    const nextUrl = params.size > 0 ? `${pathname}?${params.toString()}` : pathname;
    router.replace(nextUrl, { scroll: false });
  }, [
    activeWorkflowDefinitionId,
    activeWorkflowVersionId,
    pathname,
    requestedStudioMode,
    activeStudioSurface,
    requestedWorkflowDefinitionId,
    requestedWorkflowVersionId,
    router,
  ]);

  const switchStudioSurface = (
    nextSurface: StudioSurface,
    options?: { clearWorkflowSelection?: boolean }
  ) => {
    const params = new URLSearchParams(searchParams.toString());
    if (nextSurface === "workbench") {
      params.set("surface", "workbench");
    } else {
      params.delete("surface");
    }
    if (options?.clearWorkflowSelection) {
      params.delete("definition");
      params.delete("version");
      params.delete("mode");
    }
    const nextUrl = params.size > 0 ? `${pathname}?${params.toString()}` : pathname;
    router.replace(nextUrl, { scroll: false });
  };

  useEffect(() => {
    if (!pendingWorkbenchWorkflowDraft) {
      return;
    }
    setGoal(pendingWorkbenchWorkflowDraft.goal);
    setContextJson(pendingWorkbenchWorkflowDraft.contextJsonText);
    setComposerDraft({
      summary: pendingWorkbenchWorkflowDraft.summary,
      nodes: pendingWorkbenchWorkflowDraft.nodes,
      edges: normalizeComposerEdges(
        pendingWorkbenchWorkflowDraft.nodes,
        pendingWorkbenchWorkflowDraft.edges
      ),
    });
    setWorkflowInterface(pendingWorkbenchWorkflowDraft.workflowInterface);
    setWorkflowRuntimeSettings(
      pendingWorkbenchWorkflowDraft.runtimeSettings || initialWorkflowRuntimeSettings()
    );
    setComposerNodePositions(pendingWorkbenchWorkflowDraft.nodePositions);
    setSavedWorkflowDefinition(null);
    setPublishedWorkflowVersion(null);
    setLoadedWorkflowVersionId(null);
    setWorkflowVersions([]);
    setWorkflowTriggers([]);
    setWorkflowRuns([]);
    resetStudioTransientState();
    setStudioNotice(pendingWorkbenchWorkflowDraft.notice || "Imported from workbench run.");
    switchStudioSurface("workflow", { clearWorkflowSelection: true });
    setPendingWorkbenchWorkflowDraft(null);
  }, [pendingWorkbenchWorkflowDraft]);

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
            title: composerDraft.summary || goal.trim() || "Workflow Builder draft",
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
      setPublishedWorkflowVersion(null);
      setLoadedWorkflowVersionId(null);
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

  const workflowSetupPanel = (
    <section className="px-3 py-3 text-slate-100">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-sky-100/68">
            Workflow Setup
          </div>
          <h2 className="mt-1 text-base font-semibold tracking-[-0.02em] text-white">
            Goal, context, and validation
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
              goal.trim()
                ? "border-emerald-300/25 bg-emerald-400/12 text-emerald-200"
                : "border-white/10 bg-white/[0.05] text-slate-200"
            }`}
          >
            {goal.trim() ? "goal set" : "goal empty"}
          </span>
          <span
            className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
              contextState.invalid
                ? "border-rose-300/25 bg-rose-400/12 text-rose-200"
                : "border-sky-300/25 bg-sky-400/12 text-sky-100"
            }`}
          >
            {contextState.invalid ? "json invalid" : "json ready"}
          </span>
          <button
            type="button"
            className="rounded-full border border-white/10 bg-slate-950/16 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-white/16 hover:bg-slate-950/24"
            onClick={() => setWorkflowSetupExpanded((prev) => !prev)}
          >
            {workflowSetupExpanded ? "Hide Setup" : "Expand Setup"}
          </button>
        </div>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-5">
        <div className="rounded-2xl border border-white/8 bg-slate-950/14 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/58">
            Goal
          </div>
          <div className="mt-1 line-clamp-2 text-sm text-slate-100">
            {goal.trim() || "Set the workflow objective."}
          </div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-slate-950/14 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/58">
            Draft
          </div>
          <div className="mt-1 line-clamp-2 text-sm text-slate-100">
            {composerDraft.summary.trim() || "Workflow Builder draft"}
          </div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-slate-950/14 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/58">
            Context User
          </div>
          <div className="mt-1 truncate text-sm text-slate-100">
            {workspaceUserId.trim() || "Not set"}
          </div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-slate-950/14 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/58">
            Context Paths
          </div>
          <div className="mt-1 text-sm text-slate-100">
            {contextState.invalid ? "Unavailable" : `${contextPathSuggestions.length} detected`}
          </div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-slate-950/14 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/58">
            Execution Mode
          </div>
          <div className="mt-1 text-sm text-slate-100">
            {workflowRuntimeSettings.executionMode === "adaptive" ? "Adaptive" : "Static"}
          </div>
          <div className="mt-1 text-[11px] text-slate-300/60">
            Max replans {workflowRuntimeSettings.adaptivePolicy?.maxReplans ?? 2}
          </div>
        </div>
      </div>

      {workflowSetupExpanded ? (
        <div className="mt-3 space-y-3 border-t border-white/8 pt-3">
          <div className="grid gap-3 lg:grid-cols-4">
            <label className="block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                Goal
              </div>
              <input
                className="mt-1 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition placeholder:text-slate-300/42 focus:border-sky-300/40 focus:bg-slate-950/28"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                placeholder="Generate a document pipeline with validation and render output"
              />
            </label>
            <label className="block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                Draft Summary
              </div>
              <input
                className="mt-1 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition placeholder:text-slate-300/42 focus:border-sky-300/40 focus:bg-slate-950/28"
                value={composerDraft.summary}
                onChange={(event) =>
                  setComposerDraft((prev) => ({ ...prev, summary: event.target.value }))
                }
                placeholder="Workflow Builder draft"
              />
            </label>
            <label className="block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                Context User ID
              </div>
              <input
                className="mt-1 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition placeholder:text-slate-300/42 focus:border-sky-300/40 focus:bg-slate-950/28"
                value={workspaceUserId}
                onChange={(event) => setWorkspaceUserId(event.target.value)}
                placeholder="narendersurabhi"
              />
              <div className="mt-2 text-xs leading-5 text-slate-200/62">
                User-scoped memory bindings inherit this id automatically unless a node overrides it
                explicitly.
              </div>
            </label>
            <label className="block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                Execution Mode
              </div>
              <select
                className="mt-1 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/40 focus:bg-slate-950/28"
                value={workflowRuntimeSettings.executionMode || "static"}
                onChange={(event) =>
                  setWorkflowRuntimeSettings((prev) => ({
                    executionMode: event.target.value === "adaptive" ? "adaptive" : "static",
                    adaptivePolicy: {
                      maxReplans: prev.adaptivePolicy?.maxReplans ?? 2,
                    },
                  }))
                }
              >
                <option value="static">Static</option>
                <option value="adaptive">Adaptive</option>
              </select>
              <div className="mt-2 text-xs leading-5 text-slate-200/62">
                Adaptive mode only affects published workflow runs. Draft compile and preflight stay deterministic.
              </div>
            </label>
          </div>

          <label className="block max-w-xs">
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
              Max Adaptive Replans
            </div>
            <input
              type="number"
              min={0}
              max={10}
              className="mt-1 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/40 focus:bg-slate-950/28"
              value={workflowRuntimeSettings.adaptivePolicy?.maxReplans ?? 2}
              onChange={(event) =>
                setWorkflowRuntimeSettings((prev) => ({
                  executionMode: prev.executionMode || "static",
                  adaptivePolicy: {
                    maxReplans: Math.max(0, Math.min(10, Number(event.target.value) || 0)),
                  },
                }))
              }
            />
            <div className="mt-2 text-xs leading-5 text-slate-200/62">
              Used only when execution mode is adaptive.
            </div>
          </label>

          <label className="block">
            <div className="flex items-center justify-between gap-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                Context JSON
              </div>
              <div className="text-xs text-slate-300/55">
                {contextState.invalid ? "Invalid JSON" : "Object ready"}
              </div>
            </div>
            <textarea
              className="mt-1 min-h-[180px] w-full rounded-[18px] border border-white/8 bg-[#233142] px-3 py-3 font-mono text-xs text-slate-100 outline-none transition placeholder:text-slate-400/40 focus:border-sky-300/40 focus:bg-[#1c2939]"
              value={contextJson}
              onChange={(event) => setContextJson(event.target.value)}
            />
          </label>
        </div>
      ) : null}

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
  );

  const nodeInspectorPanel = (
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
      compactMode={studioWorkspaceMode === "focus_graph"}
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
  );

  const workflowLibraryLauncherPanel = (
    <section className="flex h-full flex-col px-3 py-3 text-slate-100">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-sky-100/68">
            Saved Workflows
          </div>
          <h3 className="mt-1 text-2xl text-white">Workflow Launcher</h3>
        </div>
        <button
          className="rounded-full border border-white/12 bg-white/[0.05] px-3 py-1.5 text-xs font-semibold text-slate-100 transition hover:border-sky-300/40 hover:bg-white/[0.08]"
          onClick={() => {
            void refreshWorkflowDefinitions();
            if (activeWorkflowDefinitionId) {
              void refreshWorkflowVersions(activeWorkflowDefinitionId);
              void refreshWorkflowTriggers(activeWorkflowDefinitionId);
              void refreshWorkflowRuns(activeWorkflowDefinitionId);
            }
          }}
        >
          Refresh
        </button>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-300/82">
        Keep Studio focused on editing. Use the full Workflows page for version history, triggers,
        run history, and draft management.
      </p>

      <div className="mt-4 flex flex-wrap gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]">
        <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-slate-100">
          drafts {workflowDefinitions.length}
        </span>
        <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-slate-100">
          versions {workflowVersions.length}
        </span>
        <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-slate-100">
          runs {workflowRuns.length}
        </span>
        <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-slate-100">
          {activeWorkflowVersionId ? "version linked" : "draft only"}
        </span>
      </div>

      <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-3">
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300/60">
          Current Workflow
        </div>
        {savedWorkflowDefinition ? (
          <>
            <div className="mt-2 truncate text-sm font-semibold text-white">
              {savedWorkflowDefinition.title}
            </div>
            <div className="mt-1 text-xs leading-5 text-slate-300/76">
              {savedWorkflowDefinition.goal || "No goal recorded for this workflow."}
            </div>
            <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
              <span className="rounded-full bg-white/[0.06] px-2.5 py-1">
                updated {formatTimestamp(savedWorkflowDefinition.updated_at)}
              </span>
              <span className="rounded-full bg-white/[0.06] px-2.5 py-1">
                {activeWorkflowVersionId ? `version ${activeWorkflowVersionId.slice(0, 8)}` : "draft"}
              </span>
            </div>
          </>
        ) : (
          <div className="mt-2 text-sm leading-6 text-slate-300/72">
            Save a draft or open one from Workflows to make this Studio session shareable.
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between gap-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300/75">
          Recent Drafts
        </div>
        <Link
          href="/workflows"
          className="rounded-full border border-white/12 bg-white/[0.05] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
        >
          Open Workflows
        </Link>
      </div>

      <div className="mt-3 flex-1 space-y-2 overflow-auto pr-1">
        {workflowDefinitionsLoading ? (
          <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-3 text-sm text-slate-300/72">
            Loading saved workflows...
          </div>
        ) : workflowDefinitionsError ? (
          <div className="rounded-2xl border border-rose-300/24 bg-rose-400/10 px-3 py-3 text-sm text-rose-100">
            {workflowDefinitionsError}
          </div>
        ) : workflowDefinitions.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/12 bg-white/[0.03] px-3 py-4 text-sm text-slate-300/72">
            No saved workflows yet. Save this draft, then use Workflows for deeper history and
            management.
          </div>
        ) : (
          workflowDefinitions.slice(0, 4).map((definition) => {
            const isActive = definition.id === activeWorkflowDefinitionId;
            return (
              <article
                key={definition.id}
                className={`rounded-2xl border px-3 py-3 ${
                  isActive
                    ? "border-sky-300/24 bg-sky-400/10"
                    : "border-white/10 bg-white/[0.04]"
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-white">
                      {definition.title}
                    </div>
                    <div className="mt-1 text-xs text-slate-300/70">
                      updated {formatTimestamp(definition.updated_at)}
                    </div>
                  </div>
                  {isActive ? (
                    <span className="rounded-full border border-sky-300/24 bg-sky-400/12 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-sky-100">
                      Active
                    </span>
                  ) : null}
                </div>
                <div className="mt-3 flex items-center justify-between gap-3">
                  <div className="line-clamp-2 text-xs leading-5 text-slate-300/76">
                    {definition.goal || "No goal recorded for this workflow."}
                  </div>
                  <button
                    className="rounded-full border border-white/12 bg-white/[0.05] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                    onClick={() => restoreWorkflowDefinition(definition)}
                  >
                    Open
                  </button>
                </div>
              </article>
            );
          })
        )}
      </div>
    </section>
  );

  const getNextFloatingStudioPanelZIndex = (
    panels: Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
  ) => Math.max(...FLOATING_STUDIO_PANEL_IDS.map((panelId) => panels[panelId].zIndex)) + 1;

  const syncStudioPanelsToStage = (
    panels: Record<FloatingStudioPanelId, FloatingStudioPanelLayout>,
    stageWidth: number,
    stageHeight: number
  ) => {
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    return FLOATING_STUDIO_PANEL_IDS.reduce<Record<FloatingStudioPanelId, FloatingStudioPanelLayout>>(
      (acc, panelId) => {
        const measured = floatingStudioPanelRefs.current[panelId];
        const defaultLayout =
          panels[panelId].mode === "docked" && panels[panelId].dockZone !== defaults[panelId].dockZone
            ? { ...defaults[panelId], dockZone: panels[panelId].dockZone }
            : defaults[panelId];
        acc[panelId] = syncFloatingStudioPanelLayoutToStage(
          panels[panelId],
          defaultLayout,
          stageWidth,
          stageHeight,
          measured?.offsetWidth,
          measured?.offsetHeight
        );
        return acc;
      },
      {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
    );
  };

  const buildFocusGraphPanelLayouts = (
    panels: Record<FloatingStudioPanelId, FloatingStudioPanelLayout>,
    stageWidth: number,
    stageHeight: number,
    preserveInspector: boolean
  ) => {
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    return FLOATING_STUDIO_PANEL_IDS.reduce<Record<FloatingStudioPanelId, FloatingStudioPanelLayout>>(
      (acc, panelId) => {
        const current = panels[panelId];
        if (panelId === "inspector" && preserveInspector) {
          acc[panelId] = syncFloatingStudioPanelLayoutToStage(
            {
              ...current,
              mode: "docked",
              dockZone: "right",
              minimized: false,
            },
            defaults.inspector,
            stageWidth,
            stageHeight
          );
          return acc;
        }
        acc[panelId] =
          current.mode === "floating"
            ? clampFloatingStudioPanelLayout(
                {
                  ...current,
                  minimized: true,
                },
                stageWidth,
                stageHeight
              )
            : syncFloatingStudioPanelLayoutToStage(
                {
                  ...current,
                  minimized: true,
                },
                current.dockZone === defaults[panelId].dockZone
                  ? defaults[panelId]
                  : { ...defaults[panelId], dockZone: current.dockZone },
                stageWidth,
                stageHeight
              );
        return acc;
      },
      {} as Record<FloatingStudioPanelId, FloatingStudioPanelLayout>
    );
  };

  useEffect(() => {
    if (!selectedDagNodeId) {
      return;
    }
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    setFloatingStudioPanels((prev) => {
      const inspector = prev.inspector;
      const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
      if (studioWorkspaceMode === "focus_graph") {
        return {
          ...prev,
          inspector: syncFloatingStudioPanelLayoutToStage(
            {
              ...inspector,
              mode: "docked",
              dockZone: "right",
              minimized: false,
            },
            defaults.inspector,
            stageWidth,
            stageHeight
          ),
        };
      }
      if (
        inspector.mode !== "docked" ||
        inspector.dockZone !== "right" ||
        inspector.minimized
      ) {
        return {
          ...prev,
          inspector: syncFloatingStudioPanelLayoutToStage(
            {
              ...inspector,
              mode: "docked",
              dockZone: "right",
              minimized: false,
            },
            defaults.inspector,
            stageWidth,
            stageHeight
          ),
        };
      }
      return prev;
    });
  }, [
    selectedDagNodeId,
    studioWorkspaceMode,
    studioWorkspaceStageSize.height,
    studioWorkspaceStageSize.width,
  ]);

  const bringFloatingStudioPanelToFront = (panelId: FloatingStudioPanelId) => {
    setFloatingStudioPanels((prev) => {
      if (prev[panelId].mode !== "floating") {
        return prev;
      }
      const nextZIndex = getNextFloatingStudioPanelZIndex(prev);
      return {
        ...prev,
        [panelId]: {
          ...prev[panelId],
          zIndex: nextZIndex,
        },
      };
    });
  };

  const setFloatingStudioPanelDocked = (
    panelId: FloatingStudioPanelId,
    dockZone: Exclude<StudioDockZone, "overlay" | "none">
  ) => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    setFloatingStudioPanels((prev) => ({
      ...prev,
      [panelId]: syncFloatingStudioPanelLayoutToStage(
        {
          ...prev[panelId],
          mode: "docked",
          dockZone,
          minimized: false,
        },
        { ...defaults[panelId], dockZone },
        stageWidth,
        stageHeight
      ),
    }));
    if (dockZone === "bottom") {
      setStudioBottomTray((prev) => ({
        ...prev,
        activePanelId: panelId,
        collapsed: false,
      }));
    }
  };

  const setFloatingStudioPanelFloating = (panelId: FloatingStudioPanelId) => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    setFloatingStudioPanels((prev) => {
      const current = prev[panelId];
      const nextZIndex = getNextFloatingStudioPanelZIndex(prev);
      return {
        ...prev,
        [panelId]: clampFloatingStudioPanelLayout(
          {
            ...defaults[panelId],
            ...current,
            minimized: false,
            mode: "floating",
            dockZone: current.dockZone === "none" ? defaults[panelId].dockZone : current.dockZone,
            zIndex: nextZIndex,
          },
          stageWidth,
          stageHeight
        ),
      };
    });
  };

  const revealStudioPanel = (
    panelId: FloatingStudioPanelId,
    dockZone?: Exclude<StudioDockZone, "overlay" | "none">
  ) => {
    if (dockZone) {
      setFloatingStudioPanelDocked(panelId, dockZone);
    } else {
      setFloatingStudioPanelMinimized(panelId, false);
    }
    if (dockZone === "bottom" || floatingStudioPanels[panelId].dockZone === "bottom") {
      setStudioBottomTray((prev) => ({
        ...prev,
        activePanelId: panelId,
        collapsed: false,
      }));
    }
  };

  const setFloatingStudioPanelMinimized = (
    panelId: FloatingStudioPanelId,
    minimized: boolean
  ) => {
    setFloatingStudioPanels((prev) => {
      const current = prev[panelId];
      if (current.minimized === minimized) {
        return prev;
      }
      if (
        !studioWorkspaceStageSize.width ||
        !studioWorkspaceStageSize.height ||
        current.mode !== "floating"
      ) {
        return {
          ...prev,
          [panelId]: {
            ...current,
            minimized,
          },
        };
      }
      return {
        ...prev,
        [panelId]: clampFloatingStudioPanelLayout(
          {
            ...current,
            minimized,
          },
          studioWorkspaceStageSize.width,
          studioWorkspaceStageSize.height
        ),
      };
    });
  };

  const toggleFloatingStudioPanelMinimized = (panelId: FloatingStudioPanelId) => {
    setFloatingStudioPanels((prev) => {
      const current = prev[panelId];
      const nextMinimized = !current.minimized;
      if (
        !studioWorkspaceStageSize.width ||
        !studioWorkspaceStageSize.height ||
        current.mode !== "floating"
      ) {
        return {
          ...prev,
          [panelId]: {
            ...current,
            minimized: nextMinimized,
          },
        };
      }
      return {
        ...prev,
        [panelId]: clampFloatingStudioPanelLayout(
          {
            ...current,
            minimized: nextMinimized,
          },
          studioWorkspaceStageSize.width,
          studioWorkspaceStageSize.height
        ),
      };
    });
  };

  const toggleFloatingStudioPanelMode = (panelId: FloatingStudioPanelId) => {
    const current = floatingStudioPanels[panelId];
    if (current.mode === "floating") {
      const nextDockZone: Exclude<StudioDockZone, "overlay" | "none"> =
        current.dockZone === "none" || current.dockZone === "overlay"
          ? (STUDIO_DOCK_ZONE_DEFAULTS[panelId].dockZone as Exclude<
              StudioDockZone,
              "overlay" | "none"
            >)
          : current.dockZone;
      setFloatingStudioPanelDocked(
        panelId,
        nextDockZone
      );
      return;
    }
    setFloatingStudioPanelFloating(panelId);
  };

  const restoreFloatingStudioPanel = (panelId: FloatingStudioPanelId) => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    setFloatingStudioPanels((prev) => {
      const current = prev[panelId];
      const nextZIndex =
        current.mode === "floating" ? getNextFloatingStudioPanelZIndex(prev) : current.zIndex;
      return {
        ...prev,
        [panelId]:
          current.mode === "floating"
            ? clampFloatingStudioPanelLayout(
                {
                  ...defaults[panelId],
                  mode: "floating",
                  dockZone:
                    current.dockZone === "none" ? defaults[panelId].dockZone : current.dockZone,
                  minimized: false,
                  zIndex: nextZIndex,
                },
                stageWidth,
                stageHeight
              )
            : syncFloatingStudioPanelLayoutToStage(
                {
                  ...defaults[panelId],
                  minimized: false,
                  zIndex: nextZIndex,
                },
                defaults[panelId],
                stageWidth,
                stageHeight
              ),
      };
    });
  };

  const resetFloatingStudioWorkspaceLayout = () => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    const defaults = createInitialFloatingStudioPanelLayouts(stageWidth, stageHeight);
    focusGraphPanelSnapshotRef.current = null;
    setStudioWorkspaceMode("default");
    setActiveStudioPanelMenuId(null);
    setStudioBottomTray(initialStudioBottomTrayState());
    setFloatingStudioPanels(defaults);
    setStudioNotice("Workspace layout reset.");
  };

  const enterFocusGraphMode = () => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    setActiveStudioPanelMenuId(null);
    setFloatingStudioPanels((prev) => {
      focusGraphPanelSnapshotRef.current = prev;
      return buildFocusGraphPanelLayouts(
        prev,
        stageWidth,
        stageHeight,
        Boolean(selectedDagNodeId)
      );
    });
    setStudioWorkspaceMode("focus_graph");
    setStudioNotice("Focus Graph mode enabled. Press F to restore the workspace.");
  };

  const exitFocusGraphMode = () => {
    const stageWidth = studioWorkspaceStageSize.width || 1440;
    const stageHeight = studioWorkspaceStageSize.height || 1040;
    const snapshot = focusGraphPanelSnapshotRef.current;
    setActiveStudioPanelMenuId(null);
    setStudioWorkspaceMode("default");
    if (!snapshot) {
      setStudioNotice("Focus Graph mode cleared.");
      return;
    }
    setFloatingStudioPanels(syncStudioPanelsToStage(snapshot, stageWidth, stageHeight));
    focusGraphPanelSnapshotRef.current = null;
    setStudioNotice("Workspace layout restored.");
  };

  const toggleFocusGraphMode = () => {
    if (studioWorkspaceMode === "focus_graph") {
      exitFocusGraphMode();
      return;
    }
    enterFocusGraphMode();
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      if (isTypingTarget(event.target)) {
        return;
      }
      const normalizedKey = event.key.toLowerCase();
      if (normalizedKey === "f") {
        event.preventDefault();
        toggleFocusGraphMode();
        return;
      }
      if (normalizedKey === "escape") {
        event.preventDefault();
        setActiveStudioPanelMenuId(null);
        setDagEdgeDraftSourceNodeId(null);
        setDagConnectorDrag(null);
        setDagCanvasDraggingNodeId(null);
        setDagConnectorHoverTargetNodeId(null);
        setSelectedDagNodeId(null);
        setStudioNotice("Transient graph interactions cleared.");
        return;
      }
      if (normalizedKey === "0") {
        event.preventDefault();
        resetDagCanvasZoom();
        return;
      }
      if (!event.shiftKey) {
        return;
      }
      if (event.code === "Digit1") {
        event.preventDefault();
        revealStudioPanel("palette", "left");
        return;
      }
      if (event.code === "Digit2") {
        if (!selectedDagNode) {
          return;
        }
        event.preventDefault();
        revealStudioPanel("inspector", "right");
        return;
      }
      if (event.code === "Digit3") {
        event.preventDefault();
        revealStudioPanel("interface", "bottom");
        setStudioBottomTray((prev) => ({
          ...prev,
          activePanelId: "interface",
          collapsed: false,
        }));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [
    floatingStudioPanels,
    selectedDagNode,
    selectedDagNodeId,
    studioWorkspaceMode,
    studioWorkspaceStageSize.height,
    studioWorkspaceStageSize.width,
  ]);

  const beginFloatingStudioPanelDrag = (
    panelId: FloatingStudioPanelId,
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    const panel = floatingStudioPanelRefs.current[panelId];
    if (!panel || floatingStudioPanels[panelId].mode !== "floating") {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const panelRect = panel.getBoundingClientRect();
    bringFloatingStudioPanelToFront(panelId);
    setFloatingStudioPanelDrag({
      id: panelId,
      offsetX: event.clientX - panelRect.left,
      offsetY: event.clientY - panelRect.top,
    });
  };

  const beginFloatingStudioPanelResize = (
    panelId: FloatingStudioPanelId,
    direction: FloatingStudioPanelResizeDirection,
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    const panel = floatingStudioPanelRefs.current[panelId];
    const layout = floatingStudioPanels[panelId];
    const canResizeDockedPanel =
      layout.mode === "docked" &&
      (layout.dockZone === "left" || layout.dockZone === "right") &&
      (direction === "e" || direction === "w");
    if ((layout.mode === "floating" && !panel) || (layout.mode !== "floating" && !canResizeDockedPanel)) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (layout.mode === "floating") {
      bringFloatingStudioPanelToFront(panelId);
    }
    setFloatingStudioPanelResize({
      id: panelId,
      direction,
      mode: layout.mode,
      startX: event.clientX,
      startY: event.clientY,
      startLeft: layout.x,
      startTop: layout.y,
      startWidth: layout.width,
      startHeight: layout.height,
    });
  };

  const renderDockedStudioPanel = (
    panelId: FloatingStudioPanelId,
    title: string,
    content: ReactNode,
    options: { bodyClassName?: string; badge?: string; panelDomId?: string } = {}
  ) => {
    const layout = floatingStudioPanels[panelId];
    const isMinimized = layout.minimized;
    return (
      <div
        id={options.panelDomId}
        key={`docked-studio-panel-${panelId}`}
        className="flex h-full min-h-0 flex-col overflow-hidden rounded-[24px] border border-white/12 bg-[linear-gradient(180deg,rgba(37,49,61,0.86),rgba(16,24,34,0.9))] shadow-[0_18px_36px_rgba(15,23,42,0.24)] backdrop-blur-xl"
      >
        <div className="flex items-center justify-between gap-3 border-b border-white/8 bg-[rgba(9,16,27,0.46)] px-3 py-2">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                aria-label={isMinimized ? `Open ${title}` : `Minimize ${title}`}
                title={isMinimized ? "Open panel" : "Minimize panel"}
                className="h-3 w-3 rounded-full bg-[#f6cf58] shadow-[inset_0_1px_0_rgba(255,255,255,0.34),0_0_0_1px_rgba(125,77,0,0.2)] transition hover:brightness-105"
                onClick={() => toggleFloatingStudioPanelMinimized(panelId)}
              />
              <button
                type="button"
                aria-label={`Restore ${title}`}
                title="Reset panel"
                className="h-3 w-3 rounded-full bg-[#50d16e] shadow-[inset_0_1px_0_rgba(255,255,255,0.28),0_0_0_1px_rgba(6,95,70,0.2)] transition hover:brightness-105"
                onClick={() => restoreFloatingStudioPanel(panelId)}
              />
            </div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100">
              {title}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-white/16 hover:bg-white/[0.1]"
              onClick={() => toggleFloatingStudioPanelMode(panelId)}
            >
              Float
            </button>
            {renderStudioPanelActionMenu(panelId)}
            {options.badge ? (
              <div className="rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-200">
                {options.badge}
              </div>
            ) : null}
          </div>
        </div>
        {!isMinimized ? (
          <div className={`min-h-0 flex-1 overflow-auto ${options.bodyClassName || ""}`.trim()}>
            {content}
          </div>
        ) : null}
      </div>
    );
  };

  const renderFloatingStudioPanel = (
    panelId: FloatingStudioPanelId,
    title: string,
    content: ReactNode,
    options: { bodyClassName?: string; badge?: string; panelDomId?: string } = {}
  ) => {
    const layout = floatingStudioPanels[panelId];
    const isMinimized = layout.minimized;
    return (
      <div
        id={options.panelDomId}
        ref={(node) => {
          floatingStudioPanelRefs.current[panelId] = node;
        }}
        key={`floating-studio-panel-${panelId}`}
        className="pointer-events-auto absolute flex flex-col overflow-hidden rounded-[24px] border border-white/12 bg-[linear-gradient(180deg,rgba(40,53,67,0.62),rgba(19,28,39,0.72))] shadow-[0_28px_64px_rgba(15,23,42,0.34)] backdrop-blur-xl"
        style={{
          left: layout.x,
          top: layout.y,
          width: layout.width,
          height: isMinimized ? FLOATING_STUDIO_PANEL_HEADER_HEIGHT : layout.height,
          zIndex: layout.zIndex,
        }}
        onMouseDown={() => bringFloatingStudioPanelToFront(panelId)}
      >
        <div
          className="flex cursor-grab items-center justify-between gap-3 border-b border-white/8 bg-[rgba(9,16,27,0.38)] px-3 py-2 active:cursor-grabbing"
          onMouseDown={(event) => beginFloatingStudioPanelDrag(panelId, event)}
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                aria-label={isMinimized ? `Open ${title}` : `Minimize ${title}`}
                title={isMinimized ? "Open panel" : "Minimize panel"}
                className="h-3 w-3 rounded-full bg-[#f6cf58] shadow-[inset_0_1px_0_rgba(255,255,255,0.34),0_0_0_1px_rgba(125,77,0,0.2)] transition hover:brightness-105"
                onMouseDown={(event) => {
                  event.stopPropagation();
                }}
                onClick={() => toggleFloatingStudioPanelMinimized(panelId)}
              />
              <button
                type="button"
                aria-label={`Restore ${title}`}
                title="Restore panel"
                className="h-3 w-3 rounded-full bg-[#50d16e] shadow-[inset_0_1px_0_rgba(255,255,255,0.28),0_0_0_1px_rgba(6,95,70,0.2)] transition hover:brightness-105"
                onMouseDown={(event) => {
                  event.stopPropagation();
                }}
                onClick={() => restoreFloatingStudioPanel(panelId)}
              />
            </div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100">
              {title}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-white/16 hover:bg-white/[0.1]"
              onMouseDown={(event) => {
                event.stopPropagation();
              }}
              onClick={() => toggleFloatingStudioPanelMode(panelId)}
            >
              Dock
            </button>
            {renderStudioPanelActionMenu(panelId, { stopMouseDownPropagation: true })}
            {options.badge ? (
              <div className="rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-200">
                {options.badge}
              </div>
            ) : null}
          </div>
        </div>
        {!isMinimized ? (
          <div className={`min-h-0 flex-1 overflow-auto ${options.bodyClassName || ""}`.trim()}>
            {content}
          </div>
        ) : null}
        {!isMinimized ? (
          <>
            {FLOATING_STUDIO_PANEL_RESIZE_HANDLES.map((handle) => (
              <div
                key={`${panelId}-${handle.direction}`}
                className={`${handle.className} transition hover:bg-white/[0.06]`.trim()}
                onMouseDown={(event) =>
                  beginFloatingStudioPanelResize(panelId, handle.direction, event)
                }
                title="Resize panel"
              />
            ))}
          </>
        ) : null}
      </div>
    );
  };

  const workspaceStageWidth = studioWorkspaceStageSize.width || 1440;
  const workspaceStageHeight = studioWorkspaceStageSize.height || 1040;
  const dockedRightPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) =>
      floatingStudioPanels[panelId].mode === "docked" &&
      floatingStudioPanels[panelId].dockZone === "right" &&
      !floatingStudioPanels[panelId].minimized &&
      (panelId !== "inspector" || Boolean(selectedDagNode))
  );
  const dockedLeftPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) =>
      floatingStudioPanels[panelId].mode === "docked" &&
      floatingStudioPanels[panelId].dockZone === "left" &&
      !floatingStudioPanels[panelId].minimized &&
      (panelId !== "inspector" || Boolean(selectedDagNode))
  );
  const dockedBottomPanelIds = STUDIO_PANEL_DISPLAY_ORDER.filter(
    (panelId) =>
      floatingStudioPanels[panelId].mode === "docked" &&
      floatingStudioPanels[panelId].dockZone === "bottom" &&
      !floatingStudioPanels[panelId].minimized &&
      (panelId !== "inspector" || Boolean(selectedDagNode))
  );
  const minimizedWorkspacePanelIds = FLOATING_STUDIO_PANEL_IDS.filter(
    (panelId) =>
      floatingStudioPanels[panelId].minimized &&
      (panelId !== "inspector" || Boolean(selectedDagNode))
  );
  const floatingWorkspacePanelIds = FLOATING_STUDIO_PANEL_IDS.filter(
    (panelId) =>
      floatingStudioPanels[panelId].mode === "floating" &&
      !floatingStudioPanels[panelId].minimized &&
      (panelId !== "inspector" || Boolean(selectedDagNode))
  );

  useEffect(() => {
    if (dockedBottomPanelIds.length === 0) {
      return;
    }
    if (!dockedBottomPanelIds.includes(studioBottomTray.activePanelId)) {
      setStudioBottomTray((prev) => ({
        ...prev,
        activePanelId: dockedBottomPanelIds[0],
      }));
    }
  }, [dockedBottomPanelIds, studioBottomTray.activePanelId]);

  const workspacePanelRects = resolveWorkspacePanelRects(
    floatingStudioPanels,
    workspaceStageWidth,
    workspaceStageHeight,
    {
      showInspector: Boolean(selectedDagNode),
      showMinimizedShelf: FLOATING_STUDIO_PANEL_IDS.some(
        (panelId) => floatingStudioPanels[panelId].minimized && (panelId !== "inspector" || Boolean(selectedDagNode))
      ),
      showBottomTray: dockedBottomPanelIds.length > 0,
      bottomTrayHeight: studioBottomTray.collapsed
        ? STUDIO_BOTTOM_TRAY_COLLAPSED_HEIGHT
        : studioBottomTray.height,
    }
  );
  const leftDockRect = workspacePanelRects.leftDock;
  const rightDockRect = workspacePanelRects.rightDock;
  const bottomDockRect = workspacePanelRects.bottomDock;
  const minimizedShelfRect = workspacePanelRects.minimizedShelf;
  const getWorkspacePanelTitle = (panelId: FloatingStudioPanelId) => STUDIO_PANEL_TITLES[panelId];
  const getWorkspacePanelBadge = (panelId: FloatingStudioPanelId) => {
    switch (panelId) {
      case "compile":
        return composerCompileLoading || chainPreflightLoading ? "busy" : "ready";
      case "interface":
        return `${workflowInterface.inputs.length}/${workflowInterface.outputs.length}`;
      case "library":
        return savedWorkflowDefinition ? "linked" : "browse";
      default:
        return null;
    }
  };
  const renderStudioPanelActionMenu = (
    panelId: FloatingStudioPanelId,
    options: { stopMouseDownPropagation?: boolean } = {}
  ) => {
    const isOpen = activeStudioPanelMenuId === panelId;
    const current = floatingStudioPanels[panelId];
    return (
      <div
        data-studio-panel-menu-root={panelId}
        className="relative"
        onMouseDown={(event) => {
          if (options.stopMouseDownPropagation) {
            event.stopPropagation();
          }
        }}
      >
        <button
          type="button"
          className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] transition ${
            isOpen
              ? "border-sky-300/35 bg-sky-400/18 text-sky-50"
              : "border-white/10 bg-white/[0.06] text-slate-100 hover:border-white/16 hover:bg-white/[0.1]"
          }`}
          aria-label={`${getWorkspacePanelTitle(panelId)} menu`}
          aria-expanded={isOpen}
          onClick={() =>
            setActiveStudioPanelMenuId((prev) => (prev === panelId ? null : panelId))
          }
        >
          Panel
        </button>
        {isOpen ? (
          <div className="absolute right-0 top-full z-40 mt-2 w-44 overflow-hidden rounded-2xl border border-white/12 bg-[rgba(12,19,31,0.96)] p-1 shadow-[0_24px_48px_rgba(2,6,23,0.4)] backdrop-blur-xl">
            {STUDIO_PANEL_ALLOWED_DOCK_ZONES[panelId].map((dockZone) => {
              const isActiveDock =
                current.mode === "docked" && current.dockZone === dockZone && !current.minimized;
              return (
                <button
                  key={`${panelId}-${dockZone}`}
                  type="button"
                  disabled={isActiveDock}
                  className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-default disabled:opacity-45"
                  onClick={() => {
                    setFloatingStudioPanelDocked(panelId, dockZone);
                    setActiveStudioPanelMenuId(null);
                  }}
                >
                  <span>{STUDIO_DOCK_ZONE_LABELS[dockZone]}</span>
                  {isActiveDock ? <span className="text-sky-100/72">Active</span> : null}
                </button>
              );
            })}
            <button
              type="button"
              disabled={current.mode === "floating" && !current.minimized}
              className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-default disabled:opacity-45"
              onClick={() => {
                setFloatingStudioPanelFloating(panelId);
                setActiveStudioPanelMenuId(null);
              }}
            >
              <span>Float</span>
            </button>
            <button
              type="button"
              disabled={current.minimized}
              className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-default disabled:opacity-45"
              onClick={() => {
                setFloatingStudioPanelMinimized(panelId, true);
                setActiveStudioPanelMenuId(null);
              }}
            >
              <span>Minimize</span>
            </button>
            <button
              type="button"
              className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:bg-white/[0.08]"
              onClick={() => {
                restoreFloatingStudioPanel(panelId);
                setActiveStudioPanelMenuId(null);
              }}
            >
              <span>Restore Defaults</span>
            </button>
          </div>
        ) : null}
      </div>
    );
  };
  const getWorkspacePanelDefinition = (panelId: FloatingStudioPanelId) => {
    if (panelId === "inspector" && !selectedDagNode) {
      return null;
    }

    switch (panelId) {
      case "palette":
        return {
          title: getWorkspacePanelTitle("palette"),
          content: (
            <div className="h-full">
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
            </div>
          ),
          panelDomId: "studio-palette-section",
          bodyClassName: "overflow-hidden p-0",
        } as const;
      case "compile":
        return {
          title: getWorkspacePanelTitle("compile"),
          content: (
            <StudioCompilePanel
              compileLoading={composerCompileLoading || chainPreflightLoading}
              compileResult={composerCompileResult}
              preflightResult={chainPreflightResult}
              issues={composerIssues}
              draftPayloadPreview={draftPayloadPreview}
              onCompile={runChainPreflight}
            />
          ),
          badge: getWorkspacePanelBadge("compile") || undefined,
        } as const;
      case "setup":
        return {
          title: getWorkspacePanelTitle("setup"),
          content: workflowSetupPanel,
        } as const;
      case "interface":
        return {
          title: getWorkspacePanelTitle("interface"),
          content: (
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
          ),
          badge: getWorkspacePanelBadge("interface") || undefined,
        } as const;
      case "library":
        return {
          title: getWorkspacePanelTitle("library"),
          content: workflowLibraryLauncherPanel,
          panelDomId: "studio-library-section",
          bodyClassName: "overflow-hidden p-0",
          badge: getWorkspacePanelBadge("library") || undefined,
        } as const;
      case "inspector":
        return {
          title: getWorkspacePanelTitle("inspector"),
          content: nodeInspectorPanel,
          panelDomId: "studio-inspector-section",
          bodyClassName: "transition-opacity duration-200 motion-reduce:transition-none",
        } as const;
      default:
        return null;
    }
  };

  const renderWorkspacePanel = (panelId: FloatingStudioPanelId) => {
    const definition = getWorkspacePanelDefinition(panelId);
    if (!definition) {
      return null;
    }
    const renderer =
      floatingStudioPanels[panelId].mode === "floating"
        ? renderFloatingStudioPanel
        : renderDockedStudioPanel;
    return renderer(panelId, definition.title, definition.content, {
      badge: definition.badge,
      panelDomId: definition.panelDomId,
      bodyClassName: definition.bodyClassName,
    });
  };

  const activeBottomTrayPanelId =
    dockedBottomPanelIds.includes(studioBottomTray.activePanelId)
      ? studioBottomTray.activePanelId
      : dockedBottomPanelIds[0];
  const activeBottomTrayDefinition = activeBottomTrayPanelId
    ? getWorkspacePanelDefinition(activeBottomTrayPanelId)
    : null;

  const beginStudioBottomTrayResize = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!bottomDockRect || studioBottomTray.collapsed) {
      return;
    }
    event.preventDefault();
    setStudioBottomTrayResize({
      startY: event.clientY,
      startHeight: studioBottomTray.height,
    });
  };

  const toggleStudioBottomTrayCollapsed = () => {
    setStudioBottomTray((prev) => ({
      ...prev,
      collapsed: !prev.collapsed,
    }));
  };

  const studioShellTitle =
    activeStudioSurface === "workbench"
      ? "Process Flow Designer"
      : `Workflow Builder: ${composerDraft.summary.trim() || "Pipeline Alpha"}`;
  const studioShellBreadcrumbLabel =
    activeStudioSurface === "workbench"
      ? "Process Flow Designer"
      : composerDraft.summary.trim() || "Workflow Builder draft";

  return (
    <AppShell
      activeScreen="studio"
      title={studioShellTitle}
      breadcrumbs={[
        { label: "Project", href: "/project" },
        { label: "Workflows", href: "/workflows" },
        { label: studioShellBreadcrumbLabel },
      ]}
      actions={
        <>
          <div className="inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/[0.04] p-1">
            {(["workflow", "workbench"] as StudioSurface[]).map((surface) => (
              <button
                key={surface}
                type="button"
                className={`rounded-lg px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] transition ${
                  activeStudioSurface === surface
                    ? "bg-sky-400/18 text-sky-50"
                    : "text-slate-100 hover:bg-white/[0.08]"
                }`}
                onClick={() => switchStudioSurface(surface)}
              >
                {surface === "workflow" ? "builder" : "canvas"}
              </button>
            ))}
          </div>
          {activeStudioSurface === "workflow" ? (
            <>
              <button
                className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                onClick={startFreshStudioDraft}
              >
                New Workflow
              </button>
              <button
                className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                onClick={saveWorkflowDefinition}
                disabled={workflowActionLoading !== null}
              >
                {workflowActionLoading === "save" ? "Saving..." : "Save"}
              </button>
              <button
                className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                onClick={publishWorkflowVersion}
                disabled={workflowActionLoading !== null}
              >
                {workflowActionLoading === "publish" ? "Publishing..." : "Publish"}
              </button>
              <button
                className="rounded-xl border border-slate-200/18 bg-slate-950/25 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-white transition hover:border-white/30 hover:bg-slate-950/35 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={runWorkflowVersion}
                disabled={workflowActionLoading !== null}
              >
                {workflowActionLoading === "run" ? "Starting..." : "Run Workflow"}
              </button>
            </>
          ) : null}
        </>
      }
    >
      {activeStudioSurface === "workflow" && studioNotice ? (
        <div className="mb-4 rounded-[24px] border border-sky-300/15 bg-sky-400/10 px-4 py-3 text-sm text-sky-50">
          {studioNotice}
        </div>
      ) : null}
      <section
        className={`relative ${activeStudioSurface === "workflow" ? "block" : "hidden"}`}
        aria-hidden={activeStudioSurface !== "workflow"}
      >
              <div className="relative">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-100/72">
                      Workflow Builder
                    </div>
                    <h2 className="mt-1 text-[30px] font-semibold tracking-[-0.03em] text-white">
                      Process Flow Designer
                    </h2>
                    <p className="mt-1 max-w-3xl text-[13px] leading-5 text-slate-200/74">
                      Map business logic into clear steps, decisions, tools, and AI actions before
                      running the automation.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]">
                    <button
                      type="button"
                      className={`rounded-full border px-3 py-1 transition ${
                        studioWorkspaceMode === "focus_graph"
                          ? "border-sky-300/35 bg-sky-400/18 text-sky-50"
                          : "border-white/10 bg-white/[0.05] text-slate-100 hover:border-white/16 hover:bg-white/[0.08]"
                      }`}
                      onClick={toggleFocusGraphMode}
                    >
                      {studioWorkspaceMode === "focus_graph" ? "exit focus" : "focus graph"}
                    </button>
                    <button
                      type="button"
                      className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100 transition hover:border-white/16 hover:bg-white/[0.08]"
                      onClick={resetFloatingStudioWorkspaceLayout}
                    >
                      reset layout
                    </button>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
                      steps {visualChainSummary.steps}
                    </span>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
                      edges {visualChainSummary.dagEdges}
                    </span>
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
                      zoom {Math.round(dagCanvasZoom * 100)}%
                    </span>
                    <span
                      className={`rounded-full border px-3 py-1 ${
                        visualChainSummary.missingInputs > 0
                          ? "border-rose-300/25 bg-rose-400/12 text-rose-100"
                          : "border-emerald-300/25 bg-emerald-400/12 text-emerald-100"
                      }`}
                    >
                      {visualChainSummary.missingInputs > 0
                        ? `missing ${visualChainSummary.missingInputs}`
                        : "inputs ready"}
                    </span>
                  </div>
                </div>

                <div
                  ref={studioWorkspaceStageRef}
                  id="studio-graph-section"
                  className={`relative mt-4 h-[calc(100vh-184px)] min-h-[980px] overflow-hidden rounded-[30px] bg-[linear-gradient(180deg,rgba(75,92,109,0.58),rgba(45,57,71,0.72))] shadow-[0_22px_56px_rgba(15,23,42,0.16)] ${
                    studioWorkspaceMode === "focus_graph"
                      ? "ring-2 ring-sky-300/25"
                      : "ring-1 ring-white/10"
                  }`}
                >
                  <div className="pointer-events-none absolute left-1/2 top-4 z-20 -translate-x-1/2 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-100/58">
                    {studioWorkspaceMode === "focus_graph"
                      ? "Focus Graph active. Press F to restore your workspace."
                      : "Minimized panels collapse into the stage shelf so the graph stays clear"}
                  </div>

                  <div
                    className="absolute"
                    style={{
                      left: workspacePanelRects.graph.x,
                      top: workspacePanelRects.graph.y,
                      width: workspacePanelRects.graph.width,
                      height: workspacePanelRects.graph.height,
                    }}
                  >
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
                      dagCanvasZoom={dagCanvasZoom}
                      showToolbar
                      showBlueprintPreview
                      onZoomIn={zoomInDagCanvas}
                      onZoomOut={zoomOutDagCanvas}
                      zoomInDisabled={dagCanvasZoom >= DAG_CANVAS_ZOOM_MAX}
                      zoomOutDisabled={dagCanvasZoom <= DAG_CANVAS_ZOOM_MIN}
                      onToggleFocusGraph={toggleFocusGraphMode}
                      focusGraphActive={studioWorkspaceMode === "focus_graph"}
                      onRunWorkflow={runWorkflowVersion}
                      runWorkflowPending={workflowActionLoading === "run"}
                      runWorkflowDisabled={workflowActionLoading !== null}
                    />
                  </div>

                  {leftDockRect && dockedLeftPanelIds.length > 0 ? (
                    <div
                      className="pointer-events-auto absolute flex flex-col gap-3 overflow-y-auto pr-1"
                      style={{
                        left: leftDockRect.x,
                        top: leftDockRect.y,
                        width: leftDockRect.width + 6,
                        height: leftDockRect.height,
                      }}
                    >
                      <div
                        className="absolute inset-y-0 right-0 z-20 w-2 cursor-ew-resize transition hover:bg-white/[0.08]"
                        onMouseDown={(event) =>
                          beginFloatingStudioPanelResize(dockedLeftPanelIds[0], "e", event)
                        }
                        title="Resize left dock"
                      />
                      {dockedLeftPanelIds.map((panelId) => (
                        <div
                          key={`docked-left-panel-${panelId}`}
                          style={{
                            height: getFloatingStudioPanelActiveHeight(floatingStudioPanels[panelId]),
                            minHeight: FLOATING_STUDIO_PANEL_HEADER_HEIGHT,
                          }}
                        >
                          {renderWorkspacePanel(panelId)}
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {rightDockRect && dockedRightPanelIds.length > 0 ? (
                    <div
                      className="pointer-events-auto absolute flex flex-col gap-3 overflow-y-auto pr-1"
                      style={{
                        left: rightDockRect.x,
                        top: rightDockRect.y,
                        width: rightDockRect.width + 6,
                        height: rightDockRect.height,
                      }}
                    >
                      <div
                        className="absolute inset-y-0 left-0 z-20 w-2 cursor-ew-resize transition hover:bg-white/[0.08]"
                        onMouseDown={(event) =>
                          beginFloatingStudioPanelResize(dockedRightPanelIds[0], "w", event)
                        }
                        title="Resize right dock"
                      />
                      {dockedRightPanelIds.map((panelId) => (
                        <div
                          key={`docked-right-panel-${panelId}`}
                          style={{
                            height: getFloatingStudioPanelActiveHeight(floatingStudioPanels[panelId]),
                            minHeight: FLOATING_STUDIO_PANEL_HEADER_HEIGHT,
                          }}
                        >
                          {renderWorkspacePanel(panelId)}
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {bottomDockRect && dockedBottomPanelIds.length > 0 ? (
                    <div
                      className="pointer-events-auto absolute"
                      style={{
                        left: bottomDockRect.x,
                        top: bottomDockRect.y,
                        width: bottomDockRect.width,
                        height: bottomDockRect.height,
                      }}
                    >
                      <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-[24px] border border-white/12 bg-[linear-gradient(180deg,rgba(37,49,61,0.88),rgba(16,24,34,0.92))] shadow-[0_18px_36px_rgba(15,23,42,0.24)] backdrop-blur-xl">
                        <div
                          className="h-2 cursor-ns-resize transition hover:bg-white/[0.08]"
                          onMouseDown={beginStudioBottomTrayResize}
                          title="Resize bottom tray"
                        />
                        <div className="flex items-center justify-between gap-3 border-b border-white/8 bg-[rgba(9,16,27,0.46)] px-3 py-2">
                          <div className="flex min-w-0 items-center gap-2 overflow-x-auto">
                            {dockedBottomPanelIds.map((panelId) => {
                              const badge = getWorkspacePanelBadge(panelId);
                              const isActive = activeBottomTrayPanelId === panelId;
                              return (
                                <button
                                  key={`studio-bottom-tray-tab-${panelId}`}
                                  type="button"
                                  className={`flex shrink-0 items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] transition ${
                                    isActive
                                      ? "border-sky-300/35 bg-sky-400/18 text-sky-50"
                                      : "border-white/10 bg-white/[0.05] text-slate-100 hover:border-white/16 hover:bg-white/[0.08]"
                                  }`}
                                  onClick={() =>
                                    setStudioBottomTray((prev) => ({
                                      ...prev,
                                      activePanelId: panelId,
                                      collapsed: false,
                                    }))
                                  }
                                >
                                  <span>{getWorkspacePanelTitle(panelId)}</span>
                                  {badge ? (
                                    <span className="rounded-full border border-white/10 bg-white/[0.08] px-2 py-0.5 text-[9px] tracking-[0.18em] text-slate-200">
                                      {badge}
                                    </span>
                                  ) : null}
                                </button>
                              );
                            })}
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              className="rounded-full border border-white/10 bg-white/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-white/16 hover:bg-white/[0.1]"
                              onClick={toggleStudioBottomTrayCollapsed}
                            >
                              {studioBottomTray.collapsed ? "Expand" : "Collapse"}
                            </button>
                          </div>
                        </div>
                        {!studioBottomTray.collapsed && activeBottomTrayDefinition ? (
                          <div
                            id={activeBottomTrayDefinition.panelDomId}
                            className={`min-h-0 flex-1 overflow-auto ${
                              activeBottomTrayDefinition.bodyClassName || ""
                            }`.trim()}
                          >
                            {activeBottomTrayDefinition.content}
                          </div>
                        ) : (
                          <div className="flex h-full items-center justify-between px-4 py-3 text-xs uppercase tracking-[0.18em] text-slate-300/68">
                            <span>
                              {activeBottomTrayPanelId
                                ? `${getWorkspacePanelTitle(activeBottomTrayPanelId)} ready`
                                : "Bottom tray ready"}
                            </span>
                            <span>Shift+3</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : null}

                  {minimizedShelfRect && minimizedWorkspacePanelIds.length > 0 ? (
                    <div
                      className="pointer-events-auto absolute"
                      style={{
                        left: minimizedShelfRect.x,
                        top: minimizedShelfRect.y,
                        width: minimizedShelfRect.width,
                        height: minimizedShelfRect.height,
                      }}
                    >
                      <div className="flex h-full items-center gap-3 overflow-x-auto rounded-[22px] border border-white/10 bg-[rgba(9,16,27,0.52)] px-3 shadow-[0_18px_40px_rgba(15,23,42,0.2)] backdrop-blur-xl">
                        <div className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-100/62">
                          Minimized
                        </div>
                        <div className="flex min-w-0 items-center gap-2">
                          {minimizedWorkspacePanelIds.map((panelId) => {
                            const title = getWorkspacePanelTitle(panelId);
                            const badge = getWorkspacePanelBadge(panelId);
                            return (
                              <button
                                key={`minimized-panel-${panelId}`}
                                type="button"
                                className="flex shrink-0 items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/30 hover:bg-white/[0.11]"
                                aria-label={`Restore ${title}`}
                                title={`Restore ${title}`}
                                onClick={() => setFloatingStudioPanelMinimized(panelId, false)}
                              >
                                <span>{title}</span>
                                {badge ? (
                                  <span className="rounded-full border border-white/10 bg-white/[0.08] px-2 py-0.5 text-[9px] tracking-[0.18em] text-slate-200">
                                    {badge}
                                  </span>
                                ) : null}
                                <span className="text-[9px] tracking-[0.18em] text-sky-100/78">
                                  Restore
                                </span>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  ) : null}

                  {floatingWorkspacePanelIds.map((panelId) => renderWorkspacePanel(panelId))}
                </div>
              </div>
      </section>

      {workbenchSurfaceMounted ? (
        <StudioWorkbenchSurface
          active={activeStudioSurface === "workbench"}
          workspaceUserId={workspaceUserId}
          onPromoteWorkflowDraft={setPendingWorkbenchWorkflowDraft}
        />
      ) : null}

      <datalist id="studio-capability-id-options">
        {availableCapabilities.map((item) => (
          <option key={`studio-capability-id-option-${item.id}`} value={item.id} />
        ))}
      </datalist>
    </AppShell>
  );
}
