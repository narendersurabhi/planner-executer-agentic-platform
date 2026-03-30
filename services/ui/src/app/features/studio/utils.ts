"use client";

import type {
  CanvasPoint,
  CapabilitySchemaField,
  CapabilityItem,
  ChainPreflightResult,
  ComposerCompileResponse,
  ComposerDraftEdge,
  ComposerDraftNode,
  ComposerValidationIssue,
} from "./types";

export const DAG_CANVAS_NODE_WIDTH = 320;
export const DAG_CANVAS_NODE_HEIGHT = 76;
export const DAG_CANVAS_PADDING = 16;
export const DAG_CANVAS_SNAP = 8;
export const DAG_CANVAS_MIN_WIDTH = 960;
export const DAG_CANVAS_MIN_HEIGHT = 460;

export const CHAINABLE_REQUIRED_FIELDS = new Set([
  "document_spec",
  "validation_report",
  "errors",
  "original_spec",
  "data",
  "path",
  "text",
  "content",
  "openapi_spec",
]);

const topLevelFieldFromPath = (path: string) => path.split(/[.[\]]/)[0].trim();

export const getCapabilityRequiredInputs = (item: CapabilityItem | undefined | null): string[] => {
  if (!item) {
    return [];
  }
  const explicit = (item.required_inputs || [])
    .map((value) => String(value || "").trim())
    .filter(Boolean);
  if (explicit.length > 0) {
    return Array.from(new Set(explicit));
  }

  const fromInputSchema =
    item.input_schema &&
    typeof item.input_schema === "object" &&
    !Array.isArray(item.input_schema) &&
    Array.isArray((item.input_schema as Record<string, unknown>).required)
      ? ((item.input_schema as Record<string, unknown>).required as unknown[])
          .map((value) => String(value || "").trim())
          .filter(Boolean)
      : [];
  if (fromInputSchema.length > 0) {
    return Array.from(new Set(fromInputSchema));
  }

  const requiredFromFields = (item.input_fields || [])
    .filter((field) => field.required)
    .map((field) => topLevelFieldFromPath(field.path))
    .filter(Boolean);
  return Array.from(new Set(requiredFromFields));
};

export const capabilityInputSchemaProperties = (
  item: CapabilityItem | undefined | null
): Record<string, Record<string, unknown>> => {
  if (!item?.input_schema || typeof item.input_schema !== "object" || Array.isArray(item.input_schema)) {
    return {};
  }
  const properties = (item.input_schema as Record<string, unknown>).properties;
  if (!properties || typeof properties !== "object" || Array.isArray(properties)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(properties).filter(
      ([, value]) => value && typeof value === "object" && !Array.isArray(value)
    )
  ) as Record<string, Record<string, unknown>>;
};

export const capabilityOutputSchemaProperties = (
  item: CapabilityItem | undefined | null
): Record<string, Record<string, unknown>> => {
  if (!item?.output_schema || typeof item.output_schema !== "object" || Array.isArray(item.output_schema)) {
    return {};
  }
  const properties = (item.output_schema as Record<string, unknown>).properties;
  if (!properties || typeof properties !== "object" || Array.isArray(properties)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(properties).filter(
      ([, value]) => value && typeof value === "object" && !Array.isArray(value)
    )
  ) as Record<string, Record<string, unknown>>;
};

export const capabilityOutputSchemaFields = (
  item: CapabilityItem | undefined | null
): CapabilitySchemaField[] => {
  if (!item) {
    return [];
  }
  const explicitFields = Array.isArray(item.output_fields)
    ? item.output_fields
        .map((field) => ({
          path: String(field.path || "").trim(),
          type: String(field.type || "string").trim() || "string",
          required: Boolean(field.required),
          description:
            typeof field.description === "string" ? field.description : null,
        }))
        .filter((field) => field.path)
    : [];
  const seen = new Set(explicitFields.map((field) => field.path));
  const schemaProperties = capabilityOutputSchemaProperties(item);
  const inferredFields = Object.entries(schemaProperties)
    .filter(([path]) => !seen.has(path))
    .map(([path, property]) => ({
      path,
      type: schemaPropertyTypeLabel(property),
      required: false,
      description:
        typeof property.description === "string" ? property.description : null,
    }));
  return [...explicitFields, ...inferredFields];
};

export const schemaPropertyTypeLabel = (property: Record<string, unknown> | undefined) => {
  if (!property) {
    return "string";
  }
  const rawType = property.type;
  if (Array.isArray(rawType)) {
    return rawType.map((value) => String(value)).join(" | ") || "string";
  }
  if (typeof rawType === "string" && rawType.trim()) {
    return rawType;
  }
  return "string";
};

export const isContextInputPresent = (value: unknown) => {
  if (value === undefined || value === null) {
    return false;
  }
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  if (typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).length > 0;
  }
  return true;
};

export const collectContextPathSuggestions = (
  value: unknown,
  options: { maxDepth?: number } = {}
): string[] => {
  const maxDepth = options.maxDepth ?? 4;
  const suggestions = new Set<string>();

  const visit = (current: unknown, path: string, depth: number) => {
    if (depth > maxDepth) {
      return;
    }
    if (path) {
      suggestions.add(path);
    }
    if (Array.isArray(current)) {
      if (current.length > 0) {
        visit(current[0], path ? `${path}[0]` : "[0]", depth + 1);
      }
      return;
    }
    if (!current || typeof current !== "object") {
      return;
    }
    Object.entries(current as Record<string, unknown>).forEach(([key, nested]) => {
      const nextPath = path ? `${path}.${key}` : key;
      visit(nested, nextPath, depth + 1);
    });
  };

  visit(value, "", 0);
  return [...suggestions].sort((left, right) => left.localeCompare(right));
};

export const inferCapabilityOutputPath = (capabilityId: string) => {
  if (capabilityId.includes("spec.validate")) {
    return "validation_report";
  }
  if (capabilityId.includes("spec")) {
    if (capabilityId.includes("openapi")) {
      return "openapi_spec";
    }
    return "document_spec";
  }
  if (capabilityId.includes("docx") || capabilityId.includes("pdf") || capabilityId.includes("render")) {
    return "path";
  }
  if (capabilityId.includes("json.transform")) {
    return "result";
  }
  if (capabilityId.includes("llm.text.generate")) {
    return "text";
  }
  return "result";
};

export const inferOutputExtensionForCapability = (capabilityId: string): string => {
  const normalized = capabilityId.toLowerCase();
  if (normalized.includes(".docx.")) {
    return "docx";
  }
  if (normalized.includes(".pdf.")) {
    return "pdf";
  }
  if (normalized.includes("write_text")) {
    return "txt";
  }
  if (normalized.includes("write_content")) {
    return "json";
  }
  return "txt";
};

export const isPathOutputReference = (sourcePath: string): boolean => {
  const normalized = sourcePath.trim().toLowerCase();
  if (!normalized) {
    return false;
  }
  return (
    normalized === "path" ||
    normalized === "output_path" ||
    normalized.endsWith(".path") ||
    normalized.endsWith(".output_path")
  );
};

export const taskNameFromCapability = (capabilityId: string) => {
  const cleaned = capabilityId.replace(/[^a-zA-Z0-9]+/g, " ").trim();
  if (!cleaned) {
    return "Task";
  }
  const pascal = cleaned
    .split(/\s+/)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join("");
  return pascal || "Task";
};

export const outputPathSuggestionsForCapability = (
  capabilityId: string,
  preferredOutputPath?: string
) => {
  const candidates = new Set<string>();
  const normalized = capabilityId.toLowerCase();
  const add = (value: string) => {
    const trimmed = value.trim();
    if (trimmed) {
      candidates.add(trimmed);
    }
  };
  add(preferredOutputPath || "");
  add(inferCapabilityOutputPath(capabilityId));
  add("result");
  if (normalized.includes("spec")) {
    add("document_spec");
    add("openapi_spec");
    add("validation_report");
  }
  if (normalized.includes("docx") || normalized.includes("pdf") || normalized.includes("render")) {
    add("path");
  }
  if (normalized.includes("filename")) {
    add("path");
    add("output_path");
  }
  if (normalized.includes("json.transform")) {
    add("result");
    add("data");
  }
  if (normalized.includes("llm.text.generate")) {
    add("text");
  }
  return [...candidates];
};

export const outputPathSuggestionsForNode = (node: ComposerDraftNode | undefined) => {
  return outputPathSuggestionsForNodeWithCapability(node);
};

export const outputPathSuggestionsForNodeWithCapability = (
  node: ComposerDraftNode | undefined,
  capability?: CapabilityItem | null
) => {
  if (!node) {
    return ["result"];
  }
  const suggestions = new Set(outputPathSuggestionsForCapability(node.capabilityId, node.outputPath));
  capabilityOutputSchemaFields(capability).forEach((field) => {
    suggestions.add(field.path);
    const topLevel = topLevelFieldFromPath(field.path);
    if (topLevel) {
      suggestions.add(topLevel);
    }
  });
  node.outputs.forEach((output) => {
    if (output.name.trim()) {
      suggestions.add(output.name.trim());
    }
    if (output.path.trim()) {
      suggestions.add(output.path.trim());
    }
  });
  return [...suggestions];
};

export const uniqueTaskName = (
  baseName: string,
  nodes: ComposerDraftNode[],
  skipNodeId?: string
) => {
  const cleaned = baseName.trim() || "Task";
  const existing = new Set(
    nodes.filter((node) => node.id !== skipNodeId).map((node) => node.taskName.trim())
  );
  if (!existing.has(cleaned)) {
    return cleaned;
  }
  let suffix = 2;
  let candidate = `${cleaned}${suffix}`;
  while (existing.has(candidate)) {
    suffix += 1;
    candidate = `${cleaned}${suffix}`;
  }
  return candidate;
};

export const normalizeComposerEdges = (
  nodes: ComposerDraftNode[],
  edges: ComposerDraftEdge[]
): ComposerDraftEdge[] => {
  const nodeIds = new Set(nodes.map((node) => node.id));
  const dedupe = new Set<string>();
  const normalized: ComposerDraftEdge[] = [];
  edges.forEach((edge) => {
    const fromNodeId = String(edge.fromNodeId || "").trim();
    const toNodeId = String(edge.toNodeId || "").trim();
    const branchLabel = String(edge.branchLabel || "").trim();
    if (!fromNodeId || !toNodeId || fromNodeId === toNodeId) {
      return;
    }
    if (!nodeIds.has(fromNodeId) || !nodeIds.has(toNodeId)) {
      return;
    }
    const key = `${fromNodeId}->${toNodeId}`;
    if (dedupe.has(key)) {
      return;
    }
    dedupe.add(key);
    normalized.push({ fromNodeId, toNodeId, ...(branchLabel ? { branchLabel } : {}) });
  });
  return normalized;
};

export const buildSequentialComposerEdges = (nodes: ComposerDraftNode[]): ComposerDraftEdge[] =>
  nodes.slice(1).map((node, index) => ({
    fromNodeId: nodes[index].id,
    toNodeId: node.id,
  }));

export const defaultDagNodePosition = (index: number): CanvasPoint => {
  const columns = 4;
  const column = index % columns;
  const row = Math.floor(index / columns);
  return {
    x: DAG_CANVAS_PADDING + column * (DAG_CANVAS_NODE_WIDTH + 32),
    y: DAG_CANVAS_PADDING + row * (DAG_CANVAS_NODE_HEIGHT + 24),
  };
};

export const isInteractiveCanvasTarget = (target: EventTarget | null) => {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return Boolean(target.closest("button,a,input,select,textarea,label"));
};

export const detectDagCycle = (nodeIds: string[], edges: ComposerDraftEdge[]) => {
  const indegree = new Map<string, number>();
  const outgoing = new Map<string, string[]>();
  nodeIds.forEach((nodeId) => {
    indegree.set(nodeId, 0);
    outgoing.set(nodeId, []);
  });
  edges.forEach((edge) => {
    if (!indegree.has(edge.fromNodeId) || !indegree.has(edge.toNodeId)) {
      return;
    }
    outgoing.get(edge.fromNodeId)?.push(edge.toNodeId);
    indegree.set(edge.toNodeId, (indegree.get(edge.toNodeId) || 0) + 1);
  });
  const queue: string[] = [];
  indegree.forEach((value, key) => {
    if (value === 0) {
      queue.push(key);
    }
  });
  let visited = 0;
  while (queue.length > 0) {
    const current = queue.shift()!;
    visited += 1;
    const neighbors = outgoing.get(current) || [];
    neighbors.forEach((next) => {
      const updated = (indegree.get(next) || 0) - 1;
      indegree.set(next, updated);
      if (updated === 0) {
        queue.push(next);
      }
    });
  }
  return visited !== nodeIds.length;
};

export const readContextObject = (value: string): { context: Record<string, unknown>; invalid: boolean } => {
  try {
    const parsed = JSON.parse(value || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return { context: parsed as Record<string, unknown>, invalid: false };
    }
    return { context: {}, invalid: true };
  } catch {
    return { context: {}, invalid: true };
  }
};

const extractStepTaskName = (message: string): string | null => {
  const stepMatch = message.match(/^Step\s+([^:]+):/);
  if (!stepMatch || !stepMatch[1]) {
    return null;
  }
  const raw = stepMatch[1].trim();
  if (!raw) {
    return null;
  }
  const indexMatch = raw.match(/^\d+\s+\((.+)\)$/);
  if (indexMatch && indexMatch[1]) {
    return indexMatch[1].trim() || null;
  }
  return raw;
};

const extractFieldHint = (message: string): string | null => {
  const quotedMatch = message.match(/'([^']+)'/);
  if (quotedMatch && quotedMatch[1]) {
    return quotedMatch[1].trim() || null;
  }
  return null;
};

const findNodeIdByTaskName = (
  taskName: string | null | undefined,
  nodes: ComposerDraftNode[]
): string | undefined => {
  if (!taskName) {
    return undefined;
  }
  const match = nodes.find((node) => node.taskName.trim() === taskName.trim());
  return match?.id;
};

export const collectComposerValidationIssues = (
  preflightResult: ChainPreflightResult | null,
  compileResult: ComposerCompileResponse | null,
  nodes: ComposerDraftNode[]
): ComposerValidationIssue[] => {
  const issues: ComposerValidationIssue[] = [];
  const push = (issue: ComposerValidationIssue) => issues.push(issue);

  if (preflightResult) {
    preflightResult.localErrors.forEach((message) => {
      const taskName = extractStepTaskName(message);
      push({
        severity: "error",
        source: "local",
        code: "local_check",
        message,
        nodeId: findNodeIdByTaskName(taskName, nodes),
        field: extractFieldHint(message) || undefined,
      });
    });
    if (Array.isArray(preflightResult.serverDiagnostics) && preflightResult.serverDiagnostics.length > 0) {
      preflightResult.serverDiagnostics.forEach((diag) => {
        const field = typeof diag.field === "string" ? diag.field : undefined;
        push({
          severity: diag.severity === "warning" ? "warning" : "error",
          source: "preflight",
          code: diag.code || "preflight_error",
          field,
          nodeId: findNodeIdByTaskName(field, nodes),
          message: diag.message || "Preflight validation failed.",
        });
      });
    } else {
      Object.entries(preflightResult.serverErrors).forEach(([field, message]) => {
        push({
          severity: "error",
          source: "preflight",
          code: "preflight_error",
          field,
          nodeId: findNodeIdByTaskName(field, nodes),
          message,
        });
      });
    }
  }

  if (compileResult) {
    compileResult.diagnostics.errors.forEach((diag) => {
      push({
        severity: "error",
        source: "compile",
        code: diag.code || "compile_error",
        field: diag.field,
        nodeId: diag.node_id,
        message: diag.message,
      });
    });
    compileResult.diagnostics.warnings.forEach((diag) => {
      push({
        severity: "warning",
        source: "compile",
        code: diag.code || "compile_warning",
        field: diag.field,
        nodeId: diag.node_id,
        message: diag.message,
      });
    });
  }

  const dedupe = new Set<string>();
  const deduped: ComposerValidationIssue[] = [];
  issues.forEach((issue) => {
    const key = `${issue.severity}|${issue.source}|${issue.code}|${issue.field || ""}|${
      issue.nodeId || ""
    }|${issue.message}`;
    if (dedupe.has(key)) {
      return;
    }
    dedupe.add(key);
    deduped.push(issue);
  });

  const severityRank: Record<ComposerValidationIssue["severity"], number> = {
    error: 0,
    warning: 1,
  };
  const sourceRank: Record<ComposerValidationIssue["source"], number> = {
    local: 0,
    compile: 1,
    preflight: 2,
  };

  deduped.sort((left, right) => {
    const severityDiff = severityRank[left.severity] - severityRank[right.severity];
    if (severityDiff !== 0) {
      return severityDiff;
    }
    const sourceDiff = sourceRank[left.source] - sourceRank[right.source];
    if (sourceDiff !== 0) {
      return sourceDiff;
    }
    return left.message.localeCompare(right.message);
  });

  return deduped;
};

export const formatTimestamp = (value?: string) => {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
};
