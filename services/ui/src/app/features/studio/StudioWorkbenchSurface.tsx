"use client";

import { useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";

import StudioWorkbenchIcon from "./StudioWorkbenchIcon";
import {
  createAgentDefinition,
  deleteAgentDefinition,
  fetchAgentDefinitions,
  fetchCapabilityCatalog,
  fetchRunDebugger,
  launchAgentRun,
  launchCapabilityRun,
  searchCapabilities,
  updateAgentDefinition,
  type CapabilitySearchItem,
  type WorkbenchDebuggerData,
  type WorkbenchRunLaunchResponse,
} from "./studioApi";
import { mapDebuggerStepToReplayDraft, mapRunToWorkbenchFork, mapRunToWorkflowPromotion } from "./studioWorkbenchMappings";
import type {
  AgentDefinition,
  AgentDefinitionCreateRequest,
  CapabilityItem,
  ReplayableCapabilityDraft,
  StudioSurface,
  WorkbenchWorkflowPromotionDraft,
} from "./types";

type WorkbenchMode = "capability" | "agent";
type AgentEditorMode = "structured" | "raw";
type AgentStepRole = "agent" | "step";

type CapabilityInputDraft = Record<string, string | boolean>;

type AgentStepDraft = {
  localId: string;
  stepId: string;
  name: string;
  description: string;
  instruction: string;
  capabilityId: string;
  dependsOnText: string;
  inputDraft: CapabilityInputDraft;
  rawInputOverrideEnabled: boolean;
  inputJsonText: string;
  retryPolicyText: string;
};

const TERMINAL_RUN_STATUSES = new Set(["succeeded", "failed", "canceled", "accepted", "completed"]);
const DEFAULT_AGENT_CAPABILITY_ID = "codegen.autonomous";
const DEFAULT_AGENT_WORKSPACE_PATH = "workbench-agent";
const DEFAULT_AGENT_MAX_STEPS = "6";

const DEFAULT_RETRY_POLICY_PREVIEW = {
  max_attempts: 1,
  retry_class: "standard",
  retryable_errors: [],
  backoff_seconds: 0,
  backoff_multiplier: 1,
  jitter_seconds: 0,
};

type WorkbenchBanner = {
  tone: "info" | "warning";
  message: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return "{}";
  }
}

function parseJsonObject(
  text: string,
  label: string,
  fallback: Record<string, unknown> = {}
): { value: Record<string, unknown> | null; error: string | null } {
  const normalized = text.trim();
  if (!normalized) {
    return { value: fallback, error: null };
  }
  try {
    const parsed = JSON.parse(normalized);
    if (!isRecord(parsed)) {
      return { value: null, error: `${label} must be a JSON object.` };
    }
    return { value: parsed, error: null };
  } catch (error) {
    return {
      value: null,
      error: error instanceof Error ? `${label}: ${error.message}` : `${label} is invalid JSON.`,
    };
  }
}

function slugify(value: string, fallback: string): string {
  const normalized = value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return normalized || fallback;
}

function defaultAgentInputDraft(capabilityId: string): CapabilityInputDraft {
  if (capabilityId === DEFAULT_AGENT_CAPABILITY_ID) {
    return {
      workspace_path: DEFAULT_AGENT_WORKSPACE_PATH,
      max_steps: DEFAULT_AGENT_MAX_STEPS,
    };
  }
  return {};
}

function defaultStepName(capabilityId: string, role: AgentStepRole): string {
  if (!capabilityId) {
    return "";
  }
  if (role === "agent") {
    return "Agent";
  }
  return capabilityId;
}

function defaultStepDescription(capabilityId: string, role: AgentStepRole): string {
  if (!capabilityId) {
    return "";
  }
  if (role === "agent") {
    return "Autonomous agent for the primary workbench task.";
  }
  return `Workbench step for ${capabilityId}`;
}

function defaultStepInstruction(capabilityId: string, role: AgentStepRole): string {
  if (!capabilityId) {
    return "";
  }
  if (role === "agent") {
    return "Plan and execute the requested task in the selected workspace.";
  }
  return `Run ${capabilityId} with the provided inputs.`;
}

function isAgenticCapability(item: CapabilityItem): boolean {
  const id = item.id.toLowerCase();
  const tags = item.tags.map((tag) => tag.toLowerCase());
  return (
    id === DEFAULT_AGENT_CAPABILITY_ID ||
    id.includes(".autonomous") ||
    tags.includes("autonomous") ||
    tags.includes("coding-agent")
  );
}

function stringInputValue(inputDraft: CapabilityInputDraft, key: string): string {
  const value = inputDraft[key];
  return typeof value === "string" ? value : "";
}

function normalizeSchemaType(schema: Record<string, unknown> | undefined): string {
  const type = schema?.type;
  if (typeof type === "string") {
    return type;
  }
  if (Array.isArray(type)) {
    const firstString = type.find((item) => typeof item === "string");
    return typeof firstString === "string" ? firstString : "string";
  }
  return "string";
}

function getCapabilitySchemaProperties(
  capability: CapabilityItem | null
): [string, Record<string, unknown>][] {
  const rawProperties = capability?.input_schema?.properties;
  return isRecord(rawProperties)
    ? Object.entries(rawProperties).filter((entry): entry is [string, Record<string, unknown>] =>
        isRecord(entry[1])
      )
    : [];
}

function buildStructuredCapabilityInputs(
  capability: CapabilityItem | null,
  inputDraft: CapabilityInputDraft,
  labelPrefix = "Input"
): { value: Record<string, unknown> | null; error: string | null } {
  if (!capability) {
    return { value: null, error: "Choose a capability to configure its inputs." };
  }
  const schemaProperties = getCapabilitySchemaProperties(capability);
  const nextInputs: Record<string, unknown> = {};
  const requiredFields = new Set(capability.required_inputs ?? []);
  for (const [fieldName, schema] of schemaProperties) {
    const rawValue = inputDraft[fieldName];
    const fieldType = normalizeSchemaType(schema);
    const required = requiredFields.has(fieldName);

    if (fieldType === "boolean") {
      if (typeof rawValue === "boolean") {
        nextInputs[fieldName] = rawValue;
      } else if (required) {
        nextInputs[fieldName] = false;
      }
      continue;
    }

    const rawText = typeof rawValue === "string" ? rawValue : "";
    if (!rawText.trim()) {
      if (required) {
        return { value: null, error: `${labelPrefix} '${fieldName}' is required.` };
      }
      continue;
    }

    if (fieldType === "integer" || fieldType === "number") {
      const parsedNumber = Number(rawText);
      if (Number.isNaN(parsedNumber)) {
        return { value: null, error: `${labelPrefix} '${fieldName}' must be a number.` };
      }
      nextInputs[fieldName] = fieldType === "integer" ? Math.trunc(parsedNumber) : parsedNumber;
      continue;
    }

    if (fieldType === "object" || fieldType === "array") {
      try {
        nextInputs[fieldName] = JSON.parse(rawText);
      } catch (error) {
        return {
          value: null,
          error:
            error instanceof Error
              ? `${labelPrefix} '${fieldName}' is invalid JSON: ${error.message}`
              : `${labelPrefix} '${fieldName}' is invalid JSON.`,
        };
      }
      continue;
    }

    nextInputs[fieldName] = rawText;
  }
  return { value: nextInputs, error: null };
}

function splitDependencyList(value: string): string[] {
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitConstraintList(value: string): string[] {
  return value
    .split(/\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function createAgentProfileInputDraft(definition: AgentDefinition): CapabilityInputDraft {
  const inputDraft: CapabilityInputDraft = {};
  if (definition.default_goal.trim()) {
    inputDraft.goal = definition.default_goal;
  }
  const workspacePath =
    definition.default_workspace_path?.trim() ||
    (definition.agent_capability_id === DEFAULT_AGENT_CAPABILITY_ID ? DEFAULT_AGENT_WORKSPACE_PATH : "");
  if (workspacePath) {
    inputDraft.workspace_path = workspacePath;
  }
  if (definition.default_constraints.length > 0) {
    inputDraft.constraints = definition.default_constraints.join("\n");
  }
  const maxSteps =
    definition.default_max_steps ??
    (definition.agent_capability_id === DEFAULT_AGENT_CAPABILITY_ID
      ? Number(DEFAULT_AGENT_MAX_STEPS)
      : null);
  if (maxSteps !== null) {
    inputDraft.max_steps = String(maxSteps);
  }
  return inputDraft;
}

function createAgentStepDraft(capabilityId = "", role: AgentStepRole = "step"): AgentStepDraft {
  const baseId =
    role === "agent"
      ? "agent"
      : slugify(capabilityId || `step_${Date.now()}`, "step");
  return {
    localId: `agent-step-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    stepId: baseId,
    name: defaultStepName(capabilityId, role),
    description: defaultStepDescription(capabilityId, role),
    instruction: defaultStepInstruction(capabilityId, role),
    capabilityId,
    dependsOnText: "",
    inputDraft: role === "agent" ? defaultAgentInputDraft(capabilityId) : {},
    rawInputOverrideEnabled: false,
    inputJsonText: "{\n  \n}",
    retryPolicyText: "",
  };
}

function createAgentStepDraftFromDefinition(definition: AgentDefinition): AgentStepDraft {
  const capabilityId = definition.agent_capability_id || DEFAULT_AGENT_CAPABILITY_ID;
  const inputDraft = createAgentProfileInputDraft(definition);
  const draft = createAgentStepDraft(capabilityId, "agent");
  return {
    ...draft,
    name: "Agent",
    description: definition.description?.trim() || defaultStepDescription(capabilityId, "agent"),
    instruction: definition.instructions || defaultStepInstruction(capabilityId, "agent"),
    inputDraft,
    rawInputOverrideEnabled: false,
    inputJsonText: formatJson(inputDraft),
  };
}

function sortAgentDefinitions(definitions: AgentDefinition[]): AgentDefinition[] {
  return [...definitions].sort((left, right) => right.updated_at.localeCompare(left.updated_at));
}

function buildCapabilityRunSpecPreview(
  capabilityId: string,
  inputs: Record<string, unknown>,
  retryPolicy?: Record<string, unknown> | null
): Record<string, unknown> {
  const stepId = slugify(capabilityId, "capability_step");
  const resolvedRetryPolicy =
    retryPolicy && Object.keys(retryPolicy).length > 0 ? retryPolicy : DEFAULT_RETRY_POLICY_PREVIEW;
  return {
    version: "1",
    kind: "api",
    planner_version: "workbench_v1",
    tasks_summary: `Workbench capability run: ${capabilityId}`,
    steps: [
      {
        step_id: stepId,
        name: stepId,
        description: `Capability sandbox run for ${capabilityId}`,
        instruction: `Execute capability ${capabilityId}.`,
        capability_request: {
          request_id: capabilityId,
          capability_id: capabilityId,
          execution_request_id: capabilityId,
        },
        input_bindings: inputs,
        retry_policy: resolvedRetryPolicy,
        acceptance_policy: {
          acceptance_criteria: [],
          critic_required: false,
        },
        depends_on: [],
      },
    ],
    dag_edges: [],
    capability_requests: [
      {
        request_id: capabilityId,
        capability_id: capabilityId,
        execution_request_id: capabilityId,
      },
    ],
    metadata: {
      surface: "studio_workbench",
      workbench_mode: "capability",
      ephemeral: true,
    },
  };
}

function collectArtifacts(debuggerData: WorkbenchDebuggerData | null): Record<string, unknown>[] {
  if (!debuggerData) {
    return [];
  }
  const artifacts: Record<string, unknown>[] = [];
  for (const step of debuggerData.steps) {
    const stepArtifacts = step.latest_result?.artifacts;
    if (!Array.isArray(stepArtifacts)) {
      continue;
    }
    for (const artifact of stepArtifacts) {
      if (isRecord(artifact)) {
        artifacts.push(artifact);
      }
    }
  }
  return artifacts;
}

function capabilityEditorStateFromInputs(
  capability: CapabilityItem | null,
  inputs: Record<string, unknown>
): {
  inputDraft: CapabilityInputDraft;
  rawOverrideEnabled: boolean;
  rawOverrideText: string;
} {
  const rawOverrideText = formatJson(inputs);
  if (!capability) {
    return {
      inputDraft: {},
      rawOverrideEnabled: true,
      rawOverrideText,
    };
  }
  const properties = capability.input_schema?.properties;
  if (!isRecord(properties)) {
    return {
      inputDraft: {},
      rawOverrideEnabled: true,
      rawOverrideText,
    };
  }
  const schemaEntries = Object.entries(properties).filter((entry): entry is [string, Record<string, unknown>] =>
    isRecord(entry[1])
  );
  if (schemaEntries.length === 0) {
    return {
      inputDraft: {},
      rawOverrideEnabled: Object.keys(inputs).length > 0,
      rawOverrideText,
    };
  }
  const structuredDraft: CapabilityInputDraft = {};
  let requiresRawOverride = false;
  const knownFieldNames = new Set(schemaEntries.map(([fieldName]) => fieldName));
  Object.keys(inputs).forEach((fieldName) => {
    if (!knownFieldNames.has(fieldName)) {
      requiresRawOverride = true;
    }
  });
  schemaEntries.forEach(([fieldName, schema]) => {
    const value = inputs[fieldName];
    if (value === undefined) {
      return;
    }
    const fieldType = normalizeSchemaType(schema);
    if (fieldType === "boolean") {
      if (typeof value === "boolean") {
        structuredDraft[fieldName] = value;
        return;
      }
      if (typeof value === "string" && (value === "true" || value === "false")) {
        structuredDraft[fieldName] = value === "true";
        return;
      }
      requiresRawOverride = true;
      return;
    }
    if (fieldType === "object" || fieldType === "array") {
      if (typeof value === "string") {
        structuredDraft[fieldName] = value;
        return;
      }
      if (Array.isArray(value) || isRecord(value)) {
        structuredDraft[fieldName] = formatJson(value);
        return;
      }
      requiresRawOverride = true;
      return;
    }
    if (fieldType === "integer" || fieldType === "number") {
      if (typeof value === "number" || typeof value === "string") {
        structuredDraft[fieldName] = String(value);
        return;
      }
      requiresRawOverride = true;
      return;
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      structuredDraft[fieldName] = String(value);
      return;
    }
    requiresRawOverride = true;
  });
  return {
    inputDraft: structuredDraft,
    rawOverrideEnabled: requiresRawOverride,
    rawOverrideText,
  };
}

function SurfacePanel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(21,30,40,0.9),rgba(10,16,25,0.94))] p-4 shadow-[0_16px_34px_rgba(15,23,42,0.18)] ${className}`.trim()}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-100/70">
            {title}
          </div>
          {subtitle ? <p className="mt-1 text-xs leading-5 text-slate-300/78">{subtitle}</p> : null}
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function JsonPreview({
  title,
  value,
  emptyLabel,
}: {
  title: string;
  value: unknown;
  emptyLabel: string;
}) {
  const isEmpty =
    value == null ||
    (Array.isArray(value) && value.length === 0) ||
    (isRecord(value) && Object.keys(value).length === 0);
  return (
    <div className="rounded-2xl border border-white/8 bg-black/20 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-300/72">
        {title}
      </div>
      {isEmpty ? (
        <div className="mt-2 text-xs text-slate-400">{emptyLabel}</div>
      ) : (
        <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded-xl border border-white/8 bg-slate-950/55 p-3 text-[11px] leading-5 text-slate-100">
          {formatJson(value)}
        </pre>
      )}
    </div>
  );
}

export default function StudioWorkbenchSurface({
  active,
  workspaceUserId,
  onPromoteWorkflowDraft,
}: {
  active: boolean;
  workspaceUserId: string;
  onPromoteWorkflowDraft?: (draft: WorkbenchWorkflowPromotionDraft) => void;
}) {
  const [workbenchMode, setWorkbenchMode] = useState<WorkbenchMode>("capability");
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<CapabilityItem[]>([]);
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogSearchError, setCatalogSearchError] = useState<string | null>(null);
  const [catalogSearchLoading, setCatalogSearchLoading] = useState(false);
  const [catalogSearchItems, setCatalogSearchItems] = useState<CapabilitySearchItem[]>([]);
  const [groupFilter, setGroupFilter] = useState("all");
  const [riskFilter, setRiskFilter] = useState("all");
  const [idempotencyFilter, setIdempotencyFilter] = useState("all");
  const [selectedCapabilityId, setSelectedCapabilityId] = useState("");
  const [capabilityTitle, setCapabilityTitle] = useState("Capability workbench run");
  const [capabilityGoal, setCapabilityGoal] = useState("");
  const [capabilityUserId, setCapabilityUserId] = useState(workspaceUserId);
  const [capabilityContextJsonText, setCapabilityContextJsonText] = useState("{\n  \n}");
  const [capabilityRetryPolicyText, setCapabilityRetryPolicyText] = useState("");
  const [capabilityInputDraft, setCapabilityInputDraft] = useState<CapabilityInputDraft>({});
  const [capabilityRawOverrideEnabled, setCapabilityRawOverrideEnabled] = useState(false);
  const [capabilityRawOverrideText, setCapabilityRawOverrideText] = useState("{\n  \n}");
  const [agentTitle, setAgentTitle] = useState("Agent workbench run");
  const [agentGoal, setAgentGoal] = useState("");
  const [agentUserId, setAgentUserId] = useState(workspaceUserId);
  const [agentContextJsonText, setAgentContextJsonText] = useState("{\n  \n}");
  const [agentEditorMode, setAgentEditorMode] = useState<AgentEditorMode>("structured");
  const [agentSteps, setAgentSteps] = useState<AgentStepDraft[]>([
    createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent"),
  ]);
  const [agentRawRunSpecText, setAgentRawRunSpecText] = useState("{\n  \n}");
  const [agentDefinitions, setAgentDefinitions] = useState<AgentDefinition[]>([]);
  const [agentDefinitionsLoading, setAgentDefinitionsLoading] = useState(false);
  const [agentDefinitionsError, setAgentDefinitionsError] = useState<string | null>(null);
  const [selectedAgentDefinitionId, setSelectedAgentDefinitionId] = useState("");
  const [agentProfileName, setAgentProfileName] = useState("");
  const [agentProfileDescription, setAgentProfileDescription] = useState("");
  const [agentProfileError, setAgentProfileError] = useState<string | null>(null);
  const [agentProfileSaving, setAgentProfileSaving] = useState(false);
  const [agentProfileDeleting, setAgentProfileDeleting] = useState(false);
  const [launchLoading, setLaunchLoading] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [launchResponse, setLaunchResponse] = useState<WorkbenchRunLaunchResponse | null>(null);
  const [debuggerData, setDebuggerData] = useState<WorkbenchDebuggerData | null>(null);
  const [debuggerLoading, setDebuggerLoading] = useState(false);
  const [debuggerError, setDebuggerError] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [workbenchBanner, setWorkbenchBanner] = useState<WorkbenchBanner | null>(null);

  const deferredCatalogQuery = useDeferredValue(catalogQuery);

  useEffect(() => {
    if (!capabilityUserId.trim() && workspaceUserId.trim()) {
      setCapabilityUserId(workspaceUserId);
    }
  }, [capabilityUserId, workspaceUserId]);

  useEffect(() => {
    if (!agentUserId.trim() && workspaceUserId.trim()) {
      setAgentUserId(workspaceUserId);
    }
  }, [agentUserId, workspaceUserId]);

  useEffect(() => {
    let cancelled = false;
    const loadCatalog = async () => {
      setCatalogLoading(true);
      setCatalogError(null);
      try {
        const response = await fetchCapabilityCatalog(true);
        if (cancelled) {
          return;
        }
        setCatalog(response.items);
        setSelectedCapabilityId((current) => current || response.items[0]?.id || "");
      } catch (error) {
        if (!cancelled) {
          setCatalogError(
            error instanceof Error ? error.message : "Failed to load capability catalog."
          );
        }
      } finally {
        if (!cancelled) {
          setCatalogLoading(false);
        }
      }
    };
    void loadCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadAgentDefinitions = async () => {
      setAgentDefinitionsLoading(true);
      setAgentDefinitionsError(null);
      try {
        const response = await fetchAgentDefinitions(workspaceUserId.trim() || undefined);
        if (!cancelled) {
          setAgentDefinitions(sortAgentDefinitions(response));
        }
      } catch (error) {
        if (!cancelled) {
          setAgentDefinitionsError(
            error instanceof Error ? error.message : "Failed to load agent profiles."
          );
        }
      } finally {
        if (!cancelled) {
          setAgentDefinitionsLoading(false);
        }
      }
    };
    void loadAgentDefinitions();
    return () => {
      cancelled = true;
    };
  }, [workspaceUserId]);

  useEffect(() => {
    const query = deferredCatalogQuery.trim();
    if (!query) {
      setCatalogSearchItems([]);
      setCatalogSearchError(null);
      setCatalogSearchLoading(false);
      return;
    }
    let cancelled = false;
    const runSearch = async () => {
      setCatalogSearchLoading(true);
      setCatalogSearchError(null);
      try {
        const response = await searchCapabilities(query, 12);
        if (!cancelled) {
          setCatalogSearchItems(response.items);
        }
      } catch (error) {
        if (!cancelled) {
          setCatalogSearchItems([]);
          setCatalogSearchError(
            error instanceof Error ? error.message : "Capability search failed."
          );
        }
      } finally {
        if (!cancelled) {
          setCatalogSearchLoading(false);
        }
      }
    };
    void runSearch();
    return () => {
      cancelled = true;
    };
  }, [deferredCatalogQuery]);

  const selectedCapability = useMemo(
    () => catalog.find((item) => item.id === selectedCapabilityId) ?? null,
    [catalog, selectedCapabilityId]
  );

  const agentCapabilities = useMemo(
    () => catalog.filter(isAgenticCapability),
    [catalog]
  );

  const primaryAgentStep = agentSteps[0] ?? null;

  const primaryAgentCapability = useMemo(
    () =>
      primaryAgentStep
        ? catalog.find((item) => item.id === primaryAgentStep.capabilityId) ?? null
        : null,
    [catalog, primaryAgentStep]
  );

  const selectedAgentDefinition = useMemo(
    () => agentDefinitions.find((definition) => definition.id === selectedAgentDefinitionId) ?? null,
    [agentDefinitions, selectedAgentDefinitionId]
  );

  const groupOptions = useMemo(
    () =>
      Array.from(new Set(catalog.map((item) => item.group || "").filter(Boolean))).sort((a, b) =>
        a.localeCompare(b)
      ),
    [catalog]
  );

  const riskOptions = useMemo(
    () =>
      Array.from(new Set(catalog.map((item) => item.risk_tier).filter(Boolean))).sort((a, b) =>
        a.localeCompare(b)
      ),
    [catalog]
  );

  const idempotencyOptions = useMemo(
    () =>
      Array.from(new Set(catalog.map((item) => item.idempotency).filter(Boolean))).sort((a, b) =>
        a.localeCompare(b)
      ),
    [catalog]
  );

  const filteredCatalog = useMemo(() => {
    const normalizedQuery = deferredCatalogQuery.trim();
    const searchRank = new Map(catalogSearchItems.map((item, index) => [item.id, index]));
    return catalog
      .filter((item) => (groupFilter === "all" ? true : item.group === groupFilter))
      .filter((item) => (riskFilter === "all" ? true : item.risk_tier === riskFilter))
      .filter((item) => (idempotencyFilter === "all" ? true : item.idempotency === idempotencyFilter))
      .filter((item) => {
        if (!normalizedQuery) {
          return true;
        }
        return searchRank.has(item.id);
      })
      .sort((left, right) => {
        if (normalizedQuery) {
          return (searchRank.get(left.id) ?? Number.MAX_SAFE_INTEGER) -
            (searchRank.get(right.id) ?? Number.MAX_SAFE_INTEGER);
        }
        return left.id.localeCompare(right.id);
      });
  }, [catalog, catalogSearchItems, deferredCatalogQuery, groupFilter, idempotencyFilter, riskFilter]);

  const capabilitySchemaProperties = useMemo(() => {
    return getCapabilitySchemaProperties(selectedCapability);
  }, [selectedCapability]);

  const capabilityStructuredInputs = useMemo(() => {
    return buildStructuredCapabilityInputs(selectedCapability, capabilityInputDraft);
  }, [capabilityInputDraft, selectedCapability]);

  const capabilityRawInputs = useMemo(
    () => parseJsonObject(capabilityRawOverrideText, "Capability raw inputs"),
    [capabilityRawOverrideText]
  );

  const capabilityContextJson = useMemo(
    () => parseJsonObject(capabilityContextJsonText, "Capability context JSON"),
    [capabilityContextJsonText]
  );

  const capabilityRetryPolicy = useMemo(
    () => parseJsonObject(capabilityRetryPolicyText, "Retry policy"),
    [capabilityRetryPolicyText]
  );

  const capabilityLaunchInputs = useMemo(() => {
    if (capabilityRawOverrideEnabled) {
      return capabilityRawInputs;
    }
    return capabilityStructuredInputs;
  }, [capabilityRawInputs, capabilityRawOverrideEnabled, capabilityStructuredInputs]);

  const capabilityRunSpecPreview = useMemo(() => {
    if (!selectedCapability || !capabilityLaunchInputs.value) {
      return null;
    }
    return buildCapabilityRunSpecPreview(
      selectedCapability.id,
      capabilityLaunchInputs.value,
      capabilityRetryPolicy.value
    );
  }, [capabilityLaunchInputs.value, capabilityRetryPolicy.value, selectedCapability]);

  const agentContextJson = useMemo(
    () => parseJsonObject(agentContextJsonText, "Agent context JSON"),
    [agentContextJsonText]
  );

  const structuredAgentRunSpec = useMemo(() => {
    const normalizedTitle = agentTitle.trim() || "Agent workbench run";
    const normalizedGoal =
      agentGoal.trim() || stringInputValue(agentSteps[0]?.inputDraft ?? {}, "goal").trim();
    if (agentSteps.length === 0) {
      return { value: null, error: "Add at least one step to launch an agent run." };
    }

    const steps: Record<string, unknown>[] = [];
    const capabilityRequests: Record<string, unknown>[] = [];
    const stepIds = new Set<string>();
    const dagEdges: string[][] = [];

    for (let index = 0; index < agentSteps.length; index += 1) {
      const step = agentSteps[index];
      const capabilityId = step.capabilityId.trim();
      if (!capabilityId) {
        return { value: null, error: `Step ${index + 1} is missing a capability id.` };
      }
      const stepId = slugify(step.stepId || step.name || capabilityId, `step_${index + 1}`);
      if (stepIds.has(stepId)) {
        return { value: null, error: `Step id '${stepId}' is duplicated.` };
      }
      stepIds.add(stepId);
      const stepCapability = catalog.find((item) => item.id === capabilityId) ?? null;
      const parsedInputs =
        step.rawInputOverrideEnabled || !stepCapability
          ? parseJsonObject(step.inputJsonText, `Inputs for step '${step.name || stepId}'`)
          : buildStructuredCapabilityInputs(
              stepCapability,
              step.inputDraft,
              `Input for step '${step.name || stepId}'`
            );
      if (!parsedInputs.value) {
        return parsedInputs;
      }
      const parsedRetryPolicy = parseJsonObject(
        step.retryPolicyText,
        `Retry policy for step '${step.name || stepId}'`,
        {}
      );
      if (!parsedRetryPolicy.value) {
        return parsedRetryPolicy;
      }
      const dependsOn = splitDependencyList(step.dependsOnText);
      for (const dependency of dependsOn) {
        dagEdges.push([dependency, stepId]);
      }
      const capabilityRequest = {
        request_id: capabilityId,
        capability_id: capabilityId,
        execution_request_id: capabilityId,
      };
      steps.push({
        step_id: stepId,
        name: step.name.trim() || stepId,
        description: step.description.trim() || `Workbench step for ${capabilityId}`,
        instruction: step.instruction.trim() || `Execute capability ${capabilityId}.`,
        capability_request: capabilityRequest,
        input_bindings: parsedInputs.value,
        retry_policy:
          Object.keys(parsedRetryPolicy.value).length > 0
            ? parsedRetryPolicy.value
            : DEFAULT_RETRY_POLICY_PREVIEW,
        acceptance_policy: {
          acceptance_criteria: [],
          critic_required: false,
        },
        depends_on: dependsOn,
      });
      capabilityRequests.push(capabilityRequest);
    }

    for (const edge of dagEdges) {
      if (!stepIds.has(edge[0])) {
        return {
          value: null,
          error: `Dependency '${edge[0]}' does not match any step id in the structured agent builder.`,
        };
      }
    }

    return {
      value: {
        version: "1",
        kind: "api",
        planner_version: "workbench_v1",
        tasks_summary: normalizedGoal || normalizedTitle,
        steps,
        dag_edges: dagEdges,
        capability_requests: capabilityRequests,
        metadata: {
          surface: "studio_workbench",
          workbench_mode: "agent",
          ephemeral: true,
        },
      },
      error: null,
    };
  }, [agentGoal, agentSteps, agentTitle, catalog]);

  const rawAgentRunSpec = useMemo(
    () => parseJsonObject(agentRawRunSpecText, "Agent RunSpec"),
    [agentRawRunSpecText]
  );

  const agentRunSpecPreview = useMemo(
    () => (agentEditorMode === "raw" ? rawAgentRunSpec : structuredAgentRunSpec),
    [agentEditorMode, rawAgentRunSpec, structuredAgentRunSpec]
  );

  useEffect(() => {
    if (agentEditorMode !== "structured" || structuredAgentRunSpec.error || !structuredAgentRunSpec.value) {
      return;
    }
    const nextRawRunSpecText = formatJson(structuredAgentRunSpec.value);
    setAgentRawRunSpecText((current) =>
      current === nextRawRunSpecText ? current : nextRawRunSpecText
    );
  }, [agentEditorMode, structuredAgentRunSpec.error, structuredAgentRunSpec.value]);

  const predictedExecutionRequestPreview = useMemo(() => {
    const previewValue =
      workbenchMode === "capability" ? capabilityRunSpecPreview : agentRunSpecPreview.value;
    if (!isRecord(previewValue)) {
      return null;
    }
    const steps = previewValue.steps;
    if (!Array.isArray(steps) || steps.length === 0 || !isRecord(steps[0])) {
      return null;
    }
    const firstStep = steps[0];
    const capabilityRequest = isRecord(firstStep.capability_request) ? firstStep.capability_request : {};
    const contextPreview =
      workbenchMode === "capability" ? capabilityContextJson.value : agentContextJson.value;
    return {
      step_id: firstStep.step_id,
      request_id: capabilityRequest.execution_request_id ?? capabilityRequest.request_id,
      capability_id: capabilityRequest.capability_id,
      attempt_number: 1,
      status: "prepared",
      request: {
        requests: [
          {
            request_id: capabilityRequest.execution_request_id ?? capabilityRequest.request_id,
            capability_binding: {
              capability_id: capabilityRequest.capability_id,
            },
            input: firstStep.input_bindings ?? {},
          },
        ],
      },
      retry_policy: firstStep.retry_policy ?? {},
      policy_snapshot: firstStep.acceptance_policy ?? {},
      context_provenance: {
        job_context_keys: contextPreview ? Object.keys(contextPreview) : [],
      },
    };
  }, [
    agentContextJson.value,
    agentRunSpecPreview.value,
    capabilityContextJson.value,
    capabilityRunSpecPreview,
    workbenchMode,
  ]);

  const currentRunStatus =
    debuggerData?.run?.status ||
    launchResponse?.run?.status ||
    null;

  const replayResultsByStep = useMemo(() => {
    const next = new Map<string, ReturnType<typeof mapDebuggerStepToReplayDraft>>();
    if (!debuggerData) {
      return next;
    }
    debuggerData.steps.forEach((stepPayload) => {
      next.set(stepPayload.step.id, mapDebuggerStepToReplayDraft(debuggerData, stepPayload.step.id));
    });
    return next;
  }, [debuggerData]);

  const forkResult = useMemo(
    () => (debuggerData ? mapRunToWorkbenchFork(debuggerData) : null),
    [debuggerData]
  );

  const workflowPromotionResult = useMemo(
    () => (debuggerData ? mapRunToWorkflowPromotion(debuggerData) : null),
    [debuggerData]
  );

  const applyAgentDefinitionDraft = (definition: AgentDefinition) => {
    const primaryStep = createAgentStepDraftFromDefinition(definition);
    setWorkbenchMode("agent");
    setAgentEditorMode("structured");
    setSelectedAgentDefinitionId(definition.id);
    setAgentProfileName(definition.name);
    setAgentProfileDescription(definition.description ?? "");
    setAgentTitle(definition.name || "Agent workbench run");
    setAgentGoal(definition.default_goal || "");
    setAgentUserId(definition.user_id || workspaceUserId);
    setAgentSteps([primaryStep]);
    setAgentProfileError(null);
    setLaunchError(null);
    setWorkbenchBanner({
      tone: "info",
      message: `Loaded agent profile '${definition.name}'.`,
    });
  };

  const buildAgentDefinitionPayload = (): AgentDefinitionCreateRequest => {
    const primaryStep = primaryAgentStep ?? createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent");
    const capabilityId = primaryStep.capabilityId.trim() || DEFAULT_AGENT_CAPABILITY_ID;
    const defaultGoal =
      agentGoal.trim() || stringInputValue(primaryStep.inputDraft, "goal").trim();
    const workspacePath = stringInputValue(primaryStep.inputDraft, "workspace_path").trim();
    const constraints = splitConstraintList(
      stringInputValue(primaryStep.inputDraft, "constraints")
    );
    const maxStepsText = stringInputValue(primaryStep.inputDraft, "max_steps").trim();
    let maxSteps: number | null = null;
    if (maxStepsText) {
      const parsedMaxSteps = Number(maxStepsText);
      if (!Number.isInteger(parsedMaxSteps) || parsedMaxSteps <= 0) {
        throw new Error("Agent profile max steps must be a positive whole number.");
      }
      maxSteps = parsedMaxSteps;
    }
    const allowedCapabilityIds = Array.from(
      new Set(
        agentSteps
          .slice(1)
          .map((step) => step.capabilityId.trim())
          .filter(Boolean)
      )
    );
    const fallbackName = defaultGoal ? defaultGoal.slice(0, 96) : "Agent profile";
    const name =
      agentProfileName.trim() ||
      agentTitle.trim() ||
      fallbackName;
    const instructions =
      primaryStep.instruction.trim() || defaultStepInstruction(capabilityId, "agent");
    return {
      name,
      description: agentProfileDescription.trim() || null,
      agent_capability_id: capabilityId,
      instructions,
      default_goal: defaultGoal,
      default_workspace_path: workspacePath || null,
      default_constraints: constraints,
      default_max_steps: maxSteps,
      model_config: selectedAgentDefinition?.model_config ?? {},
      allowed_capability_ids: allowedCapabilityIds,
      memory_policy: selectedAgentDefinition?.memory_policy ?? {},
      guardrail_policy: selectedAgentDefinition?.guardrail_policy ?? {},
      workspace_policy: selectedAgentDefinition?.workspace_policy ?? {},
      user_id: agentUserId.trim() || workspaceUserId.trim() || null,
      metadata: {
        ...(selectedAgentDefinition?.metadata ?? {}),
        surface: "studio_workbench",
      },
    };
  };

  const handleNewAgentProfile = () => {
    setWorkbenchMode("agent");
    setAgentEditorMode("structured");
    setSelectedAgentDefinitionId("");
    setAgentProfileName("");
    setAgentProfileDescription("");
    setAgentTitle("Agent workbench run");
    setAgentGoal("");
    setAgentSteps([createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent")]);
    setAgentProfileError(null);
    setLaunchError(null);
    setWorkbenchBanner({
      tone: "info",
      message: "Started a new unsaved agent profile draft.",
    });
  };

  const handleSaveAgentProfile = async () => {
    if (!selectedAgentDefinitionId) {
      setAgentProfileError("Select a saved profile first, or use Save as to create one.");
      return;
    }
    setAgentProfileSaving(true);
    setAgentProfileError(null);
    try {
      const payload = buildAgentDefinitionPayload();
      const updated = await updateAgentDefinition(selectedAgentDefinitionId, payload);
      setAgentDefinitions((current) =>
        sortAgentDefinitions(
          current.map((definition) =>
            definition.id === updated.id ? updated : definition
          )
        )
      );
      setSelectedAgentDefinitionId(updated.id);
      setAgentProfileName(updated.name);
      setAgentProfileDescription(updated.description ?? "");
      setWorkbenchBanner({
        tone: "info",
        message: `Saved agent profile '${updated.name}'.`,
      });
    } catch (error) {
      setAgentProfileError(
        error instanceof Error ? error.message : "Failed to save agent profile."
      );
    } finally {
      setAgentProfileSaving(false);
    }
  };

  const handleSaveAgentProfileAs = async () => {
    setAgentProfileSaving(true);
    setAgentProfileError(null);
    try {
      const payload = buildAgentDefinitionPayload();
      const created = await createAgentDefinition(payload);
      setAgentDefinitions((current) =>
        sortAgentDefinitions([
          created,
          ...current.filter((definition) => definition.id !== created.id),
        ])
      );
      setSelectedAgentDefinitionId(created.id);
      setAgentProfileName(created.name);
      setAgentProfileDescription(created.description ?? "");
      setWorkbenchBanner({
        tone: "info",
        message: `Saved new agent profile '${created.name}'.`,
      });
    } catch (error) {
      setAgentProfileError(
        error instanceof Error ? error.message : "Failed to create agent profile."
      );
    } finally {
      setAgentProfileSaving(false);
    }
  };

  const handleDeleteAgentProfile = async () => {
    if (!selectedAgentDefinitionId) {
      return;
    }
    const definitionName = selectedAgentDefinition?.name || "this agent profile";
    if (!window.confirm(`Delete ${definitionName}?`)) {
      return;
    }
    setAgentProfileDeleting(true);
    setAgentProfileError(null);
    try {
      await deleteAgentDefinition(selectedAgentDefinitionId);
      setAgentDefinitions((current) =>
        current.filter((definition) => definition.id !== selectedAgentDefinitionId)
      );
      setSelectedAgentDefinitionId("");
      setAgentProfileName("");
      setAgentProfileDescription("");
      setWorkbenchBanner({
        tone: "info",
        message: "Deleted the agent profile.",
      });
    } catch (error) {
      setAgentProfileError(
        error instanceof Error ? error.message : "Failed to delete agent profile."
      );
    } finally {
      setAgentProfileDeleting(false);
    }
  };

  const applyCapabilityReplayDraft = (draft: ReplayableCapabilityDraft) => {
    const targetCapability = catalog.find((item) => item.id === draft.capabilityId) ?? null;
    const nextCapabilityState = capabilityEditorStateFromInputs(targetCapability, draft.inputs);
    setWorkbenchMode("capability");
    setSelectedCapabilityId(draft.capabilityId);
    setCapabilityTitle(draft.title || `Capability run: ${draft.capabilityId}`);
    setCapabilityGoal(draft.goal);
    setCapabilityUserId(draft.userId || workspaceUserId);
    setCapabilityContextJsonText(formatJson(draft.contextJson));
    setCapabilityRetryPolicyText(
      draft.retryPolicy && Object.keys(draft.retryPolicy).length > 0
        ? formatJson(draft.retryPolicy)
        : ""
    );
    setCapabilityInputDraft(nextCapabilityState.inputDraft);
    setCapabilityRawOverrideEnabled(nextCapabilityState.rawOverrideEnabled);
    setCapabilityRawOverrideText(nextCapabilityState.rawOverrideText);
    setLaunchError(null);
    setWorkbenchBanner({
      tone: nextCapabilityState.rawOverrideEnabled ? "warning" : "info",
      message: nextCapabilityState.rawOverrideEnabled
        ? `${draft.notice} Some inputs stayed in raw JSON because the current capability schema could not represent them as structured fields.`
        : draft.notice,
    });
  };

  const applyForkResult = () => {
    if (!forkResult) {
      return;
    }
    if (forkResult.mode === "capability") {
      applyCapabilityReplayDraft(forkResult.draft);
      return;
    }
    setWorkbenchMode("agent");
    setSelectedAgentDefinitionId("");
    setAgentProfileName("");
    setAgentProfileDescription("");
    setAgentProfileError(null);
    setAgentTitle(forkResult.draft.title || "Agent workbench run");
    setAgentGoal(forkResult.draft.goal);
    setAgentUserId(forkResult.draft.userId || workspaceUserId);
    setAgentContextJsonText(formatJson(forkResult.draft.contextJson));
    setLaunchError(null);
    if (forkResult.mode === "agent_structured") {
      setAgentEditorMode("structured");
      setAgentSteps(
        forkResult.draft.steps.length > 0
          ? forkResult.draft.steps.map((step, index) => ({
              ...(() => {
                const stepCapability = catalog.find((item) => item.id === step.capabilityId) ?? null;
                const nextInputState = capabilityEditorStateFromInputs(
                  stepCapability,
                  step.inputBindings
                );
                return {
                  localId: `forked-step-${step.stepId}-${Math.random().toString(36).slice(2, 8)}`,
                  stepId: step.stepId,
                  name: step.name,
                  description: step.description,
                  instruction: step.instruction,
                  capabilityId: step.capabilityId,
                  dependsOnText: step.dependsOn.join(", "),
                  inputDraft: nextInputState.inputDraft,
                  rawInputOverrideEnabled:
                    index === 0 ? false : nextInputState.rawOverrideEnabled,
                  inputJsonText: nextInputState.rawOverrideText,
                  retryPolicyText:
                    step.retryPolicy && Object.keys(step.retryPolicy).length > 0
                      ? formatJson(step.retryPolicy)
                      : "",
                };
              })(),
            }))
          : [createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent")]
      );
      setWorkbenchBanner({
        tone: "info",
        message: forkResult.draft.notice,
      });
      return;
    }
    setAgentEditorMode("raw");
    setAgentRawRunSpecText(formatJson(forkResult.draft.runSpec));
    setWorkbenchBanner({
      tone: "warning",
      message: forkResult.draft.notice,
    });
  };

  const handleReplayStep = (stepId: string) => {
    const replayResult = replayResultsByStep.get(stepId);
    if (!replayResult || !replayResult.replayable) {
      return;
    }
    applyCapabilityReplayDraft(replayResult.draft);
  };

  const handlePromoteWorkflowDraft = () => {
    if (!workflowPromotionResult?.promotable || !onPromoteWorkflowDraft) {
      return;
    }
    onPromoteWorkflowDraft(workflowPromotionResult.draft);
  };

  useEffect(() => {
    if (!active || !activeRunId) {
      return;
    }
    let cancelled = false;
    const refreshDebugger = async () => {
      setDebuggerLoading(true);
      setDebuggerError(null);
      try {
        const response = await fetchRunDebugger(activeRunId);
        if (!cancelled) {
          setDebuggerData(response);
        }
      } catch (error) {
        if (!cancelled) {
          setDebuggerError(
            error instanceof Error ? error.message : "Failed to load workbench run debugger."
          );
        }
      } finally {
        if (!cancelled) {
          setDebuggerLoading(false);
        }
      }
    };
    void refreshDebugger();
    if (currentRunStatus && TERMINAL_RUN_STATUSES.has(currentRunStatus)) {
      return () => {
        cancelled = true;
      };
    }
    const intervalId = window.setInterval(() => {
      void refreshDebugger();
    }, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [active, activeRunId, currentRunStatus]);

  const handleCapabilityInsert = (item: CapabilityItem) => {
    setWorkbenchMode("capability");
    setSelectedCapabilityId(item.id);
    setCapabilityTitle(`Capability run: ${item.id}`);
    setWorkbenchBanner(null);
  };

  const handleAgentInsert = (item: CapabilityItem) => {
    setWorkbenchMode("agent");
    setAgentEditorMode("structured");
    setAgentSteps((current) => {
      if (isAgenticCapability(item)) {
        const [first, ...rest] =
          current.length > 0
            ? current
            : [createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent")];
        return [
          {
            ...first,
            stepId: first.stepId || "agent",
            name: first.name || defaultStepName(item.id, "agent"),
            description: first.description || defaultStepDescription(item.id, "agent"),
            instruction: first.instruction || defaultStepInstruction(item.id, "agent"),
            capabilityId: item.id,
            inputDraft: {
              ...defaultAgentInputDraft(item.id),
              ...first.inputDraft,
            },
          },
          ...rest,
        ];
      }
      return [...current, createAgentStepDraft(item.id)];
    });
    setWorkbenchBanner(null);
  };

  const updateAgentStepCapability = (localId: string, capabilityId: string) => {
    const targetCapability = catalog.find((item) => item.id === capabilityId) ?? null;
    setAgentSteps((current) =>
      current.map((step) => {
        if (step.localId !== localId) {
          return step;
        }
        if (!targetCapability) {
          return {
            ...step,
            capabilityId,
            name: step.name || capabilityId,
          };
        }
        const parsedInputs = parseJsonObject(step.inputJsonText, "Step inputs");
        const nextInputState = capabilityEditorStateFromInputs(
          targetCapability,
          parsedInputs.value ?? {}
        );
        return {
          ...step,
          capabilityId,
          name: step.name || capabilityId,
          inputDraft: nextInputState.inputDraft,
          rawInputOverrideEnabled: nextInputState.rawOverrideEnabled,
          inputJsonText: nextInputState.rawOverrideText,
        };
      })
    );
  };

  const updateAgentStep = (
    localId: string,
    updater: (current: AgentStepDraft) => AgentStepDraft
  ) => {
    setAgentSteps((current) => current.map((step) => (step.localId === localId ? updater(step) : step)));
  };

  const updatePrimaryAgentStep = (updater: (current: AgentStepDraft) => AgentStepDraft) => {
    setAgentSteps((current) => {
      const [first, ...rest] =
        current.length > 0
          ? current
          : [createAgentStepDraft(DEFAULT_AGENT_CAPABILITY_ID, "agent")];
      return [updater(first), ...rest];
    });
  };

  const updatePrimaryAgentCapability = (capabilityId: string) => {
    updatePrimaryAgentStep((current) => ({
      ...current,
      capabilityId,
      stepId: current.stepId || "agent",
      name: current.name || defaultStepName(capabilityId, "agent"),
      description: current.description || defaultStepDescription(capabilityId, "agent"),
      instruction: current.instruction || defaultStepInstruction(capabilityId, "agent"),
      inputDraft: {
        ...defaultAgentInputDraft(capabilityId),
        ...current.inputDraft,
      },
      rawInputOverrideEnabled: false,
    }));
  };

  const updatePrimaryAgentInput = (key: string, value: string) => {
    if (key === "goal") {
      setAgentGoal(value);
    }
    updatePrimaryAgentStep((current) => ({
      ...current,
      inputDraft: {
        ...current.inputDraft,
        [key]: value,
      },
      rawInputOverrideEnabled: false,
    }));
  };

  const launchCurrentWorkbenchRun = async () => {
    setLaunchLoading(true);
    setLaunchError(null);
    setDebuggerError(null);
    setDebuggerData(null);
    setWorkbenchBanner(null);

    try {
      let response: WorkbenchRunLaunchResponse;
      if (workbenchMode === "capability") {
        if (!selectedCapability) {
          throw new Error("Choose a capability before launching a workbench run.");
        }
        if (capabilityContextJson.error) {
          throw new Error(capabilityContextJson.error);
        }
        if (capabilityRetryPolicy.error) {
          throw new Error(capabilityRetryPolicy.error);
        }
        if (capabilityLaunchInputs.error || !capabilityLaunchInputs.value) {
          throw new Error(capabilityLaunchInputs.error || "Capability inputs are invalid.");
        }
        response = await launchCapabilityRun({
          title: capabilityTitle.trim(),
          goal: capabilityGoal.trim(),
          user_id: capabilityUserId.trim() || null,
          context_json: capabilityContextJson.value ?? {},
          capability_id: selectedCapability.id,
          inputs: capabilityLaunchInputs.value,
          retry_policy:
            capabilityRetryPolicy.value && Object.keys(capabilityRetryPolicy.value).length > 0
              ? capabilityRetryPolicy.value
              : null,
        });
      } else {
        if (agentContextJson.error) {
          throw new Error(agentContextJson.error);
        }
        if (agentRunSpecPreview.error || !agentRunSpecPreview.value) {
          throw new Error(agentRunSpecPreview.error || "Agent RunSpec is invalid.");
        }
        const primaryAgentGoal = stringInputValue(primaryAgentStep?.inputDraft ?? {}, "goal").trim();
        response = await launchAgentRun({
          title: agentTitle.trim(),
          goal: agentGoal.trim() || primaryAgentGoal,
          user_id: agentUserId.trim() || null,
          context_json: agentContextJson.value ?? {},
          run_spec: agentRunSpecPreview.value,
          ...(selectedAgentDefinitionId
            ? { agent_definition_id: selectedAgentDefinitionId }
            : {}),
        });
      }

      setLaunchResponse(response);
      setActiveRunId(response.run.id);
      try {
        const initialDebugger = await fetchRunDebugger(response.run.id);
        setDebuggerData(initialDebugger);
      } catch (error) {
        setDebuggerError(
          error instanceof Error ? error.message : "Failed to load workbench run debugger."
        );
      }
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Workbench launch failed.");
    } finally {
      setLaunchLoading(false);
    }
  };

  const artifacts = useMemo(() => collectArtifacts(debuggerData), [debuggerData]);
  const activeSurface: StudioSurface = "workbench";
  const launchWorkbenchModeLabel =
    typeof launchResponse?.run?.metadata?.workbench_mode === "string"
      ? launchResponse.run.metadata.workbench_mode
      : workbenchMode;

  return (
    <section className={active ? "block" : "hidden"} aria-hidden={!active}>
      <div className="relative">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-100/72">
              Studio Surface
            </div>
            <h2 className="mt-1 flex items-center gap-3 text-[30px] font-semibold tracking-[-0.03em] text-white">
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-sky-300/22 bg-sky-400/14 text-sky-100">
                <StudioWorkbenchIcon kind="run" className="h-5 w-5" />
              </span>
              Agent + Capability Workbench
            </h2>
            <p className="mt-1 max-w-3xl text-[13px] leading-5 text-slate-200/74">
              Launch ephemeral capability and agent runs through the canonical runtime, then inspect
              the resulting debugger state without leaving <span className="font-semibold">{activeSurface}</span>.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]">
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              catalog {catalog.length}
            </span>
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              mode {workbenchMode}
            </span>
            {activeRunId ? (
              <span className="rounded-full border border-sky-300/28 bg-sky-400/14 px-3 py-1 text-sky-50">
                run {activeRunId.slice(0, 8)}
              </span>
            ) : null}
          </div>
        </div>

        {workbenchBanner ? (
          <div
            className={`mt-4 rounded-[24px] border px-4 py-3 text-sm ${
              workbenchBanner.tone === "warning"
                ? "border-amber-300/18 bg-amber-400/12 text-amber-50"
                : "border-sky-300/15 bg-sky-400/10 text-sky-50"
            }`}
          >
            {workbenchBanner.message}
          </div>
        ) : null}

        <datalist id="studio-capability-id-options">
          {catalog.map((item) => (
            <option key={item.id} value={item.id} />
          ))}
        </datalist>
        <datalist id="studio-agent-capability-options">
          {agentCapabilities.map((item) => (
            <option key={item.id} value={item.id} />
          ))}
        </datalist>

        <div className="mt-4 overflow-hidden rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(75,92,109,0.58),rgba(45,57,71,0.72))] p-4 shadow-[0_22px_56px_rgba(15,23,42,0.16)]">
          <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)_340px]">
            <SurfacePanel
              title="Catalog"
              subtitle="Search live capabilities, filter the catalog, and insert into either sandbox."
            >
              <div className="space-y-3">
                <input
                  value={catalogQuery}
                  onChange={(event) => setCatalogQuery(event.target.value)}
                  placeholder="Search capabilities"
                  className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                />
                <div className="grid gap-2">
                  <select
                    value={groupFilter}
                    onChange={(event) => setGroupFilter(event.target.value)}
                    className="rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-xs text-slate-100 outline-none"
                  >
                    <option value="all">All groups</option>
                    {groupOptions.map((group) => (
                      <option key={group} value={group}>
                        {group}
                      </option>
                    ))}
                  </select>
                  <select
                    value={riskFilter}
                    onChange={(event) => setRiskFilter(event.target.value)}
                    className="rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-xs text-slate-100 outline-none"
                  >
                    <option value="all">All risk tiers</option>
                    {riskOptions.map((risk) => (
                      <option key={risk} value={risk}>
                        {risk}
                      </option>
                    ))}
                  </select>
                  <select
                    value={idempotencyFilter}
                    onChange={(event) => setIdempotencyFilter(event.target.value)}
                    className="rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-xs text-slate-100 outline-none"
                  >
                    <option value="all">All idempotency</option>
                    {idempotencyOptions.map((mode) => (
                      <option key={mode} value={mode}>
                        {mode}
                      </option>
                    ))}
                  </select>
                </div>
                {catalogLoading ? (
                  <div className="rounded-2xl border border-white/8 bg-black/20 px-3 py-3 text-xs text-slate-300/74">
                    Loading capability catalog...
                  </div>
                ) : null}
                {catalogError ? (
                  <div className="rounded-2xl border border-rose-300/18 bg-rose-400/12 px-3 py-3 text-xs text-rose-100">
                    {catalogError}
                  </div>
                ) : null}
                {catalogSearchLoading ? (
                  <div className="text-[11px] text-slate-300/74">Searching capability catalog...</div>
                ) : null}
                {catalogSearchError ? (
                  <div className="text-[11px] text-rose-100">{catalogSearchError}</div>
                ) : null}
                <div className="max-h-[460px] space-y-2 overflow-auto pr-1">
                  {filteredCatalog.map((item) => {
                    const searchHit = catalogSearchItems.find((candidate) => candidate.id === item.id);
                    return (
                      <div
                        key={item.id}
                        className={`rounded-2xl border px-3 py-3 ${
                          item.id === selectedCapabilityId
                            ? "border-sky-300/28 bg-sky-400/12"
                            : "border-white/8 bg-black/18"
                        }`}
                      >
                        <button
                          type="button"
                          className="w-full text-left"
                          onClick={() => setSelectedCapabilityId(item.id)}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-sm font-semibold text-white">{item.id}</div>
                            <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] text-slate-200">
                              {item.group || "ungrouped"}
                            </span>
                          </div>
                          <div className="mt-1 text-xs leading-5 text-slate-300/78">
                            {item.description}
                          </div>
                        </button>
                        <div className="mt-3 flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.14em] text-slate-300/76">
                          <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                            {item.risk_tier}
                          </span>
                          <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                            {item.idempotency}
                          </span>
                          {searchHit ? (
                            <span className="rounded-full border border-sky-300/22 bg-sky-400/12 px-2 py-1 text-sky-100">
                              {searchHit.source}
                            </span>
                          ) : null}
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            type="button"
                            className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                            onClick={() => handleCapabilityInsert(item)}
                          >
                            capability
                          </button>
                          <button
                            type="button"
                            className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                            onClick={() => handleAgentInsert(item)}
                          >
                            {isAgenticCapability(item) ? "agent" : "agent step"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                  {!catalogLoading && filteredCatalog.length === 0 ? (
                    <div className="rounded-2xl border border-white/8 bg-black/18 px-3 py-4 text-xs text-slate-300/74">
                      No capabilities matched the current search and filters.
                    </div>
                  ) : null}
                </div>
              </div>
            </SurfacePanel>

            <SurfacePanel
              title="Workbench Editor"
              subtitle="Structured builders are the default. Raw JSON editors stay available as advanced overrides."
            >
              <div className="flex flex-wrap items-center gap-2">
                {(["capability", "agent"] as WorkbenchMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] transition ${
                      workbenchMode === mode
                        ? "border-sky-300/35 bg-sky-400/18 text-sky-50"
                        : "border-white/10 bg-white/[0.05] text-slate-100 hover:border-white/16 hover:bg-white/[0.08]"
                    }`}
                    onClick={() => {
                      setWorkbenchMode(mode);
                      setWorkbenchBanner(null);
                    }}
                  >
                    {mode === "capability" ? "Capability Sandbox" : "Agent Sandbox"}
                  </button>
                ))}
              </div>

              {workbenchMode === "capability" ? (
                <div className="mt-4 space-y-4">
                  <div className="grid gap-3 lg:grid-cols-2">
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Capability Id
                      </span>
                      <input
                        list="studio-capability-id-options"
                        value={selectedCapabilityId}
                        onChange={(event) => setSelectedCapabilityId(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        User Id
                      </span>
                      <input
                        value={capabilityUserId}
                        onChange={(event) => setCapabilityUserId(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Title
                      </span>
                      <input
                        value={capabilityTitle}
                        onChange={(event) => setCapabilityTitle(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Goal
                      </span>
                      <input
                        value={capabilityGoal}
                        onChange={(event) => setCapabilityGoal(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                  </div>

                  <label className="block text-xs text-slate-200">
                    <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                      Context JSON
                    </span>
                    <textarea
                      rows={5}
                      value={capabilityContextJsonText}
                      onChange={(event) => setCapabilityContextJsonText(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                    />
                  </label>

                  <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-300/74">
                          Generated input form
                        </div>
                        <div className="mt-1 text-xs text-slate-300/70">
                          Required fields come from the capability input schema.
                        </div>
                      </div>
                      <label className="inline-flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-300/76">
                        <input
                          type="checkbox"
                          checked={capabilityRawOverrideEnabled}
                          onChange={(event) => setCapabilityRawOverrideEnabled(event.target.checked)}
                        />
                        raw override
                      </label>
                    </div>
                    <div className="mt-3 space-y-3">
                      {capabilitySchemaProperties.map(([fieldName, schema]) => {
                        const fieldType = normalizeSchemaType(schema);
                        const required = selectedCapability?.required_inputs?.includes(fieldName) ?? false;
                        const rawFieldValue = capabilityInputDraft[fieldName];
                        const fieldValue = typeof rawFieldValue === "string" ? rawFieldValue : "";
                        if (fieldType === "boolean") {
                          return (
                            <label
                              key={fieldName}
                              className="flex items-center justify-between gap-3 rounded-2xl border border-white/8 bg-slate-950/38 px-3 py-2 text-sm text-slate-100"
                            >
                              <span>
                                {fieldName}
                                {required ? <span className="ml-2 text-[10px] uppercase text-sky-100/70">required</span> : null}
                              </span>
                              <input
                                type="checkbox"
                                checked={capabilityInputDraft[fieldName] === true}
                                onChange={(event) =>
                                  setCapabilityInputDraft((current) => ({
                                    ...current,
                                    [fieldName]: event.target.checked,
                                  }))
                                }
                              />
                            </label>
                          );
                        }
                        const multiLine = fieldType === "object" || fieldType === "array";
                        return (
                          <label key={fieldName} className="block text-xs text-slate-200">
                            <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                              {fieldName}
                              {required ? " *" : ""}
                            </span>
                            {multiLine ? (
                              <textarea
                                rows={4}
                                value={fieldValue}
                                onChange={(event) =>
                                  setCapabilityInputDraft((current) => ({
                                    ...current,
                                    [fieldName]: event.target.value,
                                  }))
                                }
                                className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                              />
                            ) : (
                              <input
                                value={fieldValue}
                                onChange={(event) =>
                                  setCapabilityInputDraft((current) => ({
                                    ...current,
                                    [fieldName]: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            )}
                          </label>
                        );
                      })}
                      {capabilitySchemaProperties.length === 0 ? (
                        <div className="rounded-2xl border border-white/8 bg-slate-950/38 px-3 py-4 text-xs text-slate-300/74">
                          This capability does not expose structured schema fields. Use the raw input override
                          for advanced payloads.
                        </div>
                      ) : null}
                    </div>
                  </div>

                  {capabilityRawOverrideEnabled ? (
                    <label className="block text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Raw input override JSON
                      </span>
                      <textarea
                        rows={6}
                        value={capabilityRawOverrideText}
                        onChange={(event) => setCapabilityRawOverrideText(event.target.value)}
                        className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                  ) : null}

                  <label className="block text-xs text-slate-200">
                    <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                      Advanced retry / policy override
                    </span>
                    <textarea
                      rows={4}
                      value={capabilityRetryPolicyText}
                      onChange={(event) => setCapabilityRetryPolicyText(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                    />
                  </label>
                </div>
              ) : (
                <div className="mt-4 space-y-4">
                  <div className="rounded-2xl border border-sky-300/16 bg-sky-400/10 p-3">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-white">Agent Profile</div>
                        <div className="mt-1 text-xs leading-5 text-slate-300/78">
                          {selectedAgentDefinition
                            ? `Saved profile updated ${new Date(
                                selectedAgentDefinition.updated_at
                              ).toLocaleString()}`
                            : "Unsaved draft"}
                        </div>
                      </div>
                      <span className="rounded-full border border-sky-300/22 bg-sky-400/12 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-sky-100">
                        {agentDefinitionsLoading ? "loading" : `${agentDefinitions.length} saved`}
                      </span>
                    </div>
                    <div className="mt-3 grid gap-3 lg:grid-cols-2">
                      <label className="text-xs text-slate-200">
                        <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                          Profile
                        </span>
                        <select
                          value={selectedAgentDefinitionId}
                          onChange={(event) => {
                            const nextId = event.target.value;
                            if (!nextId) {
                              setSelectedAgentDefinitionId("");
                              setAgentProfileName("");
                              setAgentProfileDescription("");
                              setAgentProfileError(null);
                              return;
                            }
                            const definition =
                              agentDefinitions.find((item) => item.id === nextId) ?? null;
                            if (definition) {
                              applyAgentDefinitionDraft(definition);
                            }
                          }}
                          className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                        >
                          <option value="">Unsaved draft</option>
                          {agentDefinitions.map((definition) => (
                            <option key={definition.id} value={definition.id}>
                              {definition.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="text-xs text-slate-200">
                        <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                          Profile name
                        </span>
                        <input
                          value={agentProfileName}
                          onChange={(event) => setAgentProfileName(event.target.value)}
                          placeholder="Agent profile"
                          className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                        />
                      </label>
                      <label className="text-xs text-slate-200 lg:col-span-2">
                        <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                          Description
                        </span>
                        <input
                          value={agentProfileDescription}
                          onChange={(event) => setAgentProfileDescription(event.target.value)}
                          className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                        />
                      </label>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                        onClick={handleNewAgentProfile}
                      >
                        New
                      </button>
                      <button
                        type="button"
                        className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => void handleSaveAgentProfile()}
                        disabled={!selectedAgentDefinitionId || agentProfileSaving}
                      >
                        {agentProfileSaving ? "Saving..." : "Save"}
                      </button>
                      <button
                        type="button"
                        className="rounded-xl border border-sky-300/26 bg-sky-400/16 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-50 transition hover:border-sky-300/36 disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => void handleSaveAgentProfileAs()}
                        disabled={agentProfileSaving}
                      >
                        {agentProfileSaving ? "Saving..." : "Save as"}
                      </button>
                      <button
                        type="button"
                        className="rounded-xl border border-rose-300/18 bg-rose-400/12 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-rose-100 transition hover:border-rose-300/28 disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => void handleDeleteAgentProfile()}
                        disabled={!selectedAgentDefinitionId || agentProfileDeleting}
                      >
                        {agentProfileDeleting ? "Deleting..." : "Delete"}
                      </button>
                    </div>
                    {agentDefinitionsError ? (
                      <div className="mt-3 rounded-2xl border border-rose-300/18 bg-rose-400/12 px-3 py-3 text-xs text-rose-100">
                        {agentDefinitionsError}
                      </div>
                    ) : null}
                    {agentProfileError ? (
                      <div className="mt-3 rounded-2xl border border-rose-300/18 bg-rose-400/12 px-3 py-3 text-xs text-rose-100">
                        {agentProfileError}
                      </div>
                    ) : null}
                  </div>

                  <div className="grid gap-3 lg:grid-cols-2">
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Title
                      </span>
                      <input
                        value={agentTitle}
                        onChange={(event) => setAgentTitle(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                    <label className="text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        User Id
                      </span>
                      <input
                        value={agentUserId}
                        onChange={(event) => setAgentUserId(event.target.value)}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                    <label className="text-xs text-slate-200 lg:col-span-2">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Run goal
                      </span>
                      <input
                        value={agentGoal}
                        onChange={(event) => {
                          const nextGoal = event.target.value;
                          if (agentEditorMode === "structured") {
                            updatePrimaryAgentInput("goal", nextGoal);
                          } else {
                            setAgentGoal(nextGoal);
                          }
                        }}
                        className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                  </div>

                  <label className="block text-xs text-slate-200">
                    <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                      Context JSON
                    </span>
                    <textarea
                      rows={5}
                      value={agentContextJsonText}
                      onChange={(event) => setAgentContextJsonText(event.target.value)}
                      className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                    />
                  </label>

                  <div className="flex flex-wrap items-center gap-2">
                    {(["structured", "raw"] as AgentEditorMode[]).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] transition ${
                          agentEditorMode === mode
                            ? "border-sky-300/35 bg-sky-400/18 text-sky-50"
                            : "border-white/10 bg-white/[0.05] text-slate-100 hover:border-white/16 hover:bg-white/[0.08]"
                        }`}
                        onClick={() => {
                          setAgentEditorMode(mode);
                          setWorkbenchBanner(null);
                        }}
                      >
                        {mode === "structured" ? "Structured Builder" : "Raw RunSpec"}
                      </button>
                    ))}
                  </div>

                  {agentEditorMode === "structured" ? (
                    <div className="space-y-3">
                      {primaryAgentStep ? (
                        <div className="rounded-2xl border border-sky-300/18 bg-sky-400/10 p-3">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-white">Agent</div>
                              {primaryAgentCapability ? (
                                <div className="mt-1 text-xs leading-5 text-slate-300/78">
                                  {primaryAgentCapability.description}
                                </div>
                              ) : null}
                            </div>
                            <span className="rounded-full border border-sky-300/22 bg-sky-400/12 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-sky-100">
                              primary step
                            </span>
                          </div>
                          <div className="mt-3 grid gap-3 lg:grid-cols-2">
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Agent capability
                              </span>
                              <input
                                list="studio-agent-capability-options"
                                value={primaryAgentStep.capabilityId}
                                onChange={(event) =>
                                  updatePrimaryAgentCapability(event.target.value)
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Step id
                              </span>
                              <input
                                value={primaryAgentStep.stepId}
                                onChange={(event) =>
                                  updatePrimaryAgentStep((current) => ({
                                    ...current,
                                    stepId: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200 lg:col-span-2">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Agent task
                              </span>
                              <textarea
                                rows={3}
                                value={
                                  stringInputValue(primaryAgentStep.inputDraft, "goal") ||
                                  agentGoal
                                }
                                onChange={(event) =>
                                  updatePrimaryAgentInput("goal", event.target.value)
                                }
                                className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Workspace path
                              </span>
                              <input
                                value={stringInputValue(
                                  primaryAgentStep.inputDraft,
                                  "workspace_path"
                                )}
                                onChange={(event) =>
                                  updatePrimaryAgentInput("workspace_path", event.target.value)
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Max steps
                              </span>
                              <input
                                type="number"
                                min={1}
                                max={12}
                                value={stringInputValue(primaryAgentStep.inputDraft, "max_steps")}
                                onChange={(event) =>
                                  updatePrimaryAgentInput("max_steps", event.target.value)
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200 lg:col-span-2">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Constraints
                              </span>
                              <textarea
                                rows={3}
                                value={stringInputValue(
                                  primaryAgentStep.inputDraft,
                                  "constraints"
                                )}
                                onChange={(event) =>
                                  updatePrimaryAgentInput("constraints", event.target.value)
                                }
                                className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200 lg:col-span-2">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Instruction
                              </span>
                              <input
                                value={primaryAgentStep.instruction}
                                onChange={(event) =>
                                  updatePrimaryAgentStep((current) => ({
                                    ...current,
                                    instruction: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                          </div>
                        </div>
                      ) : null}
                      {agentSteps.slice(1).map((step, index) => {
                        const stepCapability =
                          catalog.find((item) => item.id === step.capabilityId) ?? null;
                        const stepSchemaProperties = getCapabilitySchemaProperties(stepCapability);
                        return (
                          <div
                            key={step.localId}
                            className="rounded-2xl border border-white/8 bg-black/18 p-3"
                          >
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-sm font-semibold text-white">Step {index + 2}</div>
                            <button
                              type="button"
                              className="rounded-xl border border-rose-300/18 bg-rose-400/12 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-rose-100 transition hover:border-rose-300/28"
                              onClick={() =>
                                setAgentSteps((current) =>
                                  current.length > 1
                                    ? current.filter((item) => item.localId !== step.localId)
                                    : current
                                )
                              }
                            >
                              remove
                            </button>
                          </div>
                          <div className="mt-3 grid gap-3 lg:grid-cols-2">
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Step id
                              </span>
                              <input
                                value={step.stepId}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    stepId: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Capability id
                              </span>
                              <input
                                list="studio-capability-id-options"
                                value={step.capabilityId}
                                onChange={(event) =>
                                  updateAgentStepCapability(step.localId, event.target.value)
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Name
                              </span>
                              <input
                                value={step.name}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    name: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Depends on
                              </span>
                              <input
                                value={step.dependsOnText}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    dependsOnText: event.target.value,
                                  }))
                                }
                                placeholder="comma separated step ids"
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200 lg:col-span-2">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Description
                              </span>
                              <input
                                value={step.description}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    description: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <label className="text-xs text-slate-200 lg:col-span-2">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Instruction
                              </span>
                              <input
                                value={step.instruction}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    instruction: event.target.value,
                                  }))
                                }
                                className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                            <div className="rounded-2xl border border-white/8 bg-slate-950/30 p-3 text-xs text-slate-200 lg:col-span-2">
                              <div className="flex items-center justify-between gap-3">
                                <div>
                                  <div className="font-semibold uppercase tracking-[0.16em] text-slate-300/74">
                                    Inputs
                                  </div>
                                  <div className="mt-1 text-slate-300/70">
                                    Fill required fields from the selected capability schema.
                                  </div>
                                </div>
                                <label className="inline-flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-300/76">
                                  <input
                                    type="checkbox"
                                    checked={step.rawInputOverrideEnabled}
                                    onChange={(event) =>
                                      updateAgentStep(step.localId, (current) => ({
                                        ...current,
                                        rawInputOverrideEnabled: event.target.checked,
                                      }))
                                    }
                                  />
                                  raw override
                                </label>
                              </div>
                              {!step.rawInputOverrideEnabled && stepSchemaProperties.length > 0 ? (
                                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                                  {stepSchemaProperties.map(([fieldName, schema]) => {
                                    const fieldType = normalizeSchemaType(schema);
                                    const required =
                                      stepCapability?.required_inputs?.includes(fieldName) ?? false;
                                    const rawFieldValue = step.inputDraft[fieldName];
                                    const fieldValue =
                                      typeof rawFieldValue === "string" ? rawFieldValue : "";
                                    if (fieldType === "boolean") {
                                      return (
                                        <label
                                          key={fieldName}
                                          className="flex items-center justify-between gap-3 rounded-2xl border border-white/8 bg-slate-950/38 px-3 py-2 text-sm text-slate-100"
                                        >
                                          <span>
                                            {fieldName}
                                            {required ? (
                                              <span className="ml-2 text-[10px] uppercase text-sky-100/70">
                                                required
                                              </span>
                                            ) : null}
                                          </span>
                                          <input
                                            type="checkbox"
                                            checked={step.inputDraft[fieldName] === true}
                                            onChange={(event) =>
                                              updateAgentStep(step.localId, (current) => ({
                                                ...current,
                                                inputDraft: {
                                                  ...current.inputDraft,
                                                  [fieldName]: event.target.checked,
                                                },
                                              }))
                                            }
                                          />
                                        </label>
                                      );
                                    }
                                    const multiLine = fieldType === "object" || fieldType === "array";
                                    return (
                                      <label
                                        key={fieldName}
                                        className={`block text-xs text-slate-200 ${
                                          multiLine ? "lg:col-span-2" : ""
                                        }`.trim()}
                                      >
                                        <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                          {fieldName}
                                          {required ? " *" : ""}
                                        </span>
                                        {multiLine ? (
                                          <textarea
                                            rows={4}
                                            value={fieldValue}
                                            onChange={(event) =>
                                              updateAgentStep(step.localId, (current) => ({
                                                ...current,
                                                inputDraft: {
                                                  ...current.inputDraft,
                                                  [fieldName]: event.target.value,
                                                },
                                              }))
                                            }
                                            className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                                          />
                                        ) : (
                                          <input
                                            value={fieldValue}
                                            onChange={(event) =>
                                              updateAgentStep(step.localId, (current) => ({
                                                ...current,
                                                inputDraft: {
                                                  ...current.inputDraft,
                                                  [fieldName]: event.target.value,
                                                },
                                              }))
                                            }
                                            className="w-full rounded-xl border border-white/10 bg-slate-950/45 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                                          />
                                        )}
                                      </label>
                                    );
                                  })}
                                </div>
                              ) : null}
                              {!step.rawInputOverrideEnabled && stepSchemaProperties.length === 0 ? (
                                <div className="mt-3 rounded-2xl border border-white/8 bg-slate-950/38 px-3 py-4 text-xs text-slate-300/74">
                                  Select a catalog capability with an input schema to show generated fields.
                                </div>
                              ) : null}
                              {step.rawInputOverrideEnabled ? (
                                <label className="mt-3 block text-xs text-slate-200">
                                  <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                    Raw input override JSON
                                  </span>
                                  <textarea
                                    rows={5}
                                    value={step.inputJsonText}
                                    onChange={(event) =>
                                      updateAgentStep(step.localId, (current) => ({
                                        ...current,
                                        inputJsonText: event.target.value,
                                      }))
                                    }
                                    className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                                  />
                                </label>
                              ) : null}
                            </div>
                            <label className="text-xs text-slate-200">
                              <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                                Retry policy JSON
                              </span>
                              <textarea
                                rows={5}
                                value={step.retryPolicyText}
                                onChange={(event) =>
                                  updateAgentStep(step.localId, (current) => ({
                                    ...current,
                                    retryPolicyText: event.target.value,
                                  }))
                                }
                                className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                              />
                            </label>
                          </div>
                        </div>
                      );
                    })}
                      <button
                        type="button"
                        className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                        onClick={() => setAgentSteps((current) => [...current, createAgentStepDraft()])}
                      >
                        add step
                      </button>
                    </div>
                  ) : (
                    <label className="block text-xs text-slate-200">
                      <span className="mb-1 block uppercase tracking-[0.16em] text-slate-300/74">
                        Raw RunSpec JSON
                      </span>
                      <textarea
                        rows={18}
                        value={agentRawRunSpecText}
                        onChange={(event) => setAgentRawRunSpecText(event.target.value)}
                        className="w-full rounded-2xl border border-white/10 bg-slate-950/45 px-3 py-3 font-mono text-[12px] text-white outline-none transition focus:border-sky-300/35"
                      />
                    </label>
                  )}
                </div>
              )}

              {(workbenchMode === "capability"
                ? capabilityLaunchInputs.error || capabilityContextJson.error || capabilityRetryPolicy.error
                : agentContextJson.error || agentRunSpecPreview.error) ? (
                <div className="mt-4 rounded-2xl border border-rose-300/18 bg-rose-400/12 px-3 py-3 text-xs text-rose-100">
                  {workbenchMode === "capability"
                    ? capabilityLaunchInputs.error || capabilityContextJson.error || capabilityRetryPolicy.error
                    : agentContextJson.error || agentRunSpecPreview.error}
                </div>
              ) : null}

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-xl border border-sky-300/26 bg-sky-400/16 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-50 transition hover:border-sky-300/36 disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={() => void launchCurrentWorkbenchRun()}
                  disabled={launchLoading}
                >
                  <StudioWorkbenchIcon kind="run" className="h-4 w-4" />
                  {launchLoading ? "Launching..." : "Launch Run"}
                </button>
                {launchError ? <div className="text-xs text-rose-100">{launchError}</div> : null}
              </div>
            </SurfacePanel>

            <SurfacePanel
              title="Preview"
              subtitle="Inspect the normalized run payload before launch, plus the initial execution-request shape."
            >
              <div className="space-y-3">
                {workbenchMode === "capability" ? (
                  <>
                    <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                      <div className="text-sm font-semibold text-white">
                        {selectedCapability?.id || "No capability selected"}
                      </div>
                      <div className="mt-1 text-xs leading-5 text-slate-300/76">
                        {selectedCapability?.description || "Pick a capability from the catalog to preview its schema."}
                      </div>
                      {selectedCapability ? (
                        <div className="mt-3 flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.14em] text-slate-300/76">
                          <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                            {selectedCapability.risk_tier}
                          </span>
                          <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                            {selectedCapability.idempotency}
                          </span>
                          {(selectedCapability.adapters || []).map((adapter) => (
                            <span
                              key={`${adapter.server_id}:${adapter.tool_name}`}
                              className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1"
                            >
                              {adapter.tool_name}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                    <JsonPreview
                      title="Capability Input Schema"
                      value={selectedCapability?.input_schema}
                      emptyLabel="No input schema is available for the selected capability."
                    />
                    <JsonPreview
                      title="Capability Output Schema"
                      value={selectedCapability?.output_schema}
                      emptyLabel="No output schema is available for the selected capability."
                    />
                  </>
                ) : null}

                <JsonPreview
                  title="RunSpec Preview"
                  value={workbenchMode === "capability" ? capabilityRunSpecPreview : agentRunSpecPreview.value}
                  emptyLabel="The current editor state does not yet produce a valid run specification."
                />
                <JsonPreview
                  title="Predicted First ExecutionRequest"
                  value={predictedExecutionRequestPreview}
                  emptyLabel="Execution request preview will appear once the first step is valid."
                />
                <JsonPreview
                  title="Retry / Policy Snapshot"
                  value={
                    workbenchMode === "capability"
                      ? {
                          retry_policy:
                            capabilityRetryPolicy.value && Object.keys(capabilityRetryPolicy.value).length > 0
                              ? capabilityRetryPolicy.value
                              : DEFAULT_RETRY_POLICY_PREVIEW,
                          acceptance_policy: {
                            acceptance_criteria: [],
                            critic_required: false,
                          },
                        }
                      : isRecord(predictedExecutionRequestPreview)
                        ? {
                            retry_policy: predictedExecutionRequestPreview.retry_policy ?? {},
                            policy_snapshot: predictedExecutionRequestPreview.policy_snapshot ?? {},
                          }
                        : null
                  }
                  emptyLabel="Policy preview will appear once the active workbench payload is valid."
                />
              </div>
            </SurfacePanel>
          </div>

          <div className="mt-4 rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(16,24,34,0.92),rgba(9,14,22,0.96))] shadow-[0_18px_36px_rgba(15,23,42,0.24)]">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 px-4 py-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300/72">
                  Run Results
                </div>
                <div className="mt-1 text-xs text-slate-300/74">
                  Canonical run status, step progression, execution requests, attempts, artifacts, and debugger events.
                </div>
              </div>
              <div className="flex flex-col items-end gap-2">
                <div className="flex flex-wrap items-center justify-end gap-2">
                  <button
                    type="button"
                    className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={applyForkResult}
                    disabled={!forkResult}
                  >
                    Fork Run
                  </button>
                  <button
                    type="button"
                    className="rounded-xl border border-sky-300/26 bg-sky-400/16 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-50 transition hover:border-sky-300/36 disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={handlePromoteWorkflowDraft}
                    disabled={!workflowPromotionResult?.promotable || !onPromoteWorkflowDraft}
                  >
                    Promote to Workflow
                  </button>
                </div>
                {!onPromoteWorkflowDraft ? (
                  <div className="max-w-[420px] text-right text-[11px] text-amber-100/90">
                    Workflow promotion handoff is unavailable in the current Studio shell.
                  </div>
                ) : null}
                {forkResult?.mode === "agent_raw" ? (
                  <div className="max-w-[420px] text-right text-[11px] text-amber-100/90">
                    Fork will open the raw RunSpec editor: {forkResult.draft.reason}
                  </div>
                ) : null}
                {workflowPromotionResult && !workflowPromotionResult.promotable ? (
                  <div className="max-w-[420px] text-right text-[11px] text-amber-100/90">
                    Promote unavailable: {workflowPromotionResult.reason}
                  </div>
                ) : null}
                <div className="flex flex-wrap items-center justify-end gap-2 text-[10px] uppercase tracking-[0.14em]">
                  <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
                    status {currentRunStatus || "idle"}
                  </span>
                  <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
                    mode {launchWorkbenchModeLabel}
                  </span>
                  {activeRunId ? (
                    <span className="rounded-full border border-sky-300/28 bg-sky-400/14 px-3 py-1 text-sky-50">
                      run {activeRunId}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
            <div className="grid gap-4 px-4 py-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
              <div className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl border border-white/8 bg-black/18 px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-slate-300/72">
                      steps
                    </div>
                    <div className="mt-2 text-2xl font-semibold text-white">
                      {debuggerData?.steps.length ?? 0}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/8 bg-black/18 px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-slate-300/72">
                      execution requests
                    </div>
                    <div className="mt-2 text-2xl font-semibold text-white">
                      {debuggerData?.execution_requests.length ?? 0}
                    </div>
                  </div>
                  <div className="rounded-2xl border border-white/8 bg-black/18 px-3 py-3">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-slate-300/72">
                      attempts / invocations
                    </div>
                    <div className="mt-2 text-2xl font-semibold text-white">
                      {(debuggerData?.attempts.length ?? 0) + (debuggerData?.invocations.length ?? 0)}
                    </div>
                  </div>
                </div>

                {debuggerLoading ? (
                  <div className="rounded-2xl border border-white/8 bg-black/18 px-3 py-3 text-xs text-slate-300/74">
                    Refreshing debugger state...
                  </div>
                ) : null}
                {debuggerError ? (
                  <div className="rounded-2xl border border-rose-300/18 bg-rose-400/12 px-3 py-3 text-xs text-rose-100">
                    {debuggerError}
                  </div>
                ) : null}
                {launchResponse?.execution_request ? (
                  <JsonPreview
                    title="Initial Prepared ExecutionRequest"
                    value={launchResponse.execution_request}
                    emptyLabel="The launch response did not include a prepared execution request."
                  />
                ) : null}

                <div className="rounded-2xl border border-white/8 bg-black/18 p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-300/72">
                    Steps
                  </div>
                  <div className="mt-3 space-y-2">
                    {debuggerData?.steps.map((stepPayload) => {
                      const replayResult = replayResultsByStep.get(stepPayload.step.id);
                      return (
                        <div
                          key={stepPayload.step.id}
                          className="rounded-2xl border border-white/8 bg-slate-950/40 px-3 py-3"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-white">
                                {stepPayload.step.name}
                              </div>
                              <div className="mt-1 text-xs text-slate-300/74">
                                {stepPayload.step.capability_id}
                              </div>
                            </div>
                            <div className="flex flex-wrap items-center justify-end gap-2 text-[10px] uppercase tracking-[0.14em] text-slate-300/76">
                              <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                                {stepPayload.step.status}
                              </span>
                              <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                                requests {stepPayload.execution_requests.length}
                              </span>
                              <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1">
                                attempts {stepPayload.attempts.length}
                              </span>
                              <button
                                type="button"
                                className="rounded-xl border border-white/10 bg-white/[0.05] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                                onClick={() => handleReplayStep(stepPayload.step.id)}
                                disabled={!replayResult?.replayable}
                              >
                                Replay Step
                              </button>
                            </div>
                          </div>
                          {!replayResult?.replayable ? (
                            <div className="mt-2 text-xs text-amber-100/90">
                              Replay unavailable: {replayResult?.reason || "The replay payload could not be reconstructed."}
                            </div>
                          ) : null}
                          {stepPayload.error?.message ? (
                            <div className="mt-2 text-xs text-rose-100">
                              {String(stepPayload.error.message)}
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                    {!debuggerData?.steps.length ? (
                      <div className="text-xs text-slate-300/74">
                        Launch a workbench run to inspect step state here.
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <JsonPreview
                  title="Execution Requests"
                  value={debuggerData?.execution_requests}
                  emptyLabel="No execution requests have been captured for this run yet."
                />
                <JsonPreview
                  title="Attempts / Tool Calls"
                  value={{
                    attempts: debuggerData?.attempts ?? [],
                    invocations: debuggerData?.invocations ?? [],
                  }}
                  emptyLabel="No attempts or tool invocations have been captured yet."
                />
                <JsonPreview
                  title="Artifacts"
                  value={artifacts}
                  emptyLabel="Artifacts will appear once a step produces durable outputs."
                />
                <JsonPreview
                  title="Debugger Timeline"
                  value={debuggerData?.events}
                  emptyLabel="Debugger events will stream in as the canonical runtime advances the run."
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
