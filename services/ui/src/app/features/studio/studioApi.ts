"use client";

/**
 * studioApi.ts — Client-side API module for Studio Workbench launches.
 *
 * All workbench fetch calls go through this module so that page components
 * stay free of raw fetch logic and retry handling.
 */

import type {
  AdaptiveReplanStatus,
  AgentDefinition,
  AgentDefinitionCreateRequest,
  AgentDefinitionUpdateRequest,
  AgentDefinitionVersion,
  AgentDefinitionVersionPublishRequest,
  CapabilityCatalog,
} from "./types";

const apiUrl =
  typeof process !== "undefined"
    ? process.env.NEXT_PUBLIC_API_URL || "/api"
    : "/api";

// ─── Shared types ────────────────────────────────────────────────────────────

export type CanonicalRun = {
  id: string;
  kind: string;
  title: string;
  goal: string;
  requested_context_json?: Record<string, unknown>;
  status: string;
  job_id: string;
  plan_id?: string | null;
  workflow_run_id?: string | null;
  source_definition_id?: string | null;
  source_version_id?: string | null;
  source_trigger_id?: string | null;
  job_status?: string | null;
  job_error?: string | null;
  latest_step_id?: string | null;
  latest_step_name?: string | null;
  latest_step_error?: string | null;
  user_id?: string | null;
  planning_mode?: string;
  current_revision_number?: number;
  adaptive_status?: AdaptiveReplanStatus;
  run_spec: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type CanonicalRunStep = {
  id: string;
  run_id: string;
  job_id: string;
  spec_step_id: string;
  name: string;
  description: string;
  instruction: string;
  status: string;
  capability_id: string;
  input_bindings: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type WorkbenchRunLaunchResponse = {
  run: CanonicalRun;
  run_spec: Record<string, unknown>;
  execution_request?: Record<string, unknown> | null;
};

export type WorkbenchCapabilityRunRequest = {
  title?: string;
  goal?: string;
  user_id?: string | null;
  context_json?: Record<string, unknown>;
  capability_id: string;
  inputs?: Record<string, unknown>;
  retry_policy?: Record<string, unknown> | null;
};

export type WorkbenchAgentRunRequest = {
  title?: string;
  goal?: string;
  user_id?: string | null;
  context_json?: Record<string, unknown>;
  run_spec: Record<string, unknown>;
  agent_definition_id?: string | null;
  agent_definition_version_id?: string | null;
};

export type WorkbenchDebuggerStep = {
  step: CanonicalRunStep;
  latest_result: Record<string, unknown>;
  execution_requests: Record<string, unknown>[];
  checkpoints: Record<string, unknown>[];
  attempts: Record<string, unknown>[];
  timeline: Record<string, unknown>[];
  error: Record<string, unknown>;
};

export type WorkbenchDebuggerData = {
  run: CanonicalRun;
  job?: Record<string, unknown>;
  plan?: Record<string, unknown> | null;
  planning_mode?: string;
  current_revision_number?: number;
  adaptive_status?: AdaptiveReplanStatus;
  revision_history?: Record<string, unknown>[];
  last_replan_reason?: string | null;
  generated_at?: string;
  steps: WorkbenchDebuggerStep[];
  execution_requests: Record<string, unknown>[];
  attempts: Record<string, unknown>[];
  invocations: Record<string, unknown>[];
  events: Record<string, unknown>[];
  checkpoints?: Record<string, unknown>[];
};

export type CapabilitySearchItem = {
  id: string;
  score: number;
  reason: string;
  source: string;
  description: string;
  group: string;
  subgroup: string;
  tags: string[];
};

export type CapabilitySearchResponse = {
  mode?: string;
  items: CapabilitySearchItem[];
  query: string;
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail: string;
    try {
      const err = await res.json();
      detail =
        typeof err?.detail === "string"
          ? err.detail
          : JSON.stringify(err?.detail ?? err);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail: string;
    try {
      const err = await res.json();
      detail =
        typeof err?.detail === "string"
          ? err.detail
          : JSON.stringify(err?.detail ?? err);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, { method: "GET" });
  if (!res.ok) {
    let detail: string;
    try {
      const err = await res.json();
      detail =
        typeof err?.detail === "string"
          ? err.detail
          : JSON.stringify(err?.detail ?? err);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

async function deleteJson<T>(path: string): Promise<T> {
  const res = await fetch(`${apiUrl}${path}`, { method: "DELETE" });
  if (!res.ok) {
    let detail: string;
    try {
      const err = await res.json();
      detail =
        typeof err?.detail === "string"
          ? err.detail
          : JSON.stringify(err?.detail ?? err);
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ─── Agent definitions ───────────────────────────────────────────────────────

export async function fetchAgentDefinitions(
  userId?: string,
  includeDisabled = false
): Promise<AgentDefinition[]> {
  const params = new URLSearchParams();
  const normalizedUserId = userId?.trim();
  if (normalizedUserId) {
    params.set("user_id", normalizedUserId);
  }
  if (includeDisabled) {
    params.set("include_disabled", "true");
  }
  const query = params.toString();
  return getJson<AgentDefinition[]>(`/agents/definitions${query ? `?${query}` : ""}`);
}

export async function createAgentDefinition(
  req: AgentDefinitionCreateRequest
): Promise<AgentDefinition> {
  return postJson<AgentDefinition>("/agents/definitions", req);
}

export async function updateAgentDefinition(
  agentId: string,
  req: AgentDefinitionUpdateRequest
): Promise<AgentDefinition> {
  return putJson<AgentDefinition>(`/agents/definitions/${encodeURIComponent(agentId)}`, req);
}

export async function deleteAgentDefinition(agentId: string): Promise<{ ok: boolean }> {
  return deleteJson<{ ok: boolean }>(`/agents/definitions/${encodeURIComponent(agentId)}`);
}

export async function publishAgentDefinitionVersion(
  agentId: string,
  req: AgentDefinitionVersionPublishRequest = {}
): Promise<AgentDefinitionVersion> {
  return postJson<AgentDefinitionVersion>(
    `/agents/definitions/${encodeURIComponent(agentId)}/versions`,
    req
  );
}

export async function fetchAgentDefinitionVersions(
  agentId: string
): Promise<AgentDefinitionVersion[]> {
  return getJson<AgentDefinitionVersion[]>(
    `/agents/definitions/${encodeURIComponent(agentId)}/versions`
  );
}

export async function fetchAgentDefinitionVersion(
  agentId: string,
  versionNumber: number
): Promise<AgentDefinitionVersion> {
  return getJson<AgentDefinitionVersion>(
    `/agents/definitions/${encodeURIComponent(agentId)}/versions/${versionNumber}`
  );
}

// ─── Workbench launch ─────────────────────────────────────────────────────────

export async function launchCapabilityRun(
  req: WorkbenchCapabilityRunRequest
): Promise<WorkbenchRunLaunchResponse> {
  return postJson<WorkbenchRunLaunchResponse>(
    "/workbench/capability-runs",
    req
  );
}

export async function launchAgentRun(
  req: WorkbenchAgentRunRequest
): Promise<WorkbenchRunLaunchResponse> {
  return postJson<WorkbenchRunLaunchResponse>("/workbench/agent-runs", req);
}

// ─── Run polling ─────────────────────────────────────────────────────────────

export async function fetchRun(runId: string): Promise<CanonicalRun> {
  return getJson<CanonicalRun>(`/runs/${encodeURIComponent(runId)}`);
}

export async function fetchRunSteps(runId: string): Promise<CanonicalRunStep[]> {
  return getJson<CanonicalRunStep[]>(
    `/runs/${encodeURIComponent(runId)}/steps`
  );
}

export async function fetchRunDebugger(
  runId: string
): Promise<WorkbenchDebuggerData> {
  return getJson<WorkbenchDebuggerData>(
    `/runs/${encodeURIComponent(runId)}/debugger`
  );
}

// ─── Capability catalog ───────────────────────────────────────────────────────

export async function fetchCapabilityCatalog(
  withSchemas = true
): Promise<CapabilityCatalog> {
  return getJson<CapabilityCatalog>(`/capabilities?with_schemas=${withSchemas}`);
}

export async function searchCapabilities(
  query: string,
  limit = 12
): Promise<CapabilitySearchResponse> {
  return postJson<CapabilitySearchResponse>("/capabilities/search", {
    query,
    limit,
    request_source: "workbench",
  });
}
