"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import AppShell from "../../components/AppShell";
import StudioWorkflowLibrary from "./StudioWorkflowLibrary";
import type {
  WorkflowDefinition,
  WorkflowRun,
  WorkflowRunResult,
  WorkflowTrigger,
  WorkflowVersion,
} from "./types";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "/api";
const DEFAULT_WORKSPACE_USER_ID = "narendersurabhi";

const detailMessage = (body: unknown, fallback: string) => {
  if (body && typeof body === "object" && typeof (body as { detail?: unknown }).detail === "string") {
    return (body as { detail: string }).detail;
  }
  return fallback;
};

export default function WorkflowLibraryPage() {
  const router = useRouter();
  const [workspaceUserId, setWorkspaceUserId] = useState(DEFAULT_WORKSPACE_USER_ID);
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
  const [activeWorkflowDefinitionId, setActiveWorkflowDefinitionId] = useState<string | null>(null);
  const [activeWorkflowVersionId, setActiveWorkflowVersionId] = useState<string | null>(null);
  const [deletingWorkflowDefinitionId, setDeletingWorkflowDefinitionId] = useState<string | null>(
    null
  );
  const [workflowActionLoading, setWorkflowActionLoading] = useState<"delete" | "save" | "run" | null>(
    null
  );
  const [notice, setNotice] = useState<string | null>(null);

  const activeWorkflowDefinition = useMemo(
    () => workflowDefinitions.find((definition) => definition.id === activeWorkflowDefinitionId) || null,
    [activeWorkflowDefinitionId, workflowDefinitions]
  );

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
        throw new Error(detailMessage(body, `Workflow library request failed (${response.status}).`));
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
        throw new Error(
          detailMessage(body, `Workflow version history request failed (${response.status}).`)
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
        throw new Error(detailMessage(body, `Workflow trigger request failed (${response.status}).`));
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
        throw new Error(
          detailMessage(body, `Workflow run history request failed (${response.status}).`)
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
    if (workflowDefinitions.length === 0) {
      setActiveWorkflowDefinitionId(null);
      return;
    }
    if (!activeWorkflowDefinitionId || !workflowDefinitions.some((item) => item.id === activeWorkflowDefinitionId)) {
      setActiveWorkflowDefinitionId(workflowDefinitions[0].id);
    }
  }, [activeWorkflowDefinitionId, workflowDefinitions]);

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

  useEffect(() => {
    if (!activeWorkflowVersionId || !workflowVersions.some((version) => version.id === activeWorkflowVersionId)) {
      setActiveWorkflowVersionId(workflowVersions[0]?.id || null);
    }
  }, [activeWorkflowVersionId, workflowVersions]);

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
    setDeletingWorkflowDefinitionId(definition.id);
    try {
      const response = await fetch(
        `${apiUrl}/workflows/definitions/${encodeURIComponent(definition.id)}`,
        { method: "DELETE" }
      );
      const body = response.status === 204 ? null : ((await response.json()) as { detail?: unknown } | null);
      if (!response.ok) {
        throw new Error(detailMessage(body, `Delete draft failed (${response.status}).`));
      }
      setWorkflowDefinitions((prev) => prev.filter((item) => item.id !== definition.id));
      setNotice(`Deleted saved draft ${definition.title}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Failed to delete saved draft.");
    } finally {
      setDeletingWorkflowDefinitionId(null);
      setWorkflowActionLoading(null);
    }
  };

  const createManualWorkflowTrigger = async () => {
    if (!activeWorkflowDefinitionId || !activeWorkflowDefinition) {
      setNotice("Select a workflow definition before creating a trigger.");
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
            title: `${activeWorkflowDefinition.title} manual trigger`,
            trigger_type: "manual",
            enabled: true,
            config: { version_mode: "latest_published" },
            user_id: workspaceUserId.trim() || undefined,
            metadata: { source: "workflow_library_page" },
          }),
        }
      );
      const body = (await response.json()) as WorkflowTrigger | { detail?: unknown };
      if (!response.ok) {
        throw new Error(detailMessage(body, `Create trigger failed (${response.status}).`));
      }
      void refreshWorkflowTriggers(activeWorkflowDefinitionId);
      setNotice(`Created manual trigger ${(body as WorkflowTrigger).title}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Failed to create workflow trigger.");
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
        throw new Error(detailMessage(body, `Trigger invoke failed (${response.status}).`));
      }
      const result = body as WorkflowRunResult;
      setActiveWorkflowDefinitionId(result.workflow_definition.id);
      setActiveWorkflowVersionId(result.workflow_version.id);
      void refreshWorkflowRuns(result.workflow_definition.id);
      setNotice(`Triggered job ${result.job.id} via ${trigger.title}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Failed to invoke workflow trigger.");
    } finally {
      setWorkflowActionLoading(null);
    }
  };

  const summaryChips = [
    `drafts ${workflowDefinitions.length}`,
    `versions ${workflowVersions.length}`,
    `runs ${workflowRuns.length}`,
    workflowActionLoading ? `${workflowActionLoading}...` : "ready",
    activeWorkflowDefinition ? `active ${activeWorkflowDefinition.title}` : "no active draft",
  ];

  return (
    <AppShell
      activeScreen="workflows"
      title="Saved Workflows"
      breadcrumbs={[
        { label: "Project", href: "/project" },
        { label: "Saved Workflows" },
      ]}
      actions={
        <>
          <Link
            href="/studio"
            className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
          >
            Open Builder
          </Link>
          <Link
            href="/studio?mode=new"
            className="rounded-xl border border-slate-200/18 bg-slate-950/25 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-white transition hover:border-white/30 hover:bg-slate-950/35"
          >
            New Workflow
          </Link>
        </>
      }
    >
      {notice ? (
        <div className="mb-4 rounded-[24px] border border-sky-300/15 bg-sky-400/10 px-4 py-3 text-sm text-sky-50">
          {notice}
        </div>
      ) : null}
      <section className="relative">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-100/72">
                    Saved Workflows
                  </div>
                  <h2 className="mt-1 text-[30px] font-semibold tracking-[-0.03em] text-white">
                    Saved Workflows
                  </h2>
                  <p className="mt-1 max-w-3xl text-[13px] leading-5 text-slate-200/74">
                    Manage reusable workflows, versions, triggers, and published automations.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]">
                  {summaryChips.map((chip) => (
                    <span
                      key={chip}
                      className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100"
                    >
                      {chip}
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-end justify-between gap-3 rounded-[24px] border border-white/10 bg-[linear-gradient(180deg,rgba(63,79,95,0.54),rgba(47,60,74,0.68))] px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
                <label className="min-w-[260px] flex-1">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-200/72">
                    Workspace User ID
                  </div>
                  <input
                    className="mt-2 w-full rounded-xl border border-white/10 bg-slate-950/18 px-3 py-2 text-sm text-white outline-none transition placeholder:text-slate-300/42 focus:border-sky-300/40 focus:bg-slate-950/28"
                    value={workspaceUserId}
                    onChange={(event) => setWorkspaceUserId(event.target.value)}
                    placeholder="narendersurabhi"
                  />
                </label>
                <div className="max-w-xl text-sm leading-6 text-slate-300/82">
                  Select a saved workflow to inspect its versions, triggers, and run history here.
                  Open Draft and Open Version send you back to Studio with the selected record loaded.
                </div>
              </div>

              <div className="mt-5 max-w-[1180px]">
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
                  deletingWorkflowDefinitionId={deletingWorkflowDefinitionId}
                  onRefresh={() => {
                    void refreshWorkflowDefinitions();
                    if (activeWorkflowDefinitionId) {
                      void refreshWorkflowVersions(activeWorkflowDefinitionId);
                      void refreshWorkflowTriggers(activeWorkflowDefinitionId);
                      void refreshWorkflowRuns(activeWorkflowDefinitionId);
                    }
                  }}
                  onSelectDefinition={(definition) => {
                    setActiveWorkflowDefinitionId(definition.id);
                  }}
                  onOpenDefinition={(definition) => {
                    router.push(`/studio?definition=${encodeURIComponent(definition.id)}`);
                  }}
                  onDeleteDefinition={(definition) => {
                    void deleteWorkflowDefinition(definition);
                  }}
                  openDefinitionLabel="Open In Studio"
                  onSelectVersion={(version) => {
                    setActiveWorkflowVersionId(version.id);
                  }}
                  onOpenVersion={(version) => {
                    router.push(
                      `/studio?definition=${encodeURIComponent(version.definition_id)}&version=${encodeURIComponent(version.id)}`
                    );
                  }}
                  openVersionLabel="Open Version"
                  onCreateManualTrigger={() => {
                    void createManualWorkflowTrigger();
                  }}
                  onInvokeTrigger={(trigger) => {
                    void invokeWorkflowTrigger(trigger);
                  }}
                />
              </div>
      </section>
    </AppShell>
  );
}
