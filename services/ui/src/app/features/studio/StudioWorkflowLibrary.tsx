"use client";

import type {
  WorkflowDefinition,
  WorkflowRun,
  WorkflowTrigger,
  WorkflowVersion,
} from "./types";
import { formatTimestamp } from "./utils";

type StudioWorkflowLibraryProps = {
  workflowDefinitions: WorkflowDefinition[];
  workflowDefinitionsLoading: boolean;
  workflowDefinitionsError: string | null;
  workflowVersions: WorkflowVersion[];
  workflowVersionsLoading: boolean;
  workflowVersionsError: string | null;
  workflowTriggers: WorkflowTrigger[];
  workflowTriggersLoading: boolean;
  workflowTriggersError: string | null;
  workflowRuns: WorkflowRun[];
  workflowRunsLoading: boolean;
  workflowRunsError: string | null;
  activeWorkflowDefinitionId: string | null;
  activeWorkflowVersionId: string | null;
  deletingWorkflowDefinitionId: string | null;
  onRefresh: () => void;
  onSelectDefinition?: (definition: WorkflowDefinition) => void;
  onOpenDefinition: (definition: WorkflowDefinition) => void;
  onDeleteDefinition: (definition: WorkflowDefinition) => void;
  openDefinitionLabel?: string;
  onSelectVersion?: (version: WorkflowVersion) => void;
  onOpenVersion: (version: WorkflowVersion) => void;
  openVersionLabel?: string;
  onCreateManualTrigger: () => void;
  onInvokeTrigger: (trigger: WorkflowTrigger) => void;
};

const libraryPanelClassName =
  "rounded-[32px] border border-[#22304a] bg-[linear-gradient(180deg,rgba(15,23,42,0.98),rgba(9,17,27,0.96))] p-4 text-slate-100 shadow-[0_24px_60px_rgba(2,8,23,0.24)] [&_.border-slate-200]:border-white/10 [&_.border-slate-300]:border-white/12 [&_.border-sky-200]:border-sky-300/25 [&_.border-emerald-200]:border-emerald-300/25 [&_.border-amber-200]:border-amber-300/25 [&_.border-rose-200]:border-rose-300/25 [&_.bg-slate-50]:bg-white/[0.04] [&_.bg-slate-100]:bg-white/[0.07] [&_.bg-slate-200]:bg-white/[0.07] [&_.bg-white]:bg-white/[0.05] [&_.bg-sky-50]:bg-sky-400/10 [&_.bg-emerald-50]:bg-emerald-400/10 [&_.bg-amber-50]:bg-amber-400/10 [&_.bg-amber-100]:bg-amber-400/12 [&_.bg-rose-50]:bg-rose-400/10 [&_.bg-rose-100]:bg-rose-400/12 [&_.bg-sky-100]:bg-sky-400/12 [&_.bg-emerald-100]:bg-emerald-400/12 [&_.text-slate-900]:text-white [&_.text-slate-700]:text-slate-200 [&_.text-slate-600]:text-slate-300/82 [&_.text-slate-500]:text-slate-400 [&_.text-amber-700]:text-amber-100 [&_.text-amber-800]:text-amber-100 [&_.text-amber-900]:text-amber-50 [&_.text-rose-700]:text-rose-100 [&_.text-rose-800]:text-rose-100 [&_.text-rose-900]:text-rose-50 [&_.text-sky-700]:text-sky-100 [&_.text-emerald-700]:text-emerald-100 [&_article]:border-white/10 [&_article]:bg-white/[0.04]";

export default function StudioWorkflowLibrary({
  workflowDefinitions,
  workflowDefinitionsLoading,
  workflowDefinitionsError,
  workflowVersions,
  workflowVersionsLoading,
  workflowVersionsError,
  workflowTriggers,
  workflowTriggersLoading,
  workflowTriggersError,
  workflowRuns,
  workflowRunsLoading,
  workflowRunsError,
  activeWorkflowDefinitionId,
  activeWorkflowVersionId,
  deletingWorkflowDefinitionId,
  onRefresh,
  onSelectDefinition,
  onOpenDefinition,
  onDeleteDefinition,
  openDefinitionLabel = "Open Draft",
  onSelectVersion,
  onOpenVersion,
  openVersionLabel = "Restore Version",
  onCreateManualTrigger,
  onInvokeTrigger,
}: StudioWorkflowLibraryProps) {
  return (
    <section className={libraryPanelClassName}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-sky-100/68">
            Saved Workflows
          </div>
          <h3 className="mt-1 font-display text-2xl text-white">Workflow Versions</h3>
        </div>
        <button
          className="rounded-full border border-white/12 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-300/40 hover:bg-white/[0.08]"
          onClick={onRefresh}
        >
          Refresh
        </button>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-300/82">
        Manage reusable workflow definitions, versions, triggers, and published automations.
      </p>

      <div className="mt-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-300/75">
          Definitions
        </div>
        {workflowDefinitionsLoading ? (
          <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Loading saved workflows...
          </div>
        ) : workflowDefinitionsError ? (
          <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {workflowDefinitionsError}
          </div>
        ) : workflowDefinitions.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            Save a Studio draft to start building version history.
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {workflowDefinitions.map((definition) => {
              const isActive = definition.id === activeWorkflowDefinitionId;
              return (
                <article
                  key={definition.id}
                  className={`rounded-2xl border px-4 py-4 ${
                    isActive
                      ? "border-sky-200 bg-sky-50/80"
                      : "border-slate-200 bg-slate-50/70"
                  } ${onSelectDefinition ? "cursor-pointer transition hover:border-sky-300/35" : ""}`}
                  onClick={() => {
                    onSelectDefinition?.(definition);
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-slate-900">
                        {definition.title}
                      </div>
                      <div className="mt-1 line-clamp-2 text-xs leading-5 text-slate-600">
                        {definition.goal || "No goal recorded for this workflow."}
                      </div>
                    </div>
                    {isActive ? (
                      <span className="rounded-full bg-sky-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-700">
                        Active
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-500">
                    <span className="rounded-full bg-white px-2.5 py-1">
                      updated {formatTimestamp(definition.updated_at)}
                    </span>
                    {definition.user_id ? (
                      <span className="rounded-full bg-white px-2.5 py-1">
                        user {definition.user_id}
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpenDefinition(definition);
                      }}
                    >
                      {openDefinitionLabel}
                    </button>
                    <button
                      className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs font-semibold text-rose-700 transition hover:border-rose-400 hover:text-rose-800 disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteDefinition(definition);
                      }}
                      disabled={deletingWorkflowDefinitionId === definition.id}
                    >
                      {deletingWorkflowDefinitionId === definition.id ? "Deleting..." : "Delete"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-6">
        <div className="flex items-center justify-between gap-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Triggers
          </div>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={onCreateManualTrigger}
            disabled={!activeWorkflowDefinitionId}
          >
            Create Manual Trigger
          </button>
        </div>
        {!activeWorkflowDefinitionId ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            Open a saved workflow definition before configuring triggers.
          </div>
        ) : workflowTriggersLoading ? (
          <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Loading triggers...
          </div>
        ) : workflowTriggersError ? (
          <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {workflowTriggersError}
          </div>
        ) : workflowTriggers.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            No triggers yet. Create a manual trigger to invoke the latest published version.
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {workflowTriggers.map((trigger) => (
              <article
                key={trigger.id}
                className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-slate-900">
                      {trigger.title}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-500">
                      <span className="rounded-full bg-white px-2.5 py-1">
                        {trigger.trigger_type}
                      </span>
                      <span
                        className={`rounded-full px-2.5 py-1 ${
                          trigger.enabled
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-slate-200 text-slate-600"
                        }`}
                      >
                        {trigger.enabled ? "enabled" : "disabled"}
                      </span>
                    </div>
                  </div>
                  <button
                    className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
                    onClick={() => onInvokeTrigger(trigger)}
                    disabled={!trigger.enabled}
                  >
                    Invoke
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      <div className="mt-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          Version History
        </div>
        {!activeWorkflowDefinitionId ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            Open a saved workflow definition to browse its published versions.
          </div>
        ) : workflowVersionsLoading ? (
          <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Loading version history...
          </div>
        ) : workflowVersionsError ? (
          <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {workflowVersionsError}
          </div>
        ) : workflowVersions.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            Publish a version to make this workflow runnable and restorable.
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {workflowVersions.map((version) => {
              const isActive = version.id === activeWorkflowVersionId;
              return (
                <article
                  key={version.id}
                  className={`rounded-2xl border px-4 py-4 ${
                    isActive
                      ? "border-emerald-200 bg-emerald-50/80"
                      : "border-slate-200 bg-slate-50/70"
                  } ${onSelectVersion ? "cursor-pointer transition hover:border-emerald-300/35" : ""}`}
                  onClick={() => {
                    onSelectVersion?.(version);
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">
                        v{version.version_number}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        {formatTimestamp(version.created_at)}
                      </div>
                    </div>
                    {isActive ? (
                      <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-emerald-700">
                        Loaded
                      </span>
                    ) : null}
                  </div>
                  <div className="mt-3 line-clamp-2 text-xs leading-5 text-slate-600">
                    {version.goal || version.title || "Published workflow version"}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      className="rounded-full border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpenVersion(version);
                      }}
                    >
                      {openVersionLabel}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-6">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          Run History
        </div>
        {!activeWorkflowDefinitionId ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            Open a saved workflow definition to browse its run history.
          </div>
        ) : workflowRunsLoading ? (
          <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Loading workflow runs...
          </div>
        ) : workflowRunsError ? (
          <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {workflowRunsError}
          </div>
        ) : workflowRuns.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
            No runs yet. Publish and run a workflow, or invoke a trigger, to populate history.
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {workflowRuns.map((run) => (
              <article
                key={run.id}
                className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-semibold text-slate-900">{run.title}</div>
                    <div className="mt-1 text-xs text-slate-500">
                      {formatTimestamp(run.updated_at || run.created_at)}
                    </div>
                  </div>
                  <span
                    className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
                      run.job_status === "succeeded"
                        ? "bg-emerald-100 text-emerald-700"
                        : run.job_status === "failed"
                          ? "bg-rose-100 text-rose-700"
                          : run.job_status === "running"
                            ? "bg-sky-100 text-sky-700"
                            : "bg-slate-200 text-slate-600"
                    }`}
                  >
                    {run.job_status || "queued"}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.14em] text-slate-500">
                  <span className="rounded-full bg-white px-2.5 py-1">
                    job {run.job_id.slice(0, 8)}
                  </span>
                  <span className="rounded-full bg-white px-2.5 py-1">
                    plan {run.plan_id.slice(0, 8)}
                  </span>
                  <span className="rounded-full bg-white px-2.5 py-1">
                    version {run.version_id.slice(0, 8)}
                  </span>
                  {run.trigger_id ? (
                    <span className="rounded-full bg-white px-2.5 py-1">
                      trigger {run.trigger_id.slice(0, 8)}
                    </span>
                  ) : null}
                </div>
                {run.latest_task_error ? (
                  <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-3 py-3 text-xs leading-5 text-rose-700">
                    <div className="font-semibold uppercase tracking-[0.14em] text-rose-800">
                      Latest Task Error
                    </div>
                    <div className="mt-1 text-rose-900">
                      {run.latest_task_name ? `${run.latest_task_name}: ` : null}
                      {run.latest_task_error}
                    </div>
                  </div>
                ) : run.job_error ? (
                  <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-3 text-xs leading-5 text-amber-800">
                    <div className="font-semibold uppercase tracking-[0.14em] text-amber-900">
                      Run Error
                    </div>
                    <div className="mt-1">{run.job_error}</div>
                  </div>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
