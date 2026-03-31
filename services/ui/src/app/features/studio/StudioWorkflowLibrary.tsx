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
  onOpenDefinition: (definition: WorkflowDefinition) => void;
  onDeleteDefinition: (definition: WorkflowDefinition) => void;
  onOpenVersion: (version: WorkflowVersion) => void;
  onCreateManualTrigger: () => void;
  onInvokeTrigger: (trigger: WorkflowTrigger) => void;
};

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
  onOpenDefinition,
  onDeleteDefinition,
  onOpenVersion,
  onCreateManualTrigger,
  onInvokeTrigger,
}: StudioWorkflowLibraryProps) {
  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            Workflow Library
          </div>
          <h3 className="mt-1 font-display text-2xl text-slate-900">Saved Drafts</h3>
        </div>
        <button
          className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
          onClick={onRefresh}
        >
          Refresh
        </button>
      </div>

      <p className="mt-3 text-sm leading-6 text-slate-600">
        Reopen saved workflow definitions into the editor, then inspect or restore published
        versions for the active draft.
      </p>

      <div className="mt-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
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
                  }`}
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
                      onClick={() => onOpenDefinition(definition)}
                    >
                      Open Draft
                    </button>
                    <button
                      className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1.5 text-xs font-semibold text-rose-700 transition hover:border-rose-400 hover:text-rose-800 disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={() => onDeleteDefinition(definition)}
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
                  }`}
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
                      onClick={() => onOpenVersion(version)}
                    >
                      Restore Version
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
