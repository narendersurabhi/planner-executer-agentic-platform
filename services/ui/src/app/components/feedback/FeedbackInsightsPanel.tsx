"use client";

import {
  formatFeedbackRate,
  type FeedbackBreakdownBucket,
  type FeedbackReasonBucket,
  type FeedbackSummaryResponse
} from "../../lib/feedback";

type FeedbackInsightsPanelProps = {
  summary: FeedbackSummaryResponse | null;
  loading?: boolean;
  error?: string | null;
  onRefresh?: () => void;
};

const renderBucketList = (
  buckets: FeedbackBreakdownBucket[],
  emptyLabel: string,
  formatter?: (bucket: FeedbackBreakdownBucket) => string
) => {
  if (!buckets.length) {
    return <div className="text-xs text-slate-500">{emptyLabel}</div>;
  }
  return (
    <div className="space-y-2">
      {buckets.map((bucket) => (
        <div
          key={`feedback-bucket-${bucket.key}`}
          className="flex items-center justify-between gap-3 text-sm text-slate-700"
        >
          <span className="truncate">{bucket.key}</span>
          <span className="shrink-0 text-xs text-slate-500">
            {formatter ? formatter(bucket) : `${bucket.total}`}
          </span>
        </div>
      ))}
    </div>
  );
};

const renderReasonList = (reasons: FeedbackReasonBucket[]) => {
  if (!reasons.length) {
    return <div className="text-xs text-slate-500">No negative or partial reasons yet.</div>;
  }
  return (
    <div className="space-y-2">
      {reasons.map((reason) => (
        <div
          key={`feedback-reason-${reason.reason_code}`}
          className="flex items-center justify-between gap-3 text-sm text-slate-700"
        >
          <span className="truncate">{reason.reason_code}</span>
          <span className="shrink-0 text-xs text-slate-500">{reason.count}</span>
        </div>
      ))}
    </div>
  );
};

export default function FeedbackInsightsPanel({
  summary,
  loading = false,
  error = null,
  onRefresh
}: FeedbackInsightsPanelProps) {
  const total = summary?.total ?? 0;
  const terminalStatuses = summary?.correlates?.terminal_statuses ?? [];

  return (
    <section className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm animate-fade-up">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-xl">Feedback Insights</h2>
          <p className="mt-1 text-xs text-slate-500">
            Read-only quality signals from explicit user feedback and linked runtime context.
          </p>
        </div>
        {onRefresh ? (
          <button
            type="button"
            className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
            onClick={onRefresh}
          >
            Refresh
          </button>
        ) : null}
      </div>

      {loading && !summary ? (
        <div className="mt-4 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
          Loading feedback analytics...
        </div>
      ) : null}

      {error ? (
        <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          {error}
        </div>
      ) : null}

      {summary ? (
        <>
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Volume</div>
              <div className="mt-2 text-2xl font-semibold text-slate-900">{total}</div>
              <div className="mt-1 text-xs text-slate-500">Explicit feedback rows</div>
            </div>
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Chat</div>
              <div className="mt-2 text-2xl font-semibold text-slate-900">
                {formatFeedbackRate(summary.metrics?.chat_helpfulness_rate)}
              </div>
              <div className="mt-1 text-xs text-slate-500">Helpful response rate</div>
            </div>
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Intent</div>
              <div className="mt-2 text-2xl font-semibold text-slate-900">
                {formatFeedbackRate(summary.metrics?.intent_agreement_rate)}
              </div>
              <div className="mt-1 text-xs text-slate-500">Agreement rate</div>
            </div>
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Plans</div>
              <div className="mt-2 text-2xl font-semibold text-slate-900">
                {formatFeedbackRate(summary.metrics?.plan_approval_rate)}
              </div>
              <div className="mt-1 text-xs text-slate-500">Approval rate</div>
            </div>
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Outcome</div>
              <div className="mt-2 text-2xl font-semibold text-slate-900">
                {formatFeedbackRate(summary.metrics?.job_outcome_positive_rate)}
              </div>
              <div className="mt-1 text-xs text-slate-500">Positive outcome rate</div>
            </div>
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-4">
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Top Negative Reasons</div>
              <div className="mt-3">
                {renderReasonList(summary.negative_reasons.slice(0, 5))}
              </div>
            </div>
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Top Models</div>
              <div className="mt-3">
                {renderBucketList(summary.llm_models.slice(0, 5), "No model-tagged feedback yet.")}
              </div>
            </div>
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Planner Versions</div>
              <div className="mt-3">
                {renderBucketList(
                  summary.planner_versions.slice(0, 5),
                  "No planner-tagged feedback yet."
                )}
              </div>
            </div>
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Workflow Sources</div>
              <div className="mt-3">
                {renderBucketList(
                  summary.workflow_sources.slice(0, 5),
                  "No workflow source dimensions yet."
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-4 xl:grid-cols-3">
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Outcome Distribution</div>
              <div className="mt-3">
                {renderBucketList(
                  terminalStatuses.slice(0, 5),
                  "No job-linked feedback yet.",
                  (bucket) => `${bucket.total} jobs`
                )}
              </div>
            </div>
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">Operational Correlates</div>
              <div className="mt-3 grid gap-2 text-sm text-slate-700">
                <div className="flex items-center justify-between gap-3">
                  <span>Jobs with feedback</span>
                  <span className="text-xs text-slate-500">{summary.correlates.job_count}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Replans</span>
                  <span className="text-xs text-slate-500">{summary.correlates.replan_count}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Retries</span>
                  <span className="text-xs text-slate-500">{summary.correlates.retry_count}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Failed tasks</span>
                  <span className="text-xs text-slate-500">
                    {summary.correlates.failed_task_count}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Plan failures</span>
                  <span className="text-xs text-slate-500">
                    {summary.correlates.plan_failure_count}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Clarification turns</span>
                  <span className="text-xs text-slate-500">
                    {summary.correlates.clarification_turn_count}
                  </span>
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-slate-100 p-4">
              <div className="text-sm font-semibold text-slate-800">By Target Type</div>
              <div className="mt-3">
                {renderBucketList(
                  summary.target_type_counts.slice(0, 6),
                  "No target breakdown yet.",
                  (bucket) => `${bucket.total} rows`
                )}
              </div>
            </div>
          </div>
        </>
      ) : null}
    </section>
  );
}
