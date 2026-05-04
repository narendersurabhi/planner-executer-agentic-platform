"use client";

import type {
  ChainPreflightResult,
  ComposerCompileResponse,
  ComposerValidationIssue,
} from "./types";
import { formatTimestamp } from "./utils";

type StudioCompilePanelProps = {
  compileLoading: boolean;
  compileResult: ComposerCompileResponse | null;
  preflightResult: ChainPreflightResult | null;
  issues: ComposerValidationIssue[];
  draftPayloadPreview: Record<string, unknown>;
  onCompile: () => void;
};

const compilePanelClassName =
  "h-full px-3 py-3 text-slate-100 [&_.border-slate-200]:border-white/10 [&_.border-amber-200]:border-amber-300/25 [&_.border-rose-200]:border-rose-300/25 [&_.bg-slate-50]:bg-white/[0.04] [&_.bg-white]:bg-white/[0.05] [&_.bg-amber-50]:bg-amber-400/10 [&_.bg-rose-50]:bg-rose-400/10 [&_.text-slate-900]:text-white [&_.text-slate-800]:text-slate-100 [&_.text-slate-700]:text-slate-200 [&_.text-slate-600]:text-slate-300/82 [&_.text-slate-500]:text-slate-400 [&_.text-amber-800]:text-amber-100 [&_.text-rose-800]:text-rose-100 [&_details]:border [&_details]:border-white/10 [&_details]:bg-white/[0.04] [&_summary]:text-slate-100";

export default function StudioCompilePanel({
  compileLoading,
  compileResult,
  preflightResult,
  issues,
  draftPayloadPreview,
  onCompile,
}: StudioCompilePanelProps) {
  const errorCount = issues.filter((issue) => issue.severity === "error").length;
  const warningCount = issues.filter((issue) => issue.severity === "warning").length;
  const hasPlan = Boolean(compileResult?.plan);

  return (
    <section className={compilePanelClassName}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-sky-100/68">
            Workflow Readiness Check
          </div>
          <h3 className="mt-1 text-[22px] font-semibold tracking-[-0.03em] text-white">Plan Preview</h3>
        </div>
        <button
          className="rounded-full border border-sky-300/30 bg-sky-400/14 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-50 transition hover:border-sky-200/50 hover:bg-sky-400/18 disabled:cursor-not-allowed disabled:opacity-60"
          onClick={onCompile}
          disabled={compileLoading}
        >
          {compileLoading ? "Checking..." : "Check Workflow"}
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.16em]">
        <span
          className={`rounded-full border px-3 py-1 ${
            hasPlan
              ? "border-emerald-300/25 bg-emerald-400/12 text-emerald-200"
              : "border-white/10 bg-white/[0.05] text-slate-200"
          }`}
        >
          {hasPlan ? "Executable plan" : "Draft only"}
        </span>
        <span className="rounded-full border border-rose-300/25 bg-rose-400/12 px-3 py-1 text-rose-200">
          errors {errorCount}
        </span>
        <span className="rounded-full border border-amber-300/25 bg-amber-400/12 px-3 py-1 text-amber-200">
          warnings {warningCount}
        </span>
      </div>

      <div className="mt-3 text-xs text-slate-400">
        Last checked: {formatTimestamp(preflightResult?.checkedAt)}
      </div>

      <details className="mt-3 rounded-[18px] p-3" open={hasPlan}>
        <summary className="cursor-pointer text-sm font-semibold">
          {hasPlan ? "Compiled plan JSON" : "Compile request preview"}
        </summary>
        <pre className="mt-3 max-h-[320px] overflow-auto rounded-2xl bg-slate-950 p-4 text-[11px] leading-5 text-slate-100">
          {JSON.stringify(hasPlan ? compileResult?.plan : draftPayloadPreview, null, 2)}
        </pre>
      </details>

      {issues.length > 0 ? (
        <details className="mt-3 rounded-[18px] p-3">
          <summary className="cursor-pointer text-sm font-semibold">
            Diagnostics
          </summary>
          <div className="mt-3 space-y-2">
            {issues.map((issue, index) => (
              <div
                key={`studio-compile-issue-${index}`}
                className={`rounded-xl border px-3 py-2 text-sm ${
                  issue.severity === "warning"
                    ? "border-amber-200 bg-amber-50 text-amber-800"
                    : "border-rose-200 bg-rose-50 text-rose-800"
                }`}
              >
                <div className="font-semibold">
                  [{issue.source}] {issue.code}
                </div>
                <div className="mt-1">{issue.message}</div>
              </div>
            ))}
          </div>
        </details>
      ) : null}
    </section>
  );
}
