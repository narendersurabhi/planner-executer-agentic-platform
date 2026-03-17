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
    <section className="rounded-[28px] border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            Compile Preview
          </div>
          <h3 className="mt-1 font-display text-2xl text-slate-900">Plan Output</h3>
        </div>
        <button
          className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
          onClick={onCompile}
          disabled={compileLoading}
        >
          {compileLoading ? "Compiling..." : "Compile Draft"}
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.16em]">
        <span
          className={`rounded-full px-3 py-1 ${
            hasPlan ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-600"
          }`}
        >
          {hasPlan ? "Executable plan" : "Draft only"}
        </span>
        <span className="rounded-full bg-rose-100 px-3 py-1 text-rose-700">errors {errorCount}</span>
        <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-700">
          warnings {warningCount}
        </span>
      </div>

      <div className="mt-3 text-xs text-slate-500">
        Last checked: {formatTimestamp(preflightResult?.checkedAt)}
      </div>

      <details className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-3" open={hasPlan}>
        <summary className="cursor-pointer text-sm font-semibold text-slate-800">
          {hasPlan ? "Compiled plan JSON" : "Compile request preview"}
        </summary>
        <pre className="mt-3 max-h-[320px] overflow-auto rounded-2xl bg-slate-950 p-4 text-[11px] leading-5 text-slate-100">
          {JSON.stringify(hasPlan ? compileResult?.plan : draftPayloadPreview, null, 2)}
        </pre>
      </details>

      {issues.length > 0 ? (
        <details className="mt-4 rounded-2xl border border-slate-200 bg-white p-3">
          <summary className="cursor-pointer text-sm font-semibold text-slate-800">
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
