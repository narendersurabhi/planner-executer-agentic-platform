"use client";

type ComposerValidationIssue = {
  severity: "error" | "warning";
  source: "local" | "compile" | "preflight";
  code: string;
  message: string;
  field?: string;
  nodeId?: string;
};

type ComposerIssueFocus = {
  nodeId: string;
  field?: string;
};

type ChainPreflightResult = {
  valid: boolean;
  localErrors: string[];
  serverErrors: Record<string, string>;
  checkedAt: string;
};

type ComposerValidationPanelProps = {
  preflightResult: ChainPreflightResult | null;
  compileLoading: boolean;
  issues: ComposerValidationIssue[];
  needsValidation: boolean;
  onIssueClick: (issue: ComposerValidationIssue) => void;
  activeIssue: ComposerIssueFocus | null;
  formatTimestamp: (value?: string) => string;
};

export default function ComposerValidationPanel({
  preflightResult,
  compileLoading,
  issues,
  needsValidation,
  onIssueClick,
  activeIssue,
  formatTimestamp,
}: ComposerValidationPanelProps) {
  const errorCount = issues.filter((issue) => issue.severity === "error").length;
  const warningCount = issues.filter((issue) => issue.severity === "warning").length;
  const isValid = Boolean(preflightResult?.valid) && errorCount === 0;
  const statusText = needsValidation
    ? preflightResult
      ? isValid
        ? "Compile + Preflight OK"
        : "Compile/Preflight Issues Found"
      : "Validation required"
    : "No chain validation required";
  const statusClass = !needsValidation
    ? "text-slate-200 font-semibold"
    : isValid
      ? "text-emerald-300 font-semibold"
      : "text-rose-300 font-semibold";

  return (
    <div className="mt-3 rounded-[24px] border border-white/10 bg-white/[0.04] px-3 py-3 text-[11px] text-slate-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
      <div className="flex items-center justify-between gap-2">
        <div className={statusClass}>{statusText}</div>
        <div className="text-slate-400">{formatTimestamp(preflightResult?.checkedAt)}</div>
      </div>
      {compileLoading ? <div className="mt-2 text-slate-400">Compiling draft...</div> : null}
      {issues.length > 0 ? (
        <div className="mt-2 space-y-1">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.15em] text-slate-400">
            <span>Issues</span>
            <span className="rounded-full border border-rose-300/25 bg-rose-400/12 px-1.5 py-0.5 text-[10px] text-rose-200">
              errors: {errorCount}
            </span>
            <span className="rounded-full border border-amber-300/25 bg-amber-400/12 px-1.5 py-0.5 text-[10px] text-amber-200">
              warnings: {warningCount}
            </span>
          </div>
          {issues.map((issue, idx) => {
            const isActive =
              activeIssue &&
              issue.nodeId === activeIssue.nodeId &&
              (!activeIssue.field || !issue.field || issue.field === activeIssue.field);
            const issueClass =
              issue.severity === "warning"
                ? "text-amber-100 border-amber-300/25 bg-amber-400/10 hover:bg-amber-400/14"
                : "text-rose-100 border-rose-300/25 bg-rose-400/10 hover:bg-rose-400/14";
            return (
              <button
                key={`composer-issue-${idx}`}
                type="button"
                className={`w-full rounded-md border px-2 py-1 text-left transition ${
                  isActive ? "ring-2 ring-sky-300/30" : ""
                } ${issueClass}`}
                onClick={() => onIssueClick(issue)}
                title={issue.nodeId ? "Focus node in DAG canvas" : "Issue detail"}
              >
                • [{issue.source}] {issue.code}
                {issue.nodeId ? ` (${issue.nodeId})` : ""}
                {issue.field ? ` ${issue.field}:` : ":"} {issue.message}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="mt-2 text-slate-400">
          {needsValidation ? "No issues detected." : "Add one or more chain steps to enable validation."}
        </div>
      )}
    </div>
  );
}
