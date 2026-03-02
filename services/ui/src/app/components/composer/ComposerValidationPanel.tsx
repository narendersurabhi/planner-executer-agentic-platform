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
    ? "text-slate-700 font-semibold"
    : isValid
      ? "text-emerald-700 font-semibold"
      : "text-rose-700 font-semibold";

  return (
    <div className="mt-2 rounded-lg border border-slate-200 bg-white px-2 py-2 text-[11px]">
      <div className="flex items-center justify-between gap-2">
        <div className={statusClass}>{statusText}</div>
        <div className="text-slate-500">{formatTimestamp(preflightResult?.checkedAt)}</div>
      </div>
      {compileLoading ? <div className="mt-2 text-slate-500">Compiling draft...</div> : null}
      {issues.length > 0 ? (
        <div className="mt-2 space-y-1">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.15em] text-slate-500">
            <span>Issues</span>
            <span className="rounded-full bg-rose-100 px-1.5 py-0.5 text-[10px] text-rose-700">
              errors: {errorCount}
            </span>
            <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] text-amber-700">
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
                ? "text-amber-700 border-amber-200 hover:bg-amber-50"
                : "text-rose-700 border-rose-200 hover:bg-rose-50";
            return (
              <button
                key={`composer-issue-${idx}`}
                type="button"
                className={`w-full rounded-md border px-2 py-1 text-left transition ${
                  isActive ? "ring-2 ring-sky-100" : ""
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
        <div className="mt-2 text-slate-500">
          {needsValidation ? "No issues detected." : "Add one or more chain steps to enable validation."}
        </div>
      )}
    </div>
  );
}

