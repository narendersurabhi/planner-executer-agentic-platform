"use client";

import type React from "react";

type ComposerInputBinding =
  | {
      kind: "step_output";
      sourceNodeId: string;
      sourcePath: string;
      defaultValue?: string;
    }
  | {
      kind: "literal";
      value: string;
    }
  | {
      kind: "context";
      path: string;
    }
  | {
      kind: "memory";
      scope: "job" | "global";
      name: string;
      key?: string;
    };

type ComposerDraftNode = {
  id: string;
  taskName: string;
  capabilityId: string;
  outputPath: string;
  inputBindings: Record<string, ComposerInputBinding>;
};

type NodeRequiredStatus = {
  field: string;
  status: "missing" | "from_chain" | "from_context" | "provided";
  detail: string;
  schemaType: string;
  schemaDescription: string;
};

type ComposerIssueFocus = {
  nodeId: string;
  field?: string;
};

type ComposerStepInspectorProps = {
  selectedDagNode: ComposerDraftNode | null;
  selectedDagNodeStatus: {
    requiredCount: number;
    requiredStatus: NodeRequiredStatus[];
  } | null;
  activeComposerIssueFocus: ComposerIssueFocus | null;
  inspectorBindingRefs: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  visualChainNodes: ComposerDraftNode[];
  outputPathSuggestionsForNode: (node: ComposerDraftNode | undefined) => string[];
  autoWireNodeBindings: (nodeId: string) => void;
  quickFixNodeBindings: (nodeId: string) => void;
  setSelectedDagNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  setVisualBindingMode: (
    nodeId: string,
    field: string,
    mode: "context" | "from" | "literal" | "memory"
  ) => void;
  clearVisualBinding: (nodeId: string, field: string) => void;
  updateVisualBindingSourceNode: (nodeId: string, field: string, sourceNodeId: string) => void;
  updateVisualBindingPath: (nodeId: string, field: string, sourcePath: string) => void;
  updateVisualBindingLiteral: (nodeId: string, field: string, value: string) => void;
  updateVisualBindingContextPath: (nodeId: string, field: string, path: string) => void;
  updateVisualBindingMemory: (
    nodeId: string,
    field: string,
    patch: { scope?: "job" | "global"; name?: string; key?: string }
  ) => void;
  setVisualBindingFromPrevious: (nodeId: string, field: string) => void;
};

export default function ComposerStepInspector({
  selectedDagNode,
  selectedDagNodeStatus,
  activeComposerIssueFocus,
  inspectorBindingRefs,
  visualChainNodes,
  outputPathSuggestionsForNode,
  autoWireNodeBindings,
  quickFixNodeBindings,
  setSelectedDagNodeId,
  setVisualBindingMode,
  clearVisualBinding,
  updateVisualBindingSourceNode,
  updateVisualBindingPath,
  updateVisualBindingLiteral,
  updateVisualBindingContextPath,
  updateVisualBindingMemory,
  setVisualBindingFromPrevious,
}: ComposerStepInspectorProps) {
  if (!selectedDagNode) {
    return (
      <div className="mt-2 rounded-lg border border-dashed border-slate-200 bg-white px-2 py-2 text-[11px] text-slate-500">
        Click a node in the canvas to edit required input mappings.
      </div>
    );
  }

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
            Node Inspector
          </div>
          <div className="text-xs font-semibold text-slate-800">{selectedDagNode.taskName}</div>
          <div className="text-[11px] text-slate-500">{selectedDagNode.capabilityId}</div>
        </div>
        <div className="flex items-center gap-2">
          <button
            className="rounded-md border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
            onClick={() => autoWireNodeBindings(selectedDagNode.id)}
            title="Map missing fields from the immediately previous step"
          >
            Auto-Wire
          </button>
          <button
            className="rounded-md border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
            onClick={() => quickFixNodeBindings(selectedDagNode.id)}
          >
            Quick Fix Missing
          </button>
          <button
            className="rounded-md border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
            onClick={() => setSelectedDagNodeId(null)}
          >
            Close
          </button>
        </div>
      </div>
      {selectedDagNodeStatus && selectedDagNodeStatus.requiredCount > 0 ? (
        <div className="mt-3 space-y-2">
          {selectedDagNodeStatus.requiredStatus.map((status) => {
            const binding = selectedDagNode.inputBindings[status.field];
            const bindingMode =
              binding?.kind === "step_output"
                ? "from"
                : binding?.kind === "literal"
                  ? "literal"
                  : binding?.kind === "memory"
                    ? "memory"
                    : "context";
            const sourceNodes = visualChainNodes.filter((candidate) => candidate.id !== selectedDagNode.id);
            const sourcePathListId = `inspector-${selectedDagNode.id}-${status.field}-source-path-options`;
            const selectedSourceNode =
              binding?.kind === "step_output"
                ? visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId) ||
                  sourceNodes[sourceNodes.length - 1]
                : sourceNodes[sourceNodes.length - 1];
            const sourcePathOptions = outputPathSuggestionsForNode(selectedSourceNode);
            return (
              <div
                key={`inspector-${selectedDagNode.id}-${status.field}`}
                ref={(element) => {
                  inspectorBindingRefs.current[`${selectedDagNode.id}::${status.field}`] = element;
                }}
                className={`rounded-md border px-2 py-2 ${
                  activeComposerIssueFocus?.nodeId === selectedDagNode.id &&
                  activeComposerIssueFocus?.field === status.field
                    ? "border-sky-300 bg-sky-50"
                    : "border-slate-200 bg-slate-50"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-[11px] font-semibold text-slate-700">{status.field}</div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] ${
                      status.status === "missing"
                        ? "bg-rose-100 text-rose-700"
                        : status.status === "from_chain"
                          ? "bg-sky-100 text-sky-700"
                          : status.status === "from_context"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-amber-100 text-amber-700"
                    }`}
                  >
                    {status.status}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500">{status.detail}</div>
                <div className="mt-1 flex items-center gap-2">
                  <label className="text-[11px] text-slate-600">Mode</label>
                  <select
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                    value={bindingMode}
                    onChange={(event) =>
                      setVisualBindingMode(
                        selectedDagNode.id,
                        status.field,
                        event.target.value as "context" | "from" | "literal" | "memory"
                      )
                    }
                  >
                    <option value="context">Context</option>
                    <option value="from" disabled={sourceNodes.length === 0}>
                      Step output
                    </option>
                    <option value="literal">Literal</option>
                    <option value="memory">Memory</option>
                  </select>
                  {status.status === "missing" && sourceNodes.length > 0 && bindingMode !== "from" ? (
                    <button
                      className="rounded-md border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
                      onClick={() => setVisualBindingFromPrevious(selectedDagNode.id, status.field)}
                    >
                      Wire from previous
                    </button>
                  ) : null}
                  <button
                    className="rounded-md border border-slate-300 px-2 py-1 text-[11px] text-slate-700"
                    onClick={() => clearVisualBinding(selectedDagNode.id, status.field)}
                  >
                    Clear
                  </button>
                </div>
                {bindingMode === "from" && binding?.kind === "step_output" ? (
                  <div className="mt-2 space-y-2">
                    <select
                      className="w-full rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                      value={binding.sourceNodeId}
                      onChange={(event) =>
                        updateVisualBindingSourceNode(
                          selectedDagNode.id,
                          status.field,
                          event.target.value
                        )
                      }
                    >
                      {sourceNodes.map((candidateNode) => (
                        <option
                          key={`inspector-source-${selectedDagNode.id}-${status.field}-${candidateNode.id}`}
                          value={candidateNode.id}
                        >
                          {candidateNode.taskName} ({candidateNode.capabilityId})
                        </option>
                      ))}
                    </select>
                    <input
                      className="w-full rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                      value={binding.sourcePath}
                      onChange={(event) =>
                        updateVisualBindingPath(selectedDagNode.id, status.field, event.target.value)
                      }
                      list={sourcePathListId}
                      placeholder="source output path"
                    />
                    <datalist id={sourcePathListId}>
                      {sourcePathOptions.map((option) => (
                        <option key={`${sourcePathListId}-${option}`} value={option} />
                      ))}
                    </datalist>
                  </div>
                ) : null}
                {bindingMode === "literal" ? (
                  <input
                    className="mt-2 w-full rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                    value={binding?.kind === "literal" ? binding.value : ""}
                    onChange={(event) =>
                      updateVisualBindingLiteral(selectedDagNode.id, status.field, event.target.value)
                    }
                    placeholder="literal value"
                  />
                ) : null}
                {bindingMode === "context" ? (
                  <input
                    className="mt-2 w-full rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                    value={binding?.kind === "context" ? binding.path : status.field}
                    onChange={(event) =>
                      updateVisualBindingContextPath(selectedDagNode.id, status.field, event.target.value)
                    }
                    placeholder="context path"
                  />
                ) : null}
                {bindingMode === "memory" ? (
                  <div className="mt-2 grid gap-2 sm:grid-cols-3">
                    <select
                      className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                      value={binding?.kind === "memory" ? binding.scope : "job"}
                      onChange={(event) =>
                        updateVisualBindingMemory(selectedDagNode.id, status.field, {
                          scope: event.target.value as "job" | "global",
                        })
                      }
                    >
                      <option value="job">job</option>
                      <option value="global">global</option>
                    </select>
                    <input
                      className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                      value={binding?.kind === "memory" ? binding.name : "task_outputs"}
                      onChange={(event) =>
                        updateVisualBindingMemory(selectedDagNode.id, status.field, {
                          name: event.target.value,
                        })
                      }
                      placeholder="memory name"
                    />
                    <input
                      className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-900"
                      value={binding?.kind === "memory" ? binding.key || "" : ""}
                      onChange={(event) =>
                        updateVisualBindingMemory(selectedDagNode.id, status.field, {
                          key: event.target.value,
                        })
                      }
                      placeholder="optional key"
                    />
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="mt-2 text-[11px] text-slate-500">No required inputs for this node.</div>
      )}
    </div>
  );
}
