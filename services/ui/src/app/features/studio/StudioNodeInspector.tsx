"use client";

import { useState } from "react";

import type {
  ComposerDraftNode,
  ComposerIssueFocus,
  ComposerInputBinding,
  StudioControlCase,
  StudioControlConfig,
  WorkflowInterface,
} from "./types";

type StudioInspectorField = {
  field: string;
  required: boolean;
  status: "missing" | "from_chain" | "from_context" | "provided";
  detail: string;
  schemaType: string;
  schemaDescription: string;
};

type StudioNodeInspectorProps = {
  selectedDagNode: ComposerDraftNode | null;
  selectedDagNodeStatus: {
    requiredCount: number;
  } | null;
  inputFields: StudioInspectorField[];
  activeComposerIssueFocus: ComposerIssueFocus | null;
  inspectorBindingRefs: React.MutableRefObject<Record<string, HTMLDivElement | null>>;
  visualChainNodes: ComposerDraftNode[];
  outputPathSuggestionsForNode: (node: ComposerDraftNode | undefined) => string[];
  contextPathSuggestions: string[];
  workflowInterface: WorkflowInterface;
  autoWireNodeBindings: (nodeId: string) => void;
  quickFixNodeBindings: (nodeId: string) => void;
  setSelectedDagNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  capabilityIdOptionsId?: string;
  onDeleteNode?: (nodeId: string) => void;
  updateNodeBasics: (
    nodeId: string,
    patch: Partial<Pick<ComposerDraftNode, "taskName" | "capabilityId" | "outputPath">>
  ) => void;
  setVisualBindingMode: (
    nodeId: string,
    field: string,
    mode:
      | "context"
      | "from"
      | "literal"
      | "memory"
      | "workflow_input"
      | "workflow_variable"
  ) => void;
  clearVisualBinding: (nodeId: string, field: string) => void;
  removeCustomInputField: (nodeId: string, field: string) => void;
  addCustomInputField: (nodeId: string, field: string) => void;
  updateVisualBindingSourceNode: (nodeId: string, field: string, sourceNodeId: string) => void;
  updateVisualBindingPath: (nodeId: string, field: string, sourcePath: string) => void;
  updateVisualBindingLiteral: (nodeId: string, field: string, value: string) => void;
  updateVisualBindingContextPath: (nodeId: string, field: string, path: string) => void;
  updateVisualBindingMemory: (
    nodeId: string,
    field: string,
    patch: { scope?: "job" | "user" | "global"; name?: string; key?: string }
  ) => void;
  updateVisualBindingWorkflowInput: (nodeId: string, field: string, inputKey: string) => void;
  updateVisualBindingWorkflowVariable: (
    nodeId: string,
    field: string,
    variableKey: string
  ) => void;
  setVisualBindingFromPrevious: (nodeId: string, field: string) => void;
  canInsertDeriveOutputPath: boolean;
  onInsertDeriveOutputPath: (nodeId: string) => void;
  addNodeOutput: (nodeId: string) => void;
  updateNodeOutput: (
    nodeId: string,
    outputId: string,
    patch: { name?: string; path?: string; description?: string }
  ) => void;
  removeNodeOutput: (nodeId: string, outputId: string) => void;
  addNodeVariable: (nodeId: string) => void;
  updateNodeVariable: (
    nodeId: string,
    variableId: string,
    patch: { key?: string; value?: string; description?: string }
  ) => void;
  removeNodeVariable: (nodeId: string, variableId: string) => void;
  updateNodeControlConfig: (nodeId: string, patch: Partial<StudioControlConfig>) => void;
  addSwitchCase: (nodeId: string) => void;
  updateSwitchCase: (
    nodeId: string,
    caseId: string,
    patch: Partial<Pick<StudioControlCase, "label" | "match">>
  ) => void;
  removeSwitchCase: (nodeId: string, caseId: string) => void;
};

const bindingModeForField = (binding: ComposerInputBinding | undefined) => {
  if (binding?.kind === "step_output") {
    return "from";
  }
  if (binding?.kind === "literal") {
    return "literal";
  }
  if (binding?.kind === "memory") {
    return "memory";
  }
  if (binding?.kind === "workflow_input") {
    return "workflow_input";
  }
  if (binding?.kind === "workflow_variable") {
    return "workflow_variable";
  }
  return "context";
};

export default function StudioNodeInspector({
  selectedDagNode,
  selectedDagNodeStatus,
  inputFields,
  activeComposerIssueFocus,
  inspectorBindingRefs,
  visualChainNodes,
  outputPathSuggestionsForNode,
  contextPathSuggestions,
  workflowInterface,
  autoWireNodeBindings,
  quickFixNodeBindings,
  setSelectedDagNodeId,
  capabilityIdOptionsId,
  onDeleteNode,
  updateNodeBasics,
  setVisualBindingMode,
  clearVisualBinding,
  removeCustomInputField,
  addCustomInputField,
  updateVisualBindingSourceNode,
  updateVisualBindingPath,
  updateVisualBindingLiteral,
  updateVisualBindingContextPath,
  updateVisualBindingMemory,
  updateVisualBindingWorkflowInput,
  updateVisualBindingWorkflowVariable,
  setVisualBindingFromPrevious,
  canInsertDeriveOutputPath,
  onInsertDeriveOutputPath,
  addNodeOutput,
  updateNodeOutput,
  removeNodeOutput,
  addNodeVariable,
  updateNodeVariable,
  removeNodeVariable,
  updateNodeControlConfig,
  addSwitchCase,
  updateSwitchCase,
  removeSwitchCase,
}: StudioNodeInspectorProps) {
  const [pendingInputField, setPendingInputField] = useState("");

  if (!selectedDagNode) {
    return (
      <section className="rounded-[28px] border border-dashed border-slate-200 bg-white px-4 py-5 text-sm text-slate-500 shadow-sm">
        Select a node to configure its inputs, outputs, and design variables.
      </section>
    );
  }

  const isControlNode = selectedDagNode.nodeKind === "control";
  const controlConfig = selectedDagNode.controlConfig || {
    expression: "",
    trueLabel: "",
    falseLabel: "",
    parallelMode: "fan_out" as const,
    switchCases: [],
  };

  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
            Node Inspector
          </div>
          <h2 className="mt-1 font-display text-2xl text-slate-900">{selectedDagNode.taskName}</h2>
          <div className="mt-1 text-xs text-slate-500">{selectedDagNode.capabilityId}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
            onClick={() => autoWireNodeBindings(selectedDagNode.id)}
          >
            Auto-Wire
          </button>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
            onClick={() => quickFixNodeBindings(selectedDagNode.id)}
          >
            Quick Fix
          </button>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700 disabled:opacity-40"
            onClick={() => onInsertDeriveOutputPath(selectedDagNode.id)}
            disabled={!canInsertDeriveOutputPath}
          >
            Insert Derive Path
          </button>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
            onClick={() => setSelectedDagNodeId(null)}
          >
            Close
          </button>
          {onDeleteNode ? (
            <button
              className="rounded-full border border-rose-200 bg-rose-50 px-3 py-1.5 text-[11px] font-semibold text-rose-700"
              onClick={() => onDeleteNode(selectedDagNode.id)}
            >
              Delete
            </button>
          ) : null}
        </div>
      </div>

      <div className="mt-4 grid gap-3">
        <label className="block">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            Task Name
          </div>
          <input
            className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            value={selectedDagNode.taskName}
            onChange={(event) =>
              updateNodeBasics(selectedDagNode.id, { taskName: event.target.value })
            }
          />
        </label>
        <label className="block">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
            {isControlNode ? "Control Node Id" : "Capability Id"}
          </div>
          <input
            className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            list={capabilityIdOptionsId}
            value={selectedDagNode.capabilityId}
            disabled={isControlNode}
            onChange={(event) =>
              updateNodeBasics(selectedDagNode.id, { capabilityId: event.target.value })
            }
          />
        </label>
        <label className="block">
          <div className="flex items-center justify-between gap-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Primary Output Path
            </div>
            <div className="text-xs text-slate-500">
              {selectedDagNode.outputs.length} extra output{selectedDagNode.outputs.length === 1 ? "" : "s"}
            </div>
          </div>
          <input
            className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            value={selectedDagNode.outputPath}
            onChange={(event) =>
              updateNodeBasics(selectedDagNode.id, { outputPath: event.target.value })
            }
          />
        </label>
      </div>

      {isControlNode ? (
        <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50/70 p-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-amber-700">
            Control Flow
          </div>
          <div className="mt-1 text-xs text-amber-900">
            These nodes are design-time Studio controls. They do not compile into backend plans yet.
          </div>
          <div className="mt-4 grid gap-3">
            <label className="block">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Expression
              </div>
              <input
                className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                value={controlConfig.expression || ""}
                onChange={(event) =>
                  updateNodeControlConfig(selectedDagNode.id, { expression: event.target.value })
                }
                placeholder={
                  selectedDagNode.controlKind === "parallel"
                    ? "Optional branch grouping note"
                    : "context.approval === true"
                }
              />
            </label>

            {selectedDagNode.controlKind === "if_else" ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="block">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    True Label
                  </div>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                    value={controlConfig.trueLabel || ""}
                    onChange={(event) =>
                      updateNodeControlConfig(selectedDagNode.id, { trueLabel: event.target.value })
                    }
                    placeholder="Approved"
                  />
                </label>
                <label className="block">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    False Label
                  </div>
                  <input
                    className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                    value={controlConfig.falseLabel || ""}
                    onChange={(event) =>
                      updateNodeControlConfig(selectedDagNode.id, { falseLabel: event.target.value })
                    }
                    placeholder="Rejected"
                  />
                </label>
              </div>
            ) : null}

            {selectedDagNode.controlKind === "parallel" ? (
              <label className="block">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  Parallel Mode
                </div>
                <select
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                  value={controlConfig.parallelMode || "fan_out"}
                  onChange={(event) =>
                    updateNodeControlConfig(selectedDagNode.id, {
                      parallelMode: event.target.value as "fan_out" | "fan_in",
                    })
                  }
                >
                  <option value="fan_out">Fan Out</option>
                  <option value="fan_in">Fan In</option>
                </select>
              </label>
            ) : null}

            {selectedDagNode.controlKind === "switch" ? (
              <div className="rounded-2xl border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      Cases
                    </div>
                    <div className="mt-1 text-xs text-slate-500">Add labels and match values for each route.</div>
                  </div>
                  <button
                    className="rounded-full border border-slate-300 px-3 py-1 text-[11px] font-semibold text-slate-700"
                    onClick={() => addSwitchCase(selectedDagNode.id)}
                  >
                    Add Case
                  </button>
                </div>
                <div className="mt-3 space-y-3">
                  {(controlConfig.switchCases || []).map((item, index) => (
                    <div key={item.id} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]">
                        <label className="block">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                            Case Label
                          </div>
                          <input
                            className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                            value={item.label}
                            onChange={(event) =>
                              updateSwitchCase(selectedDagNode.id, item.id, { label: event.target.value })
                            }
                            placeholder={`Case ${index + 1}`}
                          />
                        </label>
                        <label className="block">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                            Match Value
                          </div>
                          <input
                            className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400"
                            value={item.match}
                            onChange={(event) =>
                              updateSwitchCase(selectedDagNode.id, item.id, { match: event.target.value })
                            }
                            placeholder="approved"
                          />
                        </label>
                        <div className="flex items-end">
                          <button
                            className="rounded-full border border-rose-200 bg-rose-50 px-3 py-2 text-[11px] font-semibold text-rose-700"
                            onClick={() => removeSwitchCase(selectedDagNode.id, item.id)}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="mt-6 border-t border-slate-100 pt-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Inputs
            </div>
            <div className="mt-1 text-xs text-slate-500">
              Required fields are listed automatically. Each field can point at context data, step outputs, literals, or memory.
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <div className="rounded-full bg-slate-100 px-3 py-1 text-slate-700">
              required {selectedDagNodeStatus?.requiredCount || 0}
            </div>
            <div className="rounded-full bg-emerald-50 px-3 py-1 text-emerald-700">
              context keys {contextPathSuggestions.length}
            </div>
          </div>
        </div>

        <div className="mt-3 flex gap-2">
          <input
            className="flex-1 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            value={pendingInputField}
            onChange={(event) => setPendingInputField(event.target.value)}
            placeholder="Add custom input field"
          />
          <button
            className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700"
            onClick={() => {
              const field = pendingInputField.trim();
              if (!field) {
                return;
              }
              addCustomInputField(selectedDagNode.id, field);
              setPendingInputField("");
            }}
          >
            Add
          </button>
        </div>

        <div className="mt-3 space-y-3">
          {inputFields.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
              No inputs configured for this node.
            </div>
          ) : null}
          {inputFields.map((status) => {
            const binding = selectedDagNode.inputBindings[status.field];
            const bindingMode = bindingModeForField(binding);
            const sourceNodes = visualChainNodes.filter((candidate) => candidate.id !== selectedDagNode.id);
            const sourcePathListId = `studio-inspector-${selectedDagNode.id}-${status.field}-source-path-options`;
            const contextPathListId = `studio-inspector-${selectedDagNode.id}-${status.field}-context-path-options`;
            const selectedSourceNode =
              binding?.kind === "step_output"
                ? visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId) ||
                  sourceNodes[sourceNodes.length - 1]
                : sourceNodes[sourceNodes.length - 1];
            const sourcePathOptions = outputPathSuggestionsForNode(selectedSourceNode);

            return (
              <div
                key={`studio-inspector-${selectedDagNode.id}-${status.field}`}
                ref={(element) => {
                  inspectorBindingRefs.current[`${selectedDagNode.id}::${status.field}`] = element;
                }}
                className={`rounded-2xl border px-3 py-3 ${
                  activeComposerIssueFocus?.nodeId === selectedDagNode.id &&
                  activeComposerIssueFocus?.field === status.field
                    ? "border-sky-300 bg-sky-50"
                    : "border-slate-200 bg-slate-50"
                }`}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-sm font-semibold text-slate-800">{status.field}</div>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] uppercase tracking-[0.14em] ${
                          status.required
                            ? "bg-slate-900 text-white"
                            : "border border-slate-300 bg-white text-slate-600"
                        }`}
                      >
                        {status.required ? "required" : "custom"}
                      </span>
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
                    <div className="mt-1 text-[11px] text-slate-500">
                      {status.schemaType}
                      {status.schemaDescription ? ` • ${status.schemaDescription}` : ""}
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500">{status.detail}</div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {status.status === "missing" && sourceNodes.length > 0 && bindingMode !== "from" ? (
                      <button
                        className="rounded-full border border-slate-300 px-2.5 py-1 text-[11px] text-slate-700"
                        onClick={() => setVisualBindingFromPrevious(selectedDagNode.id, status.field)}
                      >
                        Wire Prev
                      </button>
                    ) : null}
                    <button
                      className="rounded-full border border-slate-300 px-2.5 py-1 text-[11px] text-slate-700"
                      onClick={() => clearVisualBinding(selectedDagNode.id, status.field)}
                    >
                      Clear
                    </button>
                    {!status.required ? (
                      <button
                        className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                        onClick={() => removeCustomInputField(selectedDagNode.id, status.field)}
                      >
                        Remove
                      </button>
                    ) : null}
                  </div>
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <label className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Mapping
                  </label>
                  <select
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                    value={bindingMode}
                    onChange={(event) =>
                      setVisualBindingMode(
                        selectedDagNode.id,
                        status.field,
                        event.target.value as
                          | "context"
                          | "from"
                          | "literal"
                          | "memory"
                          | "workflow_input"
                          | "workflow_variable"
                      )
                    }
                  >
                    <option value="context">Context data</option>
                    <option value="from" disabled={sourceNodes.length === 0}>
                      Step output
                    </option>
                    <option value="literal">Literal value</option>
                    <option value="memory">Memory</option>
                    <option value="workflow_input">Workflow input</option>
                    <option value="workflow_variable">Workflow variable</option>
                  </select>
                </div>

                {bindingMode === "from" && binding?.kind === "step_output" ? (
                  <div className="mt-3 grid gap-2">
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={binding.sourceNodeId}
                      onChange={(event) =>
                        updateVisualBindingSourceNode(selectedDagNode.id, status.field, event.target.value)
                      }
                    >
                      {sourceNodes.map((candidateNode) => (
                        <option
                          key={`studio-inspector-source-${selectedDagNode.id}-${status.field}-${candidateNode.id}`}
                          value={candidateNode.id}
                        >
                          {candidateNode.taskName} ({candidateNode.capabilityId})
                        </option>
                      ))}
                    </select>
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
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
                    className="mt-3 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                    value={binding?.kind === "literal" ? binding.value : ""}
                    onChange={(event) =>
                      updateVisualBindingLiteral(selectedDagNode.id, status.field, event.target.value)
                    }
                    placeholder="literal value"
                  />
                ) : null}

                {bindingMode === "context" ? (
                  <div className="mt-3 grid gap-2">
                    <input
                      className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      list={contextPathListId}
                      value={binding?.kind === "context" ? binding.path : status.field}
                      onChange={(event) =>
                        updateVisualBindingContextPath(selectedDagNode.id, status.field, event.target.value)
                      }
                      placeholder="context path"
                    />
                    <datalist id={contextPathListId}>
                      {contextPathSuggestions.map((option) => (
                        <option key={`${contextPathListId}-${option}`} value={option} />
                      ))}
                    </datalist>
                    <div className="text-[11px] text-slate-500">
                      Select a value from <code>context_json</code> or type a custom path.
                    </div>
                  </div>
                ) : null}

                {bindingMode === "memory" ? (
                  <div className="mt-3 grid gap-2">
                    <div className="grid gap-2 sm:grid-cols-3">
                      <select
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                        value={binding?.kind === "memory" ? binding.scope : "job"}
                        onChange={(event) =>
                          updateVisualBindingMemory(selectedDagNode.id, status.field, {
                            scope: event.target.value as "job" | "user" | "global",
                          })
                        }
                      >
                        <option value="job">job</option>
                        <option value="user">user</option>
                        <option value="global">global</option>
                      </select>
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                        value={binding?.kind === "memory" ? binding.name : "task_outputs"}
                        onChange={(event) =>
                          updateVisualBindingMemory(selectedDagNode.id, status.field, {
                            name: event.target.value,
                          })
                        }
                        placeholder="memory name"
                      />
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                        value={binding?.kind === "memory" ? binding.key || "" : ""}
                        onChange={(event) =>
                          updateVisualBindingMemory(selectedDagNode.id, status.field, {
                            key: event.target.value,
                          })
                        }
                        placeholder="optional key"
                      />
                    </div>
                  </div>
                ) : null}

                {bindingMode === "workflow_input" ? (
                  <div className="mt-3 grid gap-2">
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={binding?.kind === "workflow_input" ? binding.inputKey : ""}
                      onChange={(event) =>
                        updateVisualBindingWorkflowInput(
                          selectedDagNode.id,
                          status.field,
                          event.target.value
                        )
                      }
                    >
                      <option value="">Select workflow input</option>
                      {workflowInterface.inputs.map((input) => (
                        <option key={`workflow-input-${input.id}`} value={input.key}>
                          {input.key || input.label || "Untitled input"}
                        </option>
                      ))}
                    </select>
                    <div className="text-[11px] text-slate-500">
                      Bind this field to a workflow-level input.
                    </div>
                  </div>
                ) : null}

                {bindingMode === "workflow_variable" ? (
                  <div className="mt-3 grid gap-2">
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={binding?.kind === "workflow_variable" ? binding.variableKey : ""}
                      onChange={(event) =>
                        updateVisualBindingWorkflowVariable(
                          selectedDagNode.id,
                          status.field,
                          event.target.value
                        )
                      }
                    >
                      <option value="">Select workflow variable</option>
                      {workflowInterface.variables.map((variable) => (
                        <option key={`workflow-variable-${variable.id}`} value={variable.key}>
                          {variable.key || "Untitled variable"}
                        </option>
                      ))}
                    </select>
                    <div className="text-[11px] text-slate-500">
                      Bind this field to a workflow-level variable.
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-6 border-t border-slate-100 pt-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Outputs
            </div>
            <div className="mt-1 text-xs text-slate-500">
              Model the fields other nodes should be able to reference from this step.
            </div>
          </div>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
            onClick={() => addNodeOutput(selectedDagNode.id)}
          >
            Add Output
          </button>
        </div>

        <div className="mt-3 space-y-3">
          {selectedDagNode.outputs.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
              No extra outputs defined. Downstream nodes can still use the primary output path above.
            </div>
          ) : null}
          {selectedDagNode.outputs.map((output) => (
            <div
              key={`studio-node-output-${selectedDagNode.id}-${output.id}`}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3"
            >
              <div className="grid gap-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-slate-800">Named Output</div>
                  <button
                    className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                    onClick={() => removeNodeOutput(selectedDagNode.id, output.id)}
                  >
                    Remove
                  </button>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  <input
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                    value={output.name}
                    onChange={(event) =>
                      updateNodeOutput(selectedDagNode.id, output.id, { name: event.target.value })
                    }
                    placeholder="output name"
                  />
                  <input
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                    value={output.path}
                    onChange={(event) =>
                      updateNodeOutput(selectedDagNode.id, output.id, { path: event.target.value })
                    }
                    placeholder="output path used in references"
                  />
                </div>
                <input
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                  value={output.description || ""}
                  onChange={(event) =>
                    updateNodeOutput(selectedDagNode.id, output.id, {
                      description: event.target.value,
                    })
                  }
                  placeholder="optional description"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6 border-t border-slate-100 pt-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Variables
            </div>
            <div className="mt-1 text-xs text-slate-500">
              Design-time notes and aliases for this node. These are kept in the Studio draft only.
            </div>
          </div>
          <button
            className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
            onClick={() => addNodeVariable(selectedDagNode.id)}
          >
            Add Variable
          </button>
        </div>

        <div className="mt-3 space-y-3">
          {selectedDagNode.variables.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
              No variables yet. Add names and values to capture local workflow intent while designing.
            </div>
          ) : null}
          {selectedDagNode.variables.map((variable) => (
            <div
              key={`studio-node-variable-${selectedDagNode.id}-${variable.id}`}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-semibold text-slate-800">Variable</div>
                <button
                  className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                  onClick={() => removeNodeVariable(selectedDagNode.id, variable.id)}
                >
                  Remove
                </button>
              </div>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <input
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                  value={variable.key}
                  onChange={(event) =>
                    updateNodeVariable(selectedDagNode.id, variable.id, { key: event.target.value })
                  }
                  placeholder="variable name"
                />
                <input
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                  value={variable.value}
                  onChange={(event) =>
                    updateNodeVariable(selectedDagNode.id, variable.id, { value: event.target.value })
                  }
                  placeholder="variable value"
                />
              </div>
              <input
                className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                value={variable.description || ""}
                onChange={(event) =>
                  updateNodeVariable(selectedDagNode.id, variable.id, {
                    description: event.target.value,
                  })
                }
                placeholder="optional description"
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
