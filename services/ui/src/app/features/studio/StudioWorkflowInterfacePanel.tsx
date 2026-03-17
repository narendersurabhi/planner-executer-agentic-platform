"use client";

import type {
  ComposerDraftNode,
  WorkflowBinding,
  WorkflowInputDefinition,
  WorkflowInterface,
  WorkflowOutputDefinition,
  WorkflowVariableDefinition,
} from "./types";

type StudioWorkflowInterfacePanelProps = {
  workflowInterface: WorkflowInterface;
  contextPathSuggestions: string[];
  visualChainNodes: ComposerDraftNode[];
  outputPathSuggestionsForNode: (node: ComposerDraftNode | undefined) => string[];
  onAddInput: () => void;
  onUpdateInput: (inputId: string, patch: Partial<WorkflowInputDefinition>) => void;
  onRemoveInput: (inputId: string) => void;
  onAddVariable: () => void;
  onUpdateVariable: (variableId: string, patch: Partial<WorkflowVariableDefinition>) => void;
  onRemoveVariable: (variableId: string) => void;
  onAddOutput: () => void;
  onUpdateOutput: (outputId: string, patch: Partial<WorkflowOutputDefinition>) => void;
  onRemoveOutput: (outputId: string) => void;
};

const workflowBindingMode = (
  binding: WorkflowBinding | null | undefined,
  fallback: "none" | "literal" | "context" | "memory" | "secret" = "none"
) => {
  if (!binding) {
    return fallback;
  }
  return binding.kind;
};

const workflowBindingForMode = (
  mode:
    | "none"
    | "literal"
    | "context"
    | "memory"
    | "secret"
    | "workflow_input"
    | "workflow_variable"
    | "step_output",
  current: WorkflowBinding | null | undefined
): WorkflowBinding | null => {
  if (mode === "none") {
    return null;
  }
  if (mode === "literal") {
    return { kind: "literal", value: current?.kind === "literal" ? current.value : "" };
  }
  if (mode === "context") {
    return { kind: "context", path: current?.kind === "context" ? current.path : "" };
  }
  if (mode === "memory") {
    return {
      kind: "memory",
      scope: current?.kind === "memory" ? current.scope : "job",
      name: current?.kind === "memory" ? current.name : "",
      ...(current?.kind === "memory" && current.key ? { key: current.key } : {}),
    };
  }
  if (mode === "secret") {
    return {
      kind: "secret",
      secretName: current?.kind === "secret" ? current.secretName : "",
    };
  }
  if (mode === "workflow_input") {
    return {
      kind: "workflow_input",
      inputKey: current?.kind === "workflow_input" ? current.inputKey : "",
    };
  }
  if (mode === "workflow_variable") {
    return {
      kind: "workflow_variable",
      variableKey: current?.kind === "workflow_variable" ? current.variableKey : "",
    };
  }
  return {
    kind: "step_output",
    sourceNodeId: current?.kind === "step_output" ? current.sourceNodeId : "",
    sourcePath: current?.kind === "step_output" ? current.sourcePath : "",
  };
};

export default function StudioWorkflowInterfacePanel({
  workflowInterface,
  contextPathSuggestions,
  visualChainNodes,
  outputPathSuggestionsForNode,
  onAddInput,
  onUpdateInput,
  onRemoveInput,
  onAddVariable,
  onUpdateVariable,
  onRemoveVariable,
  onAddOutput,
  onUpdateOutput,
  onRemoveOutput,
}: StudioWorkflowInterfacePanelProps) {
  return (
    <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Workflow Interface
          </div>
          <h2 className="mt-1 font-display text-2xl text-slate-900">Inputs, Variables, Outputs</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Define the workflow contract once, then bind nodes to workflow-level inputs and variables
            instead of depending on raw <code>context_json</code>.
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-6">
        <div>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Inputs
              </div>
              <div className="mt-1 text-xs text-slate-500">
                External values the workflow expects at run time.
              </div>
            </div>
            <button
              className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
              onClick={onAddInput}
            >
              Add Input
            </button>
          </div>
          <div className="mt-3 space-y-3">
            {workflowInterface.inputs.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                No workflow inputs yet.
              </div>
            ) : null}
            {workflowInterface.inputs.map((input) => {
              const bindingMode = workflowBindingMode(input.binding);
              const inputLiteralBinding: Extract<WorkflowBinding, { kind: "literal" }> | null =
                input.binding?.kind === "literal" ? input.binding : null;
              const inputContextBinding: Extract<WorkflowBinding, { kind: "context" }> | null =
                input.binding?.kind === "context" ? input.binding : null;
              const inputMemoryBinding: Extract<WorkflowBinding, { kind: "memory" }> | null =
                input.binding?.kind === "memory" ? input.binding : null;
              const inputSecretBinding: Extract<WorkflowBinding, { kind: "secret" }> | null =
                input.binding?.kind === "secret" ? input.binding : null;
              return (
                <div
                  key={`workflow-input-${input.id}`}
                  className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-900">
                      {input.label || input.key || "Untitled input"}
                    </div>
                    <button
                      className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                      onClick={() => onRemoveInput(input.id)}
                    >
                      Remove
                    </button>
                  </div>
                  <div className="mt-3 grid gap-3 lg:grid-cols-2">
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={input.key}
                      onChange={(event) => onUpdateInput(input.id, { key: event.target.value })}
                      placeholder="input key"
                    />
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={input.label}
                      onChange={(event) => onUpdateInput(input.id, { label: event.target.value })}
                      placeholder="label"
                    />
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={input.valueType}
                      onChange={(event) =>
                        onUpdateInput(input.id, {
                          valueType: event.target.value as WorkflowInputDefinition["valueType"],
                        })
                      }
                    >
                      <option value="string">string</option>
                      <option value="number">number</option>
                      <option value="boolean">boolean</option>
                      <option value="object">object</option>
                      <option value="array">array</option>
                    </select>
                    <label className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                      <input
                        type="checkbox"
                        checked={input.required}
                        onChange={(event) =>
                          onUpdateInput(input.id, { required: event.target.checked })
                        }
                      />
                      Required
                    </label>
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                      value={input.defaultValue || ""}
                      onChange={(event) =>
                        onUpdateInput(input.id, { defaultValue: event.target.value })
                      }
                      placeholder="default value"
                    />
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                      value={input.description || ""}
                      onChange={(event) =>
                        onUpdateInput(input.id, { description: event.target.value })
                      }
                      placeholder="description"
                    />
                  </div>

                  <div className="mt-3 grid gap-2">
                    <label className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Binding
                    </label>
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={bindingMode}
                      onChange={(event) =>
                        onUpdateInput(input.id, {
                          binding: workflowBindingForMode(
                            event.target.value as
                              | "none"
                              | "literal"
                              | "context"
                              | "memory"
                              | "secret",
                            input.binding
                          ),
                        })
                      }
                    >
                      <option value="none">No binding</option>
                      <option value="literal">Literal</option>
                      <option value="context">Context</option>
                      <option value="memory">Memory</option>
                      <option value="secret">Secret</option>
                    </select>

                    {inputLiteralBinding ? (
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                        value={inputLiteralBinding.value}
                        onChange={(event) =>
                          onUpdateInput(input.id, {
                            binding: { kind: "literal", value: event.target.value },
                          })
                        }
                        placeholder="literal fallback"
                      />
                    ) : null}

                    {inputContextBinding ? (
                      <>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          list={`workflow-input-context-${input.id}`}
                          value={inputContextBinding.path}
                          onChange={(event) =>
                          onUpdateInput(input.id, {
                            binding: { kind: "context", path: event.target.value },
                          })
                        }
                          placeholder="context path"
                        />
                        <datalist id={`workflow-input-context-${input.id}`}>
                          {contextPathSuggestions.map((path) => (
                            <option key={`workflow-input-context-${input.id}-${path}`} value={path} />
                          ))}
                        </datalist>
                      </>
                    ) : null}

                    {inputMemoryBinding ? (
                      <div className="grid gap-2 sm:grid-cols-3">
                        <select
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={inputMemoryBinding.scope}
                          onChange={(event) =>
                          onUpdateInput(input.id, {
                              binding: {
                                kind: "memory",
                                scope: event.target.value as "job" | "user" | "global",
                                name: inputMemoryBinding.name,
                                ...(inputMemoryBinding.key ? { key: inputMemoryBinding.key } : {}),
                              },
                            })
                          }
                        >
                          <option value="job">job</option>
                          <option value="user">user</option>
                          <option value="global">global</option>
                        </select>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={inputMemoryBinding.name}
                          onChange={(event) =>
                          onUpdateInput(input.id, {
                              binding: {
                                kind: "memory",
                                scope: inputMemoryBinding.scope,
                                name: event.target.value,
                                ...(inputMemoryBinding.key ? { key: inputMemoryBinding.key } : {}),
                              },
                            })
                          }
                          placeholder="memory name"
                        />
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={inputMemoryBinding.key || ""}
                          onChange={(event) =>
                          onUpdateInput(input.id, {
                              binding: {
                                kind: "memory",
                                scope: inputMemoryBinding.scope,
                                name: inputMemoryBinding.name,
                                key: event.target.value,
                              },
                            })
                          }
                          placeholder="optional key"
                        />
                      </div>
                    ) : null}

                    {inputSecretBinding ? (
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                        value={inputSecretBinding.secretName}
                        onChange={(event) =>
                          onUpdateInput(input.id, {
                            binding: { kind: "secret", secretName: event.target.value },
                          })
                        }
                        placeholder="ENV secret name"
                      />
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Variables
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Derived values resolved once and reused across node bindings.
              </div>
            </div>
            <button
              className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
              onClick={onAddVariable}
            >
              Add Variable
            </button>
          </div>
          <div className="mt-3 space-y-3">
            {workflowInterface.variables.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                No workflow variables yet.
              </div>
            ) : null}
            {workflowInterface.variables.map((variable) => {
              const bindingMode = workflowBindingMode(variable.binding, "literal");
              const variableLiteralBinding: Extract<WorkflowBinding, { kind: "literal" }> | null =
                variable.binding?.kind === "literal" ? variable.binding : null;
              const variableContextBinding: Extract<WorkflowBinding, { kind: "context" }> | null =
                variable.binding?.kind === "context" ? variable.binding : null;
              const variableMemoryBinding: Extract<WorkflowBinding, { kind: "memory" }> | null =
                variable.binding?.kind === "memory" ? variable.binding : null;
              const variableSecretBinding: Extract<WorkflowBinding, { kind: "secret" }> | null =
                variable.binding?.kind === "secret" ? variable.binding : null;
              const variableInputBinding: Extract<
                WorkflowBinding,
                { kind: "workflow_input" }
              > | null = variable.binding?.kind === "workflow_input" ? variable.binding : null;
              return (
                <div
                  key={`workflow-variable-${variable.id}`}
                  className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-900">
                      {variable.key || "Untitled variable"}
                    </div>
                    <button
                      className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                      onClick={() => onRemoveVariable(variable.id)}
                    >
                      Remove
                    </button>
                  </div>
                  <div className="mt-3 grid gap-3 lg:grid-cols-2">
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={variable.key}
                      onChange={(event) =>
                        onUpdateVariable(variable.id, { key: event.target.value })
                      }
                      placeholder="variable key"
                    />
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={variable.description || ""}
                      onChange={(event) =>
                        onUpdateVariable(variable.id, { description: event.target.value })
                      }
                      placeholder="description"
                    />
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                      value={bindingMode}
                      onChange={(event) =>
                        onUpdateVariable(variable.id, {
                          binding: workflowBindingForMode(
                            event.target.value as
                              | "literal"
                              | "context"
                              | "memory"
                              | "secret"
                              | "workflow_input",
                            variable.binding
                          ),
                        })
                      }
                    >
                      <option value="literal">Literal</option>
                      <option value="context">Context</option>
                      <option value="memory">Memory</option>
                      <option value="secret">Secret</option>
                      <option value="workflow_input">Workflow input</option>
                    </select>

                    {variableLiteralBinding ? (
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={variableLiteralBinding.value}
                        onChange={(event) =>
                          onUpdateVariable(variable.id, {
                            binding: { kind: "literal", value: event.target.value },
                          })
                        }
                        placeholder="literal value"
                      />
                    ) : null}

                    {variableContextBinding ? (
                      <>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                          list={`workflow-variable-context-${variable.id}`}
                          value={variableContextBinding.path}
                          onChange={(event) =>
                          onUpdateVariable(variable.id, {
                              binding: { kind: "context", path: event.target.value },
                            })
                          }
                          placeholder="context path"
                        />
                        <datalist id={`workflow-variable-context-${variable.id}`}>
                          {contextPathSuggestions.map((path) => (
                            <option
                              key={`workflow-variable-context-${variable.id}-${path}`}
                              value={path}
                            />
                          ))}
                        </datalist>
                      </>
                    ) : null}

                    {variableMemoryBinding ? (
                      <div className="grid gap-2 lg:col-span-2 sm:grid-cols-3">
                        <select
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={variableMemoryBinding.scope}
                          onChange={(event) =>
                          onUpdateVariable(variable.id, {
                              binding: {
                                kind: "memory",
                                scope: event.target.value as "job" | "user" | "global",
                                name: variableMemoryBinding.name,
                                ...(variableMemoryBinding.key
                                  ? { key: variableMemoryBinding.key }
                                  : {}),
                              },
                            })
                          }
                        >
                          <option value="job">job</option>
                          <option value="user">user</option>
                          <option value="global">global</option>
                        </select>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={variableMemoryBinding.name}
                          onChange={(event) =>
                          onUpdateVariable(variable.id, {
                              binding: {
                                kind: "memory",
                                scope: variableMemoryBinding.scope,
                                name: event.target.value,
                                ...(variableMemoryBinding.key
                                  ? { key: variableMemoryBinding.key }
                                  : {}),
                              },
                            })
                          }
                          placeholder="memory name"
                        />
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={variableMemoryBinding.key || ""}
                          onChange={(event) =>
                          onUpdateVariable(variable.id, {
                              binding: {
                                kind: "memory",
                                scope: variableMemoryBinding.scope,
                                name: variableMemoryBinding.name,
                                key: event.target.value,
                              },
                            })
                          }
                          placeholder="optional key"
                        />
                      </div>
                    ) : null}

                    {variableSecretBinding ? (
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={variableSecretBinding.secretName}
                        onChange={(event) =>
                          onUpdateVariable(variable.id, {
                            binding: { kind: "secret", secretName: event.target.value },
                          })
                        }
                        placeholder="ENV secret name"
                      />
                    ) : null}

                    {variableInputBinding ? (
                      <select
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={variableInputBinding.inputKey}
                        onChange={(event) =>
                          onUpdateVariable(variable.id, {
                            binding: { kind: "workflow_input", inputKey: event.target.value },
                          })
                        }
                      >
                        <option value="">Select workflow input</option>
                        {workflowInterface.inputs.map((input) => (
                          <option key={`workflow-variable-input-${input.id}`} value={input.key}>
                            {input.key || input.label || "Untitled input"}
                          </option>
                        ))}
                      </select>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Outputs
              </div>
              <div className="mt-1 text-xs text-slate-500">
                Public outputs you expect callers or downstream systems to consume.
              </div>
            </div>
            <button
              className="rounded-full border border-slate-300 px-3 py-1.5 text-[11px] font-semibold text-slate-700"
              onClick={onAddOutput}
            >
              Add Output
            </button>
          </div>
          <div className="mt-3 space-y-3">
            {workflowInterface.outputs.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                No workflow outputs yet.
              </div>
            ) : null}
            {workflowInterface.outputs.map((output) => {
              const bindingMode = workflowBindingMode(output.binding);
              const outputLiteralBinding: Extract<WorkflowBinding, { kind: "literal" }> | null =
                output.binding?.kind === "literal" ? output.binding : null;
              const outputContextBinding: Extract<WorkflowBinding, { kind: "context" }> | null =
                output.binding?.kind === "context" ? output.binding : null;
              const outputInputBinding: Extract<
                WorkflowBinding,
                { kind: "workflow_input" }
              > | null = output.binding?.kind === "workflow_input" ? output.binding : null;
              const outputVariableBinding: Extract<
                WorkflowBinding,
                { kind: "workflow_variable" }
              > | null =
                output.binding?.kind === "workflow_variable" ? output.binding : null;
              const outputStepBinding: Extract<WorkflowBinding, { kind: "step_output" }> | null =
                output.binding?.kind === "step_output" ? output.binding : null;
              const sourceNode = outputStepBinding
                ? visualChainNodes.find((node) => node.id === outputStepBinding.sourceNodeId)
                : undefined;
              const pathOptions = outputPathSuggestionsForNode(sourceNode);
              return (
                <div
                  key={`workflow-output-${output.id}`}
                  className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-900">
                      {output.label || output.key || "Untitled output"}
                    </div>
                    <button
                      className="rounded-full border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] text-rose-700"
                      onClick={() => onRemoveOutput(output.id)}
                    >
                      Remove
                    </button>
                  </div>
                  <div className="mt-3 grid gap-3 lg:grid-cols-2">
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={output.key}
                      onChange={(event) =>
                        onUpdateOutput(output.id, { key: event.target.value })
                      }
                      placeholder="output key"
                    />
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                      value={output.label}
                      onChange={(event) =>
                        onUpdateOutput(output.id, { label: event.target.value })
                      }
                      placeholder="label"
                    />
                    <input
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                      value={output.description || ""}
                      onChange={(event) =>
                        onUpdateOutput(output.id, { description: event.target.value })
                      }
                      placeholder="description"
                    />
                    <select
                      className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                      value={bindingMode}
                      onChange={(event) =>
                        onUpdateOutput(output.id, {
                          binding: workflowBindingForMode(
                            event.target.value as
                              | "none"
                              | "literal"
                              | "context"
                              | "workflow_input"
                              | "workflow_variable"
                              | "step_output",
                            output.binding
                          ),
                        })
                      }
                    >
                      <option value="none">No binding</option>
                      <option value="step_output">Step output</option>
                      <option value="workflow_input">Workflow input</option>
                      <option value="workflow_variable">Workflow variable</option>
                      <option value="context">Context</option>
                      <option value="literal">Literal</option>
                    </select>

                    {outputLiteralBinding ? (
                      <input
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={outputLiteralBinding.value}
                        onChange={(event) =>
                          onUpdateOutput(output.id, {
                            binding: { kind: "literal", value: event.target.value },
                          })
                        }
                        placeholder="literal value"
                      />
                    ) : null}

                    {outputContextBinding ? (
                      <>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                          list={`workflow-output-context-${output.id}`}
                          value={outputContextBinding.path}
                          onChange={(event) =>
                          onUpdateOutput(output.id, {
                              binding: { kind: "context", path: event.target.value },
                            })
                          }
                          placeholder="context path"
                        />
                        <datalist id={`workflow-output-context-${output.id}`}>
                          {contextPathSuggestions.map((path) => (
                            <option key={`workflow-output-context-${output.id}-${path}`} value={path} />
                          ))}
                        </datalist>
                      </>
                    ) : null}

                    {outputInputBinding ? (
                      <select
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={outputInputBinding.inputKey}
                        onChange={(event) =>
                          onUpdateOutput(output.id, {
                            binding: { kind: "workflow_input", inputKey: event.target.value },
                          })
                        }
                      >
                        <option value="">Select workflow input</option>
                        {workflowInterface.inputs.map((input) => (
                          <option key={`workflow-output-input-${input.id}`} value={input.key}>
                            {input.key || input.label || "Untitled input"}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {outputVariableBinding ? (
                      <select
                        className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 lg:col-span-2"
                        value={outputVariableBinding.variableKey}
                        onChange={(event) =>
                          onUpdateOutput(output.id, {
                            binding: {
                              kind: "workflow_variable",
                              variableKey: event.target.value,
                            },
                          })
                        }
                      >
                        <option value="">Select workflow variable</option>
                        {workflowInterface.variables.map((variable) => (
                          <option key={`workflow-output-variable-${variable.id}`} value={variable.key}>
                            {variable.key || "Untitled variable"}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {outputStepBinding ? (
                      <>
                        <select
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          value={outputStepBinding.sourceNodeId}
                          onChange={(event) =>
                          onUpdateOutput(output.id, {
                              binding: {
                                kind: "step_output",
                                sourceNodeId: event.target.value,
                                sourcePath: outputStepBinding.sourcePath,
                              },
                            })
                          }
                        >
                          <option value="">Select source step</option>
                          {visualChainNodes.map((node) => (
                            <option key={`workflow-output-node-${node.id}`} value={node.id}>
                              {node.taskName}
                            </option>
                          ))}
                        </select>
                        <input
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900"
                          list={`workflow-output-path-${output.id}`}
                          value={outputStepBinding.sourcePath}
                          onChange={(event) =>
                          onUpdateOutput(output.id, {
                              binding: {
                                kind: "step_output",
                                sourceNodeId: outputStepBinding.sourceNodeId,
                                sourcePath: event.target.value,
                              },
                            })
                          }
                          placeholder="source output path"
                        />
                        <datalist id={`workflow-output-path-${output.id}`}>
                          {pathOptions.map((path) => (
                            <option key={`workflow-output-path-${output.id}-${path}`} value={path} />
                          ))}
                        </datalist>
                      </>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
