# Agentic Workflow Studio User Guide

This guide is for people using the product through the UI. It explains when to use each screen, how to run workflows, and the common playbooks that map to the current capabilities in the repo.

## What the Product Does

Agentic Workflow Studio gives you four main ways to get work done:

- `Compose`: create a structured job from a goal and context.
- `Chat`: stay conversational until a direct tool call or a workflow is actually needed.
- `Workflow Studio`: author explicit DAG workflows, save them, publish versions, and run them.
- `Memory`: manage user-scoped memory such as profile data, preferences, and reusable facts.

At runtime, work flows through the platform as:

1. You define a goal or workflow.
2. The API creates a job or workflow run.
3. Planner and worker services execute the workflow when needed.
4. Results, artifacts, and run history are available in the UI.

## Core Concepts

- `Job`: one unit of work submitted from Compose, Chat, or a workflow run.
- `Capability`: a named execution unit such as `document.spec.generate` or `github.repo.list`.
- `Tool`: the concrete runtime implementation behind a capability or direct call.
- `Workflow definition`: a saved Studio draft.
- `Workflow version`: a published, runnable snapshot of a workflow definition.
- `Workflow run`: an execution record created when you run a published workflow or invoke a trigger.
- `Memory`: scoped data available to workflows and chat, such as `user_profile` and `semantic_memory`.

## Access the UI

For local Kubernetes access, use the commands in [k8s-port-forward.md](k8s-port-forward.md).

Typical local access:

- UI: `http://localhost:8510`
- API direct access: `http://localhost:18000`

## Home Screen

The home screen is the entry point. It contains capsules for:

- `Compose`
- `Chat`
- `Workflow Studio`
- `Memory`

Use it when you want to choose a working mode instead of being dropped directly into one.

## Compose

Use Compose when:

- you already know the goal
- you want to provide structured context
- you want the planner to generate the workflow for you

Main actions:

- `Analyze Intent`: checks how the goal will likely decompose.
- `Submit Job`: creates a normal planner-led job.

Recommended Compose flow:

1. Open `Compose`.
2. Enter a clear goal.
3. Add or edit `Context JSON`.
4. Run `Analyze Intent` if the goal is complex.
5. Submit the job.
6. Watch the job list and task results in the workspace.

Use Compose instead of Studio when you care more about outcome than explicit DAG design.

## Chat

Use Chat when:

- you want a conversational starting point
- you are not yet sure whether you need a workflow
- you want the system to stay lightweight unless execution is required

Chat can currently resolve a turn into one of these modes:

- `respond`: normal conversation, no tool or workflow
- `tool_call`: direct safe read-only capability execution
- `ask_clarification`: the request likely needs execution, but more detail is needed
- `submit_job`: create a normal job and let the platform run it

Practical notes:

- Chat only uses direct tools for bounded, safe, usually read-only actions.
- If a request needs a multi-step workflow, chat creates a job instead of improvising a hidden workflow.
- If you enable context attachment, Compose context is sent along with the chat turn.

## Workflow Studio

Use Workflow Studio when:

- you want full control over the workflow graph
- you want repeatable, versioned automation
- you need explicit branching, joins, memory bindings, or reusable runs

### Studio Layout

- Left: `Capability Palette`
- Center: graph canvas
- Right: `Node Inspector`, workflow interface, compile preview, and workflow library

### Main Studio Actions

- `New Draft`
- `Save Draft`
- `Publish Version`
- `Run Workflow`
- `Compile Preview`

### Typical Studio Flow

1. Open `Workflow Studio`.
2. Set `Memory User ID` if the workflow should use user-scoped memory.
3. Add capability nodes from the palette.
4. Connect nodes on the canvas.
5. Configure inputs in the `Node Inspector`.
6. Define workflow-level inputs, variables, and outputs in the workflow interface panel.
7. Run `Compile Preview`.
8. Fix any compile or preflight issues.
9. `Save Draft`.
10. `Publish Version`.
11. `Run Workflow` or create a manual trigger.

### Node Inspector

The node inspector lets you:

- edit node name and capability
- configure required inputs
- map inputs from:
  - literal values
  - context data
  - step outputs
  - memory
- define node outputs
- define node-local variables

### Workflow Interface

Workflow-level inputs, variables, and outputs are useful when you want a workflow to behave like a reusable component instead of a one-off graph.

Use workflow-level inputs for:

- runtime values supplied at run time
- reusable values such as repo name, report topic, or output path prefix

Use workflow-level variables for:

- derived or shared values used by multiple nodes

Use workflow-level outputs for:

- values you want the workflow to expose as its final result contract

Supported workflow bindings include:

- `literal`
- `context`
- `memory`
- `secret`
- `workflow_input`
- `workflow_variable`
- `step_output`

Important behavior:

- secret bindings are late-bound at execution time
- the worker resolves the secret just before the live tool or capability call
- the secret value is not stored as plain workflow input history

### Supported Control Flow

Currently supported execution control patterns:

- `if`
- `if_else`
- `parallel` with `fan_out`
- `parallel` with `fan_in`

Current limitation:

- `switch` can still appear as an authoring concept, but is not yet lowered into executable runtime behavior

### Workflow Library

The Studio library panel gives you:

- `Saved Drafts`
- `Triggers`
- `Version History`
- `Run History`

From the library you can:

- `Open Draft`
- `Restore Version`
- `Create Manual Trigger`
- `Invoke` an enabled trigger

### Trigger Model

In the current UI, the main trigger path is:

- manual trigger creation
- manual trigger invocation

Use triggers when you want a stable published workflow to be runnable without reopening and recompiling the draft each time.

## Memory

Use the Memory screen to manage user-specific reusable state.

Current supported memory types in the UI:

- `user_profile`
- `semantic_memory`

Good uses for `user_profile`:

- contact information
- GitHub username
- education
- certifications
- preferences

Good uses for `semantic_memory`:

- short facts
- reusable resume statements
- user preferences stated in natural language

Important rule:

- user-scoped memory only works when the correct `user_id` is provided

The UI helps with this by propagating `Memory User ID` from the workspace into Compose, Chat, and Studio.

## Recommended Playbooks

### 1. Create a Document from Structured Inputs

Use when you want a generated document from a prompt-like brief.

Recommended path:

1. Use `document.spec.generate`
2. Use `document.spec.validate`
3. Use `document.output.derive`
4. Use `document.docx.generate` or `document.pdf.generate`

Important note:

- `document.spec.generate` is for explicit content-generation inputs such as `instruction`, `topic`, `audience`, and `tone`
- markdown transformation is a separate task and should use `document.spec.generate_from_markdown`

### 2. Convert Markdown into a DOCX

Use when the source content already exists as markdown.

Recommended path:

1. `document.spec.generate_from_markdown`
2. `document.spec.validate`
3. `document.output.derive`
4. `document.docx.generate`

### 3. Inspect a GitHub Repository

Use when you want a read-only repo check or repo metadata inspection.

Fast path:

- use `Chat` for a safe direct read-only capability if the request is simple

Structured path:

1. Use `Compose` or `Workflow Studio`
2. Add `github.repo.list` or the relevant GitHub capability
3. Provide explicit repo inputs
4. Run and inspect the results

### 4. Build a Reusable Workflow

Use when you expect to run the same automation repeatedly.

Recommended path:

1. Build it in `Workflow Studio`
2. Add workflow-level inputs
3. Save the draft
4. Publish a version
5. Create a manual trigger
6. Invoke the trigger as needed

### 5. Personalize Workflows with Memory

Use when workflows should reuse profile or preference data.

Recommended path:

1. Add entries in `Memory`
2. Set the workspace `Memory User ID`
3. Bind inputs from `memory` in Studio or let chat/workflows inherit `user_id`
4. Run the workflow

### 6. Build a Qdrant-Backed RAG Workflow

Use when you want grounded retrieval over your own indexed chunks.

Recommended path:

1. `rag.collection.ensure`
2. `rag.index.upsert_texts` or `rag.index.workspace_file`
3. `rag.retrieve`
4. optional answer-generation step

Use `tenant_id`, `workspace_id`, `user_id`, and `namespace` consistently at both index time and retrieval time.

For the full engineering flow, see [rag-playbook.md](rag-playbook.md).

## Troubleshooting

### Compile Preview fails

Check:

- missing required node inputs
- unsupported control-flow pattern
- invalid context path
- invalid workflow input type

### A workflow cannot find memory

Check:

- `Memory User ID`
- memory `scope`
- memory `name`
- optional `key`

### Chat keeps asking for clarification

This usually means:

- the request is too ambiguous
- required safety constraints are missing
- the system decided a workflow is needed but lacks enough detail

### GitHub direct calls fail with auth errors

Check:

- cluster secret wiring
- GitHub token validity
- org SSO authorization
- MCP GitHub service health

## Best Practices

- Use `Chat` for lightweight conversational work.
- Use `Compose` for planner-led workflows from a goal.
- Use `Workflow Studio` for durable, versioned, manually authored workflows.
- Keep workflow interfaces explicit instead of overloading raw `context_json`.
- Store stable profile data in `user_profile`.
- Store searchable facts in `semantic_memory`.
- Use `Compile Preview` before publishing.
- Publish versions before relying on triggers.

## Related Docs

- [README.md](../README.md)
- [api.md](api.md)
- [architecture.md](architecture.md)
- [semantic-memory.md](semantic-memory.md)
- [tools.md](tools.md)
- [k8s-port-forward.md](k8s-port-forward.md)
