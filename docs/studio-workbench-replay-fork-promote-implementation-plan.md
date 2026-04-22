# Studio Workbench Replay, Fork, and Promote Implementation Plan

## Summary
- Turn the workbench from `launch + inspect` into an iterative loop: `replay failed step`, `fork run back into an editor`, and `promote agent run into Workflow Studio`.
- Keep v1 entirely inside the existing `/studio` route and current Studio surface split.
- Reuse the current canonical runtime and current workflow save/publish path. Promoted workflows start as unsaved drafts only.
- Keep replay scope to `step replay + run fork`. Do not add one-click full-run rerun in this phase.

## Implementation Changes
- `WorkflowStudio.tsx` should own a parent-level handoff state and callbacks so the workbench can send a converted draft into the workflow surface without URL payloads or persistence. Applying a promote action should:
  - replace the active workflow draft state with the imported draft payload,
  - set a Studio notice such as `Imported from workbench run`,
  - switch `surface=workflow`,
  - leave the workbench surface state intact so switching back preserves the prior workbench session.
- `StudioWorkbenchSurface.tsx` should add debugger-driven actions:
  - run-level actions: `Fork Run` and `Promote to Workflow`,
  - step-level action: `Replay Step` for executable capability steps,
  - disabled states with explicit reason text when a run or step cannot be represented in the target editor.
- Add one shared Studio mapping module for all conversions so the rules are centralized and testable. It should produce:
  - a capability replay draft from a debugger step,
  - an agent workbench draft from a normalized `RunSpec`,
  - a workflow composer draft from a normalized `RunSpec` when the shape is representable.
- Replay behavior:
  - `Replay Step` should never auto-launch immediately.
  - It should prefill the capability sandbox with the selected step’s capability id, source run title/goal/context/user, replayable inputs, and retry policy, then keep the user in the workbench to edit and relaunch with the existing `POST /workbench/capability-runs`.
  - Use the first persisted execution-request input as the source of truth when available; fall back to the step `input_bindings` when no prepared request exists.
- Fork behavior:
  - one-step capability runs should hydrate the capability sandbox,
  - multi-step runs that contain only standard executable capability steps with simple `depends_on` relationships should hydrate the structured agent builder,
  - unsupported or advanced runs should hydrate the raw `RunSpec` tab instead of failing silently.
- Promote behavior:
  - only agent runs are promotable in v1,
  - promotion should be client-side and unsaved,
  - each executable capability step becomes a workflow draft node,
  - `depends_on` becomes workflow edges,
  - non-reference input values should be carried into draft bindings as editable literal JSON strings,
  - goal, summary/title, and context JSON should carry over from the source run,
  - workflow interface should start empty,
  - node positions should use the existing default/auto-layout behavior rather than trying to preserve runtime layout.
- Unsupported promote shapes should be blocked with a clear reason. Treat these as non-promotable in v1:
  - non-capability synthetic/control steps,
  - steps with advanced fields that the workflow draft cannot represent safely, such as unsupported `execution_gate` semantics,
  - run shapes that would require inventing workflow binding kinds not already supported by the Studio compiler.

## Public Interfaces and Types
- No new backend endpoints in v1.
- Reuse existing APIs:
  - `GET /runs/{id}/debugger`
  - `POST /workbench/capability-runs`
  - `POST /workbench/agent-runs`
  - existing workflow definition save/publish endpoints
- Add Studio-local types for:
  - replayable capability draft payload,
  - fork target mode (`capability`, `agent_structured`, `agent_raw`),
  - promotable workflow draft payload plus non-promotable diagnostics.
- Keep all new behavior ephemeral until the user explicitly saves from the workflow surface.

## Test Plan
- Workbench step replay:
  - replaying a capability step prefills the capability sandbox with source capability id, inputs, context, and retry policy,
  - launching from the replayed draft creates a normal canonical workbench run.
- Run fork:
  - one-step capability run forks into capability mode,
  - representable multi-step run forks into structured agent mode,
  - unsupported run forks into raw `RunSpec` mode with a visible reason banner.
- Promote to workflow:
  - representable agent run switches to `surface=workflow` and loads a compileable unsaved draft,
  - compile preview still works after promotion,
  - no workflow definition is created until the user presses Save,
  - switching back to `surface=workbench` preserves the earlier workbench state.
- Failure and compatibility cases:
  - `Promote to Workflow` is hidden or disabled for unsupported run shapes,
  - `Replay Step` is hidden or disabled for non-executable steps,
  - existing workflow authoring, save, publish, and run behavior remain unchanged.

## Assumptions and Defaults
- Promote target: unsaved workflow draft only.
- Replay scope: step replay plus run fork; no one-click full-run replay in this phase.
- Conversion logic lives in the Studio client because the target is local unsaved editor state and the existing debugger payload already contains the required source data.
