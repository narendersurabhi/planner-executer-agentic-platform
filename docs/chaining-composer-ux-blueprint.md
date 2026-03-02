# Chaining Composer UX Blueprint

## Goal
Make workflow composition feel like:

1. pick capabilities
2. wire inputs/outputs
3. fix issues inline
4. submit with confidence

without forcing users to hand-author large JSON.

## Current Baseline
- UI already models draft graph and bindings in `services/ui/src/app/page.tsx`.
- Capabilities API already returns schema metadata:
  - `input_fields`
  - `output_fields`
  - `required_inputs`
  - `output_schema`
  - `adapters`
  from `services/api/app/main.py:2228`.
- Composer compile API already exists in `services/api/app/main.py:2844` and returns:
  - `valid`
  - `diagnostics.errors[]`
  - `diagnostics.warnings[]`
  - `plan`
  - `preflight_errors`

This means UX gains are mostly presentation + interaction improvements, not backend re-architecture.

## UX State Model
Use explicit editor states:

- `empty`: no steps yet.
- `editing`: nodes/edges/bindings changing.
- `validating`: compile/preflight running.
- `invalid`: diagnostics/preflight errors exist.
- `ready`: compile + preflight clean.
- `submitting`: creating job.
- `submitted`: job created.

State transitions should be deterministic:
- any edit => `editing` and invalidates prior `ready`.
- preflight success => `ready`.
- submit allowed only from `ready`.

## Recommended Component Breakdown
Refactor `page.tsx` into composable units:

- `ComposerShell`
  - owns draft, validation state, save/apply actions.
- `CapabilityPalette`
  - searchable grouped capability list.
  - drag to canvas or add button.
- `DagCanvas`
  - node cards, edges, drag/move, auto-layout.
  - connection handles to create dependencies.
- `StepInspector`
  - selected node editor.
  - task name, capability, output path.
- `BindingsTable`
  - one row per required input.
  - source mode picker:
    - `Context`
    - `Step Output`
    - `Memory`
    - `Literal`
- `ValidationPanel`
  - compile errors, warnings, preflight errors.
  - click-to-focus node/field.
- `SubmitBar`
  - compile status badge.
  - `Compile + Preflight` and `Submit`.

## Binding UX Rules
- Default binding mode recommendation:
  - if compatible upstream output exists -> suggest `Step Output`.
  - else if context path appears present -> suggest `Context`.
  - else show unresolved with one-click “Auto-wire best match”.
- Show schema type chip (`string`, `object`, etc.) and description for each required input.
- For `Step Output`, source path dropdown should be built from source node `output_fields`.

## Compile/Preflight UX Rules
- Keep server as source of truth: call `POST /composer/compile`.
- Treat `diagnostics.errors` and `preflight_errors` as blocking.
- Treat warnings as non-blocking but visible.
- Remove duplicate user confusion by showing one merged issue list:
  - `severity`
  - `code`
  - `message`
  - optional `node_id`
  - optional `field`

## Submit Guardrails
- Disable submit unless:
  - at least one node exists
  - local DAG checks pass (no cycles, no missing node refs)
  - server compile/preflight returns valid
- Any graph or binding change after a clean validation must require re-validation.

## High-Impact UI Improvements (Next)
1. **Guided starter wizard**
   - collect goal type, output type, optional constraints
   - seed a starter chain template.
2. **Inspector-first editing**
   - click node opens right-side inspector drawer with full bindings.
3. **Issue-to-fix loop**
   - click error scrolls/focuses node + field.
4. **Run-in-place**
   - after submit, keep canvas visible with live task statuses.

## Implementation Plan

### Phase 1 (fast wins)
- Extract `BindingsTable`, `ValidationPanel`, `SubmitBar` from `page.tsx`.
- Normalize compile result rendering into one issue list.
- Enforce submit gating strictly on compile/preflight freshness.

### Phase 2 (structure)
- Extract `DagCanvas` and `StepInspector`.
- Introduce `ComposerEditorState` enum and reducer for transitions.

### Phase 3 (guided UX)
- Add starter wizard and template-to-chain bootstrapping.
- Add node-level quick actions: auto-wire, fix missing, derive output path.

## Acceptance Criteria
- User can create a valid 4-step chain without editing raw JSON.
- All required inputs are visible and bindable in one inspector.
- Errors are actionable in one click (focus node/field).
- Submit is impossible when compile/preflight is stale or invalid.
- Time to first successful run is reduced vs. current flow.

## Intent Engineering Roadmap (Phased)

### Phase 1: Intent Decomposition Foundation
- Add `POST /intent/decompose` to return `goal_intent_graph` with ordered segments.
- Persist `goal_intent_graph` in job metadata at create time.
- Segment schema:
  - `id`, `intent`, `objective`, `confidence`, `source`
  - `depends_on`, `required_inputs`, `suggested_capabilities`
- Status: implemented.

### Phase 2: Planner Consumption
- Inject `goal_intent_graph` into planner prompt as an ordering hint.
- Use graph intent sequence to fill missing task intents deterministically.
- Keep fallback behavior when metadata graph is absent.
- Status: implemented.

### Phase 3: Composer/Submit Clarification
- Add submit-time intent clarification gate based on confidence threshold.
- Collect clarification answers and append to goal/context on submit.
- Block submit until required clarification is answered.
- Status: implemented.

### Phase 4: Runtime Enforcement
- Enforce capability intent compatibility in worker for capability-backed execution.
- Fail fast with `contract.intent_mismatch` on graph/intent policy violations.
- Status: implemented.

### Phase 5: Observability and Rollout
- Track decomposition and mismatch rates by model/version.
- Add feature flags for staged rollout:
  - `INTENT_CLARIFICATION_ON_CREATE`
  - `INTENT_MIN_CONFIDENCE`
  - `INTENT_DECOMPOSE_ENABLED`
- Status: in progress.
