# Intent Normalization Implementation Plan

This document translates the intent normalization architecture into concrete
code changes for the current Agentic Workflow Studio codebase.

It is not a greenfield roadmap. It assumes the current implementation in:

- `services/api/app/intent_service.py`
- `services/api/app/main.py`
- `services/planner/app/planner_service.py`
- `libs/core/intent_contract.py`
- `libs/core/workflow_contracts.py`

## 1. Objectives

The implementation should achieve five things:

1. one canonical normalization artifact
2. one normalization pipeline in the API
3. capability-driven clarification
4. planner consumption of normalized artifacts instead of raw re-inference
5. durable normalization traces and eval coverage

## 2. Entry Points For Change

### Shared contracts

- `libs/core/workflow_contracts.py`
- `libs/core/intent_contract.py`

### API normalization path

- `services/api/app/intent_service.py`
- `services/api/app/main.py`

### Planner handoff and validation

- `services/planner/app/planner_service.py`
- `services/planner/app/main.py`

### Tests

- `libs/core/tests/test_intent_contract.py`
- `libs/core/tests/test_workflow_contracts.py`
- `services/api/tests/test_api.py`
- planner tests where goal-intent segments are validated

## 3. New Shared Contracts To Add

Add to `libs/core/workflow_contracts.py`:

- `NormalizationTrace`
- `ClarificationState`
- `NormalizedIntentEnvelope`

Recommended responsibilities:

- `NormalizationTrace`
  - model
  - provider
  - assess mode
  - decompose mode
  - fallback reason
  - candidate capability ids
- `ClarificationState`
  - `needs_clarification`
  - `requires_blocking_clarification`
  - `missing_inputs`
  - `questions`
- `NormalizedIntentEnvelope`
  - `schema_version`
  - `goal`
  - `profile`
  - `graph`
  - `candidate_capabilities`
  - `clarification`
  - `trace`

Also add:

- `parse_normalized_intent_envelope(...)`
- `dump_normalized_intent_envelope(...)`

## 4. Phase 0: Baseline And Guardrails

Before changing behavior, make the current flow measurable.

### Changes

- Add metrics/log labels for:
  - assessment source
  - decomposition source
  - fallback used
  - clarification required
  - invalid segment rejection reason
- Capture a small golden set of representative goals:
  - document generation
  - render with explicit path
  - repo tasks
  - search/retrieval tasks
  - high-risk write tasks

### Why

This makes later cutovers measurable instead of subjective.

## 5. Phase 1: Contract Consolidation

Introduce the canonical envelope without changing planner behavior yet.

### Shared changes

In `libs/core/workflow_contracts.py`:

- add `NormalizedIntentEnvelope`
- keep `GoalIntentProfile` and `IntentGraph` unchanged for compatibility

### API changes

In `services/api/app/intent_service.py`:

- add a new top-level orchestration function:
  - `normalize_goal_intent(...)`

It should:

1. call assessment
2. call decomposition
3. assemble candidate capability sets
4. derive clarification
5. return the envelope

In `services/api/app/main.py`:

- keep existing `goal_intent_profile` and `goal_intent_graph` writes
- additionally persist `normalized_intent_envelope`

### Exit criteria

- all current callers can continue working
- new metadata is persisted without changing runtime behavior

## 6. Phase 2: Capability-Driven Clarification

Move clarification logic away from broad intent heuristics.

### Shared changes

In `libs/core/intent_contract.py`:

- add helpers to compute missing inputs from:
  - normalized segment slots
  - candidate capability ids
  - required input aliases

Suggested helper names:

- `derive_segment_missing_inputs(...)`
- `derive_envelope_clarification(...)`
- `required_input_question(...)`

### API changes

In `services/api/app/intent_service.py`:

- replace most uses of `blocking_clarification_slots(...)`
- use capability contracts plus normalized aliases to build questions

### Result

Clarification becomes tied to what the system can actually execute, not just to
high-level intent labels.

## 7. Phase 3: Unify LLM And Heuristic Output Shapes

Make assessment and decomposition feed the same canonical envelope shape.

### API changes

In `services/api/app/main.py`:

- refactor `_infer_goal_intent_with_metadata(...)`
- refactor `_decompose_goal_intent(...)`
- keep `_llm_decompose_goal_intent(...)` as an implementation detail

The public API-layer contract should become:

- `normalize_goal_intent(...)` returns a validated envelope

### Rules

- no raw LLM normalization output should be persisted directly
- all paths must pass through `intent_contract` normalization
- fallback paths must still populate the same envelope shape

## 8. Phase 4: Planner Handoff Cleanup

Make the planner consume normalized artifacts instead of relying on raw goal
text plus secondary inference.

### API changes

In `services/api/app/main.py`:

- pass `normalized_intent_envelope` into planner request metadata
- continue passing legacy fields during the migration window

### Planner changes

In `services/planner/app/planner_service.py`:

- prefer normalized envelope segments when selecting a segment for each task
- avoid fresh intent inference from raw task text unless the envelope is absent

In `services/planner/app/main.py`:

- read the envelope from job metadata for planner-side operations
- expose a fallback path for old jobs that only have profile/graph fields

### Exit criteria

- planner validation succeeds against the same normalized object the API stored
- planner behavior is stable for both new and old jobs

## 9. Phase 5: Observability And Debugger Read Path

Make normalization visible in job details and debugger surfaces.

### API changes

In `services/api/app/main.py`:

- add the normalized envelope to job detail responses
- expose normalization trace metadata
- expose clarification state and candidate capabilities

### UI changes

Optional but recommended:

- show normalized intent source
- show missing inputs and clarification blockers
- show candidate capabilities per segment

### Why

This makes misrouting and bad slot derivation debuggable without log diving.

## 10. Phase 6: Cutover And Cleanup

After new-path stability is proven:

- switch planner call sites to require the normalized envelope for new jobs
- de-emphasize direct consumers of raw `goal_intent_profile` and
  `goal_intent_graph`
- keep legacy parsing only for backward-compatible record reads

Do not remove the old fields until:

- old records no longer depend on them
- tests cover envelope-only planner intake
- job detail consumers have migrated

## 11. Concrete Function Targets

### `services/api/app/intent_service.py`

Current functions to preserve or wrap:

- `assess_goal_intent(...)`
- `decompose_goal_intent(...)`
- `resolve_intent_confidence_threshold(...)`
- `blocking_clarification_slots(...)`

Recommended additions:

- `normalize_goal_intent(...)`
- `_derive_capability_driven_clarification(...)`
- `_assemble_normalized_envelope(...)`

### `services/api/app/main.py`

Current functions that should become implementation details:

- `_infer_goal_intent_with_metadata(...)`
- `_decompose_goal_intent(...)`
- `_llm_decompose_goal_intent(...)`

Call sites to update:

- chat job creation path
- `POST /jobs`
- intent clarify endpoint
- workflow compile/publish/run paths that currently persist intent metadata

### `libs/core/intent_contract.py`

Keep as the deterministic authority for:

- intent label normalization
- slot normalization
- segment contract validation

Add:

- capability-aware missing-input derivation
- user-facing question generation from normalized required inputs
- envelope-level helper functions

### `services/planner/app/planner_service.py`

Update:

- `validate_plan_request(...)`
- segment selection and task-intent resolution helpers

Goal:

- validate against normalized envelope data first
- use raw goal text only as fallback

## 12. Suggested Feature Flags

Add a narrow rollout flag instead of a big-bang cutover.

Suggested env vars:

- `INTENT_NORMALIZATION_ENVELOPE_ENABLED=true|false`
- `INTENT_NORMALIZATION_CLARIFICATION_MODE=heuristic|capability`
- `INTENT_NORMALIZATION_PLANNER_USE_ENVELOPE=true|false`

These should allow:

- write-only shadow mode first
- planner read-path opt-in second
- capability-driven clarification cutover third

## 13. Test Plan

### Shared contract tests

Add tests for:

- envelope parsing and dumping
- backward-compatible field preservation
- clarification derivation from capability-required inputs

### API tests

Add tests for:

- heuristic envelope generation
- LLM envelope generation
- fallback to heuristic with the same envelope shape
- blocking clarification when capability-required inputs are absent
- non-blocking normalization when optional fields are absent

### Planner tests

Add tests for:

- planner validation with envelope-backed segments
- fallback behavior when only legacy graph/profile data exists
- reduced drift for document-spec generation and render tasks

## 14. Rollout Sequence

1. Phase 0 metrics and golden set
2. Phase 1 envelope contract added and persisted in shadow mode
3. Phase 2 capability-driven clarification
4. Phase 3 unified API normalization pipeline
5. Phase 4 planner consumption of the envelope
6. Phase 5 debugger and job-detail visibility
7. Phase 6 cleanup of legacy-only code paths

## 15. Recommended First PR

The first PR should be intentionally small.

Scope:

- add `NormalizedIntentEnvelope`
- add parse/dump helpers
- add `normalize_goal_intent(...)` in `intent_service.py`
- persist the envelope on new jobs without changing planner behavior

That gives the repo one canonical artifact immediately while keeping the
behavioral risk low.
