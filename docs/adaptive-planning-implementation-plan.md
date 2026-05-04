# Adaptive Planning Implementation Plan

This document turns the adaptive-planning tradeoff memo into a concrete implementation plan for this repo.

It assumes the current adaptive replanning shell remains in place and is extended incrementally rather than replaced.

## Objective

Upgrade the current replanning flow from:

- failure happens
- orchestrator marks `pending_replan`
- planner is invoked again with limited recovery context

to:

- failure or acceptance signal is classified
- a deterministic controller chooses the repair strategy
- the planner receives typed prior-execution state as a first-class input
- the runtime preserves successful work and patches only what needs to change

## Current Insertion Points

The current adaptive-planning behavior is concentrated in:

- `libs/core/models.py`
  - `AdaptiveReplanStatus`
  - `PlanRevisionSummary`
  - `JobDetails`
  - `AdaptivePlanningPolicy`
- `libs/core/planner_contracts.py`
  - `PlanRequest`
  - `intent_mismatch_recovery(...)`
- `services/planner/app/planner_service.py`
  - `build_plan_request(...)`
  - `build_llm_prompt(...)`
  - `_format_intent_mismatch_recovery_block(...)`
- `services/api/app/main.py`
  - `_adaptive_replan_status_from_metadata(...)`
  - `_mark_plan_revision_active(...)`
  - `_handle_plan_created(...)`
  - `_handle_task_failed(...)`
  - `_completed_step_snapshots_for_plan(...)`
  - `_supersede_unfinished_plan_tail(...)`
  - `_should_trigger_retry_exhausted_replan(...)`
  - `_request_job_replan(...)`
  - `_replan_job_for_intent_mismatch(...)`
  - `replan_job(...)`

Existing adaptive tests already give a starting safety net:

- `services/api/tests/test_api.py::test_contract_intent_mismatch_triggers_auto_replan_with_recovery_metadata`
- `services/api/tests/test_api.py::test_retry_exhausted_recoverable_failure_triggers_auto_replan_with_completed_context`
- `services/api/tests/test_api.py::test_manual_replan_creates_plan_revision_history_without_deleting_prior_records`

## Implementation Principles

- Keep the current API and event flow working.
- Do not require a database migration in Phase 1.
- Add typed contracts before adding more prompt complexity.
- Make controller decisions deterministic before considering model-assisted repair selection.
- Preserve debugger continuity and revision history.
- Prefer shadow or optional behavior when introducing new control logic.

## Phase 1: First-Class Planner Execution State

### Goal

Give the planner an explicit, typed view of prior execution state instead of forcing it to infer everything from `job_payload` and ad hoc metadata.

### Deliverables

- Add new typed models for prior execution state.
- Extend `PlanRequest` to carry this state explicitly.
- Build a prompt section from typed state, not only `intent_mismatch_recovery`.
- Keep current `metadata_json` storage as the persistence layer for now.

### Files

- `libs/core/models.py`
- `libs/core/planner_contracts.py`
- `services/planner/app/planner_service.py`
- `services/api/app/main.py`
- `services/api/tests/test_api.py`
- `libs/core/tests/test_planner_contracts.py`

### Proposed new models

Add to `libs/core/models.py`:

```python
class FailedStepContext(BaseModel):
    task_id: str | None = None
    task_name: str | None = None
    capability_id: str | None = None
    task_intent: str | None = None
    error_message: str | None = None
    failure_category: str | None = None
    retry_classification: str | None = None
    retryable: bool = False
    attempt_number: int = 0
    max_attempts: int = 0


class CompletedStepContext(BaseModel):
    task_id: str
    name: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class PlanRevisionContext(BaseModel):
    revision_number: int = 0
    prior_plan_id: str | None = None
    trigger_reason: str | None = None
    completed_steps: list[CompletedStepContext] = Field(default_factory=list)
    failed_step: FailedStepContext | None = None
    remaining_goals: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    human_feedback: list[dict[str, Any]] = Field(default_factory=list)
```

Extend `PlanRequest` in `libs/core/planner_contracts.py`:

```python
revision_context: models.PlanRevisionContext | None = None
```

### API-side changes

In `services/api/app/main.py`:

- add a helper that builds `PlanRevisionContext` from current metadata, active plan, and failed task information
- use that helper in `_request_job_replan(...)`
- keep writing raw metadata for compatibility, but also write a structured `revision_context` payload into `metadata_json`
- extend debugger and job-detail projections to surface the structured revision context

Recommended helper additions:

- `_build_failed_step_context(...)`
- `_build_plan_revision_context(...)`
- `_remaining_goals_from_plan(...)` or a simpler first-pass placeholder

### Planner-side changes

In `services/planner/app/planner_service.py`:

- replace `_format_intent_mismatch_recovery_block(...)` with a more general formatter such as `_format_revision_context_block(...)`
- keep `intent_mismatch_recovery` as one field inside the typed revision context for backward compatibility
- update `build_llm_prompt(...)` to instruct the planner to:
  - preserve successful completed work
  - avoid repeating the known failure mode
  - prefer minimal change to the plan shape when possible

### Acceptance criteria

- planner receives a typed `revision_context` on adaptive replans
- intent mismatch recovery still works
- retry exhaustion replan still works
- debugger and `/jobs/{id}` expose structured replan context
- no DB migration required

### Test additions

Add tests covering:

- `PlanRequest` includes `revision_context` when a replan is pending
- completed-step snapshots serialize into typed `CompletedStepContext`
- failed-step context includes retry and error metadata
- manual replan populates structured revision context
- existing adaptive tests still pass unchanged or with minimal fixture updates

## Phase 2: Deterministic Replan Controller

### Goal

Separate repair decisioning from plan generation.

The planner should not decide whether the system should retry the same step, replan the suffix, or stop for human input. That decision should happen before the planner is called.

### Deliverables

- introduce a deterministic controller module
- classify recovery events
- choose a repair strategy
- surface the chosen strategy in debugger and job details

### Files

- `services/api/app/replan_controller.py` (new)
- `services/api/app/main.py`
- `libs/core/models.py`
- `services/api/tests/test_api.py`
- `services/api/tests/test_runs_api.py`

### Proposed strategy enum

Add to `libs/core/models.py`:

```python
class ReplanStrategy(str, Enum):
    retry_same_step = "retry_same_step"
    switch_capability = "switch_capability"
    patch_suffix = "patch_suffix"
    full_replan = "full_replan"
    pause_for_human = "pause_for_human"
```

Extend `PlanRevisionContext` or `AdaptiveReplanStatus` with:

- `selected_strategy`
- `strategy_reason`

### Controller responsibilities

The controller should classify current events into at least:

- missing input or clarification required
- policy or governance block
- retryable transient failure
- capability or intent mismatch
- repeated failure after retries
- acceptance failure

Then map them deterministically:

- missing input -> `pause_for_human`
- policy block -> no replan
- transient infra failure -> `retry_same_step`
- intent mismatch -> `switch_capability` or `patch_suffix`
- retry exhaustion with reusable prefix -> `patch_suffix`
- corrupted or broad failure -> `full_replan`

### API-side changes

In `services/api/app/main.py`:

- replace direct strategy assumptions inside `_handle_task_failed(...)`
- call the controller from:
  - `_handle_task_failed(...)`
  - `replan_job(...)`
  - later, acceptance/evaluator failure hooks
- keep existing logic as fallback until the controller is stable

### Acceptance criteria

- controller decision is recorded on every adaptive replan attempt
- transient failures can bypass planner re-entry when the right action is retry
- manual replan still works
- intent mismatch and retry-exhausted paths remain supported

### Test additions

Add tests covering:

- clarification-like failures do not trigger planner replan
- transient retryable failures choose `retry_same_step`
- intent mismatch chooses a repair strategy and records it
- exhausted retries with successful prefix choose `patch_suffix`
- policy blocks do not replan

## Phase 3: Delta Replanning and Prefix Preservation

### Goal

Preserve successful work and replace only the broken suffix.

### Deliverables

- extend planner output or plan-merge logic so the system can preserve the successful prefix
- keep completed task and run-step history stable where possible
- reduce full-plan churn in debugger output

### Files

- `libs/core/models.py`
- `libs/core/planner_contracts.py`
- `services/planner/app/planner_service.py`
- `services/api/app/main.py`
- `services/api/tests/test_api.py`

### Design choice

Do not start by inventing a full plan-diff DSL. First implement controlled suffix replacement using existing revision boundaries.

Recommended first pass:

- treat completed tasks as immutable prefix
- allow the planner to generate only the remaining tasks
- merge the new suffix under a new plan revision
- preserve revision history and link old-to-new plan IDs

Possible planner-contract extension:

```python
class ReplanPatchInstruction(BaseModel):
    preserve_completed_prefix: bool = True
    replace_from_task_name: str | None = None
```

This can stay internal before becoming part of the public planner contract.

### API-side changes

In `services/api/app/main.py`:

- update `_handle_plan_created(...)` so replan activation can preserve completed prefix semantics explicitly, not only through metadata
- update `_supersede_unfinished_plan_tail(...)` to preserve already-completed tasks and steps without ambiguity
- optionally tag task and run-step metadata with:
  - `revision_origin`
  - `preserved_from_revision`
  - `superseded_by_revision`

### Acceptance criteria

- replans preserve completed tasks as the execution prefix
- debugger makes the preserved prefix and replaced suffix obvious
- task and plan history remains understandable across revisions

### Test additions

Add tests covering:

- completed tasks are not regenerated or canceled on suffix replan
- unfinished tail is superseded cleanly
- revision history links the old and new suffix correctly
- debugger output remains consistent after multiple replans

## Phase 4: Evaluator-Driven Replanning

### Goal

Allow replans to trigger from acceptance and evaluator signals, not only terminal failures.

### Deliverables

- acceptance-stage signals can request repair
- low-confidence or schema-invalid outputs can trigger controlled rework or replan
- strategy selection remains deterministic

### Files

- `services/api/app/main.py`
- acceptance or critic modules already participating in step review
- `libs/core/models.py`
- `services/api/tests/test_api.py`

### Design constraints

- not every soft signal should cause replan
- evaluator thresholds must be explicit and conservative
- rework and replan should remain distinct outcomes

### Acceptance criteria

- evaluator-triggered replans are auditable
- low-confidence signals can trigger repair without collapsing into infinite churn
- per-step acceptance and replan history is visible in debugger output

### Test additions

Add tests covering:

- schema-invalid outputs trigger rework or replan according to policy
- low-confidence outputs below threshold trigger controller evaluation
- repeated evaluator failures respect adaptive budgets

## Phase 5: Checkpoints and Replay-Aware Adaptive Planning

### Goal

Make adaptive planning useful for long-running and side-effectful steps by allowing replay from durable checkpoints instead of blunt restart.

### Deliverables

- checkpoint-aware failure classification
- replay-aware strategy selection
- revision context includes checkpoint lineage when relevant

### Files

- scheduler and run model files aligned with the broader orchestration roadmap
- `services/api/app/main.py`
- run-step and attempt persistence models
- debugger projection code

### Notes

This phase should align with `docs/best-in-class-agentic-orchestration-implementation-plan.md`.

Do not start here. This is where adaptive planning becomes runtime-infrastructure work, not only planner-contract work.

## Proposed Sequence of Code Changes

### Change set 1

- add `PlanRevisionContext`, `CompletedStepContext`, `FailedStepContext`
- extend `PlanRequest`
- build revision context in API
- update planner prompt to use it

### Change set 2

- add `ReplanStrategy`
- add `replan_controller.py`
- route `_handle_task_failed(...)` and `replan_job(...)` through controller
- expose strategy in debugger and job details

### Change set 3

- implement suffix-preservation semantics
- stabilize revision-link metadata
- add debugger visibility for preserved prefix vs replaced suffix

### Change set 4

- evaluator-triggered repair hooks
- bounded thresholds and budgets

### Change set 5

- checkpoint-aware recovery
- replay lineage in revision context

## API and Compatibility Notes

- Keep `POST /jobs/{job_id}/replan` unchanged in early phases.
- Keep adaptive policy stored in metadata until a larger canonical-run migration justifies a stronger persistence model.
- Keep `JobDetails` and debugger payloads backward-compatible by adding fields rather than renaming existing ones.
- Keep current event names in place until the broader runtime migration changes event semantics.

## Metrics to Add Early

Add these metrics as soon as the controller exists:

- `adaptive_replan_requested_total`
- `adaptive_replan_strategy_total{strategy=...}`
- `adaptive_replan_success_total`
- `adaptive_replan_prefix_preserved_total`
- `adaptive_replan_full_replan_total`
- `adaptive_replan_blocked_total{reason=...}`

Add debugger or response fields for:

- revision trigger reason
- selected strategy
- failed-step classification
- prefix preserved count
- superseded task count

## Non-Goals for the First Delivery

- no learned controller
- no multi-agent planner
- no DB migration just to carry revision context
- no broad memory-injection system for repair traces
- no full canonical-run rewrite as part of this first adaptive-planning iteration

## Suggested First PR Scope

The best first PR is:

- Phase 1 only
- plus the smallest scaffolding needed so Phase 2 can follow cleanly

That means:

- add typed revision context models
- extend `PlanRequest`
- thread the new state from API to planner
- update planner prompt construction
- surface the structured state in debugger and job details
- add focused tests around the existing replan paths

This is the highest-value change with the lowest architectural risk.

## Success Criteria

The implementation is on track if, after Phases 1 and 2:

- replans are easier to explain from the debugger
- similar failures lead to consistent repair decisions
- planner outputs preserve successful work more often
- the number of unnecessary full replans drops
- test coverage for adaptive behavior grows with each new strategy

## Relationship to Existing Docs

This plan is the near-term implementation path for:

- `docs/adaptive-planning-tradeoffs.md`
- `docs/best-in-class-agentic-orchestration-implementation-plan.md`

Use this document for scoped delivery planning.
Use the tradeoff memo for deciding what not to build yet.
Use the broader orchestration roadmap when adaptive planning work begins to depend on canonical runs, checkpoints, and scheduler unification.
