# Context Construction Impact Analysis and Implementation Plan

## Objective

Turn context construction into a first-class control-plane capability so the platform can:

- improve relevance of chat, intent, planning, and execution decisions
- reduce failures caused by missing, stale, conflicting, or noisy inputs
- make context assembly measurable instead of implicit

This document is the repo-specific plan behind the statement:

> Designed context construction pipelines improving relevance and reducing failure due to missing or noisy inputs.

## Why This Matters in This Repo

This platform already depends on context assembly in multiple places:

- chat turn merging in `services/api/app/chat_service.py`
- intent normalization in `services/api/app/intent_service.py` and `services/api/app/main.py`
- profile-aware memory hydration in `services/api/app/memory_profile_service.py`
- semantic memory promotion in `services/api/app/memory_promotion_service.py`
- workflow and job context merging in `services/api/app/main.py`
- planner request assembly in `libs/core/planner_contracts.py`
- payload resolution and validation in `libs/core/payload_resolver.py`
- feedback-driven analysis in `services/api/app/feedback_service.py`

The current system has the right building blocks, but context is still assembled across several local merge points instead of one consistent pipeline.

That creates predictable failure modes:

- relevant information exists, but is not included at the right stage
- noisy fields survive too long and pollute planning or execution
- the same context is merged differently in chat, preflight, planner, and workflow execution
- missing inputs are discovered too late
- useful prior information is present in memory or metadata but not surfaced when needed

## Current Evidence in the Codebase

The repo already demonstrates the need for stronger context construction:

- `normalized_intent_envelope` is now a canonical planner-facing artifact in `libs/core/planner_contracts.py`.
- chat uses a merged turn context plus session context in `services/api/app/chat_service.py`.
- user profile hydration and profile updates already exist in `services/api/app/memory_profile_service.py`.
- semantic memory writes and feedback exports already exist, but they are not yet part of one explicit context pipeline.
- `payload_resolver` and preflight validation prove that many failures are really context-shape and required-input problems, not raw model problems.

So this is not a speculative improvement area. It is already a control-plane concern spread across the repo.

## Impact Analysis

## 1. User-Facing Impact

### 1.1 Better Relevance

Stronger context construction should improve:

- chat answers
- clarification quality
- capability discovery responses
- planner step selection
- workflow runtime input binding

Expected user-visible improvements:

- fewer generic answers
- fewer unnecessary clarification turns
- better scoped capability suggestions
- more correct first-pass plans
- fewer execution failures caused by missing inputs

### 1.2 Lower Failure Rate From Missing Inputs

This repo has already hit failures caused by:

- missing `main_topic`
- bad `goal` propagation
- render-path ambiguity
- task-intent and render-step mismatches

Context construction directly addresses this class of problem by making required inputs visible earlier and noise removable earlier.

### 1.3 Lower Failure Rate From Noisy Inputs

Common noise sources here include:

- raw chat phrasing that should not become planner authority
- client-provided context fields that should not override server-derived identity
- transient workflow hints that should not persist into later turns
- derived filenames or ad hoc aliases that should be normalized before validation

Reducing that noise improves planner correctness and runtime predictability.

## 2. System Impact

### 2.1 Better Separation of Concerns

A formal context pipeline would make the boundaries clearer:

- raw user input
- normalized intent
- exact profile memory
- semantic memory hints
- capability candidates
- runtime metadata
- execution-safe payload

Today these are mixed in `context_json`, session metadata, job metadata, and workflow context assembly.

### 2.2 Better Planner Inputs

The planner should receive:

- one bounded, normalized, relevance-ranked context bundle
- not arbitrary raw chat/session baggage

This reduces:

- planner drift
- capability mismatch
- invalid must-have inputs
- late-stage validation failures

### 2.3 Better Evaluation

Once context construction is explicit, you can evaluate it directly:

- which sources were included
- which were dropped
- which signals improved outcome quality
- which noisy inputs correlated with failures

That is much stronger than evaluating only final model output.

## 3. Operational Impact

### 3.1 Better Debugging

A context envelope makes failures easier to explain:

- what context was available
- what context was selected
- what context was dropped
- what input was still missing

That would shorten triage for planner and worker failures.

### 3.2 Better Observability

The platform already tracks:

- planner version
- llm model
- feedback summary
- retries, replans, and plan failures

Context construction would add a missing dimension:

- context quality

That is the layer between user input and planner/runtime behavior.

## 4. Risks and Tradeoffs

### 4.1 Latency

More context retrieval and ranking can increase latency.

Mitigation:

- keep exact profile reads cheap
- cap semantic retrieval
- use bounded top-k capability candidates
- build a context budget per stage

### 4.2 Token Bloat

More context can reduce quality if the prompt becomes noisy.

Mitigation:

- define source-specific budgets
- prefer summaries over raw payloads
- include only fields tied to the current objective

### 4.3 Cross-Context Leakage

Context can leak across sessions or user scopes if not bounded.

Mitigation:

- continue binding chat to server-derived `user_id`
- keep profile exact and scoped
- filter semantic retrieval by user and namespace

### 4.4 Hidden Coupling

If every feature adds custom context rules ad hoc, the pipeline becomes harder to reason about.

Mitigation:

- centralize assembly logic
- keep one policy layer for inclusion and ranking
- make stage-specific views derived from one common envelope

## 5. How To Measure Impact

Use existing signals first.

### Relevance Signals

- `chat_helpfulness_rate` from `GET /feedback/summary`
- negative chat reasons like:
  - `too_generic`
  - `missed_request`
  - `unclear`
- capability search eval metrics
- intent eval metrics

### Failure Signals

- plan failure count
- retry count
- replan count
- clarification turn count
- failed task count
- job outcome positive rate

### New Metrics To Add

If this work is implemented, add:

- `context_sources_used_total`
- `context_source_dropped_total`
- `context_missing_required_input_total`
- `context_noise_drop_total`
- `context_budget_truncation_total`
- `context_retrieval_latency_ms`

## Implementation Plan

## Phase 1: Define a Canonical Context Envelope

### Objective

Introduce one explicit artifact for control-plane context assembly.

### Design

Add a typed `ContextEnvelope` in `libs/core/models.py` or `libs/core/workflow_contracts.py`.

Suggested fields:

- `goal`
- `user_scope`
- `session_scope`
- `workflow_scope`
- `normalized_intent_envelope`
- `profile`
- `semantic_memory_hints`
- `interaction_summaries`
- `capability_candidates`
- `runtime_metadata`
- `missing_inputs`
- `dropped_inputs`
- `trace`

This envelope should be the internal assembled view, not a public client payload.

### Files

- `libs/core/models.py` or `libs/core/workflow_contracts.py`
- `services/api/app/chat_service.py`
- `services/api/app/main.py`

### Deliverable

- one typed internal context artifact shared across chat, preflight, and planner handoff

## Phase 2: Centralize Source Collection and Normalization

### Objective

Collect context from all current sources through one reusable service.

### Sources

- turn `context_json`
- session context
- authenticated `user_id`
- `user_profile`
- compact interaction summaries
- semantic memory hints
- workflow interface context
- job metadata
- normalized intent envelope
- capability candidates

### Design

Add a new module:

- `services/api/app/context_service.py`

Core functions:

- `collect_context_sources(...)`
- `normalize_context_sources(...)`
- `build_context_envelope(...)`

### Files

- new `services/api/app/context_service.py`
- `services/api/app/chat_service.py`
- `services/api/app/main.py`
- `services/api/app/memory_profile_service.py`

### Deliverable

- one shared assembly path instead of several local merge rules

## Phase 3: Add Relevance Ranking and Noise Filtering

### Objective

Ensure the envelope includes the most relevant context and removes harmful noise.

### Design

Add deterministic policies for:

- exact-vs-semantic precedence
- stage-specific source budgets
- required-input promotion
- noise dropping
- conflict resolution

Examples:

- exact `user_profile` beats fuzzy memory
- required render inputs are preserved even if low frequency
- stale or conflicting transient values are dropped
- raw chat filler like `yes`, `go ahead`, `thanks` does not become long-lived planner context

### New Helpers

- `rank_context_items(...)`
- `drop_noisy_context_items(...)`
- `derive_missing_inputs(...)`
- `budget_context_for_stage(...)`

### Files

- `services/api/app/context_service.py`
- `libs/core/intent_contract.py`
- `libs/core/payload_resolver.py`
- `services/api/app/feedback_service.py`

### Deliverable

- bounded, relevance-ranked, stage-safe context assembly

## Phase 4: Inject Stage-Specific Context Views

### Objective

Use the same envelope, but render smaller views for each control-plane stage.

### Stages

- chat response
- capability discovery
- intent assessment
- intent decomposition
- planner preflight
- workflow interface validation
- worker payload resolution

### Design

Do not pass the full envelope everywhere.

Instead derive:

- `chat_context_view`
- `intent_context_view`
- `planner_context_view`
- `execution_context_view`

### Files

- `services/api/app/chat_service.py`
- `services/api/app/main.py`
- `libs/core/planner_contracts.py`
- `services/api/app/dispatch_service.py`

### Deliverable

- consistent context logic with stage-specific bounded projections

## Phase 5: Add Evaluation and Observability

### Objective

Measure whether the new context pipeline actually improves relevance and reduces failures.

### Design

Add:

- explicit context metrics
- context trace fields on feedback snapshots
- offline context-quality eval cases

Recommended new artifacts:

- `eval/context_quality_gold.yaml`
- `scripts/eval_context_quality.py`

Metrics should correlate context choices with:

- feedback sentiment
- planner approval
- retries
- replans
- failed tasks

### Files

- `libs/core/feedback_eval.py`
- `services/api/app/feedback_service.py`
- `services/api/app/main.py`
- new `libs/core/context_eval.py`
- new `scripts/eval_context_quality.py`

### Deliverable

- measurable evidence that context construction is driving better outcomes

## Phase 6: Use Feedback To Improve Context Policies

### Objective

Close the loop from failures back into context-construction policy.

### Design

Use negative or partial feedback slices to identify:

- missing context sources
- over-included noisy sources
- stale profile memory
- irrelevant semantic memory retrieval
- capability candidate drift

This should inform:

- source inclusion rules
- ranking policies
- prompt context budgets

### Files

- `services/api/app/feedback_service.py`
- `libs/core/feedback_eval.py`
- `docs/feedback-model-optimization-playbook.md`

### Deliverable

- feedback-guided context quality improvement loop

## Recommended Implementation Order

Implement in this order:

1. Phase 1: canonical context envelope
2. Phase 2: centralized source collection
3. Phase 3: ranking and noise filtering
4. Phase 4: stage-specific projections
5. Phase 5: metrics and eval
6. Phase 6: feedback-driven optimization

This order matters because the main risk is building too many local fixes before the context artifact itself is stable.

## Immediate First Slice

If you want the highest-value first increment, build:

1. `context_service.py`
2. `ContextEnvelope`
3. chat-time envelope assembly
4. planner-time envelope projection
5. context trace metadata on feedback snapshots

That gives you:

- cleaner relevance signals
- earlier missing-input detection
- better debugging
- a measurable path to prove the impact claim

## Summary

For this repo, the statement is justified **if** the work is implemented and measured through:

- better relevance in chat, capability discovery, and planning
- fewer failures from missing required inputs
- fewer failures from noisy or stale context
- explicit context-quality observability

The highest-value path is not “more memory” or “more prompt text.”

It is:

- canonical context construction
- deterministic filtering
- stage-specific projection
- evaluation tied to real failures and user feedback
