# Intent Understanding Next-Phase Implementation Plan

## Objective

Improve intent understanding by making normalization context-aware, slot-complete, disagreement-aware, and reviewable.

The immediate goal is to reduce cases where chat correctly escalates into execution, but the downstream normalized intent envelope is still weak, incomplete, or inconsistent with the selected capabilities.

## Status

- Phase 1 implemented: context-aware intent views
- Phase 2 implemented: disagreement detection and targeted clarification
- Phase 3 implemented: intent review queue and export path
- Phase 4 implemented: offline intent eval and CI gate expansion
- Phase 5 in progress: tuning report and candidate export implemented; prompt/evidence iteration pending

## Why This Matters

Recent work improved:

- chat boundary classification
- chat clarification normalization
- capability-first planning
- shared context construction
- feedback and eval infrastructure

That means more requests now reach intent normalization and planning correctly. The next failures are increasingly of this shape:

- execution is selected, but the normalized intent is still too generic
- clarified values are present in chat context but do not consistently land in the normalized intent envelope
- heuristic and LLM intent interpretations disagree, but the system still proceeds
- required capability inputs are discovered too late
- bad intent envelopes make planner failures look like tool or runtime bugs

So the next high-value step is not another routing layer. It is a stronger intent-understanding layer.

## Current Repo Building Blocks

This repo already has the right pieces to support a stronger intent phase:

- shared context envelopes in `services/api/app/context_service.py`
- clarification normalization in `services/api/app/chat_clarification_normalizer.py`
- canonical slot and required-input logic in `libs/core/intent_contract.py`
- normalized intent envelopes in `libs/core/workflow_contracts.py`
- intent decomposition and assessment in `services/api/app/main.py`
- intent eval harnesses in `libs/core/intent_eval.py` with:
  - `eval/intent_gold.yaml`
  - `eval/intent_normalization_gold.yaml`
- explicit user feedback and review/export plumbing in `services/api/app/feedback_service.py`

The gap is that these parts are still only partially coupled.

## Target Design

### 1. Intent Uses a Dedicated Context View

Intent normalization should consume a bounded `intent_context_view(...)` derived from the shared `ContextEnvelope`, not mostly raw goal text plus ad hoc metadata.

That view should include:

- candidate goal
- exact profile hints when relevant
- normalized clarification slots
- missing required inputs
- top capability candidates
- relevant workflow/job context
- compact recent interaction summaries when useful

### 2. Clarified Canonical Slots Feed Intent Directly

If clarification normalization already produced canonical values like:

- `instruction`
- `topic`
- `audience`
- `tone`
- `path`

those values should be merged into the normalized intent envelope before planning, not only later into `job_context`.

### 3. Intent Disagreement Becomes a First-Class State

When these signals disagree:

- heuristic intent assessment
- LLM intent assessment
- capability evidence
- required-input expectations

the system should prefer a targeted clarification over silently pushing a weak intent envelope forward.

### 4. Intent Mistakes Become Reviewable

Negative `intent_assessment` feedback should feed:

- a review queue
- eval fixtures
- exportable examples for prompt and model tuning

This should match the operating model already used for chat-boundary review.

## Phase 1

### Deliverables

- add `intent_context_view(...)` on top of the shared context envelope
- route `_normalize_goal_intent(...)` through that view
- ensure normalized clarification slots are visible during intent normalization
- preserve provenance for:
  - explicit values
  - normalized clarification values
  - inferred values

### Files

- `services/api/app/context_service.py`
- `services/api/app/main.py`
- `libs/core/workflow_contracts.py`
- `services/api/tests/test_context_service.py`
- `services/api/tests/test_chat_api.py`
- `services/api/tests/test_api.py`

### Acceptance Criteria

- intent normalization reads a bounded stage-specific context view
- clarified canonical slot values are available to intent normalization before planning
- raw chat/session baggage is not passed through unchanged

## Phase 2

### Deliverables

- strengthen slot and capability reconciliation in `intent_contract`
- add explicit disagreement detection across:
  - heuristic intent
  - LLM intent
  - capability evidence
  - required-input state
- return targeted clarification when disagreement is high

### Files

- `libs/core/intent_contract.py`
- `services/api/app/main.py`
- `services/api/app/chat_service.py`
- `libs/core/tests/test_intent_contract.py`
- `services/api/tests/test_chat_api.py`

### Acceptance Criteria

- known bad envelopes are blocked earlier
- ambiguous intent/capability mismatches ask for clarification instead of creating fragile plans
- required-input and capability mismatch diagnostics are clearer

## Phase 3

### Deliverables

- add intent review queue and export path from explicit negative feedback
- add intent-specific dimensions to feedback summaries and exports when missing
- expose likely bad intent cases for operator review

### Files

- `services/api/app/feedback_service.py`
- `services/api/app/main.py`
- `libs/core/models.py`
- `services/api/tests/test_feedback_api.py`

### Candidate Endpoint

- `GET /feedback/intent/review`

### Acceptance Criteria

- operators can inspect likely false or weak intent interpretations
- exported intent examples are easy to convert into gold cases

## Phase 4

### Deliverables

- strengthen the offline intent gate using the existing gold sets
- add repeatable intent-quality reports for:
  - intent agreement
  - capability alignment
  - missing-input precision
  - disagreement-triggered clarification rate
- wire intent gates into CI if thresholds become stable enough

### Files

- `libs/core/intent_eval.py`
- `scripts/eval_intent_decompose.py`
- `Makefile`
- `.github/workflows/ci.yml`
- `eval/intent_gold.yaml`
- `eval/intent_normalization_gold.yaml`

### Acceptance Criteria

- intent regressions are caught before merge
- clarification-trigger behavior can be evaluated, not just guessed

## Phase 5

### Deliverables

- use reviewed intent failures to tune prompts and evidence budgets
- optionally add a small intent disagreement classifier if prompt and rule tuning plateaus
- initial implementation:
  - `GET /feedback/intent/tuning-report`
  - `GET /feedback/intent/tuning-candidates`
  - reviewed-failure scaffolds for converting operator feedback into gold cases
  - `scripts/build_intent_tuning_candidates.py`
  - `make build-intent-tuning-candidates`

### Notes

- this should happen only after the review queue and eval loop are producing useful examples
- do not jump to a learned classifier before the context and disagreement signals are stable

## Implementation Notes

- keep the canonical intent envelope as the contract boundary
- do not let LLMs invent new required-input names
- prefer targeted clarifications over permissive weak envelopes
- use the shared context pipeline instead of adding another intent-only merge stack
- keep raw API behavior and chat behavior aligned where possible, with chat only adding ergonomic normalization

## Success Criteria

- more execution requests produce planner-ready normalized intent envelopes on the first pass
- clarified slot values consistently survive into intent normalization
- ambiguous intent/capability mismatches ask better clarifying questions
- negative `intent_assessment` feedback becomes actionable through review and eval
- intent failures move left from planner/runtime to normalization-time diagnostics
