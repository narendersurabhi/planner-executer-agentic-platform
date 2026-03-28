# Chat Boundary Semantic Classification Implementation Plan

## Objective

Improve chat routing quality by replacing mostly free-form boundary classification with a bounded, evidence-driven decision layer.

The immediate goal is to reduce false `chat_reply` decisions for requests that should become `submit_job`, while preserving low latency for clearly conversational turns.

## Why This Matters

The current `response_first` chat path can misclassify artifact-creation and automation requests as ordinary conversation. That failure happens before the router runs, so no job or workflow is ever created.

This plan makes the chat boundary use:

- conversation-state signals
- lightweight heuristic intent evidence
- semantic capability-family evidence
- explicit workflow-availability state

instead of relying on raw turn text alone.

## Target Design

Introduce a small internal `ChatBoundaryEvidence` artifact that is assembled before `_generate_chat_boundary_decision(...)`.

Suggested evidence fields:

- `conversation_mode_hint`
- `pending_clarification`
- `workflow_target_available`
- `intent`
- `risk_level`
- `needs_clarification`
- `missing_inputs`
- `top_capabilities`
- `top_families`

The boundary model should classify only:

- `chat_reply`
- `execution_request`
- `continue_pending`
- `exit_pending_to_chat`
- `meta_clarification`

The boundary should not decide `submit_job` versus `run_workflow`; that remains the router’s job.

## Phase 1

### Deliverables

- add typed boundary-evidence contracts
- assemble semantic capability-family evidence from the existing capability search path
- add lightweight heuristic intent evidence for non-conversational turns
- inject evidence into the boundary prompt
- persist the boundary decision and evidence in assistant message metadata

### Files

- `libs/core/chat_contracts.py`
- `services/api/app/main.py`
- `services/api/app/chat_service.py`
- `services/api/tests/test_chat_api.py`

## Phase 2

### Deliverables

- add offline boundary eval fixtures from real misroutes
- add boundary metrics for:
  - false chat replies
  - execution escalation rate
  - pending-clarification continuation rate
- attach boundary evidence to feedback exports

### Files

- `eval/chat_boundary_gold.yaml`
- `libs/core/context_eval.py` or new `libs/core/chat_boundary_eval.py`
- `services/api/app/feedback_service.py`
- `services/api/app/main.py`

## Phase 3

### Deliverables

- use boundary feedback and real misroutes to tune:
  - evidence budgets
  - capability-family scoring
  - prompt wording
- optionally add a small specialized classifier on top of the evidence object if the bounded LLM decision remains noisy

## Phase 4

### Deliverables

- add a repeatable boundary eval command and CI gate
- expose a lightweight review queue for likely false `chat_reply` and false `execution_request` cases
- connect boundary eval/reporting to the same feedback/export loop already used for prompt tuning

### Files

- `scripts/eval_chat_boundary.py`
- `eval/chat_boundary_gold.yaml`
- `Makefile`
- `.github/workflows/ci.yml`
- `services/api/app/feedback_service.py`
- `services/api/app/main.py`

## Status

- Phase 1: implemented
- Phase 2: implemented
- Phase 3: implemented
- Phase 4: implemented

## What Exists Now

- `ChatBoundaryEvidence` is assembled before `_generate_chat_boundary_decision(...)`
- boundary decisions and evidence are persisted in assistant-message metadata
- runtime metrics track decision type, reason code, and feedback by prior boundary decision
- negative and partial chat-message feedback carries boundary dimensions through summary and example export
- `eval/chat_boundary_gold.yaml` and `libs/core/chat_boundary_eval.py` provide a reusable gold-set harness
- `scripts/eval_chat_boundary.py`, `make eval-chat-boundary`, and `make eval-chat-boundary-gate` make boundary gating repeatable in local dev and CI
- `GET /feedback/chat-boundary/review` exposes a small operator review queue for likely boundary misroutes

## Implementation Notes

- keep deterministic rules minimal and high precision
- use semantic capability evidence from the registry, not hardcoded artifact noun lists
- keep heuristic intent evidence cheap and skip it for clearly conversational turns
- persist evidence so boundary mistakes are explainable and evaluable

## Success Criteria

- requests like “create a document on Kubernetes” stop dying in `chat_reply`
- clearly conversational turns remain in chat
- assistant message metadata shows the evidence that drove the boundary decision
- new gold cases can be added from live misroutes without changing the runtime shape
