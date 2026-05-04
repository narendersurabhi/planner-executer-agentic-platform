# Chat Architecture Hardening Implementation Plan

## Objective

Reduce recurring chat-to-execution failures by hardening the architecture around:

- clarification state
- capability selection
- slot persistence
- boundary routing
- cross-turn consistency

The goal is to stop fixing the same class of bugs with local patches and instead make the chat execution path structurally reliable.

## Why This Plan Exists

Recent fixes improved important failure modes:

- pending clarification state is more durable
- prior chat turns now help clarification normalization
- boundary routing is less likely to accidentally exit execution
- capability-first planning is in place

But the architecture still has several systemic gaps:

1. clarification state is still spread across raw transcript text, session metadata, normalized intent, and job context
2. submit normalization still operates over candidate capabilities instead of one explicit active execution target
3. prior answers are partly replayed from chat history instead of accumulated as durable slot state
4. boundary routing still searches the full capability space even after an execution family is obvious
5. multiple components can still ask clarification questions without sharing one authoritative clarification state machine
6. session metadata is rewritten as a single JSON blob each turn, which is fragile under concurrent or near-concurrent updates

## Suggested Improvements

### 1. Introduce a First-Class Clarification State

Create one authoritative execution/clarification state object instead of reconstructing state from:

- `draft_goal`
- `pending_clarification`
- `context_json`
- message history

That state should carry:

- `original_goal`
- `active_segment_id`
- `active_capability_id`
- `resolved_slots`
- `pending_slots`
- `slot_provenance`
- `question_history`
- `answer_history`
- `state_version`

### 2. Add an Explicit Active Execution Target

Once chat becomes execution-oriented, the system should select and persist:

- an active capability family
- an active intent segment
- optionally an active capability

Then clarification, normalization, and required-input checks should scope to that target instead of re-evaluating the entire candidate capability set every turn.

### 3. Replace Transcript Replay with a Slot Ledger

Recent history replay is a useful fallback, but it should not be the main source of truth.

Resolved answers should be stored as canonical slots with provenance:

- `explicit_user`
- `clarification_normalized`
- `inferred`
- `defaulted`

That lets later stages consume stable slot state instead of reparsing old user text.

### 4. Narrow Boundary Evidence Once Execution Is Established

Boundary retrieval should not keep treating the entire capability catalog as equally relevant once the chat is clearly in a document, GitHub, filesystem, or workflow path.

After the first strong execution commitment, boundary evidence should be biased toward:

- the active family
- the active segment
- the remaining missing slots

This reduces drift from unrelated capabilities.

### 5. Separate Clarification Ownership from Chat Reply Ownership

Execution clarification should be owned by one path only. If a component asks an execution question, it must do so through the same clarification state machinery.

The system should avoid execution-style questions emerging from generic `respond` or `meta_clarification` paths unless they also establish durable clarification state.

### 6. Harden Session-State Persistence

Session state should not rely on one mutable JSON blob being rewritten every turn.

Introduce:

- explicit versioning
- optimistic concurrency checks
- or normalized state tables for clarification/session execution state

This is important for:

- fast repeated user turns
- multiple browser tabs
- future streaming or background state updates

## Target Architecture

### ExecutionFrame

Add a single execution frame for chat-driven work:

- `frame_id`
- `original_goal`
- `mode`: `chat` | `clarification` | `execution`
- `active_family`
- `active_segment_id`
- `active_capability_id`
- `workflow_target`
- `state_version`

### ClarificationState

The clarification state should be nested under the execution frame:

- `resolved_slots`
- `pending_slots`
- `required_slots`
- `slot_provenance`
- `questions`
- `answered_questions`
- `last_user_answer`
- `auto_path_allowed`

### Stage Responsibilities

- boundary:
  - choose `chat` vs `execution`
  - preserve/continue the active execution frame
- router:
  - choose `ask_clarification`, `submit_job`, `tool_call`, or `run_workflow`
- clarification mapper:
  - map latest answer into canonical slots
- intent normalization:
  - operate on active frame + slot ledger
- planner:
  - consume normalized intent and canonical slots

## Phase 1

### Create First-Class Clarification Contracts

Status:

- Implemented on March 29, 2026

Add typed models for persistent clarification state and execution frame.

Files:

- `libs/core/chat_contracts.py`
- `libs/core/workflow_contracts.py`
- `services/api/app/chat_service.py`
- `services/api/tests/test_chat_api.py`

Deliverables:

- `ExecutionFrame`
- `ClarificationState`
- `SlotProvenance`
- `state_version`

Acceptance criteria:

- chat no longer relies on `draft_goal` as the main clarification anchor
- pending clarification state has explicit active target fields

## Phase 2

### Persist Active Capability / Active Segment

Status:

- Implemented on March 29, 2026

When execution is selected, persist the active execution target and carry it through clarification turns.

Files:

- `services/api/app/main.py`
- `services/api/app/chat_service.py`
- `libs/core/intent_contract.py`
- `services/api/tests/test_chat_api.py`

Deliverables:

- active family selection
- active segment selection
- optional active capability binding

Acceptance criteria:

- clarification normalization is scoped to the active target
- unrelated capability candidates no longer control the next clarification question

## Phase 3

### Replace Replay-First Normalization with Slot Accumulation

Status:

- Implemented on March 29, 2026

Store canonical slot updates durably on every clarification turn and use transcript replay only as fallback.

Files:

- `services/api/app/chat_clarification_normalizer.py`
- `services/api/app/chat_service.py`
- `services/api/app/context_service.py`
- `services/api/tests/test_chat_api.py`

Deliverables:

- slot ledger
- slot provenance
- explicit merge rules

Acceptance criteria:

- earlier answers survive long chats without depending on transcript length
- later submit logic reads canonical slots first, transcript second

## Phase 4

### Narrow Boundary Retrieval by Established Execution Family

Status:

- Implemented on March 29, 2026

Once execution is active, boundary evidence should prefer the active family and remaining missing slots.

Files:

- `services/api/app/main.py`
- `config/capability_registry.yaml`
- `services/api/tests/test_chat_api.py`
- `eval/chat_boundary_gold.yaml`

Deliverables:

- family-scoped retrieval
- family-aware boundary evidence
- fewer unrelated top-capability distractions

Acceptance criteria:

- document flows do not keep surfacing unrelated GitHub/file capabilities as dominant evidence
- pending clarification turns stay in-family unless the user explicitly changes direction

## Phase 5

### Unify Clarification Ownership

Status:

- Implemented on March 29, 2026

Any execution-style clarification question must go through the same clarification state path.

Files:

- `services/api/app/main.py`
- `services/api/app/chat_service.py`
- `services/api/tests/test_chat_api.py`

Deliverables:

- remove execution-question behavior from generic `respond`
- convert execution-oriented `meta_clarification` into router-owned clarification or explicit chat-only exit

Acceptance criteria:

- there is one authoritative way to enter, continue, and exit clarification

## Phase 6

### Harden Session Persistence and Concurrency

Status:

- Implemented on March 29, 2026 as an optimistic merge/retry slice on chat-session writes

Move from whole-blob session rewrites toward versioned or normalized state persistence.

Files:

- `services/api/app/models.py`
- `services/api/app/chat_service.py`
- `services/api/app/database.py`
- migration files under `services/api/app/alembic/versions/`
- `services/api/tests/test_chat_api.py`

Deliverables:

- optimistic concurrency or separate clarification-state storage
- explicit version checks
- better conflict handling

Acceptance criteria:

- fast repeated turns do not silently overwrite resolved clarification state
- multiple tabs do not lose execution context

## Phase 7

### Observability and Eval Expansion

Status:

- Implemented on March 29, 2026

Make clarification-state quality measurable.

Files:

- `services/api/app/feedback_service.py`
- `libs/core/feedback_eval.py`
- `libs/core/chat_boundary_eval.py`
- `Makefile`
- `eval/`

Deliverables:

- clarification-state review queue
- slot-loss metrics
- active-family drift metrics
- eval fixtures for long multi-turn clarification flows

Acceptance criteria:

- slot-loss regressions are visible before users report them
- long-turn document flows are part of offline gating

## Recommended Order

Implement in this order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 5
5. Phase 4
6. Phase 6
7. Phase 7

Reason:

- the main reliability win comes from authoritative clarification state first
- retrieval narrowing is useful, but only after active target state exists
- concurrency hardening is important, but it becomes easier once the execution-frame model is explicit

## Short-Term Wins

If you do not want the full refactor immediately, the highest-value near-term improvements are:

1. persist `active_capability_id` and `active_segment_id` in `pending_clarification`
2. persist canonical `resolved_slots` separately from `context_json`
3. treat transcript history as fallback only, not as the primary slot source
4. gate boundary evidence to the active family during pending clarification

## Success Criteria

The architecture should be considered improved when:

- document-generation chats no longer lose earlier answers
- filename/path, audience, tone, and topic survive across long clarification loops
- execution-oriented chats do not drift into unrelated capability families
- clarification questions come from one consistent path
- repeated or concurrent turns do not overwrite resolved state
- planner failures caused by dropped clarification answers materially decline
