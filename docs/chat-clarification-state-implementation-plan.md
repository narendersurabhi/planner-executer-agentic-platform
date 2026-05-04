## Chat Clarification State Fix Plan

### Problem

The current chat clarification loop stores most state in two fragile forms:

- a growing raw `draft_goal` string
- a thin `pending_clarification` object that only keeps the latest questions/profile

That causes two failure modes:

1. Earlier clarification answers are not durable unless they are promoted into structured `context_json`.
2. Short follow-up turns can drift the working goal away from the original request.

This is why document flows can lose filename/path answers before submit and later fail in planner validation.

### Target Behavior

- The original user goal remains stable throughout clarification.
- Clarification answers accumulate instead of being reconstructed from scratch each turn.
- Submit-time normalization can recover structured fields from the full clarification history.
- Filename/path answers survive into job context for render flows.

### Phase 1

Preserve stable clarification state in session metadata.

- Store `original_goal` inside `pending_clarification`.
- Keep question history and structured known slot values alongside the latest questions.
- Use `pending_clarification.original_goal` as the anchor when building the next candidate goal.

### Phase 2

Make submit-time normalization cumulative across relevant capability contracts.

- Evaluate all relevant capability contracts, not just the first one.
- Normalize missing collectible fields even when they are not strict schema-required inputs, as long as they are useful structured chat inputs.
- Aggregate normalization updates into one `clarification_normalization` payload.

### Phase 3

Persist render-path related clarification answers.

- Add `path` as a chat-collectible field for document render/path capabilities.
- Normalize filename/path answers into canonical `path`.
- Carry those values into `job.context_json` before planner validation.

### Phase 4

Regression coverage.

- Document flow with filename, workspace destination, and later tone answer keeps all prior answers.
- Pending clarification keeps the original goal stable across short replies.
- Submit-time normalization persists `path`/`filename` for document render flows.

### Later Work

- Add a dedicated per-turn clarification-answer mapper using the typed contracts already added in `libs/core/chat_contracts.py`.
- Support auto-path consent values like `any`, `auto`, and `you choose`.
