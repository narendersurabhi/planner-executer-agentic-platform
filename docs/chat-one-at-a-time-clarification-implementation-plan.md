# Chat One-At-A-Time Clarification Implementation Plan

## Goal

Reduce clarification-mapping errors by asking only one execution clarification question at a time while preserving the full unresolved slot queue in structured state.

## Problem

The current chat clarification flow often asks multiple questions in one assistant turn. Short user replies such as `docx`, `yes`, `any`, or `practical` then have to be mapped against several possible missing fields at once, which increases ambiguity and contributes to wrong slot assignment or clarification drift.

## Design

### Phase 1: Single Active Clarification Question

Persist one active question and one active field in `ClarificationState`, while keeping the full remaining queue in `pending_questions` and `pending_fields`.

Implementation:
- add `current_question` and `current_question_field` to the persisted clarification state
- show only the first question to the user
- keep the full remaining queue in state for later turns
- preserve the existing slot ledger and active execution target

Acceptance criteria:
- assistant clarification turns display only one question
- `pending_clarification.questions` contains only the active question
- `pending_clarification.pending_questions` keeps the full queue
- `pending_clarification.current_question_field` points at the active slot

### Phase 2: Mapper Preference For The Active Field

Use `current_question_field` as the primary field when mapping the next user answer, while still allowing opportunistic extraction of extra high-confidence fields from the same turn.

Implementation:
- pass `current_question_field` into the clarification normalizer / mapper
- prefer that field when the user answer is short or underspecified
- continue harvesting extra fields only when confidence is high

Acceptance criteria:
- terse answers like `docx` or `practical` prefer the currently asked field
- multi-field answers can still fill more than one slot

### Phase 3: Observability

Make the active-question flow visible in feedback and debugging.

Implementation:
- project `current_question` and `current_question_field` into session/context views
- expose them in clarification review/debug output

Acceptance criteria:
- session/debug state clearly shows the active clarification field

## Implementation Status

- Phase 1: implemented
- Phase 2: implemented
- Phase 3: implemented
