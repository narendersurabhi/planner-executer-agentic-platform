# Chat Real-Time Clarification Mapping Implementation Plan

## Goal

Map clarification answers into canonical slot state on every pending-clarification turn, instead of waiting until submit time.

## Problem

The current clarification flow now asks one question at a time, but the strongest slot mapping still happens during submit normalization. That leaves a gap:

- short answers are not always reflected in durable slot state immediately
- the next active question can lag behind the user answer
- routing still sees stale clarification state for the current turn

## Design

### Phase 1: Pre-Route Clarification Answer Mapping

When `pending_clarification` exists, run the clarification normalizer before routing and merge any canonical slot updates into the current context/state.

Implementation:
- reuse the existing clarification normalizer on pending turns
- update the chat context envelope before route-time context projection
- persist refreshed clarification state before routing so the active question can advance immediately

Acceptance criteria:
- route-time context can see newly mapped clarification slots from the same user turn
- pending clarification state can advance from one field to the next without waiting for submit

### Phase 2: Queue Advancement

Advance `current_question` and `current_question_field` as soon as the active field is resolved, while keeping opportunistic extra-field extraction.

Implementation:
- rebuild `pending_fields` and `pending_questions` from refreshed clarification assessment
- update the durable slot ledger and answer history on every mapped turn

Acceptance criteria:
- resolving `audience` immediately advances the active field to `tone`
- resolving the last blocking field allows the router to proceed to submit in the same turn

### Phase 3: Intent-Change Handling

Detect when the user is no longer answering the active clarification and is instead changing the request.

Implementation:
- distinguish likely slot answers from intent-changing turns
- preserve or exit the clarification frame deterministically

Acceptance criteria:
- genuine request changes do not get forced into the current slot
- ordinary slot answers stay on the active execution path

### Phase 4: Observability And Evals

Track how often real-time clarification mapping resolves the active field, advances the queue, or misassigns the turn.

Implementation:
- expose per-turn mapping outcomes in feedback/debug output
- add focused eval cases for slot resolution and queue advancement

Acceptance criteria:
- real-time clarification mapping is reviewable and regression-testable

## Implementation Status

- Phase 1: implemented
- Phase 2: implemented
- Phase 3: implemented
- Phase 4: implemented
