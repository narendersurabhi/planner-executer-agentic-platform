## User Feedback Implementation Plan

### Goal
Collect explicit user feedback on:

- chat response quality
- intent understanding quality
- planner plan quality
- final job outcome quality

The design should attach feedback to persisted runtime artifacts instead of creating a separate survey subsystem.

### Core Principles

- Store feedback in Postgres as first-class product telemetry.
- Attach feedback to existing targets:
  - `chat_message`
  - `intent_assessment`
  - `plan`
  - `job_outcome`
- Keep snapshots of the rated artifact at submission time.
- Support anonymous or weakly identified actors now via `actor_key`.
- Reuse auth-derived `user_id` later without changing target semantics.
- Keep explicit feedback separate from implicit signals like retries or replans.

### Data Model

One shared feedback record should contain:

- `id`
- `target_type`
- `target_id`
- `session_id`
- `job_id`
- `plan_id`
- `message_id`
- `user_id`
- `actor_key`
- `sentiment`
- `score`
- `reason_codes`
- `comment`
- `snapshot_json`
- `metadata_json`
- `created_at`
- `updated_at`

### Phase 1: Contracts, Storage, and Resolution

Files:

- `libs/core/models.py`
- `services/api/app/models.py`
- `services/api/app/alembic/versions/..._add_feedback_records.py`
- `services/api/app/feedback_service.py`

Work:

- Add typed feedback contracts and enums.
- Add SQLAlchemy `FeedbackRecord`.
- Add Alembic migration for the feedback table and indexes.
- Add target-resolution helpers that load:
  - chat message snapshots
  - intent snapshots from job normalization metadata
  - plan snapshots from the stored plan and tasks
  - outcome snapshots from terminal job state
- Add create-or-update semantics for the same `actor_key` on the same target.

### Phase 2: API Endpoints

Files:

- `services/api/app/main.py`
- `docs/api.md`

Work:

- Add `POST /feedback`
- Add `GET /feedback`
- Add `GET /jobs/{job_id}/feedback`
- Add `GET /chat/sessions/{session_id}/feedback`
- Resolve `actor_key` from auth user id first, then fallback header.
- Emit `feedback.submitted` runtime events.

### Phase 3: UI Collection

Files:

- `services/ui/src/app/components/feedback/FeedbackControl.tsx`
- `services/ui/src/app/lib/feedback.ts`
- `services/ui/src/app/page.tsx`

Work:

- Add assistant-message feedback controls in chat.
- Add intent feedback card in Job Details.
- Add plan feedback card in Job Details.
- Add terminal outcome feedback card in Job Details.
- Persist or reuse a local feedback actor id for dedupe.
- Show submitted state and existing feedback when the same target is reopened.

### Phase 4: Analytics and Review

Files:

- `libs/core/events.py`
- future analytics or eval files

Work:

- Track explicit feedback rates by:
  - model
  - planner version
  - route type
  - target type
- Add reason-code breakdowns and trend reporting.
- Add separate offline evals for feedback quality and target coverage.

### Reason Codes

Chat response:

- `incorrect`
- `unclear`
- `missed_request`
- `too_generic`
- `too_verbose`

Intent understanding:

- `wrong_goal`
- `wrong_scope`
- `missed_constraint`
- `asked_unnecessary_clarification`

Plan quality:

- `missing_step`
- `wrong_order`
- `wrong_capability`
- `too_complex`
- `unsafe`

Outcome quality:

- `did_not_finish`
- `wrong_result`
- `poor_quality`
- `took_too_long`

### Verification Strategy

- API tests for create, update, and list flows.
- Snapshot validation tests per target type.
- UI tests can come later; first slice should focus on backend correctness and a simple manual UI path.

### Initial Scope

Implement phases 1 through 3 first. Keep summaries and aggregate dashboards out of the first patch unless they come nearly free from the stored feedback records.
