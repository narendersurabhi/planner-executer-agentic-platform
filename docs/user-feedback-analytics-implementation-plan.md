## User Feedback Analytics Implementation Plan

### Goal
Turn the new explicit feedback collection system into an operational feedback loop.

The current implementation already captures explicit feedback for:

- `chat_message`
- `intent_assessment`
- `plan`
- `job_outcome`

This next phase should make that data useful for:

- product monitoring
- planner and model regression detection
- debugging low-quality runs
- building evaluation datasets from real user signals

### Guiding Principles

- Keep explicit feedback primary. Do not mix explicit and implicit signals into one score.
- Preserve raw feedback rows as the source of truth.
- Add derived analytics through read paths and metadata enrichment, not by overwriting original feedback.
- Start with summary and analysis first; only add heavier export or review tooling after the read model is stable.
- Avoid coupling analytics to authorization work for now.

## Phase 4A: Summary Analytics and Context Enrichment

### Objective
Make feedback queryable by the dimensions that matter in this system:

- `target_type`
- `sentiment`
- `reason_codes`
- `llm_provider`
- `llm_model`
- `planner_version`
- `workflow_source`
- `job_status_at_feedback`
- `assistant_action_type` for chat messages

### Design

Do not add a second analytics table yet.

Instead:

1. Keep `feedback` as the canonical store.
2. Enrich `FeedbackRecord.metadata_json` at submit time with a normalized `dimensions` block.
3. Add summary endpoints that aggregate from `feedback` rows and their derived dimensions.

### Dimensions to Capture at Write Time

At feedback submission time, derive and persist:

- `target_type`
- `target_id`
- `workflow_source`
- `llm_provider`
- `llm_model`
- `planner_version`
- `job_status_at_feedback`
- `assistant_action_type`
- `has_comment`
- `reason_count`

These fields should live in `metadata_json["dimensions"]`.

### Files

- `services/api/app/feedback_service.py`
  - add:
    - `derive_feedback_dimensions(...)`
    - `summarize_feedback_rows(...)`
    - `reason_code_breakdown(...)`
    - `dimension_breakdown(...)`
  - enrich feedback metadata on every submit

- `libs/core/models.py`
  - add:
    - `FeedbackSummaryRequest`
    - `FeedbackBreakdownBucket`
    - `FeedbackReasonBucket`
    - `FeedbackSummaryResponse`

- `services/api/app/main.py`
  - add:
    - `GET /feedback/summary`
  - support filters:
    - `target_type`
    - `sentiment`
    - `workflow_source`
    - `llm_model`
    - `planner_version`
    - `since`
    - `until`
    - `limit`

- `services/api/tests/test_feedback_api.py`
  - add:
    - summary response tests
    - breakdown tests
    - filtered summary tests
    - dimension enrichment tests

### Response Shape

`GET /feedback/summary` should return:

- total rows
- sentiment counts
- target-type counts
- negative reason breakdown
- breakdown by:
  - `workflow_source`
  - `llm_model`
  - `planner_version`
  - `job_status_at_feedback`

### Recommended v1 Metrics

- `chat_helpfulness_rate`
- `intent_agreement_rate`
- `plan_approval_rate`
- `job_outcome_positive_rate`
- `top_negative_reasons`

## Phase 4B: UI Feedback Insights Panel

### Objective
Expose feedback health directly in the existing operator UI.

### Design

Add a lightweight “Feedback Insights” panel that reads from `GET /feedback/summary`.

It should show:

- overall explicit feedback volume
- chat helpfulness rate
- intent agreement rate
- plan approval rate
- outcome distribution
- top negative reasons
- top affected planner versions and models

### Files

- `services/ui/src/app/page.tsx`
  - add panel state and summary fetch
  - render a compact analytics card or section

- `services/ui/src/app/lib/feedback.ts`
  - add:
    - `FeedbackSummaryResponse` type
    - summary formatting helpers

- optional:
  - `services/ui/src/app/components/feedback/FeedbackInsightsPanel.tsx`

### UX Constraints

- This should stay read-only in v1.
- Keep it compact and operator-friendly.
- Do not build a full admin analytics page yet.

## Phase 4C: Correlate Explicit Feedback with Implicit Signals

### Objective
Use existing runtime behaviors to add context around poor feedback without collapsing them into the same metric.

### Implicit Signals to Correlate

- job replans
- retries
- retry-failed
- cancels
- plan failures
- task failures
- repeated clarification turns

### Design

For jobs that have explicit feedback, compute linked operational counters:

- `replan_count`
- `retry_count`
- `failed_task_count`
- `terminal_status`
- `had_plan_failure`
- `clarification_turn_count`

These should appear in analytics responses as separate fields or separate breakdowns.

### Files

- `services/api/app/feedback_service.py`
  - add:
    - `collect_job_feedback_correlates(...)`

- `services/api/app/main.py`
  - extend `GET /feedback/summary`
  - optional:
    - `GET /feedback/examples`

- `services/api/tests/test_feedback_api.py`
  - add correlation tests

### Important Rule

Do not compute a blended “quality score” in this phase.
Keep explicit and implicit signals separate.

## Phase 4D: Negative Example Export and Evaluation Pipeline

### Objective
Convert real user feedback into reusable evaluation data.

### Design

Add export paths for negatively rated or partially rated examples:

- bad chat responses
- bad intent understanding cases
- rejected plans
- failed outcomes

Each exported row should include:

- feedback row
- target snapshot
- normalized dimensions
- linked job/plan identifiers

### Files

- `services/api/app/feedback_service.py`
  - add export helpers

- `services/api/app/main.py`
  - add:
    - `GET /feedback/examples`
    - optional CSV or JSONL support

- `libs/core/feedback_eval.py`
  - new module for converting exports into eval fixtures

- `eval/feedback_examples_gold.jsonl`
  - optional later

### First Export Filters

- `sentiment=negative`
- `sentiment=partial`
- `target_type`
- `reason_code`
- `planner_version`
- `llm_model`

## Phase 4E: Observability and Metrics

### Objective
Make feedback visible in operational dashboards and alerting.

### Design

Add metrics:

- `feedback_submitted_total{target_type,sentiment}`
- `feedback_reason_total{target_type,reason_code}`
- `feedback_summary_requests_total`
- `feedback_examples_export_total`

### Files

- `services/api/app/main.py`
  - Prometheus counters

- `libs/core/events.py`
  - `feedback.submitted` already exists; extend downstream consumers later if needed

### Future Use

- alert on spikes in negative plan feedback
- alert on sudden drops in intent agreement rate

## File-by-File Implementation Order

### Step 1
Enrich feedback metadata and add summary models.

Files:

- `libs/core/models.py`
- `services/api/app/feedback_service.py`

### Step 2
Add summary endpoint and tests.

Files:

- `services/api/app/main.py`
- `services/api/tests/test_feedback_api.py`

### Step 3
Add UI Feedback Insights panel.

Files:

- `services/ui/src/app/lib/feedback.ts`
- `services/ui/src/app/page.tsx`
- optional `services/ui/src/app/components/feedback/FeedbackInsightsPanel.tsx`

### Step 4
Add implicit correlation logic.

Files:

- `services/api/app/feedback_service.py`
- `services/api/app/main.py`
- `services/api/tests/test_feedback_api.py`

### Step 5
Add export and eval scaffolding.

Files:

- `services/api/app/feedback_service.py`
- `services/api/app/main.py`
- `libs/core/feedback_eval.py`

## Test Plan

### Backend

- summary counts are correct across mixed targets
- negative reason aggregation is correct
- planner version and model breakdowns are stable
- filters apply correctly
- metadata dimensions are written consistently
- implicit correlate fields compute correctly
- exports include snapshots and dimensions

### UI

- summary panel renders fetched totals
- empty summary state renders cleanly
- filter or refresh behavior does not break existing screens

## Recommended Immediate Next Slice

Implement Phase 4A and Phase 4B first:

- enrich feedback rows with derived dimensions
- add `GET /feedback/summary`
- render a simple “Feedback Insights” panel in the existing UI

That is the smallest next step that turns raw feedback collection into something useful for day-to-day iteration.
