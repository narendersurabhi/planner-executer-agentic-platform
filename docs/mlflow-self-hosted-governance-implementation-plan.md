# MLflow Self-Hosted Governance Implementation Plan

## Goal
Adopt **MLflow as the open, self-hosted system of record** for model, prompt, eval, and runtime-governance metadata in Agentic Workflow Studio without replacing the current execution architecture.

The platform should continue to execute through:

- API
- planner
- worker
- chat
- Studio workflows

MLflow should become the place where we track:

- offline eval runs
- prompt/model/version lineage
- deployment candidates and approvals
- runtime quality summaries
- links between user feedback, eval results, and deployed model choices

## Why MLflow Fits This Repo

This repo already has the core ingredients MLflow can organize:

- explicit eval harnesses in:
  - `scripts/eval_intent_decompose.py`
  - `scripts/eval_capability_search.py`
  - `scripts/eval_capability_search_feedback.py`
- durable runtime metadata such as:
  - `llm_model`
  - `planner_version`
  - `prompt_version`
  - `policy_version`
  - `tool_version`
- user feedback capture plus analytics:
  - `POST /feedback`
  - `GET /feedback/summary`
  - `GET /feedback/examples`
- runtime artifacts and job lineage in Postgres, Redis, and the shared artifact store

MLflow is the right fit here when the priority is:

- self-hosting
- open formats
- artifact lineage
- lightweight governance over prompts, evaluations, and deployment decisions

## Non-Goals

This plan does **not** make MLflow the runtime executor or serving layer.

Specifically:

- do not route inference traffic through MLflow serving
- do not replace Postgres as the application database
- do not replace Prometheus/Grafana/Jaeger for observability
- do not replace the feedback store with MLflow

MLflow should complement the platform, not absorb it.

## Guiding Principles

1. Keep MLflow optional behind config.
2. Start with offline eval tracking before runtime-wide instrumentation.
3. Treat MLflow as governance metadata and artifact lineage, not as the source of truth for live job state.
4. Log stable, reviewable artifacts first; avoid noisy per-token or per-turn spam.
5. Use registry aliases and approval tags for rollout decisions.
6. Keep deployment reversible through environment-variable model selection.

## Target Operating Model

### What MLflow Tracks

Use MLflow for:

- offline eval results
- exported feedback datasets
- prompt bundles
- model candidate metadata
- deployment approvals
- rollout notes
- links to runtime job/plan/task records

### What Stays Outside MLflow

Keep these in existing systems:

- jobs, plans, tasks, sessions, workflows: Postgres
- event streams: Redis
- metrics and alerting: Prometheus/Grafana
- traces: OTLP/Jaeger
- user feedback rows: Postgres
- vector memory/capability retrieval: Qdrant

### Recommended Experiment Layout

Start with these experiments:

- `awe/offline/intent_eval`
- `awe/offline/capability_search_eval`
- `awe/offline/capability_feedback_eval`
- `awe/offline/feedback_examples`
- `awe/runtime/jobs`
- `awe/runtime/chat`
- `awe/releases`

### Recommended Common Tags

Every MLflow run should carry stable tags where available:

- `service`
- `environment`
- `git_sha`
- `llm_provider`
- `llm_model`
- `prompt_version`
- `planner_version`
- `policy_version`
- `tool_version`
- `workflow_source`
- `job_id`
- `session_id`
- `plan_id`
- `task_id`
- `eval_dataset`
- `eval_slice`

## Architecture

### Self-Hosted MLflow Deployment

Use a standard self-hosted deployment:

- MLflow tracking server
- Postgres backend store
- S3-compatible object store or shared artifact storage for artifacts

For this repo, the cleanest shape is:

- Docker Compose:
  - optional `mlflow` service
  - optional object store service if needed
- Kubernetes:
  - `deploy/k8s/mlflow/`
  - separate from `deploy/k8s/observability/`

### Integration Pattern

Add a small internal adapter rather than scattering direct `mlflow.*` calls everywhere.

Recommended new module:

- `libs/core/mlflow_tracking.py`

Responsibilities:

- read env/config
- create or reuse experiments
- start and close runs
- log params, metrics, tags, artifacts
- gracefully no-op when disabled

This keeps the rest of the codebase clean and testable.

## Phased Implementation

## Phase 1: MLflow Foundation and Offline Eval Logging

### Objective
Stand up self-hosted MLflow and log the existing offline evaluation flows first.

### Scope

- add MLflow config/env support
- add a reusable tracking adapter
- log eval metrics and artifacts from current scripts
- keep runtime services untouched initially

### Files

- `pyproject.toml`
  - add optional `mlflow` dependency

- `.env.example`
  - add:
    - `MLFLOW_ENABLED=false`
    - `MLFLOW_TRACKING_URI=`
    - `MLFLOW_EXPERIMENT_PREFIX=awe`
    - `MLFLOW_ARTIFACT_LOCATION=`
    - `MLFLOW_RUN_ENV=local`
    - `MLFLOW_DEFAULT_TAGS_JSON=`

- `docker-compose.yml`
  - add optional `mlflow` service
  - wire backend store / artifact storage config

- `deploy/k8s/mlflow/`
  - add deployment, service, config, optional ingress

- `libs/core/mlflow_tracking.py`
  - add:
    - `tracking_enabled()`
    - `MlflowTrackingConfig`
    - `tracking_client()`
    - `start_run(...)`
    - `log_eval_report(...)`
    - `log_dataset_artifact(...)`
    - `set_common_tags(...)`

- `scripts/eval_intent_decompose.py`
  - log:
    - params
    - metrics
    - gold-set reference
    - JSON report artifact

- `scripts/eval_capability_search.py`
  - same pattern

- `scripts/eval_capability_search_feedback.py`
  - same pattern

- `libs/core/feedback_eval.py`
  - optionally add helper for emitting MLflow-friendly dataset metadata

### Deliverables

- self-hosted MLflow reachable locally and in Kubernetes
- offline eval runs visible in MLflow
- reports and gold-set references attached as artifacts

### Test Plan

- unit test adapter no-op behavior when disabled
- unit test adapter tag/metric normalization
- smoke test scripts log a run when MLflow is enabled

## Phase 2: Runtime Lineage for Jobs, Plans, and Feedback

### Objective
Link live runtime behavior to model/version choices without turning every event into a separate MLflow run.

### Scope

Track coarse-grained runtime runs:

- chat session or chat action summaries
- job-level planning runs
- selected task execution summaries
- feedback summary links

Do **not** log every message token or every Redis event.

### Recommended Run Structure

- parent run: job or chat session
- child run: planner evaluation
- child run: worker execution summary
- child run: feedback summary attachment

### Files

- `services/api/app/main.py`
  - log job creation metadata
  - log feedback export and summary snapshots as optional artifacts

- `services/api/app/feedback_service.py`
  - add helper to create stable feedback summary artifacts suitable for MLflow logging

- `services/planner/app/runtime_service.py`
  - log planner run summary:
    - plan size
    - planner version
    - selected capabilities
    - success/failure

- `services/worker/app/main.py`
  - log worker execution summary:
    - tool/capability used
    - status
    - duration
    - retries

- `libs/core/events.py`
  - optional: define a minimal event-to-tracking handoff if async logging is needed later

### Deliverables

- one MLflow parent run per durable job
- linked planner and worker child runs
- stable tags tying MLflow runs back to job/plan/task IDs

### Test Plan

- API job creation path logs correct tags when enabled
- planner logs summary only once per plan
- worker logs summary only on terminal task outcome

## Phase 3: Prompt and Model Governance

### Objective
Use MLflow as the review and approval surface for prompts, provider model IDs, and evaluation evidence.

### Design

This repo currently selects runtime models through environment variables such as:

- `OPENAI_MODEL`
- `CHAT_ROUTER_MODEL`
- `CHAT_RESPONSE_MODEL`
- `CHAT_PENDING_CORRECTION_MODEL`
- `INTENT_ASSESS_MODEL`
- `INTENT_DECOMPOSE_MODEL`

The planner and worker still use `OPENAI_MODEL`.

MLflow should track the governance metadata behind those choices:

- prompt bundle used
- provider model ID
- eval report artifacts
- approval status
- rollout alias

### Registry Strategy

Use MLflow Registry primarily as a **governance record**, even for hosted provider models.

That means:

- register candidate models or prompt/model bundles
- attach tags for the actual provider-hosted model ID
- attach eval artifacts and feedback summaries
- use aliases like:
  - `candidate`
  - `staging`
  - `production`
  - `rollback`

Because this repo calls provider-hosted models directly, MLflow Registry is mostly a metadata and approval layer, not the serving path.

### Files

- `libs/core/mlflow_tracking.py`
  - add:
    - `register_candidate(...)`
    - `set_registry_alias(...)`
    - `record_approval(...)`

- `docs/feedback-model-optimization-playbook.md`
  - cross-link MLflow governance flow once implemented

- `README.md`
  - add MLflow setup and governance summary after implementation lands

- new helper module:
  - `libs/core/release_tracking.py`
  - track:
    - selected model IDs
    - prompt versions
    - rollout notes
    - approval actor

### Deliverables

- candidate-to-production promotion flow in MLflow
- approval tags and artifacts attached to releases
- rollback candidate preserved

### Test Plan

- registry metadata creation test
- alias promotion test
- release metadata serialization test

## Phase 4: Feedback-to-Eval Governance Loop

### Objective
Close the loop from user feedback to governed model decisions.

### Scope

Use the new feedback system to populate MLflow artifacts and release evidence:

- `GET /feedback/examples`
- `GET /feedback/summary`
- exported JSONL slices
- offline comparison reports

### Design

When a model candidate is evaluated:

1. export a narrow slice of feedback examples
2. attach the dataset artifact to an MLflow run
3. attach eval results against that slice
4. record approval or rejection in MLflow
5. promote aliases only when thresholds are met

### Files

- `services/api/app/main.py`
  - optional admin/export helper for MLflow-bound dataset snapshots

- `libs/core/feedback_eval.py`
  - add metadata helpers for dataset lineage:
    - dataset id
    - source filter
    - created_at
    - target_type
    - reason_code filters

- `scripts/`
  - add:
    - `scripts/export_feedback_examples.py`
    - `scripts/log_feedback_dataset_to_mlflow.py`

### Deliverables

- repeatable feedback-slice exports
- MLflow artifacts linking candidate runs to real user feedback slices
- stronger evidence for release decisions

### Test Plan

- dataset export metadata test
- CLI smoke tests for artifact logging

## Phase 5: Release Gates and Rollout Discipline

### Objective
Make MLflow the approval and evidence system for model/prompt changes without forcing inference through it.

### Release Decision Inputs

A candidate should not be promoted unless it has:

- offline eval report
- feedback-slice eval report
- linked prompt version
- linked provider model ID
- rollout owner
- approval note
- rollback target

### Release Outputs

Promotion should produce:

- MLflow alias change
- release record artifact
- deployment env update
- release note stored as artifact

### Files

- new script:
  - `scripts/promote_mlflow_candidate.py`

- deployment helpers:
  - `scripts/setup_k8s_env.sh`
  - optionally `scripts/docker_images.sh` if env bundles are versioned there

- CI or Makefile additions:
  - `make mlflow-log-intent-eval`
  - `make mlflow-log-capability-eval`
  - `make mlflow-promote`

### Deliverables

- repeatable release gate for model/prompt changes
- explicit rollback record
- audit trail for why a candidate was promoted

## Data Model and Naming Recommendations

### Params

Good MLflow params for this repo:

- `planner_mode`
- `worker_mode`
- `intent_assess_mode`
- `intent_decompose_mode`
- `chat_response_mode`
- `chat_routing_mode`
- `render_path_mode`
- `capability_mode`
- `capability_top_k`
- `feedback_export_limit`

### Metrics

Good MLflow metrics for this repo:

- intent eval:
  - `intent_f1`
  - `capability_f1`
  - `segment_hit_rate`
- capability search eval:
  - `hit_rate_at_1`
  - `hit_rate_at_3`
  - `mrr`
  - `ndcg`
- feedback eval:
  - `negative_rate`
  - `partial_rate`
  - `chat_helpfulness_rate`
  - `plan_approval_rate`
- runtime summary:
  - `task_success_rate`
  - `plan_failure_rate`
  - `retry_rate`
  - `replan_rate`

### Artifacts

Store these as artifacts:

- eval report JSON
- gold set or dataset manifest
- feedback slice JSONL
- prompt bundle
- release note
- candidate comparison summary

## Security and Self-Hosting Notes

Because the goal is open/self-hosted governance:

- host MLflow inside the same trust boundary as the platform
- use Postgres and object storage you control
- do not send raw secrets, tokens, or user-sensitive memory payloads to MLflow
- log IDs, versions, summaries, and redacted artifacts instead of raw confidential payloads

Follow the same redaction discipline already used for semantic memory work.

## Recommended Implementation Order

Start in this order:

1. Phase 1: self-hosted MLflow plus offline eval logging
2. Phase 3: prompt/model governance metadata
3. Phase 4: feedback-slice lineage
4. Phase 2: runtime lineage
5. Phase 5: release gates

Reason:

- offline eval logging is the fastest win
- registry/governance metadata gives immediate value for candidate tracking
- runtime lineage is useful, but should come after the low-noise offline path is working
- release gates should be built only after the evidence chain is trustworthy

## Immediate First Slice

If only one slice is implemented first, it should be:

1. self-hosted MLflow service
2. `libs/core/mlflow_tracking.py`
3. MLflow logging from:
   - `scripts/eval_intent_decompose.py`
   - `scripts/eval_capability_search.py`
   - `scripts/eval_capability_search_feedback.py`
4. feedback-slice JSONL artifact logging

That gives the platform a working self-hosted governance backbone without touching hot runtime paths yet.
