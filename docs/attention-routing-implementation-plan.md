# Agentic Workflow Studio Attention Router — Implementation Plan (Concrete, Class-Level)

This document translates the architecture proposal into specific code insertion points in the current codebase.

## 1) Entry points for change

- Planner prompt construction: `services/planner/app/main.py`
  - `_llm_prompt(...)`
  - `plan_job(...)`
- Plan persistence and task payload emission: `services/api/app/main.py`
  - `_plan_created_payload(...)`
  - `_handle_plan_created(...)`
  - `_enqueue_ready_tasks(...)`
  - `_task_payload_from_record(...)`
- Worker execution path: `services/worker/app/main.py`
  - `_process_task_ready_message(...)`
  - `execute_task(...)`
- Feature/rerank shared library: `libs/core` (new)

## 2) New shared components to add

Add:
- `libs/core/attention_router.py`
  - `AttentionCandidate` data object
  - `AttentionSignal` data object
  - `AttentionDecision` data object
  - `AttentionConfig` with weights/temperature/top-k
  - `build_candidate_set(...)`
  - `score_candidates(...)`
  - `rerank_candidates(...)`
  - `build_attention_payload(...)`
- `libs/core/attention_metrics.py`
  - helper for scorecard emission to logs/metrics

No external ML dependency required at first. Use numeric heuristics and existing history counters.

## 3) Planner-side implementation (low-risk path)

### 3.1 Candidate set generation

Inside `plan_job(...)`, after registry load and before LLM prompt:

1) Build candidate capabilities from:
- enabled tools (`tool_registry.default_registry`)
- enabled capabilities (`_planner_capabilities()` already used in `_llm_prompt`)

2) For each candidate, compute static features:
- intent compatibility (from `task_intent`/goal intent hints)
- risk level (`tool.risk_level`)
- historical viability (from job metadata if available, else neutral)
- semantic overlap using existing `capability_search` token matching

### 3.2 Inject ranked candidates into planning prompt

In `_llm_prompt(...)`, replace static “allowed names” block with:
- `allowed_tool_names` as guardrail (unchanged)
- `attention_prioritized_tool_catalog` (ordered by new score)
- optional `attention_plan` section containing top-k candidates + reasoning

### 3.3 Persist planner decision context

In `services/planner/app/main.py`:
- Add optional `router_decisions` dictionary returned from `plan_job(...)` and attach to logs.
- Keep the final `models.PlanCreate` unchanged at first for compatibility.
- (Optional stricter phase) add optional `planner_attention` in `models.TaskCreate` metadata once end-to-end works.

## 4) API-side implementation (where worker can consume ranking context)

### 4.1 Persist/forward routing metadata

In `services/api/app/main.py`:

- Extend `_task_payload_from_record(...)` to include:
  - `routing`: `{"planner_rank": [...], "planner_selected": [...], "planner_scores": {...}}`
- Include in task-ready payload emitted from `_enqueue_ready_tasks(...)` without breaking existing consumers.

### 4.2 Optional DB schema extension

For long-lived audit:
- Add `routing_metadata JSON` to `services/api/app/models.py.TaskRecord`.
- Populate from payload during `_handle_plan_created(...)`.

This is optional for phase 1. You can skip and only pass via event payload.

## 5) Worker-side implementation (highest immediate value)

### 5.1 Tool reorder before execution

In `services/worker/app/main.py::_process_task_ready_message(...)`:

1) Call `attention_router.rerank_candidates(...)` using:
   - `task_payload["tool_requests"]`
   - `task_payload["intent"]`, `task_payload["intent_source"]`, `task_payload["intent_confidence"]`
   - `task_payload.get("routing")` from API
   - recent task history from payload context if available

2) Replace `task_payload["tool_requests"]` with ordered list before `execute_task(task_payload)`.

3) Inject fallback:
- If scoring fails or confidence < threshold, keep original list and set:
  - `task_payload["routing"]["fallback_reason"]`

### 5.2 Scoring heads (small to start)

Implement in `libs/core/attention_router.py` as 3 additive heads:
- Relevance head (semantic match or intent overlap)
- Recency head (task age + tenant/job context similarity)
- Risk head (penalize high risk with low confidence tasks)

Score formula:
- `weighted_softmax` for multi-head weighted sum
- output top-k + top-1 confidence + entropy

## 6) Rollout phases

1. **Phase A (planner only, telemetry)**
   - rank candidates for prompt injection only
   - emit decision in logs (`attention_*`)
2. **Phase B (worker only, hard fallback)**
   - reorder `tool_requests` in worker
   - fallback to original order on low confidence
3. **Phase C (planner + worker with feedback loop)**
   - append observed outcomes to rerank feedback source
   - use feedback in scoring

## 7) Code sketch locations (exact function targets)

- `services/planner/app/main.py`
  - `_planner_capabilities` / `_llm_prompt`: add ranked capability context.
  - `plan_job`: gate planner routing toggle and pass config to prompt context.
- `services/api/app/main.py`
  - `_handle_plan_created`: persist routing bundle in task metadata (or payload only).
  - `_enqueue_ready_tasks`: ensure all ready tasks carry routing metadata.
  - `_task_payload_from_record`: populate `routing` field.
- `services/worker/app/main.py`
  - `_process_task_ready_message`: apply reranker and log decision.
  - `execute_task`: optional per-tool gating/debug from `task_payload["routing"]`.

## 8) Env/feature flags

Add env vars in API/Planner/Worker:
- `ATTN_ROUTER_ENABLED=true|false`
- `ATTN_ROUTER_TOP_K=3`
- `ATTN_ROUTER_MIN_CONF=0.55`
- `ATTN_ROUTER_FALLBACK_TO_ORIGINAL=true`
- `ATTN_ROUTER_WEIGHTS=relevance:0.5,risk:0.2,recency:0.3`

## 9) Observability additions

- Add logs in each phase:
  - `attention_rerank_plan`
  - `attention_rerank_task`
  - `attention_fallback`
- Include `candidate_count`, `selected_top`, `confidence`, `entropy`, `fallback_reason`.
- Hook counters in the existing Prometheus path where task events are emitted.

## 10) Migration notes

- Keep output contract unchanged first pass.
- Keep deterministic path stable and reversible.
- If you want, next step is adding schema changes and dashboards only after Phase B is stable.
