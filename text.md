Good next steps, in priority order:

1. **Build an intent eval harness first**  
Create a small gold set of goals + expected intent graph + expected capability IDs, and score precision/recall over time.  
This is the control loop for every future change.

2. **Constrain intent output to capability catalog**  
In `POST /intent/decompose`, force `suggested_capabilities` to valid IDs only (no free-form capability names), with top-k ranking + reasons.

3. **Add intent schema v2 (slots + constraints)**  
Extend each segment with normalized slots like `entity`, `artifact_type`, `output_format`, `risk_level`, `must_have_inputs`.  
Use this contract in planner and worker, not just free text.

4. **Calibrate confidence and gating thresholds**  
Track confidence vs actual success and tune `INTENT_MIN_CONFIDENCE` by intent type/risk tier instead of one global threshold.

5. **Improve clarification from generic Q&A to targeted slot-filling**  
Ask only for missing high-impact slots (for example `output_format`, `target_system`, `safety constraints`) and block submit only on blocking slots.

6. **Add mismatch recovery policy**  
When worker sees `contract.intent_mismatch`, auto-replan with a constrained repair prompt before hard-fail.

7. **Close the memory loop for intent**  
Store decomposed intent + selected capabilities + run outcomes in semantic memory; retrieve similar successful chains to improve future decomposition.

8. **Operationalize observability**  
Add dashboards for:
- intent decomposition latency/cost
- mismatch rate
- clarification rate
- plan success after clarification
- capability suggestion hit rate

If you want, I can implement step 2 next (capability-grounded decomposition with strict validation and tests).

Implemented.

**What changed**
- Added capability-grounded normalization and fallback selection for intent decomposition in `services/api/app/main.py:1306`.
- LLM intent-decompose prompt now includes the allowed capability catalog and requires exact IDs in `services/api/app/main.py:1598`.
- Enforced:
  - case-insensitive capability ID canonicalization,
  - max 3 suggested capabilities per segment,
  - auto-fill from catalog when LLM suggestions are invalid/empty.
- Added summary counters for selected/autofilled capability suggestions in `services/api/app/main.py:1578` and defaults in `services/api/app/main.py:1792`.
- Wired catalog data through `_decompose_goal_intent` into LLM decomposition in `services/api/app/main.py:1860`.

**Tests updated**
- Strengthened unknown-capability filtering test and added casing normalization test in `services/api/tests/test_api.py:106` and `services/api/tests/test_api.py:133`.

**Validation run**
- `python3 -m pytest services/api/tests/test_api.py -k "intent_decompose"`  
- Result: `8 passed`.

If you want, I can do the next step now: expose `capability_match_rate` + `autofilled` as badges in the UI Intent Graph panel.

Implemented.

**Capability search eval phase 1**
- Saved the implementation strategy in `docs/capability-search-eval-strategy.md`.
- Added the gold-set contract in `schemas/capability_search_eval_case.json`.
- Added a starter benchmark set in `eval/capability_search_gold.jsonl`.
- Added reusable evaluation logic in `libs/core/capability_search_eval.py`.
- Added the offline evaluator CLI in `scripts/eval_capability_search.py`.
- Added test coverage in `libs/core/tests/test_capability_search_eval.py`.
- Added Make targets:
  - `make eval-capability-search`
  - `make eval-capability-search-gate`

**Metrics included**
- `hit_rate@1`
- `hit_rate@3`
- `recall@5`
- `mrr`
- `ndcg`
- `must_have_hit_rate`

**Next steps**
- Emit capability-search telemetry from API and planner.
- Join retrieval, planner selection, and execution outcomes into a feedback dataset.
- Add dashboards and regression gates before reranking/fine-tuning.

Implemented.

**Capability search telemetry phase**
- Added event types in `libs/core/events.py`:
  - `plan.capability_search`
  - `plan.capability_selection`
- Added telemetry schemas:
  - `schemas/capability_search_event.json`
  - `schemas/planner_capability_selection_event.json`
- Added API-side capability search telemetry in `services/api/app/main.py`:
  - `/capabilities/search` now emits `plan.capability_search`
  - intent semantic-hint retrieval now emits `plan.capability_search`
  - added Prometheus counters:
    - `capability_search_requests_total`
    - `capability_search_results_total`
- Added planner-side telemetry in `services/planner/app/main.py`:
  - planner semantic capability hint retrieval emits `plan.capability_search`
  - successful plans emit `plan.capability_selection`

**Tests**
- Added API telemetry assertions in `services/api/tests/test_api.py`
- Added planner telemetry assertions in `services/planner/tests/test_tool_usage_validation.py`

**Next steps**
- Join capability-search events with planner/execution outcomes into a feedback artifact.
- Add dashboards for retrieval hit rate, planner selection rate, and execution success rate.
- Use joined feedback to build hard negatives for reranking/fine-tuning.

Implemented.

**Capability search feedback + monitoring**
- Added feedback join logic in `libs/core/capability_search_feedback.py`
- Added feedback schema in `schemas/capability_search_feedback_row.json`
- Added feedback builder CLI in `scripts/build_capability_search_feedback.py`
- Added test coverage in `libs/core/tests/test_capability_search_feedback.py`
- Added outcome metrics in `services/api/app/main.py`:
  - `planner_capability_selection_total`
  - `capability_execution_outcomes_total`
- Added Grafana dashboard assets:
  - `docs/grafana/dashboards/capability-search.json`
  - `deploy/k8s/observability/grafana-dashboard-capability-search.yaml`
- Registered the dashboard in `deploy/k8s/observability/kustomization.yaml`
- Added Make target:
  - `make build-capability-feedback`

**Feedback rows include**
- retrieved capabilities
- selected capabilities
- executed capabilities
- retrieved-selected overlap
- retrieved-executed overlap
- planner overrides
- hard negatives

**Next steps**
- Feed hard negatives into a reranker training set.
- Add CI regression checks for feedback-derived planner hit rate.
- Tune search ranking for the current misses (`memory.semantic.write`, `github.pull_request.create`).

Implemented.

**Capability reranker dataset builder**
- Added training example builder in `libs/core/capability_search_training.py`
- Added tests in `libs/core/tests/test_capability_search_training.py`
- Added dataset builder CLI in `training/build_capability_reranker_dataset.py`
- Added schema in `schemas/capability_reranker_training_example.json`
- Added Make target:
  - `make build-capability-reranker-dataset`

**Training example shape**
- `query`
- `positive_capability_id`
- `negative_capability_ids`
- `selected_capabilities`
- `retrieved_capabilities`
- `execution_succeeded`
- `source=feedback`

**Positive/negative policy**
- positives prefer successfully executed retrieved capabilities
- fallback positives use retrieved-selected capabilities
- negatives come from `hard_negative_ids`

Implemented.

**Reranking + feedback eval**
- Added deterministic reranker in `libs/core/capability_reranker.py`
- Wired reranking into `libs/core/capability_search.py`
- Added feedback-eval helper in `libs/core/capability_search_feedback_eval.py`
- Added tests:
  - `libs/core/tests/test_capability_reranker.py`
  - `libs/core/tests/test_capability_search_feedback_eval.py`
  - updated `libs/core/tests/test_capability_search.py`
- Added feedback eval CLI:
  - `scripts/eval_capability_search_feedback.py`
- Added Make target:
  - `make eval-capability-feedback`

**Deterministic reranker policy**
- boosts retrieved capabilities with successful query-matched feedback
- boosts historically selected/executed capabilities
- penalizes failed and hard-negative capabilities
- degrades to base semantic ranking if no feedback rows exist
