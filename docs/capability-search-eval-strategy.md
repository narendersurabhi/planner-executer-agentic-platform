# Capability Search Evaluation Strategy

## Goal
Create a measurable loop for `capability search -> planner choice -> execution outcome -> search improvement`.

## Phase 1
1. Define a gold evaluation set in `eval/capability_search_gold.jsonl`.
2. Validate the case contract with `schemas/capability_search_eval_case.json`.
3. Run an offline evaluator via `scripts/eval_capability_search.py`.
4. Track baseline metrics:
   - `hit_rate@1`
   - `hit_rate@3`
   - `recall@5`
   - `mrr`
   - `ndcg`
   - `must_have_hit_rate`

## Phase 2
1. Emit capability-search telemetry from:
   - `POST /capabilities/search`
   - intent decomposition semantic hints
   - planner semantic hint generation
2. Record:
   - query
   - intent
   - retrieved top-k
   - scores
   - latency
   - source

## Phase 3
1. Join search events with:
   - planner selections
   - executed capabilities
   - run outcomes
2. Build hard-negative datasets from:
   - retrieved-but-not-selected results
   - selected-but-failed capabilities
   - planner overrides

## Phase 4
1. Add dashboards for:
   - capability search latency
   - hit rate by source
   - planner selection rate
   - execution success rate
   - fallback rate
2. Gate changes on offline metrics before deployment.

## Fine-Tuning Guidance
1. Improve retrieval features first:
   - description
   - tags
   - required inputs
   - group/subgroup
   - intent-aware boosts
2. Add reranking before model fine-tuning.
3. Fine-tune only after production feedback has enough hard negatives.

## Immediate Deliverables
1. `eval/capability_search_gold.jsonl`
2. `schemas/capability_search_eval_case.json`
3. `libs/core/capability_search_eval.py`
4. `scripts/eval_capability_search.py`
5. CI/test hooks and Make targets
