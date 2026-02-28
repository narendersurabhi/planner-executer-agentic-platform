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