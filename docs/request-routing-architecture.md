# Request Routing Architecture

This document is a generic implementation plan for a front-door router that directs each user request to the right agent or workflow. It is written to be customized later for a specific repo, but it includes a short appendix that maps the design to the current routing stack in this repository.

The target problem is simple to state and easy to get wrong in practice:

- conversational requests should stay conversational
- safe single-step requests should go to a direct agent or tool path
- multi-step execution requests should go to a generic workflow or job path
- requests for an already published workflow should invoke that workflow directly
- ambiguous requests should be clarified before execution

The v1 design here is correctness-first. It deliberately prefers a safe fallback over an aggressive but brittle router.

## 1. Design Goal

Build a front-door router that converts an incoming request into one of these outcomes:

1. `chat_reply`
2. `direct_agent`
3. `submit_job`
4. `run_workflow`
5. `ask_clarification`

Success means:

- execution-oriented requests rarely die in plain chat
- published workflow invocations are recognized reliably
- risky or underspecified requests are clarified instead of guessed
- policy and governance are enforced before any learned routing choice
- staging can block promotion when routing regresses

## 2. Research-Backed Design Choices

The design borrows from current routing and compound-system work, but adapts it to agent and workflow routing instead of model-only routing.

### 2.1 Use a cascade, not one giant classifier

The router should be hierarchical:

1. boundary decision: chat vs execution
2. execution family decision: direct agent vs generic workflow/job vs published workflow
3. target selection among allowed candidates
4. clarification or fallback when confidence is low

This matches the spirit of modern routing systems: small bounded decisions perform better and are easier to debug than one free-form "do everything" classifier.

### 2.2 Route from evidence, not raw text alone

The router should score from a structured evidence object:

- user message
- normalized goal
- session state
- pending clarification state
- workflow context
- retrieved candidate agents/workflows
- policy constraints
- historical outcome features

This is aligned with the evaluation mindset in RouterBench and with recent routing work that treats routing as a retrieval-plus-selection problem rather than prompt-only intent classification.

### 2.3 Learn reranking and calibration from preference and outcome data

RouteLLM shows that routing quality improves when the router is trained from preference data instead of only hand rules. For request routing, the equivalent signal is:

- human override or correction
- downstream execution success or failure
- latency and cost
- user satisfaction or negative feedback

In practice, the first learned component should be a reranker or calibrator over a deterministic candidate set, not a free-form replacement for the whole stack.

### 2.4 Optimize each module locally before optimizing the full compound system

Optimizing Model Selection for Compound AI Systems argues for per-module optimization inside a larger system. Applied here:

- optimize front-door routing first
- planner-side routing later
- worker-side tool routing later

Do not start by solving all routing surfaces at once.

### 2.5 Add online adaptation only after offline routing is stable

MixLLM and later contextual-bandit work are useful, but only after the system already has:

- stable route labels
- confidence calibration
- safe deterministic fallback
- reliable staging regression gates

Online learning is a phase-2 or phase-3 improvement, not the first deliverable.

## 3. Recommended Architecture

Implement the front-door router as five stages.

### Stage A: Hard Guards

Apply non-negotiable rules before learned routing:

- policy-denied actions cannot be selected
- disabled agents/workflows cannot be selected
- workflow invocation requires an explicit workflow reference
- high-risk requests without required constraints must clarify
- active pending clarification keeps the request in the same execution thread unless the user explicitly exits

This stage must be deterministic.

### Stage B: Boundary Gate

Classify the turn into one of:

- `chat_reply`
- `execution_request`
- `continue_pending`
- `exit_pending_to_chat`
- `meta_clarification`

This is a small bounded decision. It should never decide the exact workflow or agent. It only decides whether the request stays conversational or enters the execution path.

### Stage C: Execution Family Router

If the boundary says execution, choose the family:

- `direct_agent`: exactly one safe synchronous agent/tool path is sufficient
- `submit_job`: generic multi-step execution path
- `run_workflow`: invoke an already published workflow
- `ask_clarification`: execution is needed, but an essential field is missing

This decision should be made from retrieved candidates plus session/workflow state, not from free-form LLM reasoning alone.

### Stage D: Candidate Selection

For the chosen family, rank actual candidates:

- direct agents from an allowlisted direct catalog
- published workflows from workflow metadata and trigger metadata
- generic workflow/job path as the default execution fallback

This stage should emit top-k candidates and confidence, even if only the top-1 is used at runtime.

### Stage E: Calibration and Fallback

Before acting, enforce:

- minimum confidence threshold
- missing-input detection
- deterministic fallback to `submit_job` if the request is clearly execution-oriented but the exact target is uncertain
- deterministic fallback to `ask_clarification` if required fields are missing

The router should almost never fall back to plain `chat_reply` once the boundary has determined that execution is needed.

## 4. Route Taxonomy

Use a small stable route ontology.

```json
{
  "chat_reply": "normal explanation or conversation",
  "direct_agent": "single safe direct capability or agent path",
  "submit_job": "generic workflow or planner/executor path",
  "run_workflow": "invoke a specific published workflow",
  "ask_clarification": "execution needed but required details are missing"
}
```

Keep this taxonomy stable. Add new subtypes only under metadata, not by growing the top-level route enum every time.

## 5. Router Contracts

The router should operate on explicit typed artifacts.

### 5.1 RouteRequest

```json
{
  "request_id": "string",
  "session_id": "string|null",
  "message": "string",
  "candidate_goal": "string",
  "session_state": {},
  "context_json": {},
  "workflow_context": {
    "target_available": false,
    "definition_id": null,
    "version_id": null,
    "trigger_id": null
  },
  "user_context": {},
  "policy_context": {}
}
```

### 5.2 CandidateDescriptor

```json
{
  "candidate_id": "string",
  "candidate_type": "direct_agent | workflow | generic_path",
  "family": "documents | github | workspace | planning | workflow",
  "risk_tier": "read_only | bounded_write | high_risk_write",
  "preconditions": [],
  "input_keys": [],
  "cost_class": "low | medium | high",
  "enabled": true
}
```

### 5.3 RoutingEvidence

```json
{
  "boundary_features": {},
  "retrieved_candidates": [],
  "workflow_target_available": false,
  "pending_clarification": false,
  "missing_inputs": [],
  "historical_success_features": {},
  "policy_filters_applied": []
}
```

### 5.4 RouteDecision

```json
{
  "route": "chat_reply | direct_agent | submit_job | run_workflow | ask_clarification",
  "confidence": 0.0,
  "selected_candidate_id": null,
  "top_k_candidates": [],
  "missing_inputs": [],
  "fallback_used": false,
  "fallback_reason": null,
  "reason_codes": []
}
```

### 5.5 RouteOutcomeFeedback

```json
{
  "request_id": "string",
  "decision_route": "string",
  "selected_candidate_id": "string|null",
  "execution_started": true,
  "execution_succeeded": true,
  "user_override": false,
  "latency_ms": 0,
  "cost_usd": 0.0,
  "feedback_label": "positive | partial | negative | none"
}
```

## 6. Decision Logic

### 6.1 Boundary Logic

Use a bounded boundary model or classifier over a structured prompt or feature object.

Primary features:

- conversation mode hint
- pending clarification state
- workflow target availability
- execution-oriented intent hints
- top capability-family evidence
- missing input count

Primary rule:

- if the request is execution-oriented and not clearly conversational, return `execution_request`

Critical invariant:

- once the boundary predicts execution, later stages may downgrade to clarification or safe generic execution, but should not silently revert to plain chat except for an explicit user opt-out

### 6.2 Execution Family Logic

Use these rules in order:

1. If a valid published workflow reference is present, prefer `run_workflow`.
2. If exactly one safe synchronous direct agent path is sufficient, allow `direct_agent`.
3. If the request is clearly execution-oriented but multi-step, prefer `submit_job`.
4. If execution is needed but required fields are missing, return `ask_clarification`.

Do not use `run_workflow` just because the request could theoretically be implemented as a workflow. It should only be selected when a concrete published workflow target exists.

### 6.3 Candidate Retrieval

Retrieve candidates from:

- direct-agent catalog
- workflow registry
- optional semantic memory or historical routing outcomes

Retrieve only policy-allowed candidates. Retrieval should be family-aware. If the request is document-oriented, document families should dominate the candidate set.

### 6.4 Reranking

Rerank candidates with a small scoring model or bounded LLM over:

- semantic fit to goal
- family fit
- required input coverage
- policy/risk compatibility
- historical success
- expected cost and latency

Start with deterministic weighted scoring plus a learned confidence calibrator. Move to a learned reranker only after enough labeled data exists.

## 7. Training And Learning Loop

### Phase 1: Deterministic + evidence-driven

- hand-built boundary evidence
- deterministic family routing
- retrieved top-k candidates
- confidence from heuristics and bounded scores

### Phase 2: Supervised reranker/calibrator

Train from:

- accepted route decisions
- explicit corrections
- negative feedback
- downstream execution success/failure

Best first learned tasks:

- boundary calibration
- family reranking
- route confidence estimation

### Phase 3: Online adaptation

After the router is stable:

- contextual-bandit adaptation for cost and latency
- tenant- or workload-specific routing preferences
- exploration only inside safe candidate sets

## 8. Evaluation And Promotion Gates

Use three evaluation layers.

### 8.1 Offline Gold Set

Must include:

- conversational messages
- direct safe reads
- document or artifact creation
- system inspection
- explicit published-workflow invocation
- pending clarification continuation
- clarification exit back to chat
- risky or underspecified writes

### 8.2 Replay Set

Replay real traffic and compare:

- predicted route
- selected candidate
- downstream success
- human override or correction

### 8.3 Live Staging Regression

Run a real API-level regression against staging before production promotion.

At minimum, include a case like:

- "Create a document on Kubernetes"

This live gate must fail if the request falls back to:

- plain `respond`
- router failure state
- boundary failure state

Suggested v1 promotion thresholds:

- critical execution-request recall: `>= 0.97`
- false chat-reply rate on execution cases: `<= 0.02`
- published-workflow invocation precision: `>= 0.98`
- clarification precision on missing-required-input cases: `>= 0.95`
- zero policy-bypass events

## 9. Rollout Plan

### Phase 0: Telemetry-only shadow mode

- compute route decisions
- emit evidence and confidence
- do not change live behavior

### Phase 1: Boundary live, family selection shadow

- use the new boundary gate
- keep family routing advisory only

### Phase 2: Family routing live with deterministic fallback

- enable direct-agent and run-workflow selection
- keep low-confidence fallback to generic workflow/job path

### Phase 3: Learned reranker live

- activate learned reranker over already allowed candidate sets
- keep deterministic fallback

### Phase 4: Online adaptation

- add cost and latency adaptation
- only after staging and production telemetry are stable

## 10. Failure Modes To Design For

- provider failure in the boundary or router model
- retrieval failure producing an empty candidate set
- stale session clarification state
- workflow registry mismatch or deleted workflow references
- overconfident chat replies on execution-oriented requests
- direct-agent misfires for requests that are actually multi-step

Required behavior:

- provider failure must degrade to safe deterministic execution or clarification, not silent plain chat when execution evidence is strong
- low-confidence routing must preserve execution intent
- every fallback must emit a machine-readable reason

## 11. What To Build First

If implementing from scratch, build in this order:

1. route taxonomy and typed request/evidence/decision contracts
2. deterministic hard guards
3. boundary evidence builder
4. execution family router
5. candidate retrieval for direct agents and published workflows
6. confidence calibration and deterministic fallback
7. offline gold set and replay eval
8. live staging regression
9. learned reranker
10. online adaptation

This order keeps the system safe while still leaving room for stronger learned routing later.

## 12. Customization Checklist

Before adapting this to a specific repo, decide:

- what counts as a direct agent in your system
- what counts as a generic workflow/job path
- how published workflows are referenced
- which fields are required before execution
- which policy gates are absolute
- which telemetry fields already exist and which must be added

## Appendix A: Mapping To This Repository

This repo already has the right front-door split:

- a boundary decision in `services/api/app/main.py`
- a router decision in `services/api/app/main.py`
- chat session and workflow handoff logic in `services/api/app/chat_service.py`
- offline boundary evaluation in `eval/chat_boundary_gold.yaml` and `scripts/eval_chat_boundary.py`
- a live boundary regression harness in `eval/chat_boundary_live_regression.yaml` and `scripts/eval_chat_boundary_live.py`

Recommended repo-specific interpretation:

- keep the current boundary step, but make it strictly decide chat vs execution
- keep the current router step, but make it evidence-driven and candidate-aware
- treat `submit_job` as the safe default execution fallback
- treat `run_workflow` as valid only when workflow context already identifies a published workflow target
- treat direct one-step capability execution as a tightly allowlisted path

The main implementation gap is not the existence of a router. It is making the router:

- candidate-aware
- confidence-calibrated
- failure-tolerant
- promotion-gated with live regressions

## References

- RouteLLM: Learning to Route LLMs with Preference Data. arXiv 2024. https://arxiv.org/abs/2406.18665
- RouterBench: A Benchmark for Multi-LLM Routing System. arXiv 2024. https://arxiv.org/abs/2403.12031
- Optimizing Model Selection for Compound AI Systems. arXiv 2025. https://arxiv.org/abs/2502.14815
- MixLLM: Dynamic Routing in Mixed Large Language Models. NAACL 2025. https://aclanthology.org/2025.naacl-long.545/
- Online Multi-LLM Selection via Contextual Bandits under Unstructured Context Evolution. arXiv 2025. https://arxiv.org/abs/2506.17670
