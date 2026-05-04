# Adaptive Planning Tradeoffs for Agentic Workflow Studio

This document is a decision memo for how to evolve the current planner and orchestrator from basic adaptive replanning into a deeper adaptive planning system.

It is written for this repo as it exists today. It is not a generic AI-agents note.

## Executive Summary

The repo already has a usable adaptive replanning shell:

- jobs and workflow runs can opt into `planning_mode = adaptive`
- the API tracks `pending_replan`, `replan_count`, `max_replans`, and revision history
- the orchestrator can trigger replans after intent mismatch or retry exhaustion
- unfinished tails of the prior plan are superseded while completed work is retained in metadata/debugger history

What the repo does not have yet is a planner that treats previous execution state as a first-class input. Today, the planner mostly sees:

- the original job payload
- normalized intent and capability hints
- a limited `intent_mismatch_recovery` block
- raw metadata embedded in `job_payload`

That means the current system can replan, but it does not yet reason deeply about:

- what succeeded and should be reused
- what specifically failed and why
- whether to retry, switch capability, patch only the remaining suffix, or stop for human input
- budgets, confidence, or recovery strategy as distinct planning inputs

## Recommendation

The right path for this repo is:

1. add first-class execution-state inputs to the planner contracts
2. add a deterministic replan controller before the planner
3. add delta replanning so the system can preserve successful prefixes and patch only the broken suffix
4. add evaluator and checkpoint support only after steps 1-3 are stable

Do not jump straight to a large multi-agent planner architecture. That is the highest-cost option, the hardest to evaluate, and the easiest way to add latency and debugging pain before the core state model is sound.

## Current State in This Repo

### What is already implemented

- Adaptive job and workflow-run mode in `services/api/app/main.py`
- `AdaptiveReplanStatus`, `PlanRevisionSummary`, and planning-mode fields in `libs/core/models.py`
- automatic replan triggers for:
  - contract or intent mismatch
  - retry exhaustion on adaptive jobs
- manual `POST /jobs/{job_id}/replan`
- revision activation and history tracking
- preservation of prior run-step records so debugger history survives replans

### Where the current boundary is

The planner request contract in `libs/core/planner_contracts.py` includes:

- `job_context`
- `job_metadata`
- `job_payload`
- capabilities
- normalized intent envelope and graph
- semantic capability hints

The planner prompt in `services/planner/app/planner_service.py` explicitly uses:

- normalized intent envelope
- goal intent graph
- semantic capability hints
- `intent_mismatch_recovery`

It does not yet explicitly reason over a typed execution-state object such as:

- completed steps
- failed step classification
- remaining objectives
- recovery strategy
- budgets
- confidence
- human approvals or checkpoints

## Decision Criteria

Use these criteria to decide what to implement first:

- **Recovery quality**: does the system choose the right repair action?
- **Prefix reuse**: does it preserve successful work instead of regenerating everything?
- **Debuggability**: can you explain why a replan happened and what changed?
- **Operational safety**: can you bound retries, cost, and side effects?
- **Latency and cost**: how many extra model calls and control stages are added?
- **Implementation blast radius**: how many core runtime semantics must change?
- **Evaluation readiness**: can you measure whether the new behavior is actually better?

## The Main Tradeoffs

### 1. Metadata-driven replanning vs typed execution-state contracts

**Option A: keep recovery context in job metadata**

Pros:

- fastest incremental change
- minimal schema churn
- low migration cost
- compatible with current event flow

Cons:

- planner behavior depends on prompt reading raw JSON blobs
- harder to validate and test
- weak contract between orchestrator and planner
- easy to accumulate ambiguous or stale fields

When to choose it:

- only if you want a short-term patch and do not expect planning complexity to grow much

**Option B: add typed execution-state contracts**

Pros:

- planner receives explicit state with stable meaning
- easier to validate, test, and evolve
- supports deterministic controller logic before prompt construction
- makes future evaluator and delta-replan work much cleaner

Cons:

- requires model and contract changes across API and planner
- more migration work up front

Recommendation:

- choose Option B

### 2. Full-plan regeneration vs suffix patching

**Option A: regenerate the whole plan**

Pros:

- easiest mental model
- simple planner interface
- fewer plan-diff rules

Cons:

- throws away successful planning structure
- more churn in task IDs and debugger history
- increases chance of regression in already-correct parts
- wastes tokens and time

**Option B: patch the remaining suffix only**

Pros:

- preserves successful prefix
- reduces planner churn and runtime disruption
- better debugger continuity
- lower token cost once implemented

Cons:

- needs explicit change semantics
- harder compiler logic
- requires dependency and artifact-compatibility checks

Recommendation:

- start with full regeneration plus explicit prefix preservation
- then move to suffix patching once first-class execution state is in place

### 3. Planner decides recovery strategy vs controller decides recovery strategy

**Option A: send failure context to the planner and let it decide everything**

Pros:

- flexible
- fewer deterministic branches to implement

Cons:

- harder to control
- harder to explain
- prone to inconsistent decisions for similar failures
- more difficult to evaluate

**Option B: add a deterministic replan controller**

Controller outputs one of:

- `continue`
- `retry_same_step`
- `switch_capability`
- `patch_suffix`
- `full_replan`
- `pause_for_human`

Pros:

- better control and observability
- easy to attach budgets and policy
- allows cheap decisions without always calling the planner
- aligns with how strong production systems separate policy from generation

Cons:

- more orchestration code
- controller logic can become too heuristic if not reviewed carefully

Recommendation:

- choose Option B
- keep the controller deterministic first
- only later consider a model-assisted controller in shadow mode

### 4. Single planner model vs specialized multi-role planner

**Option A: one planner model with better inputs**

Pros:

- lowest complexity increase
- easiest debugging story
- best near-term fit for this repo
- enough for most workflow automation use cases

Cons:

- may plateau on long-horizon or adversarial planning tasks
- mixes decomposition, repair, and validation in one prompt

**Option B: specialized roles such as actor, monitor, evaluator, predictor**

Pros:

- can improve planning quality on harder multi-step tasks
- separates concerns
- closer to recent public research patterns such as modular planners

Cons:

- more latency and cost
- much harder eval design
- larger prompt and state-management burden
- easy to overbuild before the core runtime is ready

Recommendation:

- stay with a single planner plus deterministic controller for now
- revisit multi-role planning only after evaluator, checkpoints, and delta-replan are stable

### 5. Replan only on hard failures vs replan on confidence and acceptance signals

**Option A: replan only after failure**

Pros:

- simple trigger model
- easier to explain
- lower model-call volume

Cons:

- late recovery
- may waste attempts on doomed paths
- ignores weak but important signals such as low-confidence outputs or failed acceptance checks

**Option B: allow replan from evaluator or acceptance-stage signals**

Pros:

- better quality control
- more proactive repair
- aligns with evaluator-optimizer patterns used publicly by major vendors

Cons:

- can increase churn if thresholds are weak
- needs robust acceptance engine and scoring

Recommendation:

- start with failure-based triggers plus one or two high-confidence evaluator triggers
- do not make every soft signal a replan trigger

### 6. Stateless replanning vs procedural memory

**Option A: treat each replan as independent**

Pros:

- simple
- avoids stale memory contamination

Cons:

- repeats the same mistakes
- misses repair patterns that worked before

**Option B: add procedural memory from execution traces**

Pros:

- enables repair strategies based on prior similar failures
- better long-term system learning
- useful for tenant- or workflow-specific adaptation

Cons:

- needs ranking, freshness, and safety rules
- easy to inject noisy or outdated traces into prompts

Recommendation:

- do not start here
- add this after first-class execution state and evaluator loops are working

### 7. Planner-side adaptive reasoning only vs runtime-level checkpointing

**Option A: improve planner only**

Pros:

- smaller initial scope
- planner changes are easier than full runtime changes

Cons:

- runtime still restarts too much
- long-running steps remain fragile
- replay quality stays limited

**Option B: add checkpoint and replay support**

Pros:

- much better resilience for long or side-effectful steps
- enables deeper adaptation without losing progress
- aligns with the repo's broader canonical-run direction

Cons:

- materially larger runtime project
- needs durable checkpoint semantics and executor cooperation

Recommendation:

- phase this after controller and delta planning

### 8. Immediate best-in-class rewrite vs staged convergence

**Option A: build the full target architecture now**

This would likely include:

- canonical run model everywhere
- single scheduler
- evaluator stage
- checkpointing
- routing
- adaptive planner contracts
- delta replan

Pros:

- cleaner long-term architecture
- fewer transitional seams

Cons:

- highest delivery risk
- long time to first value
- difficult parallel migration
- more moving parts to debug at once

**Option B: staged convergence**

Pros:

- lower risk
- each phase is measurable
- easier to stop after enough value is achieved

Cons:

- temporary duplication
- some transitional code remains for a while

Recommendation:

- choose staged convergence

## Implementation Options

### Option 0: Small patch only

Scope:

- improve planner prompt wording
- pass more replan metadata into prompt
- add a few more failure-trigger rules

Effort: Low  
Benefit: Low to Medium  
Risk: Low  
When to choose: only if you need a quick demo improvement

Why not recommended:

- it makes the prompt smarter without improving the contract
- it does not materially improve explainability or reuse

### Option 1: First-class adaptive planner inputs

Scope:

- add typed `PlanRevisionContext` or `ExecutionState`
- include completed steps, failed step, reason, remaining goals, budgets, revision metadata
- update prompt builder and validators to use this contract explicitly

Effort: Medium  
Benefit: High  
Risk: Medium  
When to choose: best next step for this repo

Expected code touchpoints:

- `libs/core/models.py`
- `libs/core/planner_contracts.py`
- `services/planner/app/planner_service.py`
- `services/api/app/main.py`
- planner and API tests

### Option 2: Replan controller plus strategy selection

Scope:

- add deterministic controller before planner
- classify failure and choose strategy
- avoid planner calls for cases that should just retry or pause

Effort: Medium  
Benefit: High  
Risk: Medium  
When to choose: immediately after Option 1

Expected code touchpoints:

- `services/api/app/main.py`
- possibly a new module such as `services/api/app/replan_controller.py`
- tests covering trigger, classification, and chosen action

### Option 3: Delta replanning and suffix patching

Scope:

- planner returns change intent, not just a fresh plan
- preserve prior successful prefix
- cancel and replace only the broken suffix

Effort: High  
Benefit: High  
Risk: Medium to High  
When to choose: after Options 1 and 2 prove stable

Expected code touchpoints:

- planner contracts and planner output validation
- plan revision activation logic in `services/api/app/main.py`
- debugger and run-step continuity logic

### Option 4: Evaluator, acceptance, and checkpoint-aware adaptive runtime

Scope:

- evaluator can trigger replans or rework
- acceptance policy becomes a real control point
- long-running steps gain checkpoints and replay support

Effort: High  
Benefit: Very High for production-grade workflows  
Risk: High  
When to choose: once you are ready to invest in runtime-quality infrastructure, not just planner quality

Expected code touchpoints:

- scheduler and run model
- acceptance engine
- step attempts and checkpoint persistence
- debugger and metrics surfaces

### Option 5: Full modular multi-agent planner

Scope:

- separate actor, monitor, evaluator, predictor, orchestrator roles
- potentially multiple model instances per replan

Effort: Very High  
Benefit: uncertain until proven by evals  
Risk: Very High  
When to choose: only if simpler architectures clearly plateau on your target workloads

Why not recommended now:

- the repo does not yet have the state model and eval harness needed to justify it
- this is where teams often buy complexity before they buy reliability

## Decision Matrix

| Option | Quality Gain | Delivery Speed | Runtime Cost | Debuggability | Architectural Fit | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 0. Prompt patch only | Low | High | Low | Low | Medium | No |
| 1. First-class planner state | High | Medium | Low | High | High | Yes |
| 2. Replan controller | High | Medium | Low | High | High | Yes |
| 3. Delta replanning | High | Medium-Low | Medium | Medium | High | Yes, after 1-2 |
| 4. Evaluator + checkpoints | Very High | Low | Medium-High | High | Very High | Yes, later |
| 5. Full modular planner | Unknown-High | Low | High | Low-Medium | Medium | Not now |

## Recommended Phased Plan

### Phase 1: First-class execution state

Add a typed object passed into the planner, for example:

```python
class FailedStepContext(BaseModel):
    task_id: str | None = None
    task_name: str | None = None
    capability_id: str | None = None
    error_message: str | None = None
    failure_category: str | None = None
    retryable: bool = False
    attempt_number: int = 0
    max_attempts: int = 0


class CompletedStepContext(BaseModel):
    task_id: str
    name: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)


class PlanRevisionContext(BaseModel):
    revision_number: int = 0
    active_plan_id: str | None = None
    trigger_reason: str | None = None
    completed_steps: list[CompletedStepContext] = Field(default_factory=list)
    failed_step: FailedStepContext | None = None
    remaining_goals: list[str] = Field(default_factory=list)
    budgets: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    human_feedback: list[dict[str, Any]] = Field(default_factory=list)
```

Goals of this phase:

- explicit planner input contract
- better prompt quality
- better tests
- no major runtime rewrite yet

### Phase 2: Deterministic replan controller

Add a controller that classifies failures and chooses a strategy before invoking the planner.

Example logic:

- missing required user input -> `pause_for_human`
- transient infra error -> `retry_same_step`
- capability-intent mismatch -> `switch_capability` or `patch_suffix`
- retry exhaustion with reusable prefix -> `patch_suffix`
- broad objective change or corrupted state -> `full_replan`

Goals of this phase:

- fewer unnecessary replans
- explainable repair decisions
- lower cost and lower churn

### Phase 3: Delta plan output

Extend planner output so it can declare:

- steps to preserve
- steps to cancel
- steps to add
- dependency rewrites

Goals of this phase:

- preserve IDs and history where possible
- reduce churn in debugger and execution state
- improve recovery speed

### Phase 4: Evaluator and checkpoints

Add:

- acceptance-stage replan triggers
- checkpointed replay for long or side-effectful steps
- stronger runtime metrics

Goals of this phase:

- production-grade resilience
- bounded autonomy
- quality control earlier than terminal failure

## What Top Companies Publicly Converge On

Across current public materials from OpenAI, Anthropic, Google, and Microsoft, the common pattern is not "one very smart planner prompt." It is a closed-loop system:

`observe -> evaluate -> decide -> act -> verify -> replan if needed`

Common themes:

- schema-first tool or function contracts
- application-owned execution, not model-owned execution
- iterative loops with real environment feedback
- bounded autonomy with stops, budgets, and policies
- durable traces and observability
- increasing use of evaluators, monitors, or multi-role control only when simpler loops are not enough

What this means for this repo:

- the next step is not a bigger planner prompt
- the next step is better state, better control, and better evaluation

## What Not To Build First

Avoid these as the first move:

- learned routing policies before deterministic state and evals exist
- a multi-agent planner before single-planner contracts are strong
- checkpointing before replan strategies are clear
- broad memory injection from past traces without ranking and freshness rules
- automatic replanning from weak confidence signals without calibrated thresholds

These all become much easier once execution state and controller logic are explicit.

## Concrete Repo Changes by Priority

### Highest-priority changes

1. Add `PlanRevisionContext`-style models in `libs/core/models.py`
2. Extend `PlanRequest` in `libs/core/planner_contracts.py`
3. Build that state explicitly in `services/api/app/main.py`
4. Update `services/planner/app/planner_service.py` prompt construction and parsing
5. Add targeted tests for:
   - contract construction
   - controller decisions
   - prefix preservation
   - replan history integrity

### Second-priority changes

1. introduce a replan-controller module
2. add failure taxonomy and deterministic strategy mapping
3. expose repair strategy in debugger output
4. track metrics such as:
   - replan trigger rate
   - success after replan
   - prefix reuse rate
   - unnecessary full-replan rate
   - mean time to recovery

### Later changes

1. delta plan patch format
2. evaluator-driven replan triggers
3. checkpoint and replay
4. procedural memory from prior repair traces

## Recommended Decision

If your goal is **better adaptive planning soon without destabilizing the platform**, implement:

- Option 1 now
- Option 2 immediately after
- Option 3 only when you have tests and debugger visibility for revisions

If your goal is **best-in-class production orchestration over the next major architecture cycle**, use the above as the entry path into Option 4, and align it with the broader canonical-run roadmap already described in `docs/best-in-class-agentic-orchestration-implementation-plan.md`.

If your goal is **maximum novelty or research value**, a modular multi-role planner is valid as an experiment, but it should run in shadow mode against the simpler architecture first. It should not be the main delivery path.

## Sources

These are the public materials most relevant to the tradeoffs above:

- OpenAI, "A practical guide to building agents": https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/
- OpenAI, "From model to agent: Equipping the Responses API with a computer environment" (March 11, 2026): https://openai.com/index/equip-responses-api-computer-environment
- OpenAI, "Introducing deep research" (February 2, 2025; updated February 10, 2026): https://openai.com/index/introducing-deep-research/
- Anthropic, "Building effective agents" (December 19, 2024): https://www.anthropic.com/engineering/building-effective-agents
- Microsoft Learn, "Planning" for Semantic Kernel (last updated June 11, 2025): https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning
- Microsoft Research, "A brain-inspired agentic architecture to improve planning with LLMs" (December 9, 2025) and related publication: https://www.microsoft.com/en-us/research/video/a-brain-inspired-agentic-architecture-to-improve-planning-with-llms/
- Google, "Try Deep Research and our new experimental model in Gemini, your AI assistant" (December 11, 2024): https://blog.google/products/gemini/google-gemini-deep-research/
- Google, "Build with Gemini Deep Research" (December 11, 2025): https://blog.google/innovation-and-ai/technology/developers-tools/deep-research-agent-gemini-api/

## Bottom Line

For this repo, the best next implementation is not "make the planner more agentic" in the abstract.

It is:

- make replan state explicit
- separate repair decisioning from plan generation
- preserve successful work
- measure whether recovery actually improved

That gives you a deeper adaptive planner without buying unnecessary architecture too early.
