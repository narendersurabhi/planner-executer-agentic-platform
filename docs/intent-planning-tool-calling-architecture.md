# Intent Parsing and Tool-Calling Planner Architecture

This document proposes a repo-specific architecture for implementing:

- intent parsing
- capability-aware planning
- LLM tool calling during planning
- canonical `RunSpec` compilation

It is intentionally written for the current Agentic Workflow Studio codebase. It
is not a generic "AI planner" document.

It assumes the current split across:

- `services/api`
- `services/planner`
- `services/worker`
- `libs/core/intent_contract.py`
- `libs/core/planner_contracts.py`
- `libs/core/run_specs.py`
- `config/capability_registry.yaml`

## 1. Executive Summary

The right model for this repo is:

1. parse user goal into a normalized intent graph
2. retrieve allowed capability candidates for each intent segment
3. let the planner use tool calling over planner-only support tools
4. compile the result into canonical `RunSpec`
5. validate contracts and runtime conformance before execution

Key opinionated choices:

- Treat `capabilities` as the planner's public vocabulary.
- Treat runtime `tools` as private executor implementation details.
- Use LLM tool calling for planning support, not for real execution.
- Keep `intent parsing` and `intent contract validation` separate concerns.
- Make `RunSpec` the only planner output that matters at runtime.
- Keep deterministic fallback and validation around every LLM step.

The planner should not directly "choose tools and execute them". It should
choose capabilities, inspect their contracts, and produce a valid compiled run.

## 2. Problem To Solve

Today the repo has the right building blocks, but they are spread across
heuristics, planner validation, chat routing, and runtime enforcement.

Current issues:

- intent inference exists in multiple places and is partly heuristic
- planner output can still drift from capability contracts
- LLM-generated intent slots can carry bad requirements like `filename` or
  `format` for `GenerateDocumentSpec`
- capability recommendation, intent decomposition, and planner compilation are
  related, but not yet one clean compiler pipeline
- planner output still risks leaking tool-level details instead of staying at
  capability level

The result is a fragile system where:

- intent parsing can be inconsistent
- planner output may be syntactically valid but semantically wrong
- execution failures happen later than they should

## 3. Design Principles

### 3.1 Intent is compile-time metadata

Intent parsing should shape planning and clarification, but should not become a
second runtime state machine.

Intent parsing should produce:

- normalized task intent
- goal intent graph
- slot hints
- clarification requirements
- ranked capability candidates

It should not directly decide execution order or bypass `RunSpec`.

### 3.2 Capabilities are public, tools are private

The planner prompt and tool-calling surface should reason about:

- capability ids
- capability descriptions
- input contracts
- output contracts
- risk tiers
- governance visibility

It should not reason about:

- local Python handler names
- internal tool registry implementation
- adapter-local wiring details

### 3.3 LLMs propose, validators decide

The LLM should be allowed to:

- classify intent
- decompose a goal into segments
- select relevant capabilities from allowed candidates
- propose step ordering and bindings

The LLM should not be trusted to:

- invent required inputs
- redefine capability schemas
- bypass governance
- emit runtime-executable plans without validation

### 3.4 Tool calling is for planning support only

Tool calling should be used by the planner to query planning state, not to do
real work.

Good planner tools:

- `search_capabilities`
- `get_capability_contract`
- `get_schema`
- `get_workflow_hints`
- `get_memory_hints`
- `finalize_run_spec`

Bad planner tools:

- `document.pdf.render`
- `github.repo.list`
- `workspace.write`
- any tool that performs real side effects during planning

### 3.5 One canonical compiled artifact

The only planner output that should matter to runtime is `RunSpec`.

Everything else is auxiliary:

- `goal_intent_profile`
- `goal_intent_graph`
- planner scratchpad
- planner tool-call trace

Those are useful for observability, but not for execution authority.

## 4. Target Architecture

### 4.1 Goal Intake Layer

The API remains the entrypoint for:

- chat turns
- `POST /jobs`
- workflow compile/publish/run

The API should:

- normalize goal text and context
- collect user identity and tenant context
- decide whether clarification is required
- attach prior workflow/memory hints

This is already close to `services/api/app/intent_service.py` plus
`services/api/app/main.py`.

### 4.2 Intent Parsing Layer

Intent parsing should have two outputs:

1. `GoalIntentProfile`
   Used for clarification, gating, and conversational routing.
2. `IntentGraph`
   Used for planning-time decomposition and capability candidate generation.

Recommended behavior:

- use a fast heuristic parser as baseline
- optionally use LLM assessment/decomposition with allowed capability context
- normalize through shared logic in `libs/core/intent_contract.py`
- never persist raw LLM slots without normalization

The intent layer should be the only place that turns free-form text into:

- `generate`
- `transform`
- `validate`
- `render`
- `io`

and into slots such as:

- `artifact_type`
- `entity`
- `output_format`
- `risk_level`
- `must_have_inputs`

### 4.3 Capability Retrieval Layer

For each goal segment, retrieve a bounded candidate set from the capability
catalog.

Inputs:

- segment objective
- normalized intent
- user/tenant context
- governance filters
- optional semantic memory hints

Outputs:

- top-k capability candidates
- ranked reasons
- confidence metadata

This layer should run before planner tool calling so the model never sees the
entire universe of runtime possibilities when a much smaller candidate set is
enough.

Recommended sources:

- `config/capability_registry.yaml`
- semantic search over capability descriptions
- successful workflow history
- memory/workflow hints for the same user or tenant

### 4.4 Planner Tool-Calling Layer

The planner should run as a bounded tool-calling loop over planner support
tools.

Suggested planner support tool surface:

#### `search_capabilities`

Input:

- `query`
- `intent_hint`
- `limit`

Output:

- ranked capability ids
- descriptions
- risk tiers
- match reasons

#### `get_capability_contract`

Input:

- `capability_id`

Output:

- description
- risk tier
- idempotency
- input schema ref
- output schema ref
- governance visibility
- adapter summary

#### `get_schema`

Input:

- `schema_ref`

Output:

- JSON schema
- human summary
- required fields

#### `get_workflow_hints`

Input:

- `goal`
- `user_id`
- `limit`

Output:

- prior successful shapes
- common step orderings
- common capability sequences

#### `finalize_run_spec`

Input:

- candidate steps
- bindings
- retry policy
- acceptance policy

Output:

- validated `RunSpec` candidate or structured validation errors

This loop should be short and bounded:

- max iterations
- max tool calls
- fallback path on malformed output

The planner should not directly emit opaque JSON and hope downstream code fixes
it later.

## 5. Compiler Pipeline

The compiler pipeline should look like this:

```text
Goal + Context
  -> GoalIntentProfile
  -> IntentGraph
  -> Capability Candidate Retrieval
  -> Planner Tool-Calling Loop
  -> Draft Step Graph
  -> RunSpec Compiler
  -> Contract Validation
  -> Runtime Conformance Validation
  -> Persisted RunSpec
```

In more detail:

### Stage A: Assess

Produce:

- top-level intent
- clarification questions
- risk estimate

If blocking clarification is required, stop here.

### Stage B: Decompose

Produce:

- ordered intent segments
- slot hints
- suggested capabilities

### Stage C: Retrieve

For each segment:

- retrieve bounded capability candidates
- retrieve contract/schema summaries
- retrieve workflow/memory hints

### Stage D: Plan with tool calling

The planner model iteratively:

- asks for more capability/schema detail when needed
- chooses capabilities
- defines step ordering
- defines bindings and gates
- asks to finalize

### Stage E: Compile

Compile into canonical `RunSpec` and `StepSpec`.

### Stage F: Validate

Run:

- intent contract validation
- capability schema validation
- dependency validation
- runtime conformance validation

Only after all of that should a run be accepted for execution.

## 6. Data Contracts

### 6.1 GoalIntentProfile

Purpose:

- clarify or gate ambiguous requests
- help chat decide `respond` vs `submit_job` vs `tool_call`

Key fields:

- `intent`
- `confidence`
- `risk_level`
- `missing_slots`
- `questions`

### 6.2 IntentGraph

Purpose:

- planning-time decomposition
- segment-scoped capability retrieval

Key fields:

- `segments[].intent`
- `segments[].objective`
- `segments[].required_inputs`
- `segments[].suggested_capabilities`
- `segments[].slots`

### 6.3 CapabilityCandidate

Purpose:

- bounded planner context
- explainable retrieval

Key fields:

- `capability_id`
- `description`
- `risk_tier`
- `input_schema_ref`
- `output_schema_ref`
- `score`
- `reason`

### 6.4 PlannerStepDraft

Purpose:

- planner scratch output before canonical compilation

Key fields:

- `name`
- `intent`
- `capability_id`
- `input_bindings`
- `depends_on`
- `expected_output_schema_ref`
- `acceptance_policy`

### 6.5 RunSpec

Purpose:

- canonical compiled runtime contract

Key fields:

- run metadata
- ordered `StepSpec`s
- dependencies
- capability requests
- input bindings
- retry policy
- acceptance policy

## 7. Validation Model

Validation should happen in layers.

### 7.1 Intent contract validation

Validate that:

- segment intent matches step intent
- segment slots are normalized
- impossible `must_have_inputs` are stripped or rejected
- planner output does not confuse generation vs render vs validation semantics

This is where `libs/core/intent_contract.py` belongs.

### 7.2 Capability contract validation

Validate that:

- capability exists
- payload shape matches schema
- bindings can produce required inputs
- risk tier is allowed

### 7.3 Runtime conformance validation

Validate that:

- runtime manifest advertises the capability
- governance allows it in the selected runtime
- required adapters and secrets exist

### 7.4 Final compile validation

Validate that:

- DAG is acyclic
- dependencies exist
- output schemas align
- `RunSpec` is serializable and stable

## 8. Chat, Planner, and Studio Positioning

This architecture should unify all three lanes.

### Chat

Chat uses:

- `GoalIntentProfile` for routing and clarification
- `IntentGraph` when the turn becomes a job
- direct capability execution only for safe one-step reads

### Planner-led jobs

Planner-led jobs use the full compiler pipeline:

- assess
- decompose
- retrieve
- tool-call planner
- compile `RunSpec`

### Studio

Studio may skip goal decomposition when the workflow is already authored, but it
should still use the same:

- capability contracts
- `RunSpec` compiler
- runtime conformance checks

## 9. Observability

We should be able to explain:

- why the parser chose an intent
- why a segment got certain slots
- why a capability was retrieved or rejected
- which planner tool calls happened
- why the final `RunSpec` passed or failed validation

Recommended artifacts:

- `goal_intent_profile`
- `goal_intent_graph`
- planner tool-call trace
- capability candidate rankings
- compile diagnostics
- validation diagnostics

These should be queryable from Postgres-backed run/debugger APIs where possible.

## 10. Failure Handling

Expected failure classes:

### 10.1 Low-confidence intent

Response:

- ask clarification
- do not plan yet

### 10.2 No good capability candidates

Response:

- ask clarification or fail compile
- surface candidate search diagnostics

### 10.3 LLM planner malformed output

Response:

- retry with bounded repair
- fall back to deterministic or heuristic plan synthesis
- never bypass validation

### 10.4 Contract mismatch

Response:

- fail compile early
- emit exact mismatch details

### 10.5 Runtime conformance failure

Response:

- block publish or run start
- do not let the scheduler discover this later

## 11. What The Planner Should Not Do

The planner should not:

- execute real tools while planning
- mutate external systems
- depend on worker-only runtime context
- emit raw implementation tool names as public plan contracts
- invent schema fields not present in capability contracts
- rely on downstream code to clean up obviously wrong intent slots

## 12. Repo Mapping

Recommended ownership by module:

- `services/api/app/intent_service.py`
  - goal assessment
  - intent decomposition orchestration
  - clarification policy

- `libs/core/intent_contract.py`
  - normalization
  - slot cleanup
  - intent compatibility validation

- `services/planner/app/planner_service.py`
  - planner tool-calling loop
  - compile orchestration
  - planner-specific repair/fallback

- `libs/core/planner_contracts.py`
  - planner request/response contracts
  - candidate/tool-call data models

- `libs/core/run_specs.py`
  - canonical compilation target

- `config/capability_registry.yaml`
  - planner-visible capability vocabulary

## 13. Incremental Implementation Plan

### Phase 1: Unify intent parsing

- keep one shared normalization path
- keep heuristic parser as baseline
- add optional LLM assessment/decomposition with capability context

### Phase 2: Add planner support tools

- create planner-only tool surface for capability search and contract lookup
- keep runtime execution tools out of the planner prompt

### Phase 3: Planner emits `RunSpec`

- stop treating raw plan JSON as final
- make tool-calling planner compile through `RunSpec`

### Phase 4: Add full compile diagnostics

- persist candidate rankings
- persist tool-call trace
- persist contract validation failures

### Phase 5: Retire plan-time raw tool leakage

- move planner prompts and outputs fully to capability vocabulary
- keep runtime tools private to executors

## 14. Recommended First Implementation Step

If only one thing is implemented next, it should be:

`planner support tools + RunSpec-first compile`

That has the highest leverage because it:

- reduces planner hallucination
- aligns planning with capability contracts
- keeps tool calling useful without letting planning execute side effects
- moves the system toward one canonical runtime contract
