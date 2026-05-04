# Instruction-Driven Agents and Capability-Driven Clarification, Planning, and Validation

## Purpose

This document describes how to make agents in this repository behave in a controlled way:

- the user instruction drives what the system is trying to do
- the capability contract drives what the system is allowed to do
- clarification is derived from capability-required fields
- planning is validated against capability and intent contracts
- runtime execution validates capability inputs and outputs

This is the design you want if the goal is:

- fewer planner surprises
- fewer hidden agent behaviors
- fewer late schema failures
- better staging confidence before production promotion

## Short Version

If you want agents to be instruction-driven and schema-governed, enforce these rules:

1. The user instruction is a first-class field, not an implementation detail.
2. The capability registry is the public execution contract.
3. Clarification is generated from capability-required user-collectible fields.
4. The planner speaks in capability language, not raw adapter tool names.
5. Planning is rejected if the plan violates intent or capability contracts.
6. The worker validates capability inputs before execution.
7. Tool handlers validate their own output schema.
8. Task-level expected outputs are validated before the run is accepted as complete.

If any one of those layers is optional, the system will drift.

## The Core Design Principle

There should be a strict split of responsibilities:

- `instruction` defines the requested work
- `capability` defines the allowed interface
- `planner` decomposes the work into valid capability requests
- `worker` executes only validated requests

In one sentence:

The instruction chooses the objective, the capability contract chooses the allowed shape, and the runtime enforces both.

## What "Instruction-Driven" Means

Instruction-driven does not mean "the model saw some prompt text somewhere."

It means:

- the original user goal is persisted
- each task has an explicit instruction
- the instruction survives clarification, planning, and execution
- the system never silently swaps the intent to something else just because an adapter or tool happens to support it

In this repo, the important instruction-bearing fields are:

- job goal
- task instruction
- normalized intent slot values
- clarified user fields merged into context

The instruction must remain visible to:

- intent normalization
- clarification
- planner validation
- runtime execution

## What "Capability-Driven" Means

Capability-driven means the public contract is the capability, not the internal tool implementation.

The capability contract should define:

- stable capability ID
- description
- risk tier
- idempotency
- input schema
- output schema
- enabled adapters

In this repo, the authority is:

- [capability_registry.yaml](/Users/narendersurabhi/planner-executer-agentic-platform/config/capability_registry.yaml)

That registry should remain the public vocabulary for:

- planner reasoning
- clarification candidate derivation
- runtime validation
- observability and governance

Internal tool names should stay implementation details wherever possible.

## The Control Plane You Want

The correct control plane has five stages.

### 1. Normalize the request around the user instruction

The system should first convert the raw user request into a normalized artifact that contains:

- goal
- intent
- candidate capabilities
- intent segments
- known slot values
- missing fields

Relevant files:

- [intent_service.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/api/app/intent_service.py)
- [intent_contract.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/intent_contract.py)
- [intent-normalization-architecture.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/intent-normalization-architecture.md)

Rule:

The LLM may propose the normalized shape, but the contract layer must decide what is valid.

### 2. Derive clarification from capability-required fields

Clarification should not be mostly heuristic.

It should come from:

- normalized intent segments
- suggested capabilities
- required inputs for those capabilities
- filtering of non-user-collectible fields

In this repo, that logic is centered around:

- [derive_envelope_clarification in intent_contract.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/intent_contract.py)
- [normalize_goal_intent in intent_service.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/api/app/intent_service.py)

Rule:

Capability-required inputs should become clarification candidates unless they are:

- derived from dependency outputs
- runtime-only
- system-managed
- safely defaultable by policy

Examples:

- ask for `audience`
- ask for `tone`
- do not ask for `document_spec`
- do not ask for `input_data` if it is produced by an earlier task

### 3. Plan in capability language

The planner should reason over capabilities, not raw local adapter names.

That means:

- capability IDs are the planner vocabulary
- adapter selection is deterministic compiler logic
- planner output is validated against capability language rules

Relevant files:

- [planner_service.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py)
- [planner_contracts.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/planner_contracts.py)
- [run_specs.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/run_specs.py)
- [planner-capability-language-implementation-plan.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/planner-capability-language-implementation-plan.md)

Rule:

The planner may not invent raw runtime tools when a capability contract exists.

### 4. Validate the plan before execution

Before a plan is accepted, validate three things:

1. intent compatibility
2. capability input compatibility
3. runtime conformance

In this repo, the important enforcement points are:

- [validate_plan_request in planner_service.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py)
- [validate_capability_inputs in planner_service.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py)
- [_compile_plan_preflight in main.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/api/app/main.py)

Rule:

If the plan does not satisfy the capability contract or the selected intent segment, reject it before the worker sees it.

### 5. Enforce contracts again at runtime

The worker must not trust planner output blindly.

Runtime enforcement should include:

- capability input validation
- pruning of unsupported fields
- tool input schema validation
- tool output schema validation
- task-level expected output validation

Relevant files:

- [_enforce_capability_input_contract in worker main.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/worker/app/main.py)
- [_validate_expected_output in worker main.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/worker/app/main.py)
- [tool_runtime.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/framework/tool_runtime.py)

Rule:

Planner validation is not enough. Runtime must enforce again.

## The Four Contracts You Need

To make the system reliable, treat these as separate but linked contracts.

### 1. Instruction contract

This answers:

- what did the user ask for?
- what should each task actually do?

Required fields:

- job goal
- task instruction

Policy:

- never silently default `instruction`
- never replace the user objective with adapter convenience behavior
- if instruction is underspecified, clarify or reject

### 2. Capability input contract

This answers:

- what inputs are valid for this capability?
- which are required?

Authority:

- capability registry entry
- input schema ref

Policy:

- planner can enrich payloads from job context and dependency outputs
- worker validates the final payload against the schema

### 3. Capability output contract

This answers:

- what shape does the capability promise to return?

Authority:

- `output_schema_ref` in the capability registry
- tool output schema in the tool runtime

Policy:

- handlers must return contract-valid data
- tool runtime must reject invalid outputs

### 4. Task expected-output contract

This answers:

- what output does this plan step promise to downstream steps?

Authority:

- `expected_output_schema_ref` on the task / run step

Policy:

- use task expected outputs for workflow-level composition guarantees
- use capability output schemas for handler-level guarantees

Both are useful. They are not the same boundary.

## Repo-Specific Enforcement Path

The desired path in this repository is:

```text
User request
  -> intent normalization
  -> candidate capability selection
  -> schema-led clarification
  -> capability-first planning
  -> planner preflight
  -> planner capability input validation
  -> RunSpec compilation
  -> worker capability input enforcement
  -> tool input/output schema validation
  -> expected output validation
```

If a request can bypass one of those layers, you will eventually get drift.

## How To Make Clarification Truly Capability-Driven

This is the most common place where systems drift.

### Required rule

Clarification must be based on capability-required fields, not only on intent heuristics like:

- render implies `output_format`
- io implies `target_system`

That heuristic layer is useful, but it should not be the main authority.

### What to do

For each candidate capability:

1. load required input fields
2. normalize aliases
3. remove non-user-collectible fields
4. check which values already exist in:
   - clarified context
   - job context
   - intent slot values
   - dependency outputs if applicable
5. ask only for the remaining user-facing fields
6. ask one question at a time

### Field classes you should maintain

Every required field should be tagged as one of:

- `user_required`
- `user_optional`
- `inferable`
- `defaultable`
- `derived`
- `runtime_only`

Without this classification, raw schema-led clarification becomes too blunt.

## How To Keep Agents Driven by the Instruction

Instruction drift usually happens in one of these ways:

- the planner chooses a capability that does not match the user objective
- clarification asks only operational fields and never captures authoring intent
- the worker receives a payload enriched from defaults but no longer anchored to the original instruction
- the agent prompt hides the real objective in internal template text

### Required controls

1. Preserve instruction at every stage

- job goal
- task instruction
- clarified instruction fields

2. Validate task intent against the selected capability

This is already part of plan validation and should remain mandatory.

3. Reject plans that only satisfy schema shape but not intent semantics

A syntactically valid payload is not enough if it violates the selected intent segment.

4. Default only non-core fields

Safe defaults can be used for:

- tone
- audience
- optional formatting constraints

Do not default:

- instruction
- target system for risky actions
- destructive safety constraints

5. Keep agent internals behind the capability boundary

The caller should not need to know:

- hidden prompt templates
- agent loop state
- scratchpad structure
- private intermediate tool decisions

## Input Validation Strategy

Use two layers of input validation.

### Layer A: Planner-time validation

At planning time:

- resolve payload references enough to validate shape
- project job context into explicit fields where policy allows
- validate against the capability input schema

This catches:

- missing required fields
- invalid enum/type shapes
- mismatched plan payloads

### Layer B: Worker-time validation

At execution time:

- prune unsupported fields
- validate final resolved payload
- reject invalid capability requests before handler execution

This catches:

- stale plans
- planner bugs
- bad rewrites
- dependency-resolution surprises

Both layers are necessary.

## Output Validation Strategy

Use two output checks.

### Layer A: Tool output schema validation

Each tool handler should validate its own declared output schema.

That prevents a capability from returning malformed data at the tool boundary.

### Layer B: Task expected output validation

Each planned task should also declare its expected output shape.

That protects:

- downstream dependency wiring
- workflow composition
- replay and regression consistency

If a capability returns structurally valid output but not the output shape the task promised, the task should still fail.

## The Operating Rules

If you want this system to stay disciplined, follow these rules.

### Rule 1: One public contract per capability

The public contract is the capability ID plus its schemas.

Do not let raw adapter names become the public interface.

### Rule 2: Instruction is mandatory and explicit

Every authoring or transformation task must carry an explicit instruction.

### Rule 3: Clarification asks only for collectible fields

Do not ask the user for dependency outputs or runtime-only fields.

### Rule 4: Planner must validate before compile

The planner may not emit a plan that only becomes valid after hand-wavy runtime fixes.

### Rule 5: Runtime must validate again

Execution never trusts planning blindly.

### Rule 6: Defaults must be policy-driven

Defaultable fields must be clearly declared and audited.

### Rule 7: Regressions become tests

Every production failure mode should become:

- an intent/clarification test
- a planner validation test
- or a worker contract test

## Implementation Checklist for This Repo

Use this checklist when adding or hardening an agent capability.

### Capability contract

- add capability to [capability_registry.yaml](/Users/narendersurabhi/planner-executer-agentic-platform/config/capability_registry.yaml)
- define `input_schema_ref`
- define `output_schema_ref` where possible
- assign risk tier and idempotency
- keep adapter details behind the contract

### Clarification

- ensure capability-required fields are exposed to intent normalization
- mark which required fields are user-collectible
- add explicit question text for high-value fields
- make chat preserve `capability_required_inputs` fields instead of dropping them

### Planning

- keep planner output in capability language
- validate task intent against capability intent
- validate payload shape before accepting the plan
- ensure plan preflight strips cross-stage pollution like render-only fields leaking into authoring steps

### Runtime

- validate capability inputs in worker
- validate tool input and output schemas
- validate task expected outputs where downstream composition matters

### Tests

- add intent clarification regression
- add planner validation regression
- add worker contract regression
- add staging replay or golden-set coverage if the path is business-critical

## Common Failure Modes

### Failure mode 1: Clarification and planner disagree

Symptom:

- chat stops asking questions
- planner later fails on missing schema fields

Cause:

- capability-required fields are not driving clarification

Fix:

- derive clarification from required capability fields and field policy

### Failure mode 2: Planner uses the wrong vocabulary

Symptom:

- raw local tools leak into plans
- registry and planner drift

Cause:

- planner is tool-driven instead of capability-driven

Fix:

- capability-first planning plus deterministic adapter compilation

### Failure mode 3: Runtime accepts invalid payloads

Symptom:

- handler crashes deep inside execution
- failures happen after expensive orchestration work

Cause:

- no worker-time schema enforcement

Fix:

- validate again at runtime

### Failure mode 4: Agent output is structurally inconsistent

Symptom:

- downstream tasks break on missing fields
- generated artifacts do not match expected schema

Cause:

- only prompt-level expectations, no output contract enforcement

Fix:

- tool output schema validation plus task expected-output validation

## Recommended End State

The end state for this repository should be:

- instruction-driven request normalization
- capability-driven clarification
- capability-first planner language
- planner-time capability validation
- worker-time capability validation
- tool-level output validation
- task-level expected-output validation
- regression coverage for every production miss

That is how you keep agents aligned to user instructions while still letting them operate inside a typed, governable capability system.

## Bottom Line

If you want agents to be driven by instructions and governed by capability schemas, do not rely on one layer.

You need a chain of authority:

- instruction for objective
- capability schema for contract
- clarification for missing user-facing fields
- planner validation for plan correctness
- runtime validation for execution correctness
- output validation for downstream safety

That full chain, not any one component by itself, is what makes the system reliable.
