# Planner Capability Language Implementation Plan

## Goal

Make the planner speak **capability language** instead of raw runtime tool language.

For this repo, that means:

- planner prompts should reason about capability IDs and capability contracts
- planner outputs should identify capability requests, not local Python tool names
- a deterministic compiler should map capability-level plans into runtime tool requests
- runtime tools should remain private executor details

This plan is repo-specific and assumes the current split across:

- `services/api`
- `services/planner`
- `services/worker`
- `libs/core/planner_contracts.py`
- `libs/core/run_specs.py`
- `config/capability_registry.yaml`

## Current State

Today the codebase is mixed:

- the intended design already says capabilities are the planner’s public vocabulary in [intent-planning-tool-calling-architecture.md](./intent-planning-tool-calling-architecture.md)
- `normalized_intent_envelope` and `suggested_capabilities` already exist and are planner-visible
- but the actual planner prompt in `services/planner/app/planner_service.py` still instructs the LLM to emit `tool_requests` and use allowed tool names
- runtime execution, policy checks, and payload resolution still depend on `tool_requests`

So the architecture target exists, but the live planner contract is still tool-first.

## Why This Matters

Using raw tool names in planner output creates avoidable problems:

- planner output leaks executor implementation details
- changing tool adapters can break planning semantics
- local tool names and capability IDs drift separately
- the LLM is exposed to too much low-level vocabulary
- planner validation becomes harder because tool identity and business capability are conflated

For this repo specifically, a capability-first planner would reduce:

- intent/capability mismatch
- bad tool-level outputs like raw `llm_generate_document_spec`
- planner drift when adapters or filenames change
- prompt complexity in `build_llm_prompt()`

## End State

The target model is:

1. API/intent normalization produces capability-ranked intent segments.
2. Planner sees only:
   - goal
   - normalized intent envelope
   - candidate capabilities
   - capability contracts
   - planner-only support tools
3. Planner emits a capability-first plan.
4. A deterministic compiler maps capability steps to runtime `tool_requests`.
5. Runtime executes compiled `RunSpec` / task requests as it does today.

In short:

- planner language = capabilities
- compiler language = capability-to-tool mapping
- runtime language = tool requests

## Guiding Principles

1. Keep runtime compatibility during rollout.
2. Do not break existing worker execution paths on day one.
3. Move planner prompts first, compiler second, runtime cleanup last.
4. Enforce capability language at validation boundaries, not only in prompt wording.
5. Use `config/capability_registry.yaml` as the authoritative mapping from capability to adapter/tool.

## Phase 1: Tighten Planner Prompt and Validation Without Breaking Runtime

### Objective

Reduce raw tool leakage immediately while keeping the current `tool_requests` runtime shape.

### Design

Keep the `PlanCreate` shape for now, but make the planner prompt capability-first:

- present capability IDs and contracts as the primary vocabulary
- instruct the planner to use segment `suggested_capabilities`
- de-emphasize raw tool names
- make any raw tool emission a repair/validation concern

Add validation that rejects or rewrites planner outputs when raw tool names are used where a canonical capability exists.

### Files

- `services/planner/app/planner_service.py`
  - rewrite `build_llm_prompt(...)`
  - remove or reduce `Allowed tool names`
  - add a capability-centric section from request capabilities and segment candidates
  - update repair prompt in `build_llm_repair_prompt(...)`

- `services/planner/app/main.py`
  - add a normalization step that canonicalizes planner-emitted request IDs through capability aliases where possible

- `libs/core/capability_registry.py`
  - ensure canonicalization helpers are available and cheap to use from planner normalization

- `libs/core/planner_contracts.py`
  - add helper:
    - `canonicalize_planner_request_ids(...)`
    - or equivalent utility to distinguish capability IDs vs raw tool names

### Deliverables

- prompt becomes capability-first
- planner validation stops accepting arbitrary raw tool names when a capability mapping exists
- runtime remains unchanged

### Test Plan

- planner prompt test includes capability vocabulary and not just tool names
- planner output using `document.docx.render` passes
- planner output using `docx_render_from_spec` is rewritten or rejected

## Phase 2: Introduce Planner-Specific Capability Request Fields

### Objective

Separate planner output from runtime execution details in the plan contract.

### Design

Extend task contracts to allow a planner-facing field such as:

- `capability_requests`

while retaining `tool_requests` temporarily for compatibility.

The planner should populate:

- `capability_requests`

The compiler should derive:

- `tool_requests`

### Files

- `libs/core/models.py`
  - extend `TaskCreate` or add a planner-specific plan model with:
    - `capability_requests: list[str]`

- `services/planner/app/planner_service.py`
  - update prompt schema and examples
  - update parsing/repair logic

- `libs/core/run_specs.py`
  - extend conversion helpers to understand `capability_requests`

- `libs/core/execution_contracts.py`
  - keep runtime execution contract tool-based for now

### Deliverables

- planner can emit capability requests explicitly
- runtime remains compatible through compilation

### Test Plan

- planner plan with `capability_requests` compiles to valid runtime requests
- legacy `tool_requests` plans still work during transition

## Phase 3: Add Deterministic Capability-to-Tool Compilation

### Objective

Move adapter resolution out of the LLM and into deterministic compiler code.

### Design

Add a compiler step that:

- takes a capability request
- loads the capability spec
- picks the enabled adapter
- produces the runtime request ID / tool request
- carries forward validated inputs

This should use:

- `config/capability_registry.yaml`
- `libs/core/planner_contracts.py`
- `libs/core/run_specs.py`

### New Helpers

- `compile_capability_request(...)`
- `compile_capability_requests_for_task(...)`
- `resolve_primary_adapter(...)`

### Files

- `libs/core/planner_contracts.py`
- `libs/core/run_specs.py`
- optionally new module:
  - `libs/core/capability_compiler.py`

### Deliverables

- LLM no longer chooses low-level runtime handlers directly
- capability registry becomes the translation boundary

### Test Plan

- `document.spec.generate` compiles to `llm_generate_document_spec`
- `document.docx.render` compiles to `docx_render_from_spec`
- disabled capability or missing adapter fails deterministically

## Phase 4: Add Planner-Only Support Tools

### Objective

If planner tool calling is enabled, expose only planner support tools, not executor tools.

### Design

Create planner-only support tools like those already described in [intent-planning-tool-calling-architecture.md](./intent-planning-tool-calling-architecture.md):

- `search_capabilities`
- `get_capability_contract`
- `get_schema`
- `get_workflow_hints`
- `get_memory_hints`
- `finalize_run_spec`

These tools should:

- query metadata
- inspect schemas
- help compile or validate capability-level plans

They should not perform side effects.

### Files

- `libs/core/tool_catalog.py`
- `libs/core/tool_bootstrap.py`
- new planner-only helper module if needed
- `services/planner/app/runtime_service.py`

### Deliverables

- planner tool calling becomes metadata-only
- runtime executor tools are no longer visible to planner prompts

### Test Plan

- planner registry excludes side-effect runtime tools
- support tools return capability/schema data only

## Phase 5: Make RunSpec Capability-First

### Objective

Ensure the canonical planner artifact stores capability intent, not just compiled tool IDs.

### Design

Right now `RunSpec` translation still centers on runtime request IDs.

Shift `RunSpec` so that:

- capability identity is preserved as first-class metadata
- compiled tool requests are attached as derived execution details

Suggested shape:

- keep `request_ids` or `tool_requests` as compiled fields
- add `capability_id` or `capability_requests`
- preserve both until runtime fully transitions

### Files

- `libs/core/models.py`
- `libs/core/run_specs.py`
- `libs/core/execution_contracts.py`

### Deliverables

- RunSpec captures planner intent in capability terms
- runtime still gets compiled request IDs

### Test Plan

- run spec round-trip preserves capability identity
- policy/runtime still execute compiled tool requests correctly

## Phase 6: Enforce Capability Language at Submission Boundaries

### Objective

Make tool-language leakage a validation error, not just a style issue.

### Design

Add hard validation so planner-generated plans cannot be accepted unless:

- every planner-selected request is a canonical capability ID
- or it is deterministically compiled from one

Reject plans that:

- emit raw local tool names directly
- refer to disabled capabilities
- refer to adapter-local names without a matching capability

### Files

- `services/planner/app/planner_service.py`
- `libs/core/planner_contracts.py`
- `services/api/app/main.py` preflight path

### Deliverables

- capability language becomes mandatory
- planner drift into runtime tool space is blocked

### Test Plan

- raw `llm_generate_document_spec` planner output is rejected if emitted directly
- canonical `document.spec.generate` output passes
- legacy compatibility can be toggled during migration if needed

## Phase 7: Clean Up Legacy Runtime/Prompt References

### Objective

Finish the migration by removing planner dependence on raw tool vocabulary.

### Scope

- prompt examples
- docs
- UI plan/debug surfaces where they present planner intent
- tests that still assert planner-visible tool names

### Files

- `README.md`
- `docs/intent-planning-tool-calling-architecture.md`
- `docs/api.md`
- `services/ui/src/app/page.tsx`
- tests under:
  - `services/planner/tests`
  - `libs/core/tests`
  - `services/api/tests`

### Deliverables

- planner-facing docs and UI become capability-first
- tool names remain runtime internals

## Recommended Rollout Order

Use this order:

1. Phase 1: prompt + validator tightening
2. Phase 2: `capability_requests`
3. Phase 3: compiler
4. Phase 4: planner-only support tools
5. Phase 5: RunSpec capability-first metadata
6. Phase 6: hard enforcement
7. Phase 7: cleanup

This order is important because it lets you improve planner behavior without destabilizing worker execution.

## Immediate First Slice

If only one slice is implemented first, it should be:

1. rewrite `build_llm_prompt(...)` to make capabilities primary
2. canonicalize/reject raw runtime tool names in planner validation
3. add regression tests for:
   - `document.spec.generate`
   - `document.docx.render`
   - rejection of `llm_generate_document_spec`
   - rejection of `docx_render_from_spec`

That gives the planner a much better boundary immediately, even before the full compiler refactor lands.

## Success Criteria

You will know this is working when:

- planner outputs mostly reference canonical capability IDs
- plan validation failures from tool/capability mismatch drop
- adapter renames stop requiring planner prompt changes
- planner docs and UI refer to capabilities, not tool handlers
- the runtime still executes through compiled requests with no loss of behavior

## Summary

For this repo, enforcing capability language is not just a prompt tweak.

It is a staged migration:

- from planner-visible tool names
- to planner-visible capability IDs
- with deterministic compilation in between

The safest path is:

- prompt and validator first
- compiler next
- runtime cleanup last
