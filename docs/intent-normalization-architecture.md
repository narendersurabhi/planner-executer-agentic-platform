# Intent Normalization Architecture

This document proposes the repo-specific architecture for intent normalization
in Agentic Workflow Studio.

It is focused on one problem:

> how to take a raw user goal and turn it into a deterministic, capability-aware
> normalized artifact that planning can trust

It is intentionally anchored to the current codebase, not to a hypothetical
greenfield system.

Relevant current modules:

- `services/api/app/intent_service.py`
- `services/api/app/main.py`
- `services/planner/app/planner_service.py`
- `libs/core/intent_contract.py`
- `libs/core/workflow_contracts.py`
- `libs/core/run_specs.py`

## 1. Executive Summary

The recommended architecture is:

1. attach request context before classification
2. retrieve a bounded allowed capability candidate set
3. run a first-pass intent normalization step against a strict schema
4. normalize and validate the result deterministically
5. derive clarification requirements from the normalized artifact plus
   candidate capability contracts
6. persist one canonical normalization envelope
7. pass that envelope into planning without re-parsing raw goal text

The key design choice is:

- the LLM may propose normalization
- the contract layer decides what is valid

This keeps normalization useful without letting it become a second source of
runtime truth.

## 2. Current State

Today the repo already has the right building blocks:

- first-pass goal assessment in `services/api/app/intent_service.py`
- heuristic and LLM-backed decomposition in `services/api/app/main.py`
- slot normalization and contract enforcement in `libs/core/intent_contract.py`
- typed graph/profile models in `libs/core/workflow_contracts.py`
- planner validation against goal intent segments in
  `services/planner/app/planner_service.py`

The problem is not missing functionality. The problem is split authority.

Current symptoms:

- assessment and decomposition are separate artifacts
- clarification is still mostly generic and intent-level
- planner validation still depends on segment selection and secondary inference
- normalization traces are spread across job metadata instead of one canonical
  object
- planner and runtime still need to reason about partially normalized state

## 3. Design Principles

### 3.1 Normalization is compile-time, not runtime authority

Intent normalization should shape:

- clarification
- capability candidate retrieval
- planner context
- validation

It should not directly decide:

- execution order
- retries
- runtime scheduling

### 3.2 Capabilities bound the normalization space

The model should normalize against:

- allowed capability ids
- capability descriptions
- capability risk
- capability input contracts

It should not normalize against an unlimited abstract ontology.

### 3.3 One canonical artifact

The repo should have one durable normalization artifact that all control-plane
paths can rely on.

Recommended new artifact:

- `NormalizedIntentEnvelope`

This should replace the practical need to juggle:

- `GoalIntentProfile`
- `IntentGraph`
- ad hoc missing-slot logic
- scattered capability candidate hints

Those can still exist internally, but the envelope becomes the canonical
higher-level handoff object.

### 3.4 Deterministic normalization stays mandatory

LLM output should always flow through deterministic normalization in
`libs/core/intent_contract.py`.

That layer should continue to own:

- task-intent normalization
- slot normalization
- alias handling
- invalid-field stripping
- required-input validation rules

### 3.5 Clarification should be capability-driven

Clarification should be derived from:

- normalized intent
- normalized slots
- candidate capabilities
- capability-required inputs
- risk level

It should not depend primarily on broad rules like:

- `render` implies `output_format`
- `io` implies `target_system`

Those are useful heuristics, but not strong enough to be the main authority.

## 4. Target Artifact

Add a canonical contract in `libs/core/workflow_contracts.py`.

Suggested shape:

```json
{
  "schema_version": "intent_envelope_v1",
  "source": "llm_hybrid",
  "goal": "Create a quarterly planning memo as a PDF",
  "profile": {
    "intent": "generate",
    "risk_level": "bounded_write",
    "confidence": 0.86,
    "needs_clarification": false
  },
  "graph": {
    "segments": [
      {
        "id": "s1",
        "intent": "generate",
        "objective": "generate requested content",
        "suggested_capabilities": ["llm_generate_document_spec"],
        "slots": {
          "artifact_type": "document_spec",
          "entity": "document_spec",
          "must_have_inputs": ["instruction"]
        }
      },
      {
        "id": "s2",
        "intent": "render",
        "objective": "render final artifact",
        "suggested_capabilities": ["document_pdf_generate"],
        "slots": {
          "output_format": "pdf",
          "must_have_inputs": ["path", "input_data"]
        }
      }
    ]
  },
  "candidate_capabilities": {
    "s1": ["llm_generate_document_spec"],
    "s2": ["document_pdf_generate"]
  },
  "missing_inputs": [],
  "clarification": {
    "needs_clarification": false,
    "blocking": false,
    "questions": []
  },
  "trace": {
    "assess_mode": "hybrid",
    "decompose_mode": "hybrid",
    "fallback_used": false
  }
}
```

Recommended fields:

- `schema_version`
- `source`
- `goal`
- `profile`
- `graph`
- `candidate_capabilities`
- `missing_inputs`
- `clarification`
- `trace`

## 5. Pipeline Architecture

The normalization pipeline should look like this:

```text
Raw goal
  -> attach request context
  -> retrieve candidate capabilities
  -> run first-pass structured normalization
  -> normalize slots and segment shapes deterministically
  -> derive clarification from capability contracts
  -> persist NormalizedIntentEnvelope
  -> hand off to planner
```

### 5.1 Context attachment

Do this in API before LLM normalization.

Input context should include:

- normalized user id
- chat/session metadata when relevant
- workflow hints from prior jobs
- semantic capability hints
- allowed capability catalog subset

This already partly exists in:

- `_infer_goal_intent_with_metadata(...)`
- `_decompose_goal_intent(...)`
- `_retrieve_intent_workflow_hints(...)`
- `_semantic_goal_capability_hints(...)`

### 5.2 Candidate capability retrieval

Normalization should receive a bounded candidate set before classification.

This should be derived from:

- capability registry
- governance filters
- semantic retrieval
- workflow history

The goal is to keep the normalization space small enough that the model does
not invent impossible routes.

### 5.3 First-pass structured normalization

The LLM step should produce a structured object, not prose.

Recommended output responsibilities:

- primary intent
- segment list
- slot candidates
- candidate capability ids
- confidence
- ambiguity or clarification hints

The first-pass LLM step should not:

- invent schema fields
- invent required inputs not supported by capability contracts
- decide execution order beyond segment dependency hints

### 5.4 Deterministic canonicalization

All LLM output should flow through deterministic canonicalization in
`libs/core/intent_contract.py`.

That layer should continue to:

- normalize task-intent labels
- normalize slot tokens and aliases
- canonicalize `document_spec`-type objectives
- strip downstream-only requirements from upstream segments
- validate intent-tool or intent-capability compatibility

This is where planner-time drift should be cut off.

### 5.5 Clarification derivation

Clarification should be produced after canonicalization, not before.

Recommended algorithm:

1. collect candidate capabilities per segment
2. inspect required inputs and accepted aliases
3. compare those against normalized slots and extracted values
4. produce missing-input records
5. map missing inputs to user-facing questions

This should replace most of the current generic slot blocking logic in
`blocking_clarification_slots(...)`.

### 5.6 Planner handoff

The planner should receive the canonical envelope and should not need to re-read
raw goal text to infer segment meaning.

Planner inputs should prefer:

- normalized segment intent
- normalized objective
- candidate capability ids
- explicit clarification state

Planner should not need:

- repeated heuristic intent inference from raw goal text
- fallback slot guessing that bypasses normalization

## 6. Service Boundaries

### API

The API remains the owner of normalization.

It should:

- construct the envelope
- persist it on jobs and workflow compile/run metadata
- stop planner submission when blocking clarification is required
- pass the envelope into planner requests

### Planner

The planner should consume normalization output, not recreate it.

It should:

- use normalized segments as compile-time hints
- validate plan output against normalized contracts
- avoid re-parsing user prose except for backward-compat fallback

### Worker

The worker should not perform first-pass normalization.

It may use normalized intent metadata for:

- observability
- execution labels
- policy context

But it should not infer intent from raw task text as a primary control path.

## 7. Observability

Normalization should emit durable control-plane traces.

Record:

- provider and model used
- assess mode
- decompose mode
- whether fallback occurred
- candidate capability ids shown to the model
- selected candidate ids
- stripped invalid fields
- derived missing inputs
- clarification questions

Recommended home:

- job metadata for compatibility
- dedicated normalized envelope field for control-plane reads
- debugger / job details endpoints later

## 8. Backward Compatibility

The cutover should preserve existing fields while introducing the envelope.

For a migration period:

- keep `goal_intent_profile`
- keep `goal_intent_graph`
- add `normalized_intent_envelope`

Planner and UI can then migrate in phases without breaking old records.

## 9. Recommended Next Changes

Highest-value changes:

1. add `NormalizedIntentEnvelope` to `libs/core/workflow_contracts.py`
2. add a single `normalize_goal_intent(...)` orchestration path in
   `services/api/app/intent_service.py`
3. move clarification derivation to capability-driven logic
4. pass the envelope into planner requests
5. stop planner-side re-inference except as fallback

That is the smallest architecture change that materially reduces intent drift
and invalid plan generation.
