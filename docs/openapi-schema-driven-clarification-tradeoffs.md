# OpenAPI for Schema-Driven Clarification: Benefits, Limits, and Trade-Offs

## Short Answer

Implementing an OpenAPI specification helps with some of the current schema-driven logic problems, but it does not solve the whole problem by itself.

It helps most when the problem is:

- contract drift between API payloads and capability schemas
- inconsistent required field definitions across services
- unclear request and response shapes
- poor discoverability of capability inputs
- weak generation of typed clients, validators, and docs

It does not fully solve problems that are actually about:

- conversational clarification policy
- deciding which fields are user-collectible vs inferred vs defaulted
- multi-step intent decomposition
- route selection across agent, planner, and workflow surfaces
- execution-time dependency wiring
- slot provenance and clarification state management

The right conclusion for this repository is:

- use OpenAPI as a stronger contract source for capability and API input shapes
- do not treat OpenAPI as the full logic engine for clarification or routing
- add explicit metadata on top of schemas for clarification semantics

## Why This Question Matters Here

The recent failures in this repository exposed a real architectural split:

- capability schemas enforce what a tool or capability requires
- intent and chat clarification decide what to ask the user
- planner validates capability payloads later

That split created a mismatch:

- `document.spec.generate` required `instruction`, `topic`, `audience`, and `tone`
- generic clarification did not always ask for `audience` and `tone`
- planner found the gap only during capability validation

In other words, the repo had a contract source, but the clarification layer was not fully driven by it.

OpenAPI can reduce that mismatch, but only if it is used carefully.

## What OpenAPI Actually Solves

### 1. A single machine-readable contract

OpenAPI gives you one place to define:

- input fields
- required fields
- types
- enums
- nested objects
- examples
- response shapes

That helps replace scattered duplication across:

- `schemas/*.json`
- capability metadata
- UI assumptions
- hand-written validation payload builders
- documentation

If the same capability is described differently in multiple places today, OpenAPI helps consolidate it.

### 2. Better tooling and validation

OpenAPI gives you stronger ecosystem support than ad hoc schema files alone:

- generated docs
- generated clients
- request/response validators
- contract testing
- schema diff tools
- change detection in CI

That helps with operational rigor. A change to a capability contract becomes easier to detect, review, and gate.

### 3. Better portability across services

For a multi-service system like this one, OpenAPI helps when:

- API receives a request
- planner normalizes or enriches it
- worker executes capability payloads
- UI or staging tooling introspects supported inputs

A shared contract format lowers ambiguity across those boundaries.

### 4. A better base for schema-led clarification

If the capability contract is published in a consistent format, clarification can ask:

- what are the required user-facing inputs?
- which are missing?
- which already exist in context?

That is better than relying on a hardcoded allowlist alone.

## What OpenAPI Does Not Solve

### 1. OpenAPI does not know which required fields should be asked from the user

This is the biggest gap.

A field can be required in schema terms but still be:

- inferred from context
- defaulted safely
- derived from another step
- injected by runtime
- filled from dependencies
- hidden from the user entirely

Examples:

- `document_spec` may be required for rendering, but should not be asked from the user in normal chat
- `path` may be required for render, but can sometimes come from workflow defaults
- `tone` may be required for document generation, but could be defaulted conservatively

OpenAPI can tell you a field is required. It cannot tell you the UX semantics of that requirement unless you add explicit metadata.

### 2. OpenAPI does not model multi-step intent graphs

The repo does not only validate one request payload. It also needs to decide:

- is this chat or execution?
- is this generate, render, validate, or io?
- should this become one task or a chain of tasks?
- which fields belong to which stage?

OpenAPI is a contract format for APIs, not an intent graph model.

It does not natively encode:

- segment ordering
- task dependencies
- candidate capabilities by segment
- route calibration
- clarification progression across turns

### 3. OpenAPI does not represent slot provenance

In this system, it matters whether a value came from:

- explicit user input
- normalized clarification
- retrieved context
- prior workflow state
- system default

That provenance affects trust, auditability, and whether to re-ask the user.

OpenAPI has no native model for that.

### 4. OpenAPI does not choose between "must ask", "may infer", and "may default"

This is a policy question, not a schema question.

For each field, the platform needs a decision like:

- must ask before execution
- can ask only if confidence is low
- may infer from job context
- may default
- runtime-only, never ask

OpenAPI alone will not provide that classification.

### 5. OpenAPI does not solve ambiguous routing

If the system misclassifies a request as render-only instead of generate-then-render, an OpenAPI document will not fix that.

That is an intent normalization and routing problem, not a request-shape problem.

## The Core Trade-Off

### Option A: Use OpenAPI as the source of truth for all required fields

Benefits:

- fewer mismatches between clarification and capability validation
- less duplicated schema logic
- easier CI contract checks
- better generated docs and tooling

Costs:

- you will over-ask unless you separately mark derived and inferable fields
- internal/runtime fields may leak into user clarification
- planner and chat will still need semantic policy layers

Risk:

- schema-led clarification becomes too literal and asks bad questions

### Option B: Keep separate clarification policy and capability schema

Benefits:

- better user experience control
- easier to suppress internal/runtime fields
- easier to prioritize only the most important clarifications

Costs:

- drift between schema and clarification
- planner catches issues late
- more duplicated logic

Risk:

- the exact class of failures seen in this repo repeats

### Recommended position

Use a hybrid model:

- OpenAPI defines structural contracts
- clarification is schema-led, but only through a user-facing field policy layer

That gives you one contract source without turning raw schema requirements into raw user questions.

## Recommended Design for This Repository

### 1. Keep OpenAPI or JSON Schema as the structural contract source

For each capability or API surface, define:

- fields
- required fields
- types
- enums
- examples
- output contracts

If OpenAPI is adopted, it should either:

- replace repo-local schema duplication, or
- become the upstream source that generates repo-local JSON Schemas

Do not maintain both manually long term.

### 2. Add clarification-specific metadata as extensions

If OpenAPI is the contract source, it should be extended with explicit field metadata.

Examples:

```yaml
audience:
  type: string
  minLength: 1
  x-user-collectible: true
  x-clarification-priority: high
  x-question: "Who is the target audience?"
  x-may-default: true
  x-default-value: "general professional audience"
  x-provenance-policy: user_or_default
```

```yaml
document_spec:
  type: object
  x-user-collectible: false
  x-derived-from: dependency_output
  x-clarification-policy: never_ask
```

Without metadata like this, schema-led clarification will stay too blunt.

### 3. Distinguish field classes explicitly

Every required field should be assigned one of these classes:

- `user_required`
- `user_optional`
- `inferable`
- `defaultable`
- `derived`
- `runtime_only`

That classification should drive clarification, not raw `required` alone.

### 4. Generate clarification candidates from schema plus field policy

The clarification pipeline should do:

1. read capability required fields from contract
2. normalize aliases
3. remove `derived` and `runtime_only`
4. skip fields already present in trusted context
5. prefer high-priority user fields first
6. default only when policy allows
7. emit one question at a time

That is the right balance between rigor and usability.

### 5. Keep intent decomposition separate

OpenAPI should not decide:

- route family
- segment ordering
- whether the request is chat vs execution
- whether the request is generate-then-render

Those remain responsibilities of:

- intent normalization
- routing
- planner decomposition

OpenAPI strengthens the contract after the target capability family is known.

## Where OpenAPI Helps the Most in the Current Stack

### API layer

OpenAPI is strongest at the API layer because FastAPI already maps well to it.

It can improve:

- `/intent/*`
- `/plans/*`
- `/chat/*`
- workflow invocation payloads
- staging and regression tooling

### Capability registry and planner validation

OpenAPI can also help if capability contracts are expressed consistently and then projected into:

- planner capability validation
- UI capability introspection
- chat clarification contracts

This is useful, but it requires a clear translation layer. OpenAPI itself is not the registry.

### Staging and regression gates

OpenAPI helps CI and staging by enabling:

- schema diff checks
- backward-compatibility checks
- generated examples
- replay validation against declared contracts

That is operationally valuable even if it does not solve routing logic.

## Where OpenAPI Helps Less

### Chat loop state machine

Pending clarification state still needs:

- current question
- pending questions
- active capability
- active segment
- known slot values
- slot provenance
- exit or restart logic

OpenAPI does not model this state machine.

### Planner dependency wiring

Examples like:

- render task depends on `GenerateDocumentSpec.document_spec`
- output path may be derived by policy
- iterative tools may require job wrappers

These are workflow semantics, not API schema semantics.

### Semantic understanding of user language

OpenAPI cannot tell whether:

- "create a document" means generate a spec, render a DOCX, or both
- "agenticai.docx" is a path answer, a title, or an output preference
- the user intends a workflow invocation or just a chat response

That remains an intent and context understanding problem.

## Practical Trade-Offs

### Benefits

- stronger contract discipline
- better visibility into required fields
- less schema drift
- easier client and docs generation
- better CI enforcement
- better support for schema-led clarification

### Drawbacks

- more upfront schema design work
- pressure to overfit UX to raw contracts
- vendor-extension sprawl if not governed
- possible duplication if capability registry and OpenAPI evolve independently
- can create a false sense of completeness if routing problems remain unsolved

### Failure mode if adopted poorly

If OpenAPI is adopted naively, the system will ask users for fields like:

- `document_spec`
- `input_data`
- internal dependency outputs
- runtime-only execution parameters

That would make the chat UX worse, not better.

## Recommended Adoption Path

### Phase 1: Contract alignment

Use OpenAPI or shared JSON Schema generation to align:

- API request payloads
- capability inputs
- planner validation expectations

Goal:

- one structural source of truth

### Phase 2: Clarification metadata

Add field-level metadata for:

- `x-user-collectible`
- `x-question`
- `x-clarification-priority`
- `x-may-default`
- `x-default-value`
- `x-derived-from`
- `x-runtime-only`

Goal:

- schema-led clarification that still respects UX

### Phase 3: Schema-led clarification engine

Replace hardcoded clarification allowlists with:

- capability-required inputs
- filtered through field policy metadata
- ranked by clarification priority

Goal:

- planner and clarification stop disagreeing

### Phase 4: Routing integration

Use contract metadata to inform routing confidence and candidate scoring, but do not let contracts replace routing logic.

Goal:

- tighter coupling between route choice and contract expectations

## Recommended Answer for This Repo

If the question is:

"Should this repository implement OpenAPI because it will solve schema-driven clarification problems?"

The answer is:

- yes, it will solve part of the problem
- no, it will not solve the whole problem

It is a good move if the goal is:

- one contract source
- fewer mismatches between clarification and planner validation
- stronger staging and CI contract checks

It is not sufficient if the goal is:

- correct routing
- correct segment decomposition
- correct user-facing clarification behavior

For this repository, the best architecture is:

- OpenAPI for structural contracts
- explicit clarification metadata for field semantics
- intent/routing logic kept separate
- planner validation still enforced downstream

## Bottom Line

OpenAPI is valuable here, but only as a contract backbone.

It should become the source of structural truth for capability inputs and API surfaces. It should not be treated as the full answer to conversational clarification, intent decomposition, or workflow routing.

The winning design is not "OpenAPI instead of clarification logic." The winning design is:

- OpenAPI for structure
- policy metadata for collection semantics
- intent/routing for execution shape
- planner validation for final enforcement

That combination solves the actual failure mode better than OpenAPI alone.
