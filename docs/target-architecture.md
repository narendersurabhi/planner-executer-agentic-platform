# Target Architecture

This document proposes the best target architecture for this repository if the
platform were rewritten or aggressively reshaped from the current design.

It is intentionally repo-specific. It is based on the current split across:

- `services/api`
- `services/planner`
- `services/worker`
- `libs/core/execution_contracts.py`
- `libs/core/workflow_contracts.py`
- `services/api/app/dispatch_service.py`
- `services/api/app/models.py`

It is not a generic workflow-platform document.

## 1. Executive Summary

The best architecture for this repo is:

1. `UI`
2. `Control Plane`
3. `Scheduler`
4. `Executor Pool`
5. `Adapter Sidecars`
6. `State + Storage`

Key opinionated choices:

- Treat `capabilities` as the only public execution contract.
- Treat `tools` as internal implementation details behind capabilities.
- Use `Postgres` as the system of record for runs, steps, attempts, and events.
- Keep `Redis` only for transient fan-out, leases, or SSE acceleration.
- Make `planner`, `policy`, and `critic` modules or execution stages, not
  mandatory standalone services.
- Compile every entry lane into the same canonical `RunSpec`.

The biggest change from the current repo is to stop modeling planner-led jobs,
Studio workflow runs, and chat-direct execution as partly separate orchestration
systems.

## 2. Why The Current Shape Is Painful

The current repo already has the right primitives, but they are split across too
many overlapping layers.

Main issues:

- `jobs`, `workflow_runs`, `plans`, and `tasks` model overlapping lifecycle
  concepts instead of one canonical run model.
- `tool_requests` plus `capability_bindings` in
  `libs/core/execution_contracts.py` creates dual execution semantics.
- The API is both control plane and partial scheduler via
  `services/api/app/dispatch_service.py`.
- The worker runtime resolves capabilities at execution time, which allows
  deploy-time drift between capability config and service-specific tool
  allowlists.
- Task results are partly durable and partly transient. For example,
  `services/api/app/main.py` stores task results in Redis under
  `task_result:*`, while run metadata and task state live in Postgres.
- Chat direct execution is a separate lane rather than a thin synchronous form
  of the same runtime model.

The failure mode is predictable: a workflow can compile, publish, and start,
then fail because the target service cannot actually execute the selected
capability at runtime.

## 3. Design Principles

The rewrite should follow these rules:

- `One run model`: every execution path becomes a run.
- `One scheduler`: dependency resolution, retries, and acceptance all happen in
  one state machine.
- `Capabilities are public`: plans persist capability requests, not raw tool
  names.
- `Tools are private`: tool names are local runtime details inside the
  capability gateway.
- `Postgres is authoritative`: execution state is queryable without Redis.
- `Executors are stateless`: they run prepared steps and report results.
- `Conformance is checked before run start`: runtime availability and governance
  drift must fail early.
- `Optional stages stay optional`: planner, policy, and critic should plug into
  the same run lifecycle instead of forcing separate service boundaries.

## 4. Target Service Boundaries

### 4.1 UI

Keep the UI as a separate Next.js application.

It should:

- author workflows
- inspect runs
- stream events
- inspect memory, artifacts, and debugger views

It should not contain execution semantics or hidden local orchestration logic.

### 4.2 Control Plane

This should become the main application service and own:

- chat sessions and turns
- workflow definitions, versions, and triggers
- run creation
- compile and preflight
- memory APIs
- debugger and observability APIs
- planner invocation for planner-led runs

The current API already owns most of this. The rewrite should make it a clean
modular monolith rather than keep pushing more orchestration responsibility into
separate services.

### 4.3 Scheduler

This should be the only component that advances run state.

It should:

- claim ready runs and steps
- resolve dependencies
- create attempts
- handle retries and DLQ transitions
- trigger optional critic/policy stages
- move runs to terminal states

It should not execute tools or call MCP servers directly.

### 4.4 Executor Pool

Executors should be pure step runners.

They should:

- accept a prepared `ExecutionRequest`
- invoke capabilities
- capture attempt results, events, and artifacts
- return success, retryable failure, or terminal failure

They should not decide orchestration policy.

### 4.5 Adapter Sidecars

Keep external integration runtimes separate:

- GitHub MCP
- coder service
- RAG service
- future external adapters

These are true process or network boundaries and should stay isolated.

### 4.6 Planner, Policy, Critic

Do not keep these as mandatory services by default.

Instead:

- planner becomes a compile strategy or run-preparation module
- policy becomes a run/step policy stage
- critic becomes an optional acceptance/rework stage

They can still be separated later if scale or team ownership demands it.

## 5. Canonical Domain Model

Everything should revolve around `runs`.

### 5.1 Core Objects

- `WorkflowDefinition`
- `WorkflowVersion`
- `WorkflowTrigger`
- `Run`
- `RunSpec`
- `StepSpec`
- `StepAttempt`
- `Invocation`
- `RunEvent`
- `ArtifactRecord`
- `MemoryEntry`

### 5.2 Run Kinds

`Run.kind` should distinguish source, not execution semantics:

- `planner`
- `studio`
- `chat_direct`
- `trigger`
- `api`

All of them should use the same scheduler and executor lifecycle.

### 5.3 RunSpec

`RunSpec` is the compiled, immutable execution snapshot attached to a run.

It should contain:

- run metadata
- ordered `StepSpec`s
- dependency graph
- capability requests
- input bindings
- retry policy
- acceptance policy
- expected outputs
- routing hints

This replaces the need to reason separately about `plans` and `workflow_runs`
for runtime behavior.

### 5.4 StepSpec

`StepSpec` should contain:

- `step_id`
- `name`
- `description`
- `intent`
- `capability_request`
- `input_bindings`
- `execution_gate`
- `expected_output_schema_ref`
- `retry_policy`
- `acceptance_policy`
- `depends_on`

This is the canonical execution unit.

## 6. Data Model Changes

The current schema in `services/api/app/models.py` is workable for the current
system, but it should be collapsed and extended.

### 6.1 Keep

Keep with relatively minor reshaping:

- `workflow_definitions`
- `workflow_versions`
- `workflow_triggers`
- `memory`

### 6.2 Replace Or Collapse

Replace or collapse:

- `jobs`
- `workflow_runs`
- `plans`
- `tasks`
- `event_outbox`

Recommended target tables:

### 6.3 Proposed Tables

#### `runs`

Replaces most of `jobs` and `workflow_runs`.

Suggested columns:

- `id`
- `kind`
- `source_definition_id`
- `source_version_id`
- `source_trigger_id`
- `title`
- `goal`
- `requested_context_json`
- `status`
- `created_at`
- `updated_at`
- `submitted_by`
- `metadata_json`

#### `run_specs`

Immutable compiled execution snapshot.

Suggested columns:

- `id`
- `run_id`
- `compiler_version`
- `spec_json`
- `created_at`

#### `step_specs`

Materialized step metadata for indexing and debugger UX.

Suggested columns:

- `id`
- `run_id`
- `run_spec_id`
- `name`
- `intent`
- `capability_id`
- `depends_on_json`
- `expected_output_schema_ref`
- `status`
- `created_at`
- `updated_at`

#### `step_attempts`

Durable attempt history.

Suggested columns:

- `id`
- `run_id`
- `step_id`
- `attempt_number`
- `status`
- `worker_id`
- `started_at`
- `finished_at`
- `error_code`
- `error_message`
- `retry_classification`
- `result_summary_json`

#### `invocations`

One row per capability invocation, even if the step has one primary capability.

Suggested columns:

- `id`
- `step_attempt_id`
- `capability_id`
- `adapter_id`
- `request_json`
- `response_json`
- `status`
- `started_at`
- `finished_at`
- `error_code`
- `error_message`

#### `run_events`

Append-only audit trail.

Suggested columns:

- `id`
- `run_id`
- `step_id`
- `step_attempt_id`
- `event_type`
- `payload_json`
- `occurred_at`

#### `artifacts`

Suggested columns:

- `id`
- `run_id`
- `step_id`
- `artifact_type`
- `storage_key`
- `metadata_json`
- `created_at`

### 6.4 Redis Changes

Remove Redis as authoritative storage for:

- task results
- debugger state
- latest step errors

Keep Redis only if needed for:

- transient scheduling leases
- pub/sub fan-out
- SSE acceleration
- rate limiting

## 7. Execution Contract Rewrite

The current execution contracts mix tools and capabilities:

- `tool_requests`
- `capability_bindings`
- `execution_gates`

That is too much leakage from runtime implementation into the persisted plan
shape.

### 7.1 Target Contract

Persist:

- `capability_id`
- `inputs`
- `execution_gate`
- `retry_policy`
- `expected_output_schema_ref`

Do not persist:

- internal tool names
- adapter-local implementation details
- service-specific tool wiring

### 7.2 Capability Gateway

Introduce a clearer capability gateway layer responsible for:

- capability registry lookup
- adapter selection
- governance enforcement
- schema validation
- adapter execution
- error normalization

This becomes the only place that knows how `memory.read` maps to `memory_read`
or how a GitHub capability maps to an MCP call.

## 8. Conformance And Deploy-Time Validation

This is the most important missing piece in the current repo.

Before a workflow version can publish or a run can start, the platform should
validate:

- the requested capability exists
- the target runtime advertises it
- governance allows it in the target service
- required adapters are reachable
- required secrets/config are present

### 8.1 Runtime Manifest

Each executor deployment should expose or publish a `RuntimeManifest`:

- available capabilities
- adapter mappings
- governance-filtered capability set
- schema version
- runtime version

The control plane should use this manifest during publish and run-start checks.

This directly prevents failures caused by capability/tool/governance drift.

## 9. Unified Execution Flows

### 9.1 Planner-Led

1. create run with `kind=planner`
2. planner module compiles `RunSpec`
3. scheduler executes steps
4. run completes

### 9.2 Studio

1. draft compiles to `RunSpec`
2. publish stores immutable `RunSpec` snapshot
3. run submission clones version snapshot into `runs`
4. scheduler executes steps

### 9.3 Chat Direct

1. chat route selects direct capability or asks for clarification
2. direct capability becomes a one-step `RunSpec`
3. control plane may synchronously wait for completion
4. same run data still exists for debugger and history

This removes a major source of special-case logic.

## 10. Observability Model

The debugger should be backed by durable tables, not reconstruction from
scattered stores.

Recommended debugger view should join:

- `runs`
- `run_specs`
- `step_specs`
- `step_attempts`
- `invocations`
- `run_events`
- `artifacts`

This enables:

- exact failure reasons
- retry history
- per-invocation latency
- adapter-level errors
- deterministic run replay analysis

## 11. Migration Plan

Do not do a big-bang rewrite.

### Phase 0: Hardening Inside Current Architecture

- Add runtime conformance checks before publish and run start.
- Persist task results to Postgres in parallel with Redis.
- Surface latest task error directly in Workflow Studio.
- Keep current services, but make failures diagnosable.

### Phase 1: Introduce Canonical RunSpec

- Add `RunSpec` and `StepSpec` models in code.
- Make Studio compile output emit the new shape in parallel with current plan
  payloads.
- Add adapters to translate `RunSpec` into current `plans/tasks`.

### Phase 2: Add Durable Attempts And Events

- Introduce `step_attempts`, `invocations`, and `run_events`.
- Write them from the current worker runtime.
- Build debugger APIs against the new durable records.

### Phase 3: Add New Scheduler

- Implement a scheduler that reads `RunSpec` plus durable step state from
  Postgres.
- Use it first for Studio runs behind a feature flag.
- Keep the old Redis/event path alive during cutover.

### Phase 4: Move Planner Lane

- Planner output becomes `RunSpec`.
- Scheduler executes planner-led runs using the same path as Studio runs.
- Remove planner-specific orchestration divergence.

### Phase 5: Move Chat Direct

- Chat direct execution becomes synchronous one-step runs.
- Remove the separate direct-capability execution path.

### Phase 6: Retire Old Runtime Paths

- retire `plans/tasks` as primary runtime tables
- retire Redis-only task result storage
- reduce or remove `event_outbox` from the critical execution path

## 12. Success Criteria

The rewrite is successful when:

- every run type uses the same scheduler and executor contract
- every persisted step references only capabilities, not raw tools
- every failure is explainable from Postgres alone
- publish-time validation catches runtime availability drift
- Workflow Studio, planner, and chat share one debugger model
- operators can answer "why did this fail?" without inspecting live pod logs

## 13. What I Would Not Do

I would not:

- add more standalone services first
- keep separate orchestration semantics for chat direct execution
- keep Redis as the authoritative store for task results
- keep tool-level details in user-facing plan contracts
- treat planner, policy, and critic as mandatory network boundaries before the
  core run model is unified

## 14. Recommended First Implementation Step

If only one architectural improvement is started next, it should be:

`runtime conformance gating`

Specifically:

- build a governance-filtered runtime manifest for executors
- validate workflow versions and runs against it
- block publish/start when a capability is not actually executable

That is the highest-leverage step because it improves safety immediately without
requiring a full rewrite.
