# Best-in-Class Agentic Orchestration Implementation Roadmap

## Summary

- Objective: turn the platform from a strong typed DAG runner into a best-in-class agentic orchestration runtime with one canonical run model, one scheduler, durable checkpoints, adaptive routing, strong acceptance and rework loops, and first-class recovery.
- Chosen direction: core-runtime first, phased delivery, rewrite-oriented convergence.
- End state: planner-led jobs, Studio workflows, triggers, and chat-direct execution all compile into the same immutable `RunSpec`, execute through the same scheduler and executor lifecycle, and expose one debugger and one observability model.

## Key Changes

### 1. Canonical Runtime Model

- Replace overlapping orchestration semantics across `jobs`, `workflow_runs`, `plans`, and `tasks` with a canonical `Run` domain centered on `Run`, `RunSpec`, `StepSpec`, `StepAttempt`, `Invocation`, `RunEvent`, and `ArtifactRecord`.
- Keep `Run.kind` only as source metadata: `planner`, `studio`, `chat_direct`, `trigger`, `api`.
- Make capabilities the only public execution contract. Raw tool names and adapters become executor-internal details and are not persisted as orchestration intent.
- Make Postgres the source of truth for run state, step state, attempts, invocations, checkpoints, and events. Redis remains only for transient dispatch, leases, and SSE acceleration.

### 2. Single Scheduler and Prepared Execution Requests

- Introduce one scheduler stage that owns dependency resolution, readiness, retry policy, acceptance transitions, cancellation, resume, and terminal state transitions.
- Persist a prepared `ExecutionRequest` snapshot per step attempt before executor pickup. It must include resolved bindings, selected capability, policy decisions, deadlines, retry class, and context provenance.
- Add lease and heartbeat semantics for running attempts, stale-attempt reclamation, and deterministic retry classification.
- Extend retry policy beyond `max_attempts` to include retry class, backoff, jitter, timeout budget, retryable versus terminal errors, and max reworks.
- Add cooperative cancellation, pause, and resume. Executors must check cancellation before each capability invocation and after long-running adapter boundaries.

### 3. Durable Checkpointing and Replay

- Add `StepCheckpoint` persistence for long or multi-phase steps so partial progress, intermediate artifacts, and resumable substate survive worker loss.
- Support replayable resume from the last valid checkpoint instead of only restarting the whole step.
- Record checkpoint metadata in the debugger and run timeline, including checkpoint source, input digest, resume count, and replay outcome.
- Keep legacy non-checkpointed steps supported, but default all new long-running or side-effectful capabilities to explicit checkpoint policy.

### 4. Agentic Decision Quality

- Add executable dynamic control flow beyond the current narrowed DAG lowering: real `switch`, bounded `fan_out` and `fan_in`, conditional branches, and controlled runtime replanning when a step fails, returns low confidence, or violates acceptance policy.
- Formalize context construction as a deterministic pipeline with stage budgets, provenance, workflow input and variable overlays, dependency projection rules, memory retrieval policy, and explicit redaction.
- Add planner-side and worker-side routing as shadow-first rerankers with confidence gates, deterministic fallback, and full decision telemetry.
- Replace the current shallow critic behavior with a true acceptance engine that evaluates per-step `acceptance_policy`, output schemas, structured checkers, and review prompts. Human approval becomes a first-class acceptance stage rather than an implicit blocked state.

### 5. Observability, Governance, and Compatibility

- Promote the debugger to the canonical run observability surface with run timeline, step context metadata, resolved input provenance, routing decisions, acceptance history, checkpoint events, retry classification, and invocation request and response capture.
- Emit unified metrics for first-pass acceptance, fallback rate, checkpoint resume success, cancel latency, stale-lease recovery, cost by capability, and retry distribution by error class.
- Keep current `/jobs` and workflow-run APIs as compatibility facades during migration, but every response must surface canonical `run_id`.
- Keep planner, policy, and critic deployable as separate services initially, but convert them into scheduler stages conceptually so they stop owning separate lifecycle semantics.

## Public APIs, Types, and Contracts

- Add canonical APIs: `/runs`, `/runs/{id}`, `/runs/{id}/steps`, `/runs/{id}/debugger`, `/runs/{id}/cancel`, `/runs/{id}/resume`, `/runs/{id}/retry`.
- Add canonical types: `Run`, `RunSpec`, `StepSpec`, `ExecutionRequest`, `StepCheckpoint`, `AcceptanceDecision`, `RoutingDecision`, `CapabilityInvocationRecord`.
- Change event schema to run and step terminology: `run.created`, `step.ready`, `step.started`, `step.completed`, `step.failed`, `routing.decision`, `acceptance.decision`, `checkpoint.saved`, `checkpoint.resumed`.
- Keep temporary translation for existing event consumers until final cutover.
- Persist capability id as the orchestration contract on every step. Tool and adapter identifiers remain diagnostic metadata only.

## Phased Delivery

### Phase 1: Canonical Run Shadow Mode

- Add canonical run tables and models.
- Dual-write new runs, steps, attempts, invocations, and events alongside current job, workflow-run, and task records.
- Add `/runs` read APIs and debugger views backed by canonical records.
- Validate dual-write parity against current `jobs` and workflow-run behavior before any cutover.

### Phase 2: Scheduler Cut-In

- Move readiness, retries, policy gating, acceptance transitions, and resume and cancel into the new scheduler.
- Persist prepared `ExecutionRequest` snapshots and lease and heartbeat state.
- Cut executors over to consume prepared execution requests rather than reconstruct orchestration state ad hoc.

### Phase 3: Checkpoints and Replay

- Introduce checkpoint persistence and replay for long-running and side-effectful capabilities.
- Add stale-attempt recovery and replay-aware retry logic.
- Add cancel, pause, and resume enforcement at executor boundaries.

### Phase 4: Agentic Runtime Quality

- Turn on planner and worker routing in telemetry-only mode, then canary mode, then primary mode with deterministic fallback.
- Implement executable `switch`, bounded runtime branching, and controlled replanning.
- Replace the heuristic critic with acceptance policies, structured checkers, and human approval checkpoints.

### Phase 5: Cutover and Cleanup

- Make canonical runs the only orchestration system of record.
- Demote or remove legacy `jobs` and `workflow_runs` runtime semantics.
- Keep compatibility endpoints only as thin projections or retire them after downstream clients are migrated.

## Test Plan and Acceptance Criteria

- Migration and parity: dual-write records must match legacy state transitions and debugger output for planner, Studio, trigger, and chat-direct lanes.
- Scheduler invariants: dependency ordering, retry and backoff behavior, rework limits, cancel and resume transitions, and policy and acceptance state changes must be covered by deterministic tests.
- Recovery: stale lease reclaim, worker crash during execution, checkpoint resume, replay after API restart, and outbox and event delivery recovery must all be exercised.
- Agentic behavior: routing shadow-versus-live comparison, low-confidence fallback, runtime replan triggers, and acceptance-engine pass and rework decisions must be tested end to end.
- Performance: no regression beyond agreed scheduler and debugger SLOs. Routing and checkpointing must stay within bounded latency budgets.
- Success targets for final cutover:
  - increase first-pass acceptance rate
  - reduce retry and DLQ rates
  - reduce mean time to useful artifact
  - preserve or improve p95 execution latency for common workflows
  - keep deterministic fallback available for low-confidence or degraded-routing cases

## Assumptions and Defaults

- Rewrite-oriented does not mean big-bang replacement. Default rollout is shadow, dual-write, canary, cutover, then cleanup.
- Postgres is authoritative for orchestration state. Redis is transient transport only.
- Capability ids remain the public runtime contract. Raw tool names are internal executor details.
- Planner, policy, and critic remain optionally deployable, but the roadmap treats them as scheduler stages rather than separate orchestration owners.
