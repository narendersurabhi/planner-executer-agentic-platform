# Why AWE (Goal-Driven Agentic Workflow Engine)

AWE is a production-style framework for building agentic workflows that are reliable, observable, and deployable. It treats “agents” as a distributed system: plans are explicit, tool contracts are validated, failures are classified and retried deterministically, and every run is traceable end-to-end.

This document makes the case for using AWE as the foundation for new AI-enabled automation, and for investing in it as a shared platform rather than building one-off agents.

## Executive Summary

- **Move from “prompt scripts” to an execution platform:** AWE models work as a typed task DAG, not an unstructured chat loop.
- **Improve reliability and operability:** retries, DLQs, timeouts, stale-claim recovery, and isolation for risky tool calls.
- **Make runs explainable:** plans, tool calls, schemas, and events are first-class and queryable.
- **Enable production deployment:** Docker Compose for local dev and Kubernetes manifests with autoscaling for production-style environments.

## The Problem AWE Solves

Most agent implementations start as a single process that calls an LLM, then calls tools, and “hopes for the best”. This breaks down quickly:

- **Unbounded failure modes:** tool timeouts, partial writes, repeated side effects, and flaky downstreams.
- **Low observability:** you can’t answer “what happened?”, “where did it fail?”, or “what changed?” without digging through logs by hand.
- **No contract surface:** tool inputs/outputs drift, and model updates silently break workflows.
- **Hard to scale:** concurrency increases tool flakiness, and scale changes behavior without a control plane.

AWE’s position is simple: if you want agents in production, you need **plans, contracts, and operations**.

## What AWE Is (And What It Is Not)

**AWE is:**

- A **goal → plan → execute** workflow engine with explicit tasks, dependencies, and tool requests.
- A **tool-orchestration runtime** that can call local tools and MCP-backed services.
- A set of services that provide: job management, state, event streaming, artifacts, and observability.

**AWE is not:**

- A “single-agent chat app” where state lives only in a prompt.
- A generic data pipeline engine (it can run pipelines, but its core value is agent-tool orchestration).

## Key Capabilities That Make AWE Worth Building On

### 1) Explicit planning (typed task DAG)

The planner converts a goal into tasks with:

- dependencies (`deps`)
- acceptance criteria
- tool requests (what must be called)
- expected output schemas (what shape the results must take)

This yields repeatability: two runs can be compared by plan, not just output text.

### 2) Reliable execution (workers as a data plane)

Workers execute ready tasks and emit structured events. Reliability features are built in:

- Redis Streams consumer groups for concurrency
- bounded retries with policy-based classification
- DLQ for terminal failures
- stale task recovery (`XAUTOCLAIM`)

### 3) Tool contracts and schema validation

Agents fail most often at tool boundaries. AWE makes tool I/O explicit:

- tool input validation (schema-driven)
- output validation (schema-driven)
- output size caps to prevent runaway payloads

This is the difference between “it worked yesterday” and “it keeps working”.

### 4) MCP tool orchestration (timeouts + isolation)

MCP-backed tools behave like external dependencies. AWE provides guardrails:

- deadline-budgeted tool calls
- deterministic retry classification for transport/session failures
- isolation modes (including process isolation) to prevent hangs from blocking worker progress

### 5) Artifacts and download paths

Many workflows produce files (DOCX, reports, exports). AWE supports:

- shared artifacts/workspace directories
- API download endpoints for artifacts and workspace files
- optional S3/object-store backend for cross-node-safe downloads

### 6) Production-style deployment and scaling

Running agent systems locally is easy; running them reliably is the hard part. AWE includes:

- local `docker compose` workflow for development
- Kubernetes manifests (`deploy/k8s`) for production-style runtime
- HPA for request-driven services and optional KEDA scaler for queue-driven workers

### 7) Observability as a first-class feature

Agents need distributed-systems observability:

- OpenTelemetry spans around semantic steps (LLM calls, tool calls, retries)
- Prometheus metrics for core SLOs (latency/error/throughput)
- Grafana dashboards and Jaeger tracing for debugging and postmortems

## Why Invest in Developing AWE (Instead of One-Off Agents)

Building one agent solves one problem. Building AWE solves many problems faster and safer.

Platform investments pay off when you have any combination of:

- multiple workflows and tool integrations
- multiple teams contributing tools/agents
- real users and operational requirements
- compliance/security constraints

With AWE, improvements become reusable primitives:

- new tool contracts benefit every workflow
- improved timeouts/retries reduce platform-wide failures
- stronger policy gates reduce security risk for all agents
- dashboards and traces accelerate every incident response

## Where AWE Fits (Use Cases)

Examples that benefit immediately from AWE’s structure:

- document generation with structured validation and artifact delivery
- code generation/patching workflows with bounded timeouts and artifact outputs
- “ops runbook” automation: triage, diagnostics, remediation plan generation
- internal platform bots that must follow guardrails and provide auditability

## Adoption Path (Pragmatic)

1. **Start with one workflow** that already requires tools and retries.
2. **Define schemas** for tool inputs/outputs early; treat them as contracts.
3. **Instrument the critical path**: tool calls, planner outputs, and worker retries.
4. **Add policy gates** for tool allowlists and environment-based restrictions.
5. **Scale deliberately** (HPA/KEDA) after you have SLOs and dashboards.

## Near-Term Roadmap (High ROI)

- tighten tool-level SLOs and error taxonomy across all MCP routes
- improve plan validation and failure reporting (fewer opaque “parse_failed” classes)
- add “run scorecards” (success/failure stage, latency, retries, cost, artifact links)
- formalize tool schemas and versioning in CI (break-glass migrations instead of silent drift)

## Risks and Mitigations

- **Hallucinated or unverifiable claims:** enforce schema validation; use independent evaluators for scoring; prefer “evidence required” prompts.
- **Tool side effects and idempotency:** require idempotency keys; enforce policy gates for write tools; add dry-run modes.
- **Runaway latency and cost:** strict timeouts; bounded retries; output caps; fallback modes (`mock`, heuristic evaluators).
- **Security (prompt injection / data exfiltration):** restrict HTTP fetch; require allowlists; isolate tool execution; keep secrets out of logs.

## Call to Action

Adopt AWE as the standard runtime for agentic automation where reliability, auditability, and operational control matter. Treat workflows as production software: define contracts, emit traces, and scale safely.

If you want to extend AWE, start by adding one new tool with a strict schema, tests, and observability around its calls. That single investment strengthens the platform for every future workflow.
