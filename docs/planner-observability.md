# Planner Observability

## Purpose

This document describes how to build observability around the planner service. The goal is to make planner behavior measurable, debuggable, and safe to promote through staging and production.

The planner converts a job and allowed capabilities into a validated execution plan. Observability around it should answer:

- Did planning start, succeed, or fail?
- Which planner mode ran: `rule_based` or `llm`?
- Which provider and model generated the plan?
- How long did each planning stage take?
- Was the LLM response parseable?
- Did repair run?
- Did validation pass?
- If validation failed, what contract failed?
- Which capabilities were selected?
- How many tasks and dependencies were produced?
- Did the resulting plan later execute successfully?

## Existing Repo Hooks

The main planner entry points are:

- [`plan_job`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py): root planning operation
- [`build_plan_request`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py): builds the planner input contract
- [`llm_plan`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py): LLM generation, parse, repair, and validation
- [`postprocess_llm_plan`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py): normalization and auto-wiring
- [`validate_plan_request`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/planner_service.py): contract validation
- [`process_stream_entry`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/runtime_service.py): stream intake and `plan.created` or `plan.failed` emission

The planner already starts a Prometheus metrics server on port `9101` through [`services/planner/app/main.py`](/Users/narendersurabhi/planner-executer-agentic-platform/services/planner/app/main.py).

Shared observability primitives already exist:

- structured logging through [`libs/core/logging.py`](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/logging.py)
- tracing helpers through [`libs/core/tracing.py`](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/tracing.py)
- general observability overview in [`docs/observability.md`](/Users/narendersurabhi/planner-executer-agentic-platform/docs/observability.md)

## Metrics

Add a planner-specific metrics module, for example:

```text
services/planner/app/metrics.py
```

Keep metric labels bounded. Labels such as `mode`, `provider`, `model`, `status`, `reason`, and `capability_id` are acceptable. Do not label by raw prompt, job goal, session ID, user ID, or free-form error text.

### Counters

Recommended counters:

```text
planner_jobs_total{status,mode,provider}
planner_llm_requests_total{operation,status,provider,model}
planner_llm_parse_failures_total{provider,model}
planner_llm_repair_attempts_total{status,provider,model}
planner_validation_total{status,reason}
planner_plan_events_total{event_type,status}
planner_selected_capabilities_total{capability_id}
```

Important validation reason examples:

```text
intent_segment.output_format_mismatch
intent_segment.must_have_inputs_missing
intent_segment.intent_mismatch
intent_segment.risk_level_mismatch
capability_intent_invalid
capability_inputs_invalid
tool_inputs_invalid
tool_not_allowed
unknown_tool_or_capability
planner_request_language_invalid
```

The recent failure:

```text
intent_segment_invalid:document.spec.generate:GenerateDocumentSpec:output_format_mismatch:expected=json:got=docx
```

should become a metric like:

```text
planner_validation_total{status="failed",reason="intent_segment.output_format_mismatch"}
```

### Histograms

Recommended histograms:

```text
planner_plan_duration_seconds{mode,provider}
planner_llm_duration_seconds{operation,provider,model}
planner_validation_duration_seconds
planner_plan_task_count
planner_plan_dependency_depth
planner_prompt_chars
planner_response_chars
```

### Gauges

Recommended gauges for stream visibility:

```text
planner_stream_pending_messages{stream,consumer_group}
planner_stream_lag{stream,consumer_group}
```

## Structured Logs

Use the shared structured logger from [`libs/core/logging.py`](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/logging.py).

Recommended planner log events:

```text
planner_job_started
planner_job_completed
planner_job_failed
planner_llm_plan_started
planner_llm_plan_parse_failed
planner_llm_plan_repair_started
planner_llm_plan_invalid
planner_validation_failed
planner_capabilities_selected
```

Each event should include:

```text
job_id
correlation_id
planner_mode
provider
model
planner_version
task_count
selected_capabilities
duration_ms
failure_code
validation_reason
```

Do not log:

- full prompts by default
- full user goals when they may contain private data
- full plan JSON by default
- secrets or environment values
- large job context fields

The planner currently logs the full candidate plan in `llm_plan`. Before production, replace that with a safe summary log containing task count, edge count, capability list, and validation status.

## Tracing

Use the shared tracing helper in [`libs/core/tracing.py`](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/tracing.py).

Planner tracing should be bootstrapped explicitly so spans export through `OTEL_EXPORTER_OTLP_ENDPOINT`. The general observability documentation notes that tracing support exists but is not uniformly initialized across all services yet.

Recommended span tree:

```text
planner.plan_job
  planner.build_plan_request
  planner.semantic_capability_hints
  planner.llm_generate_plan
  planner.llm_repair_plan
  planner.postprocess_plan
    planner.ensure_task_intents
    planner.ensure_renderer_required_inputs
    planner.ensure_execution_bindings
  planner.validate_plan
  planner.emit_plan_created
```

Recommended span attributes:

```text
job.id
correlation_id
planner.mode
llm.provider
llm.model
goal.length
semantic_hint.count
plan.task_count
plan.edge_count
plan.max_depth
validation.status
validation.reason
```

On exceptions:

- record the exception on the span
- set span status to error
- attach a normalized failure reason, not raw error text

For validation failures, prefer normalized reasons such as:

```text
intent_segment.output_format_mismatch
capability_intent_invalid
tool_inputs_invalid
```

## Event Stream Observability

The planner reads job events and emits plan events through Redis streams.

Observe these stream-level behaviors:

- number of job events consumed
- number of `plan.created` events emitted
- number of `plan.failed` events emitted
- message processing duration
- pending messages by consumer group
- retry or stuck message count

The planner should emit a structured log and increment a metric for every `plan.created` and `plan.failed` event.

## Dashboards

Create a Grafana dashboard named `Planner`.

Recommended panels:

- planning success rate
- planning failure rate by reason
- p50, p95, and p99 planning latency
- LLM latency by provider/model
- LLM parse failure rate
- LLM repair attempt rate
- validation failure rate by reason
- selected capability distribution
- plan task count distribution
- plan dependency depth distribution
- `plan.created` and `plan.failed` events over time
- Redis stream pending messages for the planner consumer group

## Alerts

Start with these alerts:

```text
PlannerPlanFailureRateHigh
PlannerValidationFailureRateHigh
PlannerP95LatencyHigh
PlannerLLMParseFailuresHigh
PlannerStreamBacklogHigh
PlannerNoPlansCreated
```

Initial threshold suggestions:

- `plan.failed` > 5% for 10 minutes
- planner validation failures > 10% for 10 minutes
- p95 planning latency > 30 seconds for 10 minutes
- LLM parse failures > 5% for 10 minutes
- planner stream pending messages > 100 for 5 minutes
- zero `plan.created` while jobs are being created

Tune these thresholds after collecting baseline data from staging.

## Implementation Plan

### Phase 1: Add Planner Metrics

Create `services/planner/app/metrics.py`.

Add helpers for:

- plan start/success/failure
- LLM request timing
- parse failure count
- repair attempt count
- validation result count
- selected capability count

Wire metrics into:

- `process_stream_entry`
- `plan_job`
- `llm_plan`
- `validate_plan_request`

### Phase 2: Improve Safe Logs

Add summary logs for:

- planner request built
- semantic hints selected
- LLM plan generated
- repair attempted
- validation failed
- plan emitted

Replace full-plan logs with safe summaries.

Example summary shape:

```json
{
  "event": "planner_plan_candidate",
  "job_id": "job-123",
  "planner_version": "1.0.0",
  "task_count": 4,
  "edge_count": 3,
  "selected_capabilities": ["document.spec.generate", "document.docx.render"],
  "duration_ms": 842
}
```

### Phase 3: Add Planner Traces

Initialize OTLP tracing in the planner service.

Wrap the main boundaries:

- `planner.plan_job`
- `planner.llm_generate_plan`
- `planner.llm_repair_plan`
- `planner.postprocess_plan`
- `planner.validate_plan`
- `planner.emit_plan_event`

### Phase 4: Add Grafana Dashboard

Add a planner dashboard JSON under:

```text
deploy/k8s/observability
```

The first dashboard should focus on:

- health
- latency
- validation failures
- provider/model behavior
- selected capability distribution

### Phase 5: Add Regression Checks

Add tests that verify planner failures are categorized into stable observability reason codes.

Good regression cases:

- output format mismatch
- missing required inputs
- capability intent mismatch
- raw adapter tool name emitted instead of capability ID
- unknown tool or capability

## Production Readiness Checklist

Before relying on planner observability in production, confirm:

- planner `/metrics` is scraped by Prometheus
- `planner_jobs_total` increments on every job
- `planner_validation_total` has bounded reason labels
- `plan.created` and `plan.failed` events are visible in logs
- planner traces appear in Jaeger or the configured OTLP backend
- logs contain `job_id` and `correlation_id`
- full prompts and full plan JSON are not logged by default
- Grafana dashboard has success rate, latency, and failure reason panels
- alerts exist for failure rate, latency, parse failures, and backlog

## Recommended First Change

The highest-value first change is to add stable validation failure metrics.

That makes failures like:

```text
output_format_mismatch:expected=json:got=docx
```

visible as a trend instead of a one-off log line.

Once validation failures are counted by reason, staging and production promotion can answer a concrete question:

Did this release reduce planner failures, or did it introduce a new class of invalid plans?
