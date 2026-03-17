# Observability

This document describes the current observability model for Agentic Workflow
Studio: structured logs, Prometheus metrics, OTLP tracing, and the optional
Kubernetes observability stack.

## 1. Overview

The platform uses three main observability channels:

- `JSON logs` for service and runtime events
- `Prometheus metrics` for service health and runtime counters
- `OpenTelemetry traces` for distributed execution visibility

Observability is split across the core services rather than centralized in a
single process.

## 2. Structured Logging

Logging is configured through [logging.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/logging.py).

Current behavior:

- services use `structlog`
- output is JSON
- log level is configured through Python `logging.basicConfig(level=logging.INFO)`
- log entries include:
  - timestamp
  - log level
  - event payload
  - bound service context where applicable

Services using the shared logging setup include:

- `api`
- `planner`
- `worker`
- `policy`
- `coder`

Typical uses:

- task lifecycle events
- retry and DLQ events
- artifact sync failures
- memory write/read activity
- planner and routing diagnostics

## 3. Metrics

Metrics are exposed with Prometheus-compatible endpoints.

### Service metrics endpoints

Current metrics surfaces include:

- `api`
  - `GET /metrics`
  - mounted directly in the FastAPI app
- `planner`
  - metrics HTTP server on port `9101`
- `worker`
  - metrics HTTP server on port `9102`
- `policy`
  - metrics HTTP server on port `9103`
- `coder`
  - `GET /metrics`

In Kubernetes, Prometheus is configured to scrape:

- `api.awe.svc.cluster.local:8000/metrics`
- `planner.awe.svc.cluster.local:9101/metrics`
- `worker.awe.svc.cluster.local:9102/metrics`
- `policy.awe.svc.cluster.local:9103/metrics`
- `coder.awe.svc.cluster.local:8000/metrics`

It also scrapes the observability services themselves:

- Prometheus
- Loki
- Grafana

### Metric categories

The exact metric set evolves by service, but common categories include:

- request and service health metrics
- task and plan lifecycle counters
- orchestration recovery counters
- runtime execution counters
- queue and retry-related counters

## 4. Tracing

Tracing support is implemented through [tracing.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/tracing.py).

Current tracing behavior:

- OTLP tracing is enabled through `OTEL_EXPORTER_OTLP_ENDPOINT`
- the endpoint is normalized to `/v1/traces` automatically when needed
- spans can be created through the shared tracing helper
- trace IDs are propagated through task execution records and surfaced in the UI debugger

### Current implementation status

Today, `worker` explicitly configures OTLP export during startup.

That means:

- worker span export is the most concrete tracing path today
- the platform is trace-aware more broadly through IDs and span helpers
- full OTLP bootstrap is not yet uniformly wired across every service

So the tracing story is real, but uneven:

- `worker`: explicit OTLP tracing setup
- other services: trace-aware code paths and IDs exist, but not all services explicitly initialize OTLP export the same way

## 5. UI Debugging and Trace Navigation

The UI exposes runtime debugging views that can link to observability systems.

Current debugger-oriented behavior includes:

- trace IDs surfaced from task/job runtime data
- links to Jaeger traces
- links to Grafana Explore / Loki logs when configured

This makes the UI a practical entrypoint for jumping from job/task state into
observability tools.

## 6. Kubernetes Observability Stack

The repo includes an optional Kubernetes observability overlay under:

- [deploy/k8s/observability](/Users/narendersurabhi/planner-executer-agentic-platform/deploy/k8s/observability)

It contains:

- `Prometheus`
- `Grafana`
- `Loki`
- `Promtail`
- prebuilt Grafana dashboards
- preconfigured Grafana datasources

### Apply and remove

Apply:

```bash
make k8s-apply-observability
```

Delete:

```bash
make k8s-delete-observability
```

### Grafana datasources

Grafana is preconfigured with:

- `Prometheus`
- `Loki`
- `Jaeger`

Jaeger itself is not part of the observability overlay. It is deployed
separately in the base Kubernetes manifests.

## 7. Local Access

For local Kubernetes access, the common forwards are:

```bash
kubectl port-forward -n awe svc/grafana 3000:3000
kubectl port-forward -n awe svc/prometheus 9090:9090
kubectl port-forward -n awe svc/jaeger 16686:16686
```

Typical local URLs:

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`
- Jaeger: `http://localhost:16686`

See also:

- [k8s-port-forward.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/k8s-port-forward.md)
- [deploy/k8s/README.md](/Users/narendersurabhi/planner-executer-agentic-platform/deploy/k8s/README.md)

## 8. Deployment Notes

### Docker Compose

Docker Compose includes:

- Jaeger
- Prometheus
- Grafana

It is useful for local service visibility and basic tracing/metrics workflows.

### Local Kubernetes

Local Kubernetes is the closer match to the intended deployed shape.

It supports:

- API, planner, worker, policy, and coder metrics scraping
- optional observability overlay
- local port-forward access to dashboards and tracing tools

## 9. Known Limitations

Current limitations to be aware of:

- OTLP tracing is not bootstrapped uniformly across every service
- metrics are present across services, but not every domain has equally rich custom metrics
- log retention and production-grade log shipping are represented by the Loki stack, but not deeply documented as a long-term operations model yet
- observability guidance previously drifted across `README.md`, `deploy/k8s/README.md`, and manifests; this doc is intended to be the canonical overview

## 10. Related Documents

- [README.md](/Users/narendersurabhi/planner-executer-agentic-platform/README.md)
  High-level summary and quick reference
- [architecture.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/architecture.md)
  Canonical system architecture overview
- [deploy/k8s/README.md](/Users/narendersurabhi/planner-executer-agentic-platform/deploy/k8s/README.md)
  Kubernetes deployment and observability overlay usage
- [k8s-port-forward.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/k8s-port-forward.md)
  Local access commands for Grafana, Prometheus, and Jaeger
