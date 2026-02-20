# Kubernetes Local Run + Port Forward

Use these commands to spin up the stack on local Kubernetes and access it from your host.

## Spin up Kubernetes deployment (local)

```bash
# 1) Use Docker Desktop Kubernetes context
kubectl config use-context docker-desktop

# 2) Build + push local images + deploy manifests + restart workloads
make k8s-up-local
```

If you already built/pushed images and only want to apply manifests:

```bash
make k8s-apply-local
```

Quick verification:

```bash
kubectl get pods -n awe
kubectl get svc -n awe
```

## Core app services

```bash
# API
kubectl port-forward -n awe svc/api 18000:8000

# UI (Service port 80 -> container 3000)
kubectl port-forward -n awe svc/ui 8510:80
```

## Optional internal services

```bash
# Coder
kubectl port-forward -n awe svc/coder 18001:8000

# Tailor
kubectl port-forward -n awe svc/tailor 18002:8000
```

## Observability services

```bash
# Jaeger UI
kubectl port-forward -n awe svc/jaeger 16686:16686

# Grafana
kubectl port-forward -n awe svc/grafana 3000:3000

# Prometheus
kubectl port-forward -n awe svc/prometheus 9090:9090
```

## Tear down (keep data)

```bash
make k8s-down-local
```

## Notes

- Run each `kubectl port-forward` in a separate terminal.
- If a local port is already in use, change the left side only (example: `8511:80`).
