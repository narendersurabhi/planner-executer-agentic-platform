# Kubernetes Deployment

This directory provides a baseline Kubernetes deployment for AWE with autoscaling.

## What is included

- Core services: `api`, `planner`, `worker`, `coder`, `tailor`, `ui`
- Data services: `postgres`, `redis`
- Autoscaling:
  - `hpa-api.yaml` (CPU + memory)
  - `hpa-coder.yaml` (CPU + memory)
  - `hpa-tailor.yaml` (CPU + memory)
  - `keda-worker-scaledobject.yaml` (queue-depth scaling for `worker`, optional)

## Prerequisites

- Kubernetes cluster with `metrics-server` installed (required for HPA)
- Container images published and accessible to the cluster
- A storage class supporting:
  - `ReadWriteOnce` for `postgres-data`
  - `ReadWriteMany` for `shared-data` (required for multi-worker shared workspace)

## 1) Set image names

Update image fields in:

- `api.yaml`
- `planner.yaml`
- `worker.yaml`
- `coder.yaml`
- `tailor.yaml`
- `ui.yaml`

You can build and push matching images from repo root:

```bash
IMAGE_OWNER=narendersurabhi IMAGE_TAG=v0.1.0 make images-build
IMAGE_OWNER=narendersurabhi IMAGE_TAG=v0.1.0 make images-push
IMAGE_OWNER=narendersurabhi IMAGE_TAG=v0.1.0 make images-list
```

## 2) Optional secrets

If using OpenAI/GitHub tools, create `awe-secrets`:

```bash
kubectl apply -f deploy/k8s/secret.example.yaml
```

## 3) Deploy

```bash
kubectl apply -k deploy/k8s
```

## 4) Verify autoscaling

```bash
kubectl get hpa -n awe
kubectl get deploy -n awe
```

## Local Docker Desktop profile

For local development, use the overlay at `deploy/k8s/overlays/local`:

```bash
IMAGE_OWNER=narendersurabhi IMAGE_TAG=local make images-build
make k8s-apply-local
```

This profile reduces replicas, mounts worker `/shared` from host path via `hostPath`, and uses `ClusterIP` for `ui`.

To copy generated files from worker `/shared` to your local checkout:

```bash
make k8s-sync-shared
```

## 5) Enable queue-depth autoscaling for workers (KEDA, optional)

Install KEDA in your cluster, then apply:

```bash
kubectl apply -f deploy/k8s/keda-worker-scaledobject.yaml
```

Verify:

```bash
kubectl get scaledobject -n awe
kubectl describe scaledobject worker-queue-scaler -n awe
```

If you cannot use KEDA, you can apply CPU-based worker HPA instead:

```bash
kubectl apply -f deploy/k8s/hpa-worker.yaml
```

## Notes

- `worker` queue scaler uses Redis Stream backlog for `tasks.events` + consumer group `workers`.
- Tune `pendingEntriesCount`, `minReplicaCount`, and `maxReplicaCount` in `keda-worker-scaledobject.yaml`.

## Optional observability stack (Grafana + Loki + Prometheus)

Apply:

```bash
make k8s-apply-observability
```

Delete:

```bash
make k8s-delete-observability
```

Access UIs locally:

```bash
kubectl port-forward -n awe svc/grafana 3000:3000
kubectl port-forward -n awe svc/prometheus 9090:9090
kubectl port-forward -n awe svc/jaeger 16686:16686
```

- Grafana: `http://localhost:3000` (anonymous admin enabled for local/dev)
- Prometheus: `http://localhost:9090`
- Jaeger: `http://localhost:16686`

Data sources are auto-provisioned in Grafana:

- Prometheus (`http://prometheus:9090`)
- Loki (`http://loki:3100`)
- Jaeger (`http://jaeger:16686`)
