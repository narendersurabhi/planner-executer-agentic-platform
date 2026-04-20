# Kubernetes Deployment

This directory provides a baseline Kubernetes deployment for AWE with autoscaling.

## What is included

- Data services: `postgres`, `redis`
- Autoscaling:
  - `hpa-api.yaml` (CPU + memory)
  - `hpa-coder.yaml` (CPU + memory)
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

This profile reduces replicas, mounts `/shared` from a node `hostPath`, pins local `/shared` consumers to one labeled node for consistent workspace visibility, and uses `ClusterIP` for `ui`.

To copy generated files from worker `/shared` to your local checkout:

```bash
make k8s-sync-shared
```

## Staging profile

For a staging deployment, use `deploy/k8s/overlays/staging`. It is designed to work either in a dedicated staging cluster or in a shared cluster under the `awe-staging` namespace.

For the full staging runbook, including Docker Desktop staging and regression gating before production promotion, see `docs/staging-deployment.md`.
For the GitHub Actions release pipeline and required deployment secrets, see `docs/cicd-pipeline.md`.

Create a sparse `.env.staging` with only staging overrides and secrets; missing non-secret values fall back to `.env.example` when `setup_k8s_env.sh` renders the ConfigMap.

Example:

```bash
cat <<'EOF' > .env.staging
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=<key>
GITHUB_TOKEN=<token>
NEXT_PUBLIC_DEV_TOOLS=false
CHAT_ROUTING_CALIBRATOR_ENABLED=true
CHAT_ROUTING_CALIBRATOR_LIVE=false
CHAT_ROUTING_CALIBRATOR_MIN_PROBABILITY=0.65
CHAT_ROUTING_CALIBRATOR_MIN_MARGIN=0.08
EOF
```

Deploy:

```bash
STAGING_IMAGE_OWNER=narendersurabhi STAGING_IMAGE_TAG=<tag> make k8s-up-staging
```

This flow:

- applies the staging overlay into namespace `awe-staging`
- creates or updates `awe-config` and `awe-secrets` from `.env.staging`
- pins the workloads to `$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-*:$(STAGING_IMAGE_TAG)`
- waits for the main service rollouts to complete

The staging overlay keeps `ui` as `ClusterIP`, so test it with port-forward:

```bash
kubectl port-forward -n awe-staging svc/ui 3001:80
```

Then open `http://localhost:3001`.

Safe teardown keeps PVCs:

```bash
make k8s-delete-staging
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
