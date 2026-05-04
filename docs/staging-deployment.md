# Staging Deployment Runbook

This document describes how to deploy AWE to staging, verify the rollout, and run the staging regression gate before production promotion.

## Profiles

There are two supported staging paths in this repo:

1. `deploy/k8s/overlays/staging`
   Use this on a real staging cluster with a storage class that supports:
   - `ReadWriteOnce` for Postgres, Redis, and Qdrant
   - `ReadWriteMany` for `shared-data`

2. `deploy/k8s/overlays/staging-desktop`
   Use this on Docker Desktop. It replaces the `shared-data` PVC with a `hostPath` mount at `/var/awe-shared` because Docker Desktop does not provide `ReadWriteMany` storage by default.

## Prerequisites

- `kubectl` configured for the target cluster
- `metrics-server` installed in the cluster
- a registry the cluster can pull from
- a populated `.env.staging`
- published images for:
  - `awe-api`
  - `awe-planner`
  - `awe-worker`
  - `awe-coder`
  - `awe-ui`
  - `awe-policy`
  - `awe-rag-retriever-mcp`

## Staging Environment File

The staging bootstrap path reads non-secret defaults from `.env.example` and overlays staging-specific values from `.env.staging`.

The repo ignores `.env.staging`, so keep real secrets there and do not commit them.

At minimum, `.env.staging` should set the provider and any provider-specific secrets you actually use. For the current Kubernetes path, these settings matter:

```bash
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agentic
POLICY_CONFIG_PATH=/app/config/policy.yaml
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=<key>
GITHUB_TOKEN=<token>
NEXT_PUBLIC_DEV_TOOLS=false
CHAT_ROUTING_CALIBRATOR_ENABLED=true
CHAT_ROUTING_CALIBRATOR_LIVE=false
CHAT_ROUTING_CALIBRATOR_MIN_PROBABILITY=0.65
CHAT_ROUTING_CALIBRATOR_MIN_MARGIN=0.08
```

Add these when your staging setup depends on them:

- `OPENAI_API_KEY` if generation or embeddings still use OpenAI
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` if `DOCUMENT_STORE_BACKEND=s3`
- `RAG_RETRIEVER_MCP_TOKEN` and `QDRANT_API_KEY` if your retriever setup requires them

The namespace-aware bootstrap is implemented in `scripts/setup_k8s_env.sh`.

## Common Deployment Flow

The staging make targets are defined in `Makefile`.

Important variables:

- `STAGING_NAMESPACE` defaults to `awe-staging`
- `STAGING_ENV_FILE` defaults to `.env.staging`
- `STAGING_OVERLAY` defaults to `deploy/k8s/overlays/staging`
- `STAGING_IMAGE_REGISTRY`
- `STAGING_IMAGE_OWNER`
- `STAGING_IMAGE_TAG`

The standard flow is:

1. build and push images
2. apply the staging overlay
3. create or update `awe-config` and `awe-secrets` from `.env.staging`
4. pin deployments to the selected image tag
5. wait for Postgres and API availability
6. run API Alembic migrations with the pinned API image
7. wait for rollouts

The `make k8s-up-staging` target runs the migration step through:

```bash
make k8s-migrate-staging
```

Run the same target manually after applying a migration-only API image or when you need to repair a partially applied rollout.

## Deploy To A Real Staging Cluster

Select the target context:

```bash
kubectl config use-context <staging-context>
```

Build and push the staging images:

```bash
IMAGE_REGISTRY=ghcr.io IMAGE_OWNER=<owner> IMAGE_TAG=staging make images-build
IMAGE_REGISTRY=ghcr.io IMAGE_OWNER=<owner> IMAGE_TAG=staging make images-push
```

Deploy:

```bash
STAGING_OVERLAY=deploy/k8s/overlays/staging \
STAGING_IMAGE_REGISTRY=ghcr.io \
STAGING_IMAGE_OWNER=<owner> \
STAGING_IMAGE_TAG=staging \
make k8s-up-staging
```

Verify:

```bash
kubectl get deployment -n awe-staging
kubectl get pods -n awe-staging -o wide
kubectl get svc -n awe-staging
```

If the UI stays internal, test it with port-forward:

```bash
kubectl port-forward -n awe-staging svc/ui 3001:80
```

Then open `http://localhost:3001`.

## Deploy To Docker Desktop

This is the path to use when you want a local staging environment on the `docker-desktop` context.

Select the context:

```bash
kubectl config use-context docker-desktop
```

Ensure the local registry exists:

```bash
docker ps --format '{{.Names}}' | grep -q '^local-registry$' \
  || docker run -d -p 5001:5000 --name local-registry registry:2
```

Build and push `staging` images to the local registry:

```bash
IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=staging make images-build
IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=staging make images-push
```

Label the shared-workspace node:

```bash
./scripts/ensure_local_shared_node.sh
```

Deploy with the desktop staging overlay:

```bash
STAGING_OVERLAY=deploy/k8s/overlays/staging-desktop \
STAGING_IMAGE_REGISTRY=localhost:5001 \
STAGING_IMAGE_OWNER=localhost \
STAGING_IMAGE_TAG=staging \
make k8s-up-staging
```

The staging overlay forces `imagePullPolicy: Always` for `api` and `planner`. This prevents stale cached images when reusing mutable tags such as `staging`.

For higher-confidence rollouts, prefer unique image tags instead of reusing `staging`, for example:

```bash
STAGING_OVERLAY=deploy/k8s/overlays/staging-desktop \
STAGING_IMAGE_REGISTRY=localhost:5001 \
STAGING_IMAGE_OWNER=localhost \
STAGING_IMAGE_TAG=staging-20260420-001 \
make k8s-up-staging
```

Verify:

```bash
kubectl get deployment -n awe-staging
kubectl get pods -n awe-staging -o wide
```

Open the UI:

```bash
kubectl port-forward -n awe-staging svc/ui 3001:80
```

Then open `http://localhost:3001`.

## Reconfigure Or Restart Staging

Refresh the cluster env and secrets after editing `.env.staging`:

```bash
make setup-k8s-env-staging
```

Restart the app tier:

```bash
make k8s-restart-staging
```

If you need to override the overlay during reapply:

```bash
STAGING_OVERLAY=deploy/k8s/overlays/staging-desktop make k8s-apply-staging
```

## Run The Staging Regression Gate

The main live regression target is the chat boundary suite:

```bash
CHAT_BOUNDARY_LIVE_BASE_URL=https://staging.example.internal \
CHAT_BOUNDARY_LIVE_BEARER_TOKEN=<token> \
CHAT_BOUNDARY_LIVE_MIN_PASS_RATE=1.0 \
make eval-chat-boundary-live
```

If you want to export staging feedback and rebuild routing calibrator artifacts first:

```bash
CHAT_ROUTING_FEEDBACK_BASE_URL=https://staging.example.internal \
CHAT_ROUTING_FEEDBACK_BEARER_TOKEN=<token> \
CHAT_ROUTING_FEEDBACK_LIMIT=5000 \
CHAT_ROUTING_CALIBRATOR_MIN_EXAMPLES=10 \
make build-chat-routing-calibrator-from-api
```

Replay the calibrator against exported feedback:

```bash
make eval-chat-routing-calibrator
```

The GitHub Actions workflow for the staging routing gate is `.github/workflows/staging-routing-gate.yml`. It supports:

- manual `workflow_dispatch`
- reusable `workflow_call`
- environment secret `STAGING_API_BASE_URL`
- optional environment secret `STAGING_API_BEARER_TOKEN`

Promotion rule:

1. staging deployment is healthy
2. live regression suite passes
3. any calibrator replay artifacts look acceptable
4. only then promote the same image tag to production

The end-to-end staging release workflow is `.github/workflows/release-staging.yml`. It builds and pushes immutable images, deploys the resolved tag to staging, and then calls the staging routing gate.

## Verify Agent Registry

Agent Registry releases have a dedicated live e2e verifier. It creates a temporary Agent Definition, confirms it appears in the enabled profile list, updates it, launches a profile-backed Agent Sandbox run, verifies the persisted `agent_definition_snapshot`, launches an unsaved direct Agent Sandbox run, soft-deletes the profile, and confirms the deleted profile no longer appears by default.

Run it against staging:

```bash
AGENT_REGISTRY_E2E_BASE_URL=https://staging.example.internal \
AGENT_REGISTRY_E2E_BEARER_TOKEN=<token> \
make verify-agent-registry-staging
```

The report is written to:

```bash
artifacts/evals/agent_registry_staging_e2e_report.json
```

The release workflow runs this verifier after staging deployment and before the broader routing regression gate.

## Troubleshooting

### API crashes on startup with `could not translate host name "db"`

Your staging env file still contains the Compose database hostname. Use:

```bash
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agentic
```

Then rerun:

```bash
make setup-k8s-env-staging
kubectl rollout restart deployment/api -n awe-staging
```

### Policy crashes on startup with `config/policy.yaml` not found

Use the in-container path:

```bash
POLICY_CONFIG_PATH=/app/config/policy.yaml
```

Then rerun:

```bash
make setup-k8s-env-staging
kubectl rollout restart deployment/policy -n awe-staging
```

### Pods in Docker Desktop cannot mount shared storage

Use the desktop overlay instead of the generic staging overlay:

```bash
STAGING_OVERLAY=deploy/k8s/overlays/staging-desktop make k8s-up-staging
```

### A deployment is stuck on the wrong image

Pin the exact staging tag again:

```bash
STAGING_IMAGE_REGISTRY=<registry> \
STAGING_IMAGE_OWNER=<owner> \
STAGING_IMAGE_TAG=staging \
make k8s-pin-staging-images
```

## Teardown

Delete the staging application resources while keeping PVC-backed data:

```bash
make k8s-delete-staging
```
