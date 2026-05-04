# CI/CD Pipeline

This repo now has a Docker plus Kubernetes CI/CD path built around GitHub Actions.

## Workflows

### `ci`

File: `.github/workflows/ci.yml`

Runs on every push and pull request.

It performs:

- lint
- format check
- type check
- pytest
- repo-native eval gates
- advisory DeepEval gate
- Docker Compose build

This is the quality gate for merge readiness.

### `release-staging`

File: `.github/workflows/release-staging.yml`

Triggers:

- automatically after `ci` succeeds on `main`
- manually through `workflow_dispatch`

Behavior:

1. resolve an immutable image tag
2. build and push Docker images to GHCR unless an existing `image_tag` was supplied
3. write `.env.staging` from a GitHub secret
4. deploy the images to the staging Kubernetes cluster
5. run the existing staging routing regression workflow

The staging deployment uses the repo-native path:

- `make k8s-up-staging`

### `promote-production`

File: `.github/workflows/promote-production.yml`

Trigger:

- manual `workflow_dispatch`

Behavior:

1. accept a previously built immutable `image_tag`
2. write `.env.production` from a GitHub secret
3. deploy that exact tag to the production Kubernetes cluster

The production deployment uses:

- `make k8s-up-production`

This keeps production promotion tag-based instead of rebuilding from a different revision.
By default, production deploys to namespace `awe` with `PRODUCTION_OVERLAY=deploy/k8s`.

## Required GitHub Secrets

### Staging

Set these in the `staging` environment:

- `STAGING_KUBECONFIG_B64`
  - base64-encoded kubeconfig for the staging cluster
- `STAGING_ENV_FILE`
  - full `.env.staging` file contents
- `STAGING_API_BASE_URL`
  - base URL used by the staging regression workflow
- `STAGING_API_BEARER_TOKEN`
  - optional token for the staging regression workflow

### Production

Set these in the `production` environment:

- `PRODUCTION_KUBECONFIG_B64`
  - base64-encoded kubeconfig for the production cluster
- `PRODUCTION_ENV_FILE`
  - full `.env.production` file contents

## Image Tag Strategy

Automatic staging releases use:

- `sha-<12-char-commit-sha>`

Example:

- `sha-4f2c1a9b7d31`

This is the tag that should later be supplied to `promote-production`.

## Local Parity

The workflow deploy logic maps directly to Make targets, so the same flow can be run locally:

### Staging

```bash
STAGING_IMAGE_REGISTRY=ghcr.io \
STAGING_IMAGE_OWNER=<owner> \
STAGING_IMAGE_TAG=sha-<commit> \
make k8s-up-staging
```

### Production

```bash
PRODUCTION_IMAGE_REGISTRY=ghcr.io \
PRODUCTION_IMAGE_OWNER=<owner> \
PRODUCTION_IMAGE_TAG=sha-<commit> \
make k8s-up-production
```

## Promotion Model

Recommended path:

1. merge to `main`
2. let `ci` pass
3. let `release-staging` build, push, deploy, and run staging regression
4. inspect staging artifacts and health
5. run `promote-production` with the same immutable tag

This keeps staging and production on the same image digest lineage.
