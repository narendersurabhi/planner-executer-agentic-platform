# Runbook

## Local Kubernetes Redeploy (Docker Desktop)

Use a fixed tag so build/push/pin all reference the same image versions.

```bash
TAG=local-$(date +%Y%m%d%H%M%S)
make k8s-up-local LOCAL_IMAGE_TAG=$TAG
```

What this does:

1. Builds local images.
2. Pushes them to `localhost:5001`.
3. Applies local overlay manifests.
4. Pins deployments to the exact `$TAG`.
5. Waits for rollout of `api`, `planner`, `policy`, `worker`, `coder`, `ui`.

## Verify Deployment

```bash
kubectl get deploy -n awe
kubectl get pods -n awe
```

Expected: every core deployment is `1/1` and pods are `Running`.

## Troubleshooting

If rollout fails with `ImagePullBackOff`:

1. Ensure a fixed `LOCAL_IMAGE_TAG` was used (do not let it change between build/push and pin).
2. Confirm image exists in local registry:
   `docker pull localhost:5001/localhost/awe-api:<tag>`
3. Re-pin images:
   `IMAGE_TAG=<tag> make k8s-pin-local-images`
4. Re-check rollouts:
   `make k8s-restart-local`

If Docker push fails with local containerd I/O errors (for example `meta.db: input/output error`):

1. Restart Docker Desktop.
2. Re-run fixed-tag deploy command above.
