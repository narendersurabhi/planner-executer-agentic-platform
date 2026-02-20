# Local Docker Desktop Overlay

This overlay is tuned for local Kubernetes testing:

- Single replica for core services (`api`, `planner`, `worker`, `coder`, `tailor`, `ui`)
- `worker` mounts `/shared` from a node host path (`/var/awe-shared`)
- `worker` mounts `/shared` from a node `hostPath` (`/var/awe-shared`)
- `ui` Service type is `ClusterIP` (use `kubectl port-forward`)
- HPA limits are reduced (`min=1`, `max=2`)

Default host path for `/shared` on the Kubernetes node:

- `/var/awe-shared`

Note: on Docker Desktop Kubernetes this is inside the Linux node VM, not directly a macOS filesystem path.

If your repo is in a different location, edit:

- `deploy/k8s/overlays/local/patch-worker-hostpath.json`

Default local image tags are set to:

- `ghcr.io/narendersurabhi/awe-api:local`
- `ghcr.io/narendersurabhi/awe-planner:local`
- `ghcr.io/narendersurabhi/awe-worker:local`
- `ghcr.io/narendersurabhi/awe-coder:local`
- `ghcr.io/narendersurabhi/awe-tailor:local`
- `ghcr.io/narendersurabhi/awe-ui:local`

Build images with matching tags from repo root:

```bash
IMAGE_OWNER=narendersurabhi IMAGE_TAG=local make images-build
```

Deploy:

```bash
make k8s-apply-local
```

Export generated workspace/artifacts from worker to local repo:

```bash
make k8s-sync-shared
```
