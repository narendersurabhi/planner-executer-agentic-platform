# Local Docker Desktop Overlay

This overlay is tuned for local Kubernetes testing:

- `worker`, `api`, and `rag-retriever-mcp` mount `/shared` from a node `hostPath` (`/var/awe-shared`)
- those `/shared` consumers are pinned to a single node labeled `awe.shared-workspace=true`
- `ui` Service type is `ClusterIP` (use `kubectl port-forward`)
- HPA limits are reduced (`min=1`, `max=2`)

Default host path for `/shared` on the Kubernetes node:

- `/var/awe-shared`

Note: on Docker Desktop Kubernetes this is inside the Linux node VM, not directly a macOS filesystem path.

The local shared-workspace node is chosen automatically by:

- `scripts/ensure_local_shared_node.sh`

Override it by setting:

- `AWE_SHARED_NODE=<node-name>`

Default local image tags are set to:

- `ghcr.io/narendersurabhi/awe-api:local`
- `ghcr.io/narendersurabhi/awe-planner:local`
- `ghcr.io/narendersurabhi/awe-worker:local`
- `ghcr.io/narendersurabhi/awe-coder:local`
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
