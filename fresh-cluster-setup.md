Use this exact flow for a **fresh Docker Desktop Kubernetes** deploy with your local registry (`localhost:5001`).

**1. Start Docker Desktop Kubernetes**
```bash
kubectl config use-context docker-desktop
kubectl get nodes
```

If this fails with `EOF`, restart Docker Desktop and wait until nodes are `Ready`.

**2. Start local registry (port 5001)**
```bash
docker ps --format '{{.Names}}' | grep -q '^local-registry$' \
  || docker run -d -p 5001:5000 --name local-registry registry:2

docker ps | grep local-registry
```

**3. Build images tagged for local registry**
```bash
IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=latest make images-build
IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=latest make images-push
```

**4. Verify images exist in registry**
```bash
curl -s http://localhost:5001/v2/localhost/awe-api/tags/list
curl -s http://localhost:5001/v2/localhost/awe-planner/tags/list
curl -s http://localhost:5001/v2/localhost/awe-worker/tags/list
curl -s http://localhost:5001/v2/localhost/awe-coder/tags/list
curl -s http://localhost:5001/v2/localhost/awe-tailor/tags/list
curl -s http://localhost:5001/v2/localhost/awe-ui/tags/list
```

**5. Create/update env + secrets from `.env`**
```bash
make setup-k8s-env
```

This now:
- ensures namespace `awe` exists
- creates/updates `awe-config`
- creates/updates `awe-secrets`
- injects Kubernetes-safe defaults for `DATABASE_URL` and `REDIS_URL` if missing

**6. Deploy local overlay**
```bash
make k8s-apply-local
```

**7. Watch rollout**
```bash
kubectl get pods -n awe -w
kubectl get deploy -n awe
```

Wait until all deployments show `READY 1/1` (or desired replicas).

**8. Access UI**
```bash
kubectl port-forward -n awe svc/ui 8501:8501
```
Then open: `http://localhost:8501`

**9. Check logs if anything fails**
```bash
kubectl logs -n awe -l app=api --tail=100
kubectl logs -n awe -l app=planner --tail=100
kubectl logs -n awe -l app=worker --tail=100
kubectl describe pod -n awe <failing-pod-name>
```

**10. Sync generated files back to host (optional)**
```bash
make k8s-sync-workspace
make k8s-sync-artifacts
```


**One Stop**
- `k8s-up-local`: full local deploy flow (context, registry, build/push images, setup env/secrets, apply manifests, wait for rollouts)
- `k8s-down-local`: delete local overlay resources
- `k8s-restart-local`: restart app deployments and wait for readiness

Use:

```bash
make k8s-up-local
```

Then access UI via port-forward:

```bash
kubectl port-forward -n awe svc/ui 8501:8501
```

Stop:

```bash
make k8s-down-local
```