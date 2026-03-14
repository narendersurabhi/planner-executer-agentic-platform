# Deploy Log

## 2026-03-13 Local Kubernetes Redeploy

- Command:
  `TAG=local-20260313021741 && make k8s-up-local LOCAL_IMAGE_TAG=$TAG`
- Result:
  success (`api`, `planner`, `policy`, `worker`, `coder`, `ui` rolled out).
- Post-check:
  `kubectl get deploy -n awe` and `kubectl get pods -n awe` both healthy.
