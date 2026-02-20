#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/k8s_sync_shared.sh <workspace|artifacts|all>

Environment overrides:
  K8S_NAMESPACE           (default: awe)
  K8S_WORKER_SELECTOR     (default: app=worker)
  K8S_WORKER_CONTAINER    (default: worker)
  K8S_SYNC_DEST_ROOT      (default: .)
  K8S_SYNC_WORKSPACE_DIR  (default: ${K8S_SYNC_DEST_ROOT}/workspace)
  K8S_SYNC_ARTIFACTS_DIR  (default: ${K8S_SYNC_DEST_ROOT}/artifacts)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
namespace="${K8S_NAMESPACE:-awe}"
selector="${K8S_WORKER_SELECTOR:-app=worker}"
container="${K8S_WORKER_CONTAINER:-worker}"
dest_root="${K8S_SYNC_DEST_ROOT:-.}"
workspace_dest="${K8S_SYNC_WORKSPACE_DIR:-${dest_root}/workspace}"
artifacts_dest="${K8S_SYNC_ARTIFACTS_DIR:-${dest_root}/artifacts}"

case "$mode" in
  workspace|artifacts|all) ;;
  *)
    usage
    exit 1
    ;;
esac

pod="$(
  kubectl get pods \
    -n "$namespace" \
    -l "$selector" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
)"

if [[ -z "$pod" ]]; then
  echo "ERROR: no running worker pod found in namespace '$namespace' with selector '$selector'." >&2
  exit 1
fi

copy_dir() {
  local remote_path="$1"
  local local_path="$2"

  mkdir -p "$local_path"
  kubectl exec -n "$namespace" -c "$container" "$pod" -- sh -lc "test -d '$remote_path'"
  kubectl cp -c "$container" "${namespace}/${pod}:${remote_path}/." "$local_path"
  echo "Synced ${remote_path} -> ${local_path}"
}

if [[ "$mode" == "workspace" || "$mode" == "all" ]]; then
  copy_dir "/shared/workspace" "$workspace_dest"
fi

if [[ "$mode" == "artifacts" || "$mode" == "all" ]]; then
  copy_dir "/shared/artifacts" "$artifacts_dest"
fi
