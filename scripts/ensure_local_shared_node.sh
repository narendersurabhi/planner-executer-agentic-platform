#!/usr/bin/env bash
set -euo pipefail

label_key="${AWE_SHARED_NODE_LABEL_KEY:-awe.shared-workspace}"
label_value="${AWE_SHARED_NODE_LABEL_VALUE:-true}"
preferred_node="${AWE_SHARED_NODE:-}"

choose_node() {
  if [[ -n "$preferred_node" ]]; then
    echo "$preferred_node"
    return
  fi

  existing="$(
    kubectl get nodes -l "${label_key}=${label_value},!node-role.kubernetes.io/control-plane" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
      | head -n 1
  )"
  if [[ -n "$existing" ]]; then
    echo "$existing"
    return
  fi

  kubectl get nodes -l '!node-role.kubernetes.io/control-plane' \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | head -n 1
}

selected_node="$(choose_node)"
if [[ -z "$selected_node" ]]; then
  echo "ERROR: unable to choose a local shared-workspace node." >&2
  exit 1
fi

kubectl label node "$selected_node" "${label_key}=${label_value}" --overwrite >/dev/null

other_nodes="$(
  kubectl get nodes -l "${label_key}=${label_value}" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
      | grep -v "^${selected_node}$" || true
)"

if [[ -n "$other_nodes" ]]; then
  while IFS= read -r node; do
    [[ -z "$node" ]] && continue
    kubectl label node "$node" "${label_key}-" >/dev/null
  done <<< "$other_nodes"
fi

echo "Using local shared-workspace node: ${selected_node}"
