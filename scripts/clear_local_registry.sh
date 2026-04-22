#!/usr/bin/env bash

set -euo pipefail

REGISTRY_CONTAINER="${REGISTRY_CONTAINER:-local-registry}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:2}"
REGISTRY_DATA_DIR="${REGISTRY_DATA_DIR:-/var/lib/registry/docker/registry/v2}"

usage() {
  cat <<'EOF'
Usage: scripts/clear_local_registry.sh [--yes]

Deletes all images and tags from the local Docker registry used by this repo.

Environment variables:
  REGISTRY_CONTAINER  Registry container name (default: local-registry)
  REGISTRY_IMAGE      Helper image used for volume cleanup (default: registry:2)
  REGISTRY_DATA_DIR   Registry data root inside the container

Examples:
  scripts/clear_local_registry.sh
  scripts/clear_local_registry.sh --yes
  REGISTRY_CONTAINER=my-registry scripts/clear_local_registry.sh --yes
EOF
}

confirm=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes)
      confirm=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! docker container inspect "$REGISTRY_CONTAINER" >/dev/null 2>&1; then
  echo "Registry container '$REGISTRY_CONTAINER' was not found." >&2
  exit 1
fi

if [[ "$confirm" -ne 1 ]]; then
  printf "Delete all images from registry container '%s'? [y/N] " "$REGISTRY_CONTAINER" >&2
  read -r response
  case "$response" in
    y|Y|yes|YES)
      ;;
    *)
      echo "Aborted." >&2
      exit 1
      ;;
  esac
fi

was_running="$(docker inspect -f '{{.State.Running}}' "$REGISTRY_CONTAINER")"

if [[ "$was_running" == "true" ]]; then
  docker stop "$REGISTRY_CONTAINER" >/dev/null
fi

restart_registry() {
  if [[ "$was_running" == "true" ]]; then
    docker start "$REGISTRY_CONTAINER" >/dev/null
  fi
}

trap restart_registry EXIT

docker run --rm --volumes-from "$REGISTRY_CONTAINER" "$REGISTRY_IMAGE" sh -lc "
  rm -rf '$REGISTRY_DATA_DIR'/*
  mkdir -p '$REGISTRY_DATA_DIR/blobs' '$REGISTRY_DATA_DIR/repositories'
"

remaining_repos="$(
  docker run --rm --volumes-from "$REGISTRY_CONTAINER" "$REGISTRY_IMAGE" sh -lc "
    find '$REGISTRY_DATA_DIR/repositories' -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' '
  "
)"

trap - EXIT
restart_registry

echo "Cleared registry '$REGISTRY_CONTAINER'. Remaining repository directories: ${remaining_repos:-0}"
