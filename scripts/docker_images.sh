#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_images.sh <list|build|push>

Required env:
  IMAGE_OWNER         GitHub org/user (example: narendersurabhi)

Optional env:
  IMAGE_REGISTRY      Container registry (default: ghcr.io)
  IMAGE_TAG           Image tag (default: latest)
  NEXT_PUBLIC_API_URL UI build arg (default: http://localhost:18000)
  NEXT_PUBLIC_DEV_TOOLS UI build arg (default: false)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

action="$1"
registry="${IMAGE_REGISTRY:-ghcr.io}"
owner="${IMAGE_OWNER:-}"
tag="${IMAGE_TAG:-latest}"

if [[ -z "$owner" ]]; then
  echo "ERROR: IMAGE_OWNER is required (example: IMAGE_OWNER=narendersurabhi)." >&2
  exit 1
fi

services=(api planner worker coder tailor ui policy)

image_for() {
  local service="$1"
  echo "${registry}/${owner}/awe-${service}:${tag}"
}

for service in "${services[@]}"; do
  image="$(image_for "$service")"

  case "$action" in
    list)
      echo "${service}=${image}"
      ;;
    build)
      if [[ "$service" == "ui" ]]; then
        docker build \
          --build-arg "NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-http://localhost:18000}" \
          --build-arg "NEXT_PUBLIC_DEV_TOOLS=${NEXT_PUBLIC_DEV_TOOLS:-false}" \
          -f "services/${service}/Dockerfile" \
          -t "${image}" \
          .
      else
        docker build \
          -f "services/${service}/Dockerfile" \
          -t "${image}" \
          .
      fi
      ;;
    push)
      docker push "${image}"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done
