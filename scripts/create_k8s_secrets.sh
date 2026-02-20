#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/create_k8s_secrets.sh

Reads the local .env file and creates/updates the awe-secrets Kubernetes secret.
Requires: kubectl context with namespace awe.
EOF
}

if [[ ! -f .env ]]; then
  echo ".env not found" >&2
  exit 1
fi

OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2-)
GITHUB_TOKEN=$(grep -E '^GITHUB_TOKEN=' .env | cut -d= -f2-)

if [[ -z "$OPENAI_KEY" || -z "$GITHUB_TOKEN" ]]; then
  echo "OPENAI_API_KEY and GITHUB_TOKEN must be set in .env" >&2
  exit 1
fi

kubectl create secret generic awe-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
  --from-literal=GITHUB_TOKEN="$GITHUB_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
