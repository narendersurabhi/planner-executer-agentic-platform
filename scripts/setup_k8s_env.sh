#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_k8s_env.sh

Creates or updates the awe-config ConfigMap and awe-secrets Secret from .env.
EOF
}

if [[ ! -f .env ]]; then
  echo ".env missing" >&2
  exit 1
fi

# Ensure target namespace exists before applying ConfigMap/Secret.
kubectl get namespace awe >/dev/null 2>&1 || kubectl create namespace awe

tmp_env=$(mktemp)
trap 'rm -f "$tmp_env"' EXIT

# Build ConfigMap input from .env but keep secrets out of ConfigMap.
grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env \
  | grep -Ev '^(OPENAI_API_KEY|GITHUB_TOKEN|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)=' >"$tmp_env"

# Ensure K8s service DNS defaults exist when .env is compose-focused.
if ! grep -q '^DATABASE_URL=' "$tmp_env"; then
  echo "DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agentic" >>"$tmp_env"
fi
if ! grep -q '^REDIS_URL=' "$tmp_env"; then
  echo "REDIS_URL=redis://redis:6379/0" >>"$tmp_env"
fi

kubectl create configmap awe-config --from-env-file="$tmp_env" --dry-run=client -o yaml | kubectl apply -n awe -f -

OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2-)
GITHUB_TOKEN=$(grep -E '^GITHUB_TOKEN=' .env | cut -d= -f2-)
AWS_ACCESS_KEY_ID=$(grep -E '^AWS_ACCESS_KEY_ID=' .env | cut -d= -f2-)
AWS_SECRET_ACCESS_KEY=$(grep -E '^AWS_SECRET_ACCESS_KEY=' .env | cut -d= -f2-)

if [[ -z "$OPENAI_KEY" || -z "$GITHUB_TOKEN" || -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
  echo "OPENAI_API_KEY, GITHUB_TOKEN, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY must be set in .env" >&2
  exit 1
fi

kubectl create secret generic awe-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
  --from-literal=GITHUB_TOKEN="$GITHUB_TOKEN" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kubectl apply -n awe -f -
