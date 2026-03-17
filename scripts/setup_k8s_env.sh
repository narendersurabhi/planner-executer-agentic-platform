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
  | grep -Ev '^(OPENAI_API_KEY|GITHUB_TOKEN|GITHUB_CLASSIC_TOKEN|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)=' >"$tmp_env"

# Ensure K8s service DNS defaults exist when .env is compose-focused.
if ! grep -q '^DATABASE_URL=' "$tmp_env"; then
  echo "DATABASE_URL=postgresql://postgres:postgres@postgres:5432/agentic" >>"$tmp_env"
fi
if ! grep -q '^REDIS_URL=' "$tmp_env"; then
  echo "REDIS_URL=redis://redis:6379/0" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_URL=' "$tmp_env"; then
  echo "QDRANT_URL=http://qdrant:6333" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_COLLECTION=' "$tmp_env"; then
  echo "QDRANT_COLLECTION=rag_default" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_TIMEOUT_S=' "$tmp_env"; then
  echo "QDRANT_TIMEOUT_S=10" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_VECTOR_SIZE=' "$tmp_env"; then
  echo "QDRANT_VECTOR_SIZE=1536" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_DISTANCE=' "$tmp_env"; then
  echo "QDRANT_DISTANCE=Cosine" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_ON_DISK_PAYLOAD=' "$tmp_env"; then
  echo "QDRANT_ON_DISK_PAYLOAD=true" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_CREATE_PAYLOAD_INDEXES=' "$tmp_env"; then
  echo "QDRANT_CREATE_PAYLOAD_INDEXES=true" >>"$tmp_env"
fi
if ! grep -q '^QDRANT_PAYLOAD_INDEX_FIELDS=' "$tmp_env"; then
  echo "QDRANT_PAYLOAD_INDEX_FIELDS=document_id,namespace,tenant_id,user_id,workspace_id,source_uri" >>"$tmp_env"
fi
if ! grep -q '^RAG_EMBEDDING_PROVIDER=' "$tmp_env"; then
  echo "RAG_EMBEDDING_PROVIDER=openai" >>"$tmp_env"
fi
if ! grep -q '^RAG_EMBEDDING_MODEL=' "$tmp_env"; then
  echo "RAG_EMBEDDING_MODEL=text-embedding-3-small" >>"$tmp_env"
fi
if ! grep -q '^RAG_EMBEDDING_TIMEOUT_S=' "$tmp_env"; then
  echo "RAG_EMBEDDING_TIMEOUT_S=15" >>"$tmp_env"
fi
if ! grep -q '^RAG_REQUIRE_SCOPE=' "$tmp_env"; then
  echo "RAG_REQUIRE_SCOPE=true" >>"$tmp_env"
fi
if ! grep -q '^RAG_TOP_K_DEFAULT=' "$tmp_env"; then
  echo "RAG_TOP_K_DEFAULT=5" >>"$tmp_env"
fi
if ! grep -q '^RAG_TOP_K_MAX=' "$tmp_env"; then
  echo "RAG_TOP_K_MAX=20" >>"$tmp_env"
fi
if ! grep -q '^RAG_WORKSPACE_ALLOWED_EXTENSIONS=' "$tmp_env"; then
  echo "RAG_WORKSPACE_ALLOWED_EXTENSIONS=.md,.txt,.rst,.json,.yaml,.yml,.toml,.py,.ts,.tsx,.js,.jsx,.css,.html,.sql,.sh" >>"$tmp_env"
fi
if ! grep -q '^RAG_WORKSPACE_MAX_FILE_BYTES=' "$tmp_env"; then
  echo "RAG_WORKSPACE_MAX_FILE_BYTES=2000000" >>"$tmp_env"
fi
if ! grep -q '^RAG_WORKSPACE_CHUNK_SIZE_CHARS=' "$tmp_env"; then
  echo "RAG_WORKSPACE_CHUNK_SIZE_CHARS=1200" >>"$tmp_env"
fi
if ! grep -q '^RAG_WORKSPACE_CHUNK_OVERLAP_CHARS=' "$tmp_env"; then
  echo "RAG_WORKSPACE_CHUNK_OVERLAP_CHARS=200" >>"$tmp_env"
fi
if ! grep -q '^RAG_WORKSPACE_MAX_CHUNKS=' "$tmp_env"; then
  echo "RAG_WORKSPACE_MAX_CHUNKS=200" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_TEXT_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_TEXT_KEY=text" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_DOCUMENT_ID_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_DOCUMENT_ID_KEY=document_id" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_SOURCE_URI_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_SOURCE_URI_KEY=source_uri" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_NAMESPACE_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_NAMESPACE_KEY=namespace" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_TENANT_ID_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_TENANT_ID_KEY=tenant_id" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_USER_ID_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_USER_ID_KEY=user_id" >>"$tmp_env"
fi
if ! grep -q '^RAG_PAYLOAD_WORKSPACE_ID_KEY=' "$tmp_env"; then
  echo "RAG_PAYLOAD_WORKSPACE_ID_KEY=workspace_id" >>"$tmp_env"
fi

kubectl create configmap awe-config --from-env-file="$tmp_env" --dry-run=client -o yaml | kubectl apply -n awe -f -

OPENAI_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2-)
GITHUB_CLASSIC_TOKEN=$(grep -E '^GITHUB_CLASSIC_TOKEN=' .env | cut -d= -f2-)
GITHUB_TOKEN=$(grep -E '^GITHUB_TOKEN=' .env | cut -d= -f2-)
AWS_ACCESS_KEY_ID=$(grep -E '^AWS_ACCESS_KEY_ID=' .env | cut -d= -f2-)
AWS_SECRET_ACCESS_KEY=$(grep -E '^AWS_SECRET_ACCESS_KEY=' .env | cut -d= -f2-)

if [[ -n "$GITHUB_CLASSIC_TOKEN" ]]; then
  GITHUB_TOKEN="$GITHUB_CLASSIC_TOKEN"
fi

if [[ -z "$OPENAI_KEY" || -z "$GITHUB_TOKEN" || -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
  echo "OPENAI_API_KEY, GITHUB_CLASSIC_TOKEN or GITHUB_TOKEN, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY must be set in .env" >&2
  exit 1
fi

kubectl create secret generic awe-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
  --from-literal=GITHUB_TOKEN="$GITHUB_TOKEN" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --dry-run=client -o yaml | kubectl apply -n awe -f -
