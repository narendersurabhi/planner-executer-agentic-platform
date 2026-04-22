#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_k8s_env.sh

Creates or updates the awe-config ConfigMap and awe-secrets Secret from an env file.

Optional env:
  ENV_FILE=.env
  DEFAULT_ENV_FILE=.env.example
  K8S_NAMESPACE=awe
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

env_file="${ENV_FILE:-.env}"
default_env_file="${DEFAULT_ENV_FILE:-.env.example}"
namespace="${K8S_NAMESPACE:-awe}"

if [[ ! -f "$env_file" ]]; then
  echo "$env_file missing" >&2
  exit 1
fi

warn() {
  echo "WARN: $*" >&2
}

merge_env_assignments() {
  local file
  for file in "$@"; do
    [[ -f "$file" ]] || continue
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$file" || true
  done | awk -F= '{ lines[$1] = $0 } END { for (key in lines) print lines[key] }'
}

read_env_value() {
  local key="$1"
  local file="$2"
  grep -E "^${key}=" "$file" | tail -n 1 | cut -d= -f2- || true
}

env_sources=()
if [[ -f "$default_env_file" && "$default_env_file" != "$env_file" ]]; then
  env_sources+=("$default_env_file")
fi
env_sources+=("$env_file")

# Ensure target namespace exists before applying ConfigMap/Secret.
kubectl get namespace "$namespace" >/dev/null 2>&1 || kubectl create namespace "$namespace"

tmp_env=$(mktemp)
tmp_merged=$(mktemp)
trap 'rm -f "$tmp_env" "$tmp_merged"' EXIT

merge_env_assignments "${env_sources[@]}" >"$tmp_merged"

# Build ConfigMap input from env sources but keep secrets out of ConfigMap.
grep -Ev '^(OPENAI_API_KEY|GITHUB_TOKEN|GITHUB_CLASSIC_TOKEN|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|GEMINI_API_KEY|QDRANT_API_KEY|RAG_RETRIEVER_MCP_TOKEN)=' "$tmp_merged" >"$tmp_env" || true

# Ensure K8s service DNS defaults exist when the env file is compose-focused.
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

kubectl create configmap awe-config --from-env-file="$tmp_env" --dry-run=client -o yaml | kubectl apply -n "$namespace" -f -

OPENAI_KEY=$(read_env_value OPENAI_API_KEY "$tmp_merged")
GITHUB_CLASSIC_TOKEN=$(read_env_value GITHUB_CLASSIC_TOKEN "$tmp_merged")
GITHUB_TOKEN=$(read_env_value GITHUB_TOKEN "$tmp_merged")
AWS_ACCESS_KEY_ID=$(read_env_value AWS_ACCESS_KEY_ID "$tmp_merged")
AWS_SECRET_ACCESS_KEY=$(read_env_value AWS_SECRET_ACCESS_KEY "$tmp_merged")
AWS_SESSION_TOKEN=$(read_env_value AWS_SESSION_TOKEN "$tmp_merged")
GEMINI_API_KEY=$(read_env_value GEMINI_API_KEY "$tmp_merged")
QDRANT_API_KEY=$(read_env_value QDRANT_API_KEY "$tmp_merged")
RAG_RETRIEVER_MCP_TOKEN=$(read_env_value RAG_RETRIEVER_MCP_TOKEN "$tmp_merged")
LLM_PROVIDER=$(read_env_value LLM_PROVIDER "$tmp_env")
RAG_EMBEDDING_PROVIDER=$(read_env_value RAG_EMBEDDING_PROVIDER "$tmp_env")

if [[ -n "$GITHUB_CLASSIC_TOKEN" ]]; then
  GITHUB_TOKEN="$GITHUB_CLASSIC_TOKEN"
fi

if [[ "$LLM_PROVIDER" == "gemini" && -z "$GEMINI_API_KEY" ]]; then
  warn "GEMINI_API_KEY is empty while LLM_PROVIDER=gemini; Gemini-backed requests will fail."
fi
if [[ ("$LLM_PROVIDER" == "openai" || "$LLM_PROVIDER" == "openai_compatible" || "$RAG_EMBEDDING_PROVIDER" == "openai") && -z "$OPENAI_KEY" ]]; then
  warn "OPENAI_API_KEY is empty while OpenAI-backed generation or embeddings are configured."
fi
if [[ -z "$GITHUB_TOKEN" ]]; then
  warn "GITHUB_TOKEN is empty; github-mcp may fail readiness and GitHub-backed tools will be unavailable."
fi
if [[ -z "$AWS_ACCESS_KEY_ID" || -z "$AWS_SECRET_ACCESS_KEY" ]]; then
  warn "AWS credentials are incomplete; S3-backed document storage will be unavailable."
fi

kubectl create secret generic awe-secrets \
  --from-literal=OPENAI_API_KEY="$OPENAI_KEY" \
  --from-literal=GITHUB_TOKEN="$GITHUB_TOKEN" \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --from-literal=AWS_SESSION_TOKEN="$AWS_SESSION_TOKEN" \
  --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY" \
  --from-literal=QDRANT_API_KEY="$QDRANT_API_KEY" \
  --from-literal=RAG_RETRIEVER_MCP_TOKEN="$RAG_RETRIEVER_MCP_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -n "$namespace" -f -
