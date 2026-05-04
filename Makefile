.PHONY: up up-workers down lint format typecheck test schemas eval-intent eval-intent-gate \
	eval-chat-clarification eval-chat-clarification-gate \
	eval-capability-search eval-capability-search-gate \
	eval-deepeval-chat eval-deepeval-planner eval-deepeval-gate \
	eval-deepeval-staging-replay \
	eval-chat-boundary-live \
	build-capability-feedback \
	build-chat-routing-feedback \
	build-chat-routing-calibrator-from-api \
	build-intent-tuning-candidates \
	build-capability-reranker-dataset \
	build-chat-routing-reranker-dataset \
	train-chat-routing-calibrator \
	eval-chat-routing-calibrator \
	eval-capability-feedback \
	k8s-apply k8s-delete k8s-apply-local k8s-delete-local \
	setup-k8s-env-staging k8s-apply-staging k8s-delete-staging \
	k8s-pin-staging-images k8s-migrate-staging k8s-up-staging k8s-restart-staging \
	setup-k8s-env-production k8s-apply-production \
	k8s-pin-production-images k8s-up-production k8s-restart-production \
	k8s-apply-observability k8s-delete-observability \
	k8s-apply-keda-worker k8s-delete-keda-worker \
	k8s-up-local k8s-down-local k8s-restart-local \
	clear-local-registry \
	k8s-sync-shared k8s-sync-workspace k8s-sync-artifacts \
	images-list images-build images-push \
	k8s-pin-local-images \
	verify-agent-registry-staging

IMAGE_REGISTRY ?= localhost:5001
IMAGE_OWNER ?= localhost
IMAGE_TAG ?= latest
LOCAL_IMAGE_TAG ?= local-$(shell date +%Y%m%d%H%M%S)
STAGING_NAMESPACE ?= awe-staging
STAGING_ENV_FILE ?= .env.staging
STAGING_OVERLAY ?= deploy/k8s/overlays/staging
STAGING_IMAGE_REGISTRY ?= ghcr.io
STAGING_IMAGE_OWNER ?=
STAGING_IMAGE_TAG ?= latest
PRODUCTION_NAMESPACE ?= awe
PRODUCTION_ENV_FILE ?= .env.production
PRODUCTION_OVERLAY ?= deploy/k8s
PRODUCTION_IMAGE_REGISTRY ?= ghcr.io
PRODUCTION_IMAGE_OWNER ?=
PRODUCTION_IMAGE_TAG ?= latest
UV_EVAL_DEPS = \
	--with typing-extensions \
	--with pydantic \
	--with pydantic-settings \
	--with jsonschema \
	--with python-docx \
	--with reportlab \
	--with boto3 \
	--with mcp \
	--with sqlalchemy \
	--with opentelemetry-api \
	--with opentelemetry-sdk \
	--with opentelemetry-exporter-otlp \
	--with structlog \
	--with pyyaml \
	--with fastapi \
	--with redis \
	--with psycopg2-binary \
	--with prometheus-client
DEEPEVAL_DEPS = \
	$(UV_EVAL_DEPS) \
	--with deepeval

CHAT_BOUNDARY_LIVE_VERBOSE ?= 1
CHAT_BOUNDARY_LIVE_VERBOSE_FLAG = $(if $(filter 1 true TRUE yes YES,$(CHAT_BOUNDARY_LIVE_VERBOSE)),--verbose,)
AGENT_REGISTRY_E2E_BASE_URL ?= http://127.0.0.1:18000
AGENT_REGISTRY_E2E_BEARER_TOKEN ?=
AGENT_REGISTRY_E2E_OUTPUT ?= artifacts/evals/agent_registry_staging_e2e_report.json
UV_QUALITY_DEPS = \
	$(UV_EVAL_DEPS) \
	--with pytest \
	--with mypy \
	--with ruff \
	--with docxtpl \
	--with httpx

up:
	docker compose up -d --build

up-workers:
	docker compose up -d --build --scale worker=$${WORKERS:-4}

images-list:
	IMAGE_REGISTRY=$(IMAGE_REGISTRY) IMAGE_OWNER=$(IMAGE_OWNER) IMAGE_TAG=$(IMAGE_TAG) \
		./scripts/docker_images.sh list

images-build:
	IMAGE_REGISTRY=$(IMAGE_REGISTRY) IMAGE_OWNER=$(IMAGE_OWNER) IMAGE_TAG=$(IMAGE_TAG) \
		./scripts/docker_images.sh build

images-push:
	IMAGE_REGISTRY=$(IMAGE_REGISTRY) IMAGE_OWNER=$(IMAGE_OWNER) IMAGE_TAG=$(IMAGE_TAG) \
		./scripts/docker_images.sh push

setup-k8s-env:
	./scripts/setup_k8s_env.sh

setup-k8s-env-staging:
	K8S_NAMESPACE=$(STAGING_NAMESPACE) ENV_FILE=$(STAGING_ENV_FILE) DEFAULT_ENV_FILE=.env.example ./scripts/setup_k8s_env.sh

setup-k8s-env-production:
	K8S_NAMESPACE=$(PRODUCTION_NAMESPACE) ENV_FILE=$(PRODUCTION_ENV_FILE) DEFAULT_ENV_FILE=.env.example ./scripts/setup_k8s_env.sh

k8s-apply:
	kubectl apply -k deploy/k8s

k8s-delete:
	kubectl delete -k deploy/k8s --ignore-not-found

k8s-apply-observability:
	kubectl apply -k deploy/k8s/observability

k8s-delete-observability:
	kubectl delete -k deploy/k8s/observability --ignore-not-found

configmap-from-env:
	kubectl create configmap awe-config --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -

k8s-apply-local:
	./scripts/ensure_local_shared_node.sh
	kubectl kustomize --load-restrictor LoadRestrictionsNone deploy/k8s/overlays/local | kubectl apply -f -
	./scripts/setup_k8s_env.sh

k8s-apply-staging:
	kubectl kustomize --load-restrictor LoadRestrictionsNone $(STAGING_OVERLAY) | kubectl apply -f -
	$(MAKE) setup-k8s-env-staging

k8s-apply-production:
	kubectl kustomize --load-restrictor LoadRestrictionsNone $(PRODUCTION_OVERLAY) | kubectl apply -f -
	$(MAKE) setup-k8s-env-production

k8s-pin-staging-images:
	@test -n "$(STAGING_IMAGE_OWNER)" || (echo "STAGING_IMAGE_OWNER is required" >&2; exit 1)
	kubectl set image deployment/api -n $(STAGING_NAMESPACE) api=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-api:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/planner -n $(STAGING_NAMESPACE) planner=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-planner:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/policy -n $(STAGING_NAMESPACE) policy=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-policy:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/worker -n $(STAGING_NAMESPACE) worker=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-worker:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/coder -n $(STAGING_NAMESPACE) coder=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-coder:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/rag-retriever-mcp -n $(STAGING_NAMESPACE) rag-retriever-mcp=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-rag-retriever-mcp:$(STAGING_IMAGE_TAG)
	kubectl set image deployment/ui -n $(STAGING_NAMESPACE) ui=$(STAGING_IMAGE_REGISTRY)/$(STAGING_IMAGE_OWNER)/awe-ui:$(STAGING_IMAGE_TAG)

k8s-migrate-staging:
	kubectl exec -n $(STAGING_NAMESPACE) deploy/api -- python -m alembic -c /app/services/api/app/alembic.ini upgrade head

k8s-pin-production-images:
	@test -n "$(PRODUCTION_IMAGE_OWNER)" || (echo "PRODUCTION_IMAGE_OWNER is required" >&2; exit 1)
	kubectl set image deployment/api -n $(PRODUCTION_NAMESPACE) api=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-api:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/planner -n $(PRODUCTION_NAMESPACE) planner=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-planner:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/policy -n $(PRODUCTION_NAMESPACE) policy=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-policy:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/worker -n $(PRODUCTION_NAMESPACE) worker=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-worker:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/coder -n $(PRODUCTION_NAMESPACE) coder=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-coder:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/rag-retriever-mcp -n $(PRODUCTION_NAMESPACE) rag-retriever-mcp=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-rag-retriever-mcp:$(PRODUCTION_IMAGE_TAG)
	kubectl set image deployment/ui -n $(PRODUCTION_NAMESPACE) ui=$(PRODUCTION_IMAGE_REGISTRY)/$(PRODUCTION_IMAGE_OWNER)/awe-ui:$(PRODUCTION_IMAGE_TAG)

k8s-pin-local-images:
	kubectl set image deployment/api -n awe api=localhost:5001/localhost/awe-api:$(IMAGE_TAG)
	kubectl set image deployment/planner -n awe planner=localhost:5001/localhost/awe-planner:$(IMAGE_TAG)
	kubectl set image deployment/policy -n awe policy=localhost:5001/localhost/awe-policy:$(IMAGE_TAG)
	kubectl set image deployment/worker -n awe worker=localhost:5001/localhost/awe-worker:$(IMAGE_TAG)
	kubectl set image deployment/coder -n awe coder=localhost:5001/localhost/awe-coder:$(IMAGE_TAG)
	kubectl set image deployment/rag-retriever-mcp -n awe rag-retriever-mcp=localhost:5001/localhost/awe-rag-retriever-mcp:$(IMAGE_TAG)
	kubectl set image deployment/ui -n awe ui=localhost:5001/localhost/awe-ui:$(IMAGE_TAG)

k8s-delete-local:
	# Safe local teardown: keep PV/PVC objects and their data.
	kubectl delete deployment -n awe api planner policy worker coder rag-retriever-mcp ui postgres redis qdrant --ignore-not-found
	kubectl delete service -n awe api coder rag-retriever-mcp postgres redis qdrant ui --ignore-not-found
	kubectl delete hpa -n awe api coder --ignore-not-found
	kubectl delete configmap -n awe awe-config --ignore-not-found
	kubectl delete secret -n awe awe-secrets --ignore-not-found

k8s-delete-staging:
	# Safe staging teardown: keep PVC objects and their data.
	kubectl delete deployment -n $(STAGING_NAMESPACE) api planner policy worker coder rag-retriever-mcp ui github-mcp postgres redis qdrant jaeger --ignore-not-found
	kubectl delete service -n $(STAGING_NAMESPACE) api planner policy worker coder rag-retriever-mcp ui github-mcp postgres redis qdrant jaeger --ignore-not-found
	kubectl delete hpa -n $(STAGING_NAMESPACE) api coder --ignore-not-found
	kubectl delete configmap -n $(STAGING_NAMESPACE) awe-config capability-registry mcp-servers-config --ignore-not-found
	kubectl delete secret -n $(STAGING_NAMESPACE) awe-secrets --ignore-not-found

k8s-up-local:
	kubectl config use-context docker-desktop
	(docker ps --format '{{.Names}}' | grep -q '^local-registry$$') || docker run -d -p 5001:5000 --name local-registry registry:2
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) images-build
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) images-push
	$(MAKE) k8s-apply-local
	IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) k8s-pin-local-images
	kubectl rollout status deployment/postgres -n awe --timeout=180s
	kubectl rollout status deployment/redis -n awe --timeout=180s
	kubectl rollout status deployment/qdrant -n awe --timeout=180s
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n awe --timeout=180s
	kubectl rollout status deployment/ui -n awe --timeout=180s

k8s-up-staging:
	$(MAKE) k8s-apply-staging
	$(MAKE) k8s-pin-staging-images
	kubectl rollout status deployment/postgres -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/api -n $(STAGING_NAMESPACE) --timeout=180s
	$(MAKE) k8s-migrate-staging
	kubectl rollout status deployment/redis -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/qdrant -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/api -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/planner -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/policy -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/worker -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/coder -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/ui -n $(STAGING_NAMESPACE) --timeout=180s

k8s-up-production:
	$(MAKE) k8s-apply-production
	$(MAKE) k8s-pin-production-images
	kubectl rollout status deployment/postgres -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/redis -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/qdrant -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/api -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/planner -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/policy -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/worker -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/coder -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/ui -n $(PRODUCTION_NAMESPACE) --timeout=180s

k8s-down-local:
	$(MAKE) k8s-delete-local

clear-local-registry:
	./scripts/clear_local_registry.sh --yes

k8s-restart-local:
	kubectl rollout restart deployment -n awe api planner policy worker coder qdrant rag-retriever-mcp ui
	kubectl rollout status deployment/postgres -n awe --timeout=180s
	kubectl rollout status deployment/redis -n awe --timeout=180s
	kubectl rollout status deployment/qdrant -n awe --timeout=180s
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n awe --timeout=180s
	kubectl rollout status deployment/ui -n awe --timeout=180s

k8s-restart-staging:
	kubectl rollout restart deployment -n $(STAGING_NAMESPACE) api planner policy worker coder qdrant rag-retriever-mcp ui
	kubectl rollout status deployment/postgres -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/redis -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/qdrant -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/api -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/planner -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/policy -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/worker -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/coder -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n $(STAGING_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/ui -n $(STAGING_NAMESPACE) --timeout=180s

k8s-restart-production:
	kubectl rollout restart deployment -n $(PRODUCTION_NAMESPACE) api planner policy worker coder qdrant rag-retriever-mcp ui
	kubectl rollout status deployment/postgres -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/redis -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/qdrant -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/api -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/planner -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/policy -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/worker -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/coder -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n $(PRODUCTION_NAMESPACE) --timeout=180s
	kubectl rollout status deployment/ui -n $(PRODUCTION_NAMESPACE) --timeout=180s

k8s-sync-shared:
	./scripts/k8s_sync_shared.sh all

k8s-sync-workspace:
	./scripts/k8s_sync_shared.sh workspace

k8s-sync-artifacts:
	./scripts/k8s_sync_shared.sh artifacts

k8s-apply-keda-worker:
	kubectl apply -f deploy/k8s/keda-worker-scaledobject.yaml

k8s-delete-keda-worker:
	kubectl delete -f deploy/k8s/keda-worker-scaledobject.yaml --ignore-not-found

down:
	docker compose down

lint:
	PYTHONPATH=. uv run $(UV_QUALITY_DEPS) ruff check libs services

format:
	PYTHONPATH=. uv run $(UV_QUALITY_DEPS) ruff format libs services

test:
	PYTHONPATH=. uv run $(UV_QUALITY_DEPS) pytest --import-mode=importlib

typecheck:
	PYTHONPATH=. uv run $(UV_QUALITY_DEPS) mypy --config-file mypy.ini

schemas:
	PYTHONPATH=. uv run $(UV_QUALITY_DEPS) python -c "from pathlib import Path; from libs.core.schemas import export_schemas; export_schemas(Path('schemas'))"

eval-intent:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_intent_decompose.py --gold eval/intent_gold.yaml --normalization-gold eval/intent_normalization_gold.yaml --mode heuristic --top-k 3 --verbose

eval-intent-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_intent_decompose.py --gold eval/intent_gold.yaml --normalization-gold eval/intent_normalization_gold.yaml --mode heuristic --top-k 3 --min-intent-f1 0.80 --min-capability-f1 0.60 --min-segment-hit-rate 0.30 --min-normalization-intent-accuracy 0.80 --min-normalization-capability-alignment 0.75 --min-missing-input-precision 0.85 --min-disagreement-clarification-rate 0.95

eval-capability-search:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_capability_search.py --gold eval/capability_search_gold.jsonl --top-k 5 --verbose

eval-capability-search-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_capability_search.py --gold eval/capability_search_gold.jsonl --top-k 5 --min-hit-rate-at-3 0.70 --min-mrr 0.55 --min-ndcg 0.60

eval-chat-boundary:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_chat_boundary.py --gold eval/chat_boundary_gold.yaml --verbose

eval-chat-boundary-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_chat_boundary.py --gold eval/chat_boundary_gold.yaml --min-accuracy 0.95 --max-false-chat-reply-rate 0.05 --min-pending-continuation-rate 0.95 --max-active-family-drift-rate 0.05

eval-chat-boundary-live:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_chat_boundary_live.py $(CHAT_BOUNDARY_LIVE_VERBOSE_FLAG)

eval-chat-clarification:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_chat_clarification.py --gold eval/chat_clarification_mapping_gold.yaml --verbose

eval-chat-clarification-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_chat_clarification.py --gold eval/chat_clarification_mapping_gold.yaml --min-overall-accuracy 0.95 --min-restart-decision-accuracy 0.95 --min-resolved-active-field-accuracy 0.95 --min-queue-advance-accuracy 0.95 --max-wrong-field-assignment-rate 0.05

eval-deepeval-chat:
	PYTHONPATH=. uv run $(DEEPEVAL_DEPS) python3 scripts/eval_deepeval_chat.py

eval-deepeval-planner:
	PYTHONPATH=. uv run $(DEEPEVAL_DEPS) python3 scripts/eval_deepeval_planner.py

eval-deepeval-gate:
	PYTHONPATH=. uv run $(DEEPEVAL_DEPS) python3 scripts/eval_deepeval_gate.py

eval-deepeval-staging-replay:
	PYTHONPATH=. uv run $(DEEPEVAL_DEPS) python3 scripts/eval_deepeval_staging_replay.py

verify-agent-registry-staging:
	AGENT_REGISTRY_E2E_BASE_URL="$(AGENT_REGISTRY_E2E_BASE_URL)" \
	AGENT_REGISTRY_E2E_BEARER_TOKEN="$(AGENT_REGISTRY_E2E_BEARER_TOKEN)" \
	AGENT_REGISTRY_E2E_OUTPUT="$(AGENT_REGISTRY_E2E_OUTPUT)" \
		python3 scripts/verify_agent_registry_e2e.py

build-capability-feedback:
	PYTHONPATH=. python3 scripts/build_capability_search_feedback.py --source auto --output artifacts/evals/capability_search_feedback.jsonl

build-chat-routing-feedback:
	PYTHONPATH=. python3 scripts/build_chat_routing_feedback.py --output artifacts/evals/chat_routing_feedback.jsonl --training-output training/chat_routing_reranker_train.jsonl

build-chat-routing-calibrator-from-api:
	PYTHONPATH=. python3 scripts/build_chat_routing_calibrator_from_api.py --feedback-output artifacts/evals/chat_routing_feedback.jsonl --training-output training/chat_routing_reranker_train.jsonl --model-output artifacts/evals/chat_routing_calibrator.json --summary-output artifacts/evals/chat_routing_calibrator_report.json

build-intent-tuning-candidates:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/build_intent_tuning_candidates.py --jsonl-output artifacts/evals/intent_tuning_candidates.jsonl --yaml-output artifacts/evals/intent_tuning_candidates.yaml

build-capability-reranker-dataset:
	PYTHONPATH=. python3 training/build_capability_reranker_dataset.py --feedback artifacts/evals/capability_search_feedback.jsonl --output training/capability_reranker_train.jsonl

build-chat-routing-reranker-dataset:
	PYTHONPATH=. python3 training/build_chat_routing_reranker_dataset.py --feedback artifacts/evals/chat_routing_feedback.jsonl --output training/chat_routing_reranker_train.jsonl

train-chat-routing-calibrator:
	PYTHONPATH=. python3 training/train_chat_routing_calibrator.py --training-data training/chat_routing_reranker_train.jsonl --output artifacts/evals/chat_routing_calibrator.json

eval-chat-routing-calibrator:
	PYTHONPATH=. python3 scripts/eval_chat_routing_calibrator.py --feedback artifacts/evals/chat_routing_feedback.jsonl --model artifacts/evals/chat_routing_calibrator.json --output artifacts/evals/chat_routing_calibrator_replay_report.json

eval-capability-feedback:
	PYTHONPATH=. python3 scripts/eval_capability_search_feedback.py --feedback artifacts/evals/capability_search_feedback.jsonl --output artifacts/evals/capability_search_feedback_report.json
