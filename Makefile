.PHONY: up up-workers down lint format typecheck test schemas eval-intent eval-intent-gate \
	eval-capability-search eval-capability-search-gate \
	build-capability-feedback \
	build-capability-reranker-dataset \
	eval-capability-feedback \
	k8s-apply k8s-delete k8s-apply-local k8s-delete-local \
	k8s-apply-observability k8s-delete-observability \
	k8s-apply-keda-worker k8s-delete-keda-worker \
	k8s-up-local k8s-down-local k8s-restart-local \
	k8s-sync-shared k8s-sync-workspace k8s-sync-artifacts \
	images-list images-build images-push \
	k8s-pin-local-images

IMAGE_REGISTRY ?= localhost:5001
IMAGE_OWNER ?= localhost
IMAGE_TAG ?= latest
LOCAL_IMAGE_TAG ?= local-$(shell date +%Y%m%d%H%M%S)
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

k8s-up-local:
	kubectl config use-context docker-desktop
	(docker ps --format '{{.Names}}' | grep -q '^local-registry$$') || docker run -d -p 5001:5000 --name local-registry registry:2
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) images-build
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) images-push
	$(MAKE) k8s-apply-local
	IMAGE_TAG=$(LOCAL_IMAGE_TAG) $(MAKE) k8s-pin-local-images
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/qdrant -n awe --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n awe --timeout=180s
	kubectl rollout status deployment/ui -n awe --timeout=180s

k8s-down-local:
	$(MAKE) k8s-delete-local

k8s-restart-local:
	kubectl rollout restart deployment -n awe api planner policy worker coder qdrant rag-retriever-mcp ui
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/qdrant -n awe --timeout=180s
	kubectl rollout status deployment/rag-retriever-mcp -n awe --timeout=180s
	kubectl rollout status deployment/ui -n awe --timeout=180s

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
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_intent_decompose.py --gold eval/intent_gold.yaml --mode heuristic --top-k 3 --verbose

eval-intent-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_intent_decompose.py --gold eval/intent_gold.yaml --mode heuristic --top-k 3 --min-intent-f1 0.80 --min-capability-f1 0.60 --min-segment-hit-rate 0.30

eval-capability-search:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_capability_search.py --gold eval/capability_search_gold.jsonl --top-k 5 --verbose

eval-capability-search-gate:
	PYTHONPATH=. uv run $(UV_EVAL_DEPS) python3 scripts/eval_capability_search.py --gold eval/capability_search_gold.jsonl --top-k 5 --min-hit-rate-at-3 0.70 --min-mrr 0.55 --min-ndcg 0.60

build-capability-feedback:
	PYTHONPATH=. python3 scripts/build_capability_search_feedback.py --source auto --output artifacts/evals/capability_search_feedback.jsonl

build-capability-reranker-dataset:
	PYTHONPATH=. python3 training/build_capability_reranker_dataset.py --feedback artifacts/evals/capability_search_feedback.jsonl --output training/capability_reranker_train.jsonl

eval-capability-feedback:
	PYTHONPATH=. python3 scripts/eval_capability_search_feedback.py --feedback artifacts/evals/capability_search_feedback.jsonl --output artifacts/evals/capability_search_feedback_report.json
