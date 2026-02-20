.PHONY: up up-workers down lint format typecheck test schemas \
	k8s-apply k8s-delete k8s-apply-local k8s-delete-local \
	k8s-apply-observability k8s-delete-observability \
	k8s-apply-keda-worker k8s-delete-keda-worker \
	k8s-up-local k8s-down-local k8s-restart-local \
	k8s-sync-shared k8s-sync-workspace k8s-sync-artifacts \
	images-list images-build images-push

IMAGE_REGISTRY ?= localhost:5001
IMAGE_OWNER ?= localhost
IMAGE_TAG ?= latest

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
	kubectl kustomize --load-restrictor LoadRestrictionsNone deploy/k8s/overlays/local | kubectl apply -f -
	./scripts/setup_k8s_env.sh
	kubectl rollout restart deployment -n awe api planner policy worker coder tailor ui

k8s-delete-local:
	# Safe local teardown: keep PV/PVC objects and their data.
	kubectl delete deployment -n awe api planner policy worker coder tailor ui postgres redis --ignore-not-found
	kubectl delete service -n awe api coder postgres redis tailor ui --ignore-not-found
	kubectl delete hpa -n awe api coder tailor --ignore-not-found
	kubectl delete configmap -n awe awe-config --ignore-not-found
	kubectl delete secret -n awe awe-secrets --ignore-not-found

k8s-up-local:
	kubectl config use-context docker-desktop
	(docker ps --format '{{.Names}}' | grep -q '^local-registry$$') || docker run -d -p 5001:5000 --name local-registry registry:2
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=latest $(MAKE) images-build
	IMAGE_REGISTRY=localhost:5001 IMAGE_OWNER=localhost IMAGE_TAG=latest $(MAKE) images-push
	$(MAKE) k8s-apply-local
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/tailor -n awe --timeout=180s
	kubectl rollout status deployment/ui -n awe --timeout=180s

k8s-down-local:
	$(MAKE) k8s-delete-local

k8s-restart-local:
	kubectl rollout restart deployment -n awe api planner policy worker coder tailor ui
	kubectl rollout status deployment/api -n awe --timeout=180s
	kubectl rollout status deployment/planner -n awe --timeout=180s
	kubectl rollout status deployment/policy -n awe --timeout=180s
	kubectl rollout status deployment/worker -n awe --timeout=180s
	kubectl rollout status deployment/coder -n awe --timeout=180s
	kubectl rollout status deployment/tailor -n awe --timeout=180s
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
	ruff check libs services

format:
	ruff format libs services

test:
	PYTHONPATH=. pytest

typecheck:
	mypy --config-file mypy.ini

schemas:
	python -c "from pathlib import Path; from libs.core.schemas import export_schemas; export_schemas(Path('schemas'))"
