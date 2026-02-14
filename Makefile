.PHONY: up down lint format typecheck test schemas

up:
	docker compose up -d --build

down:
	docker compose down

lint:
	ruff check libs services

format:
	ruff format libs services

test:
	pytest

typecheck:
	mypy --config-file mypy.ini

schemas:
	python -c "from pathlib import Path; from libs.core.schemas import export_schemas; export_schemas(Path('schemas'))"
