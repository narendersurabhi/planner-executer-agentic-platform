from __future__ import annotations

import json
from datetime import UTC, datetime

from libs.core import events, llm_provider, models
from services.planner.app import runtime_service


class _FakeRegistry:
    def __init__(self, specs: list[models.ToolSpec]) -> None:
        self._specs = specs

    def list_specs(self) -> list[models.ToolSpec]:
        return list(self._specs)


class _FakeRedis:
    def __init__(self) -> None:
        self.added: list[tuple[str, dict[str, str]]] = []
        self.acked: list[tuple[str, str, str]] = []

    def xadd(self, stream: str, payload: dict[str, str]) -> None:
        self.added.append((stream, payload))

    def xack(self, stream: str, group: str, message_id: str) -> None:
        self.acked.append((stream, group, message_id))


def _job() -> models.Job:
    return models.Job(
        id="job-1",
        goal="Create a plan",
        context_json={},
        status=models.JobStatus.queued,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        priority=0,
        metadata={},
    )


def _plan() -> models.PlanCreate:
    return models.PlanCreate(
        planner_version="1.0.0",
        tasks_summary="demo",
        dag_edges=[],
        tasks=[
            models.TaskCreate(
                name="TaskA",
                description="desc",
                instruction="instr",
                acceptance_criteria=["ok"],
                expected_output_schema_ref="schemas/test",
                intent=models.ToolIntent.generate,
                deps=[],
                tool_requests=["llm_generate"],
                tool_inputs={},
                critic_required=False,
            )
        ],
    )


def test_resolve_execution_context_uses_llm_provider_when_enabled(monkeypatch) -> None:
    provider = llm_provider.MockLLMProvider()
    tool = models.ToolSpec(
        name="llm_generate",
        description="generate",
        input_schema={},
        output_schema={},
    )

    monkeypatch.setattr(runtime_service.llm_provider, "resolve_provider", lambda *args, **kwargs: provider)
    monkeypatch.setattr(
        runtime_service.tool_bootstrap,
        "build_default_registry",
        lambda **kwargs: _FakeRegistry([tool]),
    )

    context = runtime_service.resolve_execution_context(
        runtime_service.PlannerRuntimeConfig(
            redis_url="redis://redis:6379/0",
            metrics_port=9101,
            planner_mode="llm",
            llm_provider_name="mock",
        )
    )

    assert context.provider is provider
    assert context.tool_specs[0].name == "llm_generate"


def test_process_stream_entry_emits_created_and_selection_events() -> None:
    fake_redis = _FakeRedis()
    config = runtime_service.PlannerRuntimeConfig(
        redis_url="redis://redis:6379/0",
        metrics_port=9101,
    )
    callbacks = runtime_service.PlannerRuntimeCallbacks(
        parse_job_event=lambda payload: _job(),
        plan_job=lambda job: _plan(),
    )

    runtime_service.process_stream_entry(
        "1-0",
        {"data": json.dumps({"payload": {"id": "job-1", "goal": "Create a plan"}})},
        redis_client=fake_redis,
        config=config,
        callbacks=callbacks,
    )

    assert fake_redis.acked == [(events.JOB_STREAM, "planner", "1-0")]
    assert len(fake_redis.added) == 2
    created = json.loads(fake_redis.added[0][1]["data"])
    selected = json.loads(fake_redis.added[1][1]["data"])
    assert fake_redis.added[0][0] == events.PLAN_STREAM
    assert created["type"] == "plan.created"
    assert selected["type"] == "plan.capability_selection"
