from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import redis

from libs.core import events, llm_provider, logging as core_logging, models, tool_bootstrap


@dataclass(frozen=True)
class PlannerRuntimeConfig:
    redis_url: str
    metrics_port: int
    consumer_group: str = "planner"
    planner_mode: str = "rule_based"
    llm_provider_name: str = "mock"
    openai_api_key: str = ""
    openai_model: str = ""
    openai_base_url: str = "https://api.openai.com"
    openai_temperature: float | None = None
    openai_max_output_tokens: int | None = None
    openai_timeout_s: float | None = None
    openai_max_retries: int | None = None


@dataclass(frozen=True)
class PlannerExecutionContext:
    provider: llm_provider.LLMProvider | None
    tool_specs: list[models.ToolSpec]


@dataclass(frozen=True)
class PlannerRuntimeCallbacks:
    parse_job_event: Callable[[dict[str, Any]], models.Job | None]
    plan_job: Callable[[models.Job], models.PlanCreate]


def build_redis_client(config: PlannerRuntimeConfig) -> redis.Redis:
    return redis.Redis.from_url(config.redis_url, decode_responses=True)


def resolve_execution_context(config: PlannerRuntimeConfig) -> PlannerExecutionContext:
    provider: llm_provider.LLMProvider | None = None
    if config.planner_mode == "llm":
        provider = llm_provider.resolve_provider(
            config.llm_provider_name,
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url,
            temperature=config.openai_temperature,
            max_output_tokens=config.openai_max_output_tokens,
            timeout_s=config.openai_timeout_s,
            max_retries=config.openai_max_retries,
        )
        registry = tool_bootstrap.build_default_registry(
            http_fetch_enabled=False,
            llm_enabled=True,
            llm_provider=provider,
            service_name="planner",
        )
    else:
        registry = tool_bootstrap.build_default_registry(
            http_fetch_enabled=False,
            service_name="planner",
        )
    return PlannerExecutionContext(
        provider=provider,
        tool_specs=registry.list_specs(),
    )


def emit_plan_event(
    redis_client: redis.Redis,
    event_type: str,
    *,
    job_id: str | None,
    correlation_id: str | None,
    payload: dict[str, Any],
) -> None:
    event = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.now(UTC),
        correlation_id=correlation_id or str(uuid.uuid4()),
        job_id=job_id,
        payload=payload,
    )
    try:
        redis_client.xadd(events.PLAN_STREAM, {"data": event.model_dump_json()})
    except Exception as exc:  # noqa: BLE001
        core_logging.get_logger("planner").warning(
            "plan_event_emit_failed",
            event_type=event_type,
            job_id=job_id,
            error=str(exc),
        )


def emit_planner_capability_selection_event(
    redis_client: redis.Redis,
    job: models.Job,
    plan: models.PlanCreate,
    *,
    correlation_id: str | None = None,
) -> None:
    selected_capabilities: list[str] = []
    for task in plan.tasks:
        for tool_name in task.tool_requests or []:
            normalized = str(tool_name or "").strip()
            if normalized and normalized not in selected_capabilities:
                selected_capabilities.append(normalized)
    emit_plan_event(
        redis_client,
        "plan.capability_selection",
        job_id=job.id,
        correlation_id=correlation_id,
        payload={
            "job_id": job.id,
            "goal": job.goal,
            "planner_version": plan.planner_version,
            "selected_capabilities": selected_capabilities,
            "task_count": len(plan.tasks),
        },
    )


def ensure_consumer_group(
    redis_client: redis.Redis,
    *,
    config: PlannerRuntimeConfig,
) -> None:
    try:
        redis_client.xgroup_create(
            events.JOB_STREAM,
            config.consumer_group,
            id="0-0",
            mkstream=True,
        )
    except redis.ResponseError:
        pass


def process_stream_entry(
    message_id: str,
    data: dict[str, Any],
    *,
    redis_client: redis.Redis,
    config: PlannerRuntimeConfig,
    callbacks: PlannerRuntimeCallbacks,
) -> None:
    payload = json.loads(data["data"])
    job = callbacks.parse_job_event(payload)
    if not job:
        redis_client.xack(events.JOB_STREAM, config.consumer_group, message_id)
        return
    try:
        plan = callbacks.plan_job(job)
        correlation_id = str(payload.get("correlation_id", str(uuid.uuid4())))
        emit_plan_event(
            redis_client,
            "plan.created",
            job_id=job.id,
            correlation_id=correlation_id,
            payload=plan.model_dump(),
        )
        emit_planner_capability_selection_event(
            redis_client,
            job,
            plan,
            correlation_id=correlation_id,
        )
    except Exception as exc:  # noqa: BLE001
        emit_plan_event(
            redis_client,
            "plan.failed",
            job_id=job.id,
            correlation_id=str(payload.get("correlation_id", str(uuid.uuid4()))),
            payload={"job_id": job.id, "error": str(exc)},
        )
    redis_client.xack(events.JOB_STREAM, config.consumer_group, message_id)


def run_loop(
    redis_client: redis.Redis,
    *,
    config: PlannerRuntimeConfig,
    callbacks: PlannerRuntimeCallbacks,
) -> None:
    ensure_consumer_group(redis_client, config=config)
    consumer = str(uuid.uuid4())
    while True:
        messages = redis_client.xreadgroup(
            config.consumer_group,
            consumer,
            {events.JOB_STREAM: ">"},
            count=1,
            block=1000,
        )
        for _, entries in messages:
            for message_id, data in entries:
                process_stream_entry(
                    message_id,
                    data,
                    redis_client=redis_client,
                    config=config,
                    callbacks=callbacks,
                )
