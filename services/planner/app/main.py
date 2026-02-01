from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import List

import redis

from libs.core import events, llm_provider, logging as core_logging, models, tool_registry

core_logging.configure_logging("planner")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PLANNER_MODE = os.getenv("PLANNER_MODE", "rule_based")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def rule_based_plan(job: models.Job, tools: List[models.ToolSpec]) -> models.PlanCreate:
    checklist_task = models.TaskCreate(
        name="create_checklist",
        description="Create implementation checklist",
        instruction="Generate a detailed checklist for adding a new tool.",
        acceptance_criteria=["Checklist has at least 5 items"],
        expected_output_schema_ref="TaskResult",
        deps=[],
        tool_requests=[],
        critic_required=True,
    )
    write_task = models.TaskCreate(
        name="write_tools_doc",
        description="Write tools.md artifact",
        instruction="Write a tools.md draft file as an artifact.",
        acceptance_criteria=["Artifact path returned"],
        expected_output_schema_ref="TaskResult",
        deps=["create_checklist"],
        tool_requests=["file_write_artifact"],
        critic_required=True,
    )
    summarize_task = models.TaskCreate(
        name="summarize_artifact",
        description="Summarize the artifact",
        instruction="Summarize the tools.md artifact.",
        acceptance_criteria=["Summary provided"],
        expected_output_schema_ref="TaskResult",
        deps=["write_tools_doc"],
        tool_requests=["text_summarize"],
        critic_required=False,
    )
    return models.PlanCreate(
        planner_version="rule_based_v1",
        tasks_summary="Checklist, write artifact, summarize",
        dag_edges=[["create_checklist", "write_tools_doc"], ["write_tools_doc", "summarize_artifact"]],
        tasks=[checklist_task, write_task, summarize_task],
    )


def llm_plan(job: models.Job, tools: List[models.ToolSpec]) -> models.PlanCreate:
    provider = llm_provider.MockLLMProvider()
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        provider = llm_provider.OpenAIProvider(OPENAI_API_KEY)
    _ = provider.generate(f"Plan tasks for goal: {job.goal}")
    return rule_based_plan(job, tools)


def plan_job(job: models.Job) -> models.PlanCreate:
    registry = tool_registry.default_registry(False)
    tool_specs = registry.list_specs()
    if PLANNER_MODE == "llm":
        return llm_plan(job, tool_specs)
    return rule_based_plan(job, tool_specs)


def run() -> None:
    group = "planner"
    consumer = str(uuid.uuid4())
    try:
        redis_client.xgroup_create(events.JOB_STREAM, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass
    while True:
        messages = redis_client.xreadgroup(group, consumer, {events.JOB_STREAM: ">"}, count=1, block=1000)
        for _, entries in messages:
            for message_id, data in entries:
                payload = json.loads(data["data"])
                job = models.Job(**payload["payload"]) if "payload" in payload else None
                if not job:
                    continue
                plan = plan_job(job)
                event = models.EventEnvelope(
                    type="plan.created",
                    version="1",
                    occurred_at=datetime.utcnow(),
                    correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                    job_id=job.id,
                    payload=plan.model_dump(),
                )
                redis_client.xadd(events.PLAN_STREAM, {"data": event.model_dump_json()})
                redis_client.xack(events.JOB_STREAM, group, message_id)


if __name__ == "__main__":
    run()
