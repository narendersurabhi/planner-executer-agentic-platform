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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
OPENAI_MAX_OUTPUT_TOKENS = os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
PLANNER_MAX_DEPTH = os.getenv("PLANNER_MAX_DEPTH")
OPENAI_TIMEOUT_S = os.getenv("OPENAI_TIMEOUT_S")
OPENAI_MAX_RETRIES = os.getenv("OPENAI_MAX_RETRIES")

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


def llm_plan(
    job: models.Job, tools: List[models.ToolSpec], provider: llm_provider.LLMProvider
) -> models.PlanCreate:
    prompt = _llm_prompt(job, tools)
    response = provider.generate(prompt)
    parsed = _parse_llm_plan(response.content)
    if parsed:
        return _ensure_llm_tool(_apply_max_depth(parsed))
    fallback = rule_based_plan(job, tools).model_copy(
        update={"planner_version": "llm_fallback_rule_based_v1"}
    )
    return _ensure_llm_tool(fallback)


def plan_job(job: models.Job) -> models.PlanCreate:
    if PLANNER_MODE == "llm":
        provider = llm_provider.resolve_provider(
            LLM_PROVIDER,
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            base_url=OPENAI_BASE_URL,
            temperature=_parse_optional_float(OPENAI_TEMPERATURE),
            max_output_tokens=_parse_optional_int(OPENAI_MAX_OUTPUT_TOKENS),
            timeout_s=_parse_optional_float(OPENAI_TIMEOUT_S),
            max_retries=_parse_optional_int(OPENAI_MAX_RETRIES),
        )
        registry = tool_registry.default_registry(False, llm_enabled=True, llm_provider=provider)
        return llm_plan(job, registry.list_specs(), provider)
    registry = tool_registry.default_registry(False)
    return rule_based_plan(job, registry.list_specs())


def _job_from_event_payload(event: dict) -> models.Job | None:
    data = event.get("payload")
    if not isinstance(data, dict):
        return None
    try:
        return models.Job(**data)
    except Exception:
        job_id = data.get("id") or data.get("job_id")
        goal = data.get("goal")
        if not job_id or not goal:
            return None
        now = datetime.utcnow()
        return models.Job(
            id=job_id,
            goal=goal,
            context_json=data.get("context_json", {}),
            status=_parse_job_status(data.get("status")),
            created_at=_parse_datetime(data.get("created_at"), now),
            updated_at=_parse_datetime(data.get("updated_at"), now),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )


def _parse_job_status(value: object) -> models.JobStatus:
    if isinstance(value, models.JobStatus):
        return value
    if isinstance(value, str):
        try:
            return models.JobStatus(value)
        except ValueError:
            return models.JobStatus.queued
    return models.JobStatus.queued


def _parse_datetime(value: object, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def _llm_prompt(job: models.Job, tools: List[models.ToolSpec]) -> str:
    tool_names = ", ".join(sorted(tool.name for tool in tools))
    tool_catalog = [
        {
            "name": tool.name,
            "description": tool.description,
            "usage_guidance": tool.usage_guidance,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "risk_level": tool.risk_level,
        }
        for tool in tools
    ]
    tool_catalog_json = json.dumps(tool_catalog, ensure_ascii=True, indent=2)
    max_depth = _parse_optional_int(PLANNER_MAX_DEPTH)
    depth_hint = ""
    if max_depth:
        depth_hint = f"Max dependency chain depth: {max_depth}.\n"
    return (
        "You are a planner. Return ONLY valid JSON for a PlanCreate object.\n"
        "Fields: planner_version (string), tasks_summary (string), dag_edges (list of [from,to]), "
        "tasks (list of tasks).\n"
        "Each task must include: name, description, instruction, acceptance_criteria (list of strings), "
        "expected_output_schema_ref (string), deps (list of strings), tool_requests (list of tool names), "
        "critic_required (bool).\n"
        "Use tool usage guidance and input schemas. Only select tools whose required inputs can be satisfied "
        "by the task instruction and dependency context. If unsure, use llm_generate.\n"
        f"{depth_hint}"
        f"Allowed tool names: {tool_names}\n"
        f"Tool catalog (JSON): {tool_catalog_json}\n"
        f"Goal: {job.goal}\n"
        f"Context JSON: {job.context_json}\n"
    )


def _parse_llm_plan(content: str) -> models.PlanCreate | None:
    cleaned = _extract_json(content)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    try:
        return models.PlanCreate.model_validate(data)
    except Exception:
        return None


def _extract_json(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        text = text.lstrip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _ensure_llm_tool(plan: models.PlanCreate) -> models.PlanCreate:
    updated_tasks = []
    for task in plan.tasks:
        if not task.tool_requests:
            updated_tasks.append(task.model_copy(update={"tool_requests": ["llm_generate"]}))
        else:
            updated_tasks.append(task)
    return plan.model_copy(update={"tasks": updated_tasks})


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _apply_max_depth(plan: models.PlanCreate) -> models.PlanCreate:
    max_depth = _parse_optional_int(PLANNER_MAX_DEPTH)
    if not max_depth or max_depth <= 0:
        return plan
    tasks = [task.model_copy(deep=True) for task in plan.tasks]
    for _ in range(len(tasks) + 1):
        depths = _task_depths(tasks, max_depth)
        current_max = max(depths.values(), default=0)
        if current_max <= max_depth:
            break
        changed = False
        for task in tasks:
            if not task.deps:
                continue
            allowed = [dep for dep in task.deps if depths.get(dep, 1) < max_depth]
            if allowed != task.deps:
                task.deps = allowed
                changed = True
        if not changed:
            for task in tasks:
                if depths.get(task.name, 1) > max_depth and task.deps:
                    task.deps = []
                    changed = True
            if not changed:
                break
    return plan.model_copy(update={"tasks": tasks})


def _task_depths(tasks: List[models.TaskCreate], limit: int) -> dict[str, int]:
    by_name = {task.name: task for task in tasks}
    depth_cache: dict[str, int] = {}
    visiting: set[str] = set()

    def depth_for(name: str) -> int:
        if name in depth_cache:
            return depth_cache[name]
        if name in visiting:
            return limit + 1
        visiting.add(name)
        task = by_name.get(name)
        if not task:
            visiting.remove(name)
            return 1
        max_depth = 1
        for dep in task.deps:
            max_depth = max(max_depth, depth_for(dep) + 1)
            if max_depth > limit:
                break
        visiting.remove(name)
        depth_cache[name] = max_depth
        return max_depth

    for task in tasks:
        depth_for(task.name)
    return depth_cache


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
                job = _job_from_event_payload(payload)
                if not job:
                    redis_client.xack(events.JOB_STREAM, group, message_id)
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
