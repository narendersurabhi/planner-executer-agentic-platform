from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from enum import Enum
from functools import wraps
from time import perf_counter
from typing import List

import redis
from pydantic import BaseModel

from libs.core import events, llm_provider, logging as core_logging, models, tool_registry, prompts

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

DEFAULT_ALLOWED_BLOCK_TYPES = [
    "text",
    "paragraph",
    "heading",
    "bullets",
    "spacer",
    "optional_paragraph",
    "repeat",
]


def _log_entry_exit(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = core_logging.get_logger("planner")
            safe_args = _safe_log_value(args)
            safe_kwargs = _safe_log_value(kwargs)
            logger.info("enter", function=name, args=safe_args, kwargs=safe_kwargs)
            started = perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = int((perf_counter() - started) * 1000)
                safe_result = _safe_log_value(result)
                logger.info(
                    "exit",
                    function=name,
                    status="ok",
                    duration_ms=duration_ms,
                    result=safe_result,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((perf_counter() - started) * 1000)
                logger.error(
                    "exit",
                    function=name,
                    status="error",
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                raise

        return wrapper

    return decorator


def _json_fallback(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_log_value(value: object, max_len: int = 10000) -> object:
    try:
        serialized = json.dumps(value, default=_json_fallback, ensure_ascii=True)
    except Exception:
        try:
            serialized = json.dumps(str(value), ensure_ascii=True)
        except Exception:
            return "<unserializable>"
    if len(serialized) > max_len:
        return f"{serialized[:max_len]}...[truncated]"
    try:
        return json.loads(serialized)
    except Exception:
        return serialized


@_log_entry_exit("rule_based_plan")
def rule_based_plan(job: models.Job, tools: List[models.ToolSpec]) -> models.PlanCreate:
    checklist_task = models.TaskCreate(
        name="create_checklist",
        description="Create implementation checklist",
        instruction="Generate a detailed checklist for adding a new tool.",
        acceptance_criteria=["Checklist has at least 5 items"],
        expected_output_schema_ref="TaskResult",
        intent=models.ToolIntent.generate,
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
        intent=models.ToolIntent.render,
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
        intent=models.ToolIntent.transform,
        deps=["write_tools_doc"],
        tool_requests=["text_summarize"],
        critic_required=False,
    )
    return models.PlanCreate(
        planner_version="rule_based_v1",
        tasks_summary="Checklist, write artifact, summarize",
        dag_edges=[
            ["create_checklist", "write_tools_doc"],
            ["write_tools_doc", "summarize_artifact"],
        ],
        tasks=[checklist_task, write_task, summarize_task],
    )


@_log_entry_exit("llm_plan")
def llm_plan(
    job: models.Job, tools: List[models.ToolSpec], provider: llm_provider.LLMProvider
) -> models.PlanCreate:
    logger = core_logging.get_logger("planner")
    prompt = _llm_prompt(job, tools)
    response = provider.generate(prompt)
    candidate = _parse_llm_plan(response.content)
    if not candidate:
        logger.warning("llm_plan_parse_retry", reason="initial_parse_failed")
        repair_prompt = _llm_plan_repair_prompt(
            original_prompt=prompt,
            raw_output=response.content,
            tools=tools,
        )
        repaired = provider.generate(repair_prompt)
        candidate = _parse_llm_plan(repaired.content)
    if not candidate:
        raise ValueError("Invalid plan generated: parse_failed")
    candidate = _ensure_llm_tool(candidate)
    candidate = _ensure_job_inputs(candidate, job, tools)
    candidate = _apply_max_depth(candidate)
    logger.info("llm_plan_candidate", plan=candidate.model_dump())
    valid, reason = _validate_plan(candidate, tools)
    if not valid:
        logger.warning("llm_plan_invalid", reason=reason)
        raise ValueError(f"Invalid plan generated: {reason}")
    return candidate


@_log_entry_exit("plan_job")
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


@_log_entry_exit("_job_from_event_payload")
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
            "input_schema": tool.input_schema,
            "usage_guidance": tool.usage_guidance,
            "risk_level": tool.risk_level.value
            if isinstance(tool.risk_level, Enum)
            else tool.risk_level,
            "tool_intent": tool.tool_intent.value
            if isinstance(tool.tool_intent, Enum)
            else tool.tool_intent,
        }
        for tool in tools
    ]
    tool_catalog_json = json.dumps(
        tool_catalog, ensure_ascii=False, indent=2, default=_json_fallback
    )
    job_json = json.dumps(
        job.model_dump(mode="json"), ensure_ascii=False, indent=2, default=_json_fallback
    )
    max_depth = _parse_optional_int(PLANNER_MAX_DEPTH)
    depth_hint = ""
    if max_depth:
        depth_hint = f"Max dependency chain depth: {max_depth}.\n"
    return (
        "You are a planner. Return ONLY valid JSON for a PlanCreate object (no prose).\n"
        "Required top-level fields: planner_version, tasks_summary, dag_edges, tasks.\n"
        "Schema rules:\n"
        '- dag_edges must be an array of 2-element string arrays, e.g. [["A","B"],["B","C"]].\n'
        "- acceptance_criteria must be an array of strings, not a single string.\n"
        "Each task must include: name, description, instruction, acceptance_criteria, "
        "expected_output_schema_ref, deps, tool_requests, tool_inputs, critic_required.\n"
        "Example:\n"
        "{\n"
        '  "planner_version": "1.0.0",\n'
        '  "tasks_summary": "...",\n'
        '  "dag_edges": [["TaskA","TaskB"],["TaskB","TaskC"]],\n'
        '  "tasks": [\n'
        "    {\n"
        '      "name": "TaskA",\n'
        '      "description": "...",\n'
        '      "instruction": "...",\n'
        '      "acceptance_criteria": ["..."],\n'
        '      "expected_output_schema_ref": "schemas/example",\n'
        '      "deps": [],\n'
        '      "tool_requests": ["llm_generate"],\n'
        '      "tool_inputs": {"llm_generate": {"text": "..."}},\n'
        '      "critic_required": false\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Rules:\n"
        "1) Use only tool names from the allowed list.\n"
        "2) deps must reference task names that appear in this plan.\n"
        "3) If a tool requires structured JSON, add a prior task to generate that JSON and set "
        "expected_output_schema_ref to that schema.\n"
        "4) If a tool requires specific inputs, put them in task.tool_inputs "
        "(a dict keyed by tool name). Do NOT embed JSON in instruction text.\n"
        "5) Do NOT use placeholder strings like ${Task.output} in tool_inputs. "
        "If a tool needs dependency output (e.g., document_spec), omit that field and rely on deps; "
        "the runtime will pull the object from context. You may still include other inputs like "
        "strict, allowed_block_types, or path.\n"
        "6) Prefer the generic validation + rendering pipeline:\n"
        "   - Generate a DocumentSpec JSON.\n"
        "   - Validate with document_spec_validate.\n"
        "   - Render with docx_generate_from_spec.\n"
        "   Use specialized tools only if explicitly requested.\n"
        "7) If unsure, use llm_generate.\n"
        "8) Keep output compact. Do NOT copy or embed large raw text from Job JSON "
        "(especially job_description or candidate_resume) into tasks, instructions, "
        "acceptance criteria, or tool_inputs.\n"
        "9) For tool_inputs include only minimal scalar params. Omit large/context fields "
        "(e.g., job, memory, tailored_resume, tailored_text, resume_doc_spec, document_spec) "
        "and rely on runtime dependency/context injection.\n"
        "10) Keep each task instruction concise (one short paragraph) and keep acceptance "
        "criteria short bullets.\n"
        "\n"
        f"{depth_hint}"
        f"Allowed tool names: {tool_names}\n"
        f"Tool catalog (JSON): {tool_catalog_json}\n"
        f"Goal: {job.goal}\n"
        f"Job (JSON): {job_json}\n"
    )


def _llm_plan_repair_prompt(
    *, original_prompt: str, raw_output: str, tools: List[models.ToolSpec]
) -> str:
    tool_names = ", ".join(sorted(tool.name for tool in tools))
    return (
        "You are fixing a malformed planner response.\n"
        "Return ONLY one valid JSON object for PlanCreate.\n"
        "Do not include markdown, comments, or prose.\n"
        "Required top-level fields: planner_version, tasks_summary, dag_edges, tasks.\n"
        "Each task must include: name, description, instruction, acceptance_criteria, "
        "expected_output_schema_ref, deps, tool_requests, tool_inputs, critic_required.\n"
        "Rules:\n"
        "- acceptance_criteria must be string array.\n"
        "- dag_edges must be array of 2-string arrays.\n"
        "- Use only allowed tool names.\n"
        "- If a field is missing, add a safe default value.\n"
        f"Allowed tool names: {tool_names}\n\n"
        f"Original planner prompt (for context):\n{original_prompt}\n\n"
        f"Malformed planner output to repair:\n{raw_output}\n"
    )


def _document_spec_prompt(job: models.Job, allowed_block_types: List[str]) -> str:
    return prompts.document_spec_prompt(job.model_dump(mode="json"), allowed_block_types)


@_log_entry_exit("_parse_llm_plan")
def _parse_llm_plan(content: str) -> models.PlanCreate | None:
    candidates = [_extract_json(content), content.strip()]
    for candidate in candidates:
        data = _decode_json_object(candidate)
        if not isinstance(data, dict):
            continue
        try:
            return models.PlanCreate.model_validate(data)
        except Exception:
            continue
    return None


@_log_entry_exit("_extract_json")
def _extract_json(content: str) -> str:
    text = _strip_markdown_fence(content.strip())
    text = _unwrap_json_string(text)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    extracted = text[start : end + 1]
    extracted = _unwrap_json_string(extracted)
    return extracted


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    if len(parts) <= 1:
        return text
    candidate = parts[1].lstrip()
    if candidate.startswith("json"):
        candidate = candidate[4:].lstrip()
    return candidate


def _unwrap_json_string(text: str) -> str:
    candidate = text.strip()
    for _ in range(2):
        if not (candidate.startswith('"') and candidate.endswith('"')):
            break
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if not isinstance(decoded, str):
            break
        candidate = decoded.strip()
    return candidate


def _decode_json_object(text: str) -> dict | None:
    queue = [text]
    seen: set[str] = set()
    while queue:
        candidate = queue.pop(0).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            unwrapped = _unwrap_json_string(candidate)
            if unwrapped and unwrapped != candidate:
                queue.append(unwrapped)
            if '\\"' in candidate:
                queue.append(candidate.replace('\\"', '"'))
            extracted = _extract_json(candidate)
            if extracted and extracted != candidate:
                queue.append(extracted)
            continue
        if isinstance(parsed, dict):
            inner = parsed.get("plan")
            if isinstance(inner, dict):
                return inner
            return parsed
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
        if isinstance(parsed, str):
            queue.append(parsed)
    return None


def _ensure_llm_tool(plan: models.PlanCreate) -> models.PlanCreate:
    updated_tasks = []
    for task in plan.tasks:
        if not task.tool_requests:
            updated_tasks.append(task.model_copy(update={"tool_requests": ["llm_generate"]}))
        else:
            updated_tasks.append(task)
    return plan.model_copy(update={"tasks": updated_tasks})


def _ensure_job_inputs(
    plan: models.PlanCreate, job: models.Job, tools: List[models.ToolSpec]
) -> models.PlanCreate:
    tool_map = {tool.name: tool for tool in tools}
    job_payload = job.model_dump(mode="json")
    updated_tasks = []
    for task in plan.tasks:
        tool_inputs = dict(task.tool_inputs) if isinstance(task.tool_inputs, dict) else {}
        changed = False
        for tool_name in task.tool_requests or []:
            tool = tool_map.get(tool_name)
            if tool is None:
                continue
            required_sets = _required_sets_from_schema(tool.input_schema or {})
            if not required_sets:
                continue
            if not any("job" in required for required in required_sets):
                continue
            payload = tool_inputs.get(tool_name)
            if isinstance(payload, dict) and "job" in payload:
                continue
            payload = dict(payload) if isinstance(payload, dict) else {}
            payload["job"] = job_payload
            tool_inputs[tool_name] = payload
            changed = True
        if changed:
            updated_tasks.append(task.model_copy(update={"tool_inputs": tool_inputs}))
        else:
            updated_tasks.append(task)
    return plan.model_copy(update={"tasks": updated_tasks})


def _validate_plan(plan: models.PlanCreate, tools: List[models.ToolSpec]) -> tuple[bool, str]:
    tool_map = {tool.name: tool for tool in tools}
    for task in plan.tasks:
        if not task.tool_requests:
            continue
        for tool_name in task.tool_requests:
            tool = tool_map.get(tool_name)
            if tool is None:
                return False, f"unknown_tool:{tool_name}"
            required_sets = _required_sets_from_schema(tool.input_schema or {})
            if not required_sets:
                continue
            tool_inputs = _extract_tool_inputs(task, tool_name)
            if _inputs_satisfy_required(required_sets, tool_inputs, bool(task.deps)):
                continue
            return False, f"tool_inputs_missing:{tool_name}:{task.name}"
    return True, "ok"


def _inputs_satisfy_required(
    required_sets: list[set[str]], inputs: set[str], has_deps: bool
) -> bool:
    context_fillable = {
        "content",
        "text",
        "prompt",
        "data",
        "document_spec",
        "validation_report",
        "errors",
        "original_spec",
        "tailored_resume",
        "resume_content",
        "tailored_text",
        "resume_doc_spec",
    }
    for required in required_sets:
        missing = required - inputs
        if not missing:
            return True
        if has_deps and missing.issubset(context_fillable):
            return True
    return False


def _required_sets_from_schema(schema: dict) -> list[set[str]]:
    if not isinstance(schema, dict):
        return []
    required_sets: list[set[str]] = []
    if isinstance(schema.get("required"), list):
        required_sets.append(set(schema["required"]))
    for keyword in ("anyOf", "oneOf"):
        if isinstance(schema.get(keyword), list):
            for entry in schema[keyword]:
                if isinstance(entry, dict) and isinstance(entry.get("required"), list):
                    required_sets.append(set(entry["required"]))
    if not required_sets:
        return []
    return required_sets


def _extract_tool_inputs(task: models.TaskCreate, tool_name: str) -> set[str]:
    if isinstance(task.tool_inputs, dict) and tool_name in task.tool_inputs:
        payload = task.tool_inputs.get(tool_name)
        if isinstance(payload, dict):
            return {key for key in payload.keys() if isinstance(key, str)}
    return set()


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
        messages = redis_client.xreadgroup(
            group, consumer, {events.JOB_STREAM: ">"}, count=1, block=1000
        )
        for _, entries in messages:
            for message_id, data in entries:
                payload = json.loads(data["data"])
                job = _job_from_event_payload(payload)
                if not job:
                    redis_client.xack(events.JOB_STREAM, group, message_id)
                    continue
                try:
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
                except Exception as exc:  # noqa: BLE001
                    fail_event = models.EventEnvelope(
                        type="plan.failed",
                        version="1",
                        occurred_at=datetime.utcnow(),
                        correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                        job_id=job.id,
                        payload={"job_id": job.id, "error": str(exc)},
                    )
                    redis_client.xadd(events.PLAN_STREAM, {"data": fail_event.model_dump_json()})
                redis_client.xack(events.JOB_STREAM, group, message_id)


if __name__ == "__main__":
    run()
