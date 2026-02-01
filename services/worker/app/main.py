from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import redis
from jsonschema import Draft202012Validator

from libs.core import events, llm_provider, logging as core_logging, models, tool_registry

core_logging.configure_logging("worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TOOL_HTTP_FETCH_ENABLED = os.getenv("TOOL_HTTP_FETCH_ENABLED", "false").lower() == "true"
WORKER_MODE = os.getenv("WORKER_MODE", "rule_based")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
OPENAI_MAX_OUTPUT_TOKENS = os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
SCHEMA_REGISTRY_PATH = os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas")
SCHEMA_VALIDATION_STRICT = os.getenv("SCHEMA_VALIDATION_STRICT", "false").lower() == "true"
OPENAI_TIMEOUT_S = os.getenv("OPENAI_TIMEOUT_S")
OPENAI_MAX_RETRIES = os.getenv("OPENAI_MAX_RETRIES")
TOOL_OUTPUT_SIZE_CAP = os.getenv("TOOL_OUTPUT_SIZE_CAP", "50000")

LLM_ENABLED = WORKER_MODE == "llm"

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


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


OUTPUT_SIZE_CAP = _parse_optional_int(TOOL_OUTPUT_SIZE_CAP) or 50000


LLM_PROVIDER_INSTANCE = None
if LLM_ENABLED:
    LLM_PROVIDER_INSTANCE = llm_provider.resolve_provider(
        LLM_PROVIDER,
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=_parse_optional_float(OPENAI_TEMPERATURE),
        max_output_tokens=_parse_optional_int(OPENAI_MAX_OUTPUT_TOKENS),
        timeout_s=_parse_optional_float(OPENAI_TIMEOUT_S),
        max_retries=_parse_optional_int(OPENAI_MAX_RETRIES),
    )


def execute_task(task_payload: dict) -> models.TaskResult:
    task_id = task_payload.get("task_id")
    tool_requests = task_payload.get("tool_requests", [])
    context = task_payload.get("context", {})
    registry = tool_registry.default_registry(
        TOOL_HTTP_FETCH_ENABLED,
        llm_enabled=LLM_ENABLED,
        llm_provider=LLM_PROVIDER_INSTANCE,
    )
    tool_calls = []
    started_at = datetime.utcnow()
    outputs = {}
    tool_error = None
    for tool_name in tool_requests:
        payload = _tool_payload(tool_name, task_payload.get("instruction", ""), context, task_payload)
        call = registry.execute(
            tool_name,
            payload=payload,
            idempotency_key=str(uuid.uuid4()),
            trace_id=task_payload.get("correlation_id", ""),
            max_output_bytes=OUTPUT_SIZE_CAP,
        )
        tool_calls.append(call)
        outputs[tool_name] = call.output_or_error
        if call.status != "completed":
            tool_error = call.output_or_error.get("error", "tool_failed")
            break
    finished_at = datetime.utcnow()
    validation_error = _validate_expected_output(task_payload, outputs)
    status = (
        models.TaskStatus.failed
        if tool_error or validation_error
        else models.TaskStatus.completed
    )
    if tool_error:
        outputs["tool_error"] = {"error": tool_error}
    if validation_error:
        outputs["validation_error"] = {"error": validation_error}
    return models.TaskResult(
        task_id=task_id,
        status=status,
        outputs=outputs,
        artifacts=[],
        tool_calls=tool_calls,
        started_at=started_at,
        finished_at=finished_at,
        error=validation_error or tool_error,
    )


def _tool_payload(tool_name: str, instruction: str, context: dict, task_payload: dict) -> dict:
    if tool_name in {"file_write_artifact", "file_write_text"}:
        content = None
        if isinstance(task_payload.get("content"), str):
            content = task_payload.get("content")
        if content is None:
            content = _extract_text_from_context(context)
        payload = {"content": content or ""}
        if isinstance(task_payload.get("path"), str):
            payload["path"] = task_payload.get("path")
        return payload
    if tool_name == "file_read_text":
        if isinstance(task_payload.get("path"), str):
            return {"path": task_payload.get("path")}
        return {"path": ""}
    if not context:
        return {"text": instruction}
    try:
        context_blob = json.dumps(context, indent=2, ensure_ascii=True)
    except (TypeError, ValueError):
        context_blob = str(context)
    return {"text": f"{instruction}\n\nContext (JSON):\n{context_blob}"}


def _extract_text_from_context(context: dict) -> str | None:
    if not isinstance(context, dict):
        return None
    for key in ("dependencies_by_name", "dependencies"):
        group = context.get(key)
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            text = _extract_text_from_outputs(output)
            if text:
                return text
    return None


def _extract_text_from_outputs(outputs: dict) -> str | None:
    if not isinstance(outputs, dict):
        return None
    for tool_key in ("llm_generate", "text_summarize", "json_transform"):
        entry = outputs.get(tool_key)
        if not isinstance(entry, dict):
            continue
        if tool_key == "llm_generate":
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
        if tool_key == "text_summarize":
            summary = entry.get("summary")
            if isinstance(summary, str) and summary:
                return summary
        if tool_key == "json_transform":
            result = entry.get("result")
            if isinstance(result, dict):
                text = result.get("text")
                if isinstance(text, str) and text:
                    return text
            elif isinstance(result, str) and result:
                return result
    for entry in outputs.values():
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
    return None


def _validate_expected_output(task_payload: dict, outputs: dict) -> str | None:
    schema_ref = task_payload.get("expected_output_schema_ref")
    tool_requests = task_payload.get("tool_requests", [])
    if "llm_generate" not in tool_requests:
        return None
    if not schema_ref:
        return None
    schema_path = _resolve_schema_path(schema_ref)
    if not schema_path:
        return None
    if not schema_path.exists():
        if SCHEMA_VALIDATION_STRICT:
            return f"schema_not_found:{schema_path}"
        return None
    text = _extract_llm_text(outputs)
    if not text:
        return "missing_llm_output"
    json_text = _extract_json(text)
    if not json_text:
        return "missing_json_payload"
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return f"invalid_json:{exc}"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"invalid_schema:{exc}"
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if errors:
        messages = "; ".join(
            f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors[:5]
        )
        return f"schema_validation_failed:{messages}"
    return None


def _resolve_schema_path(schema_ref: str) -> Path | None:
    if not schema_ref:
        return None
    name = schema_ref
    if schema_ref.startswith("schema/"):
        name = schema_ref.split("/", 1)[1]
    if not name:
        return None
    if not name.endswith(".json"):
        name = f"{name}.json"
    return Path(SCHEMA_REGISTRY_PATH) / name


def _extract_llm_text(outputs: dict) -> str:
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_json(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1]
        content = content.lstrip()
        if content.startswith("json"):
            content = content[4:].lstrip()
    first_obj = content.find("{")
    first_arr = content.find("[")
    if first_obj == -1 and first_arr == -1:
        return ""
    if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
        start = first_obj
        end = content.rfind("}")
    else:
        start = first_arr
        end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return ""
    return content[start : end + 1]


def run() -> None:
    group = "workers"
    consumer = str(uuid.uuid4())
    try:
        redis_client.xgroup_create(events.TASK_STREAM, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass
    while True:
        messages = redis_client.xreadgroup(group, consumer, {events.TASK_STREAM: ">"}, count=1, block=1000)
        for _, entries in messages:
            for message_id, data in entries:
                payload = json.loads(data["data"])
                if payload.get("type") != "task.ready":
                    redis_client.xack(events.TASK_STREAM, group, message_id)
                    continue
                task_payload = payload.get("payload", {})
                result = execute_task(task_payload)
                event_type = "task.failed" if result.status == models.TaskStatus.failed else "task.completed"
                event = models.EventEnvelope(
                    type=event_type,
                    version="1",
                    occurred_at=datetime.utcnow(),
                    correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                    job_id=payload.get("job_id"),
                    task_id=task_payload.get("task_id"),
                    payload=result.model_dump(),
                )
                redis_client.xadd(events.TASK_STREAM, {"data": event.model_dump_json()})
                redis_client.xack(events.TASK_STREAM, group, message_id)
                time.sleep(0.1)


if __name__ == "__main__":
    run()
