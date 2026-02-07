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
LOGGER = core_logging.get_logger("worker")

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

DEFAULT_ALLOWED_BLOCK_TYPES = [
    "text",
    "paragraph",
    "heading",
    "bullets",
    "spacer",
    "optional_paragraph",
    "repeat",
]

TASK_INTENTS = {
    "transform",
    "generate",
    "validate",
    "render",
    "io",
}


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
    tool_inputs = task_payload.get("tool_inputs", {})
    registry = tool_registry.default_registry(
        TOOL_HTTP_FETCH_ENABLED,
        llm_enabled=LLM_ENABLED,
        llm_provider=LLM_PROVIDER_INSTANCE,
    )
    tool_calls = []
    started_at = datetime.utcnow()
    outputs = {}
    tool_error = None
    task_intent = _infer_task_intent(task_payload)
    for tool_name in tool_requests:
        try:
            tool = registry.get(tool_name)
        except KeyError:
            tool_error = f"unknown_tool:{tool_name}"
            outputs[tool_name] = {"error": tool_error}
            break
        mismatch = _intent_mismatch(task_intent, tool.spec.tool_intent, tool_name)
        if mismatch:
            tool_error = mismatch
            outputs[tool_name] = {"error": tool_error}
            tool_calls.append(
                models.ToolCall(
                    tool_name=tool_name,
                    input={},
                    idempotency_key=str(uuid.uuid4()),
                    trace_id=task_payload.get("correlation_id", ""),
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                    status="failed",
                    output_or_error={"error": tool_error},
                )
            )
            break
        payload = _tool_payload(
            tool_name,
            task_payload.get("instruction", ""),
            context,
            task_payload,
            tool_inputs,
        )
        payload["_registry"] = registry
        idempotency_key = str(uuid.uuid4())
        trace_id = task_payload.get("correlation_id", "")
        core_logging.log_event(
            LOGGER,
            "tool_call_started",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "trace_id": trace_id,
                "idempotency_key": idempotency_key,
                "payload_keys": sorted(payload.keys()),
            },
        )
        call = registry.execute(
            tool_name,
            payload=payload,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            max_output_bytes=OUTPUT_SIZE_CAP,
        )
        core_logging.log_event(
            LOGGER,
            "tool_call_finished",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "trace_id": trace_id,
                "idempotency_key": idempotency_key,
                "status": call.status,
            },
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


def _infer_task_intent(task_payload: dict) -> str:
    for key in ("intent", "task_intent"):
        value = task_payload.get(key)
        if isinstance(value, str) and value in TASK_INTENTS:
            return value
    parts: list[str] = []
    for key in ("description", "instruction"):
        value = task_payload.get(key)
        if isinstance(value, str):
            parts.append(value)
    criteria = task_payload.get("acceptance_criteria")
    if isinstance(criteria, list):
        parts.extend([item for item in criteria if isinstance(item, str)])
    text = " ".join(parts).lower()
    keyword_map = [
        ("validate", ["validate", "verify", "check", "lint", "schema"]),
        ("render", ["render", "rendering", "docx", "pdf"]),
        ("generate", ["generate", "create", "draft", "write", "compose", "produce", "build"]),
        ("transform", ["transform", "reshape", "wrap", "convert", "summarize", "repair"]),
        ("io", ["read", "fetch", "list", "search", "load", "save", "download", "upload"]),
    ]
    for intent, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            return intent
    return "generate"


def _intent_mismatch(task_intent: str, tool_intent: models.ToolIntent, tool_name: str) -> str | None:
    if task_intent == "generate" and tool_intent == models.ToolIntent.transform:
        return f"tool_intent_mismatch:{tool_name}:{tool_intent.value}:{task_intent}"
    return None


def _tool_payload(
    tool_name: str,
    instruction: str,
    context: dict,
    task_payload: dict,
    tool_inputs: dict,
) -> dict:
    if task_payload.get("tool_inputs_resolved"):
        resolved = tool_inputs.get(tool_name) if isinstance(tool_inputs, dict) else None
        if isinstance(resolved, dict):
            return resolved
    has_tool_inputs = False
    payload = {}
    if isinstance(tool_inputs, dict) and tool_name in tool_inputs:
        tool_payload = tool_inputs.get(tool_name)
        if isinstance(tool_payload, dict):
            payload = dict(tool_payload)
            has_tool_inputs = True
    payload = _merge_payload_from_task(payload, task_payload, instruction)
    payload = _fill_payload_from_context(payload, context)
    if tool_name == "llm_generate":
        base_text = payload.get("text") or payload.get("prompt") or instruction
        if context:
            try:
                context_blob = json.dumps(context, indent=2, ensure_ascii=True)
            except (TypeError, ValueError):
                context_blob = str(context)
            base_text = f"{base_text}\n\nContext (JSON):\n{context_blob}"
        return {"text": base_text}
    if has_tool_inputs:
        return payload
    if payload:
        return payload
    if not context:
        return {"text": instruction}
    try:
        context_blob = json.dumps(context, indent=2, ensure_ascii=True)
    except (TypeError, ValueError):
        context_blob = str(context)
    return {"text": f"{instruction}\n\nContext (JSON):\n{context_blob}"}


def _merge_payload_from_task(payload: dict, task_payload: dict, instruction: str) -> dict:
    merged = dict(payload)
    if not isinstance(task_payload, dict):
        task_payload = {}
    for key in (
        "data",
        "schema_ref",
        "template_id",
        "template_path",
        "output_path",
        "path",
        "content",
        "render_context",
    ):
        if key not in merged and key in task_payload:
            merged[key] = task_payload.get(key)
    if "schema_ref" not in merged:
        value = _extract_schema_ref(instruction)
        if value:
            merged["schema_ref"] = value
    if "template_id" not in merged:
        value = _extract_template_id(instruction)
        if value:
            merged["template_id"] = value
    if "template_path" not in merged:
        value = _extract_template_path(instruction)
        if value:
            merged["template_path"] = value
    if "output_path" not in merged:
        value = _extract_output_path(instruction)
        if value:
            merged["output_path"] = value
    if "path" not in merged and isinstance(task_payload.get("path"), str):
        merged["path"] = task_payload.get("path")
    if "content" not in merged and isinstance(task_payload.get("content"), str):
        merged["content"] = task_payload.get("content")
    return merged


def _fill_payload_from_context(payload: dict, context: dict) -> dict:
    filled = dict(payload)
    if "document_spec" not in filled:
        doc = _extract_document_spec_from_context(context)
        if isinstance(doc, dict):
            filled["document_spec"] = doc
    if "data" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["data"] = doc
    if "errors" not in filled:
        errors = _extract_validation_errors_from_context(context)
        if isinstance(errors, list):
            filled["errors"] = errors
    if "validation_report" not in filled:
        report = _extract_validation_report_from_context(context)
        if isinstance(report, dict):
            filled["validation_report"] = report
    if "original_spec" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["original_spec"] = doc
    if "tailored_resume" not in filled:
        resume = _extract_resume_content_from_context(context)
        if resume is not None:
            filled["tailored_resume"] = resume
    if "resume_content" not in filled:
        resume = _extract_resume_content_from_context(context)
        if resume is not None:
            filled["resume_content"] = resume
    if "resume_doc_spec" not in filled:
        resume_spec = _extract_resume_doc_spec_from_context(context)
        if isinstance(resume_spec, dict):
            filled["resume_doc_spec"] = resume_spec
    if "tailored_text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["tailored_text"] = text
    if "content" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["content"] = text
    if "text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["text"] = text
    return filled


def _extract_json_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            json_candidate = _extract_json_from_outputs(output)
            if isinstance(json_candidate, dict):
                return json_candidate
    return None


def _extract_document_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            doc = _extract_document_spec_from_outputs(output)
            if isinstance(doc, dict):
                return doc
    return None


def _extract_resume_doc_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            resume_spec = _extract_resume_doc_spec_from_outputs(output)
            if isinstance(resume_spec, dict):
                return resume_spec
    return None


def _extract_validation_errors_from_context(context: dict) -> list[dict] | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                errors = entry.get("errors")
                if isinstance(errors, list):
                    return errors
    return None


def _extract_validation_report_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                return entry
    return None


def _extract_resume_content_from_context(context: dict) -> dict | str | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            value = _extract_resume_content_from_outputs(output)
            if value is not None:
                return value
    return None


def _extract_json_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    spec_output = outputs.get("llm_generate_document_spec")
    if isinstance(spec_output, dict):
        document_spec = spec_output.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    resume_spec_output = outputs.get("llm_generate_resume_doc_spec")
    if isinstance(resume_spec_output, dict):
        resume_doc_spec = resume_spec_output.get("resume_doc_spec")
        if isinstance(resume_doc_spec, dict):
            return resume_doc_spec
    resume_spec_text_output = outputs.get("llm_generate_resume_doc_spec_from_text")
    if isinstance(resume_spec_text_output, dict):
        resume_doc_spec = resume_spec_text_output.get("resume_doc_spec")
        if isinstance(resume_doc_spec, dict):
            return resume_doc_spec
    converted = outputs.get("resume_doc_spec_to_document_spec")
    if isinstance(converted, dict):
        document_spec = converted.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            json_text = _extract_json(text)
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return None
    llm_repair = outputs.get("llm_repair_json")
    if isinstance(llm_repair, dict):
        document_spec = llm_repair.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_improve = outputs.get("llm_improve_document_spec")
    if isinstance(llm_improve, dict):
        document_spec = llm_improve.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    json_transform = outputs.get("json_transform")
    if isinstance(json_transform, dict):
        result = json_transform.get("result")
        if isinstance(result, dict):
            return result
    payload = outputs.get("result")
    if isinstance(payload, dict):
        return payload
    return None


def _extract_document_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    for key in (
        "resume_doc_spec_to_document_spec",
        "llm_generate_document_spec",
        "llm_repair_json",
        "llm_improve_document_spec",
    ):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            document_spec = candidate.get("document_spec")
            if isinstance(document_spec, dict):
                return document_spec
    direct = outputs.get("document_spec")
    if isinstance(direct, dict):
        return direct
    return None


def _extract_resume_doc_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    for key in (
        "llm_generate_resume_doc_spec",
        "llm_generate_resume_doc_spec_from_text",
    ):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            resume_doc_spec = candidate.get("resume_doc_spec")
            if isinstance(resume_doc_spec, dict):
                return resume_doc_spec
    direct = outputs.get("resume_doc_spec")
    if isinstance(direct, dict):
        return direct
    return None


def _extract_resume_content_from_outputs(outputs: dict) -> dict | str | None:
    if not isinstance(outputs, dict):
        return None
    for key in ("tailored_resume", "resume_content"):
        value = outputs.get(key)
        if isinstance(value, (dict, str)):
            return value
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            json_text = _extract_json(text)
            if json_text:
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError:
                    return None
                if isinstance(data, dict):
                    for key in ("tailored_resume", "resume_content"):
                        value = data.get(key)
                        if isinstance(value, (dict, str)):
                            return value
                    return data
    json_transform = outputs.get("json_transform")
    if isinstance(json_transform, dict):
        result = json_transform.get("result")
        if isinstance(result, dict):
            for key in ("tailored_resume", "resume_content"):
                value = result.get(key)
                if isinstance(value, (dict, str)):
                    return value
            return result
        if isinstance(result, str):
            return result
    return None


def _extract_template_id(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("template_id")
            if isinstance(value, str) and value:
                return value
            output = data.get("output")
            if isinstance(output, dict):
                value = output.get("template_id")
                if isinstance(value, str) and value:
                    return value
    value = _extract_key_from_text(instruction, "template_id")
    if value:
        return value
    return None


def _extract_template_path(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("template_path")
            if isinstance(value, str) and value:
                return value
    value = _extract_key_from_text(instruction, "template_path")
    if value:
        return value
    return None


def _extract_output_path(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("output_path")
            if isinstance(value, str) and value:
                return value
    value = _extract_key_from_text(instruction, "output_path")
    if value:
        return value
    return None


def _extract_schema_ref(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("schema_ref")
            if isinstance(value, str) and value:
                return value
            output = data.get("output")
            if isinstance(output, dict):
                value = output.get("schema_ref")
                if isinstance(value, str) and value:
                    return value
    value = _extract_key_from_text(instruction, "schema_ref")
    if value:
        return value
    return None


def _extract_key_from_text(text: str, key: str) -> str | None:
    import re

    patterns = [
        rf"{key}\\s*[:=]\\s*\"([^\"]+)\"",
        rf"{key}\\s*[:=]\\s*'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_template_id_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("template_id")
    return value if isinstance(value, str) and value else None


def _extract_template_path_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("template_path")
    return value if isinstance(value, str) and value else None


def _extract_output_path_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("output_path")
    return value if isinstance(value, str) and value else None


def _extract_schema_ref_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("schema_ref")
    return value if isinstance(value, str) and value else None


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
    direct = outputs.get("tailored_text")
    if isinstance(direct, str) and direct:
        return direct
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
            value = entry.get("tailored_text")
            if isinstance(value, str) and value:
                return value
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
    registry_path = Path(SCHEMA_REGISTRY_PATH) / name
    if registry_path.exists():
        return registry_path
    template_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    return template_dir / name


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


def _extract_instruction_payload(instruction: str) -> dict:
    if not instruction:
        return {}
    json_text = _extract_json(instruction)
    if not json_text:
        return {}
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


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
