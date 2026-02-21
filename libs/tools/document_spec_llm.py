from __future__ import annotations

import json
from typing import Any, Callable

from libs.core import prompts
from libs.core.llm_provider import LLMProvider
from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolExecutionError

SanitizeDocumentSpecFn = Callable[[dict[str, Any]], dict[str, Any]]


def register_document_spec_llm_tools(
    registry,
    llm_provider: LLMProvider,
    *,
    timeout_s: int,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_repair_document_spec",
                description="Repair invalid JSON using an LLM",
                usage_guidance=(
                    "Provide original_spec and validation errors. "
                    "Returns a repaired JSON object in document_spec."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "errors": {"type": "array", "items": {"type": "object"}},
                        "original_spec": {"type": "object"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["errors", "original_spec"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"document_spec": {"type": "object"}},
                    "required": ["document_spec"],
                },
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload, provider=llm_provider: llm_repair_document_spec(
                payload,
                provider,
                sanitize_document_spec=sanitize_document_spec,
            ),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate_document_spec",
                description="Generate a DocumentSpec JSON using an LLM",
                usage_guidance=(
                    "Provide job context and allowed_block_types. "
                    "Returns a document_spec object."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "object"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["job", "allowed_block_types"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"document_spec": {"type": "object"}},
                    "required": ["document_spec"],
                },
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=lambda payload, provider=llm_provider: llm_generate_document_spec(
                payload,
                provider,
                sanitize_document_spec=sanitize_document_spec,
            ),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_improve_document_spec",
                description="Improve a DocumentSpec JSON using a validation report",
                usage_guidance=(
                    "Provide document_spec and validation_report (output from "
                    "document_spec_validate). Returns an improved document_spec."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "validation_report": {"type": "object"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["document_spec", "validation_report"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"document_spec": {"type": "object"}},
                    "required": ["document_spec"],
                },
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload, provider=llm_provider: llm_improve_document_spec(
                payload,
                provider,
                sanitize_document_spec=sanitize_document_spec,
            ),
        )
    )


def llm_repair_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    errors = payload.get("errors", [])
    original_spec = payload.get("original_spec", {})
    allowed = payload.get("allowed_block_types")
    if not isinstance(original_spec, dict):
        raise ToolExecutionError("original_spec must be an object")
    if not isinstance(errors, list):
        errors = []
    if not errors:
        return {"document_spec": original_spec}
    errors_json = json.dumps(errors, ensure_ascii=False, indent=2)
    original_json = json.dumps(original_spec, ensure_ascii=False, indent=2)
    allowed_json = json.dumps(allowed, ensure_ascii=False) if isinstance(allowed, list) else "null"
    prompt = (
        "You are repairing a DocumentSpec JSON object. "
        "Fix the errors and return ONLY the JSON object.\n"
        f"Allowed block types: {allowed_json}\n"
        f"Validation errors: {errors_json}\n"
        f"Original DocumentSpec: {original_json}\n"
        "Return ONLY the repaired JSON object."
    )
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        document_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(document_spec, dict):
        raise ToolExecutionError("Repaired JSON must be an object")
    return {"document_spec": sanitize_document_spec(document_spec)}


def llm_generate_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    job = payload.get("job")
    allowed = payload.get("allowed_block_types")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    if not isinstance(allowed, list):
        raise ToolExecutionError("allowed_block_types must be an array")
    prompt = prompts.document_spec_prompt(job, allowed)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        document_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(document_spec, dict):
        raise ToolExecutionError("DocumentSpec must be an object")
    return {"document_spec": sanitize_document_spec(document_spec)}


def llm_improve_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    document_spec = payload.get("document_spec")
    validation_report = payload.get("validation_report")
    allowed = payload.get("allowed_block_types")
    if not isinstance(document_spec, dict):
        raise ToolExecutionError("document_spec must be an object")
    if not isinstance(validation_report, dict):
        raise ToolExecutionError("validation_report must be an object")
    prompt = prompts.document_spec_improve_prompt(document_spec, validation_report, allowed)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        improved_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(improved_spec, dict):
        raise ToolExecutionError("Improved DocumentSpec must be an object")
    return {"document_spec": sanitize_document_spec(improved_spec)}


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
