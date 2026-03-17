from __future__ import annotations

import json
from typing import Any, Callable

from libs.core import prompts
from libs.core.job_projection import (
    project_document_generation_inputs,
    project_markdown_document_generation_inputs,
)
from libs.core.llm_provider import LLMProvider, LLMProviderError
from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolExecutionError

SanitizeDocumentSpecFn = Callable[[dict[str, Any]], dict[str, Any]]

_DEFAULT_ALLOWED_BLOCK_TYPES = [
    "text",
    "paragraph",
    "heading",
    "bullets",
    "spacer",
    "optional_paragraph",
    "repeat",
]

_DOCUMENT_SPEC_REQUIRED_INPUTS = (
    "instruction",
    "topic",
    "audience",
    "tone",
    "today",
    "output_dir",
)

_DOCUMENT_SPEC_MARKDOWN_REQUIRED_INPUTS = ("markdown_text",)


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
                name="llm_generate_document_spec_from_markdown",
                description="Transform markdown source into a DocumentSpec JSON using an LLM",
                usage_guidance=(
                    "Provide explicit fields with markdown_text as source content, "
                    "plus optional topic, audience, tone, today, output_dir, and "
                    "optional allowed_block_types. Returns a document_spec object."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "markdown_text": {"type": "string", "minLength": 1},
                        "topic": {"type": "string", "minLength": 1},
                        "audience": {"type": "string", "minLength": 1},
                        "tone": {"type": "string", "minLength": 1},
                        "today": {"type": "string", "minLength": 4},
                        "output_dir": {"type": "string", "minLength": 1},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": list(_DOCUMENT_SPEC_MARKDOWN_REQUIRED_INPUTS),
                    "not": {"required": ["job"]},
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
            handler=lambda payload, provider=llm_provider: llm_generate_document_spec_from_markdown(
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
                    "Provide explicit fields "
                    "(instruction, topic, audience, tone, today, output_dir), "
                    "plus optional allowed_block_types. Returns a document_spec object."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "minLength": 1},
                        "topic": {"type": "string", "minLength": 1},
                        "audience": {"type": "string", "minLength": 1},
                        "tone": {"type": "string", "minLength": 1},
                        "today": {"type": "string", "minLength": 4},
                        "output_dir": {"type": "string", "minLength": 1},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": list(_DOCUMENT_SPEC_REQUIRED_INPUTS),
                    "not": {"required": ["job"]},
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
    try:
        document_spec = provider.generate_json_object(prompt)
    except LLMProviderError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {"document_spec": sanitize_document_spec(document_spec)}


def llm_generate_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    if "job" in payload:
        raise ToolExecutionError(
            "job is not supported for llm_generate_document_spec; "
            "provide explicit fields: instruction, topic, audience, tone, today, output_dir"
        )
    allowed = _resolve_allowed_block_types(payload.get("allowed_block_types"))
    explicit_inputs = _compact_document_spec_inputs(payload)
    missing = [key for key in _DOCUMENT_SPEC_REQUIRED_INPUTS if not _non_empty_string(explicit_inputs.get(key))]
    if missing:
        raise ToolExecutionError(
            "Missing required explicit fields for llm_generate_document_spec: "
            + ", ".join(missing)
        )
    prompt = prompts.document_spec_prompt(explicit_inputs, allowed)
    try:
        document_spec = provider.generate_json_object(prompt)
    except LLMProviderError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {"document_spec": sanitize_document_spec(document_spec)}


def llm_generate_document_spec_from_markdown(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    if "job" in payload:
        raise ToolExecutionError(
            "job is not supported for llm_generate_document_spec_from_markdown; "
            "provide explicit fields with markdown_text"
        )
    allowed = _resolve_allowed_block_types(payload.get("allowed_block_types"))
    explicit_inputs = _compact_markdown_document_spec_inputs(payload)
    missing = [
        key
        for key in _DOCUMENT_SPEC_MARKDOWN_REQUIRED_INPUTS
        if not _non_empty_string(explicit_inputs.get(key))
    ]
    if missing:
        raise ToolExecutionError(
            "Missing required explicit fields for llm_generate_document_spec_from_markdown: "
            + ", ".join(missing)
        )
    prompt = prompts.markdown_to_document_spec_prompt(explicit_inputs, allowed)
    try:
        document_spec = provider.generate_json_object(prompt)
    except LLMProviderError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {"document_spec": sanitize_document_spec(document_spec)}


def _compact_document_spec_inputs(
    payload: dict[str, Any]
) -> dict[str, Any]:
    return project_document_generation_inputs(payload)


def _compact_markdown_document_spec_inputs(
    payload: dict[str, Any]
) -> dict[str, Any]:
    return project_markdown_document_generation_inputs(payload)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
    try:
        improved_spec = provider.generate_json_object(prompt)
    except LLMProviderError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return {"document_spec": sanitize_document_spec(improved_spec)}


def _resolve_allowed_block_types(raw: Any) -> list[str]:
    if raw is None:
        return list(_DEFAULT_ALLOWED_BLOCK_TYPES)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ToolExecutionError("allowed_block_types must be an array of strings")
    normalized = [item.strip() for item in raw if item.strip()]
    if not normalized:
        raise ToolExecutionError("allowed_block_types must contain at least one value")
    return normalized
