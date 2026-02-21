from __future__ import annotations

import json
from typing import Any, Callable

from libs.core import prompts
from libs.core.llm_provider import LLMProvider
from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolExecutionError
from libs.tools.document_spec_validate import _document_spec_validate

GenerateDocumentSpecFn = Callable[[dict[str, Any], LLMProvider], dict[str, Any]]
ImproveDocumentSpecFn = Callable[[dict[str, Any], LLMProvider], dict[str, Any]]
SanitizeDocumentSpecFn = Callable[[dict[str, Any]], dict[str, Any]]


def register_document_spec_iterative_tools(
    registry,
    llm_provider: LLMProvider,
    *,
    timeout_s: int,
    generate_document_spec: GenerateDocumentSpecFn,
    improve_document_spec: ImproveDocumentSpecFn,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_iterative_improve_document_spec",
                description="Iteratively generate/improve a DocumentSpec until it validates",
                usage_guidance=(
                    "Provide job (to generate) or document_spec (to improve), plus allowed_block_types. "
                    "The tool validates with document_spec_validate and calls the LLM to fix errors "
                    "until valid or max_iterations is reached."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "object"},
                        "document_spec": {"type": "object"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                        "render_context": {"type": "object"},
                        "strict": {"type": "boolean"},
                        "max_iterations": {"type": "integer", "minimum": 1, "maximum": 10},
                        "ats_mode": {"type": "boolean"},
                    },
                    "required": ["allowed_block_types"],
                    "anyOf": [{"required": ["job"]}, {"required": ["document_spec"]}],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "validation_report": {"type": "object"},
                        "iterations": {"type": "integer"},
                        "reached_threshold": {"type": "boolean"},
                        "history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "iteration": {"type": "integer"},
                                    "valid": {"type": "boolean"},
                                    "error_count": {"type": "integer"},
                                    "warning_count": {"type": "integer"},
                                },
                                "required": [
                                    "iteration",
                                    "valid",
                                    "error_count",
                                    "warning_count",
                                ],
                            },
                        },
                    },
                    "required": [
                        "document_spec",
                        "validation_report",
                        "iterations",
                        "reached_threshold",
                        "history",
                    ],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=lambda payload, provider=llm_provider: llm_iterative_improve_document_spec(
                payload,
                provider,
                generate_document_spec=generate_document_spec,
                improve_document_spec=improve_document_spec,
                sanitize_document_spec=sanitize_document_spec,
            ),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_iterative_improve_runbook_spec",
                description="Generate and iteratively improve a runbook DocumentSpec",
                usage_guidance=(
                    "Provide job and allowed_block_types. The tool generates a runbook as a DocumentSpec, "
                    "validates it, and iteratively improves until valid or max_iterations is reached."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "object"},
                        "document_spec": {"type": "object"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                        "render_context": {"type": "object"},
                        "strict": {"type": "boolean"},
                        "max_iterations": {"type": "integer", "minimum": 1, "maximum": 10},
                        "ats_mode": {"type": "boolean"},
                    },
                    "required": ["job", "allowed_block_types"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "validation_report": {"type": "object"},
                        "iterations": {"type": "integer"},
                        "reached_threshold": {"type": "boolean"},
                        "history": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "iteration": {"type": "integer"},
                                    "valid": {"type": "boolean"},
                                    "error_count": {"type": "integer"},
                                    "warning_count": {"type": "integer"},
                                },
                                "required": [
                                    "iteration",
                                    "valid",
                                    "error_count",
                                    "warning_count",
                                ],
                            },
                        },
                    },
                    "required": [
                        "document_spec",
                        "validation_report",
                        "iterations",
                        "reached_threshold",
                        "history",
                    ],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=lambda payload, provider=llm_provider: llm_iterative_improve_runbook_spec(
                payload,
                provider,
                sanitize_document_spec=sanitize_document_spec,
            ),
        )
    )


def llm_iterative_improve_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    generate_document_spec: GenerateDocumentSpecFn,
    improve_document_spec: ImproveDocumentSpecFn,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    job = payload.get("job")
    document_spec = payload.get("document_spec")
    allowed = payload.get("allowed_block_types")
    render_context = payload.get("render_context", {})
    strict = payload.get("strict", True)
    max_iterations = payload.get("max_iterations", 3)
    ats_mode = payload.get("ats_mode", False)

    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ToolExecutionError("allowed_block_types must be an array of strings")
    if render_context is None:
        render_context = {}
    if not isinstance(render_context, dict):
        raise ToolExecutionError("render_context must be an object")
    if not isinstance(strict, bool):
        raise ToolExecutionError("strict must be a boolean")
    if not isinstance(ats_mode, bool):
        raise ToolExecutionError("ats_mode must be a boolean")
    if not isinstance(max_iterations, int) or max_iterations < 1 or max_iterations > 10:
        raise ToolExecutionError("max_iterations must be an int in [1..10]")

    if isinstance(document_spec, dict):
        current_spec = sanitize_document_spec(document_spec)
    else:
        if not isinstance(job, dict):
            raise ToolExecutionError("job must be an object when document_spec is not provided")
        current_spec = generate_document_spec(
            {"job": job, "allowed_block_types": allowed}, provider
        )["document_spec"]

    history: list[dict[str, Any]] = []
    last_report: dict[str, Any] = {}

    for iteration in range(1, max_iterations + 1):
        report = _document_spec_validate(
            {
                "document_spec": current_spec,
                "render_context": render_context,
                "strict": strict,
                "allowed_block_types": allowed,
            }
        )
        if not isinstance(report, dict):
            raise ToolExecutionError("document_spec_validate returned invalid report")
        report = _filter_document_spec_validation_report(report, ats_mode=ats_mode)
        last_report = report
        errors = report.get("errors", [])
        warnings = report.get("warnings", [])
        error_count = len(errors) if isinstance(errors, list) else 0
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        valid = bool(report.get("valid")) and error_count == 0
        history.append(
            {
                "iteration": int(iteration),
                "valid": bool(valid),
                "error_count": int(error_count),
                "warning_count": int(warning_count),
            }
        )
        if valid:
            break
        current_spec = improve_document_spec(
            {
                "document_spec": current_spec,
                "validation_report": report,
                "allowed_block_types": allowed,
            },
            provider,
        )["document_spec"]

    reached = bool(last_report.get("valid")) and not bool(last_report.get("errors"))
    return {
        "document_spec": current_spec,
        "validation_report": last_report,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }


def llm_iterative_improve_runbook_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    job = payload.get("job")
    document_spec = payload.get("document_spec")
    allowed = payload.get("allowed_block_types")
    render_context = payload.get("render_context", {})
    strict = payload.get("strict", True)
    max_iterations = payload.get("max_iterations", 3)
    ats_mode = payload.get("ats_mode", False)

    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ToolExecutionError("allowed_block_types must be an array of strings")
    if render_context is None:
        render_context = {}
    if not isinstance(render_context, dict):
        raise ToolExecutionError("render_context must be an object")
    if not isinstance(strict, bool):
        raise ToolExecutionError("strict must be a boolean")
    if not isinstance(ats_mode, bool):
        raise ToolExecutionError("ats_mode must be a boolean")
    if not isinstance(max_iterations, int) or max_iterations < 1 or max_iterations > 10:
        raise ToolExecutionError("max_iterations must be an int in [1..10]")

    current_spec = (
        sanitize_document_spec(document_spec)
        if isinstance(document_spec, dict)
        else _llm_generate_runbook_document_spec(
            {"job": job, "allowed_block_types": allowed},
            provider,
            sanitize_document_spec=sanitize_document_spec,
        )["document_spec"]
    )

    history: list[dict[str, Any]] = []
    last_report: dict[str, Any] = {}

    for iteration in range(1, max_iterations + 1):
        report = _document_spec_validate(
            {
                "document_spec": current_spec,
                "render_context": render_context,
                "strict": strict,
                "allowed_block_types": allowed,
            }
        )
        if not isinstance(report, dict):
            raise ToolExecutionError("document_spec_validate returned invalid report")
        report = _filter_document_spec_validation_report(report, ats_mode=ats_mode)
        last_report = report
        errors = report.get("errors", [])
        warnings = report.get("warnings", [])
        error_count = len(errors) if isinstance(errors, list) else 0
        warning_count = len(warnings) if isinstance(warnings, list) else 0
        valid = bool(report.get("valid")) and error_count == 0
        history.append(
            {
                "iteration": int(iteration),
                "valid": bool(valid),
                "error_count": int(error_count),
                "warning_count": int(warning_count),
            }
        )
        if valid:
            break
        current_spec = _llm_improve_runbook_document_spec(
            {
                "document_spec": current_spec,
                "validation_report": report,
                "job": job,
                "allowed_block_types": allowed,
            },
            provider,
            sanitize_document_spec=sanitize_document_spec,
        )["document_spec"]

    reached = bool(last_report.get("valid")) and not bool(last_report.get("errors"))
    return {
        "document_spec": current_spec,
        "validation_report": last_report,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }


def _llm_generate_runbook_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    job = payload.get("job")
    allowed = payload.get("allowed_block_types")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ToolExecutionError("allowed_block_types must be an array of strings")
    prompt = prompts.runbook_document_spec_prompt(job, allowed)
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


def _llm_improve_runbook_document_spec(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    sanitize_document_spec: SanitizeDocumentSpecFn,
) -> dict[str, Any]:
    document_spec = payload.get("document_spec")
    validation_report = payload.get("validation_report")
    job = payload.get("job")
    allowed = payload.get("allowed_block_types")
    if not isinstance(document_spec, dict):
        raise ToolExecutionError("document_spec must be an object")
    if not isinstance(validation_report, dict):
        raise ToolExecutionError("validation_report must be an object")
    if not isinstance(job, dict):
        job = None
    if not isinstance(allowed, list):
        allowed = None
    prompt = prompts.runbook_document_spec_improve_prompt(
        document_spec, validation_report, job=job, allowed_block_types=allowed
    )
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


def _filter_document_spec_validation_report(
    report: dict[str, Any], *, ats_mode: bool
) -> dict[str, Any]:
    if ats_mode:
        return dict(report)
    warnings = report.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    filtered_warnings: list[dict[str, Any]] = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        message = warning.get("message")
        if isinstance(message, str) and message.startswith("Non-standard heading for ATS"):
            continue
        filtered_warnings.append(warning)
    filtered = dict(report)
    filtered["warnings"] = filtered_warnings
    return filtered


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
