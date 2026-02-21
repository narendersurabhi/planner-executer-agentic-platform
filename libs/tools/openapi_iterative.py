from __future__ import annotations

import json
from typing import Any

from libs.core import prompts
from libs.core.llm_provider import LLMProvider
from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolExecutionError


def register_openapi_iterative_tools(registry, llm_provider: LLMProvider, timeout_s: int) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_iterative_improve_openapi_spec",
                description="Iteratively generate/improve an OpenAPI spec until it validates",
                usage_guidance=(
                    "Provide job (to generate) or openapi_spec (to improve). The tool validates basic "
                    "OpenAPI structure and calls the LLM to fix errors until valid or max_iterations "
                    "is reached."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "object"},
                        "openapi_spec": {"type": "object"},
                        "max_iterations": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "anyOf": [{"required": ["job"]}, {"required": ["openapi_spec"]}],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "openapi_spec": {"type": "object"},
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
                        "openapi_spec",
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
            handler=lambda payload, provider=llm_provider: llm_iterative_improve_openapi_spec(
                payload, provider
            ),
        )
    )


def llm_iterative_improve_openapi_spec(payload: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    job = payload.get("job")
    openapi_spec = payload.get("openapi_spec")
    max_iterations = payload.get("max_iterations", 3)

    if not isinstance(max_iterations, int) or max_iterations < 1 or max_iterations > 10:
        raise ToolExecutionError("max_iterations must be an int in [1..10]")

    if isinstance(openapi_spec, dict):
        current_spec = dict(openapi_spec)
    else:
        if not isinstance(job, dict):
            raise ToolExecutionError("job must be an object when openapi_spec is not provided")
        current_spec = _llm_generate_openapi_spec({"job": job}, provider)["openapi_spec"]

    history: list[dict[str, Any]] = []
    last_report: dict[str, Any] = {}

    for iteration in range(1, max_iterations + 1):
        report = validate_openapi_spec(current_spec)
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
        current_spec = _llm_improve_openapi_spec(
            {"openapi_spec": current_spec, "validation_report": report, "job": job}, provider
        )["openapi_spec"]

    reached = bool(last_report.get("valid")) and not bool(last_report.get("errors"))
    return {
        "openapi_spec": current_spec,
        "validation_report": last_report,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }


def validate_openapi_spec(spec: Any) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(spec, dict):
        return {
            "valid": False,
            "errors": [_openapi_err("<root>", "openapi_spec must be an object")],
            "warnings": [],
            "stats": {"path_count": 0, "operation_count": 0, "schema_count": 0},
        }

    openapi_version = spec.get("openapi")
    if not isinstance(openapi_version, str) or not openapi_version.strip().startswith("3."):
        errors.append(_openapi_err("/openapi", "openapi must be a string starting with '3.'"))

    info = spec.get("info")
    if not isinstance(info, dict):
        errors.append(_openapi_err("/info", "info must be an object"))
        info = {}
    title = info.get("title")
    version = info.get("version")
    if not isinstance(title, str) or not title.strip():
        errors.append(_openapi_err("/info/title", "info.title must be a non-empty string"))
    if not isinstance(version, str) or not version.strip():
        errors.append(_openapi_err("/info/version", "info.version must be a non-empty string"))

    paths = spec.get("paths")
    if not isinstance(paths, dict) or not paths:
        errors.append(_openapi_err("/paths", "paths must be a non-empty object"))
        paths = {}

    operation_count = 0
    for path_key, path_item in paths.items():
        if not isinstance(path_key, str) or not path_key.startswith("/"):
            errors.append(_openapi_err("/paths", f"Invalid path key: {path_key!r}"))
            continue
        if not isinstance(path_item, dict):
            errors.append(_openapi_err(f"/paths/{path_key}", "path item must be an object"))
            continue
        for method, operation in path_item.items():
            if method.lower() not in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "head",
                "options",
            }:
                continue
            if not isinstance(operation, dict):
                errors.append(
                    _openapi_err(f"/paths/{path_key}/{method}", "operation must be an object")
                )
                continue
            operation_count += 1
            if not isinstance(operation.get("responses"), dict) or not operation.get("responses"):
                errors.append(
                    _openapi_err(
                        f"/paths/{path_key}/{method}/responses",
                        "responses must be a non-empty object",
                    )
                )
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id.strip():
                warnings.append(
                    _openapi_err(
                        f"/paths/{path_key}/{method}/operationId",
                        "operationId should be a non-empty string",
                    )
                )

    if operation_count == 0 and paths:
        errors.append(_openapi_err("/paths", "No HTTP operations found under paths"))

    servers = spec.get("servers")
    if servers is None:
        warnings.append(_openapi_err("/servers", "servers is recommended"))
    elif not isinstance(servers, list):
        warnings.append(_openapi_err("/servers", "servers should be an array"))

    components = spec.get("components", {})
    schemas = {}
    if isinstance(components, dict):
        schemas = components.get("schemas") if isinstance(components.get("schemas"), dict) else {}
    schema_count = len(schemas) if isinstance(schemas, dict) else 0

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "path_count": len(paths) if isinstance(paths, dict) else 0,
            "operation_count": int(operation_count),
            "schema_count": int(schema_count),
        },
    }


def _openapi_err(path: str, message: str) -> dict[str, Any]:
    return {"path": path, "message": message}


def _llm_generate_openapi_spec(payload: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    prompt = prompts.openapi_spec_prompt(job)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        openapi_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(openapi_spec, dict):
        raise ToolExecutionError("openapi_spec must be an object")
    return {"openapi_spec": openapi_spec}


def _llm_improve_openapi_spec(payload: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    openapi_spec = payload.get("openapi_spec")
    validation_report = payload.get("validation_report")
    job = payload.get("job")
    if not isinstance(openapi_spec, dict):
        raise ToolExecutionError("openapi_spec must be an object")
    if not isinstance(validation_report, dict):
        raise ToolExecutionError("validation_report must be an object")
    if not isinstance(job, dict):
        job = None
    prompt = prompts.openapi_spec_improve_prompt(openapi_spec, validation_report, job=job)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        improved = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(improved, dict):
        raise ToolExecutionError("Improved OpenAPI spec must be an object")
    return {"openapi_spec": improved}


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
