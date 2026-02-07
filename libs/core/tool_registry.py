from __future__ import annotations

import json
import math
import os
import time
from subprocess import CompletedProcess, run
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from .llm_provider import LLMProvider
from . import prompts
from .models import RiskLevel, ToolCall, ToolIntent, ToolSpec
from libs.tools.docx_generate import register_docx_tools
from libs.tools.document_spec_validate import register_document_spec_tools
from libs.tools.resume_doc_spec_validate import register_resume_doc_spec_tools
from libs.tools.resume_doc_spec_to_document_spec import register_resume_doc_spec_convert_tools
from libs.tools.cover_letter_generate_ats_docx import register_cover_letter_generate_tools

class ToolExecutionError(Exception):
    pass


tool_input_type = Dict[str, Any]
tool_output_type = Dict[str, Any]


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[tool_input_type], tool_output_type]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def list_specs(self) -> List[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def execute(
        self,
        name: str,
        payload: tool_input_type,
        idempotency_key: str,
        trace_id: str,
        max_output_bytes: int = 50000,
    ) -> ToolCall:
        tool = self.get(name)
        started_at = time.time()
        payload_for_call = _sanitize_payload(payload)
        try:
            tool.spec.model_validate(tool.spec.model_dump())
            _validate_schema(tool.spec.input_schema, payload, "input")
            result = _run_with_timeout(lambda: tool.handler(payload), tool.spec.timeout_s)
            _validate_schema(tool.spec.output_schema, result, "output")
            result_bytes = json.dumps(result, ensure_ascii=True).encode("utf-8")
            if len(result_bytes) > max_output_bytes:
                raise ToolExecutionError("Tool output exceeded max size")
            status = "completed"
            output = result
        except (ToolExecutionError, ValidationError) as exc:
            status = "failed"
            output = {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            output = {"error": str(exc)}
        finished_at = time.time()
        return ToolCall(
            tool_name=name,
            input=payload_for_call,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            started_at=_to_datetime(started_at),
            finished_at=_to_datetime(finished_at),
            status=status,
            output_or_error=output,
        )


def _to_datetime(timestamp: float):
    from datetime import datetime

    return datetime.utcfromtimestamp(timestamp)


def _sanitize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    def sanitize(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(key, str) and key.startswith("_"):
                    continue
                cleaned_item = sanitize(item)
                if cleaned_item is not _SKIP:
                    cleaned[key] = cleaned_item
            return cleaned
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                cleaned_item = sanitize(item)
                if cleaned_item is not _SKIP:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        try:
            json.dumps(value, ensure_ascii=True)
            return value
        except Exception:  # noqa: BLE001
            return str(value)

    _SKIP = object()
    return sanitize(payload) or {}


def _run_with_timeout(handler: Callable[[], tool_output_type], timeout_s: int | None) -> tool_output_type:
    if not timeout_s or timeout_s <= 0:
        return handler()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(handler)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            raise ToolExecutionError(f"Tool execution timed out after {timeout_s}s") from exc


def _validate_schema(schema: Dict[str, Any] | None, payload: Dict[str, Any], label: str) -> None:
    if not schema:
        return
    try:
        validator = Draft202012Validator(schema)
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"Invalid {label} schema: {exc}") from exc
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if errors:
        messages = "; ".join(
            f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors[:5]
        )
        raise ToolExecutionError(f"{label} schema validation failed: {messages}")


def _safe_artifact_path(path: str, default_name: str) -> Path:
    base_dir = Path("/shared/artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside /shared/artifacts")
    return resolved


def _resolve_schema_path(schema_ref: str) -> Path:
    registry_dir = Path(os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas"))
    template_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    name = schema_ref
    if schema_ref.startswith("schema/"):
        name = schema_ref.split("/", 1)[1]
    if not name.endswith(".json"):
        name = f"{name}.json"
    candidate = registry_dir / name
    if candidate.exists():
        return candidate
    return template_dir / name


def _validate_schema_from_registry(schema_ref: str, payload: Dict[str, Any]) -> None:
    schema_path = _resolve_schema_path(schema_ref)
    if not schema_path.exists():
        raise ToolExecutionError(f"schema_not_found:{schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"invalid_schema:{exc}") from exc
    _validate_schema(schema, payload, "input")


def _parse_http_allowlist() -> List[str]:
    raw = os.getenv("TOOL_HTTP_FETCH_ALLOWLIST", "")
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _resolve_llm_timeout_s(provider: Optional[LLMProvider]) -> int:
    if provider is not None and hasattr(provider, "timeout_s"):
        try:
            return max(1, int(math.ceil(getattr(provider, "timeout_s"))))
        except (TypeError, ValueError):
            pass
    env_timeout = os.getenv("OPENAI_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _host_allowed(host: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return False
    if "*" in allowlist:
        return True
    for entry in allowlist:
        if entry.startswith("*."):
            suffix = entry[1:]
            if host.endswith(suffix):
                return True
        elif entry.startswith("."):
            if host.endswith(entry):
                return True
        elif host == entry:
            return True
    return False



def default_registry(
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="json_transform",
                description="Wrap or reshape already-available JSON input",
                usage_guidance=(
                    "Use only when you already have structured JSON and need to pass it forward. "
                    "Provide the JSON as the 'input' field. This tool does NOT generate new content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"input": {}},
                    "required": ["input"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"result": {}},
                    "required": ["result"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload: {"result": payload.get("input")},
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="math_eval",
                description="Evaluate a safe math expression",
                usage_guidance=(
                    "Use for deterministic math when you can provide a concrete expression. "
                    "Pass the expression as a string in the 'expr' field (example: '14*12')."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"expr": {"type": "string", "minLength": 1}},
                    "required": ["expr"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
                timeout_s=3,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=_math_eval,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="text_summarize",
                description="Summarize text by truncation",
                usage_guidance="Use to shorten an existing text. Provide input in the 'text' field.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "minLength": 1}},
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=_text_summarize,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_artifact",
                description="Write artifact content to shared volume",
                usage_guidance=(
                    "Use to write text to /shared/artifacts. Provide 'content' (required) "
                    "and optional 'path' (relative filename). Defaults to artifact.txt."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=lambda payload: _write_text_file(payload, default_filename="artifact.txt"),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_text",
                description="Write text content to a file under /shared/artifacts",
                usage_guidance=(
                    "Use to write text to /shared/artifacts. Provide 'content' (required) "
                    "and 'path' (required, include the filename)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=lambda payload: _write_text_file(payload, default_filename="output.txt"),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_code",
                description="Write code content to a file under /shared/artifacts",
                usage_guidance=(
                    "Use to write code files to /shared/artifacts. Provide 'content' and 'path' "
                    "(required, include the filename). The tool creates missing directories and "
                    "expects a code file extension (e.g., .py, .js, .ts, .html, .css)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_file_write_code,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_read_text",
                description="Read text content from a file under /shared/artifacts",
                usage_guidance=(
                    "Use to read a text file from /shared/artifacts. Provide the 'path' "
                    "relative to /shared/artifacts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "minLength": 1}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_file_read_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="list_files",
                description="List files under /shared/artifacts",
                usage_guidance=(
                    "Use to list files under /shared/artifacts. Provide optional 'path' (relative "
                    "subdirectory), 'recursive' (bool), and 'max_files' (int)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_files": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["path", "type"],
                            },
                        }
                    },
                    "required": ["entries"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=_list_files,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="run_tests",
                description="Run tests within /shared/artifacts using an allowlisted command",
                usage_guidance=(
                    "Use to run tests in /shared/artifacts. Provide 'command' and optional 'args' "
                    "and 'cwd' (relative). Only allowlisted commands are permitted."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "minLength": 1},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                    "required": ["command"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "exit_code": {"type": "integer"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                    },
                    "required": ["exit_code", "stdout", "stderr"],
                },
                timeout_s=30,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.validate,
            ),
            handler=_run_tests,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="search_text",
                description="Search for text in files under /shared/artifacts",
                usage_guidance=(
                    "Use to find text in files under /shared/artifacts. Provide 'query' (required), "
                    "optional 'path', 'glob', 'case_sensitive', 'regex', 'context_lines', and 'max_matches'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "path": {"type": "string"},
                        "glob": {"type": "string"},
                        "case_sensitive": {"type": "boolean"},
                        "regex": {"type": "boolean"},
                        "context_lines": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "text": {"type": "string"},
                                    "context": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["path", "line", "text"],
                            },
                        }
                    },
                    "required": ["matches"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=_search_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="docx_render",
                description="Render a DOCX file from structured JSON and a DOCX template",
                usage_guidance=(
                    "Provide data (object), plus either template_id (resolved under /shared/templates) "
                    "or template_path. Optionally include schema_ref to validate data against a schema "
                    "from the registry before rendering. Optionally set output_path for the rendered DOCX."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "schema_ref": {"type": "string"},
                        "template_id": {"type": "string"},
                        "template_path": {"type": "string"},
                        "output_path": {"type": "string"},
                    },
                    "required": ["data", "output_path"],
                    "anyOf": [
                        {"required": ["template_id"]},
                        {"required": ["template_path"]},
                    ],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=20,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.render,
            ),
            handler=_docx_render,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="sleep",
                description="Sleep for a number of seconds",
                usage_guidance="Use only for testing delays. Provide seconds as a number.",
                input_schema={
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"slept": {"type": "number"}},
                    "required": ["slept"],
                },
                timeout_s=10,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=_sleep,
        )
    )

    register_docx_tools(registry)
    register_document_spec_tools(registry)
    register_resume_doc_spec_tools(registry)
    register_resume_doc_spec_convert_tools(registry)
    register_cover_letter_generate_tools(registry)

    if http_fetch_enabled:
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="http_fetch",
                    description="Fetch HTTP content",
                    usage_guidance=(
                        "Use to fetch public HTTP(S) URLs. The host must be in TOOL_HTTP_FETCH_ALLOWLIST."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"url": {"type": "string", "minLength": 1}},
                        "required": ["url"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"body": {"type": "string"}},
                        "required": ["body"],
                    },
                    timeout_s=10,
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.io,
                ),
                handler=_http_fetch,
            )
        )

    if llm_enabled:
        if llm_provider is None:
            raise ValueError("llm_enabled requires a llm_provider instance")
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_generate",
                    description="Generate text with an LLM",
                    usage_guidance=(
                        "Use for open-ended text generation or reasoning. "
                        "Provide the prompt in 'text' (preferred) or 'prompt'."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "prompt": {"type": "string", "minLength": 1},
                        },
                        "anyOf": [{"required": ["text"]}, {"required": ["prompt"]}],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"tailored_text": {"type": "string"}},
                        "required": ["tailored_text"],
                    },
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_generate(payload, provider),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_iterative_improve_tailored_resume_text",
                    description=(
                        "Iteratively improve tailored resume text until alignment threshold or max iterations"
                    ),
                    usage_guidance=(
                        "Provide tailored_text (required). Optionally provide job, min_alignment_score "
                        "(0-100), and max_iterations. Returns the best tailored_text plus alignment stats."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "job": {"type": "object"},
                            "min_alignment_score": {"type": "number"},
                            "max_iterations": {"type": "integer"},
                        },
                        "required": ["tailored_text"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "alignment_score": {"type": "number"},
                            "alignment_summary": {"type": "string"},
                            "iterations": {"type": "integer"},
                            "reached_threshold": {"type": "boolean"},
                            "history": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": [
                            "tailored_text",
                            "alignment_score",
                            "alignment_summary",
                            "iterations",
                            "reached_threshold",
                        ],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_iterative_improve_tailored_resume_text(
                    payload, provider
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_tailor_resume_text",
                    description="Tailor a resume to a job description and return plain text sections",
                    usage_guidance=(
                        "Provide a full job object in 'job'. The job should include context_json "
                        "with the job description, candidate resume, target role name, and seniority "
                        "level. The tool returns plain text with SECTION headings."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"job": {"type": "object"}},
                        "required": ["job"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"tailored_text": {"type": "string"}},
                        "required": ["tailored_text"],
                    },
                    memory_reads=["job_context"],
                    memory_writes=["task_outputs"],
                    examples=[
                        {
                            "task": {
                                "name": "TailorResumeText",
                                "tool_requests": ["llm_tailor_resume_text"],
                                "tool_inputs": {
                                    "llm_tailor_resume_text": {
                                        "job": {
                                            "id": "job-id",
                                            "goal": "Tailor resume for target role",
                                            "context_json": {
                                                "job_description": "Paste JD here",
                                                "candidate_resume": "Paste resume here",
                                                "target_role_name": "Target role",
                                                "seniority_level": "Senior",
                                            },
                                        }
                                    }
                                },
                            }
                        }
                    ],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_tailor_resume_text(
                    payload, provider
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_improve_tailored_resume_text",
                    description="Review and improve tailored resume text while preserving truthfulness",
                    usage_guidance=(
                        "Provide tailored_text (required) and job context (optional). "
                        "Returns improved tailored_text plus alignment score and summary."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "job": {"type": "object"},
                        },
                        "required": ["tailored_text"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "alignment_score": {"type": "number"},
                            "alignment_summary": {"type": "string"},
                        },
                        "required": ["tailored_text", "alignment_score", "alignment_summary"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.transform,
                ),
                handler=lambda payload, provider=llm_provider: _llm_improve_tailored_resume_text(
                    payload, provider
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_generate_resume_doc_spec_from_text",
                    description="Generate a ResumeDocSpec JSON from tailored resume text",
                    usage_guidance=(
                        "Provide tailored resume text in 'tailored_text'. Optionally provide "
                        "job context in 'job'. Returns a resume_doc_spec object."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "job": {"type": "object"},
                        },
                        "required": ["tailored_text"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"resume_doc_spec": {"type": "object"}},
                        "required": ["resume_doc_spec"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    examples=[
                        {
                            "task": {
                                "name": "BuildResumeDocSpecFromText",
                                "tool_requests": ["llm_generate_resume_doc_spec_from_text"],
                                "deps": ["TailorResumeText"],
                                "tool_inputs": {
                                    "llm_generate_resume_doc_spec_from_text": {
                                        "tailored_text": {
                                            "$ref": "tasks.TailorResumeText.output.text"
                                        },
                                        "job": {"id": "job-id", "goal": "Tailor resume"},
                                    }
                                },
                            }
                        }
                    ],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_generate_resume_doc_spec_from_text(
                    payload, provider
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
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_generate_document_spec(
                    payload, provider
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_generate_resume_doc_spec",
                    description="Generate a ResumeDocSpec JSON using an LLM",
                    usage_guidance=(
                        "Use for resume or CV generation when a ResumeDocSpec is needed. "
                        "Provide job context in 'job'. You MUST provide 'tailored_resume' from a "
                        "prior task. The output is a 'resume_doc_spec' object "
                        "matching the required schema and style (header, sections, roles, education, "
                        "certifications, and styles)."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "job": {"type": "object"},
                            "tailored_resume": {"type": ["object", "string"]},
                        },
                        "required": ["job", "tailored_resume"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"resume_doc_spec": {"type": "object"}},
                        "required": ["resume_doc_spec"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    examples=[
                        {
                            "task": {
                                "name": "BuildResumeDocSpec",
                                "tool_requests": ["llm_generate_resume_doc_spec"],
                                "deps": ["TailorResumeContent"],
                                "tool_inputs": {
                                    "llm_generate_resume_doc_spec": {
                                        "job": {"id": "job-id", "goal": "Tailor resume"},
                                        "tailored_resume": {
                                            "$ref": "tasks.TailorResumeContent.output.tailored_resume"
                                        },
                                    }
                                },
                            }
                        }
                    ],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _llm_generate_resume_doc_spec(
                    payload, provider
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_repair_json",
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
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.transform,
                ),
                handler=lambda payload, provider=llm_provider: _llm_repair_json(payload, provider),
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
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.transform,
                ),
                handler=lambda payload, provider=llm_provider: _llm_improve_document_spec(
                    payload, provider
                ),
            )
        )

    return registry



def _math_eval(payload: Dict[str, Any]) -> Dict[str, Any]:
    expr = payload.get("expr", "0")
    allowed = {"sqrt": math.sqrt, "pow": pow}
    value = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
    return {"value": value}



def _text_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    summary = text[:200]
    return {"summary": summary}



def _write_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_artifact_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_text_file(payload, default_filename="output.txt")


def _ensure_code_extension(path: str) -> None:
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".json",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".sh",
        ".sql",
        ".txt",
    }
    suffix = Path(path).suffix.lower()
    if not suffix or suffix not in code_extensions:
        raise ToolExecutionError("Unsupported code file extension")


def _file_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_artifact_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}


def _list_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"entries": []}
    if root.is_file():
        return {"entries": [{"path": str(root), "type": "file"}]}
    entries = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for entry in iterator:
        entry_type = "dir" if entry.is_dir() else "file"
        entries.append({"path": str(entry), "type": entry_type})
        if len(entries) >= max_files:
            break
    return {"entries": entries}


def _run_tests(payload: Dict[str, Any]) -> Dict[str, Any]:
    command = payload.get("command", "")
    args = payload.get("args") or []
    cwd = payload.get("cwd", "")
    if not isinstance(args, list):
        args = []
    allowlist = {"pytest", "python"}
    if command not in allowlist:
        raise ToolExecutionError("Command not allowed")
    cmd = [command, *args]
    if command == "python":
        if len(args) < 2 or args[0] != "-m" or args[1] != "pytest":
            raise ToolExecutionError("Only python -m pytest is allowed")
    run_cwd = _safe_artifact_path(cwd, "")
    if not run_cwd.exists():
        raise ToolExecutionError("Working directory not found")
    result: CompletedProcess[str] = run(cmd, cwd=str(run_cwd), capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _search_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "")
    if not isinstance(query, str) or not query:
        raise ToolExecutionError("Missing query")
    path = payload.get("path", "")
    glob = payload.get("glob", "")
    case_sensitive = bool(payload.get("case_sensitive", False))
    use_regex = bool(payload.get("regex", False))
    context_lines = payload.get("context_lines", 0)
    if not isinstance(context_lines, int) or context_lines < 0:
        context_lines = 0
    max_matches = payload.get("max_matches", 200)
    if not isinstance(max_matches, int) or max_matches < 1:
        max_matches = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"matches": []}
    pattern = glob or "**/*"
    matches = []
    needle = query if case_sensitive else query.lower()
    regex = None
    if use_regex:
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags=flags)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid regex: {exc}") from exc
    for file_path in root.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
                for idx, line in enumerate(lines, start=1):
                    hay = line if case_sensitive else line.lower()
                    matched = False
                    if regex is not None:
                        matched = regex.search(line) is not None
                    else:
                        matched = needle in hay
                    if matched:
                        start = max(0, idx - 1 - context_lines)
                        end = min(len(lines), idx - 1 + context_lines + 1)
                        context = [item.rstrip("\n") for item in lines[start:end]]
                        entry = {"path": str(file_path), "line": idx, "text": line.rstrip("\n")}
                        if context_lines:
                            entry["context"] = context
                        matches.append(entry)
                        if len(matches) >= max_matches:
                            return {"matches": matches}
        except OSError:
            continue
    return {"matches": matches}


def _resolve_template_path(template_path: str, template_id: str) -> Path:
    base_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(template_path or "")
    if not template_path:
        candidate = Path(f"{template_id}.docx")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid template path outside template directory")
    return resolved


def _docx_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from docxtpl import DocxTemplate
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError("docxtpl is not installed") from exc
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ToolExecutionError("data must be an object")
    schema_ref = payload.get("schema_ref", "")
    template_id = payload.get("template_id", "")
    template_path = payload.get("template_path", "")
    output_path = payload.get("output_path", "")
    if not output_path:
        raise ToolExecutionError("Missing output_path")
    if not schema_ref and template_id:
        schema_ref = template_id
    if schema_ref:
        _validate_schema_from_registry(schema_ref, data)
    if not template_id and not template_path:
        raise ToolExecutionError("Missing template_id or template_path")
    template_file = _resolve_template_path(template_path, template_id)
    if not template_file.exists():
        raise ToolExecutionError("DOCX template not found")
    candidate = _safe_artifact_path(output_path, f"docx_{template_id or 'output'}.docx")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    doc = DocxTemplate(str(template_file))
    doc.render(data)
    doc.save(str(candidate))
    return {"path": str(candidate)}



def _sleep(payload: Dict[str, Any]) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0))
    time.sleep(seconds)
    return {"slept": seconds}



def _http_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request

    url = payload.get("url", "")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError("Unsupported URL scheme")
    host = parsed.hostname or ""
    allowlist = _parse_http_allowlist()
    if not _host_allowed(host, allowlist):
        raise ToolExecutionError("URL host not in allowlist")
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
    return {"body": body}


def _llm_generate(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    prompt = payload.get("text") or payload.get("prompt") or ""
    response = provider.generate(prompt)
    return {"text": response.content}


def _llm_generate_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
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
    return {"document_spec": document_spec}


def _llm_tailor_resume_text(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    context_json = job.get("context_json") if isinstance(job, dict) else None
    candidate_resume = None
    if isinstance(context_json, dict):
        candidate_resume = context_json.get("candidate_resume")
    if not isinstance(candidate_resume, str) or not candidate_resume.strip():
        raise ToolExecutionError("candidate_resume_missing")
    prompt = prompts.resume_tailoring_prompt(job)
    response = provider.generate(prompt)
    text = response.content.strip()
    if not text:
        raise ToolExecutionError("Empty response from LLM")
    _ensure_required_resume_sections(text)
    return {"tailored_text": text}


def _llm_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    job = payload.get("job")
    if not isinstance(tailored_text, str) or not tailored_text.strip():
        raise ToolExecutionError("tailored_text must be a non-empty string")
    _ensure_required_resume_sections(tailored_text)
    prompt = prompts.resume_tailoring_improve_prompt(tailored_text, job=job)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        payload_out = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(payload_out, dict):
        raise ToolExecutionError("Response must be a JSON object")
    improved = payload_out.get("tailored_text")
    score = payload_out.get("alignment_score")
    summary = payload_out.get("alignment_summary")
    if not isinstance(improved, str) or not improved.strip():
        raise ToolExecutionError("tailored_text must be a non-empty string")
    if not isinstance(score, (int, float)):
        raise ToolExecutionError("alignment_score must be a number")
    if not isinstance(summary, str) or not summary.strip():
        raise ToolExecutionError("alignment_summary must be a non-empty string")
    return {
        "tailored_text": improved,
        "alignment_score": float(score),
        "alignment_summary": summary.strip(),
    }


def _llm_iterative_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    job = payload.get("job")
    if not isinstance(tailored_text, str) or not tailored_text.strip():
        raise ToolExecutionError("tailored_text must be a non-empty string")

    threshold = payload.get("min_alignment_score", 85)
    max_iterations = payload.get("max_iterations", 2)
    if not isinstance(threshold, (int, float)):
        raise ToolExecutionError("min_alignment_score must be a number")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ToolExecutionError("max_iterations must be a positive integer")

    best_text = tailored_text
    best_score = -1.0
    best_summary = ""
    reached = False
    history: list[dict[str, Any]] = []

    current_text = tailored_text
    for idx in range(max_iterations):
        prompt = prompts.resume_tailoring_improve_prompt(current_text, job=job)
        response = provider.generate(prompt)
        json_text = _extract_json(response.content)
        if not json_text:
            raise ToolExecutionError("Failed to extract JSON from LLM response")
        try:
            payload_out = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
        if not isinstance(payload_out, dict):
            raise ToolExecutionError("Response must be a JSON object")

        improved = payload_out.get("tailored_text")
        score = payload_out.get("alignment_score")
        summary = payload_out.get("alignment_summary")
        if not isinstance(improved, str) or not improved.strip():
            raise ToolExecutionError("tailored_text must be a non-empty string")
        if not isinstance(score, (int, float)):
            raise ToolExecutionError("alignment_score must be a number")
        if not isinstance(summary, str) or not summary.strip():
            raise ToolExecutionError("alignment_summary must be a non-empty string")

        score_value = float(score)
        history.append(
            {
                "iteration": idx + 1,
                "alignment_score": score_value,
                "alignment_summary": summary.strip(),
            }
        )
        if score_value > best_score:
            best_score = score_value
            best_text = improved
            best_summary = summary.strip()

        current_text = improved
        if score_value >= float(threshold):
            reached = True
            break

    return {
        "tailored_text": best_text,
        "alignment_score": best_score,
        "alignment_summary": best_summary,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }


def _llm_generate_resume_doc_spec_from_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    job = payload.get("job")
    if not isinstance(tailored_text, str) or not tailored_text.strip():
        raise ToolExecutionError("tailored_text must be a non-empty string")
    _ensure_required_resume_sections(tailored_text)
    prompt = prompts.resume_doc_spec_from_text_prompt(tailored_text, job=job)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_doc_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_doc_spec, dict):
        raise ToolExecutionError("ResumeDocSpec must be an object")
    return {"resume_doc_spec": resume_doc_spec}


def _llm_generate_resume_doc_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    job = payload.get("job")
    tailored_resume = payload.get("tailored_resume")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    prompt = prompts.resume_doc_spec_prompt(job, tailored_resume=tailored_resume)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_doc_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_doc_spec, dict):
        raise ToolExecutionError("ResumeDocSpec must be an object")
    return {"resume_doc_spec": resume_doc_spec}


def _ensure_required_resume_sections(text: str) -> None:
    required = [
        "SECTION 1 SUMMARY",
        "SECTION 2 SKILLS",
        "SECTION 3 EXPERIENCE",
        "SECTION 6 EDUCATION",
        "SECTION 7 CERTIFICATIONS",
    ]
    missing = [section for section in required if section not in text]
    if missing:
        raise ToolExecutionError(f"tailored_text_missing_sections:{','.join(missing)}")


def _llm_generate_tailored_resume_content(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    prompt = prompts.tailored_resume_content_prompt(job)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_content = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_content, dict):
        raise ToolExecutionError("resume_content must be an object")
    return {"resume_content": resume_content}


def _llm_repair_json(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
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
    allowed_json = (
        json.dumps(allowed, ensure_ascii=False)
        if isinstance(allowed, list)
        else "null"
    )
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
    return {"document_spec": document_spec}


def _llm_improve_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
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
    return {"document_spec": improved_spec}


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
