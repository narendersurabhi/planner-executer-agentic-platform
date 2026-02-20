from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import shutil
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from subprocess import CompletedProcess, run
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from .llm_provider import LLMProvider
from . import prompts, tracing as core_tracing
from .models import RiskLevel, ToolCall, ToolIntent, ToolSpec
from libs.tools.docx_generate_from_spec import register_docx_tools
from libs.tools.document_spec_validate import register_document_spec_tools
from libs.tools.resume_doc_spec_validate import register_resume_doc_spec_tools
from libs.tools.resume_doc_spec_to_document_spec import register_resume_doc_spec_convert_tools
from libs.tools.coverletter_doc_spec_to_document_spec import (
    register_coverletter_doc_spec_convert_tools,
)
from libs.tools.cover_letter_generate_ats_docx import register_cover_letter_generate_tools
from libs.tools.github_tools import register_github_tools


class ToolExecutionError(Exception):
    pass


tool_input_type = Dict[str, Any]
tool_output_type = Dict[str, Any]
LOGGER = logging.getLogger(__name__)


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
            raw_error = str(exc)
            output = {"error": raw_error, "error_code": _classify_tool_error(raw_error)}
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            raw_error = str(exc)
            output = {"error": raw_error, "error_code": "runtime.unhandled"}
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


def _classify_tool_error(error_text: str) -> str:
    normalized = (error_text or "").strip()
    lowered = normalized.lower()
    if lowered.startswith("contract."):
        return lowered.split(":", 1)[0]
    if normalized.startswith("input schema validation failed"):
        return "contract.input_invalid"
    if normalized.startswith("output schema validation failed"):
        return "contract.output_invalid"
    if normalized.startswith("schema_not_found:"):
        return "contract.schema_not_found"
    if normalized.startswith("invalid_schema:"):
        return "contract.schema_invalid"
    if normalized.startswith("unknown_tool:"):
        return "contract.tool_not_found"
    if normalized.startswith("memory_only_inputs_missing:"):
        return "contract.input_missing"
    if normalized.startswith("tool_intent_mismatch:"):
        return "contract.intent_mismatch"
    if (
        "missing_required_fields" in lowered
        or "tailored_resume_missing_fields" in lowered
        or "invalid_json" in lowered
        or "coder_response_missing_files" in lowered
    ):
        return "contract.output_invalid"
    if normalized.startswith("guardrail_blocked:"):
        return "policy.blocked"
    if (
        normalized.startswith("tool_call_timed_out:")
        or normalized.startswith("mcp_sdk_timeout:")
        or "timed out" in lowered
        or "timeout" in lowered
    ):
        return "runtime.timeout"
    if normalized.startswith("mcp_sdk_all_routes_failed:"):
        return "runtime.upstream_unavailable"
    if normalized.startswith("mcp_sdk_error:"):
        return "runtime.upstream_error"
    if normalized.startswith("coder_http_error:"):
        return "runtime.http_error"
    return "runtime.tool_error"


def _extract_mcp_error_phase(error_text: str) -> str | None:
    match = re.search(r"(?:^|[;,\s])phase=([a-zA-Z0-9_.:/-]+)", error_text or "")
    if not match:
        return None
    return match.group(1)


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


def _run_with_timeout(
    handler: Callable[[], tool_output_type], timeout_s: int | None
) -> tool_output_type:
    if not timeout_s or timeout_s <= 0:
        return handler()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(handler)
    try:
        return future.result(timeout=float(timeout_s))
    except FuturesTimeoutError as exc:
        future.cancel()
        raise ToolExecutionError(f"tool_call_timed_out:timed out after {timeout_s}s") from exc
    finally:
        # Never block caller on hung worker threads.
        executor.shutdown(wait=False, cancel_futures=True)


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


def _workspace_root() -> Path:
    env_root = os.getenv("WORKSPACE_DIR")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def _safe_workspace_path(path: str, default_name: str) -> Path:
    base_dir = _workspace_root()
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside workspace")
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


def _resolve_coding_agent_timeout_s() -> int:
    env_timeout = os.getenv("CODING_AGENT_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _resolve_coder_http_timeout_s() -> int:
    for key in ("CODER_HTTP_TIMEOUT_S", "CODING_AGENT_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if env_timeout:
            try:
                return max(1, int(math.ceil(float(env_timeout))))
            except ValueError:
                return 30
    return 30


def _resolve_mcp_timeout_s() -> float:
    for key in ("MCP_TOOL_TIMEOUT_S", "MCP_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if not env_timeout:
            continue
        try:
            return max(1.0, float(env_timeout))
        except ValueError:
            return 45.0
    for key in ("TAILOR_OPENAI_TIMEOUT_S", "TAILOR_EVAL_OPENAI_TIMEOUT_S", "OPENAI_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if not env_timeout:
            continue
        try:
            # Avoid hanging indefinitely when global model timeout is very large.
            return min(180.0, max(5.0, float(env_timeout)))
        except ValueError:
            continue
    return 45.0


def _resolve_mcp_outer_timeout_headroom_s() -> float:
    env_headroom = os.getenv("MCP_TOOL_OUTER_TIMEOUT_HEADROOM_S")
    if env_headroom:
        try:
            return max(0.0, min(120.0, float(env_headroom)))
        except ValueError:
            return 15.0
    return 15.0


def _resolve_mcp_tool_timeout_s() -> int:
    timeout_s = _resolve_mcp_timeout_s() + _resolve_mcp_outer_timeout_headroom_s()
    return max(1, int(math.ceil(timeout_s)))


def _resolve_mcp_max_retries() -> int:
    env_retries = os.getenv("MCP_TOOL_MAX_RETRIES")
    if env_retries:
        try:
            return max(0, int(env_retries))
        except ValueError:
            return 1
    return 1


def _resolve_mcp_retry_sleep_s() -> float:
    env_sleep = os.getenv("MCP_TOOL_RETRY_SLEEP_S")
    if env_sleep:
        try:
            return max(0.0, float(env_sleep))
        except ValueError:
            return 0.25
    return 0.25


def _resolve_mcp_first_attempt_reserve_s(timeout_s: float) -> float:
    env_reserve = os.getenv("MCP_FIRST_ATTEMPT_RESERVE_S")
    if env_reserve:
        try:
            return max(0.05, float(env_reserve))
        except ValueError:
            return min(30.0, max(5.0, timeout_s * 0.1))
    return min(30.0, max(5.0, timeout_s * 0.1))


def _resolve_mcp_transport_timeout_s(timeout_s: float) -> float:
    env_timeout = os.getenv("MCP_TRANSPORT_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1.0, float(env_timeout))
        except ValueError:
            return timeout_s
    return timeout_s


def _resolve_mcp_isolation_mode() -> str:
    mode = os.getenv("MCP_TOOL_ISOLATION_MODE", "process").strip().lower()
    return "process" if mode == "process" else "thread"


def _streamable_http_client_kwargs(
    client_factory: Callable[..., Any], timeout_s: float
) -> Dict[str, Any]:
    """Best-effort timeout kwargs for different MCP SDK versions."""
    transport_timeout = _resolve_mcp_transport_timeout_s(timeout_s)
    connect_timeout = min(10.0, transport_timeout)
    try:
        params = inspect.signature(client_factory).parameters
    except (TypeError, ValueError):
        return {}
    candidates: Dict[str, Any] = {
        "timeout": transport_timeout,
        "request_timeout": transport_timeout,
        "read_timeout": transport_timeout,
        "write_timeout": transport_timeout,
        "connect_timeout": connect_timeout,
        "sse_read_timeout": transport_timeout,
        "http_timeout": transport_timeout,
    }
    return {name: value for name, value in candidates.items() if name in params}


def _is_retryable_mcp_error(message: str) -> bool:
    lower = message.lower()
    return (
        lower.startswith("mcp_sdk_error:")
        or lower.startswith("mcp_sdk_timeout:")
        or "session terminated" in lower
        or "timeout" in lower
    )


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
                    "properties": {
                        "path": {"type": "string"},
                        "document_type": {"type": "string"},
                    },
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
                name="workspace_write_text",
                description="Write text content to a file under the workspace",
                usage_guidance=(
                    "Use to write text to the workspace. Provide 'content' (required) "
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
            handler=lambda payload: _write_workspace_text_file(
                payload, default_filename="output.txt"
            ),
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_write_code",
                description="Write code content to a file under the workspace",
                usage_guidance=(
                    "Use to write code files in the workspace. Provide 'content' and 'path' "
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
            handler=_workspace_write_code,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_read_text",
                description="Read text content from a file under the workspace",
                usage_guidance=(
                    "Use to read a text file from the workspace. Provide the 'path' "
                    "relative to the workspace root."
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
            handler=_workspace_read_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_list_files",
                description="List files under the workspace",
                usage_guidance=(
                    "Use to list files under the workspace. Provide optional 'path' (relative "
                    "subdirectory), 'recursive' (bool), and 'max_files' (int)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_files": {"type": "integer", "minimum": 1, "maximum": 2000},
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
            handler=_list_workspace_files,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_move",
                description="Move a file from /shared/artifacts into the workspace",
                usage_guidance=(
                    "Use to move an artifact into the workspace. Provide 'source_path' (relative "
                    "to /shared/artifacts) and 'destination_path' (relative to the workspace). "
                    "Set 'overwrite' true to replace an existing destination."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_artifact_move_to_workspace,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="derive_output_filename",
                description="Derive a filesystem-safe DOCX path for resumes or general documents",
                usage_guidance=(
                    "Use to create a safe output path for docx_generate_from_spec. "
                    "Set document_type='cover_letter' for cover-letter naming. "
                    "Resume format: provide candidate_name (or first_name+last_name), "
                    "target_role_name (or role_name), and company_name (or company) to get "
                    "'Firstname Lastname Resume - Target Role - Company.docx'. "
                    "If target role/company/name are missing, the tool can derive them from "
                    "job_description plus candidate_resume/tailored_text when provided. "
                    "For general documents, 'topic' can be used in place of role name. "
                    "Fallback format: provide target_role_name (or role_name or topic) and "
                    "date/today (YYYY-MM-DD) to get role_date naming. "
                    "Optionally provide 'output_dir' (default: resumes)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "target_role_name": {"type": "string", "minLength": 1},
                        "role_name": {"type": "string", "minLength": 1},
                        "topic": {"type": "string", "minLength": 1},
                        "candidate_name": {"type": "string", "minLength": 1},
                        "first_name": {"type": "string", "minLength": 1},
                        "last_name": {"type": "string", "minLength": 1},
                        "company_name": {"type": "string", "minLength": 1},
                        "company": {"type": "string", "minLength": 1},
                        "job_description": {"type": "string", "minLength": 1},
                        "candidate_resume": {"type": "string", "minLength": 1},
                        "tailored_text": {"type": "string", "minLength": 1},
                        "date": {"type": "string", "minLength": 4},
                        "today": {"type": "string", "minLength": 4},
                        "output_dir": {"type": "string"},
                        "document_type": {"type": "string"},
                    },
                    "allOf": [
                        {
                            "anyOf": [
                                {
                                    "allOf": [
                                        {
                                            "anyOf": [
                                                {"required": ["target_role_name"]},
                                                {"required": ["role_name"]},
                                                {"required": ["topic"]},
                                                {"required": ["job_description"]},
                                            ]
                                        },
                                        {
                                            "anyOf": [
                                                {"required": ["date"]},
                                                {"required": ["today"]},
                                            ]
                                        },
                                    ]
                                },
                                {
                                    "allOf": [
                                        {
                                            "anyOf": [
                                                {"required": ["candidate_name"]},
                                                {"required": ["first_name", "last_name"]},
                                            ]
                                        },
                                        {
                                            "anyOf": [
                                                {"required": ["company_name"]},
                                                {"required": ["company"]},
                                            ]
                                        },
                                        {
                                            "anyOf": [
                                                {"required": ["target_role_name"]},
                                                {"required": ["role_name"]},
                                                {"required": ["topic"]},
                                                {"required": ["job_description"]},
                                            ]
                                        },
                                    ]
                                },
                                {
                                    "allOf": [
                                        {"required": ["job_description"]},
                                        {
                                            "anyOf": [
                                                {"required": ["candidate_name"]},
                                                {"required": ["first_name", "last_name"]},
                                                {"required": ["candidate_resume"]},
                                                {"required": ["tailored_text"]},
                                            ]
                                        },
                                    ]
                                },
                                {"required": ["memory"]},
                            ]
                        }
                    ],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=2,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=_derive_output_filename,
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
    register_coverletter_doc_spec_convert_tools(registry)
    register_cover_letter_generate_tools(registry)
    register_github_tools(registry)

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
                        "Provide the prompt in 'text' (preferred) or 'prompt'. "
                        "Returns the raw completion in the 'text' field."
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
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
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
                    name="coding_agent_generate",
                    description="Generate code files using the coding agent service",
                    usage_guidance=(
                        "Use to generate code for a repo or feature. Provide 'goal' and optional "
                        "'files' (list of relative paths), 'constraints', and 'workspace_path'. "
                        "The tool calls the coding agent service and writes files to the workspace."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "minLength": 1},
                            "files": {"type": "array", "items": {"type": "string"}},
                            "constraints": {"type": "string"},
                            "workspace_path": {"type": "string"},
                        },
                        "required": ["goal"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["path", "content"],
                                },
                            },
                            "written_paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["files", "written_paths"],
                    },
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_coding_agent_timeout_s(),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=_coding_agent_generate,
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="coding_agent_autonomous",
                    description=(
                        "Autonomously plan and implement a codebase in steps using the coding agent"
                    ),
                    usage_guidance=(
                        "Provide 'goal' and 'workspace_path'. The tool creates "
                        "IMPLEMENTATION_PLAN.md, then implements each step and updates status "
                        "in the plan file until complete."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "minLength": 1},
                            "workspace_path": {"type": "string", "minLength": 1},
                            "constraints": {"type": "string"},
                            "max_steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        },
                        "required": ["goal", "workspace_path"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "plan_path": {"type": "string"},
                            "steps_total": {"type": "integer"},
                            "steps_completed": {"type": "integer"},
                            "written_paths": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "plan_path",
                            "steps_total",
                            "steps_completed",
                            "written_paths",
                        ],
                    },
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_coding_agent_timeout_s(),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: _coding_agent_autonomous(
                    payload, provider
                ),
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
                        "Provide tailored_resume (preferred). Optionally provide job, min_alignment_score "
                        "(0-100), and max_iterations. Returns the best tailored_resume plus alignment stats."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "tailored_resume": {"type": "object"},
                            "job": {"type": "object"},
                            "min_alignment_score": {"type": "number"},
                            "max_iterations": {"type": "integer"},
                        },
                        "required": [],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "tailored_resume": {"type": "object"},
                            "alignment_score": {"type": "number"},
                            "alignment_summary": {"type": "string"},
                            "alignment_feedback": {
                                "type": "object",
                                "properties": {
                                    "top_gaps": {"type": "array", "items": {"type": "string"}},
                                    "must_fix_before_95": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "missing_evidence": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "recommended_edits": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                            "iterations": {"type": "integer"},
                            "reached_threshold": {"type": "boolean"},
                            "history": {"type": "array", "items": {"type": "object"}},
                        },
                        "required": [
                            "tailored_resume",
                            "alignment_score",
                            "alignment_summary",
                            "iterations",
                            "reached_threshold",
                        ],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_mcp_tool_timeout_s(),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: (
                    _llm_iterative_improve_tailored_resume_text(payload, provider)
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_tailor_resume_text",
                    description="Tailor a resume to a job description and return structured JSON content",
                    usage_guidance=(
                        "Provide a full job object in 'job'. The job should include context_json "
                        "with the job description, candidate resume, target role name, and seniority "
                        "level. The tool returns JSON resume content."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"job": {"type": "object"}},
                        "required": ["job"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"tailored_resume": {"type": "object"}},
                        "required": ["tailored_resume"],
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
                    timeout_s=_resolve_mcp_tool_timeout_s(),
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
                        "Provide tailored_resume (preferred) and job context (optional). "
                        "Returns improved tailored_resume plus alignment score and summary."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "tailored_resume": {"type": "object"},
                            "job": {"type": "object"},
                        },
                        "required": [],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "tailored_resume": {"type": "object"},
                            "alignment_score": {"type": "number"},
                            "alignment_summary": {"type": "string"},
                            "alignment_feedback": {
                                "type": "object",
                                "properties": {
                                    "top_gaps": {"type": "array", "items": {"type": "string"}},
                                    "must_fix_before_95": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "missing_evidence": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "recommended_edits": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "required": ["tailored_resume", "alignment_score", "alignment_summary"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_mcp_tool_timeout_s(),
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
                        "Provide tailored_resume (preferred) or tailored_text. Optionally provide "
                        "job context in 'job'. Optionally provide target_pages (1 or 2) to "
                        "bias output density. Generates and validates resume_doc_spec "
                        "(resume_doc_spec_validate, strict=true) before returning it."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "tailored_resume": {"type": "object"},
                            "job": {"type": "object"},
                            "target_pages": {"type": "integer", "enum": [1, 2]},
                        },
                        "required": [],
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
                handler=lambda payload, provider=llm_provider: (
                    _llm_generate_resume_doc_spec_from_text(payload, provider)
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_generate_coverletter_doc_spec_from_text",
                    description="Generate a CoverLetterDocSpec JSON from tailored resume text and job context",
                    usage_guidance=(
                        "Provide tailored_resume (preferred) or tailored_text, plus job context in 'job'. "
                        "Generates and validates coverletter_doc_spec (strict) before returning it."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "tailored_resume": {"type": "object"},
                            "job": {"type": "object"},
                            "today_pretty": {"type": "string"},
                            "today": {"type": "string"},
                        },
                        "required": [],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"coverletter_doc_spec": {"type": "object"}},
                        "required": ["coverletter_doc_spec"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: (
                    _llm_generate_coverletter_doc_spec_from_text(payload, provider)
                ),
            )
        )
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="llm_generate_cover_letter_from_resume",
                    description="Generate a structured cover_letter JSON from tailored resume and job description",
                    usage_guidance=(
                        "Provide tailored_resume (preferred) or tailored_text. Optionally provide "
                        "job context in 'job'. Returns cover_letter JSON suitable for "
                        "cover_letter_generate_ats_docx."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tailored_text": {"type": "string"},
                            "tailored_resume": {"type": "object"},
                            "job": {"type": "object"},
                        },
                        "required": [],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"cover_letter": {"type": "object"}},
                        "required": ["cover_letter"],
                    },
                    memory_reads=["job_context", "task_outputs"],
                    memory_writes=["task_outputs"],
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.generate,
                ),
                handler=lambda payload, provider=llm_provider: (
                    _llm_generate_cover_letter_from_resume(payload, provider)
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
                        "certifications, and styles). Optionally provide target_pages (1 or 2) "
                        "to control output density."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "job": {"type": "object"},
                            "tailored_resume": {"type": ["object", "string"]},
                            "target_pages": {"type": "integer", "enum": [1, 2]},
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


def _write_workspace_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_workspace_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_text_file(payload, default_filename="output.txt")


def _workspace_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_workspace_text_file(payload, default_filename="output.txt")


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


def _workspace_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_workspace_path(path, "output.txt")
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


def _list_workspace_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_workspace_path(path, "")
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


def _artifact_move_to_workspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file name in destination_path")
    source = _safe_artifact_path(source_path, "")
    if not source.exists():
        raise ToolExecutionError("Source file not found")
    destination = _safe_workspace_path(destination_path, "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ToolExecutionError("Destination already exists")
    shutil.move(str(source), str(destination))
    return {"path": str(destination)}


def _derive_output_filename(payload: Dict[str, Any]) -> Dict[str, Any]:
    def pick_str(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    memory_context = _select_job_context_from_memory(payload.get("memory"))
    nested_context = memory_context.get("context_json")
    if not isinstance(nested_context, dict):
        nested_context = {}

    role_name = pick_str(
        payload.get("target_role_name"),
        payload.get("role_name"),
        payload.get("topic"),
        memory_context.get("target_role_name"),
        memory_context.get("role_name"),
        memory_context.get("topic"),
        nested_context.get("target_role_name"),
        nested_context.get("role_name"),
        nested_context.get("topic"),
    )
    company_name = pick_str(
        payload.get("company_name"),
        payload.get("company"),
        memory_context.get("company_name"),
        memory_context.get("company"),
        nested_context.get("company_name"),
        nested_context.get("company"),
    )
    candidate_name = pick_str(
        payload.get("candidate_name"),
        memory_context.get("candidate_name"),
        nested_context.get("candidate_name"),
    )
    first_name = pick_str(
        payload.get("first_name"),
        memory_context.get("first_name"),
        nested_context.get("first_name"),
    )
    last_name = pick_str(
        payload.get("last_name"),
        memory_context.get("last_name"),
        nested_context.get("last_name"),
    )
    job_description = pick_str(
        payload.get("job_description"),
        memory_context.get("job_description"),
        nested_context.get("job_description"),
    )
    candidate_resume = pick_str(
        payload.get("candidate_resume"),
        memory_context.get("candidate_resume"),
        nested_context.get("candidate_resume"),
    )
    tailored_text = pick_str(
        payload.get("tailored_text"),
        memory_context.get("tailored_text"),
        nested_context.get("tailored_text"),
    )
    date_value = pick_str(
        payload.get("date"),
        payload.get("today"),
        memory_context.get("date"),
        memory_context.get("today"),
        nested_context.get("date"),
        nested_context.get("today"),
    )
    output_dir = (
        pick_str(
            payload.get("output_dir"),
            memory_context.get("output_dir"),
            nested_context.get("output_dir"),
        )
        or "resumes"
    )
    document_type = pick_str(
        payload.get("document_type"),
        memory_context.get("document_type"),
        nested_context.get("document_type"),
    )
    normalized_doc_type = document_type.lower().replace("-", "_")
    is_cover_letter = normalized_doc_type in {"cover_letter", "coverletter"}
    if not isinstance(output_dir, str):
        output_dir = "resumes"
    output_dir = output_dir.strip().strip("/")
    if not output_dir:
        output_dir = "resumes"
    if output_dir.startswith("/") or ".." in Path(output_dir).parts:
        raise ToolExecutionError("Invalid output_dir")

    def clean_label(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        # Keep output human-readable while stripping filesystem-unsafe characters.
        cleaned = re.sub(r'[<>:"/\\\\|?*]', " ", value)
        cleaned = re.sub(r"[,_;:]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned

    def slugify(value: str, pattern: str) -> str:
        cleaned = re.sub(pattern, "_", value.lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned

    if (not isinstance(role_name, str) or not role_name.strip()) and isinstance(
        job_description, str
    ):
        role_name = _derive_role_name_from_jd(job_description)
    if (not isinstance(company_name, str) or not company_name.strip()) and isinstance(
        job_description, str
    ):
        company_name = _derive_company_name_from_jd(job_description)

    if (not isinstance(candidate_name, str) or not candidate_name.strip()) and not (
        isinstance(first_name, str)
        and first_name.strip()
        and isinstance(last_name, str)
        and last_name.strip()
    ):
        candidate_name = _derive_candidate_name_from_texts(
            candidate_resume, tailored_text, job_description
        )

    if (
        (
            (not isinstance(first_name, str) or not first_name.strip())
            or (not isinstance(last_name, str) or not last_name.strip())
        )
        and isinstance(candidate_name, str)
        and candidate_name.strip()
    ):
        tokens = [token for token in candidate_name.split() if token]
        if len(tokens) >= 2:
            if not isinstance(first_name, str) or not first_name.strip():
                first_name = tokens[0]
            if not isinstance(last_name, str) or not last_name.strip():
                last_name = tokens[-1]

    role_label = clean_label(role_name)
    company_label = clean_label(company_name)
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        name_parts = [clean_label(first_name), clean_label(last_name)]
        candidate_label = " ".join([part for part in name_parts if part]).strip()
    else:
        candidate_label = clean_label(candidate_name)

    if candidate_label and role_label and company_label:
        doc_label = "Cover Letter" if is_cover_letter else "Resume"
        filename = f"{candidate_label} {doc_label} - {role_label} - {company_label}.docx"
        return {
            "path": f"{output_dir}/{filename}",
            "document_type": "cover_letter" if is_cover_letter else "resume",
        }

    if not role_label:
        raise ToolExecutionError("Missing target_role_name")
    if not isinstance(date_value, str) or not date_value.strip():
        raise ToolExecutionError("Missing date")

    role_slug = slugify(role_label or str(role_name), r"[^a-z0-9]+") or "document"
    date_slug = slugify(date_value, r"[^0-9]+")
    if not date_slug:
        raise ToolExecutionError("Invalid date")
    if is_cover_letter:
        filename = f"cover_letter_{role_slug}_{date_slug}.docx"
    else:
        filename = f"{role_slug}_{date_slug}.docx"
    return {
        "path": f"{output_dir}/{filename}",
        "document_type": "cover_letter" if is_cover_letter else "resume",
    }


def _derive_role_name_from_jd(job_description: str) -> str:
    if not isinstance(job_description, str):
        return ""
    lines = _normalized_non_empty_lines(job_description)
    if not lines:
        return ""
    role_keywords = (
        "engineer",
        "developer",
        "scientist",
        "architect",
        "manager",
        "lead",
        "principal",
        "staff",
        "analyst",
    )
    stop_markers = (
        "about ",
        "about us",
        "benefits",
        "compensation",
        "salary",
        "we're looking",
        "we are looking",
        "you will",
        "responsibilities",
        "requirements",
        "qualifications",
    )
    for line in lines[:20]:
        lower = line.lower()
        if any(marker in lower for marker in stop_markers):
            continue
        if len(line.split()) > 12:
            continue
        if any(keyword in lower for keyword in role_keywords):
            return line
    for line in lines[:5]:
        if len(line.split()) <= 10 and len(line) <= 90:
            return line
    return ""


def _derive_company_name_from_jd(job_description: str) -> str:
    if not isinstance(job_description, str):
        return ""
    patterns = [
        r"(?i)\bas a [^,\n]{2,120}\bat\s+([A-Z][-A-Za-z0-9&'., ]{1,80})[,:\n]",
        r"(?i)\b([A-Z][-A-Za-z0-9&'., ]{1,80})\s+is\s+(?:the|a|an)\b",
        r"(?i)\bjoin\s+([A-Z][-A-Za-z0-9&'., ]{1,80})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, job_description)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
        if candidate and len(candidate.split()) <= 8:
            return candidate
    return ""


def _derive_candidate_name_from_texts(*texts: Any) -> str:
    disallowed = {
        "summary",
        "skills",
        "experience",
        "education",
        "certifications",
        "professional",
        "open",
        "source",
    }

    def strip_markdown_links(line: str) -> str:
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        lines = _normalized_non_empty_lines(text)
        for raw_line in lines[:12]:
            line = strip_markdown_links(raw_line)
            lower = line.lower()
            if "@" in line or "http://" in lower or "https://" in lower:
                continue
            if "linkedin" in lower or "github" in lower:
                continue
            cleaned = re.sub(r"[^A-Za-z' -]", " ", line)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            parts = [part for part in cleaned.split(" ") if part]
            if len(parts) < 2 or len(parts) > 4:
                continue
            if any(part.lower() in disallowed for part in parts):
                continue
            if not all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", part) for part in parts):
                continue
            normalized = [part.capitalize() if part.isupper() else part for part in parts]
            return " ".join(normalized)
    return ""


def _normalized_non_empty_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    return lines


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
    result: CompletedProcess[str] = run(
        cmd, cwd=str(run_cwd), capture_output=True, text=True, check=False
    )
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


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_resolve_coder_http_timeout_s()) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise ToolExecutionError(f"coder_http_error:{detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise ToolExecutionError(f"coder_http_error:{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"coder_invalid_json:{exc}") from exc


def _post_mcp_tool_call(
    service_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    base_url = service_url.rstrip("/")
    attempts = ("/mcp/rpc/mcp", "/mcp/rpc")
    timeout_s = _resolve_mcp_timeout_s()
    max_retries = _resolve_mcp_max_retries()
    retry_sleep_s = _resolve_mcp_retry_sleep_s()
    isolation_mode = _resolve_mcp_isolation_mode()
    first_attempt_reserve_s = _resolve_mcp_first_attempt_reserve_s(timeout_s)
    started_at = time.monotonic()
    deadline = started_at + timeout_s
    total_slots = len(attempts) * (max_retries + 1)
    prompt_version = os.getenv("PROMPT_VERSION", "unknown")
    policy_version = os.getenv("POLICY_VERSION", "unknown")
    tool_version = os.getenv("TOOL_VERSION", "unknown")
    errors: List[str] = []
    common_context = {
        "tool_name": tool_name,
        "service_url": base_url,
        "mcp_timeout_s": timeout_s,
        "max_retries": max_retries,
        "mcp_isolation_mode": isolation_mode,
        "prompt_version": prompt_version,
        "policy_version": policy_version,
        "tool_version": tool_version,
    }

    def _log_mcp_event(level: str, event_name: str, payload: Dict[str, Any]) -> None:
        full_payload = {**common_context, **payload}
        message = f"{event_name} payload=%s"
        encoded = json.dumps(full_payload, ensure_ascii=True, sort_keys=True)
        if level == "warning":
            LOGGER.warning(message, encoded, extra=full_payload)
            return
        if level == "error":
            LOGGER.error(message, encoded, extra=full_payload)
            return
        LOGGER.info(message, encoded, extra=full_payload)

    with core_tracing.start_span(
        "tool_registry.mcp_tool_call",
        attributes={
            "mcp.tool_name": tool_name,
            "mcp.service_url": base_url,
            "mcp.timeout_s": timeout_s,
            "mcp.max_retries": max_retries,
        },
    ) as span:

        def _finalize_and_raise(error_text: str, status: str) -> None:
            elapsed_ms = int(max(0.0, time.monotonic() - started_at) * 1000)
            payload = {
                "status": status,
                "error": error_text,
                "error_code": _classify_tool_error(error_text),
                "elapsed_ms": elapsed_ms,
                "attempt_errors": list(errors),
            }
            _log_mcp_event("error", "mcp_call_failed", payload)
            core_tracing.set_span_attributes(
                span,
                {
                    "mcp.status": status,
                    "mcp.error": error_text,
                    "mcp.elapsed_s": max(0.0, time.monotonic() - started_at),
                },
            )
            raise ToolExecutionError(error_text)

        for path_idx, mcp_path in enumerate(attempts):
            mcp_url = f"{base_url}{mcp_path}"
            with core_tracing.start_span(
                "tool_registry.mcp_route_attempt",
                attributes={
                    "mcp.tool_name": tool_name,
                    "mcp.path": mcp_path,
                    "mcp.url": mcp_url,
                },
            ) as route_span:
                for attempt in range(max_retries + 1):
                    with core_tracing.start_span(
                        "tool_registry.mcp_route_attempt_try",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.path": mcp_path,
                            "mcp.url": mcp_url,
                            "mcp.route_attempt": attempt + 1,
                            "mcp.route_attempts_total": max_retries + 1,
                        },
                    ) as attempt_span:
                        slot_idx = path_idx * (max_retries + 1) + attempt
                        remaining_s = deadline - time.monotonic()
                        remaining_slots_after_current = max(0, total_slots - slot_idx - 1)
                        attempt_started_at = time.monotonic()
                        start_payload = {
                            "mcp_url": mcp_url,
                            "route_path": mcp_path,
                            "attempt": attempt + 1,
                            "attempts_total": max_retries + 1,
                            "deadline_remaining_ms": int(max(0.0, remaining_s) * 1000),
                        }
                        _log_mcp_event("info", "mcp_attempt_start", start_payload)
                        if remaining_s <= 0.05:
                            detail = f"mcp_call_timed_out_after_{timeout_s:.1f}s"
                            timeout_error = f"mcp_sdk_timeout:{detail}"
                            _log_mcp_event(
                                "warning",
                                "mcp_attempt_timeout",
                                {
                                    **start_payload,
                                    "error": detail,
                                    "error_code": _classify_tool_error(timeout_error),
                                },
                            )
                            core_tracing.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": timeout_error,
                                },
                            )
                            core_tracing.set_span_attributes(
                                route_span,
                                {"mcp.route_status": "failed", "mcp.route_error": timeout_error},
                            )
                            _finalize_and_raise(timeout_error, "timeout")

                        if slot_idx == 0 and remaining_slots_after_current > 0:
                            reserve_s = min(
                                max(0.05, first_attempt_reserve_s),
                                max(0.0, remaining_s - 0.05),
                            )
                            per_attempt_timeout_s = max(0.05, remaining_s - reserve_s)
                            timeout_allocation = "first_attempt_bias"
                        else:
                            slots_including_current = remaining_slots_after_current + 1
                            per_attempt_timeout_s = max(0.05, remaining_s / slots_including_current)
                            timeout_allocation = "equal_remaining"
                        try:
                            core_tracing.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_timeout_s": per_attempt_timeout_s,
                                    "mcp.deadline_remaining_s": remaining_s,
                                    "mcp.timeout_allocation": timeout_allocation,
                                    "mcp.remaining_slots_after_current": remaining_slots_after_current,
                                },
                            )
                            result = _call_mcp_tool_sdk(
                                mcp_url,
                                tool_name,
                                arguments,
                                timeout_s=per_attempt_timeout_s,
                            )
                            elapsed_ms = int(max(0.0, time.monotonic() - attempt_started_at) * 1000)
                            _log_mcp_event(
                                "info",
                                "mcp_attempt_success",
                                {
                                    **start_payload,
                                    "route": "mcp_sdk",
                                    "attempt_timeout_ms": int(per_attempt_timeout_s * 1000),
                                    "attempt_elapsed_ms": elapsed_ms,
                                    "timeout_allocation": timeout_allocation,
                                    "deadline_remaining_ms": int(
                                        max(0.0, deadline - time.monotonic()) * 1000
                                    ),
                                },
                            )
                            core_tracing.set_span_attributes(
                                attempt_span, {"mcp.route_status": "ok"}
                            )
                            core_tracing.set_span_attributes(route_span, {"mcp.route_status": "ok"})
                            core_tracing.set_span_attributes(
                                span,
                                {
                                    "mcp.selected_path": mcp_path,
                                    "mcp.status": "ok",
                                    "mcp.elapsed_s": time.monotonic() - started_at,
                                },
                            )
                            return result
                        except Exception as exc:  # noqa: BLE001
                            if isinstance(exc, ToolExecutionError):
                                error_text = str(exc)
                            else:
                                error_detail = "; ".join(_flatten_exception_messages(exc))
                                error_text = (
                                    "mcp_sdk_error:"
                                    f"phase=route_call;error_type={exc.__class__.__name__};{error_detail}"
                                )
                            error_code = _classify_tool_error(error_text)
                            error_phase = _extract_mcp_error_phase(error_text)
                            if error_text.startswith("mcp_tool_error"):
                                _log_mcp_event(
                                    "warning",
                                    "mcp_attempt_tool_error",
                                    {
                                        **start_payload,
                                        "error": error_text,
                                        "error_code": error_code,
                                        "error_phase": error_phase,
                                    },
                                )
                                core_tracing.set_span_attributes(
                                    attempt_span, {"mcp.route_status": "tool_error"}
                                )
                                core_tracing.set_span_attributes(
                                    route_span, {"mcp.route_status": "tool_error"}
                                )
                                _finalize_and_raise(error_text, "tool_error")
                            retryable = _is_retryable_mcp_error(error_text)
                            if retryable and attempt < max_retries:
                                _log_mcp_event(
                                    "info",
                                    "mcp_attempt_retrying",
                                    {
                                        **start_payload,
                                        "error": error_text,
                                        "error_code": error_code,
                                        "error_phase": error_phase,
                                        "retry_sleep_s": retry_sleep_s * float(attempt + 1),
                                        "timeout_allocation": timeout_allocation,
                                        "deadline_remaining_ms": int(
                                            max(0.0, deadline - time.monotonic()) * 1000
                                        ),
                                    },
                                )
                                core_tracing.set_span_attributes(
                                    attempt_span,
                                    {
                                        "mcp.route_status": "retrying",
                                        "mcp.route_error": error_text,
                                    },
                                )
                                core_tracing.set_span_attributes(
                                    route_span,
                                    {
                                        "mcp.route_status": "retrying",
                                        "mcp.route_error": error_text,
                                    },
                                )
                                if retry_sleep_s > 0:
                                    next_remaining_s = deadline - time.monotonic()
                                    if next_remaining_s <= 0:
                                        detail = f"mcp_call_timed_out_after_{timeout_s:.1f}s"
                                        _finalize_and_raise(f"mcp_sdk_timeout:{detail}", "timeout")
                                    sleep_s = min(
                                        retry_sleep_s * float(attempt + 1),
                                        max(0.0, next_remaining_s - 0.05),
                                    )
                                    if sleep_s > 0:
                                        time.sleep(sleep_s)
                                continue
                            attempt_elapsed_ms = int(
                                max(0.0, time.monotonic() - attempt_started_at) * 1000
                            )
                            failed_payload = {
                                **start_payload,
                                "error": error_text,
                                "error_code": error_code,
                                "error_phase": error_phase,
                                "retryable": retryable,
                                "attempt_timeout_ms": int(per_attempt_timeout_s * 1000),
                                "attempt_elapsed_ms": attempt_elapsed_ms,
                                "timeout_allocation": timeout_allocation,
                                "deadline_remaining_ms": int(
                                    max(0.0, deadline - time.monotonic()) * 1000
                                ),
                            }
                            core_tracing.set_span_attributes(
                                attempt_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": error_text,
                                },
                            )
                            core_tracing.set_span_attributes(
                                route_span,
                                {
                                    "mcp.route_status": "failed",
                                    "mcp.route_error": error_text,
                                },
                            )
                            _log_mcp_event("warning", "mcp_attempt_failed", failed_payload)
                            errors.append(f"{mcp_path}#{attempt + 1}:{error_text}")
                            break
        joined = " | ".join(errors) if errors else "no_route_errors_recorded"
        _finalize_and_raise(f"mcp_sdk_all_routes_failed:{joined}", "all_routes_failed")


def _call_mcp_tool_sdk(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    mode = _resolve_mcp_isolation_mode()
    if mode == "process":
        return _call_mcp_tool_sdk_process(mcp_url, tool_name, arguments, timeout_s)
    return _call_mcp_tool_sdk_inproc(mcp_url, tool_name, arguments, timeout_s)


def _call_mcp_tool_sdk_inproc(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"mcp_sdk_unavailable:{exc}") from exc

    streamable_kwargs = _streamable_http_client_kwargs(streamable_http_client, timeout_s)
    with core_tracing.start_span(
        "tool_registry.mcp_sdk_call",
        attributes={
            "mcp.tool_name": tool_name,
            "mcp.url": mcp_url,
            "mcp.timeout_s": timeout_s,
        },
    ) as sdk_span:
        sdk_started_at = time.monotonic()
        phase = "stream_open"

        async def _run_call() -> Any:
            nonlocal phase
            async with streamable_http_client(mcp_url, **streamable_kwargs) as (
                read_stream,
                write_stream,
                _session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    phase = "initialize"
                    with core_tracing.start_span(
                        "tool_registry.mcp_sdk_initialize",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.url": mcp_url,
                        },
                    ):
                        await session.initialize()
                    phase = "call_tool"
                    with core_tracing.start_span(
                        "tool_registry.mcp_sdk_call_tool",
                        attributes={
                            "mcp.tool_name": tool_name,
                            "mcp.url": mcp_url,
                        },
                    ):
                        return await session.call_tool(tool_name, arguments)

        try:
            result = asyncio.run(asyncio.wait_for(_run_call(), timeout=timeout_s))
        except TimeoutError as exc:
            elapsed_s = time.monotonic() - sdk_started_at
            detail = (
                f"phase={phase};mcp_call_timed_out_after_{timeout_s:.1f}s;elapsed_s={elapsed_s:.3f}"
            )
            core_tracing.set_span_attributes(
                sdk_span,
                {
                    "mcp.error": detail,
                    "mcp.phase": phase,
                    "mcp.elapsed_s": elapsed_s,
                },
            )
            raise ToolExecutionError(f"mcp_sdk_timeout:{detail}") from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.monotonic() - sdk_started_at
            detail = "; ".join(_flatten_exception_messages(exc))
            error_detail = f"phase={phase};error_type={exc.__class__.__name__};{detail}"
            core_tracing.set_span_attributes(
                sdk_span,
                {
                    "mcp.error": error_detail,
                    "mcp.phase": phase,
                    "mcp.elapsed_s": elapsed_s,
                },
            )
            raise ToolExecutionError(f"mcp_sdk_error:{error_detail}") from exc
        core_tracing.set_span_attributes(sdk_span, {"mcp.status": "ok"})
        return _extract_mcp_sdk_result(result)


def _mcp_process_entry(
    queue: Any, mcp_url: str, tool_name: str, arguments: Dict[str, Any], timeout_s: float
) -> None:
    try:
        result = _call_mcp_tool_sdk_inproc(mcp_url, tool_name, arguments, timeout_s)
        queue.put({"ok": result})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "traceback": traceback.format_exc(),
            }
        )


def _call_mcp_tool_sdk_process(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_mcp_process_entry,
        args=(queue, mcp_url, tool_name, arguments, timeout_s),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout_s + 2.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        detail = f"phase=process.join;mcp_call_timed_out_after_{timeout_s:.1f}s"
        raise ToolExecutionError(f"mcp_sdk_timeout:{detail}")

    outcome: Dict[str, Any] | None = None
    try:
        if not queue.empty():
            outcome = queue.get_nowait()
    except Exception:  # noqa: BLE001
        outcome = None
    finally:
        queue.close()

    if isinstance(outcome, dict):
        if isinstance(outcome.get("ok"), dict):
            return outcome["ok"]
        if isinstance(outcome.get("error"), str):
            detail = outcome["error"]
            error_type = outcome.get("error_type")
            if isinstance(error_type, str) and error_type.strip():
                detail = f"{detail};child_error_type={error_type}"
            child_traceback = outcome.get("traceback")
            if isinstance(child_traceback, str) and child_traceback.strip():
                LOGGER.warning(
                    "mcp_child_process_error_traceback tool=%s url=%s traceback=%s",
                    tool_name,
                    mcp_url,
                    child_traceback,
                )
            raise ToolExecutionError(detail)

    exit_code = process.exitcode
    raise ToolExecutionError(f"mcp_sdk_error:process_exit_code_{exit_code}")


def _flatten_exception_messages(exc: BaseException) -> List[str]:
    messages = [str(exc)]
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, (list, tuple)):
        for child in nested:
            if isinstance(child, BaseException):
                messages.extend(_flatten_exception_messages(child))
    deduped: List[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = message.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [exc.__class__.__name__]


def _extract_mcp_sdk_result(result: Any) -> Dict[str, Any]:
    is_error = getattr(result, "isError", False)
    if is_error:
        error_detail = _extract_mcp_error_detail(result)
        if error_detail:
            raise ToolExecutionError(f"mcp_tool_error:{error_detail}")
        raise ToolExecutionError("mcp_tool_error")

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return _normalize_mcp_structured_result(structured)

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

    raise ToolExecutionError("mcp_sdk_result_invalid")


def _normalize_mcp_structured_result(structured: Dict[str, Any]) -> Dict[str, Any]:
    # FastMCP wraps tool outputs as {"result": <tool_output>}.
    # Internal callers expect the raw tool_output dictionary.
    result_value = structured.get("result")
    if isinstance(result_value, dict):
        return result_value
    return structured


def _extract_mcp_error_detail(result: Any) -> str:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return json.dumps(structured, ensure_ascii=True)
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return " | ".join(parts)
    return ""


def _coding_agent_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ToolExecutionError("Missing goal")
    files = payload.get("files")
    if files is not None and not isinstance(files, list):
        raise ToolExecutionError("files must be a list of strings")
    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, str):
        raise ToolExecutionError("constraints must be a string")
    workspace_path = payload.get("workspace_path")
    if workspace_path is not None and not isinstance(workspace_path, str):
        raise ToolExecutionError("workspace_path must be a string")

    coder_url = os.getenv("CODER_API_URL", "http://coder:8000").rstrip("/")
    request_payload: Dict[str, Any] = {"goal": goal}
    if files:
        request_payload["files"] = files
    if constraints:
        request_payload["constraints"] = constraints

    response = _post_mcp_tool_call(
        coder_url,
        "generate_code",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("coder_response_invalid")
    file_entries = response.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        raise ToolExecutionError("coder_response_missing_files")

    written_paths: list[str] = []
    for entry in file_entries:
        if not isinstance(entry, dict):
            raise ToolExecutionError("coder_file_entry_invalid")
        path = entry.get("path")
        content = entry.get("content")
        if not isinstance(path, str) or not path.strip():
            raise ToolExecutionError("coder_file_entry_missing_path")
        if path.startswith("/") or path.startswith(".."):
            raise ToolExecutionError("coder_file_entry_invalid_path")
        if not isinstance(content, str):
            raise ToolExecutionError("coder_file_entry_missing_content")
        target_path = str(Path(workspace_path) / path) if workspace_path else path
        _write_workspace_text_file(
            {"path": target_path, "content": content}, default_filename="output.txt"
        )
        written_paths.append(target_path)

    return {"files": file_entries, "written_paths": written_paths}


def _build_plan_prompt(goal: str, constraints: Optional[str], max_steps: int) -> str:
    constraints_block = f"Constraints:\n{constraints}\n" if constraints else ""
    return (
        "You are a coding planner. Return ONLY JSON (no prose, no markdown).\n"
        "Output schema:\n"
        "{\n"
        '  "steps": [\n'
        '    {"title": "Short step title", "files": ["relative/path.ext"]}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        f"- Provide between 3 and {max_steps} steps.\n"
        "- Each step must list only the files it should create or modify.\n"
        "- File paths must be relative, no leading slash, no '..' segments.\n"
        "- Keep each step small: 5 files max per step.\n"
        "- Avoid large or binary assets; prefer small placeholder text or simple SVGs.\n"
        "- Do not include repos/, repositories/, or repo-name prefixes in file paths.\n"
        "- Keep the file list minimal but complete (include README, tests, Dockerfile if needed).\n"
        f"{constraints_block}"
        f"Goal:\n{goal}\n"
    )


def _render_plan_markdown(goal: str, steps: list[dict[str, Any]], statuses: list[bool]) -> str:
    lines = ["# IMPLEMENTATION_PLAN", "", f"Goal: {goal}", "", "## Steps"]
    for idx, step in enumerate(steps, start=1):
        status = "x" if statuses[idx - 1] else " "
        title = step.get("title", f"Step {idx}")
        files = step.get("files", [])
        files_text = ", ".join(files) if isinstance(files, list) else ""
        lines.append(f"- [{status}] Step {idx}: {title} (files: {files_text})")
    lines.append("")
    return "\n".join(lines)


def _coding_agent_autonomous(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ToolExecutionError("Missing goal")
    workspace_path = payload.get("workspace_path")
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        raise ToolExecutionError("Missing workspace_path")
    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, str):
        raise ToolExecutionError("constraints must be a string")
    max_steps = payload.get("max_steps") or 6
    if not isinstance(max_steps, int):
        max_steps = 6
    max_steps = max(3, min(12, max_steps))

    plan_prompt = _build_plan_prompt(goal, constraints, max_steps)
    response = provider.generate(plan_prompt)
    plan_json = _extract_json(response.content)
    if not plan_json:
        raise ToolExecutionError("plan_json_missing")
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"plan_json_invalid:{exc}") from exc
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ToolExecutionError("plan_steps_missing")
    steps = steps[:max_steps]
    statuses = [False for _ in steps]

    plan_path = f"{workspace_path.rstrip('/')}/IMPLEMENTATION_PLAN.md"
    plan_md = _render_plan_markdown(goal, steps, statuses)
    _write_workspace_text_file({"path": plan_path, "content": plan_md}, "IMPLEMENTATION_PLAN.md")

    coder_url = os.getenv("CODER_API_URL", "http://coder:8000").rstrip("/")
    written_paths: list[str] = []

    for idx, step in enumerate(steps, start=1):
        title = step.get("title", f"Step {idx}")
        files = step.get("files")
        if not isinstance(files, list) or not files:
            raise ToolExecutionError(f"plan_step_missing_files:{idx}")
        step_goal = (
            f"{goal}\n\nCurrent step {idx}: {title}\nOnly implement the files listed for this step."
        )
        request_payload: Dict[str, Any] = {"goal": step_goal, "files": files}
        if constraints:
            request_payload["constraints"] = constraints
        attempts = 0
        response: Dict[str, Any] | None = None
        last_error: str | None = None
        while attempts < 2 and response is None:
            attempts += 1
            try:
                response = _post_mcp_tool_call(
                    coder_url,
                    "generate_code",
                    request_payload,
                )
            except ToolExecutionError as exc:
                last_error = str(exc)
                if "invalid_json" not in last_error:
                    raise
                tighten = (
                    "STRICT JSON ONLY. Return a single JSON object. "
                    "Escape newlines as \\n and avoid unescaped quotes. "
                    "Keep output minimal. Avoid large strings or binary data."
                )
                if request_payload.get("constraints"):
                    request_payload["constraints"] = f"{request_payload['constraints']}\n{tighten}"
                else:
                    request_payload["constraints"] = tighten
                response = None
        if response is None:
            raise ToolExecutionError(last_error or "coder_invalid_json")
        if not isinstance(response, dict):
            raise ToolExecutionError("coder_response_invalid")
        file_entries = response.get("files")
        if not isinstance(file_entries, list) or not file_entries:
            raise ToolExecutionError("coder_response_missing_files")
        for entry in file_entries:
            if not isinstance(entry, dict):
                raise ToolExecutionError("coder_file_entry_invalid")
            path = entry.get("path")
            content = entry.get("content")
            if not isinstance(path, str) or not path.strip():
                raise ToolExecutionError("coder_file_entry_missing_path")
            if path.startswith("/") or path.startswith(".."):
                raise ToolExecutionError("coder_file_entry_invalid_path")
            if not isinstance(content, str):
                raise ToolExecutionError("coder_file_entry_missing_content")
            target_path = str(Path(workspace_path) / path)
            _write_workspace_text_file(
                {"path": target_path, "content": content}, default_filename="output.txt"
            )
            written_paths.append(target_path)
        statuses[idx - 1] = True
        plan_md = _render_plan_markdown(goal, steps, statuses)
        _write_workspace_text_file(
            {"path": plan_path, "content": plan_md}, "IMPLEMENTATION_PLAN.md"
        )

    return {
        "plan_path": plan_path,
        "steps_total": len(steps),
        "steps_completed": len(steps),
        "written_paths": written_paths,
    }


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
    return {"document_spec": _sanitize_document_spec(document_spec)}


def _sanitize_document_spec(document_spec: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(document_spec)
    blocks = sanitized.get("blocks")
    if isinstance(blocks, list):
        sanitized["blocks"] = _sanitize_document_blocks(blocks)
    return sanitized


def _sanitize_document_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block = dict(raw)
        block_type = block.get("type")

        if block_type == "spacer":
            # Spacer maps to an empty paragraph in DOCX; skip to avoid blank lines.
            continue

        if block_type in {"text", "paragraph", "heading", "optional_paragraph"}:
            text = block.get("text")
            if isinstance(text, str) and not text.strip():
                continue

        if block_type == "bullets":
            items = block.get("items")
            if isinstance(items, list):
                filtered_items = []
                for item in items:
                    if isinstance(item, str) and not item.strip():
                        continue
                    filtered_items.append(item)
                if not filtered_items:
                    continue
                block["items"] = filtered_items

        if block_type == "repeat":
            template = block.get("template")
            if isinstance(template, list):
                cleaned_template = _sanitize_document_blocks(template)
                if not cleaned_template:
                    continue
                block["template"] = cleaned_template

        cleaned.append(block)
    return cleaned


_RESUME_CONTEXT_KEYS = (
    "job_description",
    "candidate_resume",
    "target_role_name",
    "seniority_level",
)


def _llm_tailor_resume_text(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    memory = payload.get("memory")
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    response = _post_mcp_tool_call(
        tailor_url,
        "tailor_resume",
        {"job": job, "memory": memory},
    )
    resume_payload = response.get("tailored_resume") if isinstance(response, dict) else None
    if not isinstance(resume_payload, dict):
        raise ToolExecutionError("tailored_resume must be an object")
    return {"tailored_resume": resume_payload}


def _select_job_context_from_memory(memory: Any) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    memory_entries = memory.get("job_context")
    if not isinstance(memory_entries, list) or not memory_entries:
        return {}
    normalized: list[Dict[str, Any]] = []
    for entry in memory_entries:
        if not isinstance(entry, dict):
            continue
        payload = entry.get("payload")
        if isinstance(payload, dict):
            normalized_entry = dict(payload)
            for meta_key in ("_memory_key", "_memory_updated_at"):
                if meta_key in entry:
                    normalized_entry[meta_key] = entry[meta_key]
            normalized.append(normalized_entry)
        else:
            normalized.append(dict(entry))
    if not normalized:
        return {}

    def score(entry: Dict[str, Any]) -> tuple[int, int, str]:
        resume = entry.get("candidate_resume")
        resume_len = len(resume.strip()) if isinstance(resume, str) else 0
        job_desc = entry.get("job_description")
        job_desc_len = len(job_desc.strip()) if isinstance(job_desc, str) else 0
        updated_at = str(entry.get("_memory_updated_at") or "")
        return (resume_len, job_desc_len, updated_at)

    best = max(normalized, key=score)
    return {k: v for k, v in best.items() if isinstance(k, str) and not k.startswith("_")}


def _merge_resume_job_context(
    job: Dict[str, Any], memory_context: Dict[str, Any]
) -> Dict[str, Any]:
    context_json = job.get("context_json")
    if not isinstance(context_json, dict):
        context_json = {}
    merged_context = dict(context_json)
    merged_context.update(memory_context)
    for key in _RESUME_CONTEXT_KEYS:
        if _is_missing_value(merged_context.get(key)) and isinstance(job.get(key), str):
            merged_context[key] = job.get(key)
    return merged_context


def _build_resume_job_payload(job: Dict[str, Any] | None, memory: Any) -> Dict[str, Any]:
    job_payload: Dict[str, Any] = dict(job) if isinstance(job, dict) else {}
    memory_context = _select_job_context_from_memory(memory)
    merged_context = _merge_resume_job_context(job_payload, memory_context)
    if merged_context:
        job_payload["context_json"] = merged_context
        for key in _RESUME_CONTEXT_KEYS:
            value = merged_context.get(key)
            if isinstance(value, str) and value.strip():
                job_payload[key] = value
    return job_payload


def _llm_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    request_payload = {
        "tailored_resume": payload.get("tailored_resume"),
        "tailored_text": payload.get("tailored_text"),
        "job": payload.get("job") or {},
        "memory": payload.get("memory"),
    }
    response = _post_mcp_tool_call(
        tailor_url,
        "improve_resume",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("tailor_response_invalid")
    return response


def _llm_iterative_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    request_payload = {
        "tailored_resume": payload.get("tailored_resume"),
        "tailored_text": payload.get("tailored_text"),
        "job": payload.get("job") or {},
        "memory": payload.get("memory"),
        "min_alignment_score": payload.get("min_alignment_score", 85),
        "max_iterations": payload.get("max_iterations", 2),
    }
    response = _post_mcp_tool_call(
        tailor_url,
        "improve_iterative",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("tailor_response_invalid")
    return response


def _parse_target_pages(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if re.fullmatch(r"\d+", trimmed):
            parsed = int(trimmed)
        else:
            match = re.search(r"\b([12])\s*page(?:s)?\b", trimmed.lower())
            if match is None:
                match = re.search(r"\bpage(?:s)?\s*([12])\b", trimmed.lower())
            if match is None:
                return None
            parsed = int(match.group(1))
    else:
        return None
    if parsed in {1, 2}:
        return parsed
    return None


def _resolve_target_pages(payload: Dict[str, Any], job: Any) -> int | None:
    candidates: list[Any] = [payload.get("target_pages"), payload.get("page_count")]
    if isinstance(job, dict):
        candidates.extend([job.get("target_pages"), job.get("page_count")])
        context_json = job.get("context_json")
        if isinstance(context_json, dict):
            candidates.extend([context_json.get("target_pages"), context_json.get("page_count")])
    for candidate in candidates:
        parsed = _parse_target_pages(candidate)
        if parsed is not None:
            return parsed
    return None


def _trim_non_empty_strings(items: Any, max_items: int) -> list[str]:
    if not isinstance(items, list):
        return []
    if max_items < 1:
        return []
    normalized = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    return normalized[:max_items]


def _trim_summary_sentences(text: str, max_sentences: int) -> str:
    cleaned = text.strip()
    if max_sentences < 1 or not cleaned:
        return cleaned
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(parts) <= max_sentences:
        return cleaned
    return " ".join(parts[:max_sentences]).strip()


def _apply_resume_target_pages_policy(
    resume_doc_spec: Dict[str, Any], target_pages: int | None
) -> None:
    if target_pages not in {1, 2}:
        return

    page = resume_doc_spec.get("page")
    if not isinstance(page, dict):
        page = {}
        resume_doc_spec["page"] = page
    margins_in = page.get("margins_in")
    if not isinstance(margins_in, dict):
        margins_in = {}
        page["margins_in"] = margins_in

    defaults = resume_doc_spec.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
        resume_doc_spec["defaults"] = defaults

    if target_pages == 1:
        margins_in.update({"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45})
        defaults["font_size_pt"] = 10.5
        defaults["line_spacing"] = 1.0
        skills_group_limit = 5
        first_role_bullet_limit = 4
        other_role_bullet_limit = 2
        experience_group_bullet_limit = 2
        summary_sentence_limit = 2
        aux_bullet_limit = 2
        total_experience_bullets_limit = 9
        total_aux_bullets_limit = 2
    else:
        # Keep two-page resumes inside budget without collapsing to one-page density.
        margins_in.update({"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45})
        defaults["font_size_pt"] = 10.75
        defaults["line_spacing"] = 1.0
        skills_group_limit = 6
        first_role_bullet_limit = 6
        other_role_bullet_limit = 3
        experience_group_bullet_limit = 3
        summary_sentence_limit = 3
        aux_bullet_limit = 3
        total_experience_bullets_limit = 16
        total_aux_bullets_limit = 3

    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    current_section = ""
    role_index = 0
    total_experience_bullets = 0
    total_aux_bullets = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "section_heading":
            heading = block.get("text")
            current_section = heading.strip().upper() if isinstance(heading, str) else ""
            continue
        if block_type == "definition_list" and current_section == "SKILLS":
            items = block.get("items")
            if isinstance(items, list):
                block["items"] = [item for item in items if isinstance(item, dict)][
                    :skills_group_limit
                ]
            continue
        if block_type == "paragraph" and current_section == "SUMMARY":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                block["text"] = _trim_summary_sentences(text, summary_sentence_limit)
            continue
        if block_type == "role" and current_section == "EXPERIENCE":
            role_index += 1
            bullets = block.get("bullets")
            if isinstance(bullets, list):
                max_items = first_role_bullet_limit if role_index == 1 else other_role_bullet_limit
                trimmed = _trim_non_empty_strings(bullets, max_items)
                remaining = max(0, total_experience_bullets_limit - total_experience_bullets)
                block["bullets"] = trimmed[:remaining]
                total_experience_bullets += len(block["bullets"])
            continue
        if block_type == "bullets":
            items = block.get("items")
            if current_section == "EXPERIENCE":
                trimmed = _trim_non_empty_strings(items, experience_group_bullet_limit)
                remaining = max(0, total_experience_bullets_limit - total_experience_bullets)
                block["items"] = trimmed[:remaining]
                total_experience_bullets += len(block["items"])
            elif current_section in {"OPEN SOURCE", "PROJECTS", "OPEN SOURCE (SELECTED)"}:
                trimmed = _trim_non_empty_strings(items, aux_bullet_limit)
                remaining = max(0, total_aux_bullets_limit - total_aux_bullets)
                block["items"] = trimmed[:remaining]
                total_aux_bullets += len(block["items"])


def _llm_generate_resume_doc_spec_from_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    tailored_resume = payload.get("tailored_resume")
    job = payload.get("job")
    target_pages = _resolve_target_pages(payload, job)
    if isinstance(tailored_text, str) and tailored_text.strip() and tailored_resume is None:
        try:
            parsed = json.loads(tailored_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tailored_resume = parsed
    if isinstance(tailored_resume, dict):
        _ensure_required_resume_sections(tailored_resume)
        prompt = prompts.resume_doc_spec_prompt(job or {}, tailored_resume=tailored_resume)
    else:
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
    _apply_resume_target_pages_policy(resume_doc_spec, target_pages)
    _normalize_skills_definition_separators(resume_doc_spec)
    _ensure_certifications_section_content(
        resume_doc_spec,
        tailored_resume=tailored_resume if isinstance(tailored_resume, dict) else None,
        tailored_text=tailored_text if isinstance(tailored_text, str) else None,
        candidate_resume_text=_extract_candidate_resume_text_from_job(job),
    )
    if isinstance(tailored_text, str) and tailored_text.strip():
        _fill_missing_dates_from_text(resume_doc_spec, tailored_text)

    # Validate the generated ResumeDocSpec before returning it.
    from libs.tools.resume_doc_spec_validate import _resume_doc_spec_validate

    validation = _resume_doc_spec_validate({"resume_doc_spec": resume_doc_spec, "strict": True})
    if not validation.get("valid", False):
        errors = validation.get("errors") or []
        raise ToolExecutionError(f"resume_doc_spec_validation_failed:{errors}")
    return {"resume_doc_spec": resume_doc_spec, "validation": validation}


def _normalize_skills_definition_separators(resume_doc_spec: Dict[str, Any]) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    in_skills_section = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "section_heading":
            heading = block.get("text")
            heading_text = heading.strip().upper() if isinstance(heading, str) else ""
            in_skills_section = heading_text in {
                "SKILLS",
                "CORE SKILLS",
                "TECHNICAL SKILLS",
            }
            continue
        if not in_skills_section or block_type != "definition_list":
            continue

        items = block.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            definition = item.get("definition")
            if not isinstance(definition, str) or ";" not in definition:
                continue
            normalized_items = [part.strip() for part in definition.split(";") if part.strip()]
            item["definition"] = ", ".join(normalized_items)


def _llm_generate_cover_letter_from_resume(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    tailored_resume = payload.get("tailored_resume")
    if isinstance(tailored_text, str) and tailored_text.strip() and tailored_resume is None:
        try:
            parsed = json.loads(tailored_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tailored_resume = parsed

    resume_source: Dict[str, Any]
    header: Dict[str, Any] = {}
    if isinstance(tailored_resume, dict):
        _ensure_required_resume_sections(tailored_resume)
        resume_source = tailored_resume
        header_value = tailored_resume.get("header")
        if isinstance(header_value, dict):
            header = header_value
    else:
        if not isinstance(tailored_text, str) or not tailored_text.strip():
            raise ToolExecutionError("tailored_resume or tailored_text must be provided")
        resume_source = {"raw_text": tailored_text}

    job_value = payload.get("job")
    job_payload = _build_resume_job_payload(
        job_value if isinstance(job_value, dict) else {},
        payload.get("memory"),
    )
    context_json = job_payload.get("context_json", {})
    if not isinstance(context_json, dict):
        context_json = {}

    prompt = prompts.cover_letter_from_resume_prompt(resume_source, job=job_payload)
    response = provider.generate(prompt)
    json_text = _extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        cover_letter = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(cover_letter, dict):
        raise ToolExecutionError("cover_letter must be an object")

    header_links = header.get("links")
    if not isinstance(header_links, dict):
        header_links = {}

    job_description = context_json.get("job_description")
    if not isinstance(job_description, str):
        job_description = ""
    candidate_resume = context_json.get("candidate_resume")
    if not isinstance(candidate_resume, str):
        candidate_resume = ""

    if _is_missing_value(cover_letter.get("full_name")):
        full_name = ""
        if isinstance(header.get("name"), str) and header.get("name", "").strip():
            full_name = header["name"].strip()
        if not full_name:
            full_name = _derive_candidate_name_from_texts(
                tailored_text, candidate_resume, job_description
            )
        if full_name:
            cover_letter["full_name"] = full_name

    if _is_missing_value(cover_letter.get("location")) and isinstance(header.get("location"), str):
        cover_letter["location"] = header.get("location", "").strip()
    if _is_missing_value(cover_letter.get("phone")) and isinstance(header.get("phone"), str):
        cover_letter["phone"] = header.get("phone", "").strip()
    if _is_missing_value(cover_letter.get("email")) and isinstance(header.get("email"), str):
        cover_letter["email"] = header.get("email", "").strip()

    if _is_missing_value(cover_letter.get("linkedin_url")):
        linkedin = header_links.get("linkedin")
        if isinstance(linkedin, str) and linkedin.strip():
            cover_letter["linkedin_url"] = linkedin.strip()
    if _is_missing_value(cover_letter.get("github_url")):
        github = header_links.get("github")
        if isinstance(github, str) and github.strip():
            cover_letter["github_url"] = github.strip()

    if _is_missing_value(cover_letter.get("role_title")):
        role_hint = (
            context_json.get("target_role_name")
            or job_payload.get("target_role_name")
            or _derive_role_name_from_jd(job_description)
        )
        if isinstance(role_hint, str) and role_hint.strip():
            cover_letter["role_title"] = role_hint.strip()

    if _is_missing_value(cover_letter.get("company")):
        company_hint = context_json.get("company_name")
        if not isinstance(company_hint, str) or not company_hint.strip():
            company_hint = _derive_company_name_from_jd(job_description)
        if isinstance(company_hint, str) and company_hint.strip():
            cover_letter["company"] = company_hint.strip()

    if _is_missing_value(cover_letter.get("recipient_line")):
        cover_letter["recipient_line"] = "Hiring Team"
    if _is_missing_value(cover_letter.get("salutation")):
        cover_letter["salutation"] = "Dear Hiring Team,"
    if _is_missing_value(cover_letter.get("closing")):
        cover_letter["closing"] = "Sincerely,"

    if _is_missing_value(cover_letter.get("full_name")):
        raise ToolExecutionError("cover_letter_missing_full_name")
    if _is_missing_value(cover_letter.get("body")):
        raise ToolExecutionError("cover_letter_missing_body")

    return {"cover_letter": cover_letter}


def _llm_generate_coverletter_doc_spec_from_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    cover_letter_result = _llm_generate_cover_letter_from_resume(payload, provider)
    cover_letter = cover_letter_result.get("cover_letter")
    if not isinstance(cover_letter, dict):
        raise ToolExecutionError("cover_letter_generation_failed")

    job_value = payload.get("job")
    job_payload = _build_resume_job_payload(
        job_value if isinstance(job_value, dict) else {},
        payload.get("memory"),
    )
    context_json = job_payload.get("context_json")
    if not isinstance(context_json, dict):
        context_json = {}

    date_text = payload.get("today_pretty")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = payload.get("today")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = context_json.get("today_pretty")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = context_json.get("today")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = "Today"

    coverletter_doc_spec = _build_coverletter_doc_spec(cover_letter, date_text.strip())
    validation = _validate_coverletter_doc_spec(coverletter_doc_spec, strict=True)
    if not validation.get("valid", False):
        errors = validation.get("errors") or []
        raise ToolExecutionError(f"coverletter_doc_spec_validation_failed:{errors}")

    return {"coverletter_doc_spec": coverletter_doc_spec, "validation": validation}


def _build_coverletter_doc_spec(cover_letter: Dict[str, Any], date_text: str) -> Dict[str, Any]:
    full_name = str(cover_letter.get("full_name") or "").strip()
    location = str(cover_letter.get("location") or "").strip()
    phone = str(cover_letter.get("phone") or "").strip()
    email = str(cover_letter.get("email") or "").strip()
    linkedin_url = str(cover_letter.get("linkedin_url") or "").strip()
    github_url = str(cover_letter.get("github_url") or "").strip()
    portfolio_url = str(cover_letter.get("portfolio_url") or "").strip()
    company = str(cover_letter.get("company") or "").strip()
    hiring_manager = str(cover_letter.get("hiring_manager") or "").strip()
    recipient_line = str(cover_letter.get("recipient_line") or "Hiring Team").strip()
    role_title = str(cover_letter.get("role_title") or "").strip()
    salutation = str(cover_letter.get("salutation") or "Dear Hiring Team,").strip()
    closing = str(cover_letter.get("closing") or "Sincerely,").strip()
    body_text = str(cover_letter.get("body") or "").strip()

    header_blocks: List[Dict[str, Any]] = []
    if full_name:
        header_blocks.append({"type": "text", "style": "name", "text": full_name})
    if location:
        header_blocks.append({"type": "text", "style": "contact", "text": location})
    phone_email = " | ".join([part for part in (phone, email) if part])
    if phone_email:
        header_blocks.append({"type": "text", "style": "contact", "text": phone_email})
    links_parts: List[str] = []
    if linkedin_url:
        links_parts.append(f"LinkedIn: {linkedin_url}")
    if github_url:
        links_parts.append(f"GitHub: {github_url}")
    if portfolio_url:
        links_parts.append(f"Portfolio: {portfolio_url}")
    links_line = " | ".join(links_parts)
    if links_line:
        header_blocks.append({"type": "text", "style": "contact", "text": links_line})

    content: List[Dict[str, Any]] = []
    content.append({"type": "header", "align": "left", "blocks": header_blocks})
    content.append({"type": "paragraph", "style": "cover_letter_date", "text": date_text})
    if recipient_line:
        content.append(
            {"type": "paragraph", "style": "cover_letter_recipient", "text": recipient_line}
        )
    if hiring_manager:
        content.append(
            {"type": "paragraph", "style": "cover_letter_recipient", "text": hiring_manager}
        )
    if company:
        content.append({"type": "paragraph", "style": "cover_letter_recipient", "text": company})
    if salutation:
        content.append(
            {"type": "paragraph", "style": "cover_letter_salutation", "text": salutation}
        )

    body_paragraphs = _split_cover_letter_paragraphs(body_text)
    for paragraph in body_paragraphs:
        content.append({"type": "paragraph", "style": "cover_letter_body", "text": paragraph})

    if closing:
        content.append({"type": "paragraph", "style": "cover_letter_closing", "text": closing})
    if full_name:
        content.append({"type": "paragraph", "style": "cover_letter_signature", "text": full_name})

    title_parts = [full_name or "Candidate", "Cover Letter"]
    if role_title:
        title_parts.append(role_title)
    if company:
        title_parts.append(company)
    title = " - ".join(title_parts)

    return {
        "schema_version": "1.0",
        "doc_type": "cover_letter",
        "title": title,
        "page": {
            "size": "LETTER",
            "margins_in": {"top": 0.8, "right": 0.8, "bottom": 0.8, "left": 0.8},
        },
        "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.15},
        "content": content,
        "styles": {
            "name": {"bold": True, "size_pt": 20},
            "contact": {"size_pt": 11},
            "cover_letter_date": {"space_before_pt": 10, "space_after_pt": 8},
            "cover_letter_recipient": {"space_after_pt": 2},
            "cover_letter_salutation": {"space_before_pt": 8, "space_after_pt": 8},
            "cover_letter_body": {"space_after_pt": 8},
            "cover_letter_closing": {"space_before_pt": 8, "space_after_pt": 2},
        },
    }


def _split_cover_letter_paragraphs(body_text: str) -> List[str]:
    if not isinstance(body_text, str):
        return []
    text = body_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def _validate_coverletter_doc_spec(spec: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    from libs.tools.resume_doc_spec_validate import _require_number, _require_str, _result, err

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    _require_str(spec, "schema_version", "/schema_version", errors)
    _require_str(spec, "doc_type", "/doc_type", errors)
    _require_str(spec, "title", "/title", errors)

    page = spec.get("page")
    if not isinstance(page, dict):
        errors.append(err("/page", "page must be an object"))
    else:
        _require_str(page, "size", "/page/size", errors)
        margins = page.get("margins_in")
        if not isinstance(margins, dict):
            errors.append(err("/page/margins_in", "margins_in must be an object"))
        else:
            for key in ("top", "right", "bottom", "left"):
                value = margins.get(key)
                if not isinstance(value, (int, float)):
                    errors.append(err(f"/page/margins_in/{key}", "must be a number"))

    defaults = spec.get("defaults")
    if not isinstance(defaults, dict):
        errors.append(err("/defaults", "defaults must be an object"))
    else:
        _require_str(defaults, "font_family", "/defaults/font_family", errors)
        _require_number(defaults, "font_size_pt", "/defaults/font_size_pt", errors)
        _require_number(defaults, "line_spacing", "/defaults/line_spacing", errors)

    styles = spec.get("styles")
    if not isinstance(styles, dict):
        errors.append(err("/styles", "styles must be an object"))

    content = spec.get("content")
    if not isinstance(content, list):
        errors.append(err("/content", "content must be an array"))
        content = []

    allowed_types = {"header", "paragraph"}
    for idx, block in enumerate(content):
        path = f"/content/{idx}"
        if not isinstance(block, dict):
            errors.append(err(path, "content item must be an object"))
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str):
            errors.append(err(f"{path}/type", "type must be a string"))
            continue
        if block_type not in allowed_types:
            message = f"unsupported content type: {block_type}"
            if strict:
                errors.append(err(f"{path}/type", message))
            else:
                warnings.append(err(f"{path}/type", message))
            continue
        if block_type == "header":
            blocks = block.get("blocks")
            if not isinstance(blocks, list):
                errors.append(err(f"{path}/blocks", "blocks must be an array"))
                continue
            if not blocks:
                errors.append(err(f"{path}/blocks", "blocks must not be empty"))
                continue
            for b_idx, b in enumerate(blocks):
                b_path = f"{path}/blocks/{b_idx}"
                if not isinstance(b, dict):
                    errors.append(err(b_path, "block must be an object"))
                    continue
                _require_str(b, "type", f"{b_path}/type", errors)
                _require_str(b, "text", f"{b_path}/text", errors)
        elif block_type == "paragraph":
            _require_str(block, "text", f"{path}/text", errors)

    return _result(len(errors) == 0, errors, warnings, len(content))


def _llm_generate_resume_doc_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    job = payload.get("job")
    tailored_resume = payload.get("tailored_resume")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    target_pages = _resolve_target_pages(payload, job)
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
    _apply_resume_target_pages_policy(resume_doc_spec, target_pages)
    _ensure_certifications_section_content(
        resume_doc_spec,
        tailored_resume=tailored_resume if isinstance(tailored_resume, dict) else None,
        tailored_text=None,
        candidate_resume_text=_extract_candidate_resume_text_from_job(job),
    )
    return {"resume_doc_spec": resume_doc_spec}


def _ensure_required_resume_sections(content: Any) -> None:
    if isinstance(content, dict):
        required_keys = [
            "schema_version",
            "header",
            "summary",
            "skills",
            "experience",
            "education",
            "certifications",
        ]
        missing = [key for key in required_keys if key not in content]
        if missing:
            raise ToolExecutionError(f"tailored_resume_missing_fields:{','.join(missing)}")
        schema_version = content.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise ToolExecutionError("tailored_resume_invalid_schema_version")
        if schema_version.strip() != "1.0":
            raise ToolExecutionError("tailored_resume_invalid_schema_version")
        header = content.get("header")
        if not isinstance(header, dict):
            raise ToolExecutionError("tailored_resume_invalid_header")
        for field in ("name", "title", "location", "phone", "email"):
            value = header.get(field)
            if _is_missing_value(value):
                raise ToolExecutionError(f"tailored_resume_missing_header:{field}")
        links = header.get("links")
        if links is not None and not isinstance(links, dict):
            raise ToolExecutionError("tailored_resume_invalid_header:links")
        if not isinstance(content.get("summary"), str) or not content["summary"].strip():
            raise ToolExecutionError("tailored_resume_invalid_summary")
        for list_key in ("skills", "experience", "education", "certifications"):
            if not isinstance(content.get(list_key), list):
                raise ToolExecutionError(f"tailored_resume_invalid_{list_key}")
        for skill in content.get("skills", []):
            if not isinstance(skill, dict):
                raise ToolExecutionError("tailored_resume_invalid_skills")
            if _is_missing_value(skill.get("group_name")):
                raise ToolExecutionError("tailored_resume_invalid_skills")
            items = skill.get("items")
            if not isinstance(items, list) or not items:
                raise ToolExecutionError("tailored_resume_invalid_skills")
            for item in items:
                if _is_missing_value(item):
                    raise ToolExecutionError("tailored_resume_invalid_skills")
        experiences = content.get("experience", [])
        if not experiences:
            raise ToolExecutionError("tailored_resume_missing_experience")
        for idx, role in enumerate(experiences):
            if not isinstance(role, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}")
            for field in ("company", "title", "location", "dates"):
                if _is_missing_value(role.get(field)):
                    raise ToolExecutionError(f"tailored_resume_missing_experience:{idx}.{field}")
            bullets = role.get("bullets")
            groups = role.get("groups")
            has_bullets = isinstance(bullets, list) and len(bullets) > 0
            has_groups = isinstance(groups, list) and len(groups) > 0
            if not (has_bullets or has_groups):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.bullets")
            if bullets is not None and not isinstance(bullets, list):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.bullets")
            if isinstance(bullets, list):
                for bullet in bullets:
                    if _is_missing_value(bullet):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.bullets"
                        )
            if groups is not None and not isinstance(groups, list):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.groups")
            if isinstance(groups, list):
                for g_idx, group in enumerate(groups):
                    if not isinstance(group, dict):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}]"
                        )
                    if _is_missing_value(group.get("heading")):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].heading"
                        )
                    group_bullets = group.get("bullets")
                    if not isinstance(group_bullets, list) or not group_bullets:
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                        )
                    for bullet in group_bullets:
                        if _is_missing_value(bullet):
                            raise ToolExecutionError(
                                f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                            )
        education = content.get("education", [])
        for idx, edu in enumerate(education):
            if not isinstance(edu, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_education:{idx}")
            for field in ("degree", "school", "location", "dates"):
                if _is_missing_value(edu.get(field)):
                    raise ToolExecutionError(f"tailored_resume_missing_education:{idx}.{field}")
        certifications = content.get("certifications", [])
        for idx, cert in enumerate(certifications):
            if not isinstance(cert, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_certifications:{idx}")
            for field in ("name", "issuer", "year"):
                value = cert.get(field)
                if field == "year" and isinstance(value, int):
                    continue
                if _is_missing_value(value):
                    raise ToolExecutionError(
                        f"tailored_resume_missing_certifications:{idx}.{field}"
                    )
        return
    if not isinstance(content, str):
        raise ToolExecutionError("tailored_resume_invalid_type")
    required = [
        "SUMMARY",
        "SKILLS",
        "EXPERIENCE",
        "EDUCATION",
        "CERTIFICATIONS",
    ]
    missing = [section for section in required if section not in content]
    if not missing:
        return
    fallback_headings = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS"]
    if _has_required_headings(content, fallback_headings):
        return
    raise ToolExecutionError(f"tailored_text_missing_sections:{','.join(missing)}")


def _has_required_headings(text: str, headings: list[str]) -> bool:
    for heading in headings:
        pattern = rf"(?m)^\\s*{re.escape(heading)}\\s*$"
        if not re.search(pattern, text):
            return False
    return True


def _fill_missing_dates_from_text(resume_doc_spec: Dict[str, Any], tailored_text: str) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return
    experience_dates = _extract_dates_from_section(
        tailored_text, start_heading="EXPERIENCE", end_heading="EDUCATION"
    )
    education_dates = _extract_dates_from_section(
        tailored_text, start_heading="EDUCATION", end_heading="CERTIFICATIONS"
    )
    exp_iter = iter(experience_dates)
    edu_iter = iter(education_dates)
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "role":
            dates = block.get("dates")
            if not isinstance(dates, str) or not dates.strip():
                next_date = next(exp_iter, None)
                if next_date:
                    block["dates"] = next_date
        if block_type == "education":
            dates = block.get("dates")
            if not isinstance(dates, str) or not dates.strip():
                next_date = next(edu_iter, None)
                if next_date:
                    block["dates"] = next_date


def _ensure_certifications_section_content(
    resume_doc_spec: Dict[str, Any],
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    cert_heading_idx = _find_section_heading_block_index(
        content, {"CERTIFICATIONS", "CERTIFICATION"}
    )
    if cert_heading_idx is None:
        return

    cert_start = cert_heading_idx + 1
    cert_end = _find_next_section_heading_block_index(content, cert_start)
    cert_section = content[cert_start:cert_end]
    existing_items = _extract_non_empty_cert_items(cert_section)
    if existing_items:
        return

    fallback_items = _collect_fallback_certification_lines(
        tailored_resume=tailored_resume,
        tailored_text=tailored_text,
        candidate_resume_text=candidate_resume_text,
    )
    if fallback_items:
        for block in cert_section:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "bullets":
                continue
            items = block.get("items")
            if isinstance(items, list):
                block["items"] = fallback_items
                return
        content.insert(cert_start, {"type": "bullets", "items": fallback_items})
        return

    # If there is no cert content and no fallback, remove the empty section.
    if _cert_section_is_removable(cert_section):
        del content[cert_heading_idx:cert_end]


def _find_section_heading_block_index(
    content: List[Dict[str, Any]],
    headings: set[str],
    start: int = 0,
) -> int | None:
    for idx in range(start, len(content)):
        block = content[idx]
        if not isinstance(block, dict):
            continue
        if block.get("type") != "section_heading":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        if text.strip().upper() in headings:
            return idx
    return None


def _find_next_section_heading_block_index(content: List[Dict[str, Any]], start: int) -> int:
    for idx in range(start, len(content)):
        block = content[idx]
        if isinstance(block, dict) and block.get("type") == "section_heading":
            return idx
    return len(content)


def _extract_non_empty_cert_items(blocks: List[Any]) -> list[str]:
    items: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "bullets":
            values = block.get("items")
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    items.append(value.strip())
        elif block_type == "paragraph":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                items.append(text.strip())
    return items


def _collect_fallback_certification_lines(
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> list[str]:
    collected: list[str] = []
    if isinstance(tailored_resume, dict):
        collected.extend(_certification_lines_from_tailored_resume(tailored_resume))
    if not collected and isinstance(tailored_text, str) and tailored_text.strip():
        collected.extend(_certification_lines_from_text(tailored_text))
    if not collected and isinstance(candidate_resume_text, str) and candidate_resume_text.strip():
        collected.extend(_certification_lines_from_text(candidate_resume_text))

    deduped: list[str] = []
    seen: set[str] = set()
    for line in collected:
        normalized = line.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _certification_lines_from_tailored_resume(tailored_resume: Dict[str, Any]) -> list[str]:
    certifications = tailored_resume.get("certifications")
    if not isinstance(certifications, list):
        return []

    lines: list[str] = []
    for cert in certifications:
        if isinstance(cert, str):
            text = cert.strip()
            if text:
                lines.append(text)
            continue
        if not isinstance(cert, dict):
            continue

        name = cert.get("name")
        issuer = cert.get("issuer")
        year = cert.get("year")
        url = (
            cert.get("url")
            or cert.get("credential_url")
            or cert.get("public_url")
            or cert.get("link")
        )
        if not isinstance(name, str) or not name.strip():
            continue

        line = name.strip()
        if isinstance(issuer, str) and issuer.strip():
            line = f"{line} - {issuer.strip()}"
        if isinstance(year, int):
            line = f"{line} ({year})"
        elif isinstance(year, str) and year.strip():
            line = f"{line} ({year.strip()})"
        if isinstance(url, str) and url.strip():
            line = f"{line} | {url.strip()}"
        lines.append(line)

    return lines


def _certification_lines_from_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = _find_heading_index(lines, "CERTIFICATIONS")
    if start_idx is None:
        start_idx = _find_heading_index(lines, "CERTIFICATION")
    if start_idx is None:
        return []

    end_headings = {
        "OPEN SOURCE",
        "OPEN SOURCE (SELECTED)",
        "PROJECTS",
        "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "SUMMARY",
        "SKILLS",
        "CORE SKILLS",
    }

    cert_lines: list[str] = []
    for raw in lines[start_idx + 1 :]:
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate.upper() in end_headings:
            break
        normalized = re.sub(r"^[*•\-]\s*", "", candidate).strip()
        if normalized:
            cert_lines.append(normalized)
    return cert_lines


def _cert_section_is_removable(blocks: List[Any]) -> bool:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "bullets":
            items = block.get("items")
            if isinstance(items, list) and any(
                isinstance(item, str) and item.strip() for item in items
            ):
                return False
            continue
        if block_type == "paragraph":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return False
            continue
        return False
    return True


def _extract_candidate_resume_text_from_job(job_payload: Any) -> str | None:
    if not isinstance(job_payload, dict):
        return None
    direct = job_payload.get("candidate_resume")
    if isinstance(direct, str) and direct.strip():
        return direct
    context = job_payload.get("context_json")
    if not isinstance(context, dict):
        return None
    candidate_resume = context.get("candidate_resume")
    if isinstance(candidate_resume, str) and candidate_resume.strip():
        return candidate_resume
    return None


def _extract_dates_from_section(text: str, start_heading: str, end_heading: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = _find_heading_index(lines, start_heading)
    if start_idx is None:
        return []
    end_idx = _find_heading_index(lines, end_heading, start=start_idx + 1)
    section = lines[start_idx + 1 : end_idx] if end_idx is not None else lines[start_idx + 1 :]
    dates: list[str] = []
    for line in section:
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        candidate = parts[-1]
        if not re.search(r"\\d{4}", candidate):
            continue
        dates.append(candidate)
    return dates


def _find_heading_index(lines: list[str], heading: str, start: int = 0) -> int | None:
    target = heading.strip().lower()
    for idx in range(start, len(lines)):
        if lines[idx].strip().lower() == target:
            return idx
    return None


def _is_missing_value(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in {"unknown", "n/a", "na", "none"}:
        return True
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return True
    return False


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
    return {"document_spec": _sanitize_document_spec(document_spec)}


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
    return {"document_spec": _sanitize_document_spec(improved_spec)}


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
