from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from .llm_provider import LLMProvider
from .models import RiskLevel, ToolCall, ToolSpec


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
            input=payload,
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
                    "and optional 'path' (relative filename)."
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
            ),
            handler=_file_write_artifact,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_text",
                description="Write text content to a file under /shared/artifacts",
                usage_guidance=(
                    "Use to write text to /shared/artifacts. Provide 'content' (required) "
                    "and optional 'path' (relative filename)."
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
            ),
            handler=_file_write_text,
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
            ),
            handler=_file_read_text,
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
            ),
            handler=_sleep,
        )
    )

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
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    timeout_s=_resolve_llm_timeout_s(llm_provider),
                    risk_level=RiskLevel.high,
                ),
                handler=lambda payload, provider=llm_provider: _llm_generate(payload, provider),
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



def _file_write_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    candidate = _safe_artifact_path(path, "artifact.txt")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_write_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    candidate = _safe_artifact_path(path, "output.txt")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_artifact_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}



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
