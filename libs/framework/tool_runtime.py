from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from libs.core.models import ToolCall, ToolSpec


class ToolExecutionError(Exception):
    pass


tool_input_type = dict[str, Any]
tool_output_type = dict[str, Any]


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[tool_input_type], tool_output_type]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def list_specs(self) -> list[ToolSpec]:
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
        payload_for_call = sanitize_payload(payload)
        try:
            tool.spec.model_validate(tool.spec.model_dump())
            validate_schema(tool.spec.input_schema, payload, "input")
            result = run_with_timeout(lambda: tool.handler(payload), tool.spec.timeout_s)
            validate_schema(tool.spec.output_schema, result, "output")
            result_bytes = json.dumps(result, ensure_ascii=True).encode("utf-8")
            if len(result_bytes) > max_output_bytes:
                raise ToolExecutionError("Tool output exceeded max size")
            status = "completed"
            output = result
        except (ToolExecutionError, ValidationError) as exc:
            status = "failed"
            raw_error = str(exc)
            output = {"error": raw_error, "error_code": classify_tool_error(raw_error)}
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


def classify_tool_error(error_text: str) -> str:
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


def sanitize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    def sanitize(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(key, str) and key.startswith("_"):
                    continue
                cleaned_item = sanitize(item)
                if cleaned_item is not skip:
                    cleaned[key] = cleaned_item
            return cleaned
        if isinstance(value, list):
            cleaned_list = []
            for item in value:
                cleaned_item = sanitize(item)
                if cleaned_item is not skip:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        try:
            json.dumps(value, ensure_ascii=True)
            return value
        except Exception:  # noqa: BLE001
            return str(value)

    skip = object()
    return sanitize(payload) or {}


def run_with_timeout(handler: Callable[[], tool_output_type], timeout_s: int | None) -> tool_output_type:
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


def validate_schema(schema: dict[str, Any] | None, payload: dict[str, Any], label: str) -> None:
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


def _to_datetime(timestamp: float) -> datetime:
    return datetime.utcfromtimestamp(timestamp)
