from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from pydantic import ValidationError

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
            result = tool.handler(payload)
            result_bytes = json.dumps(result).encode("utf-8")
            if len(result_bytes) > max_output_bytes:
                raise ToolExecutionError("Tool output exceeded max size")
            status = "completed"
            output = result
        except ValidationError as exc:
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



def default_registry(http_fetch_enabled: bool = False) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="json_transform",
                description="Apply a simple JSON transformation",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=5,
                risk_level=RiskLevel.low,
            ),
            handler=lambda payload: {"result": payload},
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="math_eval",
                description="Evaluate a safe math expression",
                input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"value": {"type": "number"}}},
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
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
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
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
                output_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                timeout_s=5,
                risk_level=RiskLevel.medium,
            ),
            handler=_file_write_artifact,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="sleep",
                description="Sleep for a number of seconds",
                input_schema={"type": "object", "properties": {"seconds": {"type": "number"}}},
                output_schema={"type": "object", "properties": {"slept": {"type": "number"}}},
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
                    input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"body": {"type": "string"}}},
                    timeout_s=10,
                    risk_level=RiskLevel.high,
                ),
                handler=_http_fetch,
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
    path = payload.get("path", "artifact.txt")
    content = payload.get("content", "")
    artifact_path = f"/shared/artifacts/{path}"
    with open(artifact_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return {"path": artifact_path}



def _sleep(payload: Dict[str, Any]) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0))
    time.sleep(seconds)
    return {"slept": seconds}



def _http_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request

    url = payload.get("url", "")
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
    return {"body": body}
