from __future__ import annotations

from typing import Any

from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool


def register_tools(registry, **_kwargs: Any) -> None:
    """Register example tools for plug-and-play loading."""
    registry.register(
        Tool(
            spec=ToolSpec(
                name="example_echo",
                description="Echo back input text with a prefix",
                usage_guidance=(
                    "Use for plugin smoke tests. Provide 'text'. Optionally provide 'prefix'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "prefix": {"type": "string"},
                    },
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=_example_echo,
        )
    )


def _example_echo(payload: dict[str, Any]) -> dict[str, str]:
    text = str(payload.get("text", "")).strip()
    if not text:
        return {"text": ""}
    prefix = str(payload.get("prefix", "echo")).strip() or "echo"
    return {"text": f"{prefix}: {text}"}

