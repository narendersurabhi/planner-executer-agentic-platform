from datetime import UTC, timedelta

from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolRegistry


def test_registry_execute_uses_timezone_aware_utc_timestamps() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            spec=ToolSpec(
                name="echo",
                description="echo payload",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=1,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload: payload,
        )
    )

    call = registry.execute(
        "echo",
        {"value": "ok"},
        idempotency_key="idem-1",
        trace_id="trace-1",
    )

    assert call.started_at.tzinfo is UTC
    assert call.finished_at.tzinfo is UTC
    assert call.started_at.utcoffset() == timedelta(0)
    assert call.finished_at.utcoffset() == timedelta(0)
