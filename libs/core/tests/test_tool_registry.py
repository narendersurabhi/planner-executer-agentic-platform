import time

from libs.core.models import RiskLevel, ToolSpec
from libs.core.tool_registry import Tool, ToolRegistry, default_registry


def test_input_schema_validation() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            spec=ToolSpec(
                name="schema_input",
                description="test",
                input_schema={
                    "type": "object",
                    "properties": {"foo": {"type": "string"}},
                    "required": ["foo"],
                },
                output_schema={"type": "object"},
                timeout_s=1,
                risk_level=RiskLevel.low,
            ),
            handler=lambda payload: {"ok": True},
        )
    )
    call = registry.execute("schema_input", {}, "idempotency", "trace")
    assert call.status == "failed"
    assert "input schema validation failed" in call.output_or_error["error"]


def test_output_schema_validation() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            spec=ToolSpec(
                name="schema_output",
                description="test",
                input_schema={"type": "object"},
                output_schema={
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
                timeout_s=1,
                risk_level=RiskLevel.low,
            ),
            handler=lambda payload: {"wrong": 1},
        )
    )
    call = registry.execute("schema_output", {}, "idempotency", "trace")
    assert call.status == "failed"
    assert "output schema validation failed" in call.output_or_error["error"]


def test_timeout_enforced() -> None:
    registry = ToolRegistry()

    def slow_handler(payload: dict) -> dict:
        time.sleep(0.05)
        return {"ok": True}

    registry.register(
        Tool(
            spec=ToolSpec(
                name="slow_tool",
                description="test",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=0.01,
                risk_level=RiskLevel.low,
            ),
            handler=slow_handler,
        )
    )
    call = registry.execute("slow_tool", {}, "idempotency", "trace")
    assert call.status == "failed"
    assert "timed out" in call.output_or_error["error"]


def test_output_size_cap() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            spec=ToolSpec(
                name="big_output",
                description="test",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=1,
                risk_level=RiskLevel.low,
            ),
            handler=lambda payload: {"text": "a" * 100},
        )
    )
    call = registry.execute("big_output", {}, "idempotency", "trace", max_output_bytes=10)
    assert call.status == "failed"
    assert "output exceeded max size" in call.output_or_error["error"]


def test_docx_render_tool_registered() -> None:
    registry = default_registry()
    tool = registry.get("docx_render")
    schema = tool.spec.input_schema
    assert "data" in schema["properties"]
    assert "template_id" in schema["properties"]
    assert "template_path" in schema["properties"]
    assert "schema_ref" in schema["properties"]
    assert any("template_id" in entry.get("required", []) for entry in schema["anyOf"])
    assert "output_path" in schema["required"]


def test_file_write_text_requires_path() -> None:
    registry = default_registry()
    call = registry.execute("file_write_text", {"content": "hello"}, "id", "trace")
    assert call.status == "failed"
    assert "input schema validation failed" in call.output_or_error["error"]


def test_file_write_code_requires_extension() -> None:
    registry = default_registry()
    call = registry.execute(
        "file_write_code", {"content": "hello", "path": "output"}, "id", "trace"
    )
    assert call.status == "failed"
    assert "Unsupported code file extension" in call.output_or_error["error"]
