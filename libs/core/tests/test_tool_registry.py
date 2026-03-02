import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from libs.core.llm_provider import LLMProvider, LLMResponse
from libs.core.models import RiskLevel, ToolSpec
from libs.core.tool_registry import (
    Tool,
    ToolExecutionError,
    ToolRegistry,
    default_registry,
    evaluate_tool_allowlist,
)
from libs.core import tool_registry as tool_registry_module


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
    assert call.output_or_error["error_code"] == "contract.input_invalid"


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
    assert call.output_or_error["error_code"] == "contract.output_invalid"


def test_timeout_enforced() -> None:
    registry = ToolRegistry()

    def slow_handler(payload: dict) -> dict:
        time.sleep(1.1)
        return {"ok": True}

    registry.register(
        Tool(
            spec=ToolSpec(
                name="slow_tool",
                description="test",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=1,
                risk_level=RiskLevel.low,
            ),
            handler=slow_handler,
        )
    )
    call = registry.execute("slow_tool", {}, "idempotency", "trace")
    assert call.status == "failed"
    assert call.output_or_error["error"].startswith("tool_call_timed_out:")
    assert "timed out" in call.output_or_error["error"]
    assert call.output_or_error["error_code"] == "runtime.timeout"


def test_timeout_returns_without_waiting_for_handler_completion() -> None:
    registry = ToolRegistry()

    def very_slow_handler(payload: dict) -> dict:
        time.sleep(5.0)
        return {"ok": True}

    registry.register(
        Tool(
            spec=ToolSpec(
                name="very_slow_tool",
                description="test",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=1,
                risk_level=RiskLevel.low,
            ),
            handler=very_slow_handler,
        )
    )
    started = time.monotonic()
    call = registry.execute("very_slow_tool", {}, "idempotency", "trace")
    elapsed = time.monotonic() - started
    assert call.status == "failed"
    assert "timed out" in call.output_or_error["error"]
    assert elapsed < 2.5


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


def test_pdf_generate_tool_registered() -> None:
    registry = default_registry()
    tool = registry.get("pdf_generate_from_spec")
    schema = tool.spec.input_schema
    assert "document_spec" in schema["properties"]
    assert "path" in schema["properties"]
    assert "render_context" in schema["properties"]
    assert "strict" in schema["properties"]


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


def test_workspace_write_text_requires_path() -> None:
    registry = default_registry()
    call = registry.execute("workspace_write_text", {"content": "hello"}, "id", "trace")
    assert call.status == "failed"
    assert "input schema validation failed" in call.output_or_error["error"]


def test_workspace_write_code_requires_extension() -> None:
    registry = default_registry()
    call = registry.execute(
        "workspace_write_code", {"content": "hello", "path": "output"}, "id", "trace"
    )
    assert call.status == "failed"
    assert "Unsupported code file extension" in call.output_or_error["error"]


def test_derive_output_filename() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {"target_role_name": "Senior AI/ML Engineer", "date": "2026-02-09"},
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == "documents/senior_ai_ml_engineer_2026_02_09.docx"




def test_derive_output_filename_with_output_extension_pdf() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "target_role_name": "Senior AI/ML Engineer",
            "date": "2026-02-09",
            "output_extension": "pdf",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == "documents/senior_ai_ml_engineer_2026_02_09.pdf"
    assert call.output_or_error["output_extension"] == "pdf"


def test_derive_output_filename_accepts_document_type_as_format_when_extension_missing() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "target_role_name": "Staff Platform Engineer",
            "date": "2026-02-09",
            "document_type": "md",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == "documents/staff_platform_engineer_2026_02_09.md"
    assert call.output_or_error["output_extension"] == "md"


def test_derive_output_filename_defaults_to_today_when_missing_date() -> None:
    registry = default_registry()
    today_slug = datetime.now(UTC).date().isoformat().replace("-", "_")
    call = registry.execute(
        "derive_output_filename",
        {"target_role_name": "Platform Engineer", "output_extension": "pdf"},
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == f"documents/platform_engineer_{today_slug}.pdf"
    assert call.output_or_error["output_extension"] == "pdf"


def test_derive_output_path_uses_generic_slug_naming() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_path",
        {
            "topic": "Latency in Distributed Systems",
            "today": "2026-02-25",
            "output_dir": "documents",
            "document_type": "document",
            "output_extension": "pdf",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "documents/latency_in_distributed_systems_2026_02_25.pdf"
    )
    assert call.output_or_error["output_extension"] == "pdf"


def test_derive_output_path_defaults_document_type_when_missing() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_path",
        {
            "topic": "Latency in Distributed Systems",
            "today": "2026-02-25",
            "output_dir": "documents",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "documents/latency_in_distributed_systems_2026_02_25.docx"
    )
    assert call.output_or_error["document_type"] == "document"
    assert call.output_or_error["output_extension"] == "docx"


def test_derive_output_path_rejects_invalid_output_dir() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_path",
        {
            "topic": "Latency in Distributed Systems",
            "today": "2026-02-25",
            "output_dir": "../documents",
            "document_type": "document",
        },
        "id",
        "trace",
    )
    assert call.status == "failed"
    assert "Invalid output_dir" in call.output_or_error["error"]


def test_default_registry_loads_module_plugins(monkeypatch, tmp_path) -> None:
    plugin_path = tmp_path / "demo_plugin.py"
    plugin_path.write_text(
        (
            "from libs.core.models import ToolSpec, RiskLevel\n"
            "from libs.framework.tool_runtime import Tool\n"
            "\n"
            "def register_tools(registry):\n"
            "    registry.register(\n"
            "        Tool(\n"
            "            spec=ToolSpec(\n"
            "                name='plugin_echo',\n"
            "                description='plugin',\n"
            "                input_schema={'type':'object'},\n"
            "                output_schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']},\n"
            "                timeout_s=1,\n"
            "                risk_level=RiskLevel.low,\n"
            "            ),\n"
            "            handler=lambda payload: {'ok': True},\n"
            "        )\n"
            "    )\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TOOL_PLUGIN_MODULES", "demo_plugin")
    registry = default_registry()
    call = registry.execute("plugin_echo", {}, "id", "trace")
    assert call.status == "completed"
    assert call.output_or_error["ok"] is True


def test_default_registry_applies_enabled_disabled_tool_filters(monkeypatch) -> None:
    monkeypatch.delenv("TOOL_PLUGIN_MODULES", raising=False)
    monkeypatch.setenv("ENABLED_TOOLS", "math_eval,text_summarize")
    monkeypatch.setenv("DISABLED_TOOLS", "text_summarize")
    registry = default_registry()
    specs = {spec.name for spec in registry.list_specs()}
    assert specs == {"math_eval"}


def test_tool_allowlist_precedence_service_and_global(monkeypatch) -> None:
    monkeypatch.setenv("ENABLED_TOOLS", "math_eval,text_summarize")
    monkeypatch.setenv("WORKER_ENABLED_TOOLS", "math_eval,sleep")
    monkeypatch.setenv("DISABLED_TOOLS", "sleep")
    monkeypatch.setenv("WORKER_DISABLED_TOOLS", "math_eval")
    assert evaluate_tool_allowlist("math_eval", "worker").allowed is False
    assert evaluate_tool_allowlist("math_eval", "worker").reason == "service_disabled"
    assert evaluate_tool_allowlist("sleep", "worker").allowed is False
    assert evaluate_tool_allowlist("sleep", "worker").reason == "global_disabled"
    assert evaluate_tool_allowlist("text_summarize", "worker").allowed is False
    assert evaluate_tool_allowlist("text_summarize", "worker").reason == "not_in_service_enabled"


def test_default_registry_applies_service_specific_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TOOL_GOVERNANCE_ENABLED", "false")
    monkeypatch.setenv("ENABLED_TOOLS", "math_eval,text_summarize,file_write_text")
    monkeypatch.setenv("WORKER_ENABLED_TOOLS", "math_eval,file_write_text")
    monkeypatch.setenv("WORKER_DISABLED_TOOLS", "file_write_text")
    registry = default_registry(service_name="worker")
    specs = {spec.name for spec in registry.list_specs()}
    assert specs == {"math_eval"}


def test_evaluate_tool_allowlist_uses_governance_config_enforce(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "tool_governance.yaml"
    cfg_path.write_text(
        (
            "tool_governance:\n"
            "  mode: enforce\n"
            "  global:\n"
            "    allow: []\n"
            "    deny: []\n"
            "  services:\n"
            "    worker:\n"
            "      allow: [math_eval]\n"
            "      deny: [sleep]\n"
            "  tenants:\n"
            "    tenant_a:\n"
            "      deny: [math_eval]\n"
            "  job_types: {}\n"
            "  risk:\n"
            "    blocked_levels_by_service:\n"
            "      worker: [high]\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOL_GOVERNANCE_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("TOOL_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("TOOL_GOVERNANCE_MODE", "enforce")
    decision = evaluate_tool_allowlist("sleep", "worker")
    assert decision.allowed is False
    assert decision.reason == "config_service_deny"
    decision = evaluate_tool_allowlist(
        "math_eval",
        "worker",
        context={"tenant_id": "tenant_a"},
    )
    assert decision.allowed is False
    assert decision.reason == "config_tenant_deny"
    high_risk_tool = ToolSpec(
        name="high_tool",
        description="high",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        timeout_s=1,
        risk_level=RiskLevel.high,
    )
    decision = evaluate_tool_allowlist(
        "high_tool",
        "worker",
        tool_spec=high_risk_tool,
    )
    assert decision.allowed is False
    assert decision.reason == "risk_blocked:high"


def test_evaluate_tool_allowlist_dry_run_does_not_block(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "tool_governance.yaml"
    cfg_path.write_text(
        (
            "tool_governance:\n"
            "  mode: dry_run\n"
            "  global:\n"
            "    deny: [sleep]\n"
            "  services: {}\n"
            "  tenants: {}\n"
            "  job_types: {}\n"
            "  risk:\n"
            "    blocked_levels_by_service: {}\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOL_GOVERNANCE_CONFIG_PATH", str(cfg_path))
    monkeypatch.setenv("TOOL_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("TOOL_GOVERNANCE_MODE", "dry_run")
    decision = evaluate_tool_allowlist("sleep", "worker")
    assert decision.allowed is True
    assert decision.violated is True
    assert decision.reason.startswith("dry_run:")
    assert decision.mode == "dry_run"


def test_default_registry_plugin_fail_fast_toggle(monkeypatch, tmp_path) -> None:
    plugin_path = tmp_path / "bad_plugin.py"
    plugin_path.write_text(
        "def register_tools(registry):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TOOL_PLUGIN_MODULES", "bad_plugin")
    monkeypatch.setenv("TOOL_PLUGIN_FAIL_FAST", "false")
    registry = default_registry()
    with pytest.raises(KeyError):
        registry.get("bad_plugin_tool")


def test_default_registry_plugin_fail_fast_raises(monkeypatch, tmp_path) -> None:
    plugin_path = tmp_path / "bad_plugin_raise.py"
    plugin_path.write_text(
        "def register_tools(registry):\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TOOL_PLUGIN_MODULES", "bad_plugin_raise")
    monkeypatch.setenv("TOOL_PLUGIN_FAIL_FAST", "true")
    with pytest.raises(RuntimeError):
        default_registry()



def test_derive_output_filename_reads_from_job_context_memory() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "memory": {
                "job_context": [
                    {
                        "payload": {
                            "target_role_name": "Software Engineer AI",
                            "today": "2026-02-13",
                        }
                    }
                ]
            }
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == "documents/software_engineer_ai_2026_02_13.docx"


def test_derive_output_filename_derives_role_from_jd_for_date_fallback() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "job_description": "Principal Backend Engineer\nAcme is building distributed systems.",
            "date": "2026-02-13",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert call.output_or_error["path"] == "documents/principal_backend_engineer_2026_02_13.docx"


def test_post_mcp_tool_call_uses_mcp_subpath_first(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        assert timeout_s > 0
        return {"ok": True, "tool_name": tool_name, "arguments": arguments}

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://service:8000",
        "example_tool",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == ["http://service:8000/mcp/rpc/mcp"]


def test_post_mcp_tool_call_biases_first_attempt_timeout(monkeypatch) -> None:
    seen_timeouts: list[float] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen_timeouts.append(timeout_s)
        return {"ok": True}

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "600")
    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "2")
    monkeypatch.setenv("MCP_TOOL_RETRY_SLEEP_S", "0")
    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://service:8000",
        "example_tool",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert len(seen_timeouts) == 1
    # With 600s global budget and reserved retry budget, first attempt should receive most of the deadline.
    assert seen_timeouts[0] >= 560.0


def test_resolve_mcp_timeout_falls_back_to_openai_timeout(monkeypatch) -> None:
    monkeypatch.delenv("MCP_TOOL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MCP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_EVAL_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "60")
    assert tool_registry_module._resolve_mcp_timeout_s() == 60.0


def test_resolve_mcp_timeout_clamps_large_openai_timeout(monkeypatch) -> None:
    monkeypatch.delenv("MCP_TOOL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MCP_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.delenv("TAILOR_EVAL_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "600")
    assert tool_registry_module._resolve_mcp_timeout_s() == 180.0


def test_resolve_mcp_tool_timeout_adds_outer_headroom(monkeypatch) -> None:
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "120")
    monkeypatch.delenv("MCP_TOOL_OUTER_TIMEOUT_HEADROOM_S", raising=False)
    assert tool_registry_module._resolve_mcp_tool_timeout_s() == 135


def test_post_mcp_tool_call_falls_back_to_legacy_mcp_root(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        assert timeout_s > 0
        if url.endswith("/mcp/rpc/mcp"):
            raise ToolExecutionError("not_found")
        return {"ok": True, "tool_name": tool_name, "arguments": arguments}

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://service:8000",
        "example_tool",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == [
        "http://service:8000/mcp/rpc/mcp",
        "http://service:8000/mcp/rpc",
    ]


def test_post_mcp_tool_call_does_not_retry_on_tool_error(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        raise ToolExecutionError("mcp_tool_error:bad_input")

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    try:
        tool_registry_module._post_mcp_tool_call(
            "http://service:8000",
            "example_tool",
            {"job": {"id": "1"}},
        )
    except ToolExecutionError as exc:
        assert str(exc) == "mcp_tool_error:bad_input"
    else:
        raise AssertionError("expected ToolExecutionError")


def test_post_mcp_tool_call_retries_retryable_sdk_error(monkeypatch) -> None:
    seen: list[str] = []
    attempts = {"count": 0}

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ToolExecutionError("mcp_sdk_error:Session terminated")
        return {"ok": True, "tool_name": tool_name, "arguments": arguments}

    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "1")
    monkeypatch.setenv("MCP_TOOL_RETRY_SLEEP_S", "0")
    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://service:8000",
        "example_tool",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == [
        "http://service:8000/mcp/rpc/mcp",
        "http://service:8000/mcp/rpc/mcp",
    ]


def test_post_mcp_tool_call_bounds_retries_by_deadline(monkeypatch) -> None:
    attempts: list[float] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        attempts.append(timeout_s)
        # Simulate consuming the full per-attempt budget.
        time.sleep(timeout_s)
        raise ToolExecutionError("mcp_sdk_error:Session terminated")

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "1")
    # Exaggerated retries to ensure deadline budget stops execution early.
    monkeypatch.setenv("MCP_TOOL_MAX_RETRIES", "50")
    monkeypatch.setenv("MCP_TOOL_RETRY_SLEEP_S", "0")
    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    started = time.monotonic()
    with pytest.raises(ToolExecutionError) as exc:
        tool_registry_module._post_mcp_tool_call(
            "http://service:8000",
            "example_tool",
            {"job": {"id": "1"}},
        )
    elapsed = time.monotonic() - started
    assert str(exc.value).startswith(("mcp_sdk_timeout:", "mcp_sdk_all_routes_failed:"))
    # 2 routes * (50 retries + 1 initial) = 102 slots; deadline must stop far earlier.
    assert len(attempts) < 102
    assert elapsed < 2.5


def test_resolve_mcp_isolation_mode_defaults_to_process(monkeypatch) -> None:
    monkeypatch.delenv("MCP_TOOL_ISOLATION_MODE", raising=False)
    assert tool_registry_module._resolve_mcp_isolation_mode() == "process"




def test_extract_mcp_sdk_result_includes_error_detail() -> None:
    class _TextItem:
        text = "Error executing tool example_tool: invalid input"

    class _Result:
        isError = True
        structuredContent = None
        content = [_TextItem()]

    try:
        tool_registry_module._extract_mcp_sdk_result(_Result())
    except ToolExecutionError as exc:
        assert "invalid input" in str(exc)
    else:
        raise AssertionError("expected ToolExecutionError")


def test_extract_mcp_sdk_result_unwraps_fastmcp_result_wrapper() -> None:
    class _Result:
        isError = False
        structuredContent = {"result": {"files": [{"path": "app.py", "content": "print('hi')"}]}}
        content = []

    payload = tool_registry_module._extract_mcp_sdk_result(_Result())
    assert "files" in payload
    assert payload["files"][0]["path"] == "app.py"


def test_sanitize_document_spec_removes_spacers_and_empty_items() -> None:
    spec = {
        "blocks": [
            {"type": "heading", "level": 1, "text": "Title"},
            {"type": "spacer", "height": 1},
            {"type": "paragraph", "text": "   "},
            {"type": "bullets", "items": ["alpha", "", "   ", "beta"]},
        ]
    }
    sanitized = tool_registry_module._sanitize_document_spec(spec)
    assert sanitized["blocks"] == [
        {"type": "heading", "level": 1, "text": "Title"},
        {"type": "bullets", "items": ["alpha", "beta"]},
    ]


def test_sanitize_document_spec_cleans_repeat_template() -> None:
    spec = {
        "blocks": [
            {
                "type": "repeat",
                "items": [{"name": "a"}],
                "as": "item",
                "template": [
                    {"type": "spacer"},
                    {"type": "paragraph", "text": "  "},
                    {"type": "paragraph", "text": "{{item.name}}"},
                ],
            }
        ]
    }
    sanitized = tool_registry_module._sanitize_document_spec(spec)
    assert sanitized["blocks"] == [
        {
            "type": "repeat",
            "items": [{"name": "a"}],
            "as": "item",
            "template": [{"type": "paragraph", "text": "{{item.name}}"}],
        }
    ]


class _SequenceLLMStub(LLMProvider):
    def __init__(self, payloads: list[dict]) -> None:
        if not payloads:
            raise ValueError("payloads must be non-empty")
        self._payloads = [json.dumps(payload) for payload in payloads]
        self._index = 0

    def generate(self, prompt: str) -> LLMResponse:
        if self._index >= len(self._payloads):
            content = self._payloads[-1]
        else:
            content = self._payloads[self._index]
        self._index += 1
        return LLMResponse(content=content)


def test_llm_iterative_improve_document_spec_stops_when_valid() -> None:
    provider = _SequenceLLMStub(
        [
            {"blocks": [{"type": "heading", "level": 1}]},
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Title"},
                    {"type": "paragraph", "text": "Hello"},
                ]
            },
        ]
    )
    registry = default_registry(llm_enabled=True, llm_provider=provider)
    call = registry.execute(
        "llm_iterative_improve_document_spec",
        {
            "job": {"goal": "Generate a short doc"},
            "allowed_block_types": [
                "heading",
                "paragraph",
                "bullets",
                "text",
                "optional_paragraph",
                "repeat",
            ],
            "max_iterations": 5,
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    out = call.output_or_error
    assert out["iterations"] == 2
    assert out["reached_threshold"] is True
    assert out["validation_report"]["valid"] is True


def test_llm_iterative_improve_document_spec_defaults_allowed_block_types() -> None:
    provider = _SequenceLLMStub(
        [
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "Title"},
                    {"type": "paragraph", "text": "Hello"},
                ]
            }
        ]
    )
    registry = default_registry(llm_enabled=True, llm_provider=provider)
    call = registry.execute(
        "llm_iterative_improve_document_spec",
        {"job": {"goal": "Generate a short doc"}, "max_iterations": 2},
        "id",
        "trace",
    )
    assert call.status == "completed"
    out = call.output_or_error
    assert out["iterations"] == 1
    assert out["reached_threshold"] is True
    assert out["validation_report"]["valid"] is True


def test_llm_iterative_improve_openapi_spec_stops_when_valid() -> None:
    provider = _SequenceLLMStub(
        [
            {"openapi": "3.1.0", "info": {}, "paths": {}},
            {
                "openapi": "3.1.0",
                "info": {"title": "Demo API", "version": "1.0.0"},
                "paths": {
                    "/health": {
                        "get": {
                            "operationId": "getHealth",
                            "responses": {"200": {"description": "ok"}},
                        }
                    }
                },
            },
        ]
    )
    registry = default_registry(llm_enabled=True, llm_provider=provider)
    call = registry.execute(
        "llm_iterative_improve_openapi_spec",
        {"job": {"goal": "Generate OpenAPI spec"}, "max_iterations": 4},
        "id",
        "trace",
    )
    assert call.status == "completed"
    out = call.output_or_error
    assert out["iterations"] == 2
    assert out["reached_threshold"] is True
    assert out["validation_report"]["valid"] is True
    assert out["openapi_spec"]["openapi"] == "3.1.0"


def test_llm_iterative_improve_runbook_spec_generates_document_spec() -> None:
    provider = _SequenceLLMStub(
        [
            {
                "blocks": [
                    {"type": "heading", "level": 1, "text": "OVERVIEW"},
                    {"type": "paragraph", "text": "Use this runbook for safe rollout."},
                ]
            }
        ]
    )
    registry = default_registry(llm_enabled=True, llm_provider=provider)
    call = registry.execute(
        "llm_iterative_improve_runbook_spec",
        {
            "job": {"goal": "Create a runbook"},
            "max_iterations": 3,
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    out = call.output_or_error
    assert out["iterations"] == 1
    assert out["reached_threshold"] is True
    assert out["validation_report"]["valid"] is True


def test_docx_generate_from_spec_auto_derives_path_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    registry = default_registry()
    call = registry.execute(
        "docx_generate_from_spec",
        {
            "document_spec": {"blocks": [{"type": "paragraph", "text": "Hello"}]},
            "topic": "Latency in Distributed Systems",
            "today": "2026-02-25",
            "output_dir": "documents",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    output_path = Path(call.output_or_error["path"])
    assert output_path.exists()
    assert output_path.suffix == ".docx"

