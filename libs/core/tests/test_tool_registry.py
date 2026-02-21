import json
import time

import pytest

from libs.core.llm_provider import LLMProvider, LLMResponse
from libs.core.models import RiskLevel, ToolSpec
from libs.core.tool_registry import Tool, ToolExecutionError, ToolRegistry, default_registry
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
    assert call.output_or_error["path"] == "resumes/senior_ai_ml_engineer_2026_02_09.docx"


def test_derive_output_filename_resume_style() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "candidate_name": "Narender Surabhi",
            "target_role_name": "Software Engineer AI",
            "company_name": "Figma",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "resumes/Narender Surabhi Resume - Software Engineer AI - Figma.docx"
    )


def test_derive_output_filename_cover_letter_style() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "candidate_name": "Narender Surabhi",
            "target_role_name": "Software Engineer AI",
            "company_name": "Figma",
            "document_type": "cover_letter",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "resumes/Narender Surabhi Cover Letter - Software Engineer AI - Figma.docx"
    )
    assert call.output_or_error["document_type"] == "cover_letter"


def test_derive_output_filename_derives_from_jd_and_resume() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "job_description": (
                "Staff Software Engineer, AI/ML\n"
                "New York, NY; Remote, US\n"
                "Maven is the world's largest virtual clinic for women and families.\n"
            ),
            "candidate_resume": "NARENDER SURABHI\nOkemos, MI, USA 48864\n",
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "resumes/Narender Surabhi Resume - Staff Software Engineer AI ML - Maven.docx"
    )


def test_derive_output_filename_reads_from_job_context_memory() -> None:
    registry = default_registry()
    call = registry.execute(
        "derive_output_filename",
        {
            "memory": {
                "job_context": [
                    {
                        "payload": {
                            "candidate_name": "Narender Surabhi",
                            "target_role_name": "Software Engineer AI",
                            "company_name": "Figma",
                        }
                    }
                ]
            }
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    assert (
        call.output_or_error["path"]
        == "resumes/Narender Surabhi Resume - Software Engineer AI - Figma.docx"
    )


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
    assert call.output_or_error["path"] == "resumes/principal_backend_engineer_2026_02_13.docx"


def test_post_mcp_tool_call_uses_mcp_subpath_first(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        assert timeout_s > 0
        return {"ok": True, "tool_name": tool_name, "arguments": arguments}

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://tailor:8000",
        "tailor_resume",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == ["http://tailor:8000/mcp/rpc/mcp"]


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
        "http://tailor:8000",
        "tailor_resume",
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
        "http://tailor:8000",
        "tailor_resume",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == [
        "http://tailor:8000/mcp/rpc/mcp",
        "http://tailor:8000/mcp/rpc",
    ]


def test_post_mcp_tool_call_does_not_retry_on_tool_error(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict, timeout_s: float) -> dict:
        seen.append(url)
        raise ToolExecutionError("mcp_tool_error:bad_input")

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    try:
        tool_registry_module._post_mcp_tool_call(
            "http://tailor:8000",
            "tailor_resume",
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
        "http://tailor:8000",
        "tailor_resume",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == [
        "http://tailor:8000/mcp/rpc/mcp",
        "http://tailor:8000/mcp/rpc/mcp",
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
            "http://tailor:8000",
            "tailor_resume",
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


def test_mcp_backed_resume_tools_use_mcp_timeout(monkeypatch) -> None:
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_S", "120")
    monkeypatch.delenv("MCP_TOOL_OUTER_TIMEOUT_HEADROOM_S", raising=False)
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "600")
    registry = default_registry(llm_enabled=True, llm_provider=_CoverLetterLLMStub())
    assert registry.get("llm_tailor_resume_text").spec.timeout_s == 135
    assert registry.get("llm_improve_tailored_resume_text").spec.timeout_s == 135
    assert registry.get("llm_iterative_improve_tailored_resume_text").spec.timeout_s == 135


class _CoverLetterLLMStub(LLMProvider):
    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            content='{"body":"I am applying for this role and bring proven distributed systems experience."}'
        )


def test_llm_generate_cover_letter_from_resume_tool() -> None:
    registry = default_registry(llm_enabled=True, llm_provider=_CoverLetterLLMStub())
    payload = {
        "tailored_resume": {
            "schema_version": "1.0",
            "header": {
                "name": "Narender Surabhi",
                "title": "Staff Software Engineer, AI/ML",
                "location": "Okemos, MI, USA 48864",
                "phone": "+1 (213) 254-8205",
                "email": "surabhinarenderrao@gmail.com",
                "links": {
                    "linkedin": "linkedin.com/in/narendersurabhi",
                    "github": "github.com/narendersurabhi",
                },
            },
            "summary": "Summary",
            "skills": [{"group_name": "Languages", "items": ["Python"]}],
            "experience": [
                {
                    "company": "Acentra Health",
                    "title": "Software Engineer, AI/ML",
                    "location": "Okemos, MI",
                    "dates": "Dec 2016 - Present",
                    "bullets": ["Built and operated production systems."],
                }
            ],
            "education": [],
            "certifications": [],
        },
        "job": {
            "context_json": {
                "job_description": "Software Engineer, Distributed Systems\nFigma is the collaborative software platform for teams."
            }
        },
    }
    call = registry.execute(
        "llm_generate_cover_letter_from_resume",
        payload,
        "id",
        "trace",
    )
    assert call.status == "completed"
    out = call.output_or_error["cover_letter"]
    assert out["full_name"] == "Narender Surabhi"
    assert out["company"] == "Figma"
    assert out["role_title"] == "Software Engineer, Distributed Systems"
    assert out["recipient_line"] == "Hiring Team"
    assert out["salutation"] == "Dear Hiring Team,"


def test_llm_generate_coverletter_doc_spec_from_text_tool() -> None:
    registry = default_registry(llm_enabled=True, llm_provider=_CoverLetterLLMStub())
    payload = {
        "tailored_resume": {
            "schema_version": "1.0",
            "header": {
                "name": "Narender Surabhi",
                "title": "Staff Software Engineer, AI/ML",
                "location": "Okemos, MI, USA 48864",
                "phone": "+1 (213) 254-8205",
                "email": "surabhinarenderrao@gmail.com",
                "links": {
                    "linkedin": "linkedin.com/in/narendersurabhi",
                    "github": "github.com/narendersurabhi",
                },
            },
            "summary": "Summary",
            "skills": [{"group_name": "Languages", "items": ["Python"]}],
            "experience": [
                {
                    "company": "Acentra Health",
                    "title": "Software Engineer, AI/ML",
                    "location": "Okemos, MI",
                    "dates": "Dec 2016 - Present",
                    "bullets": ["Built and operated production systems."],
                }
            ],
            "education": [],
            "certifications": [],
        },
        "job": {
            "context_json": {
                "job_description": "Software Engineer, Distributed Systems\nFigma is the collaborative software platform for teams."
            }
        },
        "today_pretty": "February 13, 2026",
    }
    call = registry.execute(
        "llm_generate_coverletter_doc_spec_from_text",
        payload,
        "id",
        "trace",
    )
    assert call.status == "completed"
    spec = call.output_or_error["coverletter_doc_spec"]
    assert spec["doc_type"] == "cover_letter"
    assert isinstance(spec["content"], list)
    assert spec["content"][0]["type"] == "header"
    assert spec["content"][1]["text"] == "February 13, 2026"


def test_coverletter_doc_spec_to_document_spec_tool() -> None:
    registry = default_registry()
    call = registry.execute(
        "coverletter_doc_spec_to_document_spec",
        {
            "coverletter_doc_spec": {
                "schema_version": "1.0",
                "doc_type": "cover_letter",
                "title": "Example",
                "page": {
                    "size": "LETTER",
                    "margins_in": {"top": 0.8, "right": 0.8, "bottom": 0.8, "left": 0.8},
                },
                "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.15},
                "styles": {"name": {"bold": True, "size_pt": 20}},
                "content": [
                    {
                        "type": "header",
                        "blocks": [{"type": "text", "style": "name", "text": "Narender Surabhi"}],
                    },
                    {"type": "paragraph", "text": "Dear Hiring Team,"},
                ],
            }
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    blocks = call.output_or_error["document_spec"]["blocks"]
    assert blocks[0]["style"] == "cover_letter_name"
    assert blocks[1]["type"] == "paragraph"


def test_extract_mcp_sdk_result_includes_error_detail() -> None:
    class _TextItem:
        text = "Error executing tool tailor_resume: invalid input"

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
            "allowed_block_types": [
                "heading",
                "paragraph",
                "bullets",
                "text",
                "optional_paragraph",
                "repeat",
            ],
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


def test_normalize_skills_definition_separators_uses_commas() -> None:
    spec = {
        "content": [
            {"type": "section_heading", "text": "SUMMARY"},
            {
                "type": "definition_list",
                "items": [{"term": "Ignore", "definition": "alpha; beta"}],
            },
            {"type": "section_heading", "text": "SKILLS"},
            {
                "type": "definition_list",
                "items": [
                    {"term": "Languages", "definition": "Python; Java; SQL"},
                    {"term": "Cloud", "definition": "AWS, Azure"},
                ],
            },
        ]
    }
    tool_registry_module._normalize_skills_definition_separators(spec)
    items = spec["content"][3]["items"]
    assert items[0]["definition"] == "Python, Java, SQL"
    assert items[1]["definition"] == "AWS, Azure"


class _ResumeSpecLLMStub(LLMProvider):
    def __init__(self, payload: dict) -> None:
        self._content = json.dumps(payload)

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(content=self._content)


def _base_resume_doc_spec() -> dict:
    return {
        "schema_version": "1.0",
        "doc_type": "resume",
        "title": "Narender Surabhi - Resume",
        "page": {
            "size": "LETTER",
            "margins_in": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5},
        },
        "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.05},
        "styles": {},
        "content": [
            {
                "type": "header",
                "blocks": [
                    {"type": "text", "text": "Narender Surabhi"},
                    {"type": "text", "text": "Staff Software Engineer"},
                    {"type": "text", "text": "Okemos, MI | +1 (213) 254-8205"},
                ],
            },
            {"type": "section_heading", "text": "SUMMARY"},
            {"type": "paragraph", "text": "Summary"},
            {"type": "section_heading", "text": "SKILLS"},
            {"type": "definition_list", "items": [{"term": "Languages", "definition": "Python"}]},
            {"type": "section_heading", "text": "EXPERIENCE"},
            {
                "type": "role",
                "company": "Acentra Health",
                "location": "Okemos, MI",
                "title": "Software Engineer, AI/ML",
                "dates": "Dec 2016 - Present",
                "bullets": ["Built production systems."],
            },
            {"type": "section_heading", "text": "EDUCATION"},
            {
                "type": "education",
                "degree": "Bachelor of Science",
                "school": "Nizam College",
                "location": "Hyderabad, India",
                "dates": "Sep 2003 - May 2007",
            },
            {"type": "section_heading", "text": "CERTIFICATIONS"},
            {"type": "bullets", "items": []},
        ],
    }


def _dense_resume_doc_spec_for_paging() -> dict:
    return {
        "schema_version": "1.0",
        "doc_type": "resume",
        "title": "Narender Surabhi - Resume",
        "page": {
            "size": "LETTER",
            "margins_in": {"top": 0.6, "right": 0.6, "bottom": 0.6, "left": 0.6},
        },
        "defaults": {"font_family": "Calibri", "font_size_pt": 11.5, "line_spacing": 1.2},
        "styles": {},
        "content": [
            {"type": "section_heading", "text": "SUMMARY"},
            {
                "type": "paragraph",
                "text": (
                    "Line one. Line two. Line three. Line four. Line five with additional details."
                ),
            },
            {"type": "section_heading", "text": "SKILLS"},
            {
                "type": "definition_list",
                "items": [
                    {"term": "Languages", "definition": "Python, Java"},
                    {"term": "Cloud", "definition": "AWS"},
                    {"term": "MLOps", "definition": "SageMaker"},
                    {"term": "Data", "definition": "PostgreSQL"},
                    {"term": "Observability", "definition": "Grafana"},
                    {"term": "Security", "definition": "IAM"},
                ],
            },
            {"type": "section_heading", "text": "EXPERIENCE"},
            {
                "type": "role",
                "company": "A",
                "location": "X",
                "title": "T1",
                "dates": "2020 - Present",
                "bullets": [
                    "b1",
                    "b2",
                    "b3",
                    "b4",
                    "b5",
                    "b6",
                ],
            },
            {
                "type": "role",
                "company": "B",
                "location": "Y",
                "title": "T2",
                "dates": "2018 - 2020",
                "bullets": ["c1", "c2", "c3", "c4"],
            },
            {"type": "bullets", "items": ["g1", "g2", "g3", "g4"]},
        ],
    }


def _base_tailored_resume() -> dict:
    return {
        "schema_version": "1.0",
        "header": {
            "name": "Narender Surabhi",
            "title": "Staff Software Engineer, AI/ML",
            "location": "Okemos, MI, USA 48864",
            "phone": "+1 (213) 254-8205",
            "email": "surabhinarenderrao@gmail.com",
            "links": {
                "linkedin": "linkedin.com/in/narendersurabhi",
                "github": "github.com/narendersurabhi",
            },
        },
        "summary": "Summary",
        "skills": [{"group_name": "Languages", "items": ["Python"]}],
        "experience": [
            {
                "company": "Acentra Health",
                "title": "Software Engineer, AI/ML",
                "location": "Okemos, MI",
                "dates": "Dec 2016 - Present",
                "bullets": ["Built production systems."],
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science",
                "school": "Nizam College",
                "location": "Hyderabad, India",
                "dates": "Sep 2003 - May 2007",
            }
        ],
        "certifications": [],
    }


def test_resume_doc_spec_backfills_certifications_from_tailored_resume() -> None:
    resume_doc_spec = _base_resume_doc_spec()
    provider = _ResumeSpecLLMStub(resume_doc_spec)
    tailored_resume = _base_tailored_resume()
    tailored_resume["certifications"] = [
        {
            "name": "AWS Certified AI Practitioner (AIF-C01)",
            "issuer": "AWS",
            "year": 2025,
            "url": "https://www.credly.com/badges/d91a3fa3-b52c-4b44-8b5d-5f75f13f58e1/public_url",
        }
    ]
    out = tool_registry_module._llm_generate_resume_doc_spec_from_text(
        {"tailored_resume": tailored_resume},
        provider,
    )
    content = out["resume_doc_spec"]["content"]
    cert_items = []
    in_cert = False
    for block in content:
        if block.get("type") == "section_heading":
            in_cert = str(block.get("text", "")).strip().upper() == "CERTIFICATIONS"
            continue
        if not in_cert:
            continue
        if block.get("type") == "bullets":
            cert_items.extend(block.get("items") or [])
    assert cert_items
    assert "AWS Certified AI Practitioner (AIF-C01) - AWS (2025)" in cert_items[0]
    assert "credly.com/badges" in cert_items[0]


def test_apply_resume_target_pages_policy_one_page_compacts_content() -> None:
    spec = _dense_resume_doc_spec_for_paging()
    tool_registry_module._apply_resume_target_pages_policy(spec, 1)

    assert spec["defaults"]["font_size_pt"] == 10.5
    assert spec["defaults"]["line_spacing"] == 1.0
    assert spec["page"]["margins_in"] == {"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45}

    content = spec["content"]
    skills_block = content[3]
    assert len(skills_block["items"]) == 5

    role1 = content[5]
    role2 = content[6]
    exp_group_bullets = content[7]
    assert len(role1["bullets"]) == 4
    assert len(role2["bullets"]) == 2
    assert len(exp_group_bullets["items"]) == 2


def test_apply_resume_target_pages_policy_two_page_still_caps_density() -> None:
    spec = _dense_resume_doc_spec_for_paging()
    tool_registry_module._apply_resume_target_pages_policy(spec, 2)

    assert spec["defaults"]["font_size_pt"] == 10.75
    assert spec["defaults"]["line_spacing"] == 1.0
    assert spec["page"]["margins_in"] == {"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45}

    content = spec["content"]
    skills_block = content[3]
    assert len(skills_block["items"]) == 6

    role1 = content[5]
    role2 = content[6]
    exp_group_bullets = content[7]
    assert len(role1["bullets"]) == 6
    assert len(role2["bullets"]) == 3
    assert len(exp_group_bullets["items"]) == 3


def test_apply_resume_target_pages_policy_two_page_enforces_total_experience_bullet_budget() -> (
    None
):
    spec = {
        "defaults": {},
        "page": {"margins_in": {}},
        "content": [
            {"type": "section_heading", "text": "EXPERIENCE"},
            {
                "type": "role",
                "title": "A",
                "bullets": [f"a{i}" for i in range(1, 10)],
            },
            {
                "type": "role",
                "title": "B",
                "bullets": [f"b{i}" for i in range(1, 10)],
            },
            {
                "type": "role",
                "title": "C",
                "bullets": [f"c{i}" for i in range(1, 10)],
            },
            {
                "type": "bullets",
                "items": [f"g{i}" for i in range(1, 10)],
            },
        ],
    }
    tool_registry_module._apply_resume_target_pages_policy(spec, 2)

    total = 0
    for block in spec["content"]:
        if block.get("type") == "role":
            total += len(block.get("bullets") or [])
        if block.get("type") == "bullets":
            total += len(block.get("items") or [])
    assert total <= 16


def test_parse_target_pages_accepts_page_phrases() -> None:
    assert tool_registry_module._parse_target_pages("2 pages") == 2
    assert tool_registry_module._parse_target_pages("page 1") == 1
    assert tool_registry_module._parse_target_pages("target is 2 page resume") == 2
    assert tool_registry_module._parse_target_pages("three pages") is None


def test_llm_generate_resume_doc_spec_from_text_applies_target_pages_from_job_context() -> None:
    provider = _ResumeSpecLLMStub(_dense_resume_doc_spec_for_paging())
    out = tool_registry_module._llm_generate_resume_doc_spec_from_text(
        {
            "tailored_resume": _base_tailored_resume(),
            "job": {"context_json": {"target_pages": 1}},
        },
        provider,
    )
    resume_doc_spec = out["resume_doc_spec"]
    assert resume_doc_spec["defaults"]["font_size_pt"] == 10.5
    assert resume_doc_spec["defaults"]["line_spacing"] == 1.0


def test_resume_doc_spec_removes_empty_certifications_section_when_no_source() -> None:
    resume_doc_spec = _base_resume_doc_spec()
    provider = _ResumeSpecLLMStub(resume_doc_spec)
    tailored_resume = _base_tailored_resume()
    out = tool_registry_module._llm_generate_resume_doc_spec_from_text(
        {"tailored_resume": tailored_resume},
        provider,
    )
    content = out["resume_doc_spec"]["content"]
    headings = [
        str(block.get("text", "")).strip().upper()
        for block in content
        if isinstance(block, dict) and block.get("type") == "section_heading"
    ]
    assert "CERTIFICATIONS" not in headings


def test_resume_doc_spec_backfills_certifications_from_candidate_resume_context() -> None:
    resume_doc_spec = _base_resume_doc_spec()
    provider = _ResumeSpecLLMStub(resume_doc_spec)
    tailored_resume = _base_tailored_resume()
    out = tool_registry_module._llm_generate_resume_doc_spec_from_text(
        {
            "tailored_resume": tailored_resume,
            "job": {
                "context_json": {
                    "candidate_resume": (
                        "SUMMARY\n...\n"
                        "CERTIFICATIONS\n"
                        "AWS Certified AI Practitioner (AIF-C01), 2025\n"
                        "MIT Data Science and Machine Learning, 2022\n"
                    )
                }
            },
        },
        provider,
    )
    content = out["resume_doc_spec"]["content"]
    cert_items = []
    in_cert = False
    for block in content:
        if block.get("type") == "section_heading":
            in_cert = str(block.get("text", "")).strip().upper() == "CERTIFICATIONS"
            continue
        if not in_cert:
            continue
        if block.get("type") == "bullets":
            cert_items.extend(block.get("items") or [])
    assert cert_items == [
        "AWS Certified AI Practitioner (AIF-C01), 2025",
        "MIT Data Science and Machine Learning, 2022",
    ]


def test_resume_doc_spec_to_document_spec_skips_empty_certifications_heading() -> None:
    registry = default_registry()
    call = registry.execute(
        "resume_doc_spec_to_document_spec",
        {
            "resume_doc_spec": {
                "schema_version": "1.0",
                "doc_type": "resume",
                "title": "Resume",
                "page": {
                    "size": "LETTER",
                    "margins_in": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5},
                },
                "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.05},
                "styles": {},
                "content": [
                    {
                        "type": "header",
                        "blocks": [{"type": "text", "text": "Narender Surabhi"}],
                    },
                    {"type": "section_heading", "text": "SUMMARY"},
                    {"type": "paragraph", "text": "Summary text"},
                    {"type": "section_heading", "text": "CERTIFICATIONS"},
                    {"type": "bullets", "items": []},
                ],
            }
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    blocks = call.output_or_error["document_spec"]["blocks"]
    headings = [b.get("text") for b in blocks if b.get("type") == "heading"]
    assert "SUMMARY" in headings
    assert "CERTIFICATIONS" not in headings


def test_resume_doc_spec_to_document_spec_keeps_non_empty_certifications_heading() -> None:
    registry = default_registry()
    call = registry.execute(
        "resume_doc_spec_to_document_spec",
        {
            "resume_doc_spec": {
                "schema_version": "1.0",
                "doc_type": "resume",
                "title": "Resume",
                "page": {
                    "size": "LETTER",
                    "margins_in": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5},
                },
                "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.05},
                "styles": {},
                "content": [
                    {
                        "type": "header",
                        "blocks": [{"type": "text", "text": "Narender Surabhi"}],
                    },
                    {"type": "section_heading", "text": "SUMMARY"},
                    {"type": "paragraph", "text": "Summary text"},
                    {"type": "section_heading", "text": "CERTIFICATIONS"},
                    {"type": "bullets", "items": ["AWS Certified AI Practitioner - AWS (2025)"]},
                ],
            }
        },
        "id",
        "trace",
    )
    assert call.status == "completed"
    blocks = call.output_or_error["document_spec"]["blocks"]
    headings = [b.get("text") for b in blocks if b.get("type") == "heading"]
    assert "CERTIFICATIONS" in headings
