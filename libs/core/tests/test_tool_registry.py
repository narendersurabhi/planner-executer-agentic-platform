import time

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

    def fake_call(url: str, tool_name: str, arguments: dict) -> dict:
        seen.append(url)
        return {"ok": True, "tool_name": tool_name, "arguments": arguments}

    monkeypatch.setattr(tool_registry_module, "_call_mcp_tool_sdk", fake_call)
    result = tool_registry_module._post_mcp_tool_call(
        "http://tailor:8000",
        "tailor_resume",
        {"job": {"id": "1"}},
    )
    assert result["ok"] is True
    assert seen == ["http://tailor:8000/mcp/rpc/mcp"]


def test_post_mcp_tool_call_falls_back_to_legacy_mcp_root(monkeypatch) -> None:
    seen: list[str] = []

    def fake_call(url: str, tool_name: str, arguments: dict) -> dict:
        seen.append(url)
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

    def fake_call(url: str, tool_name: str, arguments: dict) -> dict:
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
