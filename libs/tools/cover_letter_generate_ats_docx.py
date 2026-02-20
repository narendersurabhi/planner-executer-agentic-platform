from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


class CoverLetter(BaseModel):
    full_name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    portfolio_url: str | None = None
    company: str | None = None
    role_title: str | None = None
    hiring_manager: str | None = None
    body: str
    closing: str | None = None


def register_cover_letter_generate_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="cover_letter_generate_ats_docx",
                description="Generate an ATS-friendly single-column cover letter .docx from a CoverLetter object",
                usage_guidance=(
                    "Provide cover_letter and optional path/today. "
                    "This tool converts CoverLetter to a standard ATS DocumentSpec, "
                    "validates it, then renders a DOCX under /shared/artifacts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "cover_letter": {"type": "object"},
                        "path": {
                            "type": "string",
                            "default": "cover_letters/cover_letter_ats.docx",
                        },
                        "today": {"type": "string", "default": "February 3, 2026"},
                        "strict": {"type": "boolean", "default": True},
                    },
                    "required": ["cover_letter"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "bytes_written": {"type": "integer"},
                        "validation_warnings": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["path", "bytes_written"],
                },
                timeout_s=30,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.render,
            ),
            handler=_cover_letter_generate_ats_docx,
        )
    )


def _cover_letter_generate_ats_docx(payload: Dict[str, Any]) -> Dict[str, Any]:
    from libs.core.tool_registry import ToolExecutionError

    raw = payload.get("cover_letter")
    if not isinstance(raw, dict):
        raise ToolExecutionError("cover_letter must be an object")

    path = payload.get("path", "cover_letters/cover_letter_ats.docx")
    today = payload.get("today", "February 3, 2026")
    strict = payload.get("strict", True)
    if not isinstance(path, str) or not path.endswith(".docx"):
        raise ToolExecutionError("path must be a .docx filename relative to /shared/artifacts")
    if not isinstance(today, str) or not today:
        raise ToolExecutionError("today must be a non-empty string")
    if not isinstance(strict, bool):
        raise ToolExecutionError("strict must be a boolean")

    try:
        cl = CoverLetter.model_validate(raw)
    except ValidationError as e:
        raise ToolExecutionError(f"Invalid CoverLetter: {e}")

    spec = _build_cover_letter_ats_document_spec(cl)

    registry = _get_registry_from_payload(payload)

    v = registry.execute(
        "document_spec_validate",
        {
            "document_spec": spec,
            "render_context": {"TODAY": today},
            "strict": strict,
        },
    )
    if not isinstance(v, dict) or not v.get("valid", False):
        raise ToolExecutionError(f"DocumentSpec validation failed: {v}")

    out = registry.execute(
        "docx_generate_from_spec",
        {
            "document_spec": spec,
            "path": path,
            "render_context": {"TODAY": today},
            "strict": strict,
        },
    )
    if not isinstance(out, dict) or "path" not in out:
        raise ToolExecutionError(f"docx_generate_from_spec failed: {out}")

    return {
        "path": out["path"],
        "bytes_written": int(out.get("bytes_written", 0)),
        "validation_warnings": v.get("warnings", []),
    }


def _build_cover_letter_ats_document_spec(cl: CoverLetter) -> Dict[str, Any]:
    # Split body into paragraphs by blank lines (preserves structure)
    body_paragraphs = _split_paragraphs(cl.body)

    # Contact line: location | email | phone | linkedin | portfolio (omit missing)
    contact_parts = [
        p
        for p in [
            cl.location,
            cl.email,
            cl.phone,
            cl.linkedin_url,
            cl.portfolio_url,
        ]
        if p
    ]
    contact_line = " | ".join(contact_parts)

    tokens: Dict[str, Any] = {
        "name": cl.full_name,
        "contact_line": contact_line,
        "company": cl.company or "",
        "role": cl.role_title or "",
        "hiring_manager": cl.hiring_manager or "Hiring Manager",
        "closing": cl.closing or "",
        "date": "{{TODAY}}",
        "body_paragraphs": body_paragraphs,
    }

    return {
        "doc_type": "cover_letter_ats_single_column",
        "version": "1.0",
        "theme": {
            "fonts": {"body": "Calibri", "heading": "Calibri"},
            "font_sizes": {"body": 11},
            "spacing": {"line": 1.15, "para_after_pt": 8},
            "page_margins_in": {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
        },
        "tokens": tokens,
        "blocks": [
            {"type": "paragraph", "style": "body_bold", "text": "{{name}}"},
            {"type": "optional_paragraph", "when": "{{contact_line}}", "text": "{{contact_line}}"},
            {"type": "spacer"},
            {"type": "paragraph", "text": "{{date}}"},
            {"type": "spacer"},
            {"type": "paragraph", "text": "{{hiring_manager}}"},
            {"type": "optional_paragraph", "when": "{{company}}", "text": "{{company}}"},
            {"type": "spacer"},
            {"type": "optional_paragraph", "when": "{{role}}", "text": "Re: {{role}}"},
            {"type": "spacer"},
            {
                "type": "repeat",
                "items": "{{body_paragraphs}}",
                "as": "p",
                "template": [{"type": "paragraph", "text": "{{p}}"}],
            },
            {"type": "spacer"},
            {"type": "optional_paragraph", "when": "{{closing}}", "text": "{{closing}}"},
            {"type": "paragraph", "text": "{{name}}"},
        ],
    }


def _split_paragraphs(body: str) -> List[str]:
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts


def _get_registry_from_payload(payload: Dict[str, Any]):
    registry = payload.get("_registry")
    if registry is None:
        from libs.core.tool_registry import ToolExecutionError

        raise ToolExecutionError(
            "Tool registry not available in payload. Inject payload['_registry'] or adjust _get_registry_from_payload."
        )
    return registry
