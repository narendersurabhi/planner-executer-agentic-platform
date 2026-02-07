from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


class ResumeExperience(BaseModel):
    title: str
    company: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    dates: str | None = None
    bullets: List[str | dict]


class ResumeEducation(BaseModel):
    degree: str
    school: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    dates: str | None = None


class ResumeCertification(BaseModel):
    name: str
    issuer: str | None = None
    year: str | None = None
    url: str | None = None


class ResumeProject(BaseModel):
    name: str
    url: str | None = None
    highlights: List[str] = []


class ResumeHeader(BaseModel):
    name: str
    location: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin: str | None = None
    github: str | None = None


class TailoredResume(BaseModel):
    header: ResumeHeader
    summary: str
    skills: Dict[str, List[str]]
    experience: List[ResumeExperience]
    education: List[ResumeEducation]
    certifications: List[ResumeCertification] = []
    projects: List[ResumeProject] = []
    resume_text: str | None = None


def register_resume_generate_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="resume_generate_ats_docx",
                description="Generate an ATS-friendly single-column resume .docx from a TailoredResume object",
                usage_guidance=(
                    "Provide tailored_resume and optional path/today. "
                    "This tool converts TailoredResume to a standard ATS DocumentSpec, "
                    "validates it, then renders a DOCX under /shared/artifacts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_resume": {"type": "object"},
                        "path": {"type": "string", "default": "resumes/tailored_resume_ats.docx"},
                        "today": {"type": "string", "default": "February 3, 2026"},
                        "strict": {"type": "boolean", "default": True},
                    },
                    "required": ["tailored_resume"],
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
            handler=_resume_generate_ats_docx,
        )
    )


def _resume_generate_ats_docx(payload: Dict[str, Any]) -> Dict[str, Any]:
    from libs.core.tool_registry import ToolExecutionError

    raw = payload.get("tailored_resume")
    if not isinstance(raw, dict):
        raise ToolExecutionError("tailored_resume must be an object")

    path = payload.get("path", "resumes/tailored_resume_ats.docx")
    today = payload.get("today", "February 3, 2026")
    strict = payload.get("strict", True)
    if not isinstance(path, str) or not path.endswith(".docx"):
        raise ToolExecutionError("path must be a .docx filename relative to /shared/artifacts")
    if not isinstance(today, str) or not today:
        raise ToolExecutionError("today must be a non-empty string")
    if not isinstance(strict, bool):
        raise ToolExecutionError("strict must be a boolean")

    try:
        resume = TailoredResume.model_validate(raw)
    except ValidationError as e:
        raise ToolExecutionError(f"Invalid TailoredResume: {e}")

    spec = _build_resume_ats_document_spec(resume)

    # validate then render using existing tools synchronously
    # NOTE: calling registry.execute inside a tool is safe in your sync worker model,
    # but ensure your registry doesn't forbid re-entrancy.
    registry = _get_registry_from_payload(payload)

    trace_id = payload.get("trace_id", "") if isinstance(payload, dict) else ""
    v = registry.execute(
        "document_spec_validate",
        {
            "document_spec": spec,
            "render_context": {"TODAY": today},
            "strict": strict,
        },
        idempotency_key=str(uuid.uuid4()),
        trace_id=trace_id,
    )
    v_out = v.output_or_error if hasattr(v, "output_or_error") else v
    if not isinstance(v_out, dict) or not v_out.get("valid", False):
        raise ToolExecutionError(f"DocumentSpec validation failed: {v}")

    out = registry.execute(
        "docx_generate_from_spec",
        {
            "document_spec": spec,
            "path": path,
            "render_context": {"TODAY": today},
            "strict": strict,
        },
        idempotency_key=str(uuid.uuid4()),
        trace_id=trace_id,
    )
    out_data = out.output_or_error if hasattr(out, "output_or_error") else out
    if not isinstance(out_data, dict) or "path" not in out_data:
        raise ToolExecutionError(f"docx_generate_from_spec failed: {out}")

    return {
        "path": out_data["path"],
        "bytes_written": int(out_data.get("bytes_written", 0)),
        "validation_warnings": v_out.get("warnings", []),
    }


def _build_resume_ats_document_spec(resume: TailoredResume) -> Dict[str, Any]:
    # Contact lines (single column)
    contact_parts = [p for p in [resume.header.location, resume.header.phone, resume.header.email] if p]
    contact_line = " | ".join(contact_parts)

    # Skills as paragraphs: "Category: item1, item2"
    skill_paragraphs: List[str] = []
    for name, items in resume.skills.items():
        if not isinstance(items, list):
            continue
        skill_paragraphs.append(f"{name}: {', '.join(items)}")

    # Experience items
    exp_items: List[Dict[str, Any]] = []
    for r in resume.experience:
        dates = r.dates or _format_dates(r.start_date, r.end_date)
        bullets: List[str] = []
        for b in r.bullets:
            if isinstance(b, str):
                bullets.append(b)
            elif isinstance(b, dict):
                text = b.get("text")
                if isinstance(text, str):
                    bullets.append(text)
        exp_items.append(
            {
                "title": r.title,
                "company": r.company,
                "location": r.location or "",
                "dates": dates,
                "bullets": bullets,
            }
        )

    # Education lines
    edu_lines: List[str] = []
    for e in resume.education:
        dates = e.dates or _format_dates(e.start_date, e.end_date)
        parts = [e.degree, e.school]
        if e.location:
            parts.append(e.location)
        if dates:
            parts.append(dates)
        edu_lines.append(" | ".join([p for p in parts if p]))

    # Certifications lines
    cert_lines: List[str] = []
    for c in resume.certifications:
        parts = [c.name]
        if c.issuer:
            parts.append(c.issuer)
        if c.year:
            parts.append(c.year)
        cert_lines.append(" - ".join([p for p in parts if p]))

    # Projects
    project_items: List[Dict[str, Any]] = []
    for p in resume.projects:
        project_items.append(
            {
                "name": p.name,
                "url": p.url or "",
                "highlights": list(p.highlights),
            }
        )
    projects_section = [1] if project_items else []

    tokens: Dict[str, Any] = {
        "name": resume.header.name,
        "headline": "",
        "contact_line": contact_line,
        "linkedin": resume.header.linkedin or "",
        "github": resume.header.github or "",
        "summary": resume.summary,
        "skill_paragraphs": skill_paragraphs,
        "experience": exp_items,
        "education_lines": edu_lines,
        "cert_lines": cert_lines,
        "projects": project_items,
        "projects_section": projects_section,
    }

    return {
        "doc_type": "resume_ats_single_column",
        "version": "1.0",
        "theme": {
            "fonts": {"body": "Calibri", "heading": "Calibri"},
            "font_sizes": {"body": 11, "name": 16, "h1": 12},
            "spacing": {"line": 1.1, "para_after_pt": 3, "tight_after_heading_pt": 2},
            "page_margins_in": {"top": 0.8, "bottom": 0.8, "left": 0.9, "right": 0.9},
        },
        "tokens": tokens,
        "blocks": [
            {"type": "text", "style": "name", "text": "{{name}}"},
            {"type": "optional_paragraph", "when": "{{headline}}", "text": "{{headline}}"},
            {"type": "optional_paragraph", "when": "{{contact_line}}", "style": "contact", "text": "{{contact_line}}"},
            {"type": "optional_paragraph", "when": "{{linkedin}}", "style": "contact", "text": "LinkedIn: {{linkedin}}"},
            {"type": "optional_paragraph", "when": "{{github}}", "style": "contact", "text": "GitHub: {{github}}"},
            {"type": "heading", "level": 1, "text": "SUMMARY"},
            {"type": "paragraph", "style": "divider", "text": "—"},
            {"type": "paragraph", "text": "{{summary}}"},
            {"type": "heading", "level": 1, "text": "SKILLS"},
            {"type": "paragraph", "style": "divider", "text": "—"},
            {"type": "repeat", "items": "{{skill_paragraphs}}", "as": "p", "template": [
                {"type": "paragraph", "text": "{{p}}"}
            ]},
            {"type": "heading", "level": 1, "text": "EXPERIENCE"},
            {"type": "paragraph", "style": "divider", "text": "—"},
            {"type": "repeat", "items": "{{experience}}", "as": "r", "template": [
                {"type": "paragraph", "style": "body_bold", "text": "{{r.title}} | {{r.company}} | {{r.location}}"},
                {"type": "paragraph", "style": "dates_right", "text": "{{r.dates}}"},
                {"type": "bullets", "items": "{{r.bullets}}"},
                {"type": "spacer"}
            ]},
            {"type": "repeat", "items": "{{projects_section}}", "as": "s", "template": [
                {"type": "heading", "level": 1, "text": "PROJECTS"},
                {"type": "paragraph", "style": "divider", "text": "—"},
                {"type": "repeat", "items": "{{projects}}", "as": "p", "template": [
                    {"type": "paragraph", "style": "body_bold", "text": "{{p.name}}"},
                    {"type": "optional_paragraph", "when": "{{p.url}}", "text": "{{p.url}}"},
                    {"type": "bullets", "items": "{{p.highlights}}"},
                    {"type": "spacer"}
                ]}
            ]},
            {"type": "heading", "level": 1, "text": "EDUCATION"},
            {"type": "paragraph", "style": "divider", "text": "—"},
            {"type": "repeat", "items": "{{education_lines}}", "as": "e", "template": [
                {"type": "paragraph", "text": "{{e}}"}
            ]},
            {"type": "heading", "level": 1, "text": "CERTIFICATIONS"},
            {"type": "paragraph", "style": "divider", "text": "—"},
            {"type": "repeat", "items": "{{cert_lines}}", "as": "c", "template": [
                {"type": "paragraph", "text": "{{c}}"}
            ]},
        ],
    }


def _format_dates(start: Optional[str], end: Optional[str]) -> str:
    s = (start or "").strip()
    e = (end or "").strip()
    if s and e:
        return f"{s} - {e}"
    if s:
        return s
    if e:
        return e
    return ""


def _get_registry_from_payload(payload: Dict[str, Any]):
    """
    If your Tool handler signature doesn't include registry, you need a way to access it.
    Options:
      1) Your ToolRegistry injects itself into payload (preferred): payload["_registry"]
      2) Global singleton registry
      3) Import from a module that holds the registry instance

    Implement the method that matches your architecture.
    """
    registry = payload.get("_registry")
    if registry is None:
        from libs.core.tool_registry import ToolExecutionError

        raise ToolExecutionError(
            "Tool registry not available in payload. Inject payload['_registry'] or adjust _get_registry_from_payload."
        )
    return registry
