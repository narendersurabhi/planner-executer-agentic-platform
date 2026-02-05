from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


class ResumeBullet(BaseModel):
    text: str
    source_experience_index: int | None = None


class ResumeExperience(BaseModel):
    title: str
    company: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    bullets: List[ResumeBullet]


class ResumeEducation(BaseModel):
    degree: str
    institution: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ResumeCertification(BaseModel):
    name: str
    issuer: str | None = None
    year: str | None = None


class ResumeSkillCategory(BaseModel):
    name: str
    items: List[str]


class TailoredResume(BaseModel):
    full_name: str
    headline: str | None = None
    location: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    summary: str
    skills: List[ResumeSkillCategory]
    experience: List[ResumeExperience]
    education: List[ResumeEducation]
    certifications: List[ResumeCertification] = []
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


def _build_resume_ats_document_spec(resume: TailoredResume) -> Dict[str, Any]:
    # Contact lines (single column)
    contact_parts = [p for p in [resume.location, resume.phone, resume.email] if p]
    contact_line = " | ".join(contact_parts)

    # Skills as paragraphs: "Category: item1, item2"
    skill_paragraphs: List[str] = []
    for cat in resume.skills:
        items = ", ".join(cat.items)
        skill_paragraphs.append(f"{cat.name}: {items}")

    # Experience items
    exp_items: List[Dict[str, Any]] = []
    for r in resume.experience:
        dates = _format_dates(r.start_date, r.end_date)
        exp_items.append(
            {
                "title": r.title,
                "company": r.company,
                "location": r.location or "",
                "dates": dates,
                "bullets": [b.text for b in r.bullets],
            }
        )

    # Education lines
    edu_lines: List[str] = []
    for e in resume.education:
        dates = _format_dates(e.start_date, e.end_date)
        parts = [e.degree, e.institution]
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

    tokens: Dict[str, Any] = {
        "name": resume.full_name,
        "headline": resume.headline or "",
        "contact_line": contact_line,
        "linkedin": resume.linkedin_url or "",
        "github": resume.github_url or "",
        "summary": resume.summary,
        "skill_paragraphs": skill_paragraphs,
        "experience": exp_items,
        "education_lines": edu_lines,
        "cert_lines": cert_lines,
    }

    return {
        "doc_type": "resume_ats_single_column",
        "version": "1.0",
        "theme": {
            "fonts": {"body": "Calibri", "heading": "Calibri"},
            "font_sizes": {"body": 11, "name": 16, "h1": 12},
            "spacing": {"line": 1.12, "para_after_pt": 6, "tight_after_heading_pt": 2},
            "page_margins_in": {"top": 0.8, "bottom": 0.8, "left": 0.9, "right": 0.9},
        },
        "tokens": tokens,
        "blocks": [
            {"type": "text", "style": "name", "text": "{{name}}"},
            {"type": "optional_paragraph", "when": "{{headline}}", "text": "{{headline}}"},
            {"type": "optional_paragraph", "when": "{{contact_line}}", "text": "{{contact_line}}"},
            {"type": "optional_paragraph", "when": "{{linkedin}}", "text": "LinkedIn: {{linkedin}}"},
            {"type": "optional_paragraph", "when": "{{github}}", "text": "GitHub: {{github}}"},
            {"type": "heading", "level": 1, "text": "SUMMARY"},
            {"type": "paragraph", "text": "{{summary}}"},
            {"type": "heading", "level": 1, "text": "SKILLS"},
            {"type": "repeat", "items": "{{skill_paragraphs}}", "as": "p", "template": [
                {"type": "paragraph", "text": "{{p}}"}
            ]},
            {"type": "heading", "level": 1, "text": "EXPERIENCE"},
            {"type": "repeat", "items": "{{experience}}", "as": "r", "template": [
                {"type": "paragraph", "style": "body_bold", "text": "{{r.title}} | {{r.company}} | {{r.location}} | {{r.dates}}"},
                {"type": "bullets", "items": "{{r.bullets}}"},
                {"type": "spacer"}
            ]},
            {"type": "heading", "level": 1, "text": "EDUCATION"},
            {"type": "repeat", "items": "{{education_lines}}", "as": "e", "template": [
                {"type": "paragraph", "text": "{{e}}"}
            ]},
            {"type": "heading", "level": 1, "text": "CERTIFICATIONS"},
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
