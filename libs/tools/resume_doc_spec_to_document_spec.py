from __future__ import annotations

from typing import Any, Dict, List

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


def register_resume_doc_spec_convert_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="resume_doc_spec_to_document_spec",
                description="Convert a ResumeDocSpec JSON into a DocumentSpec JSON",
                usage_guidance=(
                    "Provide resume_doc_spec. Returns document_spec suitable for document_spec_validate "
                    "and docx_generate_from_spec."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"resume_doc_spec": {"type": "object"}},
                    "required": ["resume_doc_spec"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"document_spec": {"type": "object"}},
                    "required": ["document_spec"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=_resume_doc_spec_to_document_spec,
        )
    )


def _resume_doc_spec_to_document_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    from libs.core.tool_registry import ToolExecutionError

    resume_spec = payload.get("resume_doc_spec")
    if not isinstance(resume_spec, dict):
        raise ToolExecutionError("resume_doc_spec must be an object")

    defaults = resume_spec.get("defaults", {})
    page = resume_spec.get("page", {})
    styles = resume_spec.get("styles", {})

    theme = _build_theme(defaults, page, styles)
    blocks = _convert_content(resume_spec.get("content", []))

    document_spec = {
        "doc_type": "resume_doc_spec_converted",
        "version": "1.0",
        "theme": theme,
        "tokens": {},
        "blocks": blocks,
    }

    return {"document_spec": document_spec}


def _build_theme(
    defaults: Dict[str, Any], page: Dict[str, Any], styles: Dict[str, Any]
) -> Dict[str, Any]:
    font_family = defaults.get("font_family", "Calibri")
    font_size = defaults.get("font_size_pt", 11)
    line_spacing = defaults.get("line_spacing", 1.05)
    margins = {}
    margins_in = page.get("margins_in") if isinstance(page, dict) else {}
    if isinstance(margins_in, dict):
        for key in ("top", "bottom", "left", "right"):
            value = margins_in.get(key)
            if isinstance(value, (int, float)):
                margins[key] = value

    heading_size = None
    if isinstance(styles, dict):
        section_heading = styles.get("section_heading")
        if isinstance(section_heading, dict):
            heading_size = section_heading.get("size_pt")

    theme = {
        "fonts": {"body": font_family, "heading": font_family},
        "font_sizes": {"body": font_size},
        "spacing": {"line": line_spacing, "para_after_pt": 3, "tight_after_heading_pt": 2},
        "page_margins_in": margins or {"top": 0.8, "bottom": 0.8, "left": 0.9, "right": 0.9},
    }
    if isinstance(heading_size, (int, float)):
        theme["font_sizes"]["h1"] = heading_size
    return theme


def _convert_content(content: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(content, list):
        return blocks

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "header":
            blocks.extend(_convert_header(item))
        elif item_type == "section_heading":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "heading", "level": 1, "text": text.strip()})
                blocks.append({"type": "paragraph", "style": "divider", "text": "—"})
        elif item_type == "paragraph":
            text = item.get("text")
            if isinstance(text, str):
                block = {"type": "paragraph", "text": text}
                style = item.get("style")
                if isinstance(style, str):
                    block["style"] = style
                blocks.append(block)
        elif item_type == "definition_list":
            blocks.extend(_convert_definition_list(item))
        elif item_type == "role":
            blocks.extend(_convert_role(item))
        elif item_type == "education":
            blocks.extend(_convert_education(item))
        elif item_type == "bullets":
            bullets = item.get("items")
            if isinstance(bullets, list):
                blocks.append({"type": "bullets", "items": bullets})
        else:
            # ignore unsupported blocks to keep conversion resilient
            continue

    return blocks


def _convert_header(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    header_blocks = item.get("blocks")
    if not isinstance(header_blocks, list):
        return blocks
    for block in header_blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        out = {"type": "text", "text": text}
        style = block.get("style")
        if isinstance(style, str):
            out["style"] = style
        blocks.append(out)
    return blocks


def _convert_definition_list(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    items = item.get("items")
    if not isinstance(items, list):
        return blocks
    for entry in items:
        if not isinstance(entry, dict):
            continue
        term = entry.get("term")
        definition = entry.get("definition")
        if not isinstance(term, str) or not isinstance(definition, str):
            continue
        blocks.append({"type": "paragraph", "text": f"{term}: {definition}"})
    return blocks


def _convert_role(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    title = item.get("title")
    company = item.get("company")
    location = item.get("location")
    dates = item.get("dates")
    if isinstance(title, str) or isinstance(company, str):
        parts = [p for p in [title, company, location] if isinstance(p, str) and p.strip()]
        if parts:
            blocks.append(
                {"type": "paragraph", "style": "body_bold", "text": " | ".join(parts)}
            )
    if isinstance(dates, str) and dates.strip():
        blocks.append({"type": "paragraph", "style": "dates_right", "text": dates.strip()})
    bullets = item.get("bullets")
    if isinstance(bullets, list):
        blocks.append({"type": "bullets", "items": bullets})
        blocks.append({"type": "spacer"})
    return blocks


def _convert_education(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    degree = item.get("degree")
    school = item.get("school")
    location = item.get("location")
    dates = item.get("dates")
    parts = [p for p in [degree, school, location, dates] if isinstance(p, str) and p.strip()]
    if parts:
        blocks.append({"type": "paragraph", "text": " | ".join(parts)})
    return blocks
