from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.tools.output_path_derivation import (
    OutputPathDerivationError,
    resolve_or_derive_output_path,
)


def _artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", "/shared/artifacts"))


def register_pdf_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="pdf_generate_from_spec",
                description="Generate a .pdf document from a DocumentSpec JSON",
                usage_guidance=(
                    "Provide path (relative .pdf filename). If omitted, a safe path is auto-derived "
                    "from topic/date/output_dir context. document_spec is resolved from memory "
                    "(document_spec:latest) unless explicitly provided. "
                    "render_context is merged into tokens, strict defaults to true. "
                    "Supported blocks: text, paragraph, heading, bullets, spacer, "
                    "optional_paragraph, repeat, page_break."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "path": {"type": "string"},
                        "render_context": {"type": "object"},
                        "strict": {"type": "boolean"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "bytes_written": {"type": "integer"},
                    },
                    "required": ["path", "bytes_written"],
                },
                memory_reads=["job_context", "task_outputs"],
                timeout_s=40,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.render,
            ),
            handler=_pdf_generate_from_spec,
        )
    )


def _pdf_generate_from_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    document_spec = payload.get("document_spec")
    if not isinstance(document_spec, dict):
        _tool_error("document_spec missing (not found in memory). Provide document_spec explicitly.")
    _ensure_validation_passed(payload)
    try:
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import (
            ListFlowable,
            ListItem,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )
    except Exception as exc:  # noqa: BLE001
        _tool_error("reportlab is not installed")
        raise AssertionError("unreachable") from exc

    try:
        path = resolve_or_derive_output_path(payload, extension="pdf")
    except OutputPathDerivationError as exc:
        _tool_error(str(exc))
    if not path.lower().endswith(".pdf"):
        _tool_error("path must end with .pdf")
    if Path(path).is_absolute():
        _tool_error("path must be relative to /shared/artifacts")

    render_context = payload.get("render_context", {})
    if render_context is None:
        render_context = {}
    if not isinstance(render_context, dict):
        _tool_error("render_context must be an object")

    strict = payload.get("strict", True)
    if not isinstance(strict, bool):
        _tool_error("strict must be a boolean")

    blocks = document_spec.get("blocks", [])
    if not isinstance(blocks, list):
        _tool_error("document_spec.blocks must be an array")

    tokens = document_spec.get("tokens", {})
    if tokens is None:
        tokens = {}
    if not isinstance(tokens, dict):
        _tool_error("document_spec.tokens must be an object")

    output_path = _safe_join_artifacts_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_tokens = {**tokens, **render_context}
    base_context = {"tokens": merged_tokens, **merged_tokens}

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "AweBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=13.5,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    heading_styles = {
        1: ParagraphStyle(
            "AweH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=4,
        ),
        2: ParagraphStyle(
            "AweH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=4,
        ),
        3: ParagraphStyle(
            "AweH3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14.5,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=3,
        ),
    }
    contact_style = ParagraphStyle(
        "AweContact",
        parent=body_style,
        fontSize=9.5,
        leading=11.5,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    section_title_style = ParagraphStyle(
        "AweSectionTitle",
        parent=heading_styles[2],
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=4,
    )
    divider_style = ParagraphStyle(
        "AweDivider",
        parent=body_style,
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=2,
    )
    role_meta_style = ParagraphStyle(
        "AweRoleMeta",
        parent=body_style,
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    role_dates_style = ParagraphStyle(
        "AweRoleDates",
        parent=body_style,
        alignment=TA_RIGHT,
        spaceAfter=2,
    )
    optional_emphasis_style = ParagraphStyle(
        "AweBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
        alignment=TA_JUSTIFY,
    )

    def style_for_hint(style_hint: Any) -> Any:
        if not isinstance(style_hint, str):
            return body_style
        normalized = style_hint.strip().lower()
        if normalized in {"name", "title"}:
            return heading_styles[1]
        if normalized in {"subtitle"}:
            return heading_styles[2]
        if normalized in {"section_title"}:
            return section_title_style
        if normalized in {"contact"}:
            return contact_style
        if normalized in {"divider"}:
            return divider_style
        if normalized in {"role_meta"}:
            return role_meta_style
        if normalized in {"dates_right"}:
            return role_dates_style
        if normalized in {"body_bold"}:
            return optional_emphasis_style
        return body_style

    flowables: list[Any] = []

    def append_paragraph(text: str, style_hint: Any = None) -> None:
        if not text.strip():
            return
        flowables.append(Paragraph(_paragraph_html(text), style_for_hint(style_hint)))

    def walk_blocks(raw_blocks: Iterable[Dict[str, Any]], context: Dict[str, Any]) -> None:
        for block in raw_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type in {"text", "paragraph"}:
                text = _render_text(str(block.get("text", "")), context, strict)
                append_paragraph(text, block.get("style"))
                continue

            if block_type == "heading":
                text = _render_text(str(block.get("text", "")), context, strict)
                level_raw = block.get("level", 1)
                level = level_raw if isinstance(level_raw, int) and level_raw in {1, 2, 3} else 1
                style = heading_styles[level]
                flowables.append(Paragraph(_paragraph_html(text), style))
                continue

            if block_type == "bullets":
                items = _resolve_items(block.get("items", []), context, strict)
                bullet_items: list[Any] = []
                for item in items:
                    item_text = _render_text(str(item), context, strict)
                    if item_text.strip():
                        bullet_items.append(
                            ListItem(Paragraph(_paragraph_html(item_text), body_style))
                        )
                if bullet_items:
                    flowables.append(
                        ListFlowable(
                            bullet_items,
                            bulletType="bullet",
                            bulletFontName="Helvetica",
                            leftIndent=14,
                            bulletDedent=8,
                            spaceBefore=2,
                            spaceAfter=6,
                        )
                    )
                continue

            if block_type == "optional_paragraph":
                should_render = _evaluate_condition(block.get("when"), context, strict)
                if should_render:
                    text = _render_text(str(block.get("text", "")), context, strict)
                    append_paragraph(text, block.get("style"))
                continue

            if block_type == "repeat":
                items = _resolve_items(block.get("items", []), context, strict)
                alias = block.get("as", "item")
                if not isinstance(alias, str) or not alias:
                    alias = "item"
                template = block.get("template", [])
                if isinstance(template, list):
                    for item in items:
                        scoped_context = dict(context)
                        scoped_context[alias] = item
                        walk_blocks(template, scoped_context)
                continue

            if block_type == "spacer":
                pt_raw = block.get("height_pt", 8)
                height_pt = float(pt_raw) if isinstance(pt_raw, (int, float)) else 8.0
                flowables.append(Spacer(1, max(1.0, height_pt)))
                continue

            if block_type == "page_break":
                flowables.append(PageBreak())
                continue

            # Ignore unknown blocks for resilience.

    walk_blocks(blocks, base_context)

    if not flowables:
        append_paragraph(" ", None)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=str(document_spec.get("title", "Document")),
    )
    doc.build(flowables)
    bytes_written = output_path.stat().st_size
    return {"path": str(output_path), "bytes_written": int(bytes_written)}


def _ensure_validation_passed(payload: Dict[str, Any]) -> None:
    validation_report = payload.get("validation_report")
    if isinstance(validation_report, dict) and validation_report.get("valid") is False:
        _tool_error(_format_validation_errors(validation_report.get("errors")))
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        _tool_error(_format_validation_errors(errors))


def _format_validation_errors(errors: Any) -> str:
    if not isinstance(errors, list) or not errors:
        return "document_spec validation failed"
    messages: list[str] = []
    for error in errors[:5]:
        if isinstance(error, dict):
            path = str(error.get("path") or "<root>")
            message = str(error.get("message") or "invalid")
            messages.append(f"{path}: {message}")
        else:
            messages.append(str(error))
    return "document_spec validation failed: " + "; ".join(messages)


def _paragraph_html(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return escaped.replace("\n", "<br/>")


def _safe_join_artifacts_path(path: str) -> Path:
    artifacts_dir = _artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidate = (artifacts_dir / path).resolve()
    root = artifacts_dir.resolve()
    if not str(candidate).startswith(str(root) + os.sep) and candidate != root:
        _tool_error("Invalid path outside /shared/artifacts")
    return candidate


def _resolve_items(value: Any, context: Dict[str, Any], strict: bool) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        resolved = _resolve_placeholder(value, context, strict, return_raw=True)
        if isinstance(resolved, list):
            return resolved
        if strict:
            _tool_error("items must resolve to an array")
        return [resolved]
    if value is None:
        return []
    if strict:
        _tool_error("items must be an array")
    return [value]


def _evaluate_condition(value: Any, context: Dict[str, Any], strict: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        resolved = _resolve_placeholder(value, context, strict, return_raw=True)
        return bool(resolved)
    if value is None:
        return False
    if strict:
        _tool_error("optional_paragraph.when must be a boolean or placeholder")
    return bool(value)


def _render_text(text: str, context: Dict[str, Any], strict: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if not expr:
            if strict:
                _tool_error("Empty placeholder")
            return match.group(0)
        value = _resolve_path(expr, context, strict)
        if value is None:
            if strict:
                _tool_error(f"Unresolved placeholder: {expr}")
            return match.group(0)
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)

    rendered = re.sub(r"\{\{\s*(.*?)\s*\}\}", replace, text)
    rendered = rendered.replace("\u2013", "-").replace("\u2014", "-")
    if strict and re.search(r"\{\{\s*(.*?)\s*\}\}", rendered):
        _tool_error("Unresolved placeholders remain after rendering")
    return rendered


def _resolve_placeholder(text: str, context: Dict[str, Any], strict: bool, return_raw: bool) -> Any:
    matches = re.findall(r"\{\{\s*(.*?)\s*\}\}", text)
    if len(matches) == 1 and text.strip() == f"{{{{{matches[0]}}}}}":
        return _resolve_path(matches[0].strip(), context, strict)
    rendered = _render_text(text, context, strict)
    return rendered if return_raw else rendered


def _resolve_path(expr: str, context: Dict[str, Any], strict: bool) -> Any:
    tokens = _tokenize_path(expr)
    current: Any = context
    for token in tokens:
        if isinstance(token, int):
            if isinstance(current, list) and 0 <= token < len(current):
                current = current[token]
                continue
            if strict:
                _tool_error(f"Index out of range: {token} for {expr}")
            return None
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if strict:
            _tool_error(f"Unknown placeholder: {expr}")
        return None
    return current


def _tokenize_path(expr: str) -> List[Any]:
    parts: List[Any] = []
    for match in re.finditer(r"([^[.\]]+)|(\[(\d+)\])", expr):
        key = match.group(1)
        index = match.group(3)
        if key is not None:
            parts.append(key)
        elif index is not None:
            parts.append(int(index))
    return parts


def _tool_error(message: str) -> None:
    from libs.core.tool_registry import ToolExecutionError

    raise ToolExecutionError(message)
