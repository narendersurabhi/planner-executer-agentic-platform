from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from docx import Document
from docx.shared import Inches, Pt

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


def _artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", "/shared/artifacts"))


def register_docx_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="docx_generate_from_spec",
                description="Generate an ATS-friendly .docx document from a DocumentSpec JSON",
                usage_guidance=(
                    "Provide document_spec and path (required, relative .docx filename), "
                    "render_context (merged into tokens), and strict (default true). "
                    "Supported blocks: text, paragraph, heading, bullets, spacer, "
                    "optional_paragraph, repeat."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "path": {"type": "string"},
                        "render_context": {"type": "object"},
                        "strict": {"type": "boolean"},
                    },
                    "required": ["document_spec", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "bytes_written": {"type": "integer"},
                    },
                    "required": ["path", "bytes_written"],
                },
                timeout_s=25,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.render,
            ),
            handler=_docx_generate_from_spec,
        )
    )


def _docx_generate_from_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    document_spec = payload.get("document_spec")
    if not isinstance(document_spec, dict):
        _tool_error("document_spec must be an object")

    path = payload.get("path")
    if not isinstance(path, str) or not path:
        _tool_error("path must be a non-empty string")
    if not path.endswith(".docx"):
        _tool_error("path must end with .docx")
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

    output_path = _safe_join_artifacts_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    theme = document_spec.get("theme", {})
    tokens = document_spec.get("tokens", {})
    if tokens is None:
        tokens = {}
    if not isinstance(tokens, dict):
        _tool_error("document_spec.tokens must be an object")

    merged_tokens = {**tokens, **render_context}
    context = {"tokens": merged_tokens, **merged_tokens}

    document = Document()
    _apply_theme(document, theme if isinstance(theme, dict) else {})

    blocks = document_spec.get("blocks", [])
    if not isinstance(blocks, list):
        _tool_error("document_spec.blocks must be an array")

    _render_blocks(document, blocks, context, strict)

    document.save(output_path)
    bytes_written = output_path.stat().st_size
    return {"path": str(output_path), "bytes_written": int(bytes_written)}


def _safe_join_artifacts_path(path: str) -> Path:
    artifacts_dir = _artifacts_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    candidate = (artifacts_dir / path).resolve()
    root = artifacts_dir.resolve()
    if not str(candidate).startswith(str(root) + os.sep) and candidate != root:
        _tool_error("Invalid path outside /shared/artifacts")
    return candidate


def _apply_theme(document: Document, theme: Dict[str, Any]) -> None:
    fonts = theme.get("fonts", {}) if isinstance(theme.get("fonts", {}), dict) else {}
    font_sizes = (
        theme.get("font_sizes", {}) if isinstance(theme.get("font_sizes", {}), dict) else {}
    )
    body_font = fonts.get("body", "Calibri")
    body_size = font_sizes.get("body", 11)

    normal_style = document.styles["Normal"]
    normal_style.font.name = body_font
    normal_style.font.size = Pt(float(body_size))

    margins = (
        theme.get("page_margins_in", {})
        if isinstance(theme.get("page_margins_in", {}), dict)
        else {}
    )
    section = document.sections[0]
    for key, attr in (
        ("top", "top_margin"),
        ("bottom", "bottom_margin"),
        ("left", "left_margin"),
        ("right", "right_margin"),
    ):
        value = margins.get(key)
        if value is not None:
            section.__setattr__(attr, Inches(float(value)))


def _render_blocks(
    document: Document, blocks: Iterable[Dict[str, Any]], context: Dict[str, Any], strict: bool
) -> None:
    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type in {"text", "paragraph"}:
            text = _render_text(str(block.get("text", "")), context, strict)
            _add_paragraph(document, text, block.get("style"))

        elif block_type == "heading":
            level = block.get("level", 1)
            if not isinstance(level, int) or level < 1 or level > 3:
                level = 1
            text = _render_text(str(block.get("text", "")), context, strict)
            document.add_heading(text, level=level)

        elif block_type == "bullets":
            items = _resolve_items(block.get("items"), context, strict)
            for item in items:
                text = _render_text(str(item), context, strict)
                _add_paragraph(document, text, block.get("style"), bullet=True)

        elif block_type == "spacer":
            document.add_paragraph("")

        elif block_type == "optional_paragraph":
            when_value = block.get("when")
            should_render = _evaluate_condition(when_value, context, strict)
            if should_render:
                text = _render_text(str(block.get("text", "")), context, strict)
                _add_paragraph(document, text, block.get("style"))

        elif block_type == "repeat":
            items = _resolve_items(block.get("items"), context, strict)
            alias = block.get("as", "item")
            template = block.get("template", [])
            if not isinstance(alias, str) or not alias:
                alias = "item"
            if not isinstance(template, list):
                continue
            for item in items:
                scoped_context = dict(context)
                scoped_context[str(alias)] = item
                _render_blocks(document, template, scoped_context, strict)

        else:
            # Ignore unknown blocks for robustness
            continue


def _add_paragraph(document: Document, text: str, style_hint: Any, bullet: bool = False) -> None:
    paragraph = document.add_paragraph()
    if bullet:
        paragraph.style = "List Bullet"
    run = paragraph.add_run(text)
    if style_hint in {"name", "body_bold"}:
        run.bold = True


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
        # If lists show up in-line, join them
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)

    rendered = re.sub(r"\{\{\s*(.*?)\s*\}\}", replace, text)
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
        # token is a string key
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
