from __future__ import annotations

from typing import Any, Dict, List

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


def register_coverletter_doc_spec_convert_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="coverletter_doc_spec_to_document_spec",
                description="Convert a CoverLetterDocSpec JSON into a DocumentSpec JSON",
                usage_guidance=(
                    "Provide coverletter_doc_spec. Returns document_spec suitable for "
                    "document_spec_validate and docx_generate_from_spec. If omitted, "
                    "coverletter_doc_spec is resolved from memory (coverletter_doc_spec:latest)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"coverletter_doc_spec": {"type": "object"}},
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
            handler=_coverletter_doc_spec_to_document_spec,
        )
    )


def _coverletter_doc_spec_to_document_spec(payload: Dict[str, Any]) -> Dict[str, Any]:
    from libs.core.tool_registry import ToolExecutionError

    spec = payload.get("coverletter_doc_spec")
    if not isinstance(spec, dict):
        raise ToolExecutionError(
            "coverletter_doc_spec missing (not found in memory). Provide coverletter_doc_spec explicitly."
        )

    defaults = spec.get("defaults", {})
    page = spec.get("page", {})
    styles = spec.get("styles", {})
    theme = _build_theme(defaults, page, styles)

    blocks: List[Dict[str, Any]] = []
    content = spec.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "header":
                blocks.extend(_convert_header(item))
            elif item_type == "paragraph":
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                block: Dict[str, Any] = {"type": "paragraph", "text": text.strip()}
                style = item.get("style")
                if isinstance(style, str) and style.strip():
                    block["style"] = style.strip()
                blocks.append(block)

    return {
        "document_spec": {
            "doc_type": "coverletter_doc_spec_converted",
            "version": "1.0",
            "theme": theme,
            "tokens": {},
            "blocks": blocks,
        }
    }


def _build_theme(
    defaults: Dict[str, Any], page: Dict[str, Any], styles: Dict[str, Any]
) -> Dict[str, Any]:
    font_family = defaults.get("font_family", "Calibri")
    font_size = defaults.get("font_size_pt", 11)
    line_spacing = defaults.get("line_spacing", 1.15)
    margins = {}
    margins_in = page.get("margins_in") if isinstance(page, dict) else {}
    if isinstance(margins_in, dict):
        for key in ("top", "bottom", "left", "right"):
            value = margins_in.get(key)
            if isinstance(value, (int, float)):
                margins[key] = value

    heading_size = None
    if isinstance(styles, dict):
        name_style = styles.get("name")
        if isinstance(name_style, dict):
            heading_size = name_style.get("size_pt")

    theme = {
        "fonts": {"body": font_family, "heading": font_family},
        "font_sizes": {"body": font_size},
        "spacing": {"line": line_spacing, "para_after_pt": 8},
        "page_margins_in": margins or {"top": 0.8, "bottom": 0.8, "left": 0.9, "right": 0.9},
    }
    if isinstance(heading_size, (int, float)):
        theme["font_sizes"]["h1"] = heading_size
    return theme


def _convert_header(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    header_blocks = item.get("blocks")
    if not isinstance(header_blocks, list):
        return blocks
    for block in header_blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        out: Dict[str, Any] = {"type": "text", "text": text.strip()}
        style = block.get("style")
        if isinstance(style, str):
            if style == "name":
                out["style"] = "cover_letter_name"
            else:
                out["style"] = style
        blocks.append(out)
    return blocks
