from __future__ import annotations

from typing import Any, Dict, List

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


def register_resume_doc_spec_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="resume_doc_spec_validate",
                description="Validate a ResumeDocSpec JSON for required structure and types",
                usage_guidance=(
                    "Provide resume_doc_spec. strict=true treats unknown section types as errors. "
                    "Returns valid/errors/warnings and basic stats."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "resume_doc_spec": {"type": "object"},
                        "strict": {"type": "boolean"},
                    },
                    "required": ["resume_doc_spec"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "valid": {"type": "boolean"},
                        "errors": {"type": "array", "items": {"type": "object"}},
                        "warnings": {"type": "array", "items": {"type": "object"}},
                        "stats": {"type": "object"},
                    },
                    "required": ["valid", "errors", "warnings", "stats"],
                },
                memory_reads=["job_context", "task_outputs"],
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.validate,
            ),
            handler=_resume_doc_spec_validate,
        )
    )


def _resume_doc_spec_validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    spec = payload.get("resume_doc_spec")
    if not isinstance(spec, dict):
        return _result(False, [err("/", "resume_doc_spec must be an object")], [], 0)

    strict = payload.get("strict", False)
    if not isinstance(strict, bool):
        strict = False

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    _require_str(spec, "schema_version", "/schema_version", errors)
    _require_str(spec, "doc_type", "/doc_type", errors)
    _require_str(spec, "title", "/title", errors)

    page = spec.get("page")
    if not isinstance(page, dict):
        errors.append(err("/page", "page must be an object"))
    else:
        _require_str(page, "size", "/page/size", errors)
        margins = page.get("margins_in")
        if not isinstance(margins, dict):
            errors.append(err("/page/margins_in", "margins_in must be an object"))
        else:
            for key in ("top", "right", "bottom", "left"):
                value = margins.get(key)
                if not isinstance(value, (int, float)):
                    errors.append(err(f"/page/margins_in/{key}", "must be a number"))

    defaults = spec.get("defaults")
    if not isinstance(defaults, dict):
        errors.append(err("/defaults", "defaults must be an object"))
    else:
        _require_str(defaults, "font_family", "/defaults/font_family", errors)
        _require_number(defaults, "font_size_pt", "/defaults/font_size_pt", errors)
        _require_number(defaults, "line_spacing", "/defaults/line_spacing", errors)

    styles = spec.get("styles")
    if not isinstance(styles, dict):
        errors.append(err("/styles", "styles must be an object"))

    content = spec.get("content")
    if not isinstance(content, list):
        errors.append(err("/content", "content must be an array"))
        content = []

    allowed_types = {
        "header",
        "section_heading",
        "paragraph",
        "definition_list",
        "role",
        "education",
        "bullets",
    }

    for idx, block in enumerate(content):
        path = f"/content/{idx}"
        if not isinstance(block, dict):
            errors.append(err(path, "content item must be an object"))
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str):
            errors.append(err(f"{path}/type", "type must be a string"))
            continue
        if block_type not in allowed_types:
            message = f"unsupported content type: {block_type}"
            if strict:
                errors.append(err(f"{path}/type", message))
            else:
                warnings.append(err(f"{path}/type", message))
            continue
        if block_type == "header":
            blocks = block.get("blocks")
            if not isinstance(blocks, list):
                errors.append(err(f"{path}/blocks", "blocks must be an array"))
                continue
            for b_idx, b in enumerate(blocks):
                b_path = f"{path}/blocks/{b_idx}"
                if not isinstance(b, dict):
                    errors.append(err(b_path, "block must be an object"))
                    continue
                _require_str(b, "type", f"{b_path}/type", errors)
                _require_str(b, "text", f"{b_path}/text", errors)
        elif block_type == "section_heading":
            _require_str(block, "text", f"{path}/text", errors)
        elif block_type == "paragraph":
            _require_str(block, "text", f"{path}/text", errors)
        elif block_type == "definition_list":
            items = block.get("items")
            if not isinstance(items, list):
                errors.append(err(f"{path}/items", "items must be an array"))
                continue
            for i_idx, item in enumerate(items):
                i_path = f"{path}/items/{i_idx}"
                if not isinstance(item, dict):
                    errors.append(err(i_path, "item must be an object"))
                    continue
                _require_str(item, "term", f"{i_path}/term", errors)
                _require_str(item, "definition", f"{i_path}/definition", errors)
        elif block_type == "role":
            _require_str(block, "company", f"{path}/company", errors)
            _require_str(block, "location", f"{path}/location", errors)
            _require_str(block, "title", f"{path}/title", errors)
            _require_str(block, "dates", f"{path}/dates", errors)
            bullets = block.get("bullets")
            if not isinstance(bullets, list):
                errors.append(err(f"{path}/bullets", "bullets must be an array"))
            else:
                for b_idx, bullet in enumerate(bullets):
                    if not isinstance(bullet, str):
                        errors.append(err(f"{path}/bullets/{b_idx}", "bullet must be a string"))
        elif block_type == "education":
            _require_str(block, "degree", f"{path}/degree", errors)
            _require_str(block, "school", f"{path}/school", errors)
            _require_str(block, "location", f"{path}/location", errors)
            _require_str(block, "dates", f"{path}/dates", errors)
        elif block_type == "bullets":
            items = block.get("items")
            if not isinstance(items, list):
                errors.append(err(f"{path}/items", "items must be an array"))
            else:
                for i_idx, item in enumerate(items):
                    if not isinstance(item, str):
                        errors.append(err(f"{path}/items/{i_idx}", "item must be a string"))

    return _result(len(errors) == 0, errors, warnings, len(content))


def _require_str(obj: Dict[str, Any], key: str, path: str, errors: List[Dict[str, Any]]) -> None:
    value = obj.get(key)
    if not isinstance(value, str) or value.strip() == "":
        errors.append(err(path, "must be a non-empty string"))


def _require_number(obj: Dict[str, Any], key: str, path: str, errors: List[Dict[str, Any]]) -> None:
    value = obj.get(key)
    if not isinstance(value, (int, float)):
        errors.append(err(path, "must be a number"))


def err(path: str, message: str) -> Dict[str, Any]:
    return {"path": path, "message": message}


def _result(
    valid: bool,
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    block_count: int,
) -> Dict[str, Any]:
    return {
        "valid": bool(valid),
        "errors": errors,
        "warnings": warnings,
        "stats": {"block_count": int(block_count)},
    }
