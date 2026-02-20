from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


_PLACEHOLDER_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def register_document_spec_tools(registry) -> None:
    from libs.core.tool_registry import Tool

    registry.register(
        Tool(
            spec=ToolSpec(
                name="document_spec_validate",
                description="Validate a DocumentSpec JSON for schema, supported blocks, and placeholder sanity",
                usage_guidance=(
                    "Provide document_spec and optional render_context merged into tokens. "
                    "strict=true fails on unresolved placeholders (where resolvable). "
                    "Validates supported blocks: text, paragraph, heading, bullets, spacer, "
                    "optional_paragraph, repeat. If document_spec is omitted, it is resolved from "
                    "memory (document_spec:latest)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_spec": {"type": "object"},
                        "render_context": {"type": "object"},
                        "strict": {"type": "boolean"},
                        "allowed_block_types": {"type": "array", "items": {"type": "string"}},
                        "max_depth": {"type": "integer"},
                        "max_blocks": {"type": "integer"},
                    },
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
                timeout_s=10,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.validate,
            ),
            handler=_document_spec_validate,
        )
    )


def _document_spec_validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    spec = payload.get("document_spec")
    if not isinstance(spec, dict):
        from libs.core.tool_registry import ToolExecutionError

        raise ToolExecutionError(
            "document_spec missing (not found in memory). Provide document_spec explicitly."
        )

    render_context = payload.get("render_context", {})
    if render_context is None:
        render_context = {}
    if not isinstance(render_context, dict):
        return _result(
            False, [err("/render_context", "render_context must be an object")], [], 0, 0, 0
        )

    strict = payload.get("strict", True)
    if not isinstance(strict, bool):
        return _result(False, [err("/strict", "strict must be a boolean")], [], 0, 0, 0)

    allowed = payload.get("allowed_block_types")
    allowed_set = set(allowed) if isinstance(allowed, list) else _default_allowed_block_types()

    max_depth = payload.get("max_depth", 6)
    max_blocks = payload.get("max_blocks", 2000)
    if not isinstance(max_depth, int) or max_depth < 1:
        max_depth = 6
    if not isinstance(max_blocks, int) or max_blocks < 1:
        max_blocks = 2000

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    # Top-level checks
    blocks = spec.get("blocks")
    if not isinstance(blocks, list):
        errors.append(err("/blocks", "document_spec.blocks must be an array"))
        return _result(False, errors, warnings, 0, 0, 0)
    if len(blocks) > max_blocks:
        errors.append(err("/blocks", f"Too many blocks: {len(blocks)} > max_blocks={max_blocks}"))
        return _result(False, errors, warnings, len(blocks), 0, 0)

    tokens = spec.get("tokens", {})
    if tokens is None:
        tokens = {}
    if not isinstance(tokens, dict):
        errors.append(err("/tokens", "document_spec.tokens must be an object"))
        return _result(False, errors, warnings, len(blocks), 0, 0)

    merged_tokens = {**tokens, **render_context}
    context = {"tokens": merged_tokens, **merged_tokens}

    # Validate blocks recursively, collect placeholders
    placeholder_count = 0
    unresolved_count = 0
    block_count = 0

    def walk_block_list(blks: List[Any], path: str, depth: int, alias: Optional[str]) -> None:
        nonlocal placeholder_count, unresolved_count, block_count
        if depth > max_depth:
            errors.append(err(path, f"Max depth exceeded: {depth} > {max_depth}"))
            return

        for i, b in enumerate(blks):
            bpath = f"{path}/{i}"
            if not isinstance(b, dict):
                warnings.append(err(bpath, "Non-object block ignored"))
                continue

            btype = b.get("type")
            if not isinstance(btype, str) or not btype:
                errors.append(err(bpath + "/type", "Block must have non-empty string type"))
                continue

            block_count += 1
            if block_count > max_blocks:
                errors.append(err("/blocks", f"Too many blocks after expansion: > {max_blocks}"))
                return

            if btype not in allowed_set:
                errors.append(err(bpath + "/type", f"Unknown block type: {btype}"))
                continue

            # Type-specific validation
            if btype in {"text", "paragraph"}:
                if "text" not in b or not isinstance(b.get("text"), str):
                    errors.append(err(bpath + "/text", "text/paragraph requires text: string"))
                placeholder_count += _count_placeholders(b.get("text"))
                unresolved_count += _check_placeholders(
                    b.get("text"), context, strict, errors, warnings, bpath + "/text", alias
                )

            elif btype == "heading":
                if "text" not in b or not isinstance(b.get("text"), str):
                    errors.append(err(bpath + "/text", "heading requires text: string"))
                level = b.get("level", 1)
                if not isinstance(level, int) or level < 1 or level > 3:
                    errors.append(err(bpath + "/level", "heading.level must be int in [1..3]"))
                placeholder_count += _count_placeholders(b.get("text"))
                unresolved_count += _check_placeholders(
                    b.get("text"), context, strict, errors, warnings, bpath + "/text", alias
                )

            elif btype == "bullets":
                items = b.get("items")
                if not (isinstance(items, list) or isinstance(items, str) or items is None):
                    errors.append(
                        err(bpath + "/items", "bullets.items must be array, string ref, or null")
                    )
                if isinstance(items, str):
                    # warn if not an exact ref (ATS mode expects "{{...}}" when string)
                    if not _is_exact_ref(items):
                        warnings.append(
                            err(
                                bpath + "/items",
                                "bullets.items string should be exact '{{...}}' ref",
                            )
                        )
                    placeholder_count += _count_placeholders(items)
                    unresolved_count += _check_placeholders(
                        items, context, strict, errors, warnings, bpath + "/items", alias
                    )
                if isinstance(items, list):
                    for j, it in enumerate(items):
                        if isinstance(it, str):
                            placeholder_count += _count_placeholders(it)
                            unresolved_count += _check_placeholders(
                                it, context, strict, errors, warnings, f"{bpath}/items/{j}", alias
                            )

            elif btype == "optional_paragraph":
                when_val = b.get("when")
                if "text" not in b or not isinstance(b.get("text"), str):
                    errors.append(err(bpath + "/text", "optional_paragraph requires text: string"))
                if not (
                    isinstance(when_val, bool) or isinstance(when_val, str) or when_val is None
                ):
                    errors.append(
                        err(bpath + "/when", "optional_paragraph.when must be bool or string ref")
                    )
                placeholder_count += _count_placeholders(b.get("text"))
                unresolved_count += _check_placeholders(
                    b.get("text"), context, strict, errors, warnings, bpath + "/text", alias
                )
                if isinstance(when_val, str):
                    if not _is_exact_ref(when_val):
                        warnings.append(
                            err(
                                bpath + "/when",
                                "optional_paragraph.when string should be exact '{{...}}' ref",
                            )
                        )
                    placeholder_count += _count_placeholders(when_val)
                    unresolved_count += _check_placeholders(
                        when_val, context, strict, errors, warnings, bpath + "/when", alias
                    )

            elif btype == "repeat":
                items = b.get("items")
                alias_name = b.get("as")
                template = b.get("template")
                if not (isinstance(items, list) or isinstance(items, str) or items is None):
                    errors.append(
                        err(bpath + "/items", "repeat.items must be array, string ref, or null")
                    )
                if not isinstance(alias_name, str) or not alias_name:
                    errors.append(err(bpath + "/as", "repeat.as must be a non-empty string"))
                    alias_name = "item"
                if not isinstance(template, list):
                    errors.append(err(bpath + "/template", "repeat.template must be an array"))
                    template = []

                if isinstance(items, str):
                    if not _is_exact_ref(items):
                        warnings.append(
                            err(
                                bpath + "/items",
                                "repeat.items string should be exact '{{...}}' ref",
                            )
                        )
                    placeholder_count += _count_placeholders(items)
                    unresolved_count += _check_placeholders(
                        items, context, strict, errors, warnings, bpath + "/items", alias
                    )

                # For template placeholders, we allow alias references without requiring resolution
                walk_block_list(template, bpath + "/template", depth + 1, alias_name)

            elif btype == "spacer":
                # no required fields
                pass

            # Simple ATS warnings
            if btype == "heading" and isinstance(b.get("text"), str):
                h = b["text"].strip().upper()
                if h and h not in {
                    "SUMMARY",
                    "SKILLS",
                    "EXPERIENCE",
                    "EDUCATION",
                    "CERTIFICATIONS",
                    "OPEN SOURCE",
                    "PROJECTS",
                }:
                    warnings.append(
                        err(bpath + "/text", f"Non-standard heading for ATS: '{b['text']}'")
                    )

            # Optional long paragraph warning
            if (
                btype in {"text", "paragraph"}
                and isinstance(b.get("text"), str)
                and len(b["text"]) > 500
            ):
                warnings.append(
                    err(bpath + "/text", "Very long paragraph (>500 chars); consider splitting")
                )

    walk_block_list(blocks, "/blocks", 1, None)

    valid = len(errors) == 0
    return _result(valid, errors, warnings, block_count, placeholder_count, unresolved_count)


def _default_allowed_block_types() -> Set[str]:
    return {"text", "paragraph", "heading", "bullets", "spacer", "optional_paragraph", "repeat"}


def _count_placeholders(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(_PLACEHOLDER_RE.findall(text))


def _is_exact_ref(s: str) -> bool:
    m = re.fullmatch(r"\{\{\s*(.*?)\s*\}\}", s.strip())
    return m is not None and s.strip().startswith("{{") and s.strip().endswith("}}")


def _check_placeholders(
    text: Any,
    context: Dict[str, Any],
    strict: bool,
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    path: str,
    alias: Optional[str],
) -> int:
    """
    Returns unresolved placeholder count increment for this text.
    Alias behavior:
      - If placeholder starts with '<alias>.' or equals '<alias>' we treat it as "allowed" (unknown shape).
      - Otherwise, we require resolvable if strict=true, warning if strict=false.
    """
    if not isinstance(text, str):
        return 0

    unresolved = 0
    for raw_expr in _PLACEHOLDER_RE.findall(text):
        expr = (raw_expr or "").strip()
        if not expr:
            # malformed placeholder
            msg = "Empty placeholder expression"
            (errors if strict else warnings).append(err(path, msg))
            unresolved += 1
            continue

        # allow alias refs inside repeat templates
        if alias and (
            expr == alias or expr.startswith(alias + ".") or expr.startswith(alias + "[")
        ):
            continue

        ok = _can_resolve(expr, context)
        if not ok:
            msg = f"Unresolved placeholder: {expr}"
            (errors if strict else warnings).append(err(path, msg))
            unresolved += 1

    return unresolved


def _can_resolve(expr: str, context: Dict[str, Any]) -> bool:
    try:
        val = _resolve_path(expr, context)
        return val is not None
    except Exception:
        return False


def _resolve_path(expr: str, context: Dict[str, Any]) -> Any:
    tokens = _tokenize_path(expr)
    current: Any = context
    for token in tokens:
        if isinstance(token, int):
            if isinstance(current, list) and 0 <= token < len(current):
                current = current[token]
                continue
            return None
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
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


def err(path: str, message: str) -> Dict[str, Any]:
    return {"path": path, "message": message}


def _result(
    valid: bool,
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    block_count: int,
    placeholder_count: int,
    unresolved_count: int,
) -> Dict[str, Any]:
    return {
        "valid": bool(valid),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "block_count": int(block_count),
            "placeholder_count": int(placeholder_count),
            "unresolved_count": int(unresolved_count),
        },
    }
