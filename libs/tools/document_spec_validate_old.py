# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# from libs.core.models import RiskLevel, ToolSpec


# _PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")
# _ALLOWED_BLOCK_TYPES = {
#     "text",
#     "paragraph",
#     "heading",
#     "bullets",
#     "spacer",
#     "optional_paragraph",
#     "repeat",
# }
# _ATS_HEADINGS = {"SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS"}


# def register_document_spec_tools(registry) -> None:
#     from libs.core.tool_registry import Tool

#     registry.register(
#         Tool(
#             spec=ToolSpec(
#                 name="document_spec_validate",
#                 description="Validate a DocumentSpec JSON for schema, supported blocks, and placeholder resolution",
#                 usage_guidance=(
#                     "Provide document_spec, optional render_context, strict, allowed_block_types, "
#                     "max_depth, and max_blocks. Returns validation errors and warnings."
#                 ),
#                 input_schema={
#                     "type": "object",
#                     "properties": {
#                         "document_spec": {"type": "object"},
#                         "render_context": {"type": "object"},
#                         "strict": {"type": "boolean"},
#                         "allowed_block_types": {"type": "array", "items": {"type": "string"}},
#                         "max_depth": {"type": "integer"},
#                         "max_blocks": {"type": "integer"},
#                     },
#                     "required": ["document_spec"],
#                 },
#                 output_schema={
#                     "type": "object",
#                     "properties": {
#                         "valid": {"type": "boolean"},
#                         "errors": {
#                             "type": "array",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {"path": {"type": "string"}, "message": {"type": "string"}},
#                                 "required": ["path", "message"],
#                             },
#                         },
#                         "warnings": {
#                             "type": "array",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {"path": {"type": "string"}, "message": {"type": "string"}},
#                                 "required": ["path", "message"],
#                             },
#                         },
#                         "stats": {
#                             "type": "object",
#                             "properties": {
#                                 "block_count": {"type": "integer"},
#                                 "placeholder_count": {"type": "integer"},
#                                 "unresolved_count": {"type": "integer"},
#                             },
#                             "required": ["block_count", "placeholder_count", "unresolved_count"],
#                         },
#                     },
#                     "required": ["valid", "errors", "warnings", "stats"],
#                 },
#                 timeout_s=10,
#                 risk_level=RiskLevel.low,
#             ),
#             handler=_document_spec_validate,
#         )
#     )


# def _document_spec_validate(payload: Dict[str, Any]) -> Dict[str, Any]:
#     document_spec = payload.get("document_spec")
#     if not isinstance(document_spec, dict):
#         _tool_error("document_spec must be an object")
#     render_context = payload.get("render_context", {})
#     if render_context is None:
#         render_context = {}
#     if not isinstance(render_context, dict):
#         _tool_error("render_context must be an object")
#     strict = payload.get("strict", True)
#     if not isinstance(strict, bool):
#         _tool_error("strict must be a boolean")
#     allowed_block_types = payload.get("allowed_block_types") or list(_ALLOWED_BLOCK_TYPES)
#     if not isinstance(allowed_block_types, list) or not all(
#         isinstance(item, str) for item in allowed_block_types
#     ):
#         _tool_error("allowed_block_types must be an array of strings")
#     max_depth = payload.get("max_depth", 6)
#     if not isinstance(max_depth, int) or max_depth < 1:
#         _tool_error("max_depth must be a positive integer")
#     max_blocks = payload.get("max_blocks", 2000)
#     if not isinstance(max_blocks, int) or max_blocks < 1:
#         _tool_error("max_blocks must be a positive integer")

#     errors: List[Dict[str, str]] = []
#     warnings: List[Dict[str, str]] = []
#     stats = _Stats()

#     _check_depth(document_spec, 0, max_depth, errors)

#     tokens = document_spec.get("tokens", {})
#     if tokens is None:
#         tokens = {}
#     if not isinstance(tokens, dict):
#         _add_error(errors, "/tokens", "tokens must be an object")
#         tokens = {}
#     theme = document_spec.get("theme")
#     if theme is not None and not isinstance(theme, dict):
#         _add_error(errors, "/theme", "theme must be an object")
#     meta = document_spec.get("meta")
#     if meta is not None and not isinstance(meta, dict):
#         _add_error(errors, "/meta", "meta must be an object")

#     blocks = document_spec.get("blocks")
#     if not isinstance(blocks, list):
#         _add_error(errors, "/blocks", "blocks must be an array")
#         blocks = []

#     context = {"tokens": {**tokens, **render_context}, **tokens, **render_context}
#     _validate_blocks(
#         blocks=blocks,
#         path="/blocks",
#         context=context,
#         strict=strict,
#         allowed_block_types=set(allowed_block_types),
#         errors=errors,
#         warnings=warnings,
#         stats=stats,
#         max_blocks=max_blocks,
#         max_depth=max_depth,
#         alias_stack=[],
#     )

#     valid = len(errors) == 0
#     return {
#         "valid": valid,
#         "errors": errors,
#         "warnings": warnings,
#         "stats": {
#             "block_count": stats.block_count,
#             "placeholder_count": stats.placeholder_count,
#             "unresolved_count": stats.unresolved_count,
#         },
#     }


# @dataclass
# class _Stats:
#     block_count: int = 0
#     placeholder_count: int = 0
#     unresolved_count: int = 0


# def _validate_blocks(
#     blocks: Iterable[Any],
#     path: str,
#     context: Dict[str, Any],
#     strict: bool,
#     allowed_block_types: Set[str],
#     errors: List[Dict[str, str]],
#     warnings: List[Dict[str, str]],
#     stats: _Stats,
#     max_blocks: int,
#     max_depth: int,
#     alias_stack: List[str],
# ) -> None:
#     for index, block in enumerate(blocks):
#         block_path = f"{path}/{index}"
#         if stats.block_count >= max_blocks:
#             _add_error(errors, block_path, f"block count exceeds max_blocks={max_blocks}")
#             return
#         stats.block_count += 1

#         if not isinstance(block, dict):
#             _add_error(errors, block_path, "block must be an object")
#             continue

#         block_type = block.get("type")
#         if not isinstance(block_type, str):
#             _add_error(errors, f"{block_path}/type", "type must be a string")
#             continue
#         if block_type not in allowed_block_types:
#             _add_error(errors, f"{block_path}/type", f"unsupported block type: {block_type}")
#             continue

#         _check_depth(block, 0, max_depth, errors, block_path)

#         if block_type in {"text", "paragraph"}:
#             _require_string(block, "text", block_path, errors)
#             _check_string_fields(block, block_path, context, strict, errors, warnings, stats, alias_stack)
#         elif block_type == "heading":
#             _require_string(block, "text", block_path, errors)
#             level = block.get("level")
#             if not isinstance(level, int) or level < 1 or level > 3:
#                 _add_error(errors, f"{block_path}/level", "level must be an integer in [1..3]")
#             _check_heading(block, block_path, warnings)
#             _check_string_fields(block, block_path, context, strict, errors, warnings, stats, alias_stack)
#         elif block_type == "bullets":
#             if "items" not in block:
#                 _add_error(errors, f"{block_path}/items", "items is required")
#             else:
#                 items = block.get("items")
#                 if isinstance(items, list):
#                     for item_index, item in enumerate(items):
#                         if isinstance(item, str):
#                             _check_placeholders(
#                                 item,
#                                 f"{block_path}/items/{item_index}",
#                                 context,
#                                 strict,
#                                 errors,
#                                 warnings,
#                                 stats,
#                                 alias_stack,
#                             )
#                             _check_long_paragraph(item, f"{block_path}/items/{item_index}", warnings)
#                         else:
#                             _add_error(errors, f"{block_path}/items/{item_index}", "items must be strings")
#                 elif isinstance(items, str):
#                     if not _is_exact_placeholder(items):
#                         _add_warning(
#                             warnings,
#                             f"{block_path}/items",
#                             "items should be a placeholder string like {{path}}",
#                         )
#                     _check_placeholders(
#                         items, f"{block_path}/items", context, strict, errors, warnings, stats, alias_stack
#                     )
#                 else:
#                     _add_error(errors, f"{block_path}/items", "items must be an array or placeholder string")
#         elif block_type == "spacer":
#             _check_string_fields(block, block_path, context, strict, errors, warnings, stats, alias_stack)
#         elif block_type == "optional_paragraph":
#             _require_string(block, "text", block_path, errors)
#             if "when" not in block:
#                 _add_error(errors, f"{block_path}/when", "when is required")
#             else:
#                 when_value = block.get("when")
#                 if isinstance(when_value, bool):
#                     pass
#                 elif isinstance(when_value, str):
#                     if not _is_exact_placeholder(when_value):
#                         _add_error(errors, f"{block_path}/when", "when must be a placeholder string or boolean")
#                     else:
#                         _check_placeholders(
#                             when_value,
#                             f"{block_path}/when",
#                             context,
#                             strict,
#                             errors,
#                             warnings,
#                             stats,
#                             alias_stack,
#                         )
#                 else:
#                     _add_error(errors, f"{block_path}/when", "when must be a boolean or placeholder string")
#             _check_string_fields(block, block_path, context, strict, errors, warnings, stats, alias_stack)
#         elif block_type == "repeat":
#             items = block.get("items")
#             alias = block.get("as")
#             template = block.get("template")
#             if items is None:
#                 _add_error(errors, f"{block_path}/items", "items is required")
#             elif isinstance(items, str):
#                 if not _is_exact_placeholder(items):
#                     _add_error(errors, f"{block_path}/items", "items must be a placeholder string or list")
#                 _check_placeholders(
#                     items, f"{block_path}/items", context, strict, errors, warnings, stats, alias_stack
#                 )
#             elif not isinstance(items, list):
#                 _add_error(errors, f"{block_path}/items", "items must be a list or placeholder string")
#             if not isinstance(alias, str) or not alias:
#                 _add_error(errors, f"{block_path}/as", "as must be a non-empty string")
#             if not isinstance(template, list):
#                 _add_error(errors, f"{block_path}/template", "template must be an array")
#             if isinstance(alias, str) and isinstance(template, list):
#                 alias_stack.append(alias)
#                 _validate_blocks(
#                     blocks=template,
#                     path=f"{block_path}/template",
#                     context=context,
#                     strict=strict,
#                     allowed_block_types=allowed_block_types,
#                     errors=errors,
#                     warnings=warnings,
#                     stats=stats,
#                     max_blocks=max_blocks,
#                     max_depth=max_depth,
#                     alias_stack=alias_stack,
#                 )
#                 alias_stack.pop()


# def _check_heading(block: Dict[str, Any], block_path: str, warnings: List[Dict[str, str]]) -> None:
#     text = block.get("text")
#     if isinstance(text, str) and text.strip().upper() not in _ATS_HEADINGS:
#         _add_warning(warnings, f"{block_path}/text", "non-standard ATS heading")


# def _check_string_fields(
#     block: Dict[str, Any],
#     block_path: str,
#     context: Dict[str, Any],
#     strict: bool,
#     errors: List[Dict[str, str]],
#     warnings: List[Dict[str, str]],
#     stats: _Stats,
#     alias_stack: List[str],
# ) -> None:
#     for key, value in block.items():
#         if isinstance(value, str):
#             _check_placeholders(
#                 value,
#                 f"{block_path}/{key}",
#                 context,
#                 strict,
#                 errors,
#                 warnings,
#                 stats,
#                 alias_stack,
#                 field_name=key,
#             )
#             _check_long_paragraph(value, f"{block_path}/{key}", warnings)
#             _check_url_path_traversal(value, f"{block_path}/{key}", key, warnings)


# def _check_placeholders(
#     text: str,
#     path: str,
#     context: Dict[str, Any],
#     strict: bool,
#     errors: List[Dict[str, str]],
#     warnings: List[Dict[str, str]],
#     stats: _Stats,
#     alias_stack: List[str],
#     field_name: Optional[str] = None,
# ) -> None:
#     for raw_expr in _PLACEHOLDER_RE.findall(text):
#         expr = raw_expr.strip()
#         stats.placeholder_count += 1
#         if not expr:
#             stats.unresolved_count += 1
#             _add_error(errors, path, "empty placeholder expression")
#             continue
#         try:
#             tokens = _tokenize_path(expr)
#         except ValueError as exc:
#             stats.unresolved_count += 1
#             _add_error(errors, path, f"invalid placeholder: {exc}")
#             continue
#         if not tokens:
#             stats.unresolved_count += 1
#             _add_error(errors, path, "invalid placeholder")
#             continue
#         if tokens[0] in alias_stack:
#             continue
#         resolved = _resolve_path(tokens, context)
#         if resolved is None:
#             stats.unresolved_count += 1
#             if strict:
#                 _add_error(errors, path, f"unresolved placeholder: {expr}")
#             else:
#                 _add_warning(warnings, path, f"unresolved placeholder: {expr}")

#     if field_name == "items" and isinstance(text, str) and text and text.startswith("{"):
#         if text.strip().startswith("{") and text.strip().endswith("}"):
#             _add_warning(warnings, path, "items should reference placeholders as {{path}}")


# def _resolve_path(tokens: List[Any], context: Dict[str, Any]) -> Optional[Any]:
#     current: Any = context
#     for token in tokens:
#         if isinstance(token, int):
#             if isinstance(current, list) and 0 <= token < len(current):
#                 current = current[token]
#                 continue
#             return None
#         if isinstance(current, dict) and token in current:
#             current = current[token]
#             continue
#         return None
#     return current


# def _tokenize_path(expr: str) -> List[Any]:
#     parts: List[Any] = []
#     idx = 0
#     while idx < len(expr):
#         char = expr[idx]
#         if char == "[":
#             end = expr.find("]", idx)
#             if end == -1:
#                 raise ValueError("missing closing bracket")
#             index_str = expr[idx + 1 : end]
#             if not index_str.isdigit():
#                 raise ValueError("index must be a number")
#             parts.append(int(index_str))
#             idx = end + 1
#         elif char == ".":
#             idx += 1
#         else:
#             start = idx
#             while idx < len(expr) and expr[idx] not in ".[":
#                 idx += 1
#             token = expr[start:idx]
#             if not token:
#                 raise ValueError("empty path segment")
#             parts.append(token)
#     return parts


# def _check_depth(
#     value: Any, depth: int, max_depth: int, errors: List[Dict[str, str]], path: str = ""
# ) -> None:
#     if depth > max_depth:
#         _add_error(errors, path or "/", f"max depth exceeded ({max_depth})")
#         return
#     if isinstance(value, dict):
#         for key, item in value.items():
#             _check_depth(item, depth + 1, max_depth, errors, f"{path}/{key}")
#     elif isinstance(value, list):
#         for index, item in enumerate(value):
#             _check_depth(item, depth + 1, max_depth, errors, f"{path}/{index}")


# def _check_long_paragraph(text: str, path: str, warnings: List[Dict[str, str]]) -> None:
#     if len(text) > 500:
#         _add_warning(warnings, path, "paragraph text exceeds 500 characters")


# def _check_url_path_traversal(
#     value: str, path: str, key: str, warnings: List[Dict[str, str]]
# ) -> None:
#     if "url" in key.lower() and ("../" in value or "..\\" in value or value.strip().startswith("..")):
#         _add_warning(warnings, path, "potential path traversal detected in URL field")


# def _require_string(
#     block: Dict[str, Any], field: str, block_path: str, errors: List[Dict[str, str]]
# ) -> None:
#     value = block.get(field)
#     if not isinstance(value, str):
#         _add_error(errors, f"{block_path}/{field}", f"{field} must be a string")


# def _is_exact_placeholder(value: str) -> bool:
#     matches = _PLACEHOLDER_RE.findall(value)
#     return len(matches) == 1 and value.strip() == f"{{{{{matches[0]}}}}}"


# def _add_error(errors: List[Dict[str, str]], path: str, message: str) -> None:
#     errors.append({"path": path, "message": message})


# def _add_warning(warnings: List[Dict[str, str]], path: str, message: str) -> None:
#     warnings.append({"path": path, "message": message})


# def _tool_error(message: str) -> None:
#     from libs.core.tool_registry import ToolExecutionError

#     raise ToolExecutionError(message)
