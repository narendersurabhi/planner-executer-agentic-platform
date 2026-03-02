from __future__ import annotations

import copy
import json
from typing import Any, Dict

from jsonschema import Draft202012Validator


class ToolInputReferenceError(ValueError):
    """Raised when a tool input reference cannot be resolved."""


def resolve_tool_inputs(
    tool_requests: list[str],
    instruction: str,
    context: dict[str, Any] | None,
    task_payload: dict[str, Any],
    tool_inputs: dict[str, Any],
) -> dict[str, Any]:
    resolved, _ = resolve_tool_inputs_with_errors(
        tool_requests, instruction, context, task_payload, tool_inputs
    )
    return resolved


def resolve_tool_inputs_with_errors(
    tool_requests: list[str],
    instruction: str,
    context: dict[str, Any] | None,
    task_payload: dict[str, Any],
    tool_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    resolved: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for tool_name in tool_requests:
        try:
            resolved_payload = resolve_tool_payload(
                tool_name,
                instruction,
                context or {},
                task_payload,
                tool_inputs,
                strict_references=True,
            )
        except ToolInputReferenceError as exc:
            errors[tool_name] = f"input reference resolution failed: {exc}"
            continue
        if isinstance(resolved_payload, dict):
            resolved[tool_name] = resolved_payload
    return resolved, errors


def validate_tool_inputs(
    tool_inputs: dict[str, Any],
    tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not isinstance(tool_inputs, dict):
        return errors
    for tool_name, payload in tool_inputs.items():
        schema = tool_schemas.get(tool_name)
        if not schema:
            continue
        if not isinstance(payload, dict):
            errors[tool_name] = "input schema validation failed: <root>: payload must be an object"
            continue
        validation_error = _validate_schema(schema, payload, "input")
        if validation_error:
            errors[tool_name] = validation_error
    return errors


def resolve_tool_payload(
    tool_name: str,
    instruction: str,
    context: dict[str, Any],
    task_payload: dict[str, Any],
    tool_inputs: dict[str, Any],
    *,
    strict_references: bool = False,
) -> dict[str, Any]:
    has_tool_inputs = False
    payload: dict[str, Any] = {}
    if isinstance(tool_inputs, dict) and tool_name in tool_inputs:
        tool_payload = tool_inputs.get(tool_name)
        if isinstance(tool_payload, dict):
            payload = _resolve_payload_references(
                dict(tool_payload),
                context,
                strict=strict_references,
            )
            has_tool_inputs = True
    payload = _merge_payload_from_task(payload, task_payload, instruction)
    payload = _fill_payload_from_context(payload, context)
    payload = _promote_document_job_fields(payload)
    if tool_name == "llm_generate":
        base_text = payload.get("text") or payload.get("prompt") or instruction
        if context:
            try:
                context_blob = json.dumps(context, indent=2, ensure_ascii=True)
            except (TypeError, ValueError):
                context_blob = str(context)
            base_text = f"{base_text}\n\nContext (JSON):\n{context_blob}"
        return {"text": base_text}
    if has_tool_inputs:
        return payload
    if payload:
        return payload
    if not context:
        return {"text": instruction}
    try:
        context_blob = json.dumps(context, indent=2, ensure_ascii=True)
    except (TypeError, ValueError):
        context_blob = str(context)
    return {"text": f"{instruction}\n\nContext (JSON):\n{context_blob}"}


def _merge_payload_from_task(payload: dict, task_payload: dict, instruction: str) -> dict:
    merged = dict(payload)
    if not isinstance(task_payload, dict):
        task_payload = {}
    for key in (
        "data",
        "schema_ref",
        "template_id",
        "template_path",
        "output_path",
        "path",
        "content",
        "render_context",
    ):
        if key not in merged and key in task_payload:
            merged[key] = task_payload.get(key)
    if "schema_ref" not in merged:
        value = _extract_schema_ref(instruction)
        if value:
            merged["schema_ref"] = value
    if "template_id" not in merged:
        value = _extract_template_id(instruction)
        if value:
            merged["template_id"] = value
    if "template_path" not in merged:
        value = _extract_template_path(instruction)
        if value:
            merged["template_path"] = value
    if "output_path" not in merged:
        value = _extract_output_path(instruction)
        if value:
            merged["output_path"] = value
    if "path" not in merged and isinstance(task_payload.get("path"), str):
        merged["path"] = task_payload.get("path")
    if "content" not in merged and isinstance(task_payload.get("content"), str):
        merged["content"] = task_payload.get("content")
    return merged


def _fill_payload_from_context(payload: dict, context: dict) -> dict:
    filled = dict(payload)
    job_context = context.get("job_context")
    if isinstance(job_context, dict):
        # Promote common planning fields from job context so tool input validation
        # can succeed before worker-side memory hydration.
        for key in (
            "instruction",
            "topic",
            "audience",
            "tone",
            "today",
            "date",
            "target_pages",
            "page_count",
            "target_role_name",
            "role_name",
            "company_name",
            "company",
            "candidate_name",
            "first_name",
            "last_name",
            "job_description",
            "output_dir",
            "document_type",
        ):
            if key in filled:
                continue
            value = job_context.get(key)
            if isinstance(value, str) and value.strip():
                filled[key] = value
    if "document_spec" not in filled:
        doc = _extract_document_spec_from_context(context)
        if isinstance(doc, dict):
            filled["document_spec"] = doc
    if "data" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["data"] = doc
    if "errors" not in filled:
        errors = _extract_validation_errors_from_context(context)
        if isinstance(errors, list):
            filled["errors"] = errors
    if "validation_report" not in filled:
        report = _extract_validation_report_from_context(context)
        if isinstance(report, dict):
            filled["validation_report"] = report
    if "original_spec" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["original_spec"] = doc
    if "content" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["content"] = text
    if "text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["text"] = text
    return filled


def _promote_document_job_fields(payload: dict[str, Any]) -> dict[str, Any]:
    promoted = dict(payload)
    job = promoted.get("job")
    if not isinstance(job, dict):
        return promoted
    for key in (
        "instruction",
        "topic",
        "audience",
        "tone",
        "today",
        "output_dir",
    ):
        if key in promoted:
            continue
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            promoted[key] = value
    return promoted


def normalize_reference_payload_for_validation(
    payload: dict[str, Any],
    *,
    dependency_defaults: dict[str, Any] | None = None,
    unknown_default: Any = "__dependency__",
) -> dict[str, Any]:
    defaults = dependency_defaults or {}

    def _normalize(value: Any, key_hint: str | None = None) -> Any:
        if isinstance(value, dict):
            if "$from" in value:
                if "$default" in value:
                    return copy.deepcopy(value.get("$default"))
                if key_hint and key_hint in defaults:
                    return copy.deepcopy(defaults[key_hint])
                return copy.deepcopy(unknown_default)
            return {k: _normalize(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize(item, key_hint) for item in value]
        return value

    normalized = _normalize(dict(payload))
    if isinstance(normalized, dict):
        normalized = _promote_document_job_fields(normalized)
    return normalized


def _resolve_payload_references(value: Any, context: dict[str, Any], *, strict: bool) -> Any:
    if isinstance(value, dict):
        if "$from" in value:
            try:
                return _resolve_reference_value(value, context)
            except ToolInputReferenceError:
                if strict:
                    raise
                return dict(value)
        return {k: _resolve_payload_references(v, context, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_payload_references(item, context, strict=strict) for item in value]
    return value


def _resolve_reference_value(reference: dict[str, Any], context: dict[str, Any]) -> Any:
    extras = set(reference.keys()) - {"$from", "$default"}
    if extras:
        raise ToolInputReferenceError(
            f"invalid reference object keys: {', '.join(sorted(extras))}"
        )
    path_spec = reference.get("$from")
    if path_spec is None:
        raise ToolInputReferenceError("reference is missing $from")
    try:
        value = _resolve_reference_path(path_spec, context)
    except ToolInputReferenceError:
        if "$default" in reference:
            return copy.deepcopy(reference.get("$default"))
        raise
    return copy.deepcopy(value)


def _resolve_reference_path(path_spec: Any, context: dict[str, Any]) -> Any:
    candidate_paths = _candidate_reference_paths(path_spec, context)
    errors: list[str] = []
    for segments in candidate_paths:
        try:
            return _walk_path(context, segments)
        except ToolInputReferenceError as exc:
            errors.append(str(exc))
    path_repr = path_spec if isinstance(path_spec, str) else json.dumps(path_spec, ensure_ascii=True)
    detail = "; ".join(errors) if errors else "path not found"
    raise ToolInputReferenceError(f"path '{path_repr}' could not be resolved ({detail})")


def _candidate_reference_paths(path_spec: Any, context: dict[str, Any]) -> list[list[Any]]:
    if isinstance(path_spec, list):
        return [list(path_spec)]
    if isinstance(path_spec, tuple):
        return [list(path_spec)]
    if not isinstance(path_spec, str):
        raise ToolInputReferenceError("$from must be a string or array path")
    raw = path_spec.strip()
    if not raw:
        raise ToolInputReferenceError("$from path is empty")
    if raw.startswith("$."):
        raw = raw[2:]
    if raw.startswith("/"):
        parts = [part.replace("~1", "/").replace("~0", "~") for part in raw.split("/")[1:]]
        return [parts]
    segments = [segment for segment in raw.split(".") if segment]
    if not segments:
        raise ToolInputReferenceError("$from path is empty")
    candidates = [segments]
    roots = {"dependencies_by_name", "dependencies", "job_context"}
    if segments[0] not in roots:
        if isinstance(context.get("dependencies_by_name"), dict):
            candidates.append(["dependencies_by_name", *segments])
        if isinstance(context.get("dependencies"), dict):
            candidates.append(["dependencies", *segments])
    return candidates


def _walk_path(root: Any, segments: list[Any]) -> Any:
    current = root
    for segment in segments:
        if isinstance(current, dict):
            if segment not in current:
                # Backward compatibility:
                # Some plans reference
                # "...document_spec_validate.validation_report"
                # even though document_spec_validate already returns
                # the report object directly.
                if segment == "validation_report" and _is_validation_report_dict(current):
                    continue
                # Some plans/reference templates use output_path while
                # path-derivation tools emit {"path": "..."}.
                if segment == "output_path" and "path" in current:
                    segment = "path"
                elif segment == "path" and "output_path" in current:
                    segment = "output_path"
                # Some plans point directly to task output, e.g.
                # "dependencies_by_name.TaskName.path", while runtime context stores
                # "dependencies_by_name.TaskName.<tool_name>.path".
                elif segment in {"path", "output_path"}:
                    nested = [
                        value
                        for value in current.values()
                        if isinstance(value, dict)
                        and (segment in value or (segment == "path" and "output_path" in value))
                    ]
                    if len(nested) == 1:
                        current = nested[0]
                        if segment == "path" and "path" not in current and "output_path" in current:
                            segment = "output_path"
                    else:
                        raise ToolInputReferenceError(f"missing key '{segment}'")
                else:
                    raise ToolInputReferenceError(f"missing key '{segment}'")
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except (TypeError, ValueError):
                raise ToolInputReferenceError(
                    f"list index '{segment}' is not a valid integer"
                ) from None
            if index < 0 or index >= len(current):
                raise ToolInputReferenceError(f"list index '{index}' out of range")
            current = current[index]
            continue
        raise ToolInputReferenceError(
            f"cannot traverse segment '{segment}' on non-container value"
        )
    return current


def _is_validation_report_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("valid"), bool)
        and isinstance(value.get("errors"), list)
        and isinstance(value.get("warnings"), list)
    )


def _extract_json_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            json_candidate = _extract_json_from_outputs(output)
            if isinstance(json_candidate, dict):
                return json_candidate
    return None


def _extract_document_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            doc = _extract_document_spec_from_outputs(output)
            if isinstance(doc, dict):
                return doc
    return None


def _extract_validation_errors_from_context(context: dict) -> list[dict] | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                errors = entry.get("errors")
                if isinstance(errors, list):
                    return errors
            for candidate in output.values():
                if not isinstance(candidate, dict):
                    continue
                errors = candidate.get("errors")
                if isinstance(errors, list):
                    return errors
    return None


def _extract_validation_report_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                return entry
            for candidate in output.values():
                if not isinstance(candidate, dict):
                    continue
                if "errors" in candidate and "valid" in candidate:
                    return candidate
    return None


def _extract_text_from_context(context: dict) -> str | None:
    if not isinstance(context, dict):
        return None
    for key in ("dependencies_by_name", "dependencies"):
        group = context.get(key)
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            text = _extract_text_from_outputs(output)
            if text:
                return text
    return None


def _extract_json_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    spec_output = outputs.get("llm_generate_document_spec")
    if isinstance(spec_output, dict):
        document_spec = spec_output.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            json_text = _extract_json(text)
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return None
    llm_repair = outputs.get("llm_repair_document_spec")
    if not isinstance(llm_repair, dict):
        llm_repair = outputs.get("llm_repair_json")
    if isinstance(llm_repair, dict):
        document_spec = llm_repair.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_improve = outputs.get("llm_improve_document_spec")
    if isinstance(llm_improve, dict):
        document_spec = llm_improve.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    json_transform = outputs.get("json_transform")
    if isinstance(json_transform, dict):
        result = json_transform.get("result")
        if isinstance(result, dict):
            return result
    payload = outputs.get("result")
    if isinstance(payload, dict):
        return payload
    return None


def _extract_document_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    for key in (
        "llm_generate_document_spec",
        "llm_repair_document_spec",
        "llm_repair_json",
        "llm_improve_document_spec",
    ):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            document_spec = candidate.get("document_spec")
            if isinstance(document_spec, dict):
                return document_spec
    direct = outputs.get("document_spec")
    if isinstance(direct, dict):
        return direct
    for candidate in outputs.values():
        if not isinstance(candidate, dict):
            continue
        document_spec = candidate.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
        result = candidate.get("result")
        if isinstance(result, dict):
            nested_document_spec = result.get("document_spec")
            if isinstance(nested_document_spec, dict):
                return nested_document_spec
    return None


def _extract_text_from_outputs(outputs: dict) -> str | None:
    if not isinstance(outputs, dict):
        return None
    for tool_key in ("llm_generate", "text_summarize", "json_transform"):
        entry = outputs.get(tool_key)
        if not isinstance(entry, dict):
            continue
        if tool_key == "llm_generate":
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
        if tool_key == "text_summarize":
            summary = entry.get("summary")
            if isinstance(summary, str) and summary:
                return summary
        if tool_key == "json_transform":
            result = entry.get("result")
            if isinstance(result, dict):
                text = result.get("text")
                if isinstance(text, str) and text:
                    return text
            elif isinstance(result, str) and result:
                return result
    for entry in outputs.values():
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
    return None


def _extract_instruction_payload(instruction: str) -> dict:
    json_text = _extract_json(instruction)
    if not json_text:
        return {}
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _validate_schema(
    schema: Dict[str, Any] | None, payload: Dict[str, Any], label: str
) -> str | None:
    if not schema:
        return None
    try:
        validator = Draft202012Validator(schema)
    except Exception as exc:  # noqa: BLE001
        return f"Invalid {label} schema: {exc}"
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if errors:
        messages = "; ".join(
            f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors[:5]
        )
        return f"{label} schema validation failed: {messages}"
    return None


def _extract_template_id(instruction: str) -> str | None:
    data = _extract_instruction_payload(instruction)
    value = data.get("template_id")
    if isinstance(value, str) and value:
        return value
    return None


def _extract_template_path(instruction: str) -> str | None:
    data = _extract_instruction_payload(instruction)
    value = data.get("template_path")
    if isinstance(value, str) and value:
        return value
    return None


def _extract_output_path(instruction: str) -> str | None:
    data = _extract_instruction_payload(instruction)
    value = data.get("output_path")
    if isinstance(value, str) and value:
        return value
    return None


def _extract_schema_ref(instruction: str) -> str | None:
    data = _extract_instruction_payload(instruction)
    value = data.get("schema_ref")
    if isinstance(value, str) and value:
        return value
    return None


def _extract_json(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1]
        content = content.lstrip()
        if content.startswith("json"):
            content = content[4:].lstrip()
    first_obj = content.find("{")
    first_arr = content.find("[")
    if first_obj == -1 and first_arr == -1:
        return ""
    if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
        start = first_obj
        end = content.rfind("}")
    else:
        start = first_arr
        end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return ""
    return content[start : end + 1]
