from __future__ import annotations

import json
from typing import Any, Dict

from jsonschema import Draft202012Validator


def resolve_tool_inputs(
    tool_requests: list[str],
    instruction: str,
    context: dict[str, Any] | None,
    task_payload: dict[str, Any],
    tool_inputs: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for tool_name in tool_requests:
        resolved_payload = resolve_tool_payload(
            tool_name, instruction, context or {}, task_payload, tool_inputs
        )
        if isinstance(resolved_payload, dict):
            resolved[tool_name] = resolved_payload
    return resolved


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
) -> dict[str, Any]:
    has_tool_inputs = False
    payload: dict[str, Any] = {}
    if isinstance(tool_inputs, dict) and tool_name in tool_inputs:
        tool_payload = tool_inputs.get(tool_name)
        if isinstance(tool_payload, dict):
            payload = dict(tool_payload)
            has_tool_inputs = True
    payload = _merge_payload_from_task(payload, task_payload, instruction)
    payload = _fill_payload_from_context(payload, context)
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
            "topic",
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
            "candidate_resume",
            "tailored_text",
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
    if "tailored_resume" not in filled:
        resume = _extract_resume_content_from_context(context)
        if resume is not None:
            filled["tailored_resume"] = resume
    if "resume_content" not in filled:
        resume = _extract_resume_content_from_context(context)
        if resume is not None:
            filled["resume_content"] = resume
    if "resume_doc_spec" not in filled:
        resume_spec = _extract_resume_doc_spec_from_context(context)
        if isinstance(resume_spec, dict):
            filled["resume_doc_spec"] = resume_spec
    if "coverletter_doc_spec" not in filled:
        coverletter_spec = _extract_coverletter_doc_spec_from_context(context)
        if isinstance(coverletter_spec, dict):
            filled["coverletter_doc_spec"] = coverletter_spec
    if "tailored_text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["tailored_text"] = text
    if "content" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["content"] = text
    if "text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["text"] = text
    return filled


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


def _extract_resume_doc_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            resume_spec = _extract_resume_doc_spec_from_outputs(output)
            if isinstance(resume_spec, dict):
                return resume_spec
    return None


def _extract_coverletter_doc_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            coverletter_spec = _extract_coverletter_doc_spec_from_outputs(output)
            if isinstance(coverletter_spec, dict):
                return coverletter_spec
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
    return None


def _extract_resume_content_from_context(context: dict) -> dict | str | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            value = _extract_resume_content_from_outputs(output)
            if value is not None:
                return value
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
    resume_spec_output = outputs.get("llm_generate_resume_doc_spec")
    if isinstance(resume_spec_output, dict):
        resume_doc_spec = resume_spec_output.get("resume_doc_spec")
        if isinstance(resume_doc_spec, dict):
            return resume_doc_spec
    resume_spec_text_output = outputs.get("llm_generate_resume_doc_spec_from_text")
    if isinstance(resume_spec_text_output, dict):
        resume_doc_spec = resume_spec_text_output.get("resume_doc_spec")
        if isinstance(resume_doc_spec, dict):
            return resume_doc_spec
    coverletter_spec_output = outputs.get("llm_generate_coverletter_doc_spec_from_text")
    if isinstance(coverletter_spec_output, dict):
        coverletter_doc_spec = coverletter_spec_output.get("coverletter_doc_spec")
        if isinstance(coverletter_doc_spec, dict):
            return coverletter_doc_spec
    converted = outputs.get("resume_doc_spec_to_document_spec")
    if isinstance(converted, dict):
        document_spec = converted.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    converted_coverletter = outputs.get("coverletter_doc_spec_to_document_spec")
    if isinstance(converted_coverletter, dict):
        document_spec = converted_coverletter.get("document_spec")
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
        "resume_doc_spec_to_document_spec",
        "coverletter_doc_spec_to_document_spec",
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
    return None


def _extract_resume_doc_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    for key in (
        "llm_generate_resume_doc_spec",
        "llm_generate_resume_doc_spec_from_text",
    ):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            resume_doc_spec = candidate.get("resume_doc_spec")
            if isinstance(resume_doc_spec, dict):
                return resume_doc_spec
    direct = outputs.get("resume_doc_spec")
    if isinstance(direct, dict):
        return direct
    return None


def _extract_coverletter_doc_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    candidate = outputs.get("llm_generate_coverletter_doc_spec_from_text")
    if isinstance(candidate, dict):
        coverletter_doc_spec = candidate.get("coverletter_doc_spec")
        if isinstance(coverletter_doc_spec, dict):
            return coverletter_doc_spec
    direct = outputs.get("coverletter_doc_spec")
    if isinstance(direct, dict):
        return direct
    return None


def _extract_resume_content_from_outputs(outputs: dict) -> dict | str | None:
    if not isinstance(outputs, dict):
        return None
    for key in ("tailored_resume", "resume_content"):
        value = outputs.get(key)
        if isinstance(value, (dict, str)):
            return value
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            json_text = _extract_json(text)
            if json_text:
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError:
                    return None
                if isinstance(data, dict):
                    for key in ("tailored_resume", "resume_content"):
                        value = data.get(key)
                        if isinstance(value, (dict, str)):
                            return value
                    return data
    json_transform = outputs.get("json_transform")
    if isinstance(json_transform, dict):
        result = json_transform.get("result")
        if isinstance(result, dict):
            for key in ("tailored_resume", "resume_content"):
                value = result.get(key)
                if isinstance(value, (dict, str)):
                    return value
            return result
        if isinstance(result, str):
            return result
    return None


def _extract_text_from_outputs(outputs: dict) -> str | None:
    if not isinstance(outputs, dict):
        return None
    direct = outputs.get("tailored_text")
    if isinstance(direct, str) and direct:
        return direct
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
            value = entry.get("tailored_text")
            if isinstance(value, str) and value:
                return value
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
