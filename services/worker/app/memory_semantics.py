from __future__ import annotations

from typing import Any, Mapping, Sequence


MEMORY_OUTPUT_KEYS = {
    "llm_tailor_resume_text": ["tailored_resume"],
    "llm_improve_tailored_resume_text": [
        "tailored_resume",
        "alignment_score",
        "alignment_summary",
    ],
    "llm_iterative_improve_tailored_resume_text": [
        "tailored_resume",
        "alignment_score",
        "alignment_summary",
        "iterations",
        "reached_threshold",
    ],
    "llm_generate_resume_doc_spec_from_text": ["resume_doc_spec", "validation"],
    "llm_generate_coverletter_doc_spec_from_text": ["coverletter_doc_spec"],
    "llm_generate_resume_doc_spec": ["resume_doc_spec"],
    "llm_generate_cover_letter_from_resume": ["cover_letter"],
    "resume_doc_spec_to_document_spec": ["document_spec"],
    "coverletter_doc_spec_to_document_spec": ["document_spec"],
    "llm_generate_document_spec": ["document_spec"],
    "llm_improve_document_spec": ["document_spec"],
    "derive_output_filename": ["path", "document_type"],
}

MEMORY_INPUT_KEYS = {
    "llm_improve_tailored_resume_text": ["tailored_resume"],
    "llm_iterative_improve_tailored_resume_text": ["tailored_resume"],
    "llm_generate_resume_doc_spec_from_text": ["tailored_resume", "tailored_text"],
    "llm_generate_coverletter_doc_spec_from_text": ["tailored_resume", "tailored_text"],
    "llm_generate_resume_doc_spec": ["tailored_resume"],
    "llm_generate_cover_letter_from_resume": ["tailored_resume", "tailored_text"],
    "resume_doc_spec_validate": ["resume_doc_spec"],
    "resume_doc_spec_to_document_spec": ["resume_doc_spec"],
    "coverletter_doc_spec_to_document_spec": ["coverletter_doc_spec"],
    "document_spec_validate": ["document_spec"],
    "docx_generate_from_spec": ["document_spec", "path"],
    "cover_letter_generate_ats_docx": ["cover_letter", "path"],
    "llm_improve_document_spec": ["document_spec"],
}

MEMORY_ONLY_INPUTS = {
    tool_name: list(keys) for tool_name, keys in MEMORY_INPUT_KEYS.items()
}
MEMORY_ONLY_INPUTS.pop("llm_generate_resume_doc_spec_from_text", None)
MEMORY_ONLY_INPUTS.pop("llm_generate_coverletter_doc_spec_from_text", None)
MEMORY_ONLY_INPUTS.pop("llm_generate_cover_letter_from_resume", None)
MEMORY_ONLY_INPUTS.pop("document_spec_validate", None)
MEMORY_ONLY_INPUTS.pop("docx_generate_from_spec", None)
MEMORY_ONLY_INPUTS.pop("cover_letter_generate_ats_docx", None)

MEMORY_PREFERRED_KEYS = {
    "tailored_resume": "tailored_resume:latest",
    "tailored_text": "tailored_text:latest",
    "alignment_score": "alignment_score:latest",
    "alignment_summary": "alignment_summary:latest",
    "resume_doc_spec": "resume_doc_spec:latest",
    "coverletter_doc_spec": "coverletter_doc_spec:latest",
    "validation": "resume_doc_spec_validation:latest",
    "document_spec": "document_spec:latest",
    "cover_letter": "cover_letter:latest",
    "path": "docx_path:latest",
}


def stable_memory_keys(tool_name: str, payload: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for field, alias in MEMORY_PREFERRED_KEYS.items():
        if field in payload:
            keys.append(alias)
    if tool_name == "derive_output_filename" and "path" in payload:
        document_type = _normalize_document_type(payload.get("document_type"))
        if document_type == "cover_letter":
            keys.append("docx_path:cover_letter:latest")
        else:
            keys.append("docx_path:resume:latest")
    return keys


def extract_memory_value(entries: Sequence[Mapping[str, Any]], key: str) -> Any:
    preferred_key = MEMORY_PREFERRED_KEYS.get(key)
    if preferred_key:
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("_memory_key") != preferred_key:
                continue
            value = _extract_entry_value(entry, key)
            if value is not None:
                return value
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        value = _extract_entry_value(entry, key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _extract_entry_value(entry: Mapping[str, Any], key: str) -> Any:
    if key in entry:
        return entry.get(key)
    payload = entry.get("payload")
    if isinstance(payload, Mapping):
        return payload.get(key)
    return None


def apply_memory_defaults(tool_name: str, payload: Mapping[str, Any]) -> dict:
    output = dict(payload)
    if tool_name in MEMORY_ONLY_INPUTS:
        for key in MEMORY_ONLY_INPUTS[tool_name]:
            output.pop(key, None)
    if tool_name not in MEMORY_INPUT_KEYS:
        return dict(payload)
    memory = output.get("memory")
    if not isinstance(memory, Mapping):
        return output
    task_outputs = memory.get("task_outputs")
    if not isinstance(task_outputs, Sequence) or not task_outputs:
        return output
    for key in MEMORY_INPUT_KEYS[tool_name]:
        existing = output.get(key)
        if isinstance(existing, str) and existing.strip():
            continue
        if key in output and not isinstance(existing, str) and existing is not None:
            continue
        if key == "path" and tool_name == "docx_generate_from_spec":
            value = _extract_docx_path_for_document(task_outputs, output.get("document_spec"))
        else:
            value = extract_memory_value(task_outputs, key)
        if value is not None:
            output[key] = value
    return output


def _normalize_document_type(value: Any) -> str:
    if not isinstance(value, str):
        return "resume"
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"cover_letter", "coverletter"}:
        return "cover_letter"
    return "resume"


def _is_cover_letter_document_spec(document_spec: Any) -> bool:
    if not isinstance(document_spec, Mapping):
        return False
    doc_type = document_spec.get("doc_type")
    if not isinstance(doc_type, str):
        return False
    normalized = doc_type.strip().lower().replace("-", "_")
    return "cover" in normalized


def _extract_docx_path_for_document(entries: Sequence[Mapping[str, Any]], document_spec: Any) -> Any:
    preferred_alias = (
        "docx_path:cover_letter:latest"
        if _is_cover_letter_document_spec(document_spec)
        else "docx_path:resume:latest"
    )
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("_memory_key") != preferred_alias:
            continue
        value = _extract_entry_value(entry, "path")
        if isinstance(value, str) and value.strip():
            return value
        if value is not None:
            return value
    return extract_memory_value(entries, "path")


def missing_memory_only_inputs(tool_name: str, payload: Mapping[str, Any]) -> list[str]:
    required = MEMORY_ONLY_INPUTS.get(tool_name)
    if not required:
        return []
    missing: list[str] = []
    for key in required:
        value = payload.get(key)
        if key == "document_spec":
            if not isinstance(value, Mapping) or not value:
                missing.append(key)
            continue
        if value is None:
            missing.append(key)
    return missing


def select_memory_payload(tool_name: str, output_or_error: Mapping[str, Any]) -> dict:
    if not isinstance(output_or_error, Mapping):
        return {}
    keys = MEMORY_OUTPUT_KEYS.get(tool_name)
    if not keys:
        return {}
    return {key: output_or_error.get(key) for key in keys if output_or_error.get(key) is not None}
