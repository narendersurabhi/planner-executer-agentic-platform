from __future__ import annotations

from typing import Any, Mapping, Sequence


MEMORY_OUTPUT_KEYS = {
    "llm_generate_document_spec": ["document_spec"],
    "llm_generate_document_spec_from_markdown": ["document_spec"],
    "llm_improve_document_spec": ["document_spec"],
    "llm_iterative_improve_document_spec": [
        "document_spec",
        "validation_report",
        "iterations",
        "reached_threshold",
    ],
    "llm_iterative_improve_runbook_spec": [
        "document_spec",
        "validation_report",
        "iterations",
        "reached_threshold",
    ],
    "llm_iterative_improve_openapi_spec": [
        "openapi_spec",
        "validation_report",
        "iterations",
        "reached_threshold",
    ],
    "derive_output_filename": ["path", "document_type"],
}

MEMORY_INPUT_KEYS = {
    "document_spec_validate": ["document_spec"],
    "docx_generate_from_spec": ["document_spec", "path"],
    "llm_improve_document_spec": ["document_spec"],
    "llm_iterative_improve_document_spec": ["document_spec"],
    "llm_iterative_improve_runbook_spec": ["document_spec"],
    "llm_iterative_improve_openapi_spec": ["openapi_spec"],
}

MEMORY_ONLY_INPUTS = {tool_name: list(keys) for tool_name, keys in MEMORY_INPUT_KEYS.items()}
MEMORY_ONLY_INPUTS.pop("document_spec_validate", None)
MEMORY_ONLY_INPUTS.pop("docx_generate_from_spec", None)
# Iterative generators support either seed spec (from memory/dependency) OR job input.
# Do not enforce memory-only for these tools.
MEMORY_ONLY_INPUTS.pop("llm_iterative_improve_document_spec", None)
MEMORY_ONLY_INPUTS.pop("llm_iterative_improve_runbook_spec", None)
MEMORY_ONLY_INPUTS.pop("llm_iterative_improve_openapi_spec", None)

MEMORY_PREFERRED_KEYS = {
    "document_spec": "document_spec:latest",
    "openapi_spec": "openapi_spec:latest",
    "path": "docx_path:latest",
}


def stable_memory_keys(tool_name: str, payload: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for field, alias in MEMORY_PREFERRED_KEYS.items():
        if field in payload:
            keys.append(alias)
    if tool_name in {"derive_output_filename", "derive_output_path"} and "path" in payload:
        keys.append("docx_path:document:latest")
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


def _extract_docx_path_for_document(
    entries: Sequence[Mapping[str, Any]], document_spec: Any
) -> Any:
    preferred_alias = "docx_path:document:latest"
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
