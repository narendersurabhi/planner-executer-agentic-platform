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
}

MEMORY_INPUT_KEYS = {
    "llm_improve_tailored_resume_text": ["tailored_resume"],
    "llm_iterative_improve_tailored_resume_text": ["tailored_resume"],
    "llm_generate_resume_doc_spec_from_text": ["tailored_resume"],
}


def extract_memory_value(entries: Sequence[Mapping[str, Any]], key: str) -> Any:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def apply_memory_defaults(tool_name: str, payload: Mapping[str, Any]) -> dict:
    if tool_name not in MEMORY_INPUT_KEYS:
        return dict(payload)
    output = dict(payload)
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
        value = extract_memory_value(task_outputs, key)
        if value is not None:
            output[key] = value
    return output


def select_memory_payload(tool_name: str, output_or_error: Mapping[str, Any]) -> dict:
    if not isinstance(output_or_error, Mapping):
        return {}
    keys = MEMORY_OUTPUT_KEYS.get(tool_name)
    if not keys:
        return {}
    return {key: output_or_error.get(key) for key in keys if output_or_error.get(key) is not None}
