from __future__ import annotations

from typing import Any, Mapping

_DOCUMENT_GENERATION_TOOL_NAMES = {
    "llm_iterative_improve_document_spec",
    "llm_iterative_improve_runbook_spec",
}

_DOCUMENT_JOB_TOP_LEVEL_KEYS = (
    "instruction",
    "topic",
    "audience",
    "tone",
    "today",
    "output_dir",
    "document_type",
)

_DOCUMENT_JOB_CONTEXT_KEYS = (
    "markdown_text",
    "instruction",
    "topic",
    "audience",
    "tone",
    "today",
    "output_dir",
    "document_type",
    "target_role_name",
    "role_name",
    "company_name",
    "company",
    "candidate_name",
    "first_name",
    "last_name",
    "job_description",
)

_DOCUMENT_GENERATION_INPUT_KEYS = (
    "instruction",
    "topic",
    "audience",
    "tone",
    "today",
    "output_dir",
    "document_type",
    "target_role_name",
    "role_name",
    "company_name",
    "company",
    "candidate_name",
    "first_name",
    "last_name",
    "job_description",
)

_DOCUMENT_MARKDOWN_INPUT_KEYS = (
    "markdown_text",
    "topic",
    "audience",
    "tone",
    "today",
    "output_dir",
    "document_type",
    "target_role_name",
    "role_name",
    "company_name",
    "company",
    "candidate_name",
    "first_name",
    "last_name",
    "job_description",
)


def project_job_payload_for_tool(
    tool_name: str,
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_payload, Mapping):
        return {}
    if str(tool_name).strip() in _DOCUMENT_GENERATION_TOOL_NAMES:
        return compact_document_job_payload(job_payload, default_goal=default_goal)
    return dict(job_payload)


def project_explicit_inputs_for_tool(
    tool_name: str,
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_payload, Mapping):
        return {}
    if str(tool_name).strip() == "llm_generate_document_spec":
        return project_document_generation_inputs(job_payload, default_goal=default_goal)
    if str(tool_name).strip() == "llm_generate_document_spec_from_markdown":
        return project_markdown_document_generation_inputs(job_payload, default_goal=default_goal)
    return {}


def project_document_generation_inputs(
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_payload, Mapping):
        return {}
    compact = compact_document_job_payload(job_payload, default_goal=default_goal)
    projected: dict[str, Any] = {}
    context_json = compact.get("context_json")

    instruction = _first_non_empty_str(
        job_payload.get("instruction"),
        compact.get("instruction"),
        context_json.get("instruction") if isinstance(context_json, Mapping) else None,
        job_payload.get("goal"),
        compact.get("goal"),
    )
    if instruction:
        projected["instruction"] = instruction

    for key in _DOCUMENT_GENERATION_INPUT_KEYS:
        if key == "instruction":
            continue
        _copy_compact_value(projected, key, job_payload.get(key))
        _copy_compact_value(projected, key, compact.get(key))

    if isinstance(context_json, Mapping):
        for key in _DOCUMENT_GENERATION_INPUT_KEYS:
            if key in projected:
                continue
            _copy_compact_value(projected, key, context_json.get(key))
    return projected


def project_markdown_document_generation_inputs(
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_payload, Mapping):
        return {}
    compact = compact_document_job_payload(job_payload, default_goal=default_goal)
    projected: dict[str, Any] = {}
    context_json = compact.get("context_json")

    for key in _DOCUMENT_MARKDOWN_INPUT_KEYS:
        _copy_compact_value(projected, key, job_payload.get(key))
        _copy_compact_value(projected, key, compact.get(key))

    if isinstance(context_json, Mapping):
        for key in _DOCUMENT_MARKDOWN_INPUT_KEYS:
            if key in projected:
                continue
            _copy_compact_value(projected, key, context_json.get(key))
    return projected


def compact_document_job_payload(
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_payload, Mapping):
        return {}

    compact: dict[str, Any] = {}
    goal = default_goal or job_payload.get("goal")
    _copy_compact_value(compact, "goal", goal)

    for key in _DOCUMENT_JOB_TOP_LEVEL_KEYS:
        _copy_compact_value(compact, key, job_payload.get(key))

    context_json = job_payload.get("context_json")
    if isinstance(context_json, Mapping):
        compact_context: dict[str, Any] = {}
        for key in _DOCUMENT_JOB_CONTEXT_KEYS:
            _copy_compact_value(compact_context, key, context_json.get(key))
        if compact_context:
            compact["context_json"] = compact_context

    return compact or dict(job_payload)


def _copy_compact_value(target: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, str):
        if value.strip():
            target[key] = value
        return
    if isinstance(value, (int, float, bool)):
        target[key] = value


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
