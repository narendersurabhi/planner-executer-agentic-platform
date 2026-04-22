from __future__ import annotations

from typing import Any, Mapping

_DOCUMENT_GENERATION_REQUEST_IDS = {
    "llm_iterative_improve_document_spec",
    "llm_iterative_improve_runbook_spec",
    "document.spec.generate_iterative",
    "document.runbook.generate_iterative",
}

_DOCUMENT_SPEC_GENERATION_REQUEST_IDS = {
    "llm_generate_document_spec",
    "document.spec.generate",
}

_DOCUMENT_SPEC_MARKDOWN_REQUEST_IDS = {
    "llm_generate_document_spec_from_markdown",
    "document.spec.generate_from_markdown",
}

_DOCUMENT_DEFAULT_AUDIENCE = "general professional audience"
_DOCUMENT_DEFAULT_TONE = "practical"

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
    job_context: Mapping[str, Any] | None = None,
    default_goal: str | None = None,
) -> dict[str, Any]:
    source_payload = merge_document_job_payload(
        job_payload,
        job_context=job_context,
        default_goal=default_goal,
    )
    if not source_payload:
        return {}
    if uses_document_job_payload(tool_name):
        return compact_document_job_payload(source_payload, default_goal=default_goal)
    return dict(source_payload)


def uses_document_job_payload(tool_name: str) -> bool:
    return str(tool_name).strip() in _DOCUMENT_GENERATION_REQUEST_IDS


def project_explicit_inputs_for_tool(
    tool_name: str,
    job_payload: Mapping[str, Any] | None,
    *,
    job_context: Mapping[str, Any] | None = None,
    default_goal: str | None = None,
) -> dict[str, Any]:
    source_payload = merge_document_job_payload(
        job_payload,
        job_context=job_context,
        default_goal=default_goal,
    )
    if not source_payload:
        return {}
    if str(tool_name).strip() in _DOCUMENT_SPEC_GENERATION_REQUEST_IDS:
        return project_document_generation_inputs(
            source_payload,
            default_goal=default_goal,
            apply_defaults=True,
        )
    if str(tool_name).strip() in _DOCUMENT_SPEC_MARKDOWN_REQUEST_IDS:
        return project_markdown_document_generation_inputs(source_payload, default_goal=default_goal)
    return {}


def project_document_generation_inputs(
    job_payload: Mapping[str, Any] | None,
    *,
    default_goal: str | None = None,
    apply_defaults: bool = False,
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
    if apply_defaults and "topic" not in projected:
        topic_fallback = _first_non_empty_str(
            projected.get("instruction"),
            default_goal,
            job_payload.get("goal"),
            compact.get("goal"),
        )
        if topic_fallback:
            projected["topic"] = topic_fallback
    if apply_defaults:
        projected.setdefault("audience", _DOCUMENT_DEFAULT_AUDIENCE)
        projected.setdefault("tone", _DOCUMENT_DEFAULT_TONE)
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


def merge_document_job_payload(
    job_payload: Mapping[str, Any] | None,
    *,
    job_context: Mapping[str, Any] | None = None,
    default_goal: str | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(job_payload) if isinstance(job_payload, Mapping) else {}
    if default_goal and not _first_non_empty_str(merged.get("goal")):
        merged["goal"] = default_goal

    if not isinstance(job_context, Mapping):
        return merged

    for key in _DOCUMENT_JOB_TOP_LEVEL_KEYS:
        if key in merged:
            continue
        _copy_compact_value(merged, key, job_context.get(key))

    existing_context = merged.get("context_json")
    context_json = dict(existing_context) if isinstance(existing_context, Mapping) else {}
    for key in _DOCUMENT_JOB_CONTEXT_KEYS:
        if key in context_json:
            continue
        _copy_compact_value(context_json, key, job_context.get(key))
    if context_json:
        merged["context_json"] = context_json
    return merged


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
