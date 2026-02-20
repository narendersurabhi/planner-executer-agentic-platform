from __future__ import annotations

from typing import Any, Dict

_RESUME_CONTEXT_KEYS = (
    "job_description",
    "candidate_resume",
    "target_role_name",
    "seniority_level",
)


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def select_job_context_from_memory(memory: Any) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    memory_entries = memory.get("job_context")
    if not isinstance(memory_entries, list) or not memory_entries:
        return {}
    normalized: list[Dict[str, Any]] = []
    for entry in memory_entries:
        if not isinstance(entry, dict):
            continue
        payload = entry.get("payload")
        if isinstance(payload, dict):
            normalized_entry = dict(payload)
            for meta_key in ("_memory_key", "_memory_updated_at"):
                if meta_key in entry:
                    normalized_entry[meta_key] = entry[meta_key]
            normalized.append(normalized_entry)
        else:
            normalized.append(dict(entry))
    if not normalized:
        return {}

    def score(entry: Dict[str, Any]) -> tuple[int, int, str]:
        resume = entry.get("candidate_resume")
        resume_len = len(resume.strip()) if isinstance(resume, str) else 0
        job_desc = entry.get("job_description")
        job_desc_len = len(job_desc.strip()) if isinstance(job_desc, str) else 0
        updated_at = str(entry.get("_memory_updated_at") or "")
        return (resume_len, job_desc_len, updated_at)

    best = max(normalized, key=score)
    return {k: v for k, v in best.items() if isinstance(k, str) and not k.startswith("_")}


def merge_resume_job_context(job: Dict[str, Any], memory_context: Dict[str, Any]) -> Dict[str, Any]:
    context_json = job.get("context_json")
    if not isinstance(context_json, dict):
        context_json = {}
    merged_context = dict(context_json)
    merged_context.update(memory_context)
    for key in _RESUME_CONTEXT_KEYS:
        if is_missing_value(merged_context.get(key)) and isinstance(job.get(key), str):
            merged_context[key] = job.get(key)
    return merged_context


def build_resume_job_payload(job: Dict[str, Any] | None, memory: Any) -> Dict[str, Any]:
    job_payload: Dict[str, Any] = dict(job) if isinstance(job, dict) else {}
    memory_context = select_job_context_from_memory(memory)
    merged_context = merge_resume_job_context(job_payload, memory_context)
    if merged_context:
        job_payload["context_json"] = merged_context
        for key in _RESUME_CONTEXT_KEYS:
            value = merged_context.get(key)
            if isinstance(value, str) and value.strip():
                job_payload[key] = value
    return job_payload
