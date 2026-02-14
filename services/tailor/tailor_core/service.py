from __future__ import annotations

import os
from typing import Any, Dict

from libs.core import llm_provider, prompts

from .context import build_resume_job_payload
from .errors import TailorError
from .validation import (
    ensure_required_resume_sections,
    extract_json,
    parse_json_object,
    resolve_tailored_resume,
)

_STRICT_JSON_SUFFIX = (
    "\n\nSTRICT OUTPUT REQUIREMENTS:\n"
    "- Return exactly one JSON object.\n"
    "- No markdown, no code fences, no prose.\n"
    "- Escape embedded quotes and newlines inside string values.\n"
    "- Ensure valid commas, braces, and brackets.\n"
)

_JSON_REPAIR_PROMPT = (
    "You repair malformed JSON.\n"
    "Return ONLY one valid JSON object.\n"
    "Do not add explanations.\n"
    "Preserve keys and values as much as possible.\n"
    "Fix only JSON syntax issues (commas, quotes, brackets, escaping).\n"
)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def create_provider_from_env() -> Any:
    return llm_provider.resolve_provider(
        os.getenv("LLM_PROVIDER", "mock"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        temperature=_parse_optional_float(os.getenv("OPENAI_TEMPERATURE")),
        max_output_tokens=_parse_optional_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS")),
        timeout_s=_parse_optional_float(os.getenv("OPENAI_TIMEOUT_S")),
        max_retries=_parse_optional_int(os.getenv("OPENAI_MAX_RETRIES")),
    )


def tailor_resume(job: Dict[str, Any], memory: Any, provider: Any) -> Dict[str, Any]:
    job_payload = build_resume_job_payload(job, memory)
    merged_context = job_payload.get("context_json", {})
    candidate_resume = merged_context.get("candidate_resume")
    if not isinstance(candidate_resume, str) or not candidate_resume.strip():
        raise TailorError("candidate_resume_missing")

    prompt = prompts.resume_tailoring_prompt(job_payload)
    response = _generate(provider, prompt)
    payload = _parse_json_response_with_repair(
        response_text=response.content,
        provider=provider,
        prompt_for_retry=prompt,
    )
    if _is_missing_header_title_error(payload):
        retry_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT:\n"
            "- If header.title is missing in the candidate resume, set header.title from target_role_name.\n"
            "- Do not return missing_required_fields for header.title when target_role_name is available.\n"
        )
        retry_response = _generate(provider, retry_prompt)
        payload = _parse_json_response_with_repair(
            response_text=retry_response.content,
            provider=provider,
            prompt_for_retry=retry_prompt,
        )
    if "error" in payload:
        if _is_missing_required_fields_error(payload):
            payload = _repair_missing_sections_payload(
                payload=payload,
                job_payload=job_payload,
                provider=provider,
                original_prompt=prompt,
                reason="missing_required_fields_error_payload",
            )
        else:
            error = payload.get("error")
            missing_fields = payload.get("missing_fields")
            if isinstance(missing_fields, list):
                missing_fields = ",".join(str(item) for item in missing_fields)
            detail = f"tailored_resume_error:{error}"
            if missing_fields:
                detail = f"{detail}:{missing_fields}"
            raise TailorError(detail)

    _normalize_top_level_sections(payload)
    _apply_header_title_fallback(payload, job_payload)
    try:
        ensure_required_resume_sections(payload)
    except TailorError as exc:
        if not _is_missing_sections_validation_error(exc):
            raise
        payload = _repair_missing_sections_payload(
            payload=payload,
            job_payload=job_payload,
            provider=provider,
            original_prompt=prompt,
            reason=exc.detail,
        )
        _normalize_top_level_sections(payload)
        _apply_header_title_fallback(payload, job_payload)
        ensure_required_resume_sections(payload)
    return payload


def _is_missing_required_fields_error(payload: Dict[str, Any]) -> bool:
    if payload.get("error") != "missing_required_fields":
        return False
    missing_fields = payload.get("missing_fields")
    if not isinstance(missing_fields, list):
        return False
    return len(missing_fields) > 0


def _is_missing_sections_validation_error(exc: TailorError) -> bool:
    return exc.detail.startswith("tailored_resume_missing_fields:")


def _normalize_top_level_sections(payload: Dict[str, Any]) -> None:
    aliases = {
        "experience": ("roles", "work_experience", "professional_experience"),
        "education": ("academics", "education_history"),
        "certifications": ("certs", "licenses"),
    }
    for canonical, options in aliases.items():
        if canonical in payload:
            continue
        for alias in options:
            candidate = payload.get(alias)
            if isinstance(candidate, list):
                payload[canonical] = candidate
                break

    if "education" not in payload:
        payload["education"] = []
    if "certifications" not in payload:
        payload["certifications"] = []


def _repair_missing_sections_payload(
    payload: Dict[str, Any],
    job_payload: Dict[str, Any],
    provider: Any,
    original_prompt: str,
    reason: str,
) -> Dict[str, Any]:
    repair_prompt = (
        f"{original_prompt}\n\n"
        "REPAIR MODE:\n"
        "- Your last output was incomplete.\n"
        f"- Failure reason: {reason}\n"
        "- Return ONE complete tailored_resume JSON object with ALL required top-level keys:\n"
        "  schema_version, header, summary, skills, experience, education, certifications.\n"
        "- experience must be a non-empty array.\n"
        "- education and certifications must be arrays (empty [] if unavailable).\n"
        "- Do not return an error object.\n"
        f"- Previous partial payload: {payload}\n"
    )
    repaired_response = _generate(provider, repair_prompt)
    repaired_payload = _parse_json_response_with_repair(
        response_text=repaired_response.content,
        provider=provider,
        prompt_for_retry=repair_prompt,
    )
    if "error" in repaired_payload:
        error = repaired_payload.get("error")
        missing_fields = repaired_payload.get("missing_fields")
        if isinstance(missing_fields, list):
            missing_fields = ",".join(str(item) for item in missing_fields)
        detail = f"tailored_resume_error:{error}"
        if missing_fields:
            detail = f"{detail}:{missing_fields}"
        raise TailorError(detail)
    return repaired_payload


def improve_resume(
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    job: Dict[str, Any],
    memory: Any,
    provider: Any,
) -> Dict[str, Any]:
    job_payload = build_resume_job_payload(job, memory)
    current_resume = resolve_tailored_resume(tailored_resume, tailored_text)
    ensure_required_resume_sections(current_resume)
    return _run_improve_once(current_resume, job_payload, provider)


def improve_resume_iterative(
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    job: Dict[str, Any],
    memory: Any,
    min_alignment_score: float,
    max_iterations: int,
    provider: Any,
) -> Dict[str, Any]:
    if max_iterations < 1:
        raise TailorError("max_iterations must be >= 1")

    job_payload = build_resume_job_payload(job, memory)
    current_resume = resolve_tailored_resume(tailored_resume, tailored_text)
    ensure_required_resume_sections(current_resume)

    best_resume = current_resume
    best_score = -1.0
    best_summary = ""
    reached = False
    history: list[dict[str, Any]] = []

    for _ in range(max_iterations):
        improved = _run_improve_once(current_resume, job_payload, provider)
        score = improved["alignment_score"]
        summary = improved["alignment_summary"]
        history.append(
            {
                "alignment_score": score,
                "alignment_summary": summary,
            }
        )
        if score > best_score:
            best_score = score
            best_resume = improved["tailored_resume"]
            best_summary = summary
        current_resume = improved["tailored_resume"]
        if score >= float(min_alignment_score):
            reached = True
            break

    return {
        "tailored_resume": best_resume,
        "alignment_score": best_score,
        "alignment_summary": best_summary,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }


def _run_improve_once(
    current_resume: Dict[str, Any], job_payload: Dict[str, Any], provider: Any
) -> Dict[str, Any]:
    prompt = prompts.resume_tailoring_improve_prompt(current_resume, job=job_payload)
    response = _generate(provider, prompt)
    payload = _parse_json_response_with_repair(
        response_text=response.content,
        provider=provider,
        prompt_for_retry=prompt,
    )
    improved = payload.get("tailored_resume")
    score = payload.get("alignment_score")
    summary = payload.get("alignment_summary")

    if not isinstance(improved, dict):
        raise TailorError("tailored_resume must be an object")
    if not isinstance(score, (int, float)):
        raise TailorError("alignment_score must be a number")
    if not isinstance(summary, str) or not summary.strip():
        raise TailorError("alignment_summary must be a non-empty string")
    _apply_header_title_fallback(improved, job_payload)
    ensure_required_resume_sections(improved)

    return {
        "tailored_resume": improved,
        "alignment_score": float(score),
        "alignment_summary": summary.strip(),
    }


def _generate(provider: Any, prompt: str) -> Any:
    try:
        return provider.generate(prompt)
    except Exception as exc:  # noqa: BLE001
        raise TailorError(str(exc), status_code=502) from exc


def _parse_json_response_with_repair(
    response_text: str,
    provider: Any,
    prompt_for_retry: str,
) -> Dict[str, Any]:
    initial_json_text = extract_json(response_text)
    try:
        return parse_json_object(initial_json_text)
    except TailorError as exc:
        if not _is_invalid_json_error(exc):
            raise

    retry_response = _generate(provider, f"{prompt_for_retry}{_STRICT_JSON_SUFFIX}")
    retry_json_text = extract_json(retry_response.content)
    try:
        return parse_json_object(retry_json_text)
    except TailorError as exc:
        if not _is_invalid_json_error(exc):
            raise

    malformed = retry_json_text.strip() or initial_json_text.strip() or response_text.strip()
    repair_prompt = f"{_JSON_REPAIR_PROMPT}\nMalformed JSON:\n{malformed}\n"
    repaired_response = _generate(provider, repair_prompt)
    repaired_json_text = extract_json(repaired_response.content)
    return parse_json_object(repaired_json_text)


def _is_invalid_json_error(exc: TailorError) -> bool:
    return exc.detail.startswith("invalid_json")


def _is_missing_header_title_error(payload: Dict[str, Any]) -> bool:
    if payload.get("error") != "missing_required_fields":
        return False
    missing_fields = payload.get("missing_fields")
    if not isinstance(missing_fields, list):
        return False
    normalized = [str(field).strip() for field in missing_fields if str(field).strip()]
    return normalized == ["header.title"]


def _apply_header_title_fallback(payload: Dict[str, Any], job_payload: Dict[str, Any]) -> None:
    header = payload.get("header")
    if not isinstance(header, dict):
        return
    title = header.get("title")
    if isinstance(title, str) and title.strip():
        return

    fallback = _resolve_header_title_fallback(payload, job_payload)
    if fallback:
        header["title"] = fallback


def _resolve_header_title_fallback(payload: Dict[str, Any], job_payload: Dict[str, Any]) -> str:
    candidates: list[Any] = []
    context_json = job_payload.get("context_json")
    if isinstance(context_json, dict):
        candidates.append(context_json.get("target_role_name"))
    candidates.append(job_payload.get("target_role_name"))
    experience = payload.get("experience")
    if isinstance(experience, list) and experience:
        first_role = experience[0]
        if isinstance(first_role, dict):
            candidates.append(first_role.get("title"))

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
