from __future__ import annotations

import re
import os
import time
from typing import Any, Dict

from libs.core import llm_provider, logging as core_logging, prompts

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

_DEFAULT_TAILOR_OPENAI_TIMEOUT_S = 20.0
_DEFAULT_TAILOR_OPENAI_MAX_RETRIES = 1
_DEFAULT_TAILOR_EVAL_OPENAI_TIMEOUT_S = 15.0
_DEFAULT_TAILOR_EVAL_OPENAI_MAX_RETRIES = 1
LOGGER = core_logging.get_logger("tailor")
_ALIGNMENT_FEEDBACK_KEYS = (
    "top_gaps",
    "must_fix_before_95",
    "missing_evidence",
    "recommended_edits",
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


def _resolve_float(primary: str | None, fallback: str | None, default: float) -> float:
    parsed_primary = _parse_optional_float(primary)
    if parsed_primary is not None:
        return parsed_primary
    parsed_fallback = _parse_optional_float(fallback)
    if parsed_fallback is not None:
        return parsed_fallback
    return default


def _resolve_int(primary: str | None, fallback: str | None, default: int) -> int:
    parsed_primary = _parse_optional_int(primary)
    if parsed_primary is not None:
        return parsed_primary
    parsed_fallback = _parse_optional_int(fallback)
    if parsed_fallback is not None:
        return parsed_fallback
    return default


def _min_changes_per_iteration() -> int:
    configured = _parse_optional_int(os.getenv("TAILOR_MIN_CHANGES_PER_ITERATION"))
    if configured is None:
        return 8
    return max(4, configured)


def _resolve_iterative_timeout_budget_s() -> float | None:
    for key in ("TAILOR_ITERATIVE_TIMEOUT_S", "MCP_TOOL_TIMEOUT_S", "MCP_TIMEOUT_S"):
        value = _parse_optional_float(os.getenv(key))
        if value is not None and value > 0:
            return float(value)
    return None


def _resolve_iterative_reserve_s(timeout_budget_s: float) -> float:
    configured = _parse_optional_float(os.getenv("TAILOR_ITERATIVE_RESERVE_S"))
    if configured is not None and configured >= 0:
        return float(configured)
    # Keep headroom between tailor's internal soft deadline and worker MCP route timeout.
    # This gives the service time to return the last completed iteration before caller deadlines.
    mcp_first_attempt_reserve = min(30.0, max(5.0, timeout_budget_s * 0.1))
    extra_return_buffer = min(30.0, max(10.0, timeout_budget_s * 0.05))
    computed = mcp_first_attempt_reserve + extra_return_buffer
    return min(max(1.0, computed), max(1.0, timeout_budget_s - 1.0))


def _remaining_deadline_s(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    return deadline_at - time.monotonic()


def _is_timeout_like_error(detail: str) -> bool:
    lowered = (detail or "").lower()
    return "timed out" in lowered or "timeout" in lowered or "deadline_exhausted" in lowered


def _provider_model(provider: Any) -> str:
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return ""


def _evaluator_mode(evaluator: Dict[str, Any] | None) -> str:
    if isinstance(evaluator, dict):
        mode = evaluator.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip().lower()
    return "self"


def _job_id(job_payload: Dict[str, Any]) -> str:
    job_id = job_payload.get("id")
    if isinstance(job_id, str) and job_id.strip():
        return job_id.strip()
    context = job_payload.get("context_json")
    if isinstance(context, dict):
        context_job_id = context.get("id")
        if isinstance(context_job_id, str) and context_job_id.strip():
            return context_job_id.strip()
    return ""


def create_provider_from_env() -> Any:
    return llm_provider.resolve_provider(
        os.getenv("LLM_PROVIDER", "mock"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        temperature=_parse_optional_float(os.getenv("OPENAI_TEMPERATURE")),
        max_output_tokens=_parse_optional_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS")),
        timeout_s=_resolve_float(
            os.getenv("TAILOR_OPENAI_TIMEOUT_S"),
            os.getenv("OPENAI_TIMEOUT_S"),
            _DEFAULT_TAILOR_OPENAI_TIMEOUT_S,
        ),
        max_retries=_resolve_int(
            os.getenv("TAILOR_OPENAI_MAX_RETRIES"),
            os.getenv("OPENAI_MAX_RETRIES"),
            _DEFAULT_TAILOR_OPENAI_MAX_RETRIES,
        ),
    )


def create_evaluator_from_env(primary_provider: Any) -> Dict[str, Any]:
    mode = os.getenv("TAILOR_EVAL_MODE", "llm").strip().lower()
    if mode not in {"llm", "heuristic", "self"}:
        mode = "llm"
    if mode in {"self", "heuristic"}:
        return {"mode": mode, "provider": primary_provider if mode == "self" else None}
    provider_name = os.getenv("TAILOR_EVAL_PROVIDER", os.getenv("LLM_PROVIDER", "mock"))
    evaluator_provider = llm_provider.resolve_provider(
        provider_name,
        api_key=os.getenv("TAILOR_EVAL_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        model=os.getenv("TAILOR_EVAL_OPENAI_MODEL", os.getenv("OPENAI_MODEL", "")),
        base_url=os.getenv(
            "TAILOR_EVAL_OPENAI_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        ),
        temperature=_parse_optional_float(
            os.getenv("TAILOR_EVAL_OPENAI_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE"))
        ),
        max_output_tokens=_parse_optional_int(
            os.getenv(
                "TAILOR_EVAL_OPENAI_MAX_OUTPUT_TOKENS",
                os.getenv("OPENAI_MAX_OUTPUT_TOKENS"),
            )
        ),
        timeout_s=_resolve_float(
            os.getenv("TAILOR_EVAL_OPENAI_TIMEOUT_S"),
            os.getenv("OPENAI_TIMEOUT_S"),
            _DEFAULT_TAILOR_EVAL_OPENAI_TIMEOUT_S,
        ),
        max_retries=_resolve_int(
            os.getenv("TAILOR_EVAL_OPENAI_MAX_RETRIES"),
            os.getenv("OPENAI_MAX_RETRIES"),
            _DEFAULT_TAILOR_EVAL_OPENAI_MAX_RETRIES,
        ),
    )
    return {"mode": "llm", "provider": evaluator_provider}


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
    _enforce_certifications_from_source(payload, job_payload)
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
        _enforce_certifications_from_source(payload, job_payload)
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


def _enforce_certifications_from_source(
    payload: Dict[str, Any], job_payload: Dict[str, Any]
) -> None:
    certs = payload.get("certifications")
    normalized = _normalize_certification_entries(certs if isinstance(certs, list) else [])
    if normalized:
        payload["certifications"] = normalized
        return

    source_resume = _extract_candidate_resume_text(job_payload)
    if not source_resume:
        payload["certifications"] = []
        return

    backfilled: list[dict[str, Any]] = []
    for line in _extract_certification_lines_from_text(source_resume):
        parsed = _parse_certification_line(line)
        if parsed is not None:
            backfilled.append(parsed)
    payload["certifications"] = backfilled


def _normalize_certification_entries(certs: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for cert in certs:
        parsed = _parse_certification_entry(cert)
        if parsed is not None:
            normalized.append(parsed)
    return normalized


def _parse_certification_entry(cert: Any) -> dict[str, Any] | None:
    if isinstance(cert, str):
        return _parse_certification_line(cert)
    if not isinstance(cert, dict):
        return None

    name = cert.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()

    issuer = cert.get("issuer")
    issuer_text = issuer.strip() if isinstance(issuer, str) and issuer.strip() else ""
    if not issuer_text:
        issuer_text = _infer_issuer(name) or "Credential"

    year = cert.get("year")
    year_value = _normalize_year_value(year)
    if year_value is None:
        year_value = "Not specified"

    parsed: dict[str, Any] = {
        "name": name,
        "issuer": issuer_text,
        "year": year_value,
    }
    url = _extract_cert_url(cert)
    if url:
        parsed["url"] = url
    return parsed


def _parse_certification_line(line: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^[*•\-]\s*", "", line).strip()
    if not cleaned:
        return None

    url_match = re.search(r"https?://\S+", cleaned)
    url = url_match.group(0) if url_match else ""
    if url:
        cleaned = cleaned.replace(url, "").strip().rstrip("|").strip()

    year_match = re.search(r"(19|20)\d{2}(?!.*(19|20)\d{2})", cleaned)
    year_value: int | str = "Not specified"
    if year_match:
        year_value = int(year_match.group(0))
        cleaned = cleaned[: year_match.start()] + cleaned[year_match.end() :]
        cleaned = re.sub(r"[,\-\|\(\)\s]+$", "", cleaned).strip()

    name = cleaned.strip(" -|,")
    if not name:
        return None
    issuer = _infer_issuer(name) or "Credential"

    parsed: dict[str, Any] = {
        "name": name,
        "issuer": issuer,
        "year": year_value,
    }
    if url:
        parsed["url"] = url
    return parsed


def _normalize_year_value(value: Any) -> int | str | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        match = re.search(r"(19|20)\d{2}", trimmed)
        if match:
            return int(match.group(0))
        return trimmed
    return None


def _extract_cert_url(cert: Dict[str, Any]) -> str:
    for key in ("url", "credential_url", "public_url", "link"):
        value = cert.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_candidate_resume_text(job_payload: Dict[str, Any]) -> str:
    direct = job_payload.get("candidate_resume")
    if isinstance(direct, str) and direct.strip():
        return direct
    context = job_payload.get("context_json")
    if isinstance(context, dict):
        candidate_resume = context.get("candidate_resume")
        if isinstance(candidate_resume, str) and candidate_resume.strip():
            return candidate_resume
    return ""


def _extract_certification_lines_from_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = _find_heading_line(lines, {"CERTIFICATIONS", "CERTIFICATION"})
    if start_idx is None:
        return []

    end_headings = {
        "SUMMARY",
        "CORE SKILLS",
        "SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "EXPERIENCE",
        "EDUCATION",
        "OPEN SOURCE",
        "OPEN SOURCE (SELECTED)",
        "PROJECTS",
    }

    collected: list[str] = []
    for raw in lines[start_idx + 1 :]:
        line = raw.strip()
        if not line:
            continue
        if line.upper() in end_headings:
            break
        collected.append(line)
    return collected


def _find_heading_line(lines: list[str], headings: set[str]) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip().upper() in headings:
            return idx
    return None


def _infer_issuer(cert_name: str) -> str:
    stripped = cert_name.strip()
    if not stripped:
        return ""
    first = re.split(r"[\s:/,-]+", stripped)[0]
    if first.isupper() and 1 <= len(first) <= 10:
        return first
    if stripped.startswith("AWS "):
        return "AWS"
    if stripped.startswith("Google "):
        return "Google"
    if stripped.startswith("Microsoft "):
        return "Microsoft"
    return ""


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
    evaluator: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    job_payload = build_resume_job_payload(job, memory)
    current_resume = resolve_tailored_resume(tailored_resume, tailored_text)
    ensure_required_resume_sections(current_resume)
    improved = _run_improve_once(current_resume, job_payload, provider)
    score, summary, feedback = _evaluate_alignment(
        improved["tailored_resume"],
        job_payload,
        evaluator=evaluator,
        fallback_score=improved["self_alignment_score"],
        fallback_summary=improved["self_alignment_summary"],
    )
    return {
        "tailored_resume": improved["tailored_resume"],
        "alignment_score": score,
        "alignment_summary": summary,
        "alignment_feedback": feedback,
    }


def improve_resume_iterative(
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    job: Dict[str, Any],
    memory: Any,
    min_alignment_score: float,
    max_iterations: int,
    provider: Any,
    evaluator: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if max_iterations < 1:
        raise TailorError("max_iterations must be >= 1")

    started_at = time.monotonic()
    timeout_budget_s = _resolve_iterative_timeout_budget_s()
    deadline_at: float | None = None
    if timeout_budget_s is not None:
        reserve_s = _resolve_iterative_reserve_s(timeout_budget_s)
        usable_budget_s = max(1.0, timeout_budget_s - reserve_s)
        deadline_at = started_at + usable_budget_s
    else:
        reserve_s = None
        usable_budget_s = None
    job_payload = build_resume_job_payload(job, memory)
    job_id = _job_id(job_payload)
    current_resume = resolve_tailored_resume(tailored_resume, tailored_text)
    ensure_required_resume_sections(current_resume)
    evaluator_mode = _evaluator_mode(evaluator)
    LOGGER.info(
        "improve_iterative_started",
        job_id=job_id,
        min_alignment_score=float(min_alignment_score),
        max_iterations=int(max_iterations),
        evaluator_mode=evaluator_mode,
        provider_type=provider.__class__.__name__,
        provider_model=_provider_model(provider),
        evaluator_provider_model=_provider_model(
            evaluator.get("provider") if isinstance(evaluator, dict) else None
        ),
        timeout_budget_s=timeout_budget_s,
        timeout_reserve_s=reserve_s,
        timeout_usable_budget_s=usable_budget_s,
    )

    best_resume = current_resume
    best_score = -1.0
    best_summary = ""
    best_feedback = _empty_alignment_feedback()
    reached = False
    history: list[dict[str, Any]] = []
    feedback_summary = ""
    feedback_score: float | None = None
    min_changes_required = _min_changes_per_iteration()

    try:
        for iteration_idx in range(max_iterations):
            iteration = iteration_idx + 1
            iteration_started_at = time.monotonic()
            remaining_before_s = _remaining_deadline_s(deadline_at)
            if remaining_before_s is not None and remaining_before_s <= 0.05:
                LOGGER.warning(
                    "improve_iterative_stopped_early",
                    job_id=job_id,
                    reason="deadline_exhausted_before_iteration",
                    iteration=iteration,
                    completed_iterations=int(len(history)),
                    best_alignment_score=float(best_score),
                )
                if history:
                    break
                raise TailorError("iterative_timeout:deadline_exhausted_before_iteration")
            LOGGER.info(
                "improve_iterative_iteration_started",
                job_id=job_id,
                iteration=iteration,
                max_iterations=int(max_iterations),
                min_alignment_score=float(min_alignment_score),
                previous_alignment_score=feedback_score,
                has_evaluator_feedback=bool(feedback_summary.strip()),
                deadline_remaining_s=remaining_before_s,
            )
            try:
                improved = _run_improve_once(
                    current_resume,
                    job_payload,
                    provider,
                    evaluator_feedback=feedback_summary,
                    previous_alignment_score=feedback_score,
                    min_required_changes=min_changes_required,
                    deadline_at=deadline_at,
                )
                score, summary, feedback = _evaluate_alignment(
                    improved["tailored_resume"],
                    job_payload,
                    evaluator=evaluator,
                    fallback_score=improved["self_alignment_score"],
                    fallback_summary=improved["self_alignment_summary"],
                    deadline_at=deadline_at,
                )
            except TailorError as exc:
                if _is_timeout_like_error(exc.detail) and history:
                    LOGGER.warning(
                        "improve_iterative_stopped_early",
                        job_id=job_id,
                        reason="iteration_timeout",
                        iteration=iteration,
                        completed_iterations=int(len(history)),
                        best_alignment_score=float(best_score),
                        error=exc.detail,
                    )
                    break
                raise
            LOGGER.info(
                "improve_iterative_iteration_scored",
                job_id=job_id,
                iteration=iteration,
                alignment_score=float(score),
                self_alignment_score=float(improved["self_alignment_score"]),
                change_count=int(improved["change_count"]),
            )
            evaluator_feedback_text = _compose_alignment_feedback_text(summary, feedback)
            if (
                score < float(min_alignment_score)
                and improved["change_count"] < min_changes_required
            ):
                strengthened_feedback = (
                    f"{evaluator_feedback_text}\n"
                    f"The previous rewrite changed only {improved['change_count']} leaf field(s). "
                    f"Apply at least {min_changes_required} major, high-impact edits that directly address this feedback."
                )
                LOGGER.info(
                    "improve_iterative_iteration_strengthen_feedback",
                    job_id=job_id,
                    iteration=iteration,
                    change_count=int(improved["change_count"]),
                    required_changes=int(min_changes_required),
                )
                improved = _run_improve_once(
                    current_resume,
                    job_payload,
                    provider,
                    evaluator_feedback=strengthened_feedback,
                    previous_alignment_score=score,
                    min_required_changes=min_changes_required,
                    deadline_at=deadline_at,
                )
                score, summary, feedback = _evaluate_alignment(
                    improved["tailored_resume"],
                    job_payload,
                    evaluator=evaluator,
                    fallback_score=improved["self_alignment_score"],
                    fallback_summary=improved["self_alignment_summary"],
                    deadline_at=deadline_at,
                )
                LOGGER.info(
                    "improve_iterative_iteration_rescored",
                    job_id=job_id,
                    iteration=iteration,
                    alignment_score=float(score),
                    self_alignment_score=float(improved["self_alignment_score"]),
                    change_count=int(improved["change_count"]),
                )
            history.append(
                {
                    "alignment_score": score,
                    "alignment_summary": summary,
                    "alignment_feedback": feedback,
                }
            )
            if score > best_score:
                best_score = score
                best_resume = improved["tailored_resume"]
                best_summary = summary
                best_feedback = feedback
            current_resume = improved["tailored_resume"]
            feedback_summary = _compose_alignment_feedback_text(summary, feedback)
            feedback_score = score
            LOGGER.info(
                "improve_iterative_iteration_finished",
                job_id=job_id,
                iteration=iteration,
                alignment_score=float(score),
                best_alignment_score=float(best_score),
                reached_threshold=bool(score >= float(min_alignment_score)),
                duration_ms=int(max(0.0, time.monotonic() - iteration_started_at) * 1000),
            )
            if score >= float(min_alignment_score):
                reached = True
                break
    except TailorError as exc:
        LOGGER.warning(
            "improve_iterative_failed",
            job_id=job_id,
            error=exc.detail,
            duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
        )
        raise

    result = {
        "tailored_resume": best_resume,
        "alignment_score": best_score,
        "alignment_summary": best_summary,
        "alignment_feedback": best_feedback,
        "iterations": len(history),
        "reached_threshold": reached,
        "history": history,
    }
    LOGGER.info(
        "improve_iterative_finished",
        job_id=job_id,
        iterations=int(len(history)),
        reached_threshold=bool(reached),
        best_alignment_score=float(best_score),
        duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
    )
    return result


def _run_improve_once(
    current_resume: Dict[str, Any],
    job_payload: Dict[str, Any],
    provider: Any,
    *,
    evaluator_feedback: str | None = None,
    previous_alignment_score: float | None = None,
    min_required_changes: int = 1,
    deadline_at: float | None = None,
) -> Dict[str, Any]:
    prompt = prompts.resume_tailoring_improve_prompt(
        current_resume,
        job=job_payload,
        evaluator_feedback=evaluator_feedback,
        previous_alignment_score=previous_alignment_score,
        min_required_changes=min_required_changes,
    )
    started_at = time.monotonic()
    LOGGER.info(
        "improve_once_started",
        job_id=_job_id(job_payload),
        provider_type=provider.__class__.__name__,
        provider_model=_provider_model(provider),
        has_evaluator_feedback=bool((evaluator_feedback or "").strip()),
        previous_alignment_score=previous_alignment_score,
        min_required_changes=int(min_required_changes),
    )
    response = _generate(provider, prompt, deadline_at=deadline_at)
    payload = _parse_json_response_with_repair(
        response_text=response.content,
        provider=provider,
        prompt_for_retry=prompt,
        deadline_at=deadline_at,
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
    applied_changes = _normalize_applied_changes(payload.get("applied_changes"))
    _normalize_top_level_sections(improved)
    _enforce_certifications_from_source(improved, job_payload)
    _apply_header_title_fallback(improved, job_payload)
    ensure_required_resume_sections(improved)
    change_count = _count_resume_leaf_changes(current_resume, improved)
    duration_ms = int(max(0.0, time.monotonic() - started_at) * 1000)
    LOGGER.info(
        "improve_once_finished",
        job_id=_job_id(job_payload),
        self_alignment_score=float(score),
        change_count=int(change_count),
        applied_changes_count=int(len(applied_changes)),
        duration_ms=duration_ms,
    )

    return {
        "tailored_resume": improved,
        "self_alignment_score": float(score),
        "self_alignment_summary": summary.strip(),
        "applied_changes": applied_changes,
        "change_count": change_count,
    }


def _empty_alignment_feedback() -> Dict[str, list[str]]:
    return {key: [] for key in _ALIGNMENT_FEEDBACK_KEYS}


def _normalize_feedback_items(value: Any, *, max_items: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        normalized.append(cleaned)
        if len(normalized) >= max_items:
            break
    return normalized


def _extract_alignment_feedback(payload: Dict[str, Any]) -> Dict[str, list[str]]:
    feedback = _empty_alignment_feedback()
    for key in _ALIGNMENT_FEEDBACK_KEYS:
        feedback[key] = _normalize_feedback_items(payload.get(key))
    return feedback


def _feedback_has_content(feedback: Dict[str, list[str]]) -> bool:
    return any(bool(feedback.get(key)) for key in _ALIGNMENT_FEEDBACK_KEYS)


def _compose_alignment_feedback_text(summary: str, feedback: Dict[str, list[str]]) -> str:
    lines: list[str] = []
    if isinstance(summary, str) and summary.strip():
        lines.append(summary.strip())
    labels = {
        "top_gaps": "Top gap",
        "must_fix_before_95": "Must fix before 95",
        "missing_evidence": "Missing evidence",
        "recommended_edits": "Recommended edit",
    }
    for key in _ALIGNMENT_FEEDBACK_KEYS:
        label = labels[key]
        for item in feedback.get(key, []):
            lines.append(f"{label}: {item}")
    return "\n".join(lines).strip()


def _evaluate_alignment(
    tailored_resume: Dict[str, Any],
    job_payload: Dict[str, Any],
    evaluator: Dict[str, Any] | None,
    fallback_score: float,
    fallback_summary: str,
    deadline_at: float | None = None,
) -> tuple[float, str, Dict[str, list[str]]]:
    mode = "self"
    provider = None
    if isinstance(evaluator, dict):
        maybe_mode = evaluator.get("mode")
        if isinstance(maybe_mode, str) and maybe_mode.strip():
            mode = maybe_mode.strip().lower()
        provider = evaluator.get("provider")

    if mode == "heuristic":
        score, summary = _heuristic_alignment(tailored_resume, job_payload)
        feedback = _empty_alignment_feedback()
        feedback["top_gaps"] = [summary.strip()]
        LOGGER.info(
            "alignment_evaluation_finished",
            job_id=_job_id(job_payload),
            evaluator_mode="heuristic",
            alignment_score=float(score),
        )
        return score, summary, feedback

    if mode == "llm":
        if provider is None:
            raise TailorError("tailor_evaluator_provider_missing")
        started_at = time.monotonic()
        LOGGER.info(
            "alignment_evaluation_started",
            job_id=_job_id(job_payload),
            evaluator_mode="llm",
            evaluator_provider_type=provider.__class__.__name__,
            evaluator_provider_model=_provider_model(provider),
        )
        prompt = prompts.resume_tailoring_evaluation_prompt(tailored_resume, job=job_payload)
        response = _generate(provider, prompt, deadline_at=deadline_at)
        payload = _parse_json_response_with_repair(
            response_text=response.content,
            provider=provider,
            prompt_for_retry=prompt,
            deadline_at=deadline_at,
        )
        score = payload.get("alignment_score")
        summary = payload.get("alignment_summary")
        if not isinstance(score, (int, float)):
            raise TailorError("alignment_score must be a number")
        feedback = _extract_alignment_feedback(payload)
        summary_text = summary.strip() if isinstance(summary, str) and summary.strip() else ""
        if not summary_text and not _feedback_has_content(feedback):
            raise TailorError("alignment_summary or structured feedback must be present")
        if not summary_text:
            summary_text = _compose_alignment_feedback_text("", feedback)
        LOGGER.info(
            "alignment_evaluation_finished",
            job_id=_job_id(job_payload),
            evaluator_mode="llm",
            alignment_score=float(score),
            top_gaps_count=len(feedback["top_gaps"]),
            must_fix_count=len(feedback["must_fix_before_95"]),
            missing_evidence_count=len(feedback["missing_evidence"]),
            recommended_edits_count=len(feedback["recommended_edits"]),
            duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
        )
        return float(score), summary_text, feedback

    # self mode falls back to the model-provided score from improve output.
    fallback_text = str(fallback_summary).strip()
    feedback = _empty_alignment_feedback()
    if fallback_text:
        feedback["top_gaps"] = [fallback_text]
    LOGGER.info(
        "alignment_evaluation_finished",
        job_id=_job_id(job_payload),
        evaluator_mode="self",
        alignment_score=float(fallback_score),
    )
    return float(fallback_score), fallback_text, feedback


def _normalize_applied_changes(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return normalized


def _count_resume_leaf_changes(before: Dict[str, Any], after: Dict[str, Any]) -> int:
    before_core = {k: v for k, v in before.items() if k != "schema_version"}
    after_core = {k: v for k, v in after.items() if k != "schema_version"}
    before_leaves = _flatten_leaf_values(before_core)
    after_leaves = _flatten_leaf_values(after_core)
    paths = set(before_leaves.keys()) | set(after_leaves.keys())
    changed = 0
    for path in paths:
        if before_leaves.get(path) != after_leaves.get(path):
            changed += 1
    return changed


def _flatten_leaf_values(value: Any, prefix: str = "") -> dict[str, str]:
    leaves: dict[str, str] = {}
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_leaf_values(value.get(key), next_prefix))
        return leaves
    if isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            leaves.update(_flatten_leaf_values(item, next_prefix))
        return leaves
    leaves[prefix or "$"] = repr(value)
    return leaves


def _heuristic_alignment(
    tailored_resume: Dict[str, Any], job_payload: Dict[str, Any]
) -> tuple[float, str]:
    context = job_payload.get("context_json")
    job_text = ""
    if isinstance(context, dict):
        maybe_jd = context.get("job_description")
        if isinstance(maybe_jd, str):
            job_text = maybe_jd
    if not job_text:
        maybe_goal = job_payload.get("goal")
        if isinstance(maybe_goal, str):
            job_text = maybe_goal

    resume_text = _tailored_resume_to_text(tailored_resume)
    job_tokens = _keyword_tokens(job_text)
    resume_tokens = _keyword_tokens(resume_text)
    if not job_tokens:
        return (
            75.0,
            "Heuristic evaluator used fallback score because job description keywords were missing.",
        )

    overlap = len(job_tokens & resume_tokens)
    coverage = overlap / max(len(job_tokens), 1)
    score = max(0.0, min(100.0, round(50.0 + coverage * 50.0, 1)))
    summary = (
        "Heuristic evaluator score based on keyword coverage between job description and tailored resume. "
        f"Matched {overlap} of {len(job_tokens)} core job keywords."
    )
    return score, summary


def _tailored_resume_to_text(tailored_resume: Dict[str, Any]) -> str:
    parts: list[str] = []
    summary = tailored_resume.get("summary")
    if isinstance(summary, str):
        parts.append(summary)
    skills = tailored_resume.get("skills")
    if isinstance(skills, list):
        for group in skills:
            if not isinstance(group, dict):
                continue
            name = group.get("group_name")
            if isinstance(name, str):
                parts.append(name)
            items = group.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        parts.append(item)
    experience = tailored_resume.get("experience")
    if isinstance(experience, list):
        for role in experience:
            if not isinstance(role, dict):
                continue
            for key in ("company", "title", "location"):
                value = role.get(key)
                if isinstance(value, str):
                    parts.append(value)
            bullets = role.get("bullets")
            if isinstance(bullets, list):
                for bullet in bullets:
                    if isinstance(bullet, str):
                        parts.append(bullet)
            groups = role.get("groups")
            if isinstance(groups, list):
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    heading = group.get("heading")
                    if isinstance(heading, str):
                        parts.append(heading)
                    g_bullets = group.get("bullets")
                    if isinstance(g_bullets, list):
                        for bullet in g_bullets:
                            if isinstance(bullet, str):
                                parts.append(bullet)
    return " ".join(parts)


def _keyword_tokens(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9+.#-]{2,}", text.lower()))
    stop = {
        "with",
        "that",
        "this",
        "from",
        "have",
        "will",
        "your",
        "role",
        "team",
        "years",
        "year",
        "plus",
        "using",
        "build",
        "built",
        "work",
        "works",
        "strong",
        "experience",
    }
    return {token for token in tokens if token not in stop}


def _generate(provider: Any, prompt: str, *, deadline_at: float | None = None) -> Any:
    started_at = time.monotonic()
    original_timeout = getattr(provider, "timeout_s", None)
    temporary_timeout: float | None = None
    if deadline_at is not None:
        remaining_s = _remaining_deadline_s(deadline_at)
        if remaining_s is None or remaining_s <= 0.05:
            raise TailorError(
                "iterative_timeout:deadline_exhausted_before_llm_call", status_code=504
            )
        temporary_timeout = float(remaining_s)
        if isinstance(original_timeout, (int, float)):
            temporary_timeout = min(float(original_timeout), temporary_timeout)
        if hasattr(provider, "timeout_s"):
            setattr(provider, "timeout_s", max(0.05, temporary_timeout))
    try:
        response = provider.generate(prompt)
        LOGGER.info(
            "llm_generate_finished",
            provider_type=provider.__class__.__name__,
            provider_model=_provider_model(provider),
            prompt_chars=int(len(prompt)),
            timeout_s=getattr(provider, "timeout_s", None),
            duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
        )
        return response
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "llm_generate_failed",
            provider_type=provider.__class__.__name__,
            provider_model=_provider_model(provider),
            prompt_chars=int(len(prompt)),
            timeout_s=getattr(provider, "timeout_s", None),
            duration_ms=int(max(0.0, time.monotonic() - started_at) * 1000),
            error=str(exc),
        )
        raise TailorError(str(exc), status_code=502) from exc
    finally:
        if temporary_timeout is not None and hasattr(provider, "timeout_s"):
            setattr(provider, "timeout_s", original_timeout)


def _parse_json_response_with_repair(
    response_text: str,
    provider: Any,
    prompt_for_retry: str,
    deadline_at: float | None = None,
) -> Dict[str, Any]:
    initial_json_text = extract_json(response_text)
    try:
        return parse_json_object(initial_json_text)
    except TailorError as exc:
        if not _is_invalid_json_error(exc):
            raise

    retry_response = _generate(
        provider,
        f"{prompt_for_retry}{_STRICT_JSON_SUFFIX}",
        deadline_at=deadline_at,
    )
    retry_json_text = extract_json(retry_response.content)
    try:
        return parse_json_object(retry_json_text)
    except TailorError as exc:
        if not _is_invalid_json_error(exc):
            raise

    malformed = retry_json_text.strip() or initial_json_text.strip() or response_text.strip()
    repair_prompt = f"{_JSON_REPAIR_PROMPT}\nMalformed JSON:\n{malformed}\n"
    repaired_response = _generate(provider, repair_prompt, deadline_at=deadline_at)
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
