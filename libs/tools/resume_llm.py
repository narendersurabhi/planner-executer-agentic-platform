from __future__ import annotations

import json
import re
import os
from typing import Any

from libs.core import prompts
from libs.core.llm_provider import LLMProvider
from libs.framework.tool_runtime import ToolExecutionError


_RESUME_CONTEXT_KEYS = (
    "job_description",
    "candidate_resume",
    "target_role_name",
    "seniority_level",
)


def llm_tailor_resume_text(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    post_mcp_tool_call: Any,
) -> dict[str, Any]:
    del provider
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    memory = payload.get("memory")
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    response = post_mcp_tool_call(
        tailor_url,
        "tailor_resume",
        {"job": job, "memory": memory},
    )
    resume_payload = response.get("tailored_resume") if isinstance(response, dict) else None
    if not isinstance(resume_payload, dict):
        raise ToolExecutionError("tailored_resume must be an object")
    return {"tailored_resume": resume_payload}


def llm_improve_tailored_resume_text(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    post_mcp_tool_call: Any,
) -> dict[str, Any]:
    del provider
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    request_payload = {
        "tailored_resume": payload.get("tailored_resume"),
        "tailored_text": payload.get("tailored_text"),
        "job": payload.get("job") or {},
        "memory": payload.get("memory"),
    }
    response = post_mcp_tool_call(
        tailor_url,
        "improve_resume",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("tailor_response_invalid")
    return response


def llm_iterative_improve_tailored_resume_text(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    post_mcp_tool_call: Any,
) -> dict[str, Any]:
    del provider
    tailor_url = os.getenv("RESUME_TAILOR_API_URL")
    if not tailor_url:
        raise ToolExecutionError("RESUME_TAILOR_API_URL not set")
    request_payload = {
        "tailored_resume": payload.get("tailored_resume"),
        "tailored_text": payload.get("tailored_text"),
        "job": payload.get("job") or {},
        "memory": payload.get("memory"),
        "min_alignment_score": payload.get("min_alignment_score", 85),
        "max_iterations": payload.get("max_iterations", 2),
    }
    response = post_mcp_tool_call(
        tailor_url,
        "improve_iterative",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("tailor_response_invalid")
    return response


def parse_target_pages(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None
        parsed = int(value)
    elif isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if re.fullmatch(r"\d+", trimmed):
            parsed = int(trimmed)
        else:
            match = re.search(r"\b([12])\s*page(?:s)?\b", trimmed.lower())
            if match is None:
                match = re.search(r"\bpage(?:s)?\s*([12])\b", trimmed.lower())
            if match is None:
                return None
            parsed = int(match.group(1))
    else:
        return None
    if parsed in {1, 2}:
        return parsed
    return None


def resolve_target_pages(payload: dict[str, Any], job: Any) -> int | None:
    candidates: list[Any] = [payload.get("target_pages"), payload.get("page_count")]
    if isinstance(job, dict):
        candidates.extend([job.get("target_pages"), job.get("page_count")])
        context_json = job.get("context_json")
        if isinstance(context_json, dict):
            candidates.extend([context_json.get("target_pages"), context_json.get("page_count")])
    for candidate in candidates:
        parsed = parse_target_pages(candidate)
        if parsed is not None:
            return parsed
    return None


def _trim_non_empty_strings(items: Any, max_items: int) -> list[str]:
    if not isinstance(items, list):
        return []
    if max_items < 1:
        return []
    normalized = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    return normalized[:max_items]


def _trim_summary_sentences(text: str, max_sentences: int) -> str:
    cleaned = text.strip()
    if max_sentences < 1 or not cleaned:
        return cleaned
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if len(parts) <= max_sentences:
        return cleaned
    return " ".join(parts[:max_sentences]).strip()


def apply_resume_target_pages_policy(
    resume_doc_spec: dict[str, Any], target_pages: int | None
) -> None:
    if target_pages not in {1, 2}:
        return

    page = resume_doc_spec.get("page")
    if not isinstance(page, dict):
        page = {}
        resume_doc_spec["page"] = page
    margins_in = page.get("margins_in")
    if not isinstance(margins_in, dict):
        margins_in = {}
        page["margins_in"] = margins_in

    defaults = resume_doc_spec.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
        resume_doc_spec["defaults"] = defaults

    if target_pages == 1:
        margins_in.update({"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45})
        defaults["font_size_pt"] = 10.5
        defaults["line_spacing"] = 1.0
        skills_group_limit = 5
        first_role_bullet_limit = 4
        other_role_bullet_limit = 2
        experience_group_bullet_limit = 2
        summary_sentence_limit = 2
        aux_bullet_limit = 2
        total_experience_bullets_limit = 9
        total_aux_bullets_limit = 2
    else:
        margins_in.update({"top": 0.45, "right": 0.45, "bottom": 0.45, "left": 0.45})
        defaults["font_size_pt"] = 10.75
        defaults["line_spacing"] = 1.0
        skills_group_limit = 6
        first_role_bullet_limit = 6
        other_role_bullet_limit = 3
        experience_group_bullet_limit = 3
        summary_sentence_limit = 3
        aux_bullet_limit = 3
        total_experience_bullets_limit = 16
        total_aux_bullets_limit = 3

    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    current_section = ""
    role_index = 0
    total_experience_bullets = 0
    total_aux_bullets = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "section_heading":
            heading = block.get("text")
            current_section = heading.strip().upper() if isinstance(heading, str) else ""
            continue
        if block_type == "definition_list" and current_section == "SKILLS":
            items = block.get("items")
            if isinstance(items, list):
                block["items"] = [item for item in items if isinstance(item, dict)][
                    :skills_group_limit
                ]
            continue
        if block_type == "paragraph" and current_section == "SUMMARY":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                block["text"] = _trim_summary_sentences(text, summary_sentence_limit)
            continue
        if block_type == "role" and current_section == "EXPERIENCE":
            role_index += 1
            bullets = block.get("bullets")
            if isinstance(bullets, list):
                max_items = first_role_bullet_limit if role_index == 1 else other_role_bullet_limit
                trimmed = _trim_non_empty_strings(bullets, max_items)
                remaining = max(0, total_experience_bullets_limit - total_experience_bullets)
                block["bullets"] = trimmed[:remaining]
                total_experience_bullets += len(block["bullets"])
            continue
        if block_type == "bullets":
            items = block.get("items")
            if current_section == "EXPERIENCE":
                trimmed = _trim_non_empty_strings(items, experience_group_bullet_limit)
                remaining = max(0, total_experience_bullets_limit - total_experience_bullets)
                block["items"] = trimmed[:remaining]
                total_experience_bullets += len(block["items"])
            elif current_section in {"OPEN SOURCE", "PROJECTS", "OPEN SOURCE (SELECTED)"}:
                trimmed = _trim_non_empty_strings(items, aux_bullet_limit)
                remaining = max(0, total_aux_bullets_limit - total_aux_bullets)
                block["items"] = trimmed[:remaining]
                total_aux_bullets += len(block["items"])


def select_job_context_from_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    memory_entries = memory.get("job_context")
    if not isinstance(memory_entries, list) or not memory_entries:
        return {}
    normalized: list[dict[str, Any]] = []
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

    def score(entry: dict[str, Any]) -> tuple[int, int, str]:
        resume = entry.get("candidate_resume")
        resume_len = len(resume.strip()) if isinstance(resume, str) else 0
        job_desc = entry.get("job_description")
        job_desc_len = len(job_desc.strip()) if isinstance(job_desc, str) else 0
        updated_at = str(entry.get("_memory_updated_at") or "")
        return (resume_len, job_desc_len, updated_at)

    best = max(normalized, key=score)
    return {k: v for k, v in best.items() if isinstance(k, str) and not k.startswith("_")}


def merge_resume_job_context(job: dict[str, Any], memory_context: dict[str, Any]) -> dict[str, Any]:
    context_json = job.get("context_json")
    if not isinstance(context_json, dict):
        context_json = {}
    merged_context = dict(context_json)
    merged_context.update(memory_context)
    for key in _RESUME_CONTEXT_KEYS:
        if is_missing_value(merged_context.get(key)) and isinstance(job.get(key), str):
            merged_context[key] = job.get(key)
    return merged_context


def build_resume_job_payload(job: dict[str, Any] | None, memory: Any) -> dict[str, Any]:
    job_payload: dict[str, Any] = dict(job) if isinstance(job, dict) else {}
    memory_context = select_job_context_from_memory(memory)
    merged_context = merge_resume_job_context(job_payload, memory_context)
    if merged_context:
        job_payload["context_json"] = merged_context
        for key in _RESUME_CONTEXT_KEYS:
            value = merged_context.get(key)
            if isinstance(value, str) and value.strip():
                job_payload[key] = value
    return job_payload


def llm_generate_resume_doc_spec_from_text(
    payload: dict[str, Any], provider: LLMProvider
) -> dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    tailored_resume = payload.get("tailored_resume")
    job = payload.get("job")
    target_pages = resolve_target_pages(payload, job)
    if isinstance(tailored_text, str) and tailored_text.strip() and tailored_resume is None:
        try:
            parsed = json.loads(tailored_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tailored_resume = parsed
    if isinstance(tailored_resume, dict):
        ensure_required_resume_sections(tailored_resume)
        prompt = prompts.resume_doc_spec_prompt(job or {}, tailored_resume=tailored_resume)
    else:
        if not isinstance(tailored_text, str) or not tailored_text.strip():
            raise ToolExecutionError("tailored_text must be a non-empty string")
        ensure_required_resume_sections(tailored_text)
        prompt = prompts.resume_doc_spec_from_text_prompt(tailored_text, job=job)
    response = provider.generate(prompt)
    json_text = extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_doc_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_doc_spec, dict):
        raise ToolExecutionError("ResumeDocSpec must be an object")
    apply_resume_target_pages_policy(resume_doc_spec, target_pages)
    normalize_skills_definition_separators(resume_doc_spec)
    ensure_certifications_section_content(
        resume_doc_spec,
        tailored_resume=tailored_resume if isinstance(tailored_resume, dict) else None,
        tailored_text=tailored_text if isinstance(tailored_text, str) else None,
        candidate_resume_text=extract_candidate_resume_text_from_job(job),
    )
    if isinstance(tailored_text, str) and tailored_text.strip():
        fill_missing_dates_from_text(resume_doc_spec, tailored_text)

    from libs.tools.resume_doc_spec_validate import _resume_doc_spec_validate

    validation = _resume_doc_spec_validate({"resume_doc_spec": resume_doc_spec, "strict": True})
    if not validation.get("valid", False):
        errors = validation.get("errors") or []
        raise ToolExecutionError(f"resume_doc_spec_validation_failed:{errors}")
    return {"resume_doc_spec": resume_doc_spec, "validation": validation}


def normalize_skills_definition_separators(resume_doc_spec: dict[str, Any]) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    in_skills_section = False
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "section_heading":
            heading = block.get("text")
            heading_text = heading.strip().upper() if isinstance(heading, str) else ""
            in_skills_section = heading_text in {
                "SKILLS",
                "CORE SKILLS",
                "TECHNICAL SKILLS",
            }
            continue
        if not in_skills_section or block_type != "definition_list":
            continue

        items = block.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            definition = item.get("definition")
            if not isinstance(definition, str) or ";" not in definition:
                continue
            normalized_items = [part.strip() for part in definition.split(";") if part.strip()]
            item["definition"] = ", ".join(normalized_items)


def llm_generate_cover_letter_from_resume(
    payload: dict[str, Any], provider: LLMProvider
) -> dict[str, Any]:
    tailored_text = payload.get("tailored_text")
    tailored_resume = payload.get("tailored_resume")
    if isinstance(tailored_text, str) and tailored_text.strip() and tailored_resume is None:
        try:
            parsed = json.loads(tailored_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            tailored_resume = parsed

    resume_source: dict[str, Any]
    header: dict[str, Any] = {}
    if isinstance(tailored_resume, dict):
        ensure_required_resume_sections(tailored_resume)
        resume_source = tailored_resume
        header_value = tailored_resume.get("header")
        if isinstance(header_value, dict):
            header = header_value
    else:
        if not isinstance(tailored_text, str) or not tailored_text.strip():
            raise ToolExecutionError("tailored_resume or tailored_text must be provided")
        resume_source = {"raw_text": tailored_text}

    job_value = payload.get("job")
    job_payload = build_resume_job_payload(
        job_value if isinstance(job_value, dict) else {},
        payload.get("memory"),
    )
    context_json = job_payload.get("context_json", {})
    if not isinstance(context_json, dict):
        context_json = {}

    prompt = prompts.cover_letter_from_resume_prompt(resume_source, job=job_payload)
    response = provider.generate(prompt)
    json_text = extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        cover_letter = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(cover_letter, dict):
        raise ToolExecutionError("cover_letter must be an object")

    header_links = header.get("links")
    if not isinstance(header_links, dict):
        header_links = {}

    job_description = context_json.get("job_description")
    if not isinstance(job_description, str):
        job_description = ""
    candidate_resume = context_json.get("candidate_resume")
    if not isinstance(candidate_resume, str):
        candidate_resume = ""

    if is_missing_value(cover_letter.get("full_name")):
        full_name = ""
        if isinstance(header.get("name"), str) and header.get("name", "").strip():
            full_name = header["name"].strip()
        if not full_name:
            full_name = derive_candidate_name_from_texts(
                tailored_text, candidate_resume, job_description
            )
        if full_name:
            cover_letter["full_name"] = full_name

    if is_missing_value(cover_letter.get("location")) and isinstance(header.get("location"), str):
        cover_letter["location"] = header.get("location", "").strip()
    if is_missing_value(cover_letter.get("phone")) and isinstance(header.get("phone"), str):
        cover_letter["phone"] = header.get("phone", "").strip()
    if is_missing_value(cover_letter.get("email")) and isinstance(header.get("email"), str):
        cover_letter["email"] = header.get("email", "").strip()

    if is_missing_value(cover_letter.get("linkedin_url")):
        linkedin = header_links.get("linkedin")
        if isinstance(linkedin, str) and linkedin.strip():
            cover_letter["linkedin_url"] = linkedin.strip()
    if is_missing_value(cover_letter.get("github_url")):
        github = header_links.get("github")
        if isinstance(github, str) and github.strip():
            cover_letter["github_url"] = github.strip()

    if is_missing_value(cover_letter.get("role_title")):
        role_hint = (
            context_json.get("target_role_name")
            or job_payload.get("target_role_name")
            or derive_role_name_from_jd(job_description)
        )
        if isinstance(role_hint, str) and role_hint.strip():
            cover_letter["role_title"] = role_hint.strip()

    if is_missing_value(cover_letter.get("company")):
        company_hint = context_json.get("company_name")
        if not isinstance(company_hint, str) or not company_hint.strip():
            company_hint = derive_company_name_from_jd(job_description)
        if isinstance(company_hint, str) and company_hint.strip():
            cover_letter["company"] = company_hint.strip()

    if is_missing_value(cover_letter.get("recipient_line")):
        cover_letter["recipient_line"] = "Hiring Team"
    if is_missing_value(cover_letter.get("salutation")):
        cover_letter["salutation"] = "Dear Hiring Team,"
    if is_missing_value(cover_letter.get("closing")):
        cover_letter["closing"] = "Sincerely,"

    if is_missing_value(cover_letter.get("full_name")):
        raise ToolExecutionError("cover_letter_missing_full_name")
    if is_missing_value(cover_letter.get("body")):
        raise ToolExecutionError("cover_letter_missing_body")

    return {"cover_letter": cover_letter}


def llm_generate_coverletter_doc_spec_from_text(
    payload: dict[str, Any], provider: LLMProvider
) -> dict[str, Any]:
    cover_letter_result = llm_generate_cover_letter_from_resume(payload, provider)
    cover_letter = cover_letter_result.get("cover_letter")
    if not isinstance(cover_letter, dict):
        raise ToolExecutionError("cover_letter_generation_failed")

    job_value = payload.get("job")
    job_payload = build_resume_job_payload(
        job_value if isinstance(job_value, dict) else {},
        payload.get("memory"),
    )
    context_json = job_payload.get("context_json")
    if not isinstance(context_json, dict):
        context_json = {}

    date_text = payload.get("today_pretty")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = payload.get("today")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = context_json.get("today_pretty")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = context_json.get("today")
    if not isinstance(date_text, str) or not date_text.strip():
        date_text = "Today"

    coverletter_doc_spec = build_coverletter_doc_spec(cover_letter, date_text.strip())
    validation = validate_coverletter_doc_spec(coverletter_doc_spec, strict=True)
    if not validation.get("valid", False):
        errors = validation.get("errors") or []
        raise ToolExecutionError(f"coverletter_doc_spec_validation_failed:{errors}")

    return {"coverletter_doc_spec": coverletter_doc_spec, "validation": validation}


def build_coverletter_doc_spec(cover_letter: dict[str, Any], date_text: str) -> dict[str, Any]:
    full_name = str(cover_letter.get("full_name") or "").strip()
    location = str(cover_letter.get("location") or "").strip()
    phone = str(cover_letter.get("phone") or "").strip()
    email = str(cover_letter.get("email") or "").strip()
    linkedin_url = str(cover_letter.get("linkedin_url") or "").strip()
    github_url = str(cover_letter.get("github_url") or "").strip()
    portfolio_url = str(cover_letter.get("portfolio_url") or "").strip()
    company = str(cover_letter.get("company") or "").strip()
    hiring_manager = str(cover_letter.get("hiring_manager") or "").strip()
    recipient_line = str(cover_letter.get("recipient_line") or "Hiring Team").strip()
    role_title = str(cover_letter.get("role_title") or "").strip()
    salutation = str(cover_letter.get("salutation") or "Dear Hiring Team,").strip()
    closing = str(cover_letter.get("closing") or "Sincerely,").strip()
    body_text = str(cover_letter.get("body") or "").strip()

    header_blocks: list[dict[str, Any]] = []
    if full_name:
        header_blocks.append({"type": "text", "style": "name", "text": full_name})
    if location:
        header_blocks.append({"type": "text", "style": "contact", "text": location})
    phone_email = " | ".join([part for part in (phone, email) if part])
    if phone_email:
        header_blocks.append({"type": "text", "style": "contact", "text": phone_email})
    links_parts: list[str] = []
    if linkedin_url:
        links_parts.append(f"LinkedIn: {linkedin_url}")
    if github_url:
        links_parts.append(f"GitHub: {github_url}")
    if portfolio_url:
        links_parts.append(f"Portfolio: {portfolio_url}")
    links_line = " | ".join(links_parts)
    if links_line:
        header_blocks.append({"type": "text", "style": "contact", "text": links_line})

    content: list[dict[str, Any]] = []
    content.append({"type": "header", "align": "left", "blocks": header_blocks})
    content.append({"type": "paragraph", "style": "cover_letter_date", "text": date_text})
    if recipient_line:
        content.append(
            {"type": "paragraph", "style": "cover_letter_recipient", "text": recipient_line}
        )
    if hiring_manager:
        content.append(
            {"type": "paragraph", "style": "cover_letter_recipient", "text": hiring_manager}
        )
    if company:
        content.append({"type": "paragraph", "style": "cover_letter_recipient", "text": company})
    if salutation:
        content.append(
            {"type": "paragraph", "style": "cover_letter_salutation", "text": salutation}
        )

    body_paragraphs = split_cover_letter_paragraphs(body_text)
    for paragraph in body_paragraphs:
        content.append({"type": "paragraph", "style": "cover_letter_body", "text": paragraph})

    if closing:
        content.append({"type": "paragraph", "style": "cover_letter_closing", "text": closing})
    if full_name:
        content.append({"type": "paragraph", "style": "cover_letter_signature", "text": full_name})

    title_parts = [full_name or "Candidate", "Cover Letter"]
    if role_title:
        title_parts.append(role_title)
    if company:
        title_parts.append(company)
    title = " - ".join(title_parts)

    return {
        "schema_version": "1.0",
        "doc_type": "cover_letter",
        "title": title,
        "page": {
            "size": "LETTER",
            "margins_in": {"top": 0.8, "right": 0.8, "bottom": 0.8, "left": 0.8},
        },
        "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.15},
        "content": content,
        "styles": {
            "name": {"bold": True, "size_pt": 20},
            "contact": {"size_pt": 11},
            "cover_letter_date": {"space_before_pt": 10, "space_after_pt": 8},
            "cover_letter_recipient": {"space_after_pt": 2},
            "cover_letter_salutation": {"space_before_pt": 8, "space_after_pt": 8},
            "cover_letter_body": {"space_after_pt": 8},
            "cover_letter_closing": {"space_before_pt": 8, "space_after_pt": 2},
        },
    }


def split_cover_letter_paragraphs(body_text: str) -> list[str]:
    if not isinstance(body_text, str):
        return []
    text = body_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def validate_coverletter_doc_spec(spec: dict[str, Any], strict: bool) -> dict[str, Any]:
    from libs.tools.resume_doc_spec_validate import _require_number, _require_str, _result, err

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    _require_str(spec, "schema_version", "/schema_version", errors)
    _require_str(spec, "doc_type", "/doc_type", errors)
    _require_str(spec, "title", "/title", errors)

    page = spec.get("page")
    if not isinstance(page, dict):
        errors.append(err("/page", "page must be an object"))
    else:
        _require_str(page, "size", "/page/size", errors)
        margins = page.get("margins_in")
        if not isinstance(margins, dict):
            errors.append(err("/page/margins_in", "margins_in must be an object"))
        else:
            for key in ("top", "right", "bottom", "left"):
                value = margins.get(key)
                if not isinstance(value, (int, float)):
                    errors.append(err(f"/page/margins_in/{key}", "must be a number"))

    defaults = spec.get("defaults")
    if not isinstance(defaults, dict):
        errors.append(err("/defaults", "defaults must be an object"))
    else:
        _require_str(defaults, "font_family", "/defaults/font_family", errors)
        _require_number(defaults, "font_size_pt", "/defaults/font_size_pt", errors)
        _require_number(defaults, "line_spacing", "/defaults/line_spacing", errors)

    styles = spec.get("styles")
    if not isinstance(styles, dict):
        errors.append(err("/styles", "styles must be an object"))

    content = spec.get("content")
    if not isinstance(content, list):
        errors.append(err("/content", "content must be an array"))
        content = []

    allowed_types = {"header", "paragraph"}
    for idx, block in enumerate(content):
        path = f"/content/{idx}"
        if not isinstance(block, dict):
            errors.append(err(path, "content item must be an object"))
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str):
            errors.append(err(f"{path}/type", "type must be a string"))
            continue
        if block_type not in allowed_types:
            message = f"unsupported content type: {block_type}"
            if strict:
                errors.append(err(f"{path}/type", message))
            else:
                warnings.append(err(f"{path}/type", message))
            continue
        if block_type == "header":
            blocks = block.get("blocks")
            if not isinstance(blocks, list):
                errors.append(err(f"{path}/blocks", "blocks must be an array"))
                continue
            if not blocks:
                errors.append(err(f"{path}/blocks", "blocks must not be empty"))
                continue
            for b_idx, b in enumerate(blocks):
                b_path = f"{path}/blocks/{b_idx}"
                if not isinstance(b, dict):
                    errors.append(err(b_path, "block must be an object"))
                    continue
                _require_str(b, "type", f"{b_path}/type", errors)
                _require_str(b, "text", f"{b_path}/text", errors)
        elif block_type == "paragraph":
            _require_str(block, "text", f"{path}/text", errors)

    return _result(len(errors) == 0, errors, warnings, len(content))


def llm_generate_resume_doc_spec(payload: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    job = payload.get("job")
    tailored_resume = payload.get("tailored_resume")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    target_pages = resolve_target_pages(payload, job)
    prompt = prompts.resume_doc_spec_prompt(job, tailored_resume=tailored_resume)
    response = provider.generate(prompt)
    json_text = extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_doc_spec = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_doc_spec, dict):
        raise ToolExecutionError("ResumeDocSpec must be an object")
    apply_resume_target_pages_policy(resume_doc_spec, target_pages)
    ensure_certifications_section_content(
        resume_doc_spec,
        tailored_resume=tailored_resume if isinstance(tailored_resume, dict) else None,
        tailored_text=None,
        candidate_resume_text=extract_candidate_resume_text_from_job(job),
    )
    return {"resume_doc_spec": resume_doc_spec}


def ensure_required_resume_sections(content: Any) -> None:
    if isinstance(content, dict):
        required_keys = [
            "schema_version",
            "header",
            "summary",
            "skills",
            "experience",
            "education",
            "certifications",
        ]
        missing = [key for key in required_keys if key not in content]
        if missing:
            raise ToolExecutionError(f"tailored_resume_missing_fields:{','.join(missing)}")
        schema_version = content.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise ToolExecutionError("tailored_resume_invalid_schema_version")
        if schema_version.strip() != "1.0":
            raise ToolExecutionError("tailored_resume_invalid_schema_version")
        header = content.get("header")
        if not isinstance(header, dict):
            raise ToolExecutionError("tailored_resume_invalid_header")
        for field in ("name", "title", "location", "phone", "email"):
            value = header.get(field)
            if is_missing_value(value):
                raise ToolExecutionError(f"tailored_resume_missing_header:{field}")
        links = header.get("links")
        if links is not None and not isinstance(links, dict):
            raise ToolExecutionError("tailored_resume_invalid_header:links")
        if not isinstance(content.get("summary"), str) or not content["summary"].strip():
            raise ToolExecutionError("tailored_resume_invalid_summary")
        for list_key in ("skills", "experience", "education", "certifications"):
            if not isinstance(content.get(list_key), list):
                raise ToolExecutionError(f"tailored_resume_invalid_{list_key}")
        for skill in content.get("skills", []):
            if not isinstance(skill, dict):
                raise ToolExecutionError("tailored_resume_invalid_skills")
            if is_missing_value(skill.get("group_name")):
                raise ToolExecutionError("tailored_resume_invalid_skills")
            items = skill.get("items")
            if not isinstance(items, list) or not items:
                raise ToolExecutionError("tailored_resume_invalid_skills")
            for item in items:
                if is_missing_value(item):
                    raise ToolExecutionError("tailored_resume_invalid_skills")
        experiences = content.get("experience", [])
        if not experiences:
            raise ToolExecutionError("tailored_resume_missing_experience")
        for idx, role in enumerate(experiences):
            if not isinstance(role, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}")
            for field in ("company", "title", "location", "dates"):
                if is_missing_value(role.get(field)):
                    raise ToolExecutionError(f"tailored_resume_missing_experience:{idx}.{field}")
            bullets = role.get("bullets")
            groups = role.get("groups")
            has_bullets = isinstance(bullets, list) and len(bullets) > 0
            has_groups = isinstance(groups, list) and len(groups) > 0
            if not (has_bullets or has_groups):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.bullets")
            if bullets is not None and not isinstance(bullets, list):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.bullets")
            if isinstance(bullets, list):
                for bullet in bullets:
                    if is_missing_value(bullet):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.bullets"
                        )
            if groups is not None and not isinstance(groups, list):
                raise ToolExecutionError(f"tailored_resume_invalid_experience:{idx}.groups")
            if isinstance(groups, list):
                for g_idx, group in enumerate(groups):
                    if not isinstance(group, dict):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}]"
                        )
                    if is_missing_value(group.get("heading")):
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].heading"
                        )
                    group_bullets = group.get("bullets")
                    if not isinstance(group_bullets, list) or not group_bullets:
                        raise ToolExecutionError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                        )
                    for bullet in group_bullets:
                        if is_missing_value(bullet):
                            raise ToolExecutionError(
                                f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                            )
        education = content.get("education", [])
        for idx, edu in enumerate(education):
            if not isinstance(edu, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_education:{idx}")
            for field in ("degree", "school", "location", "dates"):
                if is_missing_value(edu.get(field)):
                    raise ToolExecutionError(f"tailored_resume_missing_education:{idx}.{field}")
        certifications = content.get("certifications", [])
        for idx, cert in enumerate(certifications):
            if not isinstance(cert, dict):
                raise ToolExecutionError(f"tailored_resume_invalid_certifications:{idx}")
            for field in ("name", "issuer", "year"):
                value = cert.get(field)
                if field == "year" and isinstance(value, int):
                    continue
                if is_missing_value(value):
                    raise ToolExecutionError(
                        f"tailored_resume_missing_certifications:{idx}.{field}"
                    )
        return
    if not isinstance(content, str):
        raise ToolExecutionError("tailored_resume_invalid_type")
    required = [
        "SUMMARY",
        "SKILLS",
        "EXPERIENCE",
        "EDUCATION",
        "CERTIFICATIONS",
    ]
    missing = [section for section in required if section not in content]
    if not missing:
        return
    fallback_headings = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "CERTIFICATIONS"]
    if has_required_headings(content, fallback_headings):
        return
    raise ToolExecutionError(f"tailored_text_missing_sections:{','.join(missing)}")


def has_required_headings(text: str, headings: list[str]) -> bool:
    for heading in headings:
        pattern = rf"(?m)^\\s*{re.escape(heading)}\\s*$"
        if not re.search(pattern, text):
            return False
    return True


def fill_missing_dates_from_text(resume_doc_spec: dict[str, Any], tailored_text: str) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return
    experience_dates = extract_dates_from_section(
        tailored_text, start_heading="EXPERIENCE", end_heading="EDUCATION"
    )
    education_dates = extract_dates_from_section(
        tailored_text, start_heading="EDUCATION", end_heading="CERTIFICATIONS"
    )
    exp_iter = iter(experience_dates)
    edu_iter = iter(education_dates)
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "role":
            dates = block.get("dates")
            if not isinstance(dates, str) or not dates.strip():
                next_date = next(exp_iter, None)
                if next_date:
                    block["dates"] = next_date
        if block_type == "education":
            dates = block.get("dates")
            if not isinstance(dates, str) or not dates.strip():
                next_date = next(edu_iter, None)
                if next_date:
                    block["dates"] = next_date


def ensure_certifications_section_content(
    resume_doc_spec: dict[str, Any],
    tailored_resume: dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> None:
    content = resume_doc_spec.get("content")
    if not isinstance(content, list):
        return

    cert_heading_idx = find_section_heading_block_index(
        content, {"CERTIFICATIONS", "CERTIFICATION"}
    )
    if cert_heading_idx is None:
        return

    cert_start = cert_heading_idx + 1
    cert_end = find_next_section_heading_block_index(content, cert_start)
    cert_section = content[cert_start:cert_end]
    existing_items = extract_non_empty_cert_items(cert_section)
    if existing_items:
        return

    fallback_items = collect_fallback_certification_lines(
        tailored_resume=tailored_resume,
        tailored_text=tailored_text,
        candidate_resume_text=candidate_resume_text,
    )
    if fallback_items:
        for block in cert_section:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "bullets":
                continue
            items = block.get("items")
            if isinstance(items, list):
                block["items"] = fallback_items
                return
        content.insert(cert_start, {"type": "bullets", "items": fallback_items})
        return

    if cert_section_is_removable(cert_section):
        del content[cert_heading_idx:cert_end]


def find_section_heading_block_index(
    content: list[dict[str, Any]],
    headings: set[str],
    start: int = 0,
) -> int | None:
    for idx in range(start, len(content)):
        block = content[idx]
        if not isinstance(block, dict):
            continue
        if block.get("type") != "section_heading":
            continue
        text = block.get("text")
        if not isinstance(text, str):
            continue
        if text.strip().upper() in headings:
            return idx
    return None


def find_next_section_heading_block_index(content: list[dict[str, Any]], start: int) -> int:
    for idx in range(start, len(content)):
        block = content[idx]
        if isinstance(block, dict) and block.get("type") == "section_heading":
            return idx
    return len(content)


def extract_non_empty_cert_items(blocks: list[Any]) -> list[str]:
    items: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "bullets":
            values = block.get("items")
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str) and value.strip():
                    items.append(value.strip())
        elif block_type == "paragraph":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                items.append(text.strip())
    return items


def collect_fallback_certification_lines(
    tailored_resume: dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> list[str]:
    collected: list[str] = []
    if isinstance(tailored_resume, dict):
        collected.extend(certification_lines_from_tailored_resume(tailored_resume))
    if not collected and isinstance(tailored_text, str) and tailored_text.strip():
        collected.extend(certification_lines_from_text(tailored_text))
    if not collected and isinstance(candidate_resume_text, str) and candidate_resume_text.strip():
        collected.extend(certification_lines_from_text(candidate_resume_text))

    deduped: list[str] = []
    seen: set[str] = set()
    for line in collected:
        normalized = line.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def certification_lines_from_tailored_resume(tailored_resume: dict[str, Any]) -> list[str]:
    certifications = tailored_resume.get("certifications")
    if not isinstance(certifications, list):
        return []

    lines: list[str] = []
    for cert in certifications:
        if isinstance(cert, str):
            text = cert.strip()
            if text:
                lines.append(text)
            continue
        if not isinstance(cert, dict):
            continue

        name = cert.get("name")
        issuer = cert.get("issuer")
        year = cert.get("year")
        url = (
            cert.get("url")
            or cert.get("credential_url")
            or cert.get("public_url")
            or cert.get("link")
        )
        if not isinstance(name, str) or not name.strip():
            continue

        line = name.strip()
        if isinstance(issuer, str) and issuer.strip():
            line = f"{line} - {issuer.strip()}"
        if isinstance(year, int):
            line = f"{line} ({year})"
        elif isinstance(year, str) and year.strip():
            line = f"{line} ({year.strip()})"
        if isinstance(url, str) and url.strip():
            line = f"{line} | {url.strip()}"
        lines.append(line)

    return lines


def certification_lines_from_text(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = find_heading_index(lines, "CERTIFICATIONS")
    if start_idx is None:
        start_idx = find_heading_index(lines, "CERTIFICATION")
    if start_idx is None:
        return []

    end_headings = {
        "OPEN SOURCE",
        "OPEN SOURCE (SELECTED)",
        "PROJECTS",
        "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "SUMMARY",
        "SKILLS",
        "CORE SKILLS",
    }

    cert_lines: list[str] = []
    for raw in lines[start_idx + 1 :]:
        candidate = raw.strip()
        if not candidate:
            continue
        if candidate.upper() in end_headings:
            break
        normalized = re.sub(r"^[*•\-]\s*", "", candidate).strip()
        if normalized:
            cert_lines.append(normalized)
    return cert_lines


def cert_section_is_removable(blocks: list[Any]) -> bool:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "bullets":
            items = block.get("items")
            if isinstance(items, list) and any(
                isinstance(item, str) and item.strip() for item in items
            ):
                return False
            continue
        if block_type == "paragraph":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return False
            continue
        return False
    return True


def extract_candidate_resume_text_from_job(job_payload: Any) -> str | None:
    if not isinstance(job_payload, dict):
        return None
    direct = job_payload.get("candidate_resume")
    if isinstance(direct, str) and direct.strip():
        return direct
    context = job_payload.get("context_json")
    if not isinstance(context, dict):
        return None
    candidate_resume = context.get("candidate_resume")
    if isinstance(candidate_resume, str) and candidate_resume.strip():
        return candidate_resume
    return None


def extract_dates_from_section(text: str, start_heading: str, end_heading: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    start_idx = find_heading_index(lines, start_heading)
    if start_idx is None:
        return []
    end_idx = find_heading_index(lines, end_heading, start=start_idx + 1)
    section = lines[start_idx + 1 : end_idx] if end_idx is not None else lines[start_idx + 1 :]
    dates: list[str] = []
    for line in section:
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        candidate = parts[-1]
        if not re.search(r"\\d{4}", candidate):
            continue
        dates.append(candidate)
    return dates


def find_heading_index(lines: list[str], heading: str, start: int = 0) -> int | None:
    target = heading.strip().lower()
    for idx in range(start, len(lines)):
        if lines[idx].strip().lower() == target:
            return idx
    return None


def is_missing_value(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in {"unknown", "n/a", "na", "none"}:
        return True
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return True
    return False


def llm_generate_tailored_resume_content(
    payload: dict[str, Any], provider: LLMProvider
) -> dict[str, Any]:
    job = payload.get("job")
    if not isinstance(job, dict):
        raise ToolExecutionError("job must be an object")
    prompt = prompts.tailored_resume_content_prompt(job)
    response = provider.generate(prompt)
    json_text = extract_json(response.content)
    if not json_text:
        raise ToolExecutionError("Failed to extract JSON from LLM response")
    try:
        resume_content = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"Invalid JSON returned: {exc}") from exc
    if not isinstance(resume_content, dict):
        raise ToolExecutionError("resume_content must be an object")
    return {"resume_content": resume_content}


def _normalized_non_empty_lines(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [line.strip() for line in text.splitlines() if isinstance(line, str) and line.strip()]


def derive_role_name_from_jd(job_description: str) -> str:
    lines = _normalized_non_empty_lines(job_description)
    if not lines:
        return ""

    skip_tokens = (
        "about",
        "overview",
        "summary",
        "responsibilities",
        "qualifications",
        "requirements",
        "preferred",
        "benefits",
        "location",
        "company",
        "team",
        "compensation",
        "salary",
        "who we are",
        "what you'll do",
        "what you will do",
    )
    noise_suffixes = (
        " at",
        " - remote",
        " (remote",
        " - hybrid",
        " (hybrid",
        " - onsite",
        " (onsite",
    )

    for line in lines[:20]:
        candidate = line.strip(" -*•|:")
        lowered = candidate.lower()
        if not candidate or len(candidate) < 3:
            continue
        if any(token in lowered for token in skip_tokens):
            continue
        if any(suffix in lowered for suffix in noise_suffixes):
            candidate = candidate.split(" - ", 1)[0].strip()
        candidate = re.sub(r"\s+", " ", candidate).strip(" .,-")
        if not candidate:
            continue
        if len(candidate.split()) > 10:
            continue
        return candidate

    return ""


def derive_company_name_from_jd(job_description: str) -> str:
    text = job_description if isinstance(job_description, str) else ""
    if not text.strip():
        return ""

    patterns = [
        r"\bat\s+([A-Z][A-Za-z0-9&.,'\- ]{1,60})",
        r"\bjoin\s+([A-Z][A-Za-z0-9&.,'\- ]{1,60})",
        r"\b([A-Z][A-Za-z0-9&.,'\- ]{1,60})\s+is\s+(?:a|an|the)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
        if len(candidate.split()) > 8:
            continue
        return candidate
    return ""


def derive_candidate_name_from_texts(*texts: Any) -> str:
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        lines = _normalized_non_empty_lines(text)
        for raw in lines[:25]:
            line = raw.strip()
            lowered = line.lower()
            if any(
                token in lowered
                for token in (
                    "summary",
                    "experience",
                    "education",
                    "skills",
                    "certification",
                    "objective",
                    "linkedin",
                    "github",
                    "http",
                    "@",
                )
            ):
                continue
            cleaned = re.sub(r"[^A-Za-z' -]", " ", line)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            parts = cleaned.split()
            if len(parts) < 2 or len(parts) > 4:
                continue
            if any(len(part) == 1 for part in parts):
                continue
            if not all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", part) for part in parts):
                continue
            return " ".join(part.capitalize() for part in parts)
    return ""


def extract_json(text: str) -> str:
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
