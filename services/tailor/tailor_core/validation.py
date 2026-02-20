from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .context import is_missing_value
from .errors import TailorError


def extract_json(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return stripped[start : end + 1]


def parse_json_object(json_text: str) -> Dict[str, Any]:
    if not json_text:
        raise TailorError("invalid_json")
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise TailorError(f"invalid_json:{exc}") from exc
    if not isinstance(payload, dict):
        raise TailorError("tailored_resume_invalid")
    return payload


def resolve_tailored_resume(
    tailored_resume: Optional[Dict[str, Any]], tailored_text: Optional[str]
) -> Dict[str, Any]:
    if isinstance(tailored_resume, dict):
        return tailored_resume
    if isinstance(tailored_text, str) and tailored_text.strip():
        try:
            parsed = json.loads(tailored_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    raise TailorError("tailored_resume must be an object")


def ensure_required_resume_sections(content: Any) -> None:
    if not isinstance(content, dict):
        raise TailorError("tailored_resume must be an object")
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
        raise TailorError(f"tailored_resume_missing_fields:{','.join(missing)}")
    schema_version = content.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise TailorError("tailored_resume_invalid_schema_version")
    if schema_version.strip() != "1.0":
        raise TailorError("tailored_resume_invalid_schema_version")
    header = content.get("header")
    if not isinstance(header, dict):
        raise TailorError("tailored_resume_invalid_header")
    for field in ("name", "title", "location", "phone", "email"):
        value = header.get(field)
        if is_missing_value(value):
            raise TailorError(f"tailored_resume_missing_header:{field}")
    links = header.get("links")
    if links is not None and not isinstance(links, dict):
        raise TailorError("tailored_resume_invalid_header:links")
    if not isinstance(content.get("summary"), str) or not content["summary"].strip():
        raise TailorError("tailored_resume_invalid_summary")
    for list_key in ("skills", "experience", "education", "certifications"):
        if not isinstance(content.get(list_key), list):
            raise TailorError(f"tailored_resume_invalid_{list_key}")
    experiences = content.get("experience", [])
    if not experiences:
        raise TailorError("tailored_resume_missing_experience")
    for idx, role in enumerate(experiences):
        if not isinstance(role, dict):
            raise TailorError(f"tailored_resume_invalid_experience:{idx}")
        for field in ("company", "title", "location", "dates"):
            if is_missing_value(role.get(field)):
                raise TailorError(f"tailored_resume_missing_experience:{idx}.{field}")
        bullets = role.get("bullets")
        groups = role.get("groups")
        has_bullets = isinstance(bullets, list) and len(bullets) > 0
        has_groups = isinstance(groups, list) and len(groups) > 0
        if not (has_bullets or has_groups):
            raise TailorError(f"tailored_resume_invalid_experience:{idx}.bullets")
        if bullets is not None and not isinstance(bullets, list):
            raise TailorError(f"tailored_resume_invalid_experience:{idx}.bullets")
        if isinstance(bullets, list):
            for bullet in bullets:
                if is_missing_value(bullet):
                    raise TailorError(f"tailored_resume_invalid_experience:{idx}.bullets")
        if groups is not None and not isinstance(groups, list):
            raise TailorError(f"tailored_resume_invalid_experience:{idx}.groups")
        if isinstance(groups, list):
            for g_idx, group in enumerate(groups):
                if not isinstance(group, dict):
                    raise TailorError(f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}]")
                if is_missing_value(group.get("heading")):
                    raise TailorError(
                        f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].heading"
                    )
                group_bullets = group.get("bullets")
                if not isinstance(group_bullets, list) or not group_bullets:
                    raise TailorError(
                        f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                    )
                for bullet in group_bullets:
                    if is_missing_value(bullet):
                        raise TailorError(
                            f"tailored_resume_invalid_experience:{idx}.groups[{g_idx}].bullets"
                        )
    education = content.get("education", [])
    for idx, edu in enumerate(education):
        if not isinstance(edu, dict):
            raise TailorError(f"tailored_resume_invalid_education:{idx}")
        for field in ("degree", "school", "location", "dates"):
            if is_missing_value(edu.get(field)):
                raise TailorError(f"tailored_resume_missing_education:{idx}.{field}")
    certifications = content.get("certifications", [])
    for idx, cert in enumerate(certifications):
        if not isinstance(cert, dict):
            raise TailorError(f"tailored_resume_invalid_certifications:{idx}")
        for field in ("name", "issuer", "year"):
            value = cert.get(field)
            if field == "year" and isinstance(value, int):
                continue
            if is_missing_value(value):
                raise TailorError(f"tailored_resume_missing_certifications:{idx}.{field}")
