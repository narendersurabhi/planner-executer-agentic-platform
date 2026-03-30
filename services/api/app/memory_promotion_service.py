from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any, Mapping

from libs.core import memory_redaction, models

_OUTPUT_FORMAT_RE = re.compile(
    r"\b(?:i prefer|prefer|default to|use)\s+(docx|pdf|markdown|md|json|yaml|csv|txt)\b",
    re.IGNORECASE,
)
_VERBOSITY_RE = re.compile(
    r"\b(?:i prefer|prefer|keep (?:it|responses|answers|output)|make (?:it|responses|answers|output))\s+"
    r"(concise|brief|short|detailed|verbose)\b",
    re.IGNORECASE,
)
_VERBOSITY_FALLBACK_RE = re.compile(
    r"\b(concise|brief|short|detailed|verbose)\s+(?:responses|answers|output)\b",
    re.IGNORECASE,
)
_FORMAT_ALIASES = {"md": "markdown", "txt": "text"}
_VERBOSITY_ALIASES = {"brief": "concise", "short": "concise", "verbose": "detailed"}


def extract_user_profile_decisions(content: str) -> list[models.MemoryPromotionDecision]:
    text = str(content or "").strip()
    if not text:
        return []

    decisions: list[models.MemoryPromotionDecision] = []
    format_match = _OUTPUT_FORMAT_RE.search(text)
    if format_match:
        raw_value = format_match.group(1).strip().lower()
        format_value = _FORMAT_ALIASES.get(raw_value, raw_value)
        decisions.append(
            models.MemoryPromotionDecision(
                candidate_type=models.MemoryCandidateType.user_profile_update,
                accepted=True,
                reason="explicit_output_format_preference",
                payload={"preferences": {"preferred_output_format": format_value}},
            )
        )

    verbosity_match = _VERBOSITY_RE.search(text) or _VERBOSITY_FALLBACK_RE.search(text)
    if verbosity_match:
        raw_value = verbosity_match.group(1).strip().lower()
        verbosity_value = _VERBOSITY_ALIASES.get(raw_value, raw_value)
        decisions.append(
            models.MemoryPromotionDecision(
                candidate_type=models.MemoryCandidateType.user_profile_update,
                accepted=True,
                reason="explicit_response_verbosity_preference",
                payload={"preferences": {"response_verbosity": verbosity_value}},
            )
        )

    return decisions


def merge_user_profile_payload(
    existing_payload: Mapping[str, Any] | None,
    decisions: list[models.MemoryPromotionDecision],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    merged = models.UserProfilePayload.model_validate(existing_payload or {}).model_dump(mode="json")
    preferences = (
        dict(merged.get("preferences"))
        if isinstance(merged.get("preferences"), Mapping)
        else {}
    )
    for decision in decisions:
        if not decision.accepted or decision.candidate_type != models.MemoryCandidateType.user_profile_update:
            continue
        payload = decision.payload if isinstance(decision.payload, Mapping) else {}
        payload_preferences = (
            dict(payload.get("preferences"))
            if isinstance(payload.get("preferences"), Mapping)
            else {}
        )
        for key, value in payload_preferences.items():
            if isinstance(value, str) and value.strip():
                preferences[str(key)] = value.strip()
    merged["preferences"] = preferences
    merged["updated_at"] = (now or datetime.now(UTC)).isoformat()
    return models.UserProfilePayload.model_validate(merged).model_dump(mode="json")


def build_semantic_memory_write(
    *,
    user_id: str,
    key: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    ttl_seconds: int | None = None,
) -> models.MemoryWrite:
    sanitized_payload, sensitivity, indexable = memory_redaction.sanitize_memory_payload(payload)
    merged_metadata = dict(metadata or {})
    merged_metadata["sensitive"] = sensitivity == models.MemorySensitivity.restricted
    merged_metadata["indexable"] = indexable
    return models.MemoryWrite(
        name="semantic_memory",
        scope=models.MemoryScope.user,
        user_id=user_id,
        key=key,
        payload=sanitized_payload,
        metadata=merged_metadata,
        ttl_seconds=ttl_seconds,
    )
