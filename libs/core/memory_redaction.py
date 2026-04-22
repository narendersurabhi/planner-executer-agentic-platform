from __future__ import annotations

import re
from typing import Any, Mapping

from . import models

REDACTED = "[redacted]"
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z\-_]{20,})\b"
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passcode|secret|token|api[_-]?key|authorization|cookie|session|ssn|dob|member[_-]?id)",
    re.IGNORECASE,
)


def redact_text(value: str) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    redacted = value
    changed = False
    for pattern in (_EMAIL_RE, _PHONE_RE, _TOKEN_RE):
        next_value, count = pattern.subn(REDACTED, redacted)
        if count:
            changed = True
            redacted = next_value
    return redacted, changed


def sanitize_memory_payload(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], models.MemorySensitivity, bool]:
    sanitized, redacted = _sanitize_value(payload if isinstance(payload, Mapping) else {})
    sensitivity = models.MemorySensitivity.restricted if redacted else models.MemorySensitivity.low
    indexable = not redacted
    return sanitized if isinstance(sanitized, dict) else {}, sensitivity, indexable


def _sanitize_value(value: Any, *, key: str | None = None) -> tuple[Any, bool]:
    normalized_key = str(key or "").strip()
    if normalized_key and _SENSITIVE_KEY_RE.search(normalized_key):
        return REDACTED, True
    if isinstance(value, Mapping):
        changed = False
        sanitized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            child_sanitized, child_changed = _sanitize_value(child_value, key=str(child_key))
            sanitized[str(child_key)] = child_sanitized
            changed = changed or child_changed
        return sanitized, changed
    if isinstance(value, list):
        changed = False
        sanitized_list: list[Any] = []
        for item in value:
            item_sanitized, item_changed = _sanitize_value(item)
            sanitized_list.append(item_sanitized)
            changed = changed or item_changed
        return sanitized_list, changed
    if isinstance(value, str):
        return redact_text(value)
    return value, False
