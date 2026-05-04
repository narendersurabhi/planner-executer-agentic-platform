from libs.core import memory_redaction, models


def test_sanitize_memory_payload_redacts_sensitive_keys() -> None:
    payload = {
        "subject": "user profile",
        "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
        "nested": {"member_id": "12345"},
    }

    sanitized, sensitivity, indexable = memory_redaction.sanitize_memory_payload(payload)

    assert sanitized["api_key"] == memory_redaction.REDACTED
    assert sanitized["nested"]["member_id"] == memory_redaction.REDACTED
    assert sensitivity == models.MemorySensitivity.restricted
    assert indexable is False


def test_sanitize_memory_payload_redacts_email_and_phone() -> None:
    payload = {
        "fact": "Contact me at narender@example.com or 555-222-1212.",
    }

    sanitized, sensitivity, indexable = memory_redaction.sanitize_memory_payload(payload)

    assert sanitized["fact"].count(memory_redaction.REDACTED) == 2
    assert sensitivity == models.MemorySensitivity.restricted
    assert indexable is False
