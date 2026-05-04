from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from libs.core import models

from . import memory_promotion_service, memory_store

USER_PROFILE_KEY = "profile"


def load_user_profile(db: Session, user_id: str) -> models.UserProfilePayload:
    entries = memory_store.read_memory(
        db,
        models.MemoryQuery(
            name="user_profile",
            scope=models.MemoryScope.user,
            user_id=user_id,
            key=USER_PROFILE_KEY,
            limit=1,
        ),
    )
    if not entries:
        return models.UserProfilePayload()
    return models.UserProfilePayload.model_validate(entries[0].payload or {})


def write_user_profile(
    db: Session,
    *,
    user_id: str,
    payload: dict[str, Any],
) -> models.MemoryEntry:
    validated = models.UserProfilePayload.model_validate(payload)
    return memory_store.write_memory(
        db,
        models.MemoryWrite(
            name="user_profile",
            scope=models.MemoryScope.user,
            user_id=user_id,
            key=USER_PROFILE_KEY,
            payload=validated.model_dump(mode="json"),
            metadata={"promotion_source": "user_profile_service"},
        ),
    )


def apply_user_profile_updates_from_text(
    db: Session,
    *,
    user_id: str,
    content: str,
) -> tuple[models.MemoryEntry | None, list[models.MemoryPromotionDecision]]:
    decisions = memory_promotion_service.extract_user_profile_decisions(content)
    if not decisions:
        return None, []
    existing = load_user_profile(db, user_id)
    merged_payload = memory_promotion_service.merge_user_profile_payload(
        existing.model_dump(mode="json"),
        decisions,
    )
    entry = write_user_profile(db, user_id=user_id, payload=merged_payload)
    return entry, decisions
