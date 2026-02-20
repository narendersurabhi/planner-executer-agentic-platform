from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from libs.core import memory_registry, models
from .models import MemoryRecord


MEMORY_REGISTRY = memory_registry.default_memory_registry()


def write_memory(db: Session, request: models.MemoryWrite) -> models.MemoryEntry:
    spec = _resolve_spec(request.name)
    scope = request.scope or spec.scope
    if request.scope and request.scope != spec.scope:
        raise ValueError("scope_mismatch")
    _validate_scope_requirements(scope, request.job_id, request.user_id, request.project_id)
    ttl_seconds = request.ttl_seconds if request.ttl_seconds is not None else spec.ttl_seconds
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None

    record = _find_existing(
        db=db,
        name=spec.name,
        scope=scope,
        key=request.key,
        job_id=request.job_id,
        user_id=request.user_id,
        project_id=request.project_id,
    )
    if record:
        if request.if_match_updated_at is not None:
            if _normalize_dt(record.updated_at) != _normalize_dt(request.if_match_updated_at):
                raise ValueError("memory_conflict")
        record.payload = request.payload
        record.metadata_json = request.metadata or {}
        record.updated_at = now
        record.expires_at = expires_at
        record.version = spec.version
    else:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            name=spec.name,
            scope=scope.value,
            key=request.key,
            job_id=request.job_id,
            user_id=request.user_id,
            project_id=request.project_id,
            payload=request.payload,
            metadata_json=request.metadata or {},
            version=spec.version,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        db.add(record)
    db.commit()
    db.refresh(record)
    return _to_entry(record)


def read_memory(db: Session, query: models.MemoryQuery) -> list[models.MemoryEntry]:
    spec = _resolve_spec(query.name)
    scope = query.scope or spec.scope
    if query.scope and query.scope != spec.scope:
        raise ValueError("scope_mismatch")
    _validate_scope_requirements(scope, query.job_id, query.user_id, query.project_id)

    q = db.query(MemoryRecord).filter(
        MemoryRecord.name == spec.name, MemoryRecord.scope == scope.value
    )
    if query.key is not None:
        q = q.filter(MemoryRecord.key == query.key)
    if query.job_id:
        q = q.filter(MemoryRecord.job_id == query.job_id)
    if query.user_id:
        q = q.filter(MemoryRecord.user_id == query.user_id)
    if query.project_id:
        q = q.filter(MemoryRecord.project_id == query.project_id)
    if not query.include_expired:
        now = datetime.utcnow()
        q = q.filter((MemoryRecord.expires_at.is_(None)) | (MemoryRecord.expires_at > now))
    limit = max(1, min(query.limit or 50, 200))
    records = q.order_by(MemoryRecord.updated_at.desc()).limit(limit).all()
    return [_to_entry(record) for record in records]


def _resolve_spec(name: str) -> models.MemorySpec:
    if not MEMORY_REGISTRY.has(name):
        raise KeyError(f"unknown_memory:{name}")
    return MEMORY_REGISTRY.get(name)


def _validate_scope_requirements(
    scope: models.MemoryScope,
    job_id: Optional[str],
    user_id: Optional[str],
    project_id: Optional[str],
) -> None:
    if scope in {models.MemoryScope.request, models.MemoryScope.session} and not job_id:
        raise ValueError("job_id_required")
    if scope == models.MemoryScope.user and not user_id:
        raise ValueError("user_id_required")
    if scope == models.MemoryScope.project and not project_id:
        raise ValueError("project_id_required")


def _find_existing(
    db: Session,
    name: str,
    scope: models.MemoryScope,
    key: Optional[str],
    job_id: Optional[str],
    user_id: Optional[str],
    project_id: Optional[str],
) -> Optional[MemoryRecord]:
    q = db.query(MemoryRecord).filter(MemoryRecord.name == name, MemoryRecord.scope == scope.value)
    if key is None:
        q = q.filter(MemoryRecord.key.is_(None))
    else:
        q = q.filter(MemoryRecord.key == key)
    if job_id:
        q = q.filter(MemoryRecord.job_id == job_id)
    else:
        q = q.filter(MemoryRecord.job_id.is_(None))
    if user_id:
        q = q.filter(MemoryRecord.user_id == user_id)
    else:
        q = q.filter(MemoryRecord.user_id.is_(None))
    if project_id:
        q = q.filter(MemoryRecord.project_id == project_id)
    else:
        q = q.filter(MemoryRecord.project_id.is_(None))
    return q.first()


def _to_entry(record: MemoryRecord) -> models.MemoryEntry:
    return models.MemoryEntry(
        id=record.id,
        name=record.name,
        scope=models.MemoryScope(record.scope),
        payload=record.payload or {},
        key=record.key,
        job_id=record.job_id,
        user_id=record.user_id,
        project_id=record.project_id,
        metadata=record.metadata_json or {},
        version=record.version,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
    )


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)
