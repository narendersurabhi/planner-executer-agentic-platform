from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    namespace: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    filters: dict[str, Any] | None = None
    min_score: float | None = None
    include_text: bool = True
    include_metadata: bool = True


class RetrieveMatch(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    source_uri: str


class RetrieveResponse(BaseModel):
    matches: list[RetrieveMatch]


class EnsureCollectionRequest(BaseModel):
    collection_name: str | None = None
    vector_size: int | None = Field(default=None, ge=1)
    distance: str | None = None
    vector_name: str | None = None
    create_payload_indexes: bool = True
    payload_keyword_fields: list[str] | None = None
    on_disk_payload: bool = True


class EnsureCollectionResponse(BaseModel):
    collection_name: str
    status: str
    vector_size: int
    distance: str
    vector_name: str | None = None
    payload_keyword_fields: list[str]


class UpsertTextEntry(BaseModel):
    chunk_id: str | None = None
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_uri: str | None = None
    namespace: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertTextsRequest(BaseModel):
    collection_name: str | None = None
    ensure_collection: bool = True
    namespace: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    entries: list[UpsertTextEntry] = Field(min_length=1, max_length=100)


class UpsertTextsResponse(BaseModel):
    collection_name: str
    upserted_count: int
    chunk_ids: list[str]


class IndexWorkspaceFileRequest(BaseModel):
    path: str = Field(min_length=1)
    collection_name: str | None = None
    ensure_collection: bool = True
    document_id: str | None = None
    source_uri: str | None = None
    namespace: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    chunk_size_chars: int | None = Field(default=None, ge=200, le=20000)
    chunk_overlap_chars: int | None = Field(default=None, ge=0, le=5000)
    max_chunks: int | None = Field(default=None, ge=1, le=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexWorkspaceFileResponse(BaseModel):
    collection_name: str
    path: str
    document_id: str
    chunk_count: int
    upserted_count: int
    chunk_ids: list[str]
