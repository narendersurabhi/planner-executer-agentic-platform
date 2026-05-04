from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import RetrieverError
from .models import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DocumentChunk,
    DocumentChunksRequest,
    DocumentChunksResponse,
    DocumentListRequest,
    DocumentListResponse,
    DocumentSummary,
    EnsureCollectionRequest,
    EnsureCollectionResponse,
    IndexMarkdownRequest,
    IndexMarkdownResponse,
    IndexWorkspaceDirectoryFileResult,
    IndexWorkspaceDirectoryRequest,
    IndexWorkspaceDirectoryResponse,
    IndexWorkspaceDirectorySkippedFile,
    IndexWorkspaceFileRequest,
    IndexWorkspaceFileResponse,
    RerankMatch,
    RerankRequest,
    RerankResponse,
    RetrieveMatch,
    RetrieveRequest,
    RetrieveResponse,
    UpsertTextEntry,
    UpsertTextsRequest,
    UpsertTextsResponse,
)


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class VectorDatabase(Protocol):
    def query(
        self,
        *,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filter_obj: dict[str, Any] | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> list[dict[str, Any]]: ...

    def get_collection(self, collection: str) -> dict[str, Any] | None: ...

    def scroll(
        self,
        *,
        collection: str,
        limit: int,
        offset: str | int | None,
        filter_obj: dict[str, Any] | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> tuple[list[dict[str, Any]], str | int | None]: ...

    def create_collection(
        self,
        *,
        collection: str,
        vector_size: int,
        distance: str,
        vector_name: str | None,
        on_disk_payload: bool,
    ) -> None: ...

    def create_payload_index(
        self,
        *,
        collection: str,
        field_name: str,
        field_schema: str = "keyword",
    ) -> None: ...

    def upsert_points(
        self,
        *,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None: ...

    def delete_points(
        self,
        *,
        collection: str,
        filter_obj: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class RetrieverServiceConfig:
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_vector_name: str | None
    qdrant_timeout_s: float
    qdrant_vector_size: int
    qdrant_distance: str
    qdrant_on_disk_payload: bool
    qdrant_create_payload_indexes: bool
    qdrant_payload_index_fields: tuple[str, ...]
    embedding_provider: str
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_timeout_s: float
    top_k_default: int
    top_k_max: int
    require_scope: bool
    workspace_dir: str
    workspace_allowed_extensions: tuple[str, ...]
    workspace_max_file_bytes: int
    workspace_chunk_size_chars: int
    workspace_chunk_overlap_chars: int
    workspace_max_chunks: int
    payload_text_key: str
    payload_document_id_key: str
    payload_source_uri_key: str
    payload_namespace_key: str
    payload_tenant_id_key: str
    payload_user_id_key: str
    payload_workspace_id_key: str

    @classmethod
    def from_env(cls) -> RetrieverServiceConfig:
        payload_text_key = os.getenv("RAG_PAYLOAD_TEXT_KEY", "text").strip() or "text"
        payload_document_id_key = (
            os.getenv("RAG_PAYLOAD_DOCUMENT_ID_KEY", "document_id").strip() or "document_id"
        )
        payload_source_uri_key = (
            os.getenv("RAG_PAYLOAD_SOURCE_URI_KEY", "source_uri").strip() or "source_uri"
        )
        payload_namespace_key = (
            os.getenv("RAG_PAYLOAD_NAMESPACE_KEY", "namespace").strip() or "namespace"
        )
        payload_tenant_id_key = (
            os.getenv("RAG_PAYLOAD_TENANT_ID_KEY", "tenant_id").strip() or "tenant_id"
        )
        payload_user_id_key = (
            os.getenv("RAG_PAYLOAD_USER_ID_KEY", "user_id").strip() or "user_id"
        )
        payload_workspace_id_key = (
            os.getenv("RAG_PAYLOAD_WORKSPACE_ID_KEY", "workspace_id").strip() or "workspace_id"
        )
        default_payload_index_fields = ",".join(
            [
                payload_document_id_key,
                payload_namespace_key,
                payload_tenant_id_key,
                payload_user_id_key,
                payload_workspace_id_key,
                payload_source_uri_key,
            ]
        )
        return cls(
            qdrant_url=os.getenv("QDRANT_URL", "").strip(),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "").strip(),
            qdrant_vector_name=_optional_str(os.getenv("QDRANT_VECTOR_NAME")),
            qdrant_timeout_s=_float_with_default(os.getenv("QDRANT_TIMEOUT_S"), 10.0),
            qdrant_vector_size=_int_with_default(
                os.getenv("QDRANT_VECTOR_SIZE"),
                _default_embedding_size(os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")),
            ),
            qdrant_distance=os.getenv("QDRANT_DISTANCE", "Cosine").strip() or "Cosine",
            qdrant_on_disk_payload=_bool_with_default(
                os.getenv("QDRANT_ON_DISK_PAYLOAD"),
                True,
            ),
            qdrant_create_payload_indexes=_bool_with_default(
                os.getenv("QDRANT_CREATE_PAYLOAD_INDEXES"),
                True,
            ),
            qdrant_payload_index_fields=tuple(
                _csv_values(
                    os.getenv("QDRANT_PAYLOAD_INDEX_FIELDS", default_payload_index_fields)
                )
            ),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "openai").strip().lower(),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            embedding_api_key=(
                os.getenv("RAG_OPENAI_API_KEY", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip()
            ),
            embedding_base_url=os.getenv(
                "RAG_OPENAI_BASE_URL",
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            ).strip(),
            embedding_timeout_s=_float_with_default(os.getenv("RAG_EMBEDDING_TIMEOUT_S"), 15.0),
            top_k_default=_int_with_default(os.getenv("RAG_TOP_K_DEFAULT"), 5),
            top_k_max=_int_with_default(os.getenv("RAG_TOP_K_MAX"), 20),
            require_scope=_bool_with_default(os.getenv("RAG_REQUIRE_SCOPE"), True),
            workspace_dir=os.getenv("WORKSPACE_DIR", "/shared/workspace").strip()
            or "/shared/workspace",
            workspace_allowed_extensions=tuple(
                _csv_values(
                    os.getenv(
                        "RAG_WORKSPACE_ALLOWED_EXTENSIONS",
                        ".md,.txt,.rst,.json,.yaml,.yml,.toml,.py,.ts,.tsx,.js,.jsx,.css,.html,.sql,.sh",
                    )
                )
            ),
            workspace_max_file_bytes=_int_with_default(
                os.getenv("RAG_WORKSPACE_MAX_FILE_BYTES"),
                2_000_000,
            ),
            workspace_chunk_size_chars=_int_with_default(
                os.getenv("RAG_WORKSPACE_CHUNK_SIZE_CHARS"),
                1200,
            ),
            workspace_chunk_overlap_chars=_int_with_default(
                os.getenv("RAG_WORKSPACE_CHUNK_OVERLAP_CHARS"),
                200,
            ),
            workspace_max_chunks=_int_with_default(
                os.getenv("RAG_WORKSPACE_MAX_CHUNKS"),
                200,
            ),
            payload_text_key=payload_text_key,
            payload_document_id_key=payload_document_id_key,
            payload_source_uri_key=payload_source_uri_key,
            payload_namespace_key=payload_namespace_key,
            payload_tenant_id_key=payload_tenant_id_key,
            payload_user_id_key=payload_user_id_key,
            payload_workspace_id_key=payload_workspace_id_key,
        )


@dataclass(frozen=True)
class _PreparedChunk:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _MarkdownSection:
    text: str
    title: str | None
    heading_level: int | None
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class OpenAIEmbeddingClient:
    api_key: str
    model: str
    base_url: str
    timeout_s: float

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RetrieverError("embedding_api_key_missing", status_code=503)
        if not self.model:
            raise RetrieverError("embedding_model_missing", status_code=503)
        payload = {"input": texts, "model": self.model}
        data = _json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/v1/embeddings",
            payload=payload,
            timeout_s=self.timeout_s,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            error_prefix="embedding_http_error",
            invalid_json_prefix="embedding_invalid_json",
        )
        entries = data.get("data")
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise RetrieverError("embedding_missing_data", status_code=502)
        vectors: list[list[float]] = []
        for entry in entries:
            vector = entry.get("embedding") if isinstance(entry, dict) else None
            if not isinstance(vector, list) or not vector:
                raise RetrieverError("embedding_missing_vector", status_code=502)
            try:
                vectors.append([float(item) for item in vector])
            except (TypeError, ValueError) as exc:
                raise RetrieverError("embedding_vector_invalid", status_code=502) from exc
        return vectors


@dataclass(frozen=True)
class QdrantClient:
    base_url: str
    api_key: str
    timeout_s: float

    def query(
        self,
        *,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filter_obj: dict[str, Any] | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> list[dict[str, Any]]:
        if not self.base_url:
            raise RetrieverError("qdrant_url_missing", status_code=503)
        if not collection:
            raise RetrieverError("qdrant_collection_missing", status_code=503)
        payload: dict[str, Any] = {
            "query": vector,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if filter_obj:
            payload["filter"] = filter_obj
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        if vector_name:
            payload["using"] = vector_name
        data = self._request(
            method="POST",
            path=f"/collections/{collection}/points/query",
            payload=payload,
        )
        result = data.get("result")
        if isinstance(result, dict):
            points = result.get("points")
            if isinstance(points, list):
                return [point for point in points if isinstance(point, dict)]
        if isinstance(result, list):
            return [point for point in result if isinstance(point, dict)]
        raise RetrieverError("qdrant_response_invalid", status_code=502)

    def get_collection(self, collection: str) -> dict[str, Any] | None:
        if not collection:
            raise RetrieverError("qdrant_collection_missing", status_code=503)
        try:
            return self._request(method="GET", path=f"/collections/{collection}")
        except RetrieverError as exc:
            if exc.status_code == 404:
                return None
            raise

    def scroll(
        self,
        *,
        collection: str,
        limit: int,
        offset: str | int | None,
        filter_obj: dict[str, Any] | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> tuple[list[dict[str, Any]], str | int | None]:
        if not self.base_url:
            raise RetrieverError("qdrant_url_missing", status_code=503)
        if not collection:
            raise RetrieverError("qdrant_collection_missing", status_code=503)
        payload: dict[str, Any] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if filter_obj:
            payload["filter"] = filter_obj
        if offset is not None:
            payload["offset"] = offset
        if vector_name:
            payload["using"] = vector_name
        data = self._request(
            method="POST",
            path=f"/collections/{collection}/points/scroll",
            payload=payload,
        )
        result = data.get("result")
        if not isinstance(result, dict):
            raise RetrieverError("qdrant_response_invalid", status_code=502)
        points = result.get("points")
        next_page_offset = result.get("next_page_offset")
        if not isinstance(points, list):
            raise RetrieverError("qdrant_response_invalid", status_code=502)
        return [point for point in points if isinstance(point, dict)], (
            next_page_offset if isinstance(next_page_offset, (str, int)) else None
        )

    def create_collection(
        self,
        *,
        collection: str,
        vector_size: int,
        distance: str,
        vector_name: str | None,
        on_disk_payload: bool,
    ) -> None:
        vectors: dict[str, Any]
        if vector_name:
            vectors = {vector_name: {"size": vector_size, "distance": distance}}
        else:
            vectors = {"size": vector_size, "distance": distance}
        self._request(
            method="PUT",
            path=f"/collections/{collection}",
            payload={
                "vectors": vectors,
                "on_disk_payload": on_disk_payload,
            },
        )

    def create_payload_index(
        self,
        *,
        collection: str,
        field_name: str,
        field_schema: str = "keyword",
    ) -> None:
        self._request(
            method="PUT",
            path=f"/collections/{collection}/index?wait=true",
            payload={"field_name": field_name, "field_schema": field_schema},
        )

    def upsert_points(
        self,
        *,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        self._request(
            method="PUT",
            path=f"/collections/{collection}/points?wait=true",
            payload={"points": points},
        )

    def delete_points(
        self,
        *,
        collection: str,
        filter_obj: dict[str, Any],
    ) -> None:
        self._request(
            method="POST",
            path=f"/collections/{collection}/points/delete?wait=true",
            payload={"filter": filter_obj},
        )

    def _request(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise RetrieverError("qdrant_url_missing", status_code=503)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        return _json_request(
            method=method,
            url=f"{self.base_url.rstrip('/')}{path}",
            payload=payload,
            timeout_s=self.timeout_s,
            headers=headers,
            error_prefix="qdrant_http_error",
            invalid_json_prefix="qdrant_invalid_json",
        )


@dataclass(frozen=True)
class RetrieverService:
    config: RetrieverServiceConfig
    embedder: TextEmbedder
    vector_db: VectorDatabase

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        query = request.query.strip()
        if not query:
            raise RetrieverError("missing_query")
        if self.config.require_scope and not any(
            [request.tenant_id, request.user_id, request.workspace_id]
        ):
            raise RetrieverError(
                "missing_scope:one_of_tenant_id_user_id_workspace_id_required"
            )
        collection = self._resolve_collection_name(None)
        top_k = request.top_k or self.config.top_k_default
        top_k = max(1, min(self.config.top_k_max, top_k))
        vector = self.embedder.embed_texts([query])[0]
        qdrant_filter = build_qdrant_filter(request, self.config)
        points = self.vector_db.query(
            collection=collection,
            vector=vector,
            limit=top_k,
            score_threshold=request.min_score,
            filter_obj=qdrant_filter,
            with_payload=True,
            vector_name=self.config.qdrant_vector_name,
        )
        matches = [
            normalize_match(
                point,
                self.config,
                include_text=request.include_text,
                include_metadata=request.include_metadata,
            )
            for point in points
        ]
        return RetrieveResponse(matches=matches)

    def rerank(self, request: RerankRequest) -> RerankResponse:
        query = request.query.strip()
        if not query:
            raise RetrieverError("missing_query")
        if not request.matches:
            raise RetrieverError("missing_matches", status_code=400)
        max_base_score = max((match.score for match in request.matches), default=0.0)
        reranked = [
            (
                _compute_rerank_score(query, match, max_base_score=max_base_score),
                index,
                match,
            )
            for index, match in enumerate(request.matches)
        ]
        reranked.sort(key=lambda item: (-item[0], -item[2].score, item[1]))
        limit = request.top_k or len(reranked)
        limited = reranked[: max(1, min(limit, len(reranked)))]
        return RerankResponse(
            strategy="heuristic_lexical_v1",
            matches=[
                RerankMatch(
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    text=match.text,
                    score=match.score,
                    rerank_score=score,
                    metadata=match.metadata,
                    source_uri=match.source_uri,
                )
                for score, _index, match in limited
            ],
        )

    def list_documents(self, request: DocumentListRequest) -> DocumentListResponse:
        self._validate_scope(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
        )
        collection = self._resolve_collection_name(request.collection_name)
        filter_obj = build_scope_filter(
            config=self.config,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
        )
        limit = max(1, min(request.limit, 500))
        scan_limit = max(limit * 50, 1000)
        points, truncated = self._scroll_points(
            collection=collection,
            filter_obj=filter_obj,
            limit=scan_limit,
        )
        documents = _summarize_documents(points, self.config)
        query = str(request.query or "").strip().lower()
        if query:
            documents = [
                document for document in documents if _document_matches_query(document, query)
            ]
        documents.sort(
            key=lambda item: (
                item.indexed_at or "",
                item.document_id.lower(),
            ),
            reverse=True,
        )
        return DocumentListResponse(
            collection_name=collection,
            truncated=truncated,
            scanned_point_count=len(points),
            documents=documents[:limit],
        )

    def get_document_chunks(self, request: DocumentChunksRequest) -> DocumentChunksResponse:
        self._validate_scope(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
        )
        collection = self._resolve_collection_name(request.collection_name)
        filter_obj = build_scope_filter(
            config=self.config,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            extra_conditions=[
                _match_condition(self.config.payload_document_id_key, request.document_id)
            ],
        )
        points, truncated = self._scroll_points(
            collection=collection,
            filter_obj=filter_obj,
            limit=request.limit,
        )
        if not points:
            raise RetrieverError("document_not_found", status_code=404)
        if truncated:
            raise RetrieverError("document_chunk_limit_exceeded", status_code=400)
        summary = _summarize_documents(points, self.config)[0]
        chunks = [
            _normalize_document_chunk(point, self.config)
            for point in sorted(points, key=_document_chunk_sort_key)
        ]
        return DocumentChunksResponse(
            collection_name=collection,
            document=summary,
            chunks=chunks,
        )

    def delete_document(self, request: DeleteDocumentRequest) -> DeleteDocumentResponse:
        chunks_response = self.get_document_chunks(
            DocumentChunksRequest(
                collection_name=request.collection_name,
                document_id=request.document_id,
                namespace=request.namespace,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                limit=2000,
            )
        )
        filter_obj = build_scope_filter(
            config=self.config,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            extra_conditions=[
                _match_condition(self.config.payload_document_id_key, request.document_id)
            ],
        )
        if filter_obj is None:
            raise RetrieverError("delete_filter_missing", status_code=400)
        self.vector_db.delete_points(
            collection=chunks_response.collection_name,
            filter_obj=filter_obj,
        )
        return DeleteDocumentResponse(
            collection_name=chunks_response.collection_name,
            document_id=request.document_id,
            deleted_chunk_count=len(chunks_response.chunks),
        )

    def ensure_collection(self, request: EnsureCollectionRequest | None = None) -> EnsureCollectionResponse:
        normalized = request or EnsureCollectionRequest()
        collection = self._resolve_collection_name(normalized.collection_name)
        vector_size = normalized.vector_size or self.config.qdrant_vector_size
        distance = (normalized.distance or self.config.qdrant_distance).strip() or "Cosine"
        vector_name = normalized.vector_name if normalized.vector_name is not None else self.config.qdrant_vector_name
        payload_keyword_fields = [
            field
            for field in (
                normalized.payload_keyword_fields or list(self.config.qdrant_payload_index_fields)
            )
            if str(field).strip()
        ]
        existing = self.vector_db.get_collection(collection)
        if existing is None:
            self.vector_db.create_collection(
                collection=collection,
                vector_size=vector_size,
                distance=distance,
                vector_name=vector_name,
                on_disk_payload=normalized.on_disk_payload,
            )
            status = "created"
        else:
            _validate_existing_collection(
                existing=existing,
                collection=collection,
                vector_size=vector_size,
                distance=distance,
                vector_name=vector_name,
            )
            status = "exists"
        if normalized.create_payload_indexes:
            for field_name in payload_keyword_fields:
                self.vector_db.create_payload_index(
                    collection=collection,
                    field_name=field_name,
                    field_schema="keyword",
                )
        return EnsureCollectionResponse(
            collection_name=collection,
            status=status,
            vector_size=vector_size,
            distance=distance,
            vector_name=vector_name,
            payload_keyword_fields=payload_keyword_fields if normalized.create_payload_indexes else [],
        )

    def upsert_texts(self, request: UpsertTextsRequest) -> UpsertTextsResponse:
        collection = self._resolve_collection_name(request.collection_name)
        if request.ensure_collection:
            self.ensure_collection(EnsureCollectionRequest(collection_name=collection))
        texts = [entry.text for entry in request.entries]
        vectors = self.embedder.embed_texts(texts)
        if len(vectors) != len(request.entries):
            raise RetrieverError("embedding_batch_size_mismatch", status_code=502)
        indexed_at = datetime.now(UTC).isoformat()
        points: list[dict[str, Any]] = []
        chunk_ids: list[str] = []
        for entry, vector in zip(request.entries, vectors, strict=False):
            normalized = _merge_upsert_scope(entry, request)
            if self.config.require_scope and not any(
                [normalized.tenant_id, normalized.user_id, normalized.workspace_id]
            ):
                raise RetrieverError(
                    "missing_scope:one_of_tenant_id_user_id_workspace_id_required"
                )
            chunk_id = normalized.chunk_id or _derive_chunk_id(
                collection=collection,
                document_id=normalized.document_id,
                text=normalized.text,
                source_uri=normalized.source_uri,
                namespace=normalized.namespace,
                tenant_id=normalized.tenant_id,
                user_id=normalized.user_id,
                workspace_id=normalized.workspace_id,
            )
            payload = _build_payload_from_entry(normalized, self.config)
            payload.setdefault("indexed_at", indexed_at)
            point: dict[str, Any] = {
                "id": chunk_id,
                "payload": payload,
            }
            if self.config.qdrant_vector_name:
                point["vector"] = {self.config.qdrant_vector_name: vector}
            else:
                point["vector"] = vector
            points.append(point)
            chunk_ids.append(chunk_id)
        self.vector_db.upsert_points(collection=collection, points=points)
        return UpsertTextsResponse(
            collection_name=collection,
            upserted_count=len(points),
            chunk_ids=chunk_ids,
        )

    def index_workspace_file(self, request: IndexWorkspaceFileRequest) -> IndexWorkspaceFileResponse:
        normalized_path, content = _read_workspace_file(request.path, self.config)
        chunk_size = request.chunk_size_chars or self.config.workspace_chunk_size_chars
        overlap = (
            request.chunk_overlap_chars
            if request.chunk_overlap_chars is not None
            else self.config.workspace_chunk_overlap_chars
        )
        max_chunks = request.max_chunks or self.config.workspace_max_chunks
        base_metadata = dict(request.metadata or {})
        base_metadata.setdefault("path", normalized_path)
        base_metadata.setdefault("filename", Path(normalized_path).name)
        base_metadata.setdefault("extension", Path(normalized_path).suffix.lower())
        base_metadata.setdefault("chunking_strategy", "text")
        prepared_chunks = _prepare_text_chunks(
            content,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks,
            base_metadata=base_metadata,
        )
        if not prepared_chunks:
            raise RetrieverError("workspace_file_empty_after_chunking", status_code=400)
        source_uri = request.source_uri or normalized_path
        document_id = request.document_id or normalized_path
        upsert_result = self._index_prepared_chunks(
            collection_name=request.collection_name,
            ensure_collection=request.ensure_collection,
            document_id=document_id,
            source_uri=source_uri,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            prepared_chunks=prepared_chunks,
        )
        return IndexWorkspaceFileResponse(
            collection_name=upsert_result.collection_name,
            path=normalized_path,
            document_id=document_id,
            chunk_count=len(prepared_chunks),
            upserted_count=upsert_result.upserted_count,
            chunk_ids=upsert_result.chunk_ids,
        )

    def index_markdown(self, request: IndexMarkdownRequest) -> IndexMarkdownResponse:
        markdown_text = request.markdown_text.strip()
        if not markdown_text:
            raise RetrieverError("markdown_text_empty", status_code=400)
        chunk_size = request.chunk_size_chars or self.config.workspace_chunk_size_chars
        overlap = (
            request.chunk_overlap_chars
            if request.chunk_overlap_chars is not None
            else self.config.workspace_chunk_overlap_chars
        )
        max_chunks = request.max_chunks or self.config.workspace_max_chunks
        source_uri = str(request.source_uri or request.document_id or "inline_markdown")
        document_id = str(request.document_id or source_uri)
        base_metadata = dict(request.metadata or {})
        base_metadata.setdefault("content_type", "markdown")
        base_metadata.setdefault("chunking_strategy", "markdown")
        prepared_chunks, section_count = _prepare_markdown_chunks(
            markdown_text,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks,
            base_metadata=base_metadata,
        )
        if not prepared_chunks:
            raise RetrieverError("markdown_empty_after_chunking", status_code=400)
        upsert_result = self._index_prepared_chunks(
            collection_name=request.collection_name,
            ensure_collection=request.ensure_collection,
            document_id=document_id,
            source_uri=source_uri,
            namespace=request.namespace,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
            prepared_chunks=prepared_chunks,
        )
        return IndexMarkdownResponse(
            collection_name=upsert_result.collection_name,
            document_id=document_id,
            source_uri=source_uri,
            section_count=section_count,
            chunk_count=len(prepared_chunks),
            upserted_count=upsert_result.upserted_count,
            chunk_ids=upsert_result.chunk_ids,
        )

    def index_workspace_directory(
        self,
        request: IndexWorkspaceDirectoryRequest,
    ) -> IndexWorkspaceDirectoryResponse:
        directory = _safe_workspace_path(request.directory_path, self.config)
        if not directory.exists():
            raise RetrieverError("workspace_directory_not_found", status_code=404)
        if not directory.is_dir():
            raise RetrieverError("workspace_directory_is_file", status_code=400)
        relative_directory = _relative_workspace_path(directory, self.config)
        candidate_paths = _list_workspace_files(
            directory,
            root=_workspace_root(self.config),
            recursive=request.recursive,
        )
        if not candidate_paths:
            raise RetrieverError("workspace_directory_no_files", status_code=400)
        max_files = request.max_files or 100
        allowed_extensions = _normalize_extensions(request.extensions) or set(
            self.config.workspace_allowed_extensions
        )
        collection = self._resolve_collection_name(request.collection_name)
        if request.ensure_collection:
            self.ensure_collection(EnsureCollectionRequest(collection_name=collection))
        files: list[IndexWorkspaceDirectoryFileResult] = []
        skipped: list[IndexWorkspaceDirectorySkippedFile] = []
        for candidate in candidate_paths[:max_files]:
            relative_path = _relative_workspace_path(candidate, self.config)
            suffix = candidate.suffix.lower()
            if allowed_extensions and suffix not in allowed_extensions:
                skipped.append(
                    IndexWorkspaceDirectorySkippedFile(
                        path=relative_path,
                        reason=f"extension_not_allowed:{suffix}",
                    )
                )
                continue
            try:
                normalized_path, content = _read_workspace_file(relative_path, self.config)
                base_metadata = dict(request.metadata or {})
                base_metadata.setdefault("path", normalized_path)
                base_metadata.setdefault("filename", Path(normalized_path).name)
                base_metadata.setdefault("extension", suffix)
                strategy = "markdown" if suffix in {".md", ".markdown", ".mdx"} else "text"
                if strategy == "markdown":
                    base_metadata.setdefault("content_type", "markdown")
                    base_metadata.setdefault("chunking_strategy", "markdown")
                    prepared_chunks, _section_count = _prepare_markdown_chunks(
                        content,
                        chunk_size=request.chunk_size_chars or self.config.workspace_chunk_size_chars,
                        overlap=(
                            request.chunk_overlap_chars
                            if request.chunk_overlap_chars is not None
                            else self.config.workspace_chunk_overlap_chars
                        ),
                        max_chunks=request.max_chunks_per_file or self.config.workspace_max_chunks,
                        base_metadata=base_metadata,
                    )
                else:
                    base_metadata.setdefault("chunking_strategy", "text")
                    prepared_chunks = _prepare_text_chunks(
                        content,
                        chunk_size=request.chunk_size_chars or self.config.workspace_chunk_size_chars,
                        overlap=(
                            request.chunk_overlap_chars
                            if request.chunk_overlap_chars is not None
                            else self.config.workspace_chunk_overlap_chars
                        ),
                        max_chunks=request.max_chunks_per_file or self.config.workspace_max_chunks,
                        base_metadata=base_metadata,
                    )
                if not prepared_chunks:
                    skipped.append(
                        IndexWorkspaceDirectorySkippedFile(
                            path=normalized_path,
                            reason="empty_after_chunking",
                        )
                    )
                    continue
                upsert_result = self._index_prepared_chunks(
                    collection_name=collection,
                    ensure_collection=False,
                    document_id=normalized_path,
                    source_uri=normalized_path,
                    namespace=request.namespace,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    prepared_chunks=prepared_chunks,
                )
                files.append(
                    IndexWorkspaceDirectoryFileResult(
                        path=normalized_path,
                        document_id=normalized_path,
                        strategy=strategy,
                        chunk_count=len(prepared_chunks),
                        upserted_count=upsert_result.upserted_count,
                        chunk_ids=upsert_result.chunk_ids,
                    )
                )
            except RetrieverError as exc:
                skipped.append(
                    IndexWorkspaceDirectorySkippedFile(
                        path=relative_path,
                        reason=exc.detail,
                    )
                )
        return IndexWorkspaceDirectoryResponse(
            collection_name=collection,
            directory_path=relative_directory,
            indexed_file_count=len(files),
            skipped_file_count=len(skipped),
            total_chunk_count=sum(item.chunk_count for item in files),
            total_upserted_count=sum(item.upserted_count for item in files),
            files=files,
            skipped=skipped,
        )

    def _index_prepared_chunks(
        self,
        *,
        collection_name: str | None,
        ensure_collection: bool,
        document_id: str,
        source_uri: str,
        namespace: str | None,
        tenant_id: str | None,
        user_id: str | None,
        workspace_id: str | None,
        prepared_chunks: list[_PreparedChunk],
    ) -> UpsertTextsResponse:
        entries = [
            UpsertTextEntry(
                document_id=document_id,
                text=chunk.text,
                source_uri=source_uri,
                namespace=namespace,
                tenant_id=tenant_id,
                user_id=user_id,
                workspace_id=workspace_id,
                metadata=chunk.metadata,
            )
            for chunk in prepared_chunks
        ]
        return self.upsert_texts(
            UpsertTextsRequest(
                collection_name=collection_name,
                ensure_collection=ensure_collection,
                namespace=namespace,
                tenant_id=tenant_id,
                user_id=user_id,
                workspace_id=workspace_id,
                entries=entries,
            )
        )

    def ensure_default_collection(self) -> EnsureCollectionResponse | None:
        if not self.config.qdrant_collection:
            return None
        return self.ensure_collection(EnsureCollectionRequest(collection_name=self.config.qdrant_collection))

    def _scroll_points(
        self,
        *,
        collection: str,
        filter_obj: dict[str, Any] | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        points: list[dict[str, Any]] = []
        offset: str | int | None = None
        truncated = False
        while len(points) < limit:
            batch_limit = min(256, limit - len(points))
            batch, next_offset = self.vector_db.scroll(
                collection=collection,
                limit=batch_limit,
                offset=offset,
                filter_obj=filter_obj,
                with_payload=True,
                vector_name=self.config.qdrant_vector_name,
            )
            if not batch:
                break
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        if offset is not None and len(points) >= limit:
            truncated = True
        return points, truncated

    def _validate_scope(
        self,
        *,
        tenant_id: str | None,
        user_id: str | None,
        workspace_id: str | None,
    ) -> None:
        if self.config.require_scope and not any([tenant_id, user_id, workspace_id]):
            raise RetrieverError("missing_scope:one_of_tenant_id_user_id_workspace_id_required")

    def _resolve_collection_name(self, collection_name: str | None) -> str:
        collection = str(collection_name or self.config.qdrant_collection).strip()
        if not collection:
            raise RetrieverError("qdrant_collection_missing", status_code=503)
        return collection


def build_service_from_env() -> RetrieverService:
    config = RetrieverServiceConfig.from_env()
    embedder = build_embedder_from_config(config)
    vector_db = QdrantClient(
        base_url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout_s=config.qdrant_timeout_s,
    )
    return RetrieverService(config=config, embedder=embedder, vector_db=vector_db)


def build_embedder_from_config(config: RetrieverServiceConfig) -> TextEmbedder:
    if config.embedding_provider == "openai":
        return OpenAIEmbeddingClient(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            timeout_s=config.embedding_timeout_s,
        )
    raise RetrieverError(
        f"unsupported_embedding_provider:{config.embedding_provider}",
        status_code=503,
    )


def build_qdrant_filter(
    request: RetrieveRequest,
    config: RetrieverServiceConfig,
) -> dict[str, Any] | None:
    must = _scope_filter_conditions(
        config=config,
        namespace=request.namespace,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        workspace_id=request.workspace_id,
    )
    raw_filters = request.filters if isinstance(request.filters, dict) else None
    if raw_filters:
        if any(
            key in raw_filters
            for key in ("must", "should", "must_not", "should_not", "min_should")
        ):
            merged = dict(raw_filters)
            existing_must = merged.get("must")
            merged_must = list(existing_must) if isinstance(existing_must, list) else []
            merged["must"] = must + merged_must
            return merged
        for key, value in raw_filters.items():
            if isinstance(value, (str, int, float, bool)):
                must.append(_match_condition(str(key), value))
            else:
                raise RetrieverError(
                    f"unsupported_filter_shape:{key}:expected_scalar_or_qdrant_filter"
                )

    if not must:
        return None
    return {"must": must}


def build_scope_filter(
    *,
    config: RetrieverServiceConfig,
    namespace: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    workspace_id: str | None = None,
    extra_conditions: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    must = _scope_filter_conditions(
        config=config,
        namespace=namespace,
        tenant_id=tenant_id,
        user_id=user_id,
        workspace_id=workspace_id,
    )
    if extra_conditions:
        must.extend(extra_conditions)
    if not must:
        return None
    return {"must": must}


def _scope_filter_conditions(
    *,
    config: RetrieverServiceConfig,
    namespace: str | None,
    tenant_id: str | None,
    user_id: str | None,
    workspace_id: str | None,
) -> list[dict[str, Any]]:
    must: list[dict[str, Any]] = []
    if namespace:
        must.append(_match_condition(config.payload_namespace_key, namespace))
    if tenant_id:
        must.append(_match_condition(config.payload_tenant_id_key, tenant_id))
    if user_id:
        must.append(_match_condition(config.payload_user_id_key, user_id))
    if workspace_id:
        must.append(_match_condition(config.payload_workspace_id_key, workspace_id))
    return must


def normalize_match(
    point: dict[str, Any],
    config: RetrieverServiceConfig,
    *,
    include_text: bool,
    include_metadata: bool,
) -> RetrieveMatch:
    payload = point.get("payload")
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    chunk_id = str(point.get("id") or payload_dict.get("chunk_id") or "")
    document_id = str(
        payload_dict.get(config.payload_document_id_key)
        or payload_dict.get("doc_id")
        or chunk_id
    )
    text = str(payload_dict.get(config.payload_text_key) or "")
    source_uri = str(
        payload_dict.get(config.payload_source_uri_key)
        or payload_dict.get("path")
        or payload_dict.get("source")
        or document_id
    )
    score = point.get("score", 0.0)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    metadata = payload_dict if include_metadata else {}
    if not include_text:
        text = ""
    return RetrieveMatch(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        score=score_value,
        metadata=metadata,
        source_uri=source_uri,
    )


def _normalize_document_chunk(
    point: dict[str, Any],
    config: RetrieverServiceConfig,
) -> DocumentChunk:
    payload = point.get("payload")
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    chunk_id = str(point.get("id") or payload_dict.get("chunk_id") or "")
    chunk_index_value = payload_dict.get("chunk_index")
    chunk_index = chunk_index_value if isinstance(chunk_index_value, int) else None
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=str(payload_dict.get(config.payload_document_id_key) or chunk_id),
        source_uri=str(
            payload_dict.get(config.payload_source_uri_key)
            or payload_dict.get("path")
            or payload_dict.get("source")
            or payload_dict.get(config.payload_document_id_key)
            or chunk_id
        ),
        text=str(payload_dict.get(config.payload_text_key) or ""),
        chunk_index=chunk_index,
        metadata=payload_dict,
    )


def _document_chunk_sort_key(point: dict[str, Any]) -> tuple[int, int, str]:
    payload = point.get("payload")
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    chunk_index = payload_dict.get("chunk_index")
    section_index = payload_dict.get("section_index")
    return (
        int(chunk_index) if isinstance(chunk_index, int) else 10**9,
        int(section_index) if isinstance(section_index, int) else 10**9,
        str(point.get("id") or ""),
    )


def _summarize_documents(
    points: list[dict[str, Any]],
    config: RetrieverServiceConfig,
) -> list[DocumentSummary]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        payload = point.get("payload")
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        document_id = str(
            payload_dict.get(config.payload_document_id_key)
            or payload_dict.get("doc_id")
            or point.get("id")
            or ""
        ).strip()
        if not document_id:
            continue
        grouped.setdefault(document_id, []).append(point)

    summaries: list[DocumentSummary] = []
    for document_id, document_points in grouped.items():
        payload = document_points[0].get("payload")
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        indexed_candidates = [
            str(point_payload.get("indexed_at") or "").strip()
            for point_payload in (
                dict(point.get("payload")) if isinstance(point.get("payload"), dict) else {}
                for point in document_points
            )
            if str(point_payload.get("indexed_at") or "").strip()
        ]
        summaries.append(
            DocumentSummary(
                document_id=document_id,
                source_uri=str(
                    payload_dict.get(config.payload_source_uri_key)
                    or payload_dict.get("path")
                    or payload_dict.get("source")
                    or document_id
                ),
                namespace=_maybe_str(payload_dict.get(config.payload_namespace_key)),
                tenant_id=_maybe_str(payload_dict.get(config.payload_tenant_id_key)),
                user_id=_maybe_str(payload_dict.get(config.payload_user_id_key)),
                workspace_id=_maybe_str(payload_dict.get(config.payload_workspace_id_key)),
                chunk_count=len(document_points),
                chunking_strategy=_maybe_str(payload_dict.get("chunking_strategy")),
                content_type=_maybe_str(payload_dict.get("content_type")),
                filename=_maybe_str(payload_dict.get("filename")),
                path=_maybe_str(payload_dict.get("path")),
                repo=_maybe_str(payload_dict.get("repo")),
                indexed_at=max(indexed_candidates) if indexed_candidates else None,
                metadata=_document_summary_metadata(payload_dict, config),
            )
        )
    return summaries


def _document_summary_metadata(
    payload: dict[str, Any],
    config: RetrieverServiceConfig,
) -> dict[str, Any]:
    excluded = {
        config.payload_text_key,
        config.payload_document_id_key,
        config.payload_source_uri_key,
        config.payload_namespace_key,
        config.payload_tenant_id_key,
        config.payload_user_id_key,
        config.payload_workspace_id_key,
        "chunk_index",
        "chunk_count",
        "section_index",
        "section_count",
        "section_chunk_index",
        "section_chunk_count",
        "indexed_at",
    }
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key in excluded:
            continue
        metadata[key] = value
    return metadata


def _document_matches_query(document: DocumentSummary, query: str) -> bool:
    haystack = " ".join(
        part
        for part in [
            document.document_id,
            document.source_uri,
            document.filename or "",
            document.path or "",
            document.repo or "",
            json.dumps(document.metadata, ensure_ascii=True, sort_keys=True),
        ]
        if part
    ).lower()
    return query in haystack


_RERANK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "use",
    "what",
    "with",
}


def _compute_rerank_score(
    query: str,
    match: RetrieveMatch,
    *,
    max_base_score: float,
) -> float:
    query_terms = _tokenize_rerank_text(query)
    if not query_terms:
        return round(_normalized_base_score(match.score, max_base_score), 6)

    primary_text = " ".join(
        part
        for part in [
            match.text,
            match.document_id,
            match.source_uri,
        ]
        if part
    )
    primary_terms = _tokenize_rerank_text(primary_text)
    primary_counter = Counter(primary_terms)
    unique_query_terms = set(query_terms)
    overlap_terms = unique_query_terms.intersection(primary_counter.keys())
    overlap_ratio = len(overlap_terms) / len(unique_query_terms)
    frequency_ratio = sum(min(primary_counter[term], 3) for term in unique_query_terms) / (
        len(unique_query_terms) * 3
    )

    metadata_text = _stringify_metadata_for_rerank(match.metadata)
    metadata_terms = set(_tokenize_rerank_text(metadata_text))
    metadata_overlap = len(unique_query_terms.intersection(metadata_terms)) / len(unique_query_terms)

    normalized_query = _normalize_rerank_phrase(query)
    normalized_text = _normalize_rerank_phrase(f"{primary_text} {metadata_text}")
    phrase_bonus = 0.18 if normalized_query and normalized_query in normalized_text else 0.0

    base_score = _normalized_base_score(match.score, max_base_score)
    rerank_score = (
        0.45 * overlap_ratio
        + 0.15 * frequency_ratio
        + 0.12 * metadata_overlap
        + 0.18 * base_score
        + phrase_bonus
    )
    return round(rerank_score, 6)


def _normalized_base_score(score: float, max_base_score: float) -> float:
    if max_base_score <= 0:
        return max(0.0, float(score))
    return max(0.0, min(float(score) / max_base_score, 1.0))


def _stringify_metadata_for_rerank(metadata: dict[str, Any]) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return ""
    parts: list[str] = []
    for key in (
        "title",
        "heading",
        "heading_path",
        "section",
        "filename",
        "path",
        "repo",
        "content_type",
    ):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value))
        elif isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
    return " ".join(parts)


def _tokenize_rerank_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", str(text or "").lower())
        if len(token) > 1 and token not in _RERANK_STOPWORDS
    ]


def _normalize_rerank_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _merge_upsert_scope(entry: UpsertTextEntry, request: UpsertTextsRequest) -> UpsertTextEntry:
    return UpsertTextEntry(
        chunk_id=entry.chunk_id,
        document_id=entry.document_id,
        text=entry.text,
        source_uri=entry.source_uri,
        namespace=entry.namespace or request.namespace,
        tenant_id=entry.tenant_id or request.tenant_id,
        user_id=entry.user_id or request.user_id,
        workspace_id=entry.workspace_id or request.workspace_id,
        metadata=dict(entry.metadata or {}),
    )


def _build_payload_from_entry(
    entry: UpsertTextEntry,
    config: RetrieverServiceConfig,
) -> dict[str, Any]:
    payload = dict(entry.metadata or {})
    payload[config.payload_text_key] = entry.text
    payload[config.payload_document_id_key] = entry.document_id
    if entry.source_uri:
        payload[config.payload_source_uri_key] = entry.source_uri
    if entry.namespace:
        payload[config.payload_namespace_key] = entry.namespace
    if entry.tenant_id:
        payload[config.payload_tenant_id_key] = entry.tenant_id
    if entry.user_id:
        payload[config.payload_user_id_key] = entry.user_id
    if entry.workspace_id:
        payload[config.payload_workspace_id_key] = entry.workspace_id
    return payload


def _derive_chunk_id(
    *,
    collection: str,
    document_id: str,
    text: str,
    source_uri: str | None,
    namespace: str | None,
    tenant_id: str | None,
    user_id: str | None,
    workspace_id: str | None,
) -> str:
    digest = hashlib.sha256(
        "|".join(
            [
                collection,
                document_id,
                source_uri or "",
                namespace or "",
                tenant_id or "",
                user_id or "",
                workspace_id or "",
                text,
            ]
        ).encode("utf-8")
    ).hexdigest()
    return digest[:32]


def _validate_existing_collection(
    *,
    existing: dict[str, Any],
    collection: str,
    vector_size: int,
    distance: str,
    vector_name: str | None,
) -> None:
    config = existing.get("result", {}).get("config", {}).get("params", {})
    vectors = config.get("vectors")
    existing_size: int | None = None
    existing_distance: str | None = None
    if vector_name:
        if isinstance(vectors, dict):
            named = vectors.get(vector_name)
            if isinstance(named, dict):
                existing_size = _coerce_int(named.get("size"))
                existing_distance = _upper_str(named.get("distance"))
    elif isinstance(vectors, dict):
        if "size" in vectors:
            existing_size = _coerce_int(vectors.get("size"))
            existing_distance = _upper_str(vectors.get("distance"))
    if existing_size is not None and existing_size != vector_size:
        raise RetrieverError(
            f"qdrant_collection_configuration_mismatch:{collection}:vector_size",
            status_code=409,
        )
    if existing_distance is not None and existing_distance != _upper_str(distance):
        raise RetrieverError(
            f"qdrant_collection_configuration_mismatch:{collection}:distance",
            status_code=409,
        )


def _json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | list[Any] | None,
    timeout_s: float,
    headers: dict[str, str],
    error_prefix: str,
    invalid_json_prefix: str,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RetrieverError(f"{error_prefix}:{detail}", status_code=exc.code) from exc
    except (URLError, TimeoutError) as exc:
        raise RetrieverError(f"{error_prefix}:{exc}", status_code=502) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RetrieverError(f"{invalid_json_prefix}:{exc}", status_code=502) from exc
    if not isinstance(data, dict):
        raise RetrieverError("json_response_invalid", status_code=502)
    return data


def _match_condition(key: str, value: str | int | float | bool) -> dict[str, Any]:
    return {"key": key, "match": {"value": value}}


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def _csv_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _workspace_root(config: RetrieverServiceConfig) -> Path:
    return Path(config.workspace_dir).resolve()


def _safe_workspace_path(path: str, config: RetrieverServiceConfig) -> Path:
    base_dir = _workspace_root(config)
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir)):
        raise RetrieverError("invalid_workspace_path_outside_workspace", status_code=400)
    return resolved


def _read_workspace_file(path: str, config: RetrieverServiceConfig) -> tuple[str, str]:
    candidate = _safe_workspace_path(path, config)
    if not candidate.exists():
        raise RetrieverError("workspace_file_not_found", status_code=404)
    if candidate.is_dir():
        raise RetrieverError("workspace_file_is_directory", status_code=400)
    suffix = candidate.suffix.lower()
    allowed = set(config.workspace_allowed_extensions)
    if allowed and suffix not in allowed:
        raise RetrieverError(f"workspace_file_extension_not_allowed:{suffix}", status_code=400)
    stat = candidate.stat()
    if stat.st_size > config.workspace_max_file_bytes:
        raise RetrieverError("workspace_file_too_large", status_code=400)
    text = candidate.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise RetrieverError("workspace_file_empty", status_code=400)
    try:
        relative_path = str(candidate.relative_to(_workspace_root(config)))
    except ValueError:
        relative_path = str(candidate)
    return relative_path, text


def _relative_workspace_path(path: Path, config: RetrieverServiceConfig) -> str:
    try:
        return str(path.resolve().relative_to(_workspace_root(config)))
    except ValueError:
        return str(path.resolve())


def _prepare_text_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
    base_metadata: dict[str, Any],
) -> list[_PreparedChunk]:
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
    prepared: list[_PreparedChunk] = []
    for index, chunk_text in enumerate(chunks):
        chunk_metadata = dict(base_metadata)
        chunk_metadata["chunk_index"] = index
        chunk_metadata["chunk_count"] = len(chunks)
        prepared.append(_PreparedChunk(text=chunk_text, metadata=chunk_metadata))
    return prepared


def _prepare_markdown_chunks(
    markdown_text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
    base_metadata: dict[str, Any],
) -> tuple[list[_PreparedChunk], int]:
    sections = _split_markdown_sections(markdown_text)
    prepared: list[_PreparedChunk] = []
    for section_index, section in enumerate(sections):
        remaining = max_chunks - len(prepared)
        if remaining <= 0:
            break
        section_base = dict(base_metadata)
        section_base["section_index"] = section_index
        section_base["section_count"] = len(sections)
        if section.title:
            section_base["section_title"] = section.title
        if section.heading_level is not None:
            section_base["heading_level"] = section.heading_level
        if section.heading_path:
            section_base["heading_path"] = list(section.heading_path)
        section_chunks = _chunk_text(
            section.text,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=remaining,
        )
        for local_index, chunk_text in enumerate(section_chunks):
            chunk_metadata = dict(section_base)
            chunk_metadata["section_chunk_index"] = local_index
            chunk_metadata["section_chunk_count"] = len(section_chunks)
            prepared.append(_PreparedChunk(text=chunk_text, metadata=chunk_metadata))
    for global_index, chunk in enumerate(prepared):
        chunk.metadata["chunk_index"] = global_index
        chunk.metadata["chunk_count"] = len(prepared)
    return prepared, len(sections)


def _split_markdown_sections(markdown_text: str) -> list[_MarkdownSection]:
    normalized = markdown_text.strip()
    if not normalized:
        return []
    sections: list[_MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_title: str | None = None
    current_level: int | None = None
    current_path: tuple[str, ...] = ()
    in_code_block = False
    heading_re = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

    def flush_section() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append(
            _MarkdownSection(
                text=text,
                title=current_title,
                heading_level=current_level,
                heading_path=current_path,
            )
        )

    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
        heading_match = None if in_code_block else heading_re.match(line)
        if heading_match:
            flush_section()
            hashes, raw_title = heading_match.groups()
            level = len(hashes)
            title = raw_title.strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_lines = [line]
            current_title = title
            current_level = level
            current_path = tuple(item[1] for item in heading_stack)
            continue
        current_lines.append(line)

    flush_section()
    return sections


def _normalize_extensions(extensions: list[str] | None) -> set[str]:
    normalized: set[str] = set()
    if not extensions:
        return normalized
    for extension in extensions:
        value = str(extension).strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        normalized.add(value)
    return normalized


def _list_workspace_files(directory: Path, *, root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    ignored_dir_names = {"node_modules", "__pycache__", ".git", ".next", ".venv", "venv"}
    files: list[Path] = []
    for candidate in sorted(directory.glob(pattern)):
        if not candidate.is_file():
            continue
        try:
            relative_parts = candidate.resolve().relative_to(root.resolve()).parts
        except ValueError:
            continue
        if any(part.startswith(".") for part in relative_parts[:-1]):
            continue
        if any(part in ignored_dir_names for part in relative_parts[:-1]):
            continue
        if relative_parts and relative_parts[-1].startswith("."):
            continue
        files.append(candidate)
    return files


def _chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    size = max(200, chunk_size)
    effective_overlap = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    start = 0
    length = len(normalized)
    while start < length and len(chunks) < max_chunks:
        end = min(length, start + size)
        if end < length:
            split_window = normalized[start:end]
            newline_break = split_window.rfind("\n\n")
            line_break = split_window.rfind("\n")
            space_break = split_window.rfind(" ")
            preferred = max(newline_break, line_break, space_break)
            if preferred > int(size * 0.6):
                end = start + preferred
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        next_start = max(start + 1, end - effective_overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _int_with_default(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_with_default(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _bool_with_default(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _default_embedding_size(model: str | None) -> int:
    normalized = (model or "").strip().lower()
    known = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return known.get(normalized, 1536)


def _upper_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper() or None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
