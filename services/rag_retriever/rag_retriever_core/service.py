from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import RetrieverError
from .models import (
    EnsureCollectionRequest,
    EnsureCollectionResponse,
    IndexWorkspaceFileRequest,
    IndexWorkspaceFileResponse,
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
        overlap = request.chunk_overlap_chars or self.config.workspace_chunk_overlap_chars
        max_chunks = request.max_chunks or self.config.workspace_max_chunks
        chunks = _chunk_text(content, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
        if not chunks:
            raise RetrieverError("workspace_file_empty_after_chunking", status_code=400)
        source_uri = request.source_uri or normalized_path
        document_id = request.document_id or normalized_path
        base_metadata = dict(request.metadata or {})
        base_metadata.setdefault("path", normalized_path)
        base_metadata.setdefault("filename", Path(normalized_path).name)
        base_metadata.setdefault("extension", Path(normalized_path).suffix.lower())
        entries: list[UpsertTextEntry] = []
        for index, chunk_text in enumerate(chunks):
            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_index"] = index
            chunk_metadata["chunk_count"] = len(chunks)
            entries.append(
                UpsertTextEntry(
                    document_id=document_id,
                    text=chunk_text,
                    source_uri=source_uri,
                    namespace=request.namespace,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    metadata=chunk_metadata,
                )
            )
        upsert_result = self.upsert_texts(
            UpsertTextsRequest(
                collection_name=request.collection_name,
                ensure_collection=request.ensure_collection,
                namespace=request.namespace,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                entries=entries,
            )
        )
        return IndexWorkspaceFileResponse(
            collection_name=upsert_result.collection_name,
            path=normalized_path,
            document_id=document_id,
            chunk_count=len(chunks),
            upserted_count=upsert_result.upserted_count,
            chunk_ids=upsert_result.chunk_ids,
        )

    def ensure_default_collection(self) -> EnsureCollectionResponse | None:
        if not self.config.qdrant_collection:
            return None
        return self.ensure_collection(EnsureCollectionRequest(collection_name=self.config.qdrant_collection))

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
    must: list[dict[str, Any]] = []
    if request.namespace:
        must.append(_match_condition(config.payload_namespace_key, request.namespace))
    if request.tenant_id:
        must.append(_match_condition(config.payload_tenant_id_key, request.tenant_id))
    if request.user_id:
        must.append(_match_condition(config.payload_user_id_key, request.user_id))
    if request.workspace_id:
        must.append(_match_condition(config.payload_workspace_id_key, request.workspace_id))

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
