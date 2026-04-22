from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_retriever_core import (
    DeleteDocumentRequest,
    DocumentChunksRequest,
    DocumentListRequest,
    EnsureCollectionRequest,
    IndexMarkdownRequest,
    IndexWorkspaceDirectoryRequest,
    IndexWorkspaceFileRequest,
    RerankRequest,
    RetrieveMatch,
    RetrieveRequest,
    UpsertTextEntry,
    UpsertTextsRequest,
)
from rag_retriever_core.errors import RetrieverError
from rag_retriever_core.service import (
    RetrieverService,
    RetrieverServiceConfig,
    build_qdrant_filter,
)


@dataclass
class _EmbedderStub:
    vectors: list[list[float]]

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if not texts:
            return []
        if not self.vectors:
            raise AssertionError("Embedder stub requires at least one vector")
        vectors = [list(vector) for vector in self.vectors]
        while len(vectors) < len(texts):
            vectors.append(list(self.vectors[-1]))
        return vectors[: len(texts)]


@dataclass
class _VectorDbStub:
    points: list[dict[str, Any]]
    existing_collection: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.query_calls: list[dict[str, Any]] = []
        self.scroll_calls: list[dict[str, Any]] = []
        self.created_collections: list[dict[str, Any]] = []
        self.created_payload_indexes: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def query(
        self,
        *,
        collection: str,
        vector: list[float],
        limit: int,
        score_threshold: float | None,
        filter_obj: dict | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> list[dict[str, Any]]:
        self.query_calls.append(
            {
                "collection": collection,
                "vector": vector,
                "limit": limit,
                "score_threshold": score_threshold,
                "filter_obj": filter_obj,
                "with_payload": with_payload,
                "vector_name": vector_name,
            }
        )
        return list(self.points)

    def get_collection(self, collection: str) -> dict[str, Any] | None:
        return self.existing_collection

    def scroll(
        self,
        *,
        collection: str,
        limit: int,
        offset: str | int | None,
        filter_obj: dict | None,
        with_payload: bool,
        vector_name: str | None,
    ) -> tuple[list[dict[str, Any]], str | int | None]:
        self.scroll_calls.append(
            {
                "collection": collection,
                "limit": limit,
                "offset": offset,
                "filter_obj": filter_obj,
                "with_payload": with_payload,
                "vector_name": vector_name,
            }
        )
        matched = [point for point in self.points if _matches_filter(point, filter_obj)]
        start_index = int(offset) if isinstance(offset, int) else 0
        batch = matched[start_index : start_index + limit]
        next_offset = start_index + limit if start_index + limit < len(matched) else None
        return batch, next_offset

    def create_collection(
        self,
        *,
        collection: str,
        vector_size: int,
        distance: str,
        vector_name: str | None,
        on_disk_payload: bool,
    ) -> None:
        self.created_collections.append(
            {
                "collection": collection,
                "vector_size": vector_size,
                "distance": distance,
                "vector_name": vector_name,
                "on_disk_payload": on_disk_payload,
            }
        )
        self.existing_collection = {
            "result": {
                "config": {
                    "params": {
                        "vectors": {
                            "size": vector_size,
                            "distance": distance,
                        }
                    }
                }
            }
        }

    def create_payload_index(
        self,
        *,
        collection: str,
        field_name: str,
        field_schema: str = "keyword",
    ) -> None:
        self.created_payload_indexes.append(
            {
                "collection": collection,
                "field_name": field_name,
                "field_schema": field_schema,
            }
        )

    def upsert_points(
        self,
        *,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        self.upsert_calls.append({"collection": collection, "points": points})

    def delete_points(
        self,
        *,
        collection: str,
        filter_obj: dict[str, Any],
    ) -> None:
        self.delete_calls.append({"collection": collection, "filter_obj": filter_obj})
        self.points = [point for point in self.points if not _matches_filter(point, filter_obj)]


def _matches_filter(point: dict[str, Any], filter_obj: dict[str, Any] | None) -> bool:
    if not isinstance(filter_obj, dict):
        return True
    must = filter_obj.get("must")
    if not isinstance(must, list):
        return True
    payload = point.get("payload")
    payload_dict = dict(payload) if isinstance(payload, dict) else {}
    for condition in must:
        if not isinstance(condition, dict):
            continue
        key = condition.get("key")
        match = condition.get("match")
        if not isinstance(key, str) or not isinstance(match, dict):
            continue
        if payload_dict.get(key) != match.get("value"):
            return False
    return True


def _config(require_scope: bool = True) -> RetrieverServiceConfig:
    return RetrieverServiceConfig(
        qdrant_url="http://qdrant:6333",
        qdrant_api_key="",
        qdrant_collection="rag_default",
        qdrant_vector_name=None,
        qdrant_timeout_s=10.0,
        qdrant_vector_size=1536,
        qdrant_distance="Cosine",
        qdrant_on_disk_payload=True,
        qdrant_create_payload_indexes=True,
        qdrant_payload_index_fields=(
            "document_id",
            "namespace",
            "tenant_id",
            "user_id",
            "workspace_id",
            "source_uri",
        ),
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key="test-key",
        embedding_base_url="https://api.openai.com",
        embedding_timeout_s=15.0,
        top_k_default=5,
        top_k_max=20,
        require_scope=require_scope,
        workspace_dir="/tmp/rag-workspace",
        workspace_allowed_extensions=(
            ".md",
            ".txt",
            ".rst",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".css",
            ".html",
            ".sql",
            ".sh",
        ),
        workspace_max_file_bytes=2_000_000,
        workspace_chunk_size_chars=1200,
        workspace_chunk_overlap_chars=200,
        workspace_max_chunks=200,
        payload_text_key="text",
        payload_document_id_key="document_id",
        payload_source_uri_key="source_uri",
        payload_namespace_key="namespace",
        payload_tenant_id_key="tenant_id",
        payload_user_id_key="user_id",
        payload_workspace_id_key="workspace_id",
    )


def test_build_qdrant_filter_merges_scope_and_simple_scalar_filters() -> None:
    request = RetrieveRequest(
        query="kubernetes rollout",
        tenant_id="tenant-a",
        workspace_id="ws-1",
        namespace="docs",
        filters={"repo": "agentic-workflow-studio"},
    )

    built = build_qdrant_filter(request, _config())

    assert built == {
        "must": [
            {"key": "namespace", "match": {"value": "docs"}},
            {"key": "tenant_id", "match": {"value": "tenant-a"}},
            {"key": "workspace_id", "match": {"value": "ws-1"}},
            {"key": "repo", "match": {"value": "agentic-workflow-studio"}},
        ]
    }


def test_retrieve_normalizes_matches_and_enforces_top_k() -> None:
    embedder = _EmbedderStub([[0.1, 0.2, 0.3]])
    vector_db = _VectorDbStub(
        [
            {
                "id": "chunk-1",
                "score": 0.91,
                "payload": {
                    "document_id": "doc-1",
                    "text": "Deployment uses API, planner, and worker services.",
                    "source_uri": "README.md#architecture",
                    "tenant_id": "tenant-a",
                },
            }
        ]
    )
    service = RetrieverService(config=_config(), embedder=embedder, vector_db=vector_db)

    result = service.retrieve(
        RetrieveRequest(
            query="How does deployment work?",
            tenant_id="tenant-a",
            top_k=20,
            include_text=True,
            include_metadata=True,
        )
    )

    assert embedder.calls == [["How does deployment work?"]]
    assert vector_db.query_calls[0]["limit"] == 20
    assert result.matches[0].chunk_id == "chunk-1"
    assert result.matches[0].document_id == "doc-1"
    assert result.matches[0].source_uri == "README.md#architecture"
    assert "planner" in result.matches[0].text
    assert result.matches[0].metadata["tenant_id"] == "tenant-a"


def test_retrieve_requires_scope_when_configured() -> None:
    service = RetrieverService(
        config=_config(require_scope=True),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    try:
        service.retrieve(RetrieveRequest(query="ungrounded query"))
    except RetrieverError as exc:
        assert exc.detail == "missing_scope:one_of_tenant_id_user_id_workspace_id_required"
    else:  # pragma: no cover
        raise AssertionError("Expected RetrieverError")


def test_rerank_prioritizes_lexical_and_metadata_matches() -> None:
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    result = service.rerank(
        RerankRequest(
            query="workflow studio versions",
            matches=[
                RetrieveMatch(
                    chunk_id="chunk-1",
                    document_id="readme",
                    text="Deployment uses Kubernetes overlays and local registry tags.",
                    score=0.96,
                    metadata={"path": "README.md"},
                    source_uri="README.md#kubernetes",
                ),
                RetrieveMatch(
                    chunk_id="chunk-2",
                    document_id="docs/user-guide.md",
                    text="Workflow Studio supports saved drafts, published versions, and run history.",
                    score=0.72,
                    metadata={"heading_path": ["Studio", "Versions"]},
                    source_uri="docs/user-guide.md#studio",
                ),
            ],
            top_k=1,
        )
    )

    assert result.strategy == "heuristic_lexical_v1"
    assert len(result.matches) == 1
    assert result.matches[0].chunk_id == "chunk-2"
    assert result.matches[0].rerank_score > 0.5


def test_rerank_preserves_best_base_score_when_matches_are_ungrounded() -> None:
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    result = service.rerank(
        RerankRequest(
            query="unrelated query",
            matches=[
                RetrieveMatch(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    text="alpha beta",
                    score=0.91,
                    metadata={},
                    source_uri="doc-1",
                ),
                RetrieveMatch(
                    chunk_id="chunk-2",
                    document_id="doc-2",
                    text="gamma delta",
                    score=0.47,
                    metadata={},
                    source_uri="doc-2",
                ),
            ],
        )
    )

    assert [match.chunk_id for match in result.matches] == ["chunk-1", "chunk-2"]
    assert result.matches[0].rerank_score >= result.matches[1].rerank_score


def test_list_documents_aggregates_chunks_into_document_summaries() -> None:
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub(
            [
                {
                    "id": "chunk-1",
                    "payload": {
                        "document_id": "docs/user-guide.md",
                        "text": "Workflow Studio supports saved drafts.",
                        "source_uri": "docs/user-guide.md",
                        "user_id": "narendersurabhi",
                        "namespace": "docs",
                        "path": "docs/user-guide.md",
                        "filename": "user-guide.md",
                        "repo": "agentic-workflow-studio",
                        "chunking_strategy": "markdown",
                        "indexed_at": "2026-03-17T20:00:00+00:00",
                    },
                },
                {
                    "id": "chunk-2",
                    "payload": {
                        "document_id": "docs/user-guide.md",
                        "text": "Published versions can be run manually.",
                        "source_uri": "docs/user-guide.md",
                        "user_id": "narendersurabhi",
                        "namespace": "docs",
                        "path": "docs/user-guide.md",
                        "filename": "user-guide.md",
                        "repo": "agentic-workflow-studio",
                        "chunking_strategy": "markdown",
                        "indexed_at": "2026-03-17T20:00:01+00:00",
                    },
                },
                {
                    "id": "chunk-3",
                    "payload": {
                        "document_id": "docs/rag-playbook.md",
                        "text": "rag.retrieve reranks grounded chunks.",
                        "source_uri": "docs/rag-playbook.md",
                        "user_id": "narendersurabhi",
                        "namespace": "docs",
                        "path": "docs/rag-playbook.md",
                        "filename": "rag-playbook.md",
                        "repo": "agentic-workflow-studio",
                        "chunking_strategy": "markdown",
                        "indexed_at": "2026-03-17T19:59:59+00:00",
                    },
                },
            ]
        ),
    )

    result = service.list_documents(
        DocumentListRequest(
            user_id="narendersurabhi",
            namespace="docs",
            query="user-guide",
        )
    )

    assert result.collection_name == "rag_default"
    assert result.truncated is False
    assert result.scanned_point_count == 3
    assert len(result.documents) == 1
    assert result.documents[0].document_id == "docs/user-guide.md"
    assert result.documents[0].chunk_count == 2
    assert result.documents[0].indexed_at == "2026-03-17T20:00:01+00:00"


def test_get_document_chunks_returns_sorted_chunks() -> None:
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub(
            [
                {
                    "id": "chunk-2",
                    "payload": {
                        "document_id": "docs/user-guide.md",
                        "text": "Second chunk",
                        "source_uri": "docs/user-guide.md",
                        "user_id": "narendersurabhi",
                        "namespace": "docs",
                        "chunk_index": 1,
                        "path": "docs/user-guide.md",
                    },
                },
                {
                    "id": "chunk-1",
                    "payload": {
                        "document_id": "docs/user-guide.md",
                        "text": "First chunk",
                        "source_uri": "docs/user-guide.md",
                        "user_id": "narendersurabhi",
                        "namespace": "docs",
                        "chunk_index": 0,
                        "path": "docs/user-guide.md",
                    },
                },
            ]
        ),
    )

    result = service.get_document_chunks(
        DocumentChunksRequest(
            document_id="docs/user-guide.md",
            user_id="narendersurabhi",
            namespace="docs",
        )
    )

    assert result.document.document_id == "docs/user-guide.md"
    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1", "chunk-2"]
    assert result.chunks[0].chunk_index == 0


def test_delete_document_removes_all_matching_points() -> None:
    vector_db = _VectorDbStub(
        [
            {
                "id": "chunk-1",
                "payload": {
                    "document_id": "docs/user-guide.md",
                    "text": "First chunk",
                    "source_uri": "docs/user-guide.md",
                    "user_id": "narendersurabhi",
                    "namespace": "docs",
                },
            },
            {
                "id": "chunk-2",
                "payload": {
                    "document_id": "docs/user-guide.md",
                    "text": "Second chunk",
                    "source_uri": "docs/user-guide.md",
                    "user_id": "narendersurabhi",
                    "namespace": "docs",
                },
            },
            {
                "id": "chunk-3",
                "payload": {
                    "document_id": "docs/rag-playbook.md",
                    "text": "Other doc",
                    "source_uri": "docs/rag-playbook.md",
                    "user_id": "narendersurabhi",
                    "namespace": "docs",
                },
            },
        ]
    )
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=vector_db,
    )

    result = service.delete_document(
        DeleteDocumentRequest(
            document_id="docs/user-guide.md",
            user_id="narendersurabhi",
            namespace="docs",
        )
    )

    assert result.deleted_chunk_count == 2
    remaining_ids = [point["id"] for point in vector_db.points]
    assert remaining_ids == ["chunk-3"]


def test_index_workspace_file_reads_chunks_and_upserts(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source = workspace_dir / "docs" / "guide.md"
    source.parent.mkdir()
    source.write_text(
        (
            "Workflow Studio saves drafts and publishes versions. "
            "Use triggers to invoke published workflows. "
            "The workflow library now includes saved drafts, version history, and run history. "
            "RAG indexing now supports workspace files so that repository and documentation content can be chunked and embedded. "
            "This lets retrieval operate against the same shared workspace used by other services.\n\n"
        )
        * 3,
        encoding="utf-8",
    )
    config = _config()
    config = RetrieverServiceConfig(**{**config.__dict__, "workspace_dir": str(workspace_dir)})
    embedder = _EmbedderStub([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    vector_db = _VectorDbStub([])
    service = RetrieverService(config=config, embedder=embedder, vector_db=vector_db)

    result = service.index_workspace_file(
        IndexWorkspaceFileRequest(
            path="docs/guide.md",
            tenant_id="tenant-a",
            workspace_id="ws-1",
            namespace="docs",
            chunk_size_chars=200,
            chunk_overlap_chars=40,
            metadata={"repo": "agentic-workflow-studio"},
        )
    )

    assert result.path == "docs/guide.md"
    assert result.document_id == "docs/guide.md"
    assert result.chunk_count == result.upserted_count
    assert result.chunk_count >= 2
    assert embedder.calls
    points = vector_db.upsert_calls[0]["points"]
    assert points[0]["payload"]["path"] == "docs/guide.md"
    assert points[0]["payload"]["repo"] == "agentic-workflow-studio"
    assert points[0]["payload"]["chunk_index"] == 0
    assert points[0]["payload"]["chunk_count"] == result.chunk_count


def test_index_workspace_file_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    config = _config()
    config = RetrieverServiceConfig(**{**config.__dict__, "workspace_dir": str(workspace_dir)})
    service = RetrieverService(
        config=config,
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    try:
        service.index_workspace_file(
            IndexWorkspaceFileRequest(
                path="../outside.txt",
                tenant_id="tenant-a",
                workspace_id="ws-1",
            )
        )
    except RetrieverError as exc:
        assert exc.detail == "invalid_workspace_path_outside_workspace"
    else:  # pragma: no cover
        raise AssertionError("Expected RetrieverError")


def test_index_markdown_chunks_by_headings_and_records_section_metadata() -> None:
    embedder = _EmbedderStub([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    vector_db = _VectorDbStub([])
    service = RetrieverService(config=_config(), embedder=embedder, vector_db=vector_db)

    result = service.index_markdown(
        IndexMarkdownRequest(
            markdown_text=(
                "# Overview\n"
                "Agentic Workflow Studio supports chat, compose, and Studio.\n\n"
                "## RAG\n"
                "RAG indexing supports markdown-aware chunking.\n"
                "Directory indexing can recurse through docs folders.\n"
            ),
            user_id="narendersurabhi",
            namespace="docs",
            source_uri="docs/overview.md",
            metadata={"repo": "agentic-workflow-studio"},
            chunk_size_chars=220,
            chunk_overlap_chars=20,
        )
    )

    assert result.document_id == "docs/overview.md"
    assert result.source_uri == "docs/overview.md"
    assert result.section_count >= 2
    assert result.chunk_count == result.upserted_count
    upsert_points = vector_db.upsert_calls[0]["points"]
    assert upsert_points[0]["payload"]["repo"] == "agentic-workflow-studio"
    assert upsert_points[0]["payload"]["content_type"] == "markdown"
    assert upsert_points[0]["payload"]["chunking_strategy"] == "markdown"
    assert "heading_path" in upsert_points[0]["payload"]


def test_index_workspace_directory_indexes_text_and_markdown_files(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    docs_dir = workspace_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "guide.md").write_text(
        "# Guide\nWorkflow Studio supports saved drafts and versions.\n\n## RAG\nUse workspace directory indexing.",
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text(
        "This text file should be indexed with plain text chunking.",
        encoding="utf-8",
    )
    (docs_dir / ".hidden.md").write_text("# Hidden\nThis should be ignored.", encoding="utf-8")
    (docs_dir / "data.bin").write_bytes(b"\x00\x01\x02")
    config = _config()
    config = RetrieverServiceConfig(**{**config.__dict__, "workspace_dir": str(workspace_dir)})
    embedder = _EmbedderStub([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    vector_db = _VectorDbStub([])
    service = RetrieverService(config=config, embedder=embedder, vector_db=vector_db)

    result = service.index_workspace_directory(
        IndexWorkspaceDirectoryRequest(
            directory_path="docs",
            user_id="narendersurabhi",
            namespace="docs",
            recursive=True,
            max_files=10,
            metadata={"repo": "agentic-workflow-studio"},
        )
    )

    assert result.directory_path == "docs"
    assert result.indexed_file_count == 2
    assert result.skipped_file_count == 1
    assert {item.path for item in result.files} == {"docs/guide.md", "docs/notes.txt"}
    strategies = {item.path: item.strategy for item in result.files}
    assert strategies["docs/guide.md"] == "markdown"
    assert strategies["docs/notes.txt"] == "text"
    assert result.skipped[0].path == "docs/data.bin"
    assert result.skipped[0].reason == "extension_not_allowed:.bin"
    upsert_payloads = [call["points"][0]["payload"] for call in vector_db.upsert_calls]
    assert any(payload["chunking_strategy"] == "markdown" for payload in upsert_payloads)
    assert any(payload["chunking_strategy"] == "text" for payload in upsert_payloads)


def test_ensure_collection_creates_collection_and_payload_indexes() -> None:
    service = RetrieverService(
        config=_config(),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    result = service.ensure_collection(EnsureCollectionRequest())

    assert result.collection_name == "rag_default"
    assert result.status == "created"
    assert result.vector_size == 1536
    assert len(service.vector_db.created_collections) == 1
    indexed_fields = [item["field_name"] for item in service.vector_db.created_payload_indexes]
    assert "document_id" in indexed_fields
    assert "workspace_id" in indexed_fields


def test_upsert_texts_embeds_and_upserts_points() -> None:
    embedder = _EmbedderStub([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    vector_db = _VectorDbStub([])
    service = RetrieverService(config=_config(), embedder=embedder, vector_db=vector_db)

    result = service.upsert_texts(
        UpsertTextsRequest(
            namespace="docs",
            tenant_id="tenant-a",
            workspace_id="ws-1",
            entries=[
                UpsertTextEntry(
                    document_id="readme",
                    text="Kubernetes deployment uses local overlay manifests.",
                    source_uri="README.md#kubernetes",
                    metadata={"repo": "agentic-workflow-studio"},
                ),
                UpsertTextEntry(
                    document_id="guide",
                    text="Workflow Studio supports saved drafts and versions.",
                    source_uri="docs/user-guide.md#studio",
                ),
            ],
        )
    )

    assert embedder.calls == [
        [
            "Kubernetes deployment uses local overlay manifests.",
            "Workflow Studio supports saved drafts and versions.",
        ]
    ]
    assert result.upserted_count == 2
    assert len(vector_db.upsert_calls) == 1
    upsert_points = vector_db.upsert_calls[0]["points"]
    assert upsert_points[0]["payload"]["namespace"] == "docs"
    assert upsert_points[0]["payload"]["tenant_id"] == "tenant-a"
    assert upsert_points[0]["payload"]["repo"] == "agentic-workflow-studio"
    assert upsert_points[1]["payload"]["workspace_id"] == "ws-1"
    assert len(result.chunk_ids) == 2


def test_upsert_texts_requires_scope_when_configured() -> None:
    service = RetrieverService(
        config=_config(require_scope=True),
        embedder=_EmbedderStub([[0.1, 0.2, 0.3]]),
        vector_db=_VectorDbStub([]),
    )

    try:
        service.upsert_texts(
            UpsertTextsRequest(
                entries=[
                    UpsertTextEntry(
                        document_id="doc-1",
                        text="A chunk without tenant or workspace scope.",
                    )
                ]
            )
        )
    except RetrieverError as exc:
        assert exc.detail == "missing_scope:one_of_tenant_id_user_id_workspace_id_required"
    else:  # pragma: no cover
        raise AssertionError("Expected RetrieverError")
