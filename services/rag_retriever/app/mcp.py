from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

from rag_retriever_core import (
    EnsureCollectionRequest,
    IndexWorkspaceFileRequest,
    RetrieverError,
    RetrieveRequest,
    UpsertTextEntry,
    UpsertTextsRequest,
)


def create_mcp_asgi_app(service: Any, logger: Any):
    default_hosts = [
        "rag-retriever-mcp",
        "rag-retriever-mcp:8086",
        "localhost",
        "localhost:8086",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:8086",
        "127.0.0.1:*",
    ]
    raw_allowed_hosts = os.getenv("MCP_ALLOWED_HOSTS", "")
    allowed_hosts = [h.strip() for h in raw_allowed_hosts.split(",") if h.strip()] or default_hosts
    mcp = FastMCP(
        "agentic-rag-retriever",
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @mcp.tool()
    def retrieve(
        query: str,
        top_k: int | None = None,
        namespace: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        filters: dict[str, Any] | None = None,
        min_score: float | None = None,
        include_text: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        try:
            request = RetrieveRequest(
                query=query,
                top_k=top_k,
                namespace=namespace,
                tenant_id=tenant_id,
                user_id=user_id,
                workspace_id=workspace_id,
                filters=filters,
                min_score=min_score,
                include_text=include_text,
                include_metadata=include_metadata,
            )
            result = service.retrieve(request)
        except RetrieverError as exc:
            logger.error("rag_retrieve_failed", extra={"error": exc.detail})
            raise RuntimeError(exc.detail) from exc
        return result.model_dump()

    @mcp.tool()
    def ensure_collection(
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str | None = None,
        vector_name: str | None = None,
        create_payload_indexes: bool = True,
        payload_keyword_fields: list[str] | None = None,
        on_disk_payload: bool = True,
    ) -> dict[str, Any]:
        try:
            result = service.ensure_collection(
                EnsureCollectionRequest(
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance=distance,
                    vector_name=vector_name,
                    create_payload_indexes=create_payload_indexes,
                    payload_keyword_fields=payload_keyword_fields,
                    on_disk_payload=on_disk_payload,
                )
            )
        except RetrieverError as exc:
            logger.error("rag_ensure_collection_failed", extra={"error": exc.detail})
            raise RuntimeError(exc.detail) from exc
        return result.model_dump()

    @mcp.tool()
    def upsert_texts(
        entries: list[dict[str, Any]],
        collection_name: str | None = None,
        ensure_collection: bool = True,
        namespace: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            normalized_entries = [UpsertTextEntry(**entry) for entry in entries]
            result = service.upsert_texts(
                UpsertTextsRequest(
                    collection_name=collection_name,
                    ensure_collection=ensure_collection,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    entries=normalized_entries,
                )
            )
        except RetrieverError as exc:
            logger.error("rag_upsert_texts_failed", extra={"error": exc.detail})
            raise RuntimeError(exc.detail) from exc
        return result.model_dump()

    @mcp.tool()
    def index_workspace_file(
        path: str,
        collection_name: str | None = None,
        ensure_collection: bool = True,
        document_id: str | None = None,
        source_uri: str | None = None,
        namespace: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        chunk_size_chars: int | None = None,
        chunk_overlap_chars: int | None = None,
        max_chunks: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            result = service.index_workspace_file(
                IndexWorkspaceFileRequest(
                    path=path,
                    collection_name=collection_name,
                    ensure_collection=ensure_collection,
                    document_id=document_id,
                    source_uri=source_uri,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    chunk_size_chars=chunk_size_chars,
                    chunk_overlap_chars=chunk_overlap_chars,
                    max_chunks=max_chunks,
                    metadata=metadata or {},
                )
            )
        except RetrieverError as exc:
            logger.error("rag_index_workspace_file_failed", extra={"error": exc.detail})
            raise RuntimeError(exc.detail) from exc
        return result.model_dump()

    mcp_app = mcp.streamable_http_app()
    session_manager = mcp_app.routes[0].app.session_manager
    return mcp_app, session_manager
