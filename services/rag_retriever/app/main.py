from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.mcp import create_mcp_asgi_app
from libs.core import logging as core_logging
from rag_retriever_core import (
    EnsureCollectionRequest,
    EnsureCollectionResponse,
    IndexWorkspaceFileRequest,
    IndexWorkspaceFileResponse,
    RetrieverError,
    RetrieveRequest,
    RetrieveResponse,
    UpsertTextsRequest,
    UpsertTextsResponse,
    build_service_from_env,
)


core_logging.configure_logging("rag-retriever")
LOGGER = core_logging.get_logger("rag-retriever")
RETRIEVER_SERVICE = build_service_from_env()

app = FastAPI(title="Agentic RAG Retriever Service")
app.state.retriever_service = RETRIEVER_SERVICE
app.state.retriever_logger = LOGGER
app.mount("/metrics", make_asgi_app())
MCP_APP, MCP_SESSION_MANAGER = create_mcp_asgi_app(RETRIEVER_SERVICE, LOGGER)
app.mount("/mcp", MCP_APP)
app.mount("/mcp/rpc", MCP_APP)
app.mount("/mcp/rpc/mcp", MCP_APP)


@app.middleware("http")
async def _optional_bearer_auth(request: Request, call_next):
    expected = os.getenv("RAG_RETRIEVER_MCP_TOKEN", "").strip()
    if not expected:
        return await call_next(request)
    path = request.url.path
    protected = path.startswith("/mcp") or path == "/retrieve"
    if not protected:
        return await call_next(request)
    actual = request.headers.get("Authorization", "").strip()
    if actual != f"Bearer {expected}":
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


@app.on_event("startup")
async def _startup_mcp_session_manager() -> None:
    try:
        ensured = RETRIEVER_SERVICE.ensure_default_collection()
        if ensured is not None:
            LOGGER.info(
                "rag_default_collection_ready",
                extra={
                    "collection_name": ensured.collection_name,
                    "status": ensured.status,
                    "vector_size": ensured.vector_size,
                },
            )
    except RetrieverError as exc:
        LOGGER.warning("rag_default_collection_not_ready", extra={"error": exc.detail})
    session_cm = MCP_SESSION_MANAGER.run()
    app.state._mcp_session_cm = session_cm
    await session_cm.__aenter__()


@app.on_event("shutdown")
async def _shutdown_mcp_session_manager() -> None:
    session_cm = getattr(app.state, "_mcp_session_cm", None)
    if session_cm is not None:
        await session_cm.__aexit__(None, None, None)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest) -> RetrieveResponse:
    try:
        return RETRIEVER_SERVICE.retrieve(request)
    except RetrieverError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/collections/ensure", response_model=EnsureCollectionResponse)
def ensure_collection_endpoint(request: EnsureCollectionRequest) -> EnsureCollectionResponse:
    try:
        return RETRIEVER_SERVICE.ensure_collection(request)
    except RetrieverError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/index/upsert_texts", response_model=UpsertTextsResponse)
def upsert_texts_endpoint(request: UpsertTextsRequest) -> UpsertTextsResponse:
    try:
        return RETRIEVER_SERVICE.upsert_texts(request)
    except RetrieverError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/index/workspace_file", response_model=IndexWorkspaceFileResponse)
def index_workspace_file_endpoint(request: IndexWorkspaceFileRequest) -> IndexWorkspaceFileResponse:
    try:
        return RETRIEVER_SERVICE.index_workspace_file(request)
    except RetrieverError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
