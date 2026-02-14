from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from libs.core import logging as core_logging
from app.mcp import create_mcp_asgi_app
from tailor_core import (
    TailorError,
    create_provider_from_env,
    improve_resume,
    improve_resume_iterative,
    tailor_resume,
)


core_logging.configure_logging("tailor")
LOGGER = core_logging.get_logger("tailor")

LLM_PROVIDER_INSTANCE = create_provider_from_env()


class TailorRequest(BaseModel):
    job: Dict[str, Any] = Field(default_factory=dict)
    memory: Optional[Dict[str, Any]] = None


class TailorResponse(BaseModel):
    tailored_resume: Dict[str, Any]


class ImproveRequest(BaseModel):
    tailored_resume: Optional[Dict[str, Any]] = None
    tailored_text: Optional[str] = None
    job: Dict[str, Any] = Field(default_factory=dict)
    memory: Optional[Dict[str, Any]] = None


class ImproveResponse(BaseModel):
    tailored_resume: Dict[str, Any]
    alignment_score: float
    alignment_summary: str


class ImproveIterativeRequest(BaseModel):
    tailored_resume: Optional[Dict[str, Any]] = None
    tailored_text: Optional[str] = None
    job: Dict[str, Any] = Field(default_factory=dict)
    memory: Optional[Dict[str, Any]] = None
    min_alignment_score: float = 85
    max_iterations: int = 2


class ImproveIterativeResponse(BaseModel):
    tailored_resume: Dict[str, Any]
    alignment_score: float
    alignment_summary: str
    iterations: int
    reached_threshold: bool
    history: List[Dict[str, Any]]


app = FastAPI(title="Resume Tailoring Service")
app.state.tailor_provider = LLM_PROVIDER_INSTANCE
MCP_APP, MCP_SESSION_MANAGER = create_mcp_asgi_app(LLM_PROVIDER_INSTANCE)
app.mount("/mcp/rpc", MCP_APP)


@app.on_event("startup")
async def _startup_mcp_session_manager() -> None:
    session_cm = MCP_SESSION_MANAGER.run()
    app.state._mcp_session_cm = session_cm
    await session_cm.__aenter__()


@app.on_event("shutdown")
async def _shutdown_mcp_session_manager() -> None:
    session_cm = getattr(app.state, "_mcp_session_cm", None)
    if session_cm is not None:
        await session_cm.__aexit__(None, None, None)


def _http_error(error: TailorError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.detail)


@app.post("/tailor", response_model=TailorResponse)
def tailor_resume_endpoint(request: TailorRequest) -> TailorResponse:
    try:
        tailored_resume = tailor_resume(request.job, request.memory, LLM_PROVIDER_INSTANCE)
    except TailorError as exc:
        raise _http_error(exc) from exc
    return TailorResponse(tailored_resume=tailored_resume)


@app.post("/improve", response_model=ImproveResponse)
def improve_resume_endpoint(request: ImproveRequest) -> ImproveResponse:
    try:
        result = improve_resume(
            tailored_resume=request.tailored_resume,
            tailored_text=request.tailored_text,
            job=request.job,
            memory=request.memory,
            provider=LLM_PROVIDER_INSTANCE,
        )
    except TailorError as exc:
        raise _http_error(exc) from exc
    return ImproveResponse(
        tailored_resume=result["tailored_resume"],
        alignment_score=float(result["alignment_score"]),
        alignment_summary=str(result["alignment_summary"]),
    )


@app.post("/improve-iterative", response_model=ImproveIterativeResponse)
def improve_resume_iterative_endpoint(request: ImproveIterativeRequest) -> ImproveIterativeResponse:
    try:
        result = improve_resume_iterative(
            tailored_resume=request.tailored_resume,
            tailored_text=request.tailored_text,
            job=request.job,
            memory=request.memory,
            min_alignment_score=request.min_alignment_score,
            max_iterations=request.max_iterations,
            provider=LLM_PROVIDER_INSTANCE,
        )
    except TailorError as exc:
        raise _http_error(exc) from exc

    return ImproveIterativeResponse(
        tailored_resume=result["tailored_resume"],
        alignment_score=float(result["alignment_score"]),
        alignment_summary=str(result["alignment_summary"]),
        iterations=int(result["iterations"]),
        reached_threshold=bool(result["reached_threshold"]),
        history=list(result["history"]),
    )
