from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.mcp import create_mcp_asgi_app
from coder_core import CoderError, create_provider_from_env, generate_code
from coder_core.models import CodeGenRequest, CodeGenResponse
from libs.core import logging as core_logging


core_logging.configure_logging("coder")
LOGGER = core_logging.get_logger("coder")

LLM_PROVIDER_INSTANCE = create_provider_from_env()

app = FastAPI(title="Agentic Coder Service")
app.state.coder_provider = LLM_PROVIDER_INSTANCE
app.state.coder_logger = LOGGER
MCP_APP, MCP_SESSION_MANAGER = create_mcp_asgi_app(LLM_PROVIDER_INSTANCE, LOGGER)
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


@app.post("/generate", response_model=CodeGenResponse)
def generate_code_endpoint(request: CodeGenRequest) -> CodeGenResponse:
    try:
        return generate_code(request, LLM_PROVIDER_INSTANCE, LOGGER)
    except CoderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
