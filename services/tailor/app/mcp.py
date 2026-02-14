from __future__ import annotations

import os
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

from tailor_core import (
    TailorError,
    improve_resume as run_improve_resume,
    improve_resume_iterative as run_improve_resume_iterative,
    tailor_resume as run_tailor_resume,
)


def create_mcp_asgi_app(provider: Any):
    default_hosts = [
        "tailor",
        "tailor:8000",
        "localhost",
        "localhost:8000",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:8000",
        "127.0.0.1:*",
    ]
    raw_allowed_hosts = os.getenv("MCP_ALLOWED_HOSTS", "")
    allowed_hosts = [h.strip() for h in raw_allowed_hosts.split(",") if h.strip()] or default_hosts
    mcp = FastMCP(
        "agentic-tailor",
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @mcp.tool()
    def tailor_resume(job: Dict[str, Any], memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            return {"tailored_resume": run_tailor_resume(job or {}, memory, provider)}
        except TailorError as exc:
            raise RuntimeError(exc.detail) from exc

    @mcp.tool()
    def improve_resume(
        tailored_resume: Dict[str, Any] | None = None,
        tailored_text: str | None = None,
        job: Dict[str, Any] | None = None,
        memory: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        try:
            return run_improve_resume(
                tailored_resume=tailored_resume,
                tailored_text=tailored_text,
                job=job or {},
                memory=memory,
                provider=provider,
            )
        except TailorError as exc:
            raise RuntimeError(exc.detail) from exc

    @mcp.tool()
    def improve_iterative(
        tailored_resume: Dict[str, Any] | None = None,
        tailored_text: str | None = None,
        job: Dict[str, Any] | None = None,
        memory: Dict[str, Any] | None = None,
        min_alignment_score: float = 85,
        max_iterations: int = 2,
    ) -> Dict[str, Any]:
        try:
            return run_improve_resume_iterative(
                tailored_resume=tailored_resume,
                tailored_text=tailored_text,
                job=job or {},
                memory=memory,
                min_alignment_score=min_alignment_score,
                max_iterations=max_iterations,
                provider=provider,
            )
        except TailorError as exc:
            raise RuntimeError(exc.detail) from exc

    mcp_app = mcp.streamable_http_app()
    session_manager = mcp_app.routes[0].app.session_manager
    return mcp_app, session_manager
