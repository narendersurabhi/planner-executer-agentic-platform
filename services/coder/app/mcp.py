from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

from coder_core import CoderError, generate_code as run_generate_code
from coder_core.models import CodeGenRequest


def create_mcp_asgi_app(provider: Any, logger: Any):
    default_hosts = [
        "coder",
        "coder:8000",
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
        "agentic-coder",
        transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
    )

    @mcp.tool()
    def generate_code(
        goal: str, files: Optional[List[str]] = None, constraints: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            req = CodeGenRequest(goal=goal, files=files, constraints=constraints)
            result = run_generate_code(req, provider, logger)
        except CoderError as exc:
            raise RuntimeError(exc.detail) from exc
        return result.model_dump()

    mcp_app = mcp.streamable_http_app()
    session_manager = mcp_app.routes[0].app.session_manager
    return mcp_app, session_manager
