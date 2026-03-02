from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from libs.core import capability_registry
from libs.core import mcp_gateway
from libs.framework.tool_runtime import ToolExecutionError


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_mcp_server_registry_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp_servers.json"
    _write_json(
        config_path,
        {
            "mcp_servers": {
                "github_remote": {
                    "base_url": "https://mcp.example.com",
                    "route_paths": ["/", "/mcp"],
                    "timeout_s": 30,
                    "max_retries": 2,
                    "retry_sleep_s": 0.5,
                    "headers": {"X-Env": "test"},
                    "bearer_env": "GITHUB_TOKEN",
                    "enabled": True,
                }
            }
        },
    )
    registry = mcp_gateway.load_mcp_server_registry(config_path)
    server = registry.require("github_remote")
    assert server.base_url == "https://mcp.example.com"
    assert server.route_paths == ("/", "/mcp")
    assert server.max_retries == 2


def test_invoke_capability_maps_and_calls_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    cap_registry = capability_registry.CapabilityRegistry(
        capabilities={
            "github.repo.list": capability_registry.CapabilitySpec(
                capability_id="github.repo.list",
                description="List repos",
                risk_tier="read_only",
                idempotency="read",
                adapters=(
                    capability_registry.CapabilityAdapterSpec(
                        type="mcp",
                        server_id="github_remote",
                        tool_name="github_repo_list",
                        route_paths=("/",),
                        timeout_s=10,
                        max_retries=1,
                        retry_sleep_s=0.1,
                        arg_map={"visibility": "scope"},
                        response_path=None,
                        enabled=True,
                    ),
                ),
                enabled=True,
            )
        }
    )
    server_registry = mcp_gateway.MCPServerRegistry(
        servers={
            "github_remote": mcp_gateway.MCPServerSpec(
                server_id="github_remote",
                base_url="https://mcp.example.com",
                route_paths=("/mcp",),
                timeout_s=20,
                max_retries=2,
                retry_sleep_s=0.2,
                headers={"X-Test": "1"},
                bearer_env="GITHUB_TOKEN",
                enabled=True,
            )
        }
    )
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    captured: dict[str, Any] = {}

    def fake_post_tool_call(service_url: str, tool_name: str, arguments: dict, **kwargs: Any) -> dict:
        captured["service_url"] = service_url
        captured["tool_name"] = tool_name
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return {"ok": True}

    result = mcp_gateway.invoke_capability(
        "github.repo.list",
        {"scope": "private", "unused": "value"},
        capability_registry=cap_registry,
        server_registry=server_registry,
        post_tool_call=fake_post_tool_call,
    )
    assert result == {"ok": True}
    assert captured["service_url"] == "https://mcp.example.com"
    assert captured["tool_name"] == "github_repo_list"
    assert captured["arguments"] == {"visibility": "private"}
    assert captured["kwargs"]["route_paths"] == ("/",)
    assert captured["kwargs"]["timeout_s_override"] == 10
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["kwargs"]["headers"]["X-Test"] == "1"


def test_invoke_capability_raises_when_all_adapters_fail() -> None:
    cap_registry = capability_registry.CapabilityRegistry(
        capabilities={
            "github.repo.list": capability_registry.CapabilitySpec(
                capability_id="github.repo.list",
                description="List repos",
                risk_tier="read_only",
                idempotency="read",
                adapters=(
                    capability_registry.CapabilityAdapterSpec(
                        type="mcp",
                        server_id="missing_server",
                        tool_name="github_repo_list",
                    ),
                ),
                enabled=True,
            )
        }
    )
    server_registry = mcp_gateway.MCPServerRegistry(servers={})
    with pytest.raises(ToolExecutionError) as exc:
        mcp_gateway.invoke_capability(
            "github.repo.list",
            {},
            capability_registry=cap_registry,
            server_registry=server_registry,
        )
    assert "mcp_gateway_all_adapters_failed" in str(exc.value)


def test_invoke_capability_with_native_tool_adapter() -> None:
    cap_registry = capability_registry.CapabilityRegistry(
        capabilities={
            "document.docx.generate": capability_registry.CapabilitySpec(
                capability_id="document.docx.generate",
                description="Generate DOCX from document spec",
                risk_tier="read_only",
                idempotency="read",
                adapters=(
                    capability_registry.CapabilityAdapterSpec(
                        type="tool",
                        server_id="local_worker",
                        tool_name="docx_generate_from_spec",
                        arg_map={"path": "output_path", "document_spec": "document_spec"},
                    ),
                ),
                enabled=True,
            )
        }
    )

    def _execute_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert tool_name == "docx_generate_from_spec"
        assert payload["path"] == "documents/out.docx"
        assert payload["document_spec"] == {"blocks": []}
        return {"path": "/shared/artifacts/documents/out.docx", "bytes_written": 123}

    result = mcp_gateway.invoke_capability(
        "document.docx.generate",
        {"output_path": "documents/out.docx", "document_spec": {"blocks": []}},
        capability_registry=cap_registry,
        execute_tool=_execute_tool,
    )
    assert result["path"] == "/shared/artifacts/documents/out.docx"
