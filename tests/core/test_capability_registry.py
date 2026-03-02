from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.core import capability_registry


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_capability_registry_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "capabilities.json"
    _write_json(
        config_path,
        {
            "capabilities": [
                {
                    "id": "docs.generate",
                    "description": "Generate document spec",
                    "risk_tier": "read_only",
                    "idempotency": "read",
                    "group": "docs",
                    "subgroup": "specs",
                    "adapters": [
                        {
                            "type": "mcp",
                            "server_id": "local_docs",
                            "tool_name": "llm_generate_document_spec",
                            "route_paths": ["/mcp/rpc/mcp", "mcp/rpc"],
                        }
                    ],
                }
            ]
        },
    )

    registry = capability_registry.load_capability_registry(config_path)
    spec = registry.require("docs.generate")
    assert spec.description == "Generate document spec"
    assert spec.group == "docs"
    assert spec.subgroup == "specs"
    assert len(spec.adapters) == 1
    assert spec.adapters[0].route_paths == ("/mcp/rpc/mcp", "/mcp/rpc")


def test_load_capability_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "capabilities.json"
    _write_json(
        config_path,
        {
            "capabilities": [
                {"id": "dup", "description": "a", "adapters": []},
                {"id": "dup", "description": "b", "adapters": []},
            ]
        },
    )
    with pytest.raises(capability_registry.CapabilityRegistryError):
        capability_registry.load_capability_registry(config_path)


def test_resolve_capability_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAPABILITY_MODE", raising=False)
    assert capability_registry.resolve_capability_mode() == "disabled"
    monkeypatch.setenv("CAPABILITY_MODE", "enabled")
    assert capability_registry.resolve_capability_mode() == "enabled"
    monkeypatch.setenv("CAPABILITY_MODE", "weird")
    assert capability_registry.resolve_capability_mode() == "disabled"
