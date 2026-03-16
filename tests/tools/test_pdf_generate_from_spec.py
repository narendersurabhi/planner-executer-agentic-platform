from __future__ import annotations

from pathlib import Path

import pytest

from libs.core.tool_registry import ToolExecutionError, ToolRegistry
from libs.tools.pdf_generate_from_spec import register_pdf_tools


def _artifact_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_pdf_tools(registry)
    return registry


def test_invalid_validation_report_blocks_pdf_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(_artifact_dir(tmp_path)))
    registry = _make_registry()
    with pytest.raises(ToolExecutionError, match="document_spec validation failed"):
        registry.get("pdf_generate_from_spec").handler(
            {
                "document_spec": {
                    "blocks": [{"type": "paragraph", "text": "Hello"}],
                },
                "path": "tests/blocked.pdf",
                "validation_report": {
                    "valid": False,
                    "errors": [
                        {
                            "path": "/blocks/0/text",
                            "message": "text/paragraph requires text: string",
                        }
                    ],
                },
            }
        )
