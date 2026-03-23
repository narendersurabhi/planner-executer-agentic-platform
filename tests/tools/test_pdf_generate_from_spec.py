from __future__ import annotations

from pathlib import Path

import pytest

from libs.core.tool_registry import ToolExecutionError, ToolRegistry
from libs.tools.pdf_render_from_spec import register_pdf_tools


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
        registry.get("pdf_render_from_spec").handler(
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


def test_missing_path_is_rejected() -> None:
    registry = _make_registry()
    call = registry.execute(
        "pdf_render_from_spec",
        {"document_spec": {"blocks": [{"type": "paragraph", "text": "Hello"}]}},
        "id",
        "trace",
    )
    assert call.status == "failed"
    assert "input schema validation failed" in call.output_or_error["error"]
