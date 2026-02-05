from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document

from libs.core.tool_registry import ToolExecutionError, ToolRegistry
from libs.tools.docx_generate import register_docx_tools


def _artifact_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_fixture(name: str) -> dict:
    path = Path("tests/fixtures") / name
    return json.loads(path.read_text(encoding="utf-8"))


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_docx_tools(registry)
    return registry


def test_generates_docx_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(_artifact_dir(tmp_path)))
    registry = _make_registry()
    spec = _load_fixture("resume_ats_single_column_spec.json")
    payload = {"document_spec": spec, "path": "tests/resume.docx"}
    call = registry.execute("docx_generate_from_spec", payload, "id", "trace")
    assert call.status == "completed"
    output_path = Path(call.output_or_error["path"])
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_path_traversal_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(_artifact_dir(tmp_path)))
    registry = _make_registry()
    spec = _load_fixture("resume_ats_single_column_spec.json")
    with pytest.raises(ToolExecutionError):
        registry.get("docx_generate_from_spec").handler(
            {"document_spec": spec, "path": "../evil.docx"}
        )


def test_missing_path_fails_schema_validation() -> None:
    registry = _make_registry()
    spec = _load_fixture("resume_ats_single_column_spec.json")
    call = registry.execute("docx_generate_from_spec", {"document_spec": spec}, "id", "trace")
    assert call.status == "failed"
    assert "input schema validation failed" in call.output_or_error["error"]


def test_strict_unresolved_placeholder_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(_artifact_dir(tmp_path)))
    registry = _make_registry()
    spec = {
        "tokens": {"name": "A"},
        "blocks": [{"type": "paragraph", "text": "{{missing}}"}],
    }
    with pytest.raises(ToolExecutionError):
        registry.get("docx_generate_from_spec").handler(
            {"document_spec": spec, "path": "tests/strict.docx", "strict": True}
        )


def test_repeat_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(_artifact_dir(tmp_path)))
    registry = _make_registry()
    spec = {
        "tokens": {
            "experience": [
                {"company": "Alpha"},
                {"company": "Beta"},
            ]
        },
        "blocks": [
            {
                "type": "repeat",
                "items": "{{experience}}",
                "as": "r",
                "template": [
                    {"type": "paragraph", "text": "{{r.company}}"},
                ],
            }
        ],
    }
    call = registry.execute(
        "docx_generate_from_spec",
        {"document_spec": spec, "path": "tests/repeat.docx"},
        "id",
        "trace",
    )
    assert call.status == "completed"
    doc = Document(call.output_or_error["path"])
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    assert "Alpha" in text
    assert "Beta" in text
