import os
from pathlib import Path

import pytest
from docx import Document

from libs.core.tool_registry import default_registry


def test_docx_render_renders_docx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / "basic.docx"
    doc = Document()
    doc.add_paragraph("{{ full_name }}")
    doc.save(str(template_path))

    monkeypatch.setenv("DOCX_TEMPLATE_DIR", str(template_dir))
    try:
        os.makedirs("/shared/artifacts", exist_ok=True)
    except OSError:
        pytest.skip("Cannot create /shared/artifacts in test environment")

    registry = default_registry()
    payload = {
        "data": {
            "full_name": "Ada Lovelace",
            "summary": "Pioneer",
            "skills": [],
            "experience": [],
            "education": [],
        },
        "template_id": "basic",
        "output_path": "docx_test.docx",
    }
    call = registry.execute("docx_render", payload, "idempotency", "trace")
    assert call.status == "completed"
    output_path = call.output_or_error["path"]
    assert os.path.exists(output_path)
