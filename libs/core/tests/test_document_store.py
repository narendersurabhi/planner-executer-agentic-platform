from __future__ import annotations

from pathlib import Path

import pytest

from libs.core import document_store


def test_artifact_relative_path_normalizes_artifacts_prefix() -> None:
    assert (
        document_store.artifact_relative_path("artifacts/documents/output.docx")
        == "documents/output.docx"
    )


def test_artifact_relative_path_rejects_parent_traversal() -> None:
    with pytest.raises(document_store.DocumentStoreError):
        document_store.artifact_relative_path("../secrets.txt")


def test_artifact_local_path_under_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    target = document_store.artifact_local_path("documents/a.docx")
    assert str(target).startswith(str(tmp_path.resolve()))
    assert target.as_posix().endswith("documents/a.docx")

