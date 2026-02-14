from __future__ import annotations

import json
from pathlib import Path


def _contract_dirs() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    return [
        root / "services" / "coder" / "contracts" / "v1",
        root / "services" / "tailor" / "contracts" / "v1",
    ]


def test_contract_json_files_are_valid() -> None:
    for directory in _contract_dirs():
        assert directory.exists(), f"missing contract directory: {directory}"
        files = sorted(directory.glob("*.json"))
        assert files, f"no contract files in: {directory}"
        for file_path in files:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            assert isinstance(data, dict), f"contract is not object: {file_path}"
            assert "$schema" in data, f"missing $schema in: {file_path}"
            assert "$id" in data, f"missing $id in: {file_path}"
