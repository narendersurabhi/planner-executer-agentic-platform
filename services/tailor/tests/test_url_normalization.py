from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tailor_core.service import (  # noqa: E402
    _normalize_url_spacing_in_object,
    _normalize_url_spacing_text,
    _parse_certification_line,
)


def test_normalize_url_spacing_text_removes_space_after_scheme() -> None:
    raw = "See cert at https: //www.credly.com/badge/abc/public_url"
    assert _normalize_url_spacing_text(raw) == "See cert at https://www.credly.com/badge/abc/public_url"


def test_normalize_url_spacing_in_object_is_recursive() -> None:
    payload = {
        "header": {
            "links": {
                "linkedin": "https: //www.linkedin.com/in/example",
                "github": "https: //github.com/example",
            }
        },
        "certifications": [
            {"name": "AWS", "url": "https: //www.credly.com/badges/xyz/public_url"}
        ],
    }
    normalized = _normalize_url_spacing_in_object(payload)
    assert normalized["header"]["links"]["linkedin"] == "https://www.linkedin.com/in/example"
    assert normalized["header"]["links"]["github"] == "https://github.com/example"
    assert (
        normalized["certifications"][0]["url"]
        == "https://www.credly.com/badges/xyz/public_url"
    )


def test_parse_certification_line_handles_spaced_url_scheme() -> None:
    line = (
        "AWS Certified AI Practitioner - Amazon Web Services (2025) | "
        "https: //www.credly.com/badges/d91a3fa3-b52c-4b44-8b5d-5f75f13f58e1/public_url"
    )
    parsed = _parse_certification_line(line)
    assert parsed is not None
    assert parsed["url"] == (
        "https://www.credly.com/badges/d91a3fa3-b52c-4b44-8b5d-5f75f13f58e1/public_url"
    )

