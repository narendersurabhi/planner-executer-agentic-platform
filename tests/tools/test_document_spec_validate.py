from libs.tools.document_spec_validate_old import _document_spec_validate


def _base_spec():
    return {
        "doc_type": "resume",
        "version": "v1",
        "tokens": {
            "full_name": "Ada Lovelace",
            "summary": "Engineer and writer.",
            "experience": [
                {"company": "Analytical Engines Inc", "bullets": ["Built systems."]},
                {"company": "Math Guild", "bullets": ["Published papers."]},
            ],
        },
        "blocks": [
            {"type": "heading", "level": 1, "text": "{{full_name}}"},
            {"type": "heading", "level": 2, "text": "SUMMARY"},
            {"type": "paragraph", "text": "{{summary}}"},
            {
                "type": "repeat",
                "items": "{{experience}}",
                "as": "r",
                "template": [
                    {"type": "paragraph", "text": "{{r.company}}"},
                    {"type": "bullets", "items": "{{r.bullets}}"},
                ],
            },
        ],
    }


def test_valid_spec_returns_valid_true():
    result = _document_spec_validate({"document_spec": _base_spec(), "strict": True})
    assert result["valid"] is True
    assert result["errors"] == []


def test_unknown_block_type_fails():
    spec = _base_spec()
    spec["blocks"].append({"type": "skill_grid", "text": "Nope"})
    result = _document_spec_validate({"document_spec": spec, "strict": True})
    assert result["valid"] is False
    assert any("unsupported block type" in err["message"] for err in result["errors"])


def test_strict_unresolved_placeholder_fails():
    spec = _base_spec()
    spec["blocks"].append({"type": "paragraph", "text": "{{missing}}"})
    result = _document_spec_validate({"document_spec": spec, "strict": True})
    assert result["valid"] is False
    assert any("unresolved placeholder" in err["message"] for err in result["errors"])


def test_non_strict_unresolved_placeholder_warns():
    spec = _base_spec()
    spec["blocks"].append({"type": "paragraph", "text": "{{missing}}"})
    result = _document_spec_validate({"document_spec": spec, "strict": False})
    assert result["valid"] is True
    assert any("unresolved placeholder" in warn["message"] for warn in result["warnings"])


def test_repeat_template_placeholder_allowed():
    spec = _base_spec()
    spec["tokens"]["experience"] = []
    result = _document_spec_validate({"document_spec": spec, "strict": True})
    assert result["valid"] is True
    assert not any("r.company" in err["message"] for err in result["errors"])
