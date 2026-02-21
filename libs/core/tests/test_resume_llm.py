from libs.tools import resume_llm


def test_apply_resume_target_pages_policy_one_page_caps_content() -> None:
    spec = {
        "content": [
            {"type": "section_heading", "text": "SUMMARY"},
            {
                "type": "paragraph",
                "text": "Sentence one. Sentence two. Sentence three.",
            },
            {"type": "section_heading", "text": "EXPERIENCE"},
            {
                "type": "role",
                "bullets": ["a", "b", "c", "d", "e", "f"],
            },
            {
                "type": "role",
                "bullets": ["g", "h", "i", "j"],
            },
        ]
    }
    resume_llm.apply_resume_target_pages_policy(spec, 1)
    assert spec["page"]["margins_in"]["left"] == 0.45
    assert spec["defaults"]["font_size_pt"] == 10.5
    assert spec["content"][3]["bullets"] == ["a", "b", "c", "d"]
    assert spec["content"][4]["bullets"] == ["g", "h"]


def test_ensure_certifications_section_content_fills_from_tailored_resume() -> None:
    spec = {
        "content": [
            {"type": "section_heading", "text": "CERTIFICATIONS"},
            {"type": "bullets", "items": []},
        ]
    }
    tailored_resume = {
        "certifications": [
            {
                "name": "AWS Solutions Architect",
                "issuer": "Amazon",
                "year": 2024,
            }
        ]
    }
    resume_llm.ensure_certifications_section_content(
        spec,
        tailored_resume=tailored_resume,
        tailored_text=None,
        candidate_resume_text=None,
    )
    assert spec["content"][1]["items"][0].startswith("AWS Solutions Architect")


def test_ensure_certifications_section_content_removes_empty_section_when_no_fallback() -> None:
    spec = {
        "content": [
            {"type": "section_heading", "text": "CERTIFICATIONS"},
            {"type": "bullets", "items": []},
            {"type": "section_heading", "text": "EDUCATION"},
        ]
    }
    resume_llm.ensure_certifications_section_content(
        spec,
        tailored_resume=None,
        tailored_text=None,
        candidate_resume_text=None,
    )
    assert spec["content"][0]["text"] == "EDUCATION"


def test_parse_target_pages_phrases() -> None:
    assert resume_llm.parse_target_pages("2 pages") == 2
    assert resume_llm.parse_target_pages("page 1") == 1
    assert resume_llm.parse_target_pages("three pages") is None
