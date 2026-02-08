from libs.core.tool_registry import ToolExecutionError, _ensure_required_resume_sections


def _valid_resume():
    return {
        "header": {
            "name": "Jane Doe",
            "location": "Okemos, MI",
            "phone": "+1 (555) 555-5555",
            "email": "jane@example.com",
            "linkedin": "https://www.linkedin.com/in/janedoe",
            "github": "https://github.com/janedoe",
        },
        "summary": "Senior engineer with relevant experience.",
        "skills": [{"term": "Python", "definition": "Backend services"}],
        "experience": [
            {
                "company": "Acme",
                "title": "Engineer",
                "location": "Remote",
                "dates": "2020-2024",
                "bullets": ["Built reliable services with measurable impact."],
            }
        ],
        "education": [],
        "certifications": [],
        "credibility_gate": [],
        "one_page_selection": [],
    }


def test_required_resume_sections_accepts_valid_payload() -> None:
    _ensure_required_resume_sections(_valid_resume())


def test_required_resume_sections_rejects_missing_header() -> None:
    resume = _valid_resume()
    resume["header"]["name"] = "Unknown"
    try:
        _ensure_required_resume_sections(resume)
    except ToolExecutionError as exc:
        assert "tailored_resume_missing_header:name" in str(exc)
    else:
        raise AssertionError("Expected missing header error")


def test_required_resume_sections_rejects_unknown_experience() -> None:
    resume = _valid_resume()
    resume["experience"][0]["company"] = "Unknown"
    try:
        _ensure_required_resume_sections(resume)
    except ToolExecutionError as exc:
        assert "tailored_resume_missing_experience:0.company" in str(exc)
    else:
        raise AssertionError("Expected missing experience error")


def test_required_resume_sections_rejects_missing_bullets() -> None:
    resume = _valid_resume()
    resume["experience"][0]["bullets"] = []
    try:
        _ensure_required_resume_sections(resume)
    except ToolExecutionError as exc:
        assert "tailored_resume_invalid_experience:0.bullets" in str(exc)
    else:
        raise AssertionError("Expected missing bullets error")
