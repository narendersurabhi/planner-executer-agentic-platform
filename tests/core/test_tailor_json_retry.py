from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAILOR_SERVICE_ROOT = ROOT / "services" / "tailor"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TAILOR_SERVICE_ROOT))
from tailor_core.service import improve_resume, tailor_resume  # type: ignore  # noqa: E402


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    def generate(self, _prompt: str) -> _FakeLLMResponse:
        if not self._outputs:
            raise RuntimeError("no_more_outputs")
        return _FakeLLMResponse(self._outputs.pop(0))


def _valid_tailored_resume() -> dict:
    return {
        "schema_version": "1.0",
        "header": {
            "name": "Jane Doe",
            "title": "Staff Software Engineer, AI/ML",
            "location": "Okemos, MI",
            "phone": "+1 (555) 555-5555",
            "email": "jane@example.com",
            "links": {
                "linkedin": "https://www.linkedin.com/in/jane",
                "github": "https://github.com/jane",
            },
        },
        "summary": "Experienced engineer shipping production ML systems.",
        "skills": [{"group_name": "Languages", "items": ["Python", "SQL"]}],
        "experience": [
            {
                "company": "Acentra Health",
                "title": "Software Engineer, AI/ML",
                "location": "Okemos, MI",
                "dates": "Dec 2016 - Present",
                "bullets": ["Built AI-powered workflow systems with measurable impact."],
            }
        ],
        "education": [
            {
                "degree": "BS",
                "school": "Nizam College",
                "location": "Hyderabad, India",
                "dates": "2003 - 2007",
            }
        ],
        "certifications": [{"name": "AWS Certified AI Practitioner", "issuer": "AWS", "year": "2025"}],
    }


def test_tailor_resume_retries_when_first_response_has_invalid_json() -> None:
    valid = _valid_tailored_resume()
    provider = _FakeProvider(
        outputs=[
            '{"schema_version":"1.0","header":{"name":"Jane Doe"}',  # malformed json
            json.dumps(valid),
        ]
    )
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": "candidate resume text",
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert result["schema_version"] == "1.0"
    assert result["header"]["name"] == "Jane Doe"


def test_improve_resume_retries_when_first_response_has_invalid_json() -> None:
    current_resume = _valid_tailored_resume()
    improved_resume = _valid_tailored_resume()
    improved_resume["summary"] = "Staff engineer improving healthcare workflows with AI and ML."
    provider = _FakeProvider(
        outputs=[
            '{"tailored_resume": {"summary":"x"}',  # malformed json
            json.dumps(
                {
                    "tailored_resume": improved_resume,
                    "alignment_score": 99,
                    "alignment_summary": "Strong alignment with role requirements.",
                }
            ),
        ]
    )

    result = improve_resume(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        provider=provider,
    )

    assert result["alignment_score"] == 99.0
    assert result["tailored_resume"]["summary"].startswith("Staff engineer")


def test_tailor_resume_falls_back_header_title_from_target_role_name() -> None:
    resume = _valid_tailored_resume()
    resume["header"]["title"] = ""
    provider = _FakeProvider(outputs=[json.dumps(resume)])
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": "candidate resume text",
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert result["header"]["title"] == "Staff Software Engineer, AI/ML"


def test_tailor_resume_retries_on_missing_header_title_error_payload() -> None:
    valid = _valid_tailored_resume()
    provider = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "error": "missing_required_fields",
                    "missing_fields": ["header.title"],
                }
            ),
            json.dumps(valid),
        ]
    )
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": "candidate resume text",
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert result["header"]["title"] == valid["header"]["title"]


def test_tailor_resume_retries_when_required_sections_are_missing() -> None:
    partial = {
        "schema_version": "1.0",
        "header": {
            "name": "Jane Doe",
            "title": "Staff Software Engineer, AI/ML",
            "location": "Okemos, MI",
            "phone": "+1 (555) 555-5555",
            "email": "jane@example.com",
        },
        "summary": "Staff engineer building production ML systems.",
        "skills": [{"group_name": "Languages", "items": ["Python", "SQL"]}],
    }
    valid = _valid_tailored_resume()
    provider = _FakeProvider(outputs=[json.dumps(partial), json.dumps(valid)])
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": "candidate resume text",
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert len(result["experience"]) > 0
    assert isinstance(result["education"], list)
    assert isinstance(result["certifications"], list)


def test_tailor_resume_defaults_optional_sections_when_only_education_and_certs_missing() -> None:
    payload = _valid_tailored_resume()
    payload.pop("education")
    payload.pop("certifications")
    provider = _FakeProvider(outputs=[json.dumps(payload)])
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": "candidate resume text",
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert result["education"] == []
    assert result["certifications"] == []
