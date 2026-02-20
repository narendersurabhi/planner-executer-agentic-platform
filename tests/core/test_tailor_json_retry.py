from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAILOR_SERVICE_ROOT = ROOT / "services" / "tailor"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TAILOR_SERVICE_ROOT))
from tailor_core.service import (  # type: ignore  # noqa: E402
    improve_resume,
    improve_resume_iterative,
    tailor_resume,
)


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    def __init__(self, outputs: list[object]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, _prompt: str) -> _FakeLLMResponse:
        self.prompts.append(_prompt)
        if not self._outputs:
            raise RuntimeError("no_more_outputs")
        next_item = self._outputs.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return _FakeLLMResponse(str(next_item))


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
        "certifications": [
            {"name": "AWS Certified AI Practitioner", "issuer": "AWS", "year": "2025"}
        ],
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


def test_improve_resume_uses_independent_llm_evaluator_score() -> None:
    current_resume = _valid_tailored_resume()
    improved_resume = _valid_tailored_resume()
    improved_resume["summary"] = "Improved summary from writer model."
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved_resume,
                    "alignment_score": 99,
                    "alignment_summary": "Writer self-score.",
                }
            )
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "alignment_score": 81,
                    "alignment_summary": "Independent evaluator score.",
                }
            )
        ]
    )

    result = improve_resume(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["alignment_score"] == 81.0
    assert result["alignment_summary"] == "Independent evaluator score."


def test_improve_resume_accepts_structured_eval_feedback_without_summary() -> None:
    current_resume = _valid_tailored_resume()
    improved_resume = _valid_tailored_resume()
    improved_resume["summary"] = "Improved summary from writer model."
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved_resume,
                    "alignment_score": 99,
                    "alignment_summary": "Writer self-score.",
                }
            )
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "alignment_score": 84,
                    "top_gaps": ["ASR platform ownership is not explicit."],
                    "must_fix_before_95": ["Add one bullet proving ASR production ownership."],
                    "missing_evidence": ["No metric tied to speech pipeline reliability."],
                    "recommended_edits": ["Add p95 latency and uptime metric for speech API."],
                }
            )
        ]
    )

    result = improve_resume(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["alignment_score"] == 84.0
    assert result["alignment_summary"]
    assert result["alignment_feedback"]["must_fix_before_95"][0].startswith(
        "Add one bullet proving ASR"
    )


def test_improve_iterative_uses_independent_evaluator_for_threshold(monkeypatch) -> None:
    monkeypatch.setenv("TAILOR_MIN_CHANGES_PER_ITERATION", "1")
    current_resume = _valid_tailored_resume()
    improved1 = _valid_tailored_resume()
    improved1["summary"] = "first pass"
    improved2 = _valid_tailored_resume()
    improved2["summary"] = "second pass"
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved1,
                    "alignment_score": 99,
                    "alignment_summary": "writer pass 1",
                }
            ),
            json.dumps(
                {
                    "tailored_resume": improved2,
                    "alignment_score": 99,
                    "alignment_summary": "writer pass 2",
                }
            ),
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps({"alignment_score": 70, "alignment_summary": "eval pass 1"}),
            json.dumps({"alignment_score": 99, "alignment_summary": "eval pass 2"}),
        ]
    )

    result = improve_resume_iterative(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        min_alignment_score=99,
        max_iterations=5,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["iterations"] == 1
    assert result["alignment_score"] == 99.0
    assert result["reached_threshold"] is True
    assert result["history"][0]["alignment_score"] == 99


def test_improve_iterative_feeds_prior_evaluator_feedback_into_next_prompt(monkeypatch) -> None:
    monkeypatch.setenv("TAILOR_MIN_CHANGES_PER_ITERATION", "1")
    current_resume = _valid_tailored_resume()
    improved1 = _valid_tailored_resume()
    improved1["summary"] = "first pass summary"
    improved2 = _valid_tailored_resume()
    improved2["summary"] = "second pass summary with kubernetes reliability details"
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved1,
                    "alignment_score": 95,
                    "alignment_summary": "writer pass 1",
                }
            ),
            json.dumps(
                {
                    "tailored_resume": improved2,
                    "alignment_score": 98,
                    "alignment_summary": "writer pass 2",
                }
            ),
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "alignment_score": 70,
                    "alignment_summary": "Missing stronger Kubernetes reliability evidence.",
                }
            ),
            json.dumps(
                {
                    "alignment_score": 99,
                    "alignment_summary": "Coverage and evidence look complete now.",
                }
            ),
        ]
    )

    result = improve_resume_iterative(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        min_alignment_score=99,
        max_iterations=3,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["iterations"] == 1
    assert "Missing stronger Kubernetes reliability evidence." in writer.prompts[1]


def test_improve_iterative_feeds_structured_feedback_into_next_prompt(monkeypatch) -> None:
    monkeypatch.setenv("TAILOR_MIN_CHANGES_PER_ITERATION", "1")
    current_resume = _valid_tailored_resume()
    improved1 = _valid_tailored_resume()
    improved1["summary"] = "first pass summary"
    improved2 = _valid_tailored_resume()
    improved2["summary"] = "second pass summary with reliability metrics"
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved1,
                    "alignment_score": 95,
                    "alignment_summary": "writer pass 1",
                }
            ),
            json.dumps(
                {
                    "tailored_resume": improved2,
                    "alignment_score": 98,
                    "alignment_summary": "writer pass 2",
                }
            ),
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "alignment_score": 70,
                    "alignment_summary": "Key gaps remain.",
                    "top_gaps": ["Kubernetes reliability evidence is weak."],
                    "must_fix_before_95": [
                        "Add one quantified bullet on Kubernetes reliability ownership."
                    ],
                    "missing_evidence": ["No SLO/SLI metric tied to Kubernetes operations."],
                    "recommended_edits": [
                        "Add production SLO numbers and on-call outcomes for Kubernetes services."
                    ],
                }
            ),
            json.dumps(
                {
                    "alignment_score": 99,
                    "alignment_summary": "Coverage and evidence look complete now.",
                }
            ),
        ]
    )

    result = improve_resume_iterative(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        min_alignment_score=99,
        max_iterations=3,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["iterations"] == 1
    assert "Add one quantified bullet on Kubernetes reliability ownership." in writer.prompts[1]


def test_improve_iterative_retries_same_iteration_when_no_changes_made() -> None:
    current_resume = _valid_tailored_resume()
    unchanged = _valid_tailored_resume()
    improved = _valid_tailored_resume()
    improved["summary"] = "Updated summary with direct JD evidence and architecture outcomes."
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": unchanged,
                    "alignment_score": 98,
                    "alignment_summary": "writer unchanged pass",
                }
            ),
            json.dumps(
                {
                    "tailored_resume": improved,
                    "alignment_score": 98,
                    "alignment_summary": "writer revised pass",
                }
            ),
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps({"alignment_score": 70, "alignment_summary": "Needs stronger evidence."}),
            json.dumps({"alignment_score": 99, "alignment_summary": "Now aligned."}),
        ]
    )

    result = improve_resume_iterative(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        min_alignment_score=99,
        max_iterations=3,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["iterations"] == 1
    assert len(writer.prompts) == 2
    assert result["alignment_score"] == 99.0


def test_improve_iterative_returns_last_completed_result_on_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TAILOR_MIN_CHANGES_PER_ITERATION", "1")
    current_resume = _valid_tailored_resume()
    improved1 = _valid_tailored_resume()
    improved1["summary"] = "first pass summary"
    improved1["header"]["title"] = "Principal AI Engineer"
    improved1["skills"][0]["items"] = ["Python", "SQL", "Go", "Terraform"]
    improved1["experience"][0]["bullets"] = [
        "Built AI-powered workflow systems with measurable impact and improved p95 latency by 30 percent.",
        "Led platform reliability initiatives with SLO governance and incident runbooks.",
    ]
    writer = _FakeProvider(
        outputs=[
            json.dumps(
                {
                    "tailored_resume": improved1,
                    "alignment_score": 94,
                    "alignment_summary": "writer pass 1",
                }
            ),
            TimeoutError("timed out while generating iteration 2"),
        ]
    )
    evaluator = _FakeProvider(
        outputs=[
            json.dumps({"alignment_score": 90, "alignment_summary": "eval pass 1"}),
        ]
    )

    result = improve_resume_iterative(
        tailored_resume=current_resume,
        tailored_text=None,
        job={},
        memory=None,
        min_alignment_score=99,
        max_iterations=5,
        provider=writer,
        evaluator={"mode": "llm", "provider": evaluator},
    )

    assert result["iterations"] == 1
    assert result["alignment_score"] == 90.0
    assert result["reached_threshold"] is False
    assert result["tailored_resume"]["summary"] == "first pass summary"


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


def test_tailor_resume_backfills_certifications_from_candidate_resume() -> None:
    payload = _valid_tailored_resume()
    payload["certifications"] = []
    provider = _FakeProvider(outputs=[json.dumps(payload)])
    job = {
        "context_json": {
            "job_description": "Staff Software Engineer, AI/ML",
            "candidate_resume": (
                "SUMMARY\n...\n"
                "CERTIFICATIONS\n"
                "AWS Certified AI Practitioner (AIF-C01), 2025 | "
                "https://www.credly.com/badges/d91a3fa3-b52c-4b44-8b5d-5f75f13f58e1/public_url\n"
            ),
            "target_role_name": "Staff Software Engineer, AI/ML",
            "seniority_level": "Senior",
        }
    }

    result = tailor_resume(job=job, memory=None, provider=provider)

    assert isinstance(result["certifications"], list)
    assert result["certifications"]
    first = result["certifications"][0]
    assert first["name"].startswith("AWS Certified AI Practitioner")
    assert first["issuer"] == "AWS"
    assert str(first["year"]) == "2025"
    assert "credly.com/badges" in first.get("url", "")
