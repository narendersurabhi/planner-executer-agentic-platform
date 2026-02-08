from services.worker.app import memory_semantics


def test_select_memory_payload_filters_keys() -> None:
    output = {
        "tailored_resume": {"summary": "hi", "skills": [], "experience": [], "education": [], "certifications": []},
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload("llm_improve_tailored_resume_text", output)
    assert selected == {
        "tailored_resume": {"summary": "hi", "skills": [], "experience": [], "education": [], "certifications": []},
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
    }


def test_apply_memory_defaults_fills_missing_tailored_text() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "tailored_resume": {
                        "summary": "from memory",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    },
                    "alignment_score": 88,
                },
                {
                    "tailored_resume": {
                        "summary": "older",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    }
                },
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "from memory"


def test_apply_memory_defaults_respects_existing_value() -> None:
    payload = {
        "tailored_resume": {
            "summary": "already set",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
        },
        "memory": {"task_outputs": [{"tailored_text": "from memory"}]},
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "already set"


def test_apply_memory_defaults_prefers_latest_keyed_entry() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {
                    "tailored_resume": {
                        "summary": "older",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    }
                },
                {
                    "_memory_key": "tailored_resume:latest",
                    "tailored_resume": {
                        "summary": "preferred",
                        "skills": [],
                        "experience": [],
                        "education": [],
                        "certifications": [],
                    },
                },
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_resume"]["summary"] == "preferred"


def test_extract_memory_value_prefers_alignment_latest_key() -> None:
    entries = [
        {"alignment_score": 80},
        {"_memory_key": "alignment_score:latest", "alignment_score": 92},
    ]
    value = memory_semantics.extract_memory_value(entries, "alignment_score")
    assert value == 92


def test_select_memory_payload_includes_resume_doc_spec() -> None:
    output = {"resume_doc_spec": {"doc_type": "resume"}, "extra": "ignore"}
    selected = memory_semantics.select_memory_payload(
        "llm_generate_resume_doc_spec_from_text", output
    )
    assert selected == {"resume_doc_spec": {"doc_type": "resume"}}
