from services.worker.app import memory_semantics


def test_select_memory_payload_filters_keys() -> None:
    output = {
        "tailored_text": "hi",
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
        "extra": "ignore",
    }
    selected = memory_semantics.select_memory_payload("llm_improve_tailored_resume_text", output)
    assert selected == {
        "tailored_text": "hi",
        "alignment_score": 92.5,
        "alignment_summary": "strong match",
    }


def test_apply_memory_defaults_fills_missing_tailored_text() -> None:
    payload = {
        "memory": {
            "task_outputs": [
                {"tailored_text": "from memory", "alignment_score": 88},
                {"tailored_text": "older"},
            ]
        }
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_text"] == "from memory"


def test_apply_memory_defaults_respects_existing_value() -> None:
    payload = {
        "tailored_text": "already set",
        "memory": {"task_outputs": [{"tailored_text": "from memory"}]},
    }
    updated = memory_semantics.apply_memory_defaults("llm_improve_tailored_resume_text", payload)
    assert updated["tailored_text"] == "already set"
