from libs.core import chat_routing_calibrator, chat_routing_calibrator_eval


def test_evaluate_feedback_rows_reports_disagreement_and_live_override() -> None:
    model = chat_routing_calibrator.train_model(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["generic:submit_job"],
                "top_k_candidates": ["workflow:release", "generic:submit_job"],
                "execution_succeeded": True,
            }
        ],
        epochs=140,
        learning_rate=0.2,
    )
    report = chat_routing_calibrator_eval.evaluate_feedback_rows(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "selected_candidate_id": "generic:submit_job",
                "top_k_candidates": ["generic:submit_job", "workflow:release"],
                "feedback_label": "negative",
                "selected_positive_ids": [],
                "selected_negative_ids": ["generic:submit_job"],
                "hard_negative_ids": [],
            }
        ],
        model=model,
        min_probability=0.55,
        min_margin=0.01,
    )

    assert report["summary"]["row_count"] == 1
    assert report["summary"]["disagreement_count"] == 1
    assert report["summary"]["live_override_count"] == 1
    assert report["summary"]["shadow_better_count"] == 1
    assert report["summary"]["live_better_count"] == 1
    assert report["cases"][0]["shadow_selected_candidate_id"] == "workflow:release"
    assert report["cases"][0]["live_selected_candidate_id"] == "workflow:release"


def test_evaluate_feedback_rows_reports_blocked_live_override_reason() -> None:
    model = chat_routing_calibrator.train_model(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "route": "run_workflow",
                "positive_candidate_id": "workflow:release",
                "negative_candidate_ids": ["generic:submit_job"],
                "top_k_candidates": ["workflow:release", "generic:submit_job"],
                "execution_succeeded": True,
            }
        ],
        epochs=140,
        learning_rate=0.2,
    )
    report = chat_routing_calibrator_eval.evaluate_feedback_rows(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "selected_candidate_id": "generic:submit_job",
                "top_k_candidates": ["generic:submit_job", "workflow:release"],
                "feedback_label": "negative",
                "selected_positive_ids": [],
                "selected_negative_ids": ["generic:submit_job"],
                "hard_negative_ids": [],
            }
        ],
        model=model,
        min_probability=0.99,
        min_margin=0.5,
    )

    assert report["summary"]["live_override_count"] == 0
    assert report["live_override_reasons"]
    assert report["cases"][0]["live_override_used"] is False
