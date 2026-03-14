from libs.core import capability_search_feedback_eval


def test_evaluate_feedback_rows_reports_core_rates() -> None:
    report = capability_search_feedback_eval.evaluate_feedback_rows(
        [
            {
                "selected_capabilities": ["document.pdf.generate"],
                "retrieved_selected": ["document.pdf.generate"],
                "executed_capabilities": [{"id": "document.pdf.generate", "status": "completed"}],
                "execution_succeeded": True,
                "planner_override": [],
                "retrieved_capabilities": ["document.pdf.generate", "llm.text.generate"],
                "hard_negative_ids": ["llm.text.generate"],
            },
            {
                "selected_capabilities": ["github.pull_request.create"],
                "retrieved_selected": [],
                "executed_capabilities": [{"id": "github.pull_request.create", "status": "failed"}],
                "execution_succeeded": False,
                "planner_override": ["github.pull_request.create"],
                "retrieved_capabilities": ["github.branch.create"],
                "hard_negative_ids": ["github.branch.create"],
            },
        ]
    )

    assert report["row_count"] == 2
    assert report["planner_selection_hit_rate"] == 0.5
    assert report["execution_success_rate"] == 0.5
    assert report["planner_override_rate"] == 0.5
    assert report["hard_negative_ratio"] > 0.0
