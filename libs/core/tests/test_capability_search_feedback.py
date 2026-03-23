from libs.core import capability_search_feedback


def test_build_feedback_rows_joins_search_selection_and_execution() -> None:
    rows = capability_search_feedback.build_feedback_rows(
        [
            {
                "type": "plan.capability_search",
                "job_id": "job-1",
                "correlation_id": "corr-1",
                "payload": {
                    "query": "render a pdf report",
                    "results": [
                        {"id": "document.spec.generate"},
                        {"id": "document.pdf.render"},
                        {"id": "llm.text.generate"},
                    ],
                },
            },
            {
                "type": "plan.capability_selection",
                "job_id": "job-1",
                "correlation_id": "corr-1",
                "payload": {
                    "selected_capabilities": ["document.spec.generate", "document.pdf.render"]
                },
            },
            {
                "type": "task.completed",
                "job_id": "job-1",
                "correlation_id": "corr-1",
                "payload": {
                    "tool_calls": [
                        {"tool_name": "document.spec.generate"},
                        {"tool_name": "document.pdf.render"},
                    ]
                },
            },
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["retrieved_selected"] == ["document.spec.generate", "document.pdf.render"]
    assert row["retrieved_executed"] == ["document.spec.generate", "document.pdf.render"]
    assert row["planner_override"] == []
    assert row["hard_negative_ids"] == ["llm.text.generate"]
    assert row["execution_succeeded"] is True
