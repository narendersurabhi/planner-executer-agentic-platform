from libs.core import capability_search_training


def test_build_reranker_training_examples_uses_completed_or_selected_positives() -> None:
    examples = capability_search_training.build_reranker_training_examples(
        [
            {
                "job_id": "job-1",
                "correlation_id": "corr-1",
                "search_queries": ["render a pdf report"],
                "retrieved_capabilities": [
                    "document.pdf.generate",
                    "document.output.derive",
                    "llm.text.generate",
                ],
                "selected_capabilities": [
                    "document.output.derive",
                    "document.pdf.generate",
                ],
                "executed_capabilities": [
                    {"id": "document.output.derive", "status": "completed"},
                    {"id": "document.pdf.generate", "status": "completed"},
                ],
                "execution_succeeded": True,
                "planner_override": [],
                "retrieved_selected": [
                    "document.output.derive",
                    "document.pdf.generate",
                ],
                "retrieved_executed": [
                    "document.output.derive",
                    "document.pdf.generate",
                ],
                "hard_negative_ids": ["llm.text.generate"],
            }
        ]
    )

    assert len(examples) == 2
    assert examples[0]["query"] == "render a pdf report"
    assert examples[0]["negative_capability_ids"] == ["llm.text.generate"]
    assert {entry["positive_capability_id"] for entry in examples} == {
        "document.output.derive",
        "document.pdf.generate",
    }
