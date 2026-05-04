from libs.core import capability_search_training


def test_build_reranker_training_examples_uses_completed_or_selected_positives() -> None:
    examples = capability_search_training.build_reranker_training_examples(
        [
            {
                "job_id": "job-1",
                "correlation_id": "corr-1",
                "search_queries": ["render a pdf report"],
                "retrieved_capabilities": [
                    "document.spec.generate",
                    "document.pdf.render",
                    "llm.text.generate",
                ],
                "selected_capabilities": ["document.spec.generate", "document.pdf.render"],
                "executed_capabilities": [
                    {"id": "document.spec.generate", "status": "completed"},
                    {"id": "document.pdf.render", "status": "completed"},
                ],
                "execution_succeeded": True,
                "planner_override": [],
                "retrieved_selected": ["document.spec.generate", "document.pdf.render"],
                "retrieved_executed": ["document.spec.generate", "document.pdf.render"],
                "hard_negative_ids": ["llm.text.generate"],
            }
        ]
    )

    assert len(examples) == 2
    assert examples[0]["query"] == "render a pdf report"
    assert examples[0]["negative_capability_ids"] == ["llm.text.generate"]
    assert {entry["positive_capability_id"] for entry in examples} == {
        "document.spec.generate",
        "document.pdf.render",
    }
