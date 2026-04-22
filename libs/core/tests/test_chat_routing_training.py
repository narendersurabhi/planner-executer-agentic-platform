from libs.core import chat_routing_training


def test_build_reranker_training_examples_uses_positive_selected_candidate() -> None:
    examples = chat_routing_training.build_reranker_training_examples(
        [
            {
                "feedback_id": "fb-1",
                "query": "Run release readiness workflow",
                "decision_route": "run_workflow",
                "selected_positive_ids": ["workflow:release"],
                "hard_negative_ids": ["generic:submit_job", "workflow:other"],
                "top_k_candidates": [
                    "workflow:release",
                    "generic:submit_job",
                    "workflow:other",
                ],
                "execution_succeeded": True,
            }
        ]
    )

    assert len(examples) == 1
    assert examples[0]["positive_candidate_id"] == "workflow:release"
    assert examples[0]["negative_candidate_ids"] == ["generic:submit_job", "workflow:other"]
    assert examples[0]["route"] == "run_workflow"
