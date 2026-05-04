from libs.core import chat_routing_reranker


def test_rerank_route_candidates_promotes_positive_workflow_match() -> None:
    candidates = [
        {
            "candidate_id": "generic:submit_job",
            "candidate_type": "generic_path",
            "score": 55.0,
            "reason_codes": ["generic_execution_fallback"],
            "metadata": {},
        },
        {
            "candidate_id": "workflow:release",
            "candidate_type": "workflow",
            "score": 44.0,
            "reason_codes": ["workflow_token_overlap"],
            "metadata": {},
        },
    ]
    feedback_rows = [
        {
            "query": "Run release readiness workflow",
            "selected_positive_ids": ["workflow:release"],
            "selected_negative_ids": [],
            "hard_negative_ids": ["generic:submit_job"],
            "selected_candidate_type": "workflow",
            "feedback_label": "positive",
            "fallback_used": False,
        }
    ]

    reranked = chat_routing_reranker.rerank_route_candidates(
        query="Run release readiness workflow",
        candidates=candidates,
        feedback_rows=feedback_rows,
        limit=5,
    )

    assert reranked[0]["candidate_id"] == "workflow:release"
    assert any(code.startswith("routing_rerank_positive=") for code in reranked[0]["reason_codes"])
