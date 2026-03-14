from libs.core import capability_reranker


def test_reranker_boosts_successful_query_specific_capability() -> None:
    results = [
        {"id": "memory.semantic.search", "score": 30.0, "reason": "semantic match", "source": "semantic_search"},
        {"id": "memory.semantic.write", "score": 28.0, "reason": "semantic match", "source": "semantic_search"},
    ]
    feedback_rows = [
        {
            "search_queries": ["store a fact in semantic memory for future reuse"],
            "retrieved_selected": ["memory.semantic.write"],
            "retrieved_executed": ["memory.semantic.write"],
            "hard_negative_ids": ["memory.semantic.search"],
            "executed_capabilities": [{"id": "memory.semantic.write", "status": "completed"}],
        }
    ]

    reranked = capability_reranker.rerank_capability_results(
        query="store a fact in semantic memory for future reuse",
        intent_hint="io",
        results=results,
        feedback_rows=feedback_rows,
        limit=5,
    )

    assert reranked[0]["id"] == "memory.semantic.write"
    assert "rerank executed=1" in reranked[0]["reason"]

