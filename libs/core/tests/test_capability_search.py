from libs.core import capability_registry, capability_search


def test_search_capabilities_prefers_pdf_rendering_for_pdf_report_query() -> None:
    registry = capability_registry.load_capability_registry()
    entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    results = capability_search.search_capabilities(
        query="Generate a PDF report from a document spec",
        capability_entries=entries,
        limit=5,
        intent_hint="render",
    )
    ids = [str(item["id"]) for item in results]
    assert "document.pdf.generate" in ids
    assert any("pdf" in str(item["reason"]).lower() for item in results)


def test_search_capabilities_prefers_semantic_memory_for_memory_query() -> None:
    registry = capability_registry.load_capability_registry()
    entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    results = capability_search.search_capabilities(
        query="Search semantic memory for user preferences",
        capability_entries=entries,
        limit=5,
        intent_hint="io",
    )
    ids = [str(item["id"]) for item in results]
    assert "memory.semantic.search" in ids


def test_search_capabilities_can_use_feedback_reranking() -> None:
    registry = capability_registry.load_capability_registry()
    entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    results = capability_search.search_capabilities(
        query="store a fact in semantic memory for future reuse",
        capability_entries=entries,
        limit=5,
        intent_hint="io",
        rerank_feedback_rows=[
            {
                "search_queries": ["store a fact in semantic memory for future reuse"],
                "retrieved_selected": ["memory.semantic.write"],
                "retrieved_executed": ["memory.semantic.write"],
                "hard_negative_ids": ["memory.semantic.search"],
                "executed_capabilities": [{"id": "memory.semantic.write", "status": "completed"}],
            }
        ],
    )
    assert results[0]["id"] == "memory.semantic.write"
