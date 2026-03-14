from pathlib import Path

from libs.core import capability_registry, capability_search, capability_search_eval


def test_load_capability_search_eval_cases_reads_jsonl(tmp_path: Path) -> None:
    gold = tmp_path / "capability_search_gold.jsonl"
    gold.write_text(
        "\n".join(
            [
                '{"id":"pdf_render","query":"render a pdf report","intent":"render","expected_capability_ids":["document.pdf.generate"]}',
                '{"id":"memory","query":"search semantic memory","intent":"io","expected_capability_ids":["memory.semantic.search"]}',
            ]
        ),
        encoding="utf-8",
    )

    cases = capability_search_eval.load_capability_search_eval_cases(gold)

    assert [case.case_id for case in cases] == ["pdf_render", "memory"]
    assert cases[0].expected_capability_ids == ("document.pdf.generate",)


def test_evaluate_capability_search_cases_reports_hits() -> None:
    registry = capability_registry.load_capability_registry()
    entries = capability_search.build_capability_search_entries(registry.enabled_capabilities())
    cases = [
        capability_search_eval.CapabilitySearchEvalCase(
            case_id="pdf",
            query="render a pdf report from a document spec",
            intent="render",
            segment_text=None,
            expected_capability_ids=("document.pdf.generate",),
            acceptable_alternates=(),
            must_have=("document.pdf.generate",),
            risk_tier="read_only",
            notes=None,
        ),
        capability_search_eval.CapabilitySearchEvalCase(
            case_id="memory",
            query="search semantic memory for user preferences",
            intent="io",
            segment_text=None,
            expected_capability_ids=("memory.semantic.search",),
            acceptable_alternates=(),
            must_have=("memory.semantic.search",),
            risk_tier="read_only",
            notes=None,
        ),
    ]

    report = capability_search_eval.evaluate_capability_search_cases(
        cases,
        capability_entries=entries,
        limit=5,
    )

    assert report["summary"]["hit_rate_at_1"] > 0.0
    assert report["summary"]["hit_rate_at_3"] == 1.0
    by_case = {entry["case_id"]: entry for entry in report["cases"]}
    assert by_case["pdf"]["planner_useful_candidates"]
    assert by_case["memory"]["metrics"]["must_have_hit_rate"] == 1.0
