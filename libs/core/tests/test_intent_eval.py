from pathlib import Path

from libs.core.intent_eval import (
    IntentEvalCase,
    evaluate_intent_case,
    evaluate_intent_cases,
    load_intent_eval_cases,
)


def test_load_intent_eval_cases_parses_yaml(tmp_path: Path) -> None:
    fixture = tmp_path / "intent_gold.yaml"
    fixture.write_text(
        (
            "version: 1\n"
            "cases:\n"
            "  - id: c1\n"
            "    goal: Generate a PDF report\n"
            "    expected_intents: [generate, render]\n"
            "    expected_capabilities: [document.spec.generate, document.pdf.generate]\n"
            "    expected_capabilities_by_segment:\n"
            "      - [document.spec.generate]\n"
            "      - [document.pdf.generate]\n"
        ),
        encoding="utf-8",
    )
    cases = load_intent_eval_cases(fixture)
    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].expected_intents == ("generate", "render")


def test_evaluate_intent_case_scores_expected_matches() -> None:
    case = IntentEvalCase(
        case_id="c1",
        goal="Generate and render.",
        expected_intents=("generate", "render"),
        expected_capabilities=("document.spec.generate", "document.pdf.generate"),
        expected_capabilities_by_segment=(
            ("document.spec.generate",),
            ("document.pdf.generate",),
        ),
    )
    graph = {
        "segments": [
            {"intent": "generate", "suggested_capabilities": ["document.spec.generate"]},
            {"intent": "render", "suggested_capabilities": ["document.pdf.generate"]},
        ]
    }
    result = evaluate_intent_case(
        case,
        graph,
        top_k=2,
        allowed_capability_ids={"document.spec.generate", "document.pdf.generate"},
    )
    assert result["intent"]["f1"] == 1.0
    assert result["capabilities"]["f1"] == 1.0
    assert result["segment_topk"]["hit_rate"] == 1.0
    assert result["capabilities"]["invalid"] == []


def test_evaluate_intent_cases_aggregates_metrics() -> None:
    cases = [
        IntentEvalCase(
            case_id="c1",
            goal="Generate and render.",
            expected_intents=("generate", "render"),
            expected_capabilities=("document.spec.generate", "document.pdf.generate"),
            expected_capabilities_by_segment=(
                ("document.spec.generate",),
                ("document.pdf.generate",),
            ),
        ),
        IntentEvalCase(
            case_id="c2",
            goal="Read and summarize.",
            expected_intents=("io", "transform"),
            expected_capabilities=("memory.read", "utility.json.transform"),
            expected_capabilities_by_segment=(("memory.read",), ("utility.json.transform",)),
        ),
    ]

    def _decompose(goal: str):  # noqa: ANN001
        if goal.startswith("Generate"):
            return {
                "segments": [
                    {"intent": "generate", "suggested_capabilities": ["document.spec.generate"]},
                    {"intent": "render", "suggested_capabilities": ["document.pdf.generate"]},
                ]
            }
        return {
            "segments": [
                {"intent": "io", "suggested_capabilities": ["memory.read"]},
                {"intent": "transform", "suggested_capabilities": ["utility.json.transform"]},
            ]
        }

    report = evaluate_intent_cases(
        cases,
        decompose_goal=_decompose,
        top_k=2,
        allowed_capability_ids={
            "document.spec.generate",
            "document.pdf.generate",
            "memory.read",
            "utility.json.transform",
        },
    )
    summary = report["summary"]
    assert summary["case_count"] == 2
    assert summary["intent_micro"]["f1"] == 1.0
    assert summary["capability_micro"]["f1"] == 1.0
    assert summary["segment_topk"]["hit_rate"] == 1.0
