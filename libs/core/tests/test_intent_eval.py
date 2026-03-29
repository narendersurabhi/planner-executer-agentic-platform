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
            "    expected_capabilities: [document.spec.generate, document.pdf.render]\n"
            "    expected_capabilities_by_segment:\n"
            "      - [document.spec.generate]\n"
            "      - [document.pdf.render]\n"
            "    expected_profile_intent: generate\n"
            "    expected_missing_inputs: [output_format]\n"
            "    expected_requires_clarification: true\n"
            "    expected_clarification_mode: capability_required_inputs\n"
        ),
        encoding="utf-8",
    )
    cases = load_intent_eval_cases(fixture)
    assert len(cases) == 1
    assert cases[0].case_id == "c1"
    assert cases[0].expected_intents == ("generate", "render")
    assert cases[0].expected_profile_intent == "generate"
    assert cases[0].expected_missing_inputs == ("output_format",)
    assert cases[0].expected_requires_clarification is True
    assert cases[0].expected_clarification_mode == "capability_required_inputs"


def test_evaluate_intent_case_scores_expected_matches() -> None:
    case = IntentEvalCase(
        case_id="c1",
        goal="Generate and render.",
        expected_intents=("generate", "render"),
        expected_capabilities=("document.spec.generate", "document.pdf.render"),
        expected_capabilities_by_segment=(
            ("document.spec.generate",),
            ("document.pdf.render",),
        ),
    )
    graph = {
        "segments": [
            {"intent": "generate", "suggested_capabilities": ["document.spec.generate"]},
            {"intent": "render", "suggested_capabilities": ["document.pdf.render"]},
        ]
    }
    result = evaluate_intent_case(
        case,
        graph,
        top_k=2,
        allowed_capability_ids={"document.spec.generate", "document.pdf.render"},
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
            expected_capabilities=("document.spec.generate", "document.pdf.render"),
            expected_capabilities_by_segment=(
                ("document.spec.generate",),
                ("document.pdf.render",),
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
                    {"intent": "render", "suggested_capabilities": ["document.pdf.render"]},
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
            "document.pdf.render",
            "memory.read",
            "utility.json.transform",
        },
    )
    summary = report["summary"]
    assert summary["case_count"] == 2
    assert summary["intent_micro"]["f1"] == 1.0
    assert summary["capability_micro"]["f1"] == 1.0
    assert summary["segment_topk"]["hit_rate"] == 1.0


def test_evaluate_intent_cases_reports_normalization_metrics() -> None:
    case = IntentEvalCase(
        case_id="c1",
        goal="Generate a memo.",
        expected_intents=("generate",),
        expected_capabilities=("document.spec.generate",),
        expected_capabilities_by_segment=(("document.spec.generate",),),
        expected_profile_intent="generate",
        expected_missing_inputs=("output_format",),
        expected_requires_clarification=True,
        expected_clarification_mode="capability_required_inputs",
    )

    report = evaluate_intent_cases(
        [case],
        decompose_goal=lambda _goal: {
            "segments": [
                {
                    "intent": "generate",
                    "suggested_capabilities": ["document.spec.generate"],
                }
            ]
        },
        normalize_goal_intent=lambda _goal, _context: {
            "profile": {
                "intent": "generate",
                "needs_clarification": True,
                "clarification_mode": "capability_required_inputs",
            },
            "graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "generate",
                        "suggested_capabilities": ["document.spec.generate"],
                    }
                ]
            },
            "candidate_capabilities": {"s1": ["document.spec.generate"]},
            "clarification": {
                "needs_clarification": True,
                "missing_inputs": ["output_format"],
                "clarification_mode": "capability_required_inputs",
            },
            "trace": {"disagreement": {}},
        },
        top_k=2,
        allowed_capability_ids={"document.spec.generate"},
    )

    summary = report["summary"]["normalization"]
    assert summary["profile_intent_accuracy"] == 1.0
    assert summary["capability_alignment_rate"] == 1.0
    assert summary["missing_inputs"]["precision"] == 1.0
    assert summary["requires_clarification_accuracy"] == 1.0
    assert summary["clarification_mode_accuracy"] == 1.0
    case_result = report["cases"][0]["normalization"]
    assert case_result["profile_intent"]["match"] is True
    assert case_result["clarification"]["mode_match"] is True


def test_phase0_intent_normalization_gold_set_loads() -> None:
    cases = load_intent_eval_cases(Path("eval/intent_normalization_gold.yaml"))

    assert [case.case_id for case in cases] == [
        "document_generation_memo",
        "render_explicit_pdf_path",
        "repo_branch_release_summary",
        "workspace_timeout_retrieval",
        "high_risk_workspace_delete",
    ]
    assert cases[-1].expected_capabilities == ("filesystem.workspace.delete",)
    assert cases[0].expected_profile_intent == "generate"
    assert cases[0].expected_missing_inputs == ("output_format",)
