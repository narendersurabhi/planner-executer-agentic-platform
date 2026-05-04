from pathlib import Path

import pytest

from libs.core.deepeval_agents import (
    build_chat_feedback_bundle,
    build_chat_gold_bundle,
    build_planner_feedback_bundle,
    build_planner_gold_bundle,
)
from libs.core.deepeval_support import (
    DeepEvalCase,
    DeepEvalMetricSpec,
    DeepEvalSettings,
    run_deepeval_cases,
    validate_settings,
)


def test_build_chat_gold_bundle_preserves_gold_expectations() -> None:
    bundle = build_chat_gold_bundle(
        boundary_gold=Path("eval/chat_boundary_gold.yaml"),
        clarification_gold=Path("eval/chat_clarification_mapping_gold.yaml"),
    )

    assert bundle.surface == "chat"
    assert bundle.dataset_id == "chat_boundary_gold+chat_clarification_mapping_gold"
    assert {spec.name for spec in bundle.metric_specs} == {
        "chat_decision_quality",
        "chat_clarification_behavior",
    }

    boundary_case = next(
        case for case in bundle.cases if case.case_id == "document_create_execution_request"
    )
    clarification_case = next(
        case for case in bundle.cases if case.case_id == "redirect_to_new_document_request_restarts"
    )

    assert boundary_case.surface == "chat_boundary"
    assert '"decision": "execution_request"' in boundary_case.expected_output
    assert boundary_case.local_pass is True
    assert boundary_case.local_scores["decision_match"] is True

    assert clarification_case.surface == "chat_clarification"
    assert '"restarted": true' in clarification_case.expected_output
    assert clarification_case.local_pass is True
    assert clarification_case.local_scores["overall_match"] is True


def test_build_planner_gold_bundle_uses_normalizer_and_tracks_missing_inputs(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "planner_gold.yaml"
    fixture.write_text(
        (
            "version: 1\n"
            "cases:\n"
            "  - id: planner_case_1\n"
            "    goal: Generate a memo.\n"
            "    expected_intents: [generate]\n"
            "    expected_capabilities: [document.spec.generate]\n"
            "    expected_capabilities_by_segment:\n"
            "      - [document.spec.generate]\n"
            "    expected_profile_intent: generate\n"
            "    expected_missing_inputs: [output_format]\n"
            "    expected_requires_clarification: true\n"
            "    expected_clarification_mode: capability_required_inputs\n"
        ),
        encoding="utf-8",
    )

    def _normalize(_goal: str, _context: dict | None) -> dict:
        return {
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
        }

    bundle = build_planner_gold_bundle(
        normalization_gold=fixture,
        normalize_goal_intent=_normalize,
    )

    assert bundle.surface == "planner"
    assert bundle.dataset_id == "planner_gold"
    assert len(bundle.cases) == 1
    case = bundle.cases[0]
    assert case.surface == "planner_boundary"
    assert '"expected_capabilities": ["document.spec.generate"]' in case.expected_output
    assert case.local_pass is True
    assert case.local_scores["capability_alignment"] is True
    assert case.local_scores["missing_inputs_precision"] == 1.0
    assert bundle.deterministic_summary["capability_alignment_rate"] == 1.0


def test_run_deepeval_cases_uses_deterministic_fallback_when_judge_disabled() -> None:
    report = run_deepeval_cases(
        surface="chat",
        dataset_id="unit",
        cases=[
            DeepEvalCase(
                case_id="case_1",
                surface="chat_boundary",
                source_dataset_id="unit",
                input_text="User turn: create a document",
                actual_output='{"decision":"execution_request"}',
                expected_output='{"decision":"execution_request"}',
                context=("Case type: unit.",),
                local_scores={"decision_match": True, "family_scope_match": True},
                local_pass=True,
                metadata={"case_kind": "unit"},
            )
        ],
        metric_specs=(DeepEvalMetricSpec(name="unit_metric", criteria="unit"),),
        settings=DeepEvalSettings(enabled=False, mode="local", judge_provider="mock"),
        threshold=0.95,
        deterministic_summary={"unit": {"accuracy": 1.0}},
    )

    assert report["schema_version"] == "deepeval_eval_report_v1"
    assert report["deepeval"]["judge_used"] is False
    assert report["case_count"] == 1
    assert report["overall_score"] == 1.0
    assert report["threshold_passed"] is True
    assert report["deterministic_metrics_summary"][0]["name"] == "decision_match"
    assert report["cases"][0]["passed"] is True
    assert report["cases"][0]["judge_metrics"] == []


def test_validate_settings_requires_model_and_key_when_enabled() -> None:
    with pytest.raises(ValueError, match="deepeval_judge_model_missing"):
        validate_settings(
            DeepEvalSettings(
                enabled=True,
                mode="local",
                judge_provider="openai",
                judge_model=None,
                judge_api_key=None,
            )
        )


def test_feedback_bundles_preserve_sentiment_labels() -> None:
    chat_examples = [
        {
            "feedback": {
                "id": "fb_chat_1",
                "target_type": "chat_message",
                "target_id": "message_1",
                "sentiment": "negative",
                "reason_codes": ["clarification_loop"],
                "comment": "It should have submitted the job.",
            },
            "snapshot": {
                "user_message_content": "create a deployment report",
                "metadata": {
                    "boundary_decision": {"decision": "meta_clarification", "evidence": {}}
                },
            },
            "dimensions": {
                "clarification_mapping_restarted": "no",
                "clarification_mapping_queue_advanced": "no",
                "clarification_mapping_resolved_active_field": "no",
            },
        }
    ]
    planner_examples = [
        {
            "feedback": {
                "id": "fb_plan_1",
                "target_type": "plan",
                "target_id": "plan_1",
                "sentiment": "partial",
                "reason_codes": ["wrong_capability"],
                "comment": "The capability choice drifted.",
            },
            "snapshot": {
                "plan_id": "plan_1",
                "planner_version": "planner_v1",
                "tasks_summary": "One task",
                "task_count": 1,
                "tasks": [{"name": "Generate"}],
            },
        }
    ]

    chat_bundle = build_chat_feedback_bundle(examples=chat_examples)
    planner_bundle = build_planner_feedback_bundle(examples=planner_examples)

    assert len(chat_bundle.cases) == 1
    assert chat_bundle.cases[0].metadata["sentiment"] == "negative"
    assert chat_bundle.cases[0].local_pass is False

    assert len(planner_bundle.cases) == 1
    assert planner_bundle.cases[0].metadata["sentiment"] == "partial"
    assert planner_bundle.cases[0].local_scores["feedback_sentiment_score"] == 0.5
