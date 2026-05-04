from pathlib import Path

from libs.core import chat_clarification_eval


def test_load_chat_clarification_eval_cases_from_repo_fixture() -> None:
    cases = chat_clarification_eval.load_chat_clarification_eval_cases(
        Path("eval/chat_clarification_mapping_gold.yaml")
    )

    assert len(cases) >= 5
    assert cases[0].active_field_before == "audience"
    assert cases[0].expected_queue_advanced is True


def test_evaluate_chat_clarification_cases_matches_repo_fixture() -> None:
    cases = chat_clarification_eval.load_chat_clarification_eval_cases(
        Path("eval/chat_clarification_mapping_gold.yaml")
    )

    report = chat_clarification_eval.evaluate_chat_clarification_cases(cases)

    assert report["summary"]["overall_accuracy"] == 1.0
    assert report["summary"]["restart_decision_accuracy"] == 1.0
    assert report["summary"]["resolved_active_field_accuracy"] == 1.0
    assert report["summary"]["queue_advance_accuracy"] == 1.0
    assert report["summary"]["wrong_field_assignment_rate"] == 0.0


def test_evaluate_chat_clarification_cases_reports_wrong_field_assignment() -> None:
    cases = [
        chat_clarification_eval.ChatClarificationEvalCase(
            case_id="wrong_field_assignment",
            content="save it as medic.docx in the workspace",
            active_field_before="audience",
            active_field_after="audience",
            resolved_fields=("path",),
            expected_restarted=False,
            expected_resolved_active_field=True,
            expected_queue_advanced=False,
            notes=None,
        )
    ]

    report = chat_clarification_eval.evaluate_chat_clarification_cases(cases)

    assert report["summary"]["overall_accuracy"] == 0.0
    assert report["summary"]["wrong_field_assignment_rate"] == 1.0
