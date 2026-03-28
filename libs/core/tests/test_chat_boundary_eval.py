from pathlib import Path

from libs.core import chat_boundary_eval, chat_contracts


def test_load_chat_boundary_eval_cases_from_repo_fixture() -> None:
    cases = chat_boundary_eval.load_chat_boundary_eval_cases(Path("eval/chat_boundary_gold.yaml"))

    assert len(cases) == 3
    assert cases[0].expected_decision == chat_contracts.ChatBoundaryDecisionType.execution_request
    assert cases[0].evidence is not None
    assert cases[0].evidence.top_families[0].family == "documents"
    assert cases[1].pending_clarification is True


def test_evaluate_chat_boundary_cases_reports_false_chat_replies() -> None:
    cases = chat_boundary_eval.load_chat_boundary_eval_cases(Path("eval/chat_boundary_gold.yaml"))

    def predict(case: chat_boundary_eval.ChatBoundaryEvalCase) -> str:
        if case.pending_clarification:
            return "continue_pending"
        return "chat_reply"

    report = chat_boundary_eval.evaluate_chat_boundary_cases(cases, predict=predict)

    assert report["summary"]["case_count"] == 3
    assert report["summary"]["accuracy"] == 2 / 3
    assert report["summary"]["false_chat_reply_rate"] == 1 / 3
    assert report["summary"]["execution_escalation_rate"] == 0.0
    assert report["summary"]["pending_continuation_rate"] == 1.0
    assert report["summary"]["expected_counts"]["execution_request"] == 1
    assert report["summary"]["predicted_counts"]["chat_reply"] == 2


def test_evaluate_chat_boundary_cases_accepts_decision_objects() -> None:
    cases = chat_boundary_eval.load_chat_boundary_eval_cases(Path("eval/chat_boundary_gold.yaml"))

    def predict(
        case: chat_boundary_eval.ChatBoundaryEvalCase,
    ) -> chat_contracts.ChatBoundaryDecision:
        if case.pending_clarification:
            decision = chat_contracts.ChatBoundaryDecisionType.continue_pending
        elif case.case_id == "document_create_execution_request":
            decision = chat_contracts.ChatBoundaryDecisionType.execution_request
        else:
            decision = chat_contracts.ChatBoundaryDecisionType.chat_reply
        return chat_contracts.ChatBoundaryDecision(decision=decision, assistant_response="")

    report = chat_boundary_eval.evaluate_chat_boundary_cases(cases, predict=predict)

    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["execution_escalation_rate"] == 1 / 3
    assert report["summary"]["false_chat_reply_rate"] == 0.0


def test_predict_chat_boundary_from_evidence_matches_repo_fixture_cases() -> None:
    cases = chat_boundary_eval.load_chat_boundary_eval_cases(Path("eval/chat_boundary_gold.yaml"))

    report = chat_boundary_eval.evaluate_chat_boundary_cases(
        cases,
        predict=chat_boundary_eval.predict_chat_boundary_from_evidence,
    )

    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["false_chat_reply_rate"] == 0.0
    assert report["summary"]["pending_continuation_rate"] == 1.0
