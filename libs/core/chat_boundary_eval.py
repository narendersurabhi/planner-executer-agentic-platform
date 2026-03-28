from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from . import chat_contracts


@dataclass(frozen=True)
class ChatBoundaryEvalCase:
    case_id: str
    content: str
    candidate_goal: str
    expected_decision: chat_contracts.ChatBoundaryDecisionType
    evidence: chat_contracts.ChatBoundaryEvidence | None
    notes: str | None = None

    @property
    def pending_clarification(self) -> bool:
        return bool(self.evidence.pending_clarification) if self.evidence is not None else False

    @property
    def workflow_target_available(self) -> bool:
        return bool(self.evidence.workflow_target_available) if self.evidence is not None else False


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _derive_execution_signal_strength(
    evidence: chat_contracts.ChatBoundaryEvidence,
) -> tuple[float, float, float, str]:
    top_capability_score = (
        float(evidence.top_capability_score or 0.0)
        if evidence.top_capability_score
        else max((float(item.score or 0.0) for item in evidence.top_capabilities), default=0.0)
    )
    top_family_score = (
        float(evidence.top_family_score or 0.0)
        if evidence.top_family_score
        else max((float(item.score or 0.0) for item in evidence.top_families), default=0.0)
    )
    total_family_score = sum(float(item.score or 0.0) for item in evidence.top_families)
    family_concentration = (
        float(evidence.family_concentration or 0.0)
        if evidence.family_concentration
        else _safe_div(top_family_score, total_family_score)
    )
    strength = str(evidence.execution_signal_strength or "").strip().lower()
    if not strength:
        has_execution_intent = str(evidence.intent or "").strip().lower() not in {
            "",
            "other",
            "inform",
            "clarify",
        }
        if top_capability_score >= 0.8 and top_family_score >= 1.1 and family_concentration >= 0.6:
            strength = "strong"
        elif (
            top_capability_score >= 0.65
            and top_family_score >= 0.8
            and family_concentration >= 0.5
            and has_execution_intent
        ):
            strength = "strong"
        elif top_capability_score >= 0.55 and top_family_score >= 0.65:
            strength = "moderate"
        elif top_capability_score > 0.0 or top_family_score > 0.0:
            strength = "weak"
        else:
            strength = "none"
    return top_capability_score, top_family_score, family_concentration, strength


def _looks_like_likely_clarification_answer(case: ChatBoundaryEvalCase) -> bool:
    evidence = case.evidence or chat_contracts.ChatBoundaryEvidence()
    if evidence.likely_clarification_answer:
        return True
    lowered = case.content.strip().lower()
    if not lowered or lowered.endswith("?"):
        return False
    token_count = len([token for token in lowered.replace("/", " ").split() if token])
    return evidence.pending_clarification and token_count <= 18


def predict_chat_boundary_from_evidence(
    case: ChatBoundaryEvalCase,
) -> chat_contracts.ChatBoundaryDecisionType:
    evidence = case.evidence or chat_contracts.ChatBoundaryEvidence()
    _top_capability_score, top_family_score, _family_concentration, execution_signal_strength = (
        _derive_execution_signal_strength(evidence)
    )
    if evidence.pending_clarification:
        if _looks_like_likely_clarification_answer(case):
            return chat_contracts.ChatBoundaryDecisionType.continue_pending
        if evidence.conversation_mode_hint == "conversational":
            return chat_contracts.ChatBoundaryDecisionType.exit_pending_to_chat
        return chat_contracts.ChatBoundaryDecisionType.meta_clarification
    if (
        execution_signal_strength == "strong"
        and evidence.conversation_mode_hint != "conversational"
    ):
        return chat_contracts.ChatBoundaryDecisionType.execution_request
    if evidence.workflow_target_available and evidence.conversation_mode_hint != "conversational":
        return chat_contracts.ChatBoundaryDecisionType.execution_request
    if (
        evidence.conversation_mode_hint == "execution_oriented"
        and evidence.intent not in {"", "other", "inform", "clarify"}
        and (
            execution_signal_strength in {"strong", "moderate"}
            or top_family_score >= 0.8
        )
    ):
        return chat_contracts.ChatBoundaryDecisionType.execution_request
    return chat_contracts.ChatBoundaryDecisionType.chat_reply


def _coerce_decision(value: Any) -> chat_contracts.ChatBoundaryDecisionType:
    if isinstance(value, chat_contracts.ChatBoundaryDecisionType):
        return value
    if isinstance(value, chat_contracts.ChatBoundaryDecision):
        return value.decision
    if isinstance(value, Mapping):
        candidate = value.get("decision") or value.get("type")
        return chat_contracts.ChatBoundaryDecisionType(str(candidate or "").strip())
    return chat_contracts.ChatBoundaryDecisionType(str(value or "").strip())


def load_chat_boundary_eval_cases(path: Path) -> list[ChatBoundaryEvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("chat_boundary_eval_cases_invalid_root")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("chat_boundary_eval_cases_missing_cases")
    cases: list[ChatBoundaryEvalCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, Mapping):
            continue
        case_id = _normalize_string(raw_case.get("id") or f"case_{index}")
        content = _normalize_string(raw_case.get("content"))
        candidate_goal = _normalize_string(raw_case.get("candidate_goal") or content)
        expected_raw = _normalize_string(raw_case.get("expected_decision"))
        notes = _normalize_string(raw_case.get("notes")) or None
        if not case_id:
            raise ValueError(f"chat_boundary_eval_case_missing_id:index={index}")
        if not content:
            raise ValueError(f"chat_boundary_eval_case_missing_content:index={index}")
        if not expected_raw:
            raise ValueError(f"chat_boundary_eval_case_missing_expected_decision:index={index}")
        try:
            expected_decision = chat_contracts.ChatBoundaryDecisionType(expected_raw)
        except ValueError as exc:
            raise ValueError(
                f"chat_boundary_eval_case_invalid_expected_decision:index={index}:{expected_raw}"
            ) from exc
        evidence_payload = raw_case.get("evidence")
        evidence = None
        if isinstance(evidence_payload, Mapping):
            evidence = chat_contracts.ChatBoundaryEvidence.model_validate(evidence_payload)
        cases.append(
            ChatBoundaryEvalCase(
                case_id=case_id,
                content=content,
                candidate_goal=candidate_goal,
                expected_decision=expected_decision,
                evidence=evidence,
                notes=notes,
            )
        )
    if not cases:
        raise ValueError("chat_boundary_eval_cases_empty")
    return cases


def evaluate_chat_boundary_case(
    case: ChatBoundaryEvalCase,
    *,
    predict: Callable[[ChatBoundaryEvalCase], Any],
) -> dict[str, Any]:
    predicted_decision = _coerce_decision(predict(case))
    top_family = None
    if case.evidence is not None:
        for family in case.evidence.top_families:
            normalized = _normalize_string(family.family)
            if normalized:
                top_family = normalized
                break
    return {
        "case_id": case.case_id,
        "content": case.content,
        "candidate_goal": case.candidate_goal,
        "expected_decision": case.expected_decision.value,
        "predicted_decision": predicted_decision.value,
        "match": predicted_decision == case.expected_decision,
        "pending_clarification": case.pending_clarification,
        "workflow_target_available": case.workflow_target_available,
        "top_family": top_family,
        "notes": case.notes,
    }


def evaluate_chat_boundary_cases(
    cases: list[ChatBoundaryEvalCase],
    *,
    predict: Callable[[ChatBoundaryEvalCase], Any],
) -> dict[str, Any]:
    results = [evaluate_chat_boundary_case(case, predict=predict) for case in cases]
    case_count = len(results)
    matches = sum(1 for result in results if bool(result["match"]))
    false_chat_replies = sum(
        1
        for result in results
        if result["predicted_decision"] == chat_contracts.ChatBoundaryDecisionType.chat_reply.value
        and result["expected_decision"] != chat_contracts.ChatBoundaryDecisionType.chat_reply.value
    )
    execution_escalations = sum(
        1
        for result in results
        if result["predicted_decision"]
        == chat_contracts.ChatBoundaryDecisionType.execution_request.value
    )
    pending_cases = [result for result in results if bool(result["pending_clarification"])]
    pending_continuations = sum(
        1
        for result in pending_cases
        if result["predicted_decision"]
        == chat_contracts.ChatBoundaryDecisionType.continue_pending.value
    )

    expected_counts = {
        decision.value: sum(1 for result in results if result["expected_decision"] == decision.value)
        for decision in chat_contracts.ChatBoundaryDecisionType
    }
    predicted_counts = {
        decision.value: sum(1 for result in results if result["predicted_decision"] == decision.value)
        for decision in chat_contracts.ChatBoundaryDecisionType
    }

    return {
        "summary": {
            "case_count": case_count,
            "accuracy": _safe_div(matches, case_count),
            "false_chat_reply_rate": _safe_div(false_chat_replies, case_count),
            "execution_escalation_rate": _safe_div(execution_escalations, case_count),
            "pending_continuation_rate": _safe_div(pending_continuations, len(pending_cases)),
            "expected_counts": expected_counts,
            "predicted_counts": predicted_counts,
        },
        "cases": results,
    }
