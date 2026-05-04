from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from . import intent_contract

_INTENT_CHANGE_ARTIFACT_TOKENS = {
    "document",
    "doc",
    "docx",
    "pdf",
    "markdown",
    "json",
    "report",
    "guide",
    "checklist",
    "workflow",
    "job",
    "issue",
    "issues",
    "repo",
    "repository",
    "file",
    "files",
    "summary",
    "runbook",
    "query",
}
_INTENT_CHANGE_ACTION_TOKENS = {
    "create",
    "generate",
    "write",
    "make",
    "render",
    "build",
    "draft",
    "search",
    "list",
    "read",
    "inspect",
    "fetch",
    "produce",
}
_OUTPUT_FORMAT_TOKENS = {"pdf", "docx", "markdown", "json", "word"}
_TONE_TOKENS = {"practical", "formal", "conversational", "executive", "technical", "concise"}


@dataclass(frozen=True)
class ChatClarificationEvalCase:
    case_id: str
    content: str
    active_field_before: str | None
    active_field_after: str | None
    resolved_fields: tuple[str, ...]
    expected_restarted: bool
    expected_resolved_active_field: bool
    expected_queue_advanced: bool
    notes: str | None = None


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _normalize_field(value: Any) -> str | None:
    normalized = intent_contract.normalize_required_input_key(value)
    return normalized or None


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _looks_like_local_clarification_field_answer(
    content: str,
    *,
    current_field: str | None,
) -> bool:
    normalized_field = _normalize_field(current_field)
    lowered = _normalize_string(content).lower()
    if not lowered or not normalized_field:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    if not tokens:
        return False

    if normalized_field == "output_format":
        return bool(tokens & _OUTPUT_FORMAT_TOKENS)
    if normalized_field == "tone":
        return bool(tokens & _TONE_TOKENS) and len(tokens) <= 8
    if normalized_field == "audience":
        return not bool(tokens & _INTENT_CHANGE_ACTION_TOKENS) and len(tokens) <= 12
    if normalized_field in {"path", "output_path"}:
        return bool(
            re.search(
                r"(\.(?:docx|pdf|md|markdown|json)\b|\b(?:workspace|folder|download|filename|path|save)\b)",
                lowered,
            )
        )
    return False


def predict_clarification_restart(
    *,
    content: str,
    active_field_before: str | None,
) -> bool:
    lowered = _normalize_string(content).lower()
    if not lowered:
        return False
    if re.search(r"\b(?:chat|answer|reply|respond|explain|here)\b", lowered) and re.search(
        r"\b(?:instead|only|just|not|no|skip|cancel|stop|rather)\b",
        lowered,
    ):
        return False
    if _looks_like_local_clarification_field_answer(
        content,
        current_field=active_field_before,
    ):
        return False
    redirect_patterns = (
        r"\bactually\b",
        r"\binstead\b",
        r"\brather\b",
        r"\bstart over\b",
        r"\brestart\b",
        r"\bnew request\b",
        r"\bdifferent request\b",
        r"\bmake this\b",
        r"\bturn this\b",
        r"\bconvert this\b",
    )
    if not any(re.search(pattern, lowered) for pattern in redirect_patterns):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    return bool(tokens & _INTENT_CHANGE_ARTIFACT_TOKENS) and bool(
        tokens & _INTENT_CHANGE_ACTION_TOKENS
    )


def load_chat_clarification_eval_cases(path: Path) -> list[ChatClarificationEvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("chat_clarification_eval_cases_invalid_root")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("chat_clarification_eval_cases_missing_cases")
    cases: list[ChatClarificationEvalCase] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, Mapping):
            continue
        case_id = _normalize_string(raw_case.get("id") or f"case_{index}")
        content = _normalize_string(raw_case.get("content"))
        active_field_before = _normalize_field(raw_case.get("active_field_before"))
        active_field_after = _normalize_field(raw_case.get("active_field_after"))
        resolved_fields: list[str] = []
        for raw_field in raw_case.get("resolved_fields") or []:
            field = _normalize_field(raw_field)
            if field and field not in resolved_fields:
                resolved_fields.append(field)
        if not case_id:
            raise ValueError(f"chat_clarification_eval_case_missing_id:index={index}")
        if not content:
            raise ValueError(f"chat_clarification_eval_case_missing_content:index={index}")
        cases.append(
            ChatClarificationEvalCase(
                case_id=case_id,
                content=content,
                active_field_before=active_field_before,
                active_field_after=active_field_after,
                resolved_fields=tuple(resolved_fields),
                expected_restarted=bool(raw_case.get("expected_restarted")),
                expected_resolved_active_field=bool(
                    raw_case.get("expected_resolved_active_field")
                ),
                expected_queue_advanced=bool(raw_case.get("expected_queue_advanced")),
                notes=_normalize_string(raw_case.get("notes")) or None,
            )
        )
    if not cases:
        raise ValueError("chat_clarification_eval_cases_empty")
    return cases


def evaluate_chat_clarification_case(
    case: ChatClarificationEvalCase,
) -> dict[str, Any]:
    restarted = predict_clarification_restart(
        content=case.content,
        active_field_before=case.active_field_before,
    )
    active_field_before = _normalize_field(case.active_field_before)
    active_field_after = _normalize_field(case.active_field_after)
    resolved_fields = list(case.resolved_fields)
    resolved_active_field = bool(
        not restarted
        and active_field_before
        and active_field_before in set(resolved_fields)
    )
    queue_advanced = bool(
        not restarted
        and resolved_active_field
        and (active_field_after is None or active_field_after != active_field_before)
    )
    restart_match = restarted == case.expected_restarted
    resolved_active_field_match = (
        resolved_active_field == case.expected_resolved_active_field
    )
    queue_advanced_match = queue_advanced == case.expected_queue_advanced
    wrong_field_assignment = bool(
        not restarted
        and not resolved_active_field
        and bool(resolved_fields)
        and case.expected_resolved_active_field
    )
    return {
        "case_id": case.case_id,
        "content": case.content,
        "active_field_before": active_field_before,
        "active_field_after": active_field_after,
        "resolved_fields": resolved_fields,
        "expected_restarted": case.expected_restarted,
        "predicted_restarted": restarted,
        "expected_resolved_active_field": case.expected_resolved_active_field,
        "predicted_resolved_active_field": resolved_active_field,
        "expected_queue_advanced": case.expected_queue_advanced,
        "predicted_queue_advanced": queue_advanced,
        "restart_match": restart_match,
        "resolved_active_field_match": resolved_active_field_match,
        "queue_advanced_match": queue_advanced_match,
        "overall_match": restart_match and resolved_active_field_match and queue_advanced_match,
        "wrong_field_assignment": wrong_field_assignment,
        "notes": case.notes,
    }


def evaluate_chat_clarification_cases(
    cases: list[ChatClarificationEvalCase],
) -> dict[str, Any]:
    results = [evaluate_chat_clarification_case(case) for case in cases]
    case_count = len(results)
    overall_matches = sum(1 for result in results if bool(result["overall_match"]))
    restart_matches = sum(1 for result in results if bool(result["restart_match"]))
    resolved_active_field_matches = sum(
        1 for result in results if bool(result["resolved_active_field_match"])
    )
    queue_advanced_matches = sum(
        1 for result in results if bool(result["queue_advanced_match"])
    )
    wrong_field_assignments = sum(
        1 for result in results if bool(result["wrong_field_assignment"])
    )
    resolved_active_field_count = sum(
        1 for result in results if bool(result["predicted_resolved_active_field"])
    )
    queue_advanced_count = sum(
        1 for result in results if bool(result["predicted_queue_advanced"])
    )
    restart_count = sum(1 for result in results if bool(result["predicted_restarted"]))
    summary = {
        "case_count": case_count,
        "overall_accuracy": round(_safe_div(overall_matches, case_count), 3),
        "restart_decision_accuracy": round(_safe_div(restart_matches, case_count), 3),
        "resolved_active_field_accuracy": round(
            _safe_div(resolved_active_field_matches, case_count),
            3,
        ),
        "queue_advance_accuracy": round(_safe_div(queue_advanced_matches, case_count), 3),
        "wrong_field_assignment_rate": round(
            _safe_div(wrong_field_assignments, case_count),
            3,
        ),
        "resolved_active_field_rate": round(
            _safe_div(resolved_active_field_count, case_count),
            3,
        ),
        "queue_advance_rate": round(_safe_div(queue_advanced_count, case_count), 3),
        "restart_rate": round(_safe_div(restart_count, case_count), 3),
    }
    return {"summary": summary, "cases": results}
