from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import capability_search


@dataclass(frozen=True)
class CapabilitySearchEvalCase:
    case_id: str
    query: str
    intent: str | None
    segment_text: str | None
    expected_capability_ids: tuple[str, ...]
    acceptable_alternates: tuple[str, ...]
    must_have: tuple[str, ...]
    risk_tier: str | None
    notes: str | None


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _normalize_string(item)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _dcg(flags: list[int]) -> float:
    score = 0.0
    for index, value in enumerate(flags, start=1):
        if value <= 0:
            continue
        if index == 1:
            score += float(value)
            continue
        score += float(value) / __import__("math").log2(index + 1)
    return score


def load_capability_search_eval_cases(path: Path) -> list[CapabilitySearchEvalCase]:
    if not path.exists():
        raise ValueError(f"capability_search_eval_cases_missing:{path}")
    cases: list[CapabilitySearchEvalCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"capability_search_eval_case_invalid_json:line={line_number}:{exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"capability_search_eval_case_invalid_root:line={line_number}")
        case_id = _normalize_string(payload.get("id") or f"case_{line_number}")
        query = _normalize_string(payload.get("query"))
        expected_capability_ids = tuple(_normalize_string_list(payload.get("expected_capability_ids")))
        acceptable_alternates = tuple(_normalize_string_list(payload.get("acceptable_alternates")))
        must_have = tuple(_normalize_string_list(payload.get("must_have")))
        if not case_id:
            raise ValueError(f"capability_search_eval_case_missing_id:line={line_number}")
        if not query:
            raise ValueError(f"capability_search_eval_case_missing_query:line={line_number}")
        if not expected_capability_ids:
            raise ValueError(
                f"capability_search_eval_case_missing_expected_capability_ids:line={line_number}"
            )
        cases.append(
            CapabilitySearchEvalCase(
                case_id=case_id,
                query=query,
                intent=_normalize_string(payload.get("intent")) or None,
                segment_text=_normalize_string(payload.get("segment_text")) or None,
                expected_capability_ids=expected_capability_ids,
                acceptable_alternates=acceptable_alternates,
                must_have=must_have,
                risk_tier=_normalize_string(payload.get("risk_tier")) or None,
                notes=_normalize_string(payload.get("notes")) or None,
            )
        )
    if not cases:
        raise ValueError("capability_search_eval_cases_empty")
    return cases


def evaluate_capability_search_case(
    case: CapabilitySearchEvalCase,
    *,
    capability_entries: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    query_text = case.segment_text or case.query
    results = capability_search.search_capabilities(
        query=query_text,
        capability_entries=capability_entries,
        limit=max(1, limit),
        intent_hint=case.intent,
    )
    predicted_ids = [str(item.get("id") or "").strip() for item in results if str(item.get("id") or "").strip()]
    relevant_ids = set(case.expected_capability_ids).union(case.acceptable_alternates)
    top_k = predicted_ids[: max(1, limit)]
    hits = [1 if capability_id in relevant_ids else 0 for capability_id in top_k]
    hit_at_1 = 1.0 if hits[:1] and hits[0] else 0.0
    hit_at_3 = 1.0 if any(hits[:3]) else 0.0
    recall_at_5 = _safe_div(sum(1 for capability_id in top_k[:5] if capability_id in relevant_ids), len(relevant_ids))

    reciprocal_rank = 0.0
    for index, capability_id in enumerate(top_k, start=1):
        if capability_id in relevant_ids:
            reciprocal_rank = 1.0 / float(index)
            break

    ideal_hits = [1] * min(len(relevant_ids), len(top_k))
    ndcg = _safe_div(_dcg(hits), _dcg(ideal_hits))

    must_have_ids = set(case.must_have or case.expected_capability_ids)
    must_have_hit = 1.0 if any(capability_id in must_have_ids for capability_id in top_k) else 0.0
    planner_useful_candidates = [capability_id for capability_id in top_k if capability_id in relevant_ids]

    return {
        "case_id": case.case_id,
        "query": case.query,
        "query_text": query_text,
        "intent": case.intent,
        "expected_capability_ids": list(case.expected_capability_ids),
        "acceptable_alternates": list(case.acceptable_alternates),
        "must_have": list(case.must_have),
        "retrieved": results,
        "metrics": {
            "hit_rate_at_1": hit_at_1,
            "hit_rate_at_3": hit_at_3,
            "recall_at_5": recall_at_5,
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "must_have_hit_rate": must_have_hit,
        },
        "planner_useful_candidates": planner_useful_candidates,
    }


def evaluate_capability_search_cases(
    cases: list[CapabilitySearchEvalCase],
    *,
    capability_entries: list[dict[str, Any]],
    limit: int = 5,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit_must_be_positive")
    results = [
        evaluate_capability_search_case(case, capability_entries=capability_entries, limit=limit)
        for case in cases
    ]
    case_count = len(results)
    summary = {
        "case_count": case_count,
        "hit_rate_at_1": _safe_div(
            sum(float(result["metrics"]["hit_rate_at_1"]) for result in results), case_count
        ),
        "hit_rate_at_3": _safe_div(
            sum(float(result["metrics"]["hit_rate_at_3"]) for result in results), case_count
        ),
        "recall_at_5": _safe_div(
            sum(float(result["metrics"]["recall_at_5"]) for result in results), case_count
        ),
        "mrr": _safe_div(sum(float(result["metrics"]["mrr"]) for result in results), case_count),
        "ndcg": _safe_div(sum(float(result["metrics"]["ndcg"]) for result in results), case_count),
        "must_have_hit_rate": _safe_div(
            sum(float(result["metrics"]["must_have_hit_rate"]) for result in results), case_count
        ),
    }
    return {"summary": summary, "cases": results}
