from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_DEFAULT_RERANK_FEEDBACK_PATH = Path(
    os.getenv("CHAT_ROUTING_RERANK_FEEDBACK_PATH", "artifacts/evals/chat_routing_feedback.jsonl")
)


def _normalize_str(value: Any) -> str:
    return str(value or "").strip()


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for entry in value:
        text = _normalize_str(entry)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def _query_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize_str(value).lower().replace("\n", " ").split()
        if token and len(token) > 2
    }


def _query_match_weight(query: str, row_query: str) -> float:
    normalized_query = _normalize_str(query).lower()
    normalized_row_query = _normalize_str(row_query).lower()
    if not normalized_query or not normalized_row_query:
        return 0.0
    if normalized_query == normalized_row_query:
        return 1.0
    if normalized_query in normalized_row_query or normalized_row_query in normalized_query:
        return 0.8
    overlap = _query_tokens(normalized_query) & _query_tokens(normalized_row_query)
    if len(overlap) >= 3:
        return 0.6
    if len(overlap) >= 2:
        return 0.4
    return 0.0


def load_feedback_rows(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _DEFAULT_RERANK_FEEDBACK_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def rerank_route_candidates(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if feedback_rows is None and os.getenv("CHAT_ROUTING_RERANK_ENABLED", "true").lower() == "true":
        feedback_rows = load_feedback_rows()
    if not feedback_rows:
        return candidates[: max(1, limit or len(candidates))]

    adjusted: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = _normalize_str(candidate.get("candidate_id") or candidate.get("id"))
        if not candidate_id:
            continue
        candidate_type = _normalize_str(candidate.get("candidate_type"))
        score = float(candidate.get("score") or 0.0)
        reason_codes = _normalize_str_list(candidate.get("reason_codes"))
        positive_hits = 0
        negative_hits = 0
        fallback_penalties = 0
        hard_negative_hits = 0
        for row in feedback_rows:
            match_weight = _query_match_weight(query, _normalize_str(row.get("query")))
            if match_weight <= 0.0:
                continue
            positive_ids = _normalize_str_list(row.get("selected_positive_ids"))
            negative_ids = _normalize_str_list(row.get("selected_negative_ids"))
            hard_negative_ids = _normalize_str_list(row.get("hard_negative_ids"))
            selected_candidate_type = _normalize_str(row.get("selected_candidate_type"))
            if candidate_id in positive_ids:
                score += 12.0 * match_weight
                positive_hits += 1
            if candidate_id in negative_ids:
                score -= 10.0 * match_weight
                negative_hits += 1
            if candidate_id in hard_negative_ids:
                score -= 6.0 * match_weight
                hard_negative_hits += 1
            if (
                candidate_type == "generic_path"
                and selected_candidate_type == "generic_path"
                and bool(row.get("fallback_used"))
                and _normalize_str(row.get("feedback_label")) in {"negative", "partial"}
            ):
                score -= 4.0 * match_weight
                fallback_penalties += 1
            if (
                candidate_type
                and candidate_type == selected_candidate_type
                and _normalize_str(row.get("feedback_label")) == "positive"
                and candidate_id not in positive_ids
            ):
                score += 2.0 * match_weight
        if positive_hits:
            reason_codes.append(f"routing_rerank_positive={positive_hits}")
        if negative_hits:
            reason_codes.append(f"routing_rerank_negative={negative_hits}")
        if hard_negative_hits:
            reason_codes.append(f"routing_rerank_hard_negative={hard_negative_hits}")
        if fallback_penalties:
            reason_codes.append(f"routing_rerank_fallback_penalty={fallback_penalties}")
        adjusted.append(
            {
                **candidate,
                "candidate_id": candidate_id,
                "score": round(score, 3),
                "reason_codes": reason_codes,
                "metadata": {
                    **(dict(candidate.get("metadata")) if isinstance(candidate.get("metadata"), dict) else {}),
                    "reranked": True,
                },
            }
        )
    adjusted.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            _normalize_str(item.get("candidate_id") or item.get("id")),
        )
    )
    return adjusted[: max(1, limit or len(adjusted))]
