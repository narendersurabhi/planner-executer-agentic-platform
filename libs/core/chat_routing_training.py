from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


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


def build_reranker_training_examples(feedback_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in feedback_rows:
        query = _normalize_str(row.get("query"))
        positives = _normalize_str_list(row.get("selected_positive_ids"))
        negatives = _normalize_str_list(row.get("hard_negative_ids"))
        if not query or not positives or not negatives:
            continue
        for positive_id in positives:
            if positive_id in negatives:
                continue
            examples.append(
                {
                    "feedback_id": _normalize_str(row.get("feedback_id")),
                    "query": query,
                    "route": _normalize_str(row.get("decision_route")),
                    "positive_candidate_id": positive_id,
                    "negative_candidate_ids": negatives,
                    "top_k_candidates": _normalize_str_list(row.get("top_k_candidates")),
                    "execution_succeeded": bool(row.get("execution_succeeded")),
                    "source": "routing_feedback",
                }
            )
    return examples
