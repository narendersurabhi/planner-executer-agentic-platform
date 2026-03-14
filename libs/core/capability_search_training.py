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
        query_candidates = _normalize_str_list(row.get("search_queries"))
        if not query_candidates:
            continue
        query = query_candidates[0]
        executed_capabilities = []
        raw_executed = row.get("executed_capabilities")
        if isinstance(raw_executed, list):
            for entry in raw_executed:
                if not isinstance(entry, Mapping):
                    continue
                capability_id = _normalize_str(entry.get("id"))
                status = _normalize_str(entry.get("status"))
                if capability_id and status == "completed" and capability_id not in executed_capabilities:
                    executed_capabilities.append(capability_id)
        retrieved_selected = _normalize_str_list(row.get("retrieved_selected"))
        positives = executed_capabilities or retrieved_selected
        negatives = _normalize_str_list(row.get("hard_negative_ids"))
        if not positives or not negatives:
            continue
        for positive_id in positives:
            if positive_id in negatives:
                continue
            examples.append(
                {
                    "job_id": _normalize_str(row.get("job_id")),
                    "correlation_id": _normalize_str(row.get("correlation_id")),
                    "query": query,
                    "positive_capability_id": positive_id,
                    "negative_capability_ids": negatives,
                    "selected_capabilities": _normalize_str_list(row.get("selected_capabilities")),
                    "retrieved_capabilities": _normalize_str_list(row.get("retrieved_capabilities")),
                    "execution_succeeded": bool(row.get("execution_succeeded")),
                    "source": "feedback",
                }
            )
    return examples
