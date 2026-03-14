from __future__ import annotations

import json
from pathlib import Path
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


def load_feedback_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def rerank_capability_results(
    *,
    query: str,
    intent_hint: str | None,
    results: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not results or not feedback_rows:
        return results[: max(1, limit or len(results))]

    normalized_query = _normalize_str(query).lower()
    normalized_intent = _normalize_str(intent_hint).lower()
    adjusted: list[dict[str, Any]] = []
    for result in results:
        capability_id = _normalize_str(result.get("id"))
        if not capability_id:
            continue
        score = float(result.get("score") or 0.0)
        reasons = [str(result.get("reason") or "").strip() or "semantic match"]
        selected_hits = 0
        executed_hits = 0
        failure_hits = 0
        hard_negative_hits = 0
        query_hits = 0
        for row in feedback_rows:
            row_queries = [item.lower() for item in _normalize_str_list(row.get("search_queries"))]
            row_selected = _normalize_str_list(row.get("retrieved_selected"))
            row_executed = _normalize_str_list(row.get("retrieved_executed"))
            row_negatives = _normalize_str_list(row.get("hard_negative_ids"))
            if normalized_query and normalized_query in row_queries:
                query_hits += 1
                if capability_id in row_selected:
                    score += 10.0
                    selected_hits += 1
                if capability_id in row_executed:
                    score += 14.0
                    executed_hits += 1
                if capability_id in row_negatives:
                    score -= 12.0
                    hard_negative_hits += 1
                continue
            if capability_id in row_selected:
                score += 3.0
                selected_hits += 1
            if capability_id in row_executed:
                score += 4.0
                executed_hits += 1
            if capability_id in row_negatives:
                score -= 4.0
                hard_negative_hits += 1
            executed_capabilities = row.get("executed_capabilities")
            if isinstance(executed_capabilities, list):
                for entry in executed_capabilities:
                    if not isinstance(entry, dict):
                        continue
                    if _normalize_str(entry.get("id")) != capability_id:
                        continue
                    if _normalize_str(entry.get("status")) == "failed":
                        score -= 5.0
                        failure_hits += 1

        if query_hits:
            reasons.append(f"rerank query-hit={query_hits}")
        if executed_hits:
            reasons.append(f"rerank executed={executed_hits}")
        if selected_hits:
            reasons.append(f"rerank selected={selected_hits}")
        if failure_hits:
            reasons.append(f"rerank failures={failure_hits}")
        if hard_negative_hits:
            reasons.append(f"rerank hard_negatives={hard_negative_hits}")
        adjusted.append(
            {
                **result,
                "score": round(score, 3),
                "reason": "; ".join(reasons),
                "source": str(result.get("source") or "semantic_search"),
                "reranked": True,
                "rerank_intent": normalized_intent or None,
            }
        )
    adjusted.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    return adjusted[: max(1, limit or len(adjusted))]
