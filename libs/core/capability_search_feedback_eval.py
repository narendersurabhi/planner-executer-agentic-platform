from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def evaluate_feedback_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    rows_with_selected = 0
    rows_with_executed = 0
    rows_with_overrides = 0
    planner_selection_hits = 0
    execution_successes = 0
    retrieved_total = 0
    hard_negative_total = 0

    for row in rows:
        total_rows += 1
        selected = row.get("selected_capabilities")
        if isinstance(selected, list) and selected:
            rows_with_selected += 1
        retrieved_selected = row.get("retrieved_selected")
        if isinstance(retrieved_selected, list) and retrieved_selected:
            planner_selection_hits += 1
        executed = row.get("executed_capabilities")
        if isinstance(executed, list) and executed:
            rows_with_executed += 1
        if bool(row.get("execution_succeeded")):
            execution_successes += 1
        overrides = row.get("planner_override")
        if isinstance(overrides, list) and overrides:
            rows_with_overrides += 1
        retrieved = row.get("retrieved_capabilities")
        if isinstance(retrieved, list):
            retrieved_total += len(retrieved)
        hard_negatives = row.get("hard_negative_ids")
        if isinstance(hard_negatives, list):
            hard_negative_total += len(hard_negatives)

    return {
        "row_count": total_rows,
        "planner_selection_hit_rate": _safe_div(planner_selection_hits, rows_with_selected),
        "execution_success_rate": _safe_div(execution_successes, rows_with_executed),
        "planner_override_rate": _safe_div(rows_with_overrides, rows_with_selected),
        "hard_negative_ratio": _safe_div(hard_negative_total, retrieved_total),
        "rows_with_selected": rows_with_selected,
        "rows_with_executed": rows_with_executed,
        "retrieved_total": retrieved_total,
        "hard_negative_total": hard_negative_total,
    }
