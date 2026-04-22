from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from . import chat_routing_calibrator


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


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


def _candidate_type(candidate_id: str) -> str:
    if candidate_id.startswith("workflow:"):
        return "workflow"
    if candidate_id.startswith("generic:"):
        return "generic_path"
    return "direct_agent"


def _default_route(candidate_type: str) -> str:
    if candidate_type == "workflow":
        return "run_workflow"
    if candidate_type == "generic_path":
        return "submit_job"
    return "tool_call"


def _candidate_payloads(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    top_k_candidates = _normalize_str_list(row.get("top_k_candidates"))
    payloads: list[dict[str, Any]] = []
    total = len(top_k_candidates)
    for index, candidate_id in enumerate(top_k_candidates):
        candidate_type = _candidate_type(candidate_id)
        payloads.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "route": _default_route(candidate_type),
                "score": float(total - index),
                "reason_codes": [],
                "metadata": {},
            }
        )
    return payloads


def _candidate_label(row: Mapping[str, Any], candidate_id: str | None) -> str:
    normalized = _normalize_str(candidate_id)
    if not normalized:
        return "unknown"
    if normalized in _normalize_str_list(row.get("selected_positive_ids")):
        return "positive"
    negative_ids = set(_normalize_str_list(row.get("selected_negative_ids")))
    negative_ids.update(_normalize_str_list(row.get("hard_negative_ids")))
    if normalized in negative_ids:
        return "negative"
    return "unknown"


def evaluate_feedback_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    model: Mapping[str, Any] | None,
    min_probability: float = 0.65,
    min_margin: float = 0.08,
) -> dict[str, Any]:
    total_rows = 0
    disagreement_count = 0
    live_override_count = 0
    shadow_better_count = 0
    shadow_worse_count = 0
    live_better_count = 0
    live_worse_count = 0
    shadow_reason_counts: Counter[str] = Counter()
    live_reason_counts: Counter[str] = Counter()
    feedback_label_counts: Counter[str] = Counter()
    shadow_outcome_counts: Counter[str] = Counter()
    live_outcome_counts: Counter[str] = Counter()
    case_rows: list[dict[str, Any]] = []

    model_payload = dict(model or {})
    for row in rows:
        top_k_candidates = _normalize_str_list(row.get("top_k_candidates"))
        base_selected_candidate_id = _normalize_str(row.get("selected_candidate_id")) or (
            top_k_candidates[0] if top_k_candidates else None
        )
        if not top_k_candidates or base_selected_candidate_id is None:
            continue
        total_rows += 1
        feedback_label = _normalize_str(row.get("feedback_label")) or "none"
        feedback_label_counts[feedback_label] += 1

        payloads = _candidate_payloads(row)
        shadow = chat_routing_calibrator.calibrate_route_candidates(
            candidates=payloads,
            model=model_payload,
            live=False,
            limit=len(payloads),
        )
        live = chat_routing_calibrator.calibrate_route_candidates(
            candidates=payloads,
            model=model_payload,
            live=True,
            min_probability=min_probability,
            min_margin=min_margin,
            limit=len(payloads),
        )
        shadow_summary = dict(shadow.get("summary") or {})
        live_summary = dict(live.get("summary") or {})
        shadow_selected_candidate_id = _normalize_str(shadow_summary.get("shadow_selected_candidate_id")) or None
        live_selected_candidate_id = (
            _normalize_str(live_summary.get("shadow_selected_candidate_id"))
            if bool(live_summary.get("live_override_used"))
            else base_selected_candidate_id
        ) or None

        if shadow_selected_candidate_id and shadow_selected_candidate_id != base_selected_candidate_id:
            disagreement_count += 1
        if bool(live_summary.get("live_override_used")):
            live_override_count += 1

        base_label = _candidate_label(row, base_selected_candidate_id)
        shadow_label = _candidate_label(row, shadow_selected_candidate_id)
        live_label = _candidate_label(row, live_selected_candidate_id)

        shadow_outcome = "neutral"
        if base_label != shadow_label:
            if shadow_label == "positive" and base_label != "positive":
                shadow_better_count += 1
                shadow_outcome = "better"
            elif base_label == "positive" and shadow_label != "positive":
                shadow_worse_count += 1
                shadow_outcome = "worse"
            elif shadow_label == "negative" and base_label != "negative":
                shadow_worse_count += 1
                shadow_outcome = "worse"
            elif base_label == "negative" and shadow_label != "negative":
                shadow_better_count += 1
                shadow_outcome = "better"
        shadow_outcome_counts[shadow_outcome] += 1

        live_outcome = "neutral"
        if base_label != live_label:
            if live_label == "positive" and base_label != "positive":
                live_better_count += 1
                live_outcome = "better"
            elif base_label == "positive" and live_label != "positive":
                live_worse_count += 1
                live_outcome = "worse"
            elif live_label == "negative" and base_label != "negative":
                live_worse_count += 1
                live_outcome = "worse"
            elif base_label == "negative" and live_label != "negative":
                live_better_count += 1
                live_outcome = "better"
        live_outcome_counts[live_outcome] += 1

        shadow_reason = _normalize_str(live_summary.get("live_override_reason"))
        if shadow_reason:
            live_reason_counts[shadow_reason] += 1
        shadow_mode_reason = _normalize_str(shadow_summary.get("mode")) or "shadow"
        shadow_reason_counts[shadow_mode_reason] += 1

        case_rows.append(
            {
                "feedback_id": _normalize_str(row.get("feedback_id")),
                "query": _normalize_str(row.get("query")),
                "feedback_label": feedback_label,
                "base_selected_candidate_id": base_selected_candidate_id,
                "shadow_selected_candidate_id": shadow_selected_candidate_id,
                "live_selected_candidate_id": live_selected_candidate_id,
                "base_label": base_label,
                "shadow_label": shadow_label,
                "live_label": live_label,
                "shadow_outcome": shadow_outcome,
                "live_outcome": live_outcome,
                "shadow_top_k_candidates": _normalize_str_list(
                    shadow_summary.get("shadow_top_k_candidates")
                ),
                "live_override_used": bool(live_summary.get("live_override_used")),
                "live_override_reason": _normalize_str(live_summary.get("live_override_reason"))
                or None,
            }
        )

    return {
        "summary": {
            "row_count": total_rows,
            "disagreement_count": disagreement_count,
            "disagreement_rate": _safe_div(disagreement_count, total_rows),
            "live_override_count": live_override_count,
            "live_override_rate": _safe_div(live_override_count, total_rows),
            "shadow_better_count": shadow_better_count,
            "shadow_worse_count": shadow_worse_count,
            "live_better_count": live_better_count,
            "live_worse_count": live_worse_count,
            "shadow_better_rate": _safe_div(shadow_better_count, total_rows),
            "shadow_worse_rate": _safe_div(shadow_worse_count, total_rows),
            "live_better_rate": _safe_div(live_better_count, total_rows),
            "live_worse_rate": _safe_div(live_worse_count, total_rows),
            "min_probability": float(min_probability),
            "min_margin": float(min_margin),
            "model_version": _normalize_str(model_payload.get("model_version")) or None,
        },
        "feedback_labels": dict(sorted(feedback_label_counts.items())),
        "shadow_outcomes": dict(sorted(shadow_outcome_counts.items())),
        "live_outcomes": dict(sorted(live_outcome_counts.items())),
        "shadow_reasons": dict(sorted(shadow_reason_counts.items())),
        "live_override_reasons": dict(sorted(live_reason_counts.items())),
        "cases": case_rows,
    }
