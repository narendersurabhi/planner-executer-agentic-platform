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


def _feedback_payload(example: Mapping[str, Any]) -> dict[str, Any]:
    feedback = example.get("feedback")
    return dict(feedback) if isinstance(feedback, Mapping) else {}


def _dimensions_payload(example: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = example.get("dimensions")
    if isinstance(dimensions, Mapping):
        return dict(dimensions)
    metadata = feedback.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("dimensions"), Mapping):
        return dict(metadata.get("dimensions"))
    return {}


def _snapshot_payload(example: Mapping[str, Any], feedback: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = example.get("snapshot")
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    if isinstance(feedback.get("snapshot"), Mapping):
        return dict(feedback.get("snapshot"))
    return {}


def build_feedback_rows(examples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        feedback = _feedback_payload(example)
        dimensions = _dimensions_payload(example, feedback)
        snapshot = _snapshot_payload(example, feedback)
        metadata = snapshot.get("metadata")
        snapshot_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        routing_decision = snapshot_metadata.get("routing_decision")
        routing_payload = dict(routing_decision) if isinstance(routing_decision, Mapping) else {}

        query = _normalize_str(
            snapshot.get("user_message_content")
            or snapshot.get("candidate_goal")
            or snapshot.get("goal")
        )
        route = _normalize_str(
            dimensions.get("routing_decision_route") or routing_payload.get("route")
        )
        selected_candidate_id = _normalize_str(
            dimensions.get("routing_selected_candidate_id")
            or routing_payload.get("selected_candidate_id")
        )
        if not query or not route:
            continue

        top_k_candidates = _normalize_str_list(routing_payload.get("top_k_candidates"))
        feedback_label = _normalize_str(feedback.get("sentiment"))
        fallback_used = _normalize_str(dimensions.get("routing_fallback_used")) == "yes"
        execution_succeeded = _normalize_str(dimensions.get("routing_execution_succeeded")) == "yes"

        hard_negative_ids: list[str] = []
        if feedback_label == "positive":
            hard_negative_ids = [
                candidate_id
                for candidate_id in top_k_candidates
                if candidate_id != selected_candidate_id
            ]

        rows.append(
            {
                "feedback_id": _normalize_str(feedback.get("id")),
                "session_id": _normalize_str(feedback.get("session_id")),
                "message_id": _normalize_str(feedback.get("message_id")),
                "query": query,
                "decision_route": route,
                "selected_candidate_id": selected_candidate_id or None,
                "selected_candidate_type": _normalize_str(
                    dimensions.get("routing_selected_candidate_type")
                )
                or None,
                "top_k_candidates": top_k_candidates,
                "fallback_used": fallback_used,
                "fallback_reason": _normalize_str(dimensions.get("routing_fallback_reason")) or None,
                "execution_started": _normalize_str(dimensions.get("routing_execution_started"))
                == "yes",
                "execution_succeeded": execution_succeeded,
                "feedback_label": feedback_label or "none",
                "reason_codes": _normalize_str_list(feedback.get("reason_codes")),
                "selected_positive_ids": (
                    [selected_candidate_id]
                    if selected_candidate_id and feedback_label == "positive"
                    else []
                ),
                "selected_negative_ids": (
                    [selected_candidate_id]
                    if selected_candidate_id and feedback_label in {"negative", "partial"}
                    else []
                ),
                "hard_negative_ids": hard_negative_ids,
            }
        )
    rows.sort(key=lambda item: (str(item.get("session_id") or ""), str(item.get("message_id") or "")))
    return rows
