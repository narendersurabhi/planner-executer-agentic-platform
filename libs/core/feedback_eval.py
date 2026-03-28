from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def build_feedback_eval_rows(examples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        feedback = example.get("feedback")
        dimensions = example.get("dimensions")
        snapshot = example.get("snapshot")
        linked_ids = example.get("linked_ids")
        if not isinstance(feedback, Mapping):
            continue
        snapshot_payload = dict(snapshot) if isinstance(snapshot, Mapping) else {}
        boundary_decision = None
        boundary_evidence = None
        snapshot_metadata = snapshot_payload.get("metadata")
        if isinstance(snapshot_metadata, Mapping):
            raw_boundary = snapshot_metadata.get("boundary_decision")
            if isinstance(raw_boundary, Mapping):
                boundary_decision = str(raw_boundary.get("decision") or "").strip() or None
                boundary_evidence = (
                    dict(raw_boundary.get("evidence"))
                    if isinstance(raw_boundary.get("evidence"), Mapping)
                    else None
                )
        rows.append(
            {
                "feedback_id": str(feedback.get("id") or "").strip(),
                "target_type": str(feedback.get("target_type") or "").strip(),
                "target_id": str(feedback.get("target_id") or "").strip(),
                "sentiment": str(feedback.get("sentiment") or "").strip(),
                "reason_codes": list(feedback.get("reason_codes") or []),
                "comment": str(feedback.get("comment") or "").strip() or None,
                "snapshot": snapshot_payload,
                "dimensions": dict(dimensions) if isinstance(dimensions, Mapping) else {},
                "linked_ids": dict(linked_ids) if isinstance(linked_ids, Mapping) else {},
                "boundary_decision": boundary_decision,
                "boundary_evidence": boundary_evidence,
            }
        )
    return rows


def dumps_feedback_eval_rows_jsonl(examples: Iterable[Mapping[str, Any]]) -> str:
    rows = build_feedback_eval_rows(examples)
    return "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows)
