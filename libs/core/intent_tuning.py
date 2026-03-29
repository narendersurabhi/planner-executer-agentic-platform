from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _non_empty_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for entry in value:
        item = _non_empty_string(entry)
        if item is None or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def build_intent_tuning_rows(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        feedback = _as_mapping(item.get("feedback"))
        dimensions = _as_mapping(item.get("dimensions"))
        linked_ids = _as_mapping(item.get("linked_ids"))
        observed_case = _as_mapping(item.get("observed_case"))
        gold_case_stub = _as_mapping(item.get("gold_case_stub"))
        if not feedback:
            continue
        rows.append(
            {
                "feedback_id": _non_empty_string(feedback.get("id")),
                "target_id": _non_empty_string(feedback.get("target_id")),
                "target_type": _non_empty_string(feedback.get("target_type")),
                "sentiment": _non_empty_string(feedback.get("sentiment")),
                "reason_codes": _string_list(feedback.get("reason_codes")),
                "comment": _non_empty_string(feedback.get("comment")),
                "review_label": _non_empty_string(item.get("review_label")),
                "review_score": int(item.get("review_score") or 0),
                "tuning_focus": _non_empty_string(item.get("tuning_focus")),
                "suggested_case_id": _non_empty_string(item.get("suggested_case_id")),
                "goal": _non_empty_string(observed_case.get("goal")),
                "observed_profile_intent": _non_empty_string(observed_case.get("profile_intent")),
                "observed_profile_source": _non_empty_string(observed_case.get("profile_source")),
                "observed_graph_intents": _string_list(observed_case.get("graph_intents")),
                "candidate_capabilities": _string_list(
                    observed_case.get("candidate_capabilities")
                ),
                "missing_inputs": _string_list(observed_case.get("missing_inputs")),
                "clarification_mode": _non_empty_string(
                    observed_case.get("clarification_mode")
                ),
                "disagreement_reason": _non_empty_string(
                    observed_case.get("disagreement_reason")
                ),
                "intent_assessment_intent": _non_empty_string(
                    dimensions.get("intent_assessment_intent")
                ),
                "intent_assessment_source": _non_empty_string(
                    dimensions.get("intent_assessment_source")
                ),
                "intent_top_capability": _non_empty_string(dimensions.get("intent_top_capability")),
                "intent_top_family": _non_empty_string(dimensions.get("intent_top_family")),
                "job_id": _non_empty_string(linked_ids.get("job_id")),
                "plan_id": _non_empty_string(linked_ids.get("plan_id")),
                "session_id": _non_empty_string(linked_ids.get("session_id")),
                "gold_case_stub": gold_case_stub,
            }
        )
    return rows


def build_intent_tuning_gold_cases(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in items:
        gold_case_stub = _as_mapping(item.get("gold_case_stub"))
        suggested_case_id = _non_empty_string(item.get("suggested_case_id"))
        if suggested_case_id is not None and not _non_empty_string(gold_case_stub.get("id")):
            gold_case_stub["id"] = suggested_case_id
        if gold_case_stub:
            cases.append(gold_case_stub)
    return cases


def build_intent_tuning_gold_bundle(
    items: Iterable[Mapping[str, Any]],
    *,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "description": description
        or "Intent tuning candidates exported from reviewed feedback.",
        "cases": build_intent_tuning_gold_cases(items),
    }


def dumps_intent_tuning_rows_jsonl(items: Iterable[Mapping[str, Any]]) -> str:
    rows = build_intent_tuning_rows(items)
    return "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows)


def dumps_intent_tuning_gold_yaml(
    items: Iterable[Mapping[str, Any]],
    *,
    description: str | None = None,
) -> str:
    if yaml is None:
        raise RuntimeError("pyyaml_required_for_intent_tuning_yaml_export")
    payload = build_intent_tuning_gold_bundle(items, description=description)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
