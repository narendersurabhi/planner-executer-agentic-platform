from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_DEFAULT_MODEL_PATH = Path(
    os.getenv("CHAT_ROUTING_CALIBRATOR_MODEL_PATH", "artifacts/evals/chat_routing_calibrator.json")
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


def _candidate_type(candidate_id: str, declared_type: str | None = None) -> str:
    normalized = _normalize_str(declared_type)
    if normalized:
        return normalized
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


def _rank_bucket(rank: int) -> str:
    if rank <= 0:
        return "0"
    if rank == 1:
        return "1"
    if rank == 2:
        return "2"
    return "3+"


def _top_k_bucket(size: int) -> str:
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    return "4+"


def _feature_keys(*, candidate_id: str, candidate_type: str) -> list[str]:
    return [
        "__bias__",
        f"candidate_type={candidate_type}",
        f"candidate_id={candidate_id}",
    ]


def build_pointwise_examples(
    reranker_rows: list[dict[str, Any]] | list[dict],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in reranker_rows:
        route = _normalize_str(row.get("route") or row.get("decision_route"))
        positive_candidate_id = _normalize_str(row.get("positive_candidate_id"))
        negative_candidate_ids = _normalize_str_list(row.get("negative_candidate_ids"))
        top_k_candidates = _normalize_str_list(row.get("top_k_candidates"))
        if not route or not positive_candidate_id or not negative_candidate_ids:
            continue
        candidate_ids: list[str] = []
        for candidate_id in [positive_candidate_id, *negative_candidate_ids, *top_k_candidates]:
            if candidate_id and candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
        for candidate_id in candidate_ids:
            label = 1 if candidate_id == positive_candidate_id else 0
            if label == 0 and candidate_id not in negative_candidate_ids:
                continue
            rank = top_k_candidates.index(candidate_id) if candidate_id in top_k_candidates else len(top_k_candidates)
            candidate_type = _candidate_type(candidate_id)
            examples.append(
                {
                    "feedback_id": _normalize_str(row.get("feedback_id")),
                    "query": _normalize_str(row.get("query")),
                    "route": route,
                    "candidate_id": candidate_id,
                    "candidate_type": candidate_type,
                    "candidate_rank": rank,
                    "top_k_size": max(len(top_k_candidates), len(candidate_ids)),
                    "label": label,
                    "weight": 1.0 if bool(row.get("execution_succeeded")) else 0.75,
                    "source": _normalize_str(row.get("source")) or "routing_feedback",
                }
            )
    return examples


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def train_model(
    reranker_rows: list[dict[str, Any]] | list[dict],
    *,
    epochs: int = 120,
    learning_rate: float = 0.18,
    l2: float = 0.0005,
) -> dict[str, Any]:
    examples = build_pointwise_examples(reranker_rows)
    weights: dict[str, float] = {}
    if not examples:
        return {
            "model_version": "chat-routing-calibrator-v1",
            "trained_at": datetime.now(UTC).isoformat(),
            "weights": weights,
            "training_examples": 0,
            "positive_examples": 0,
            "negative_examples": 0,
            "epochs": 0,
            "learning_rate": learning_rate,
            "l2": l2,
            "accuracy": 0.0,
            "log_loss": 0.0,
        }

    for _epoch in range(max(1, epochs)):
        for example in examples:
            feature_keys = _feature_keys(
                candidate_id=_normalize_str(example.get("candidate_id")),
                candidate_type=_normalize_str(example.get("candidate_type")),
            )
            label = 1.0 if int(example.get("label") or 0) > 0 else 0.0
            sample_weight = max(0.1, float(example.get("weight") or 1.0))
            linear = sum(weights.get(key, 0.0) for key in feature_keys)
            prediction = _sigmoid(linear)
            error = (label - prediction) * sample_weight
            for key in feature_keys:
                updated = weights.get(key, 0.0) + (learning_rate * error)
                updated -= learning_rate * l2 * updated
                weights[key] = updated

    correct = 0
    log_loss = 0.0
    positive_count = 0
    negative_count = 0
    for example in examples:
        probability = predict_probability(
            {
                "weights": weights,
            },
            candidate_id=_normalize_str(example.get("candidate_id")),
            candidate_type=_normalize_str(example.get("candidate_type")),
        )
        label = 1 if int(example.get("label") or 0) > 0 else 0
        if label:
            positive_count += 1
        else:
            negative_count += 1
        if (probability >= 0.5 and label == 1) or (probability < 0.5 and label == 0):
            correct += 1
        clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
        log_loss += -(label * math.log(clipped) + ((1 - label) * math.log(1.0 - clipped)))

    return {
        "model_version": "chat-routing-calibrator-v1",
        "trained_at": datetime.now(UTC).isoformat(),
        "weights": {key: round(value, 6) for key, value in sorted(weights.items())},
        "training_examples": len(examples),
        "positive_examples": positive_count,
        "negative_examples": negative_count,
        "epochs": max(1, epochs),
        "learning_rate": learning_rate,
        "l2": l2,
        "accuracy": round(correct / max(1, len(examples)), 4),
        "log_loss": round(log_loss / max(1, len(examples)), 6),
    }


def save_model(model: dict[str, Any], path: Path | None = None) -> Path:
    target = path or _DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_model(path: Path | None = None) -> dict[str, Any] | None:
    target = path or _DEFAULT_MODEL_PATH
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else None


def predict_probability(
    model: dict[str, Any] | dict,
    *,
    candidate_id: str,
    candidate_type: str,
) -> float:
    weights = dict(model.get("weights") or {})
    linear = 0.0
    for key in _feature_keys(
        candidate_id=_normalize_str(candidate_id),
        candidate_type=_normalize_str(candidate_type),
    ):
        linear += float(weights.get(key) or 0.0)
    return round(_sigmoid(linear), 6)


def calibrate_route_candidates(
    *,
    candidates: list[dict[str, Any]],
    model: dict[str, Any] | None = None,
    live: bool = False,
    min_probability: float = 0.65,
    min_margin: float = 0.08,
    limit: int | None = None,
) -> dict[str, Any]:
    if not candidates or model is None:
        return {"candidates": candidates[: max(1, limit or len(candidates))], "summary": {}}

    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = _normalize_str(candidate.get("candidate_id") or candidate.get("id"))
        if not candidate_id:
            continue
        candidate_type = _candidate_type(
            candidate_id,
            declared_type=_normalize_str(candidate.get("candidate_type")),
        )
        route = _normalize_str(candidate.get("route")) or _default_route(candidate_type)
        probability = predict_probability(
            model,
            candidate_id=candidate_id,
            candidate_type=candidate_type,
        )
        reason_codes = _normalize_str_list(candidate.get("reason_codes"))
        reason_codes.append(f"routing_calibration={probability:.3f}")
        metadata = (
            dict(candidate.get("metadata"))
            if isinstance(candidate.get("metadata"), dict)
            else {}
        )
        metadata.update(
            {
                "calibration_probability": probability,
                "calibration_mode": "live" if live else "shadow",
                "calibration_model_version": _normalize_str(model.get("model_version"))
                or "chat-routing-calibrator-v1",
            }
        )
        annotated.append(
            {
                **candidate,
                "candidate_id": candidate_id,
                "candidate_type": candidate_type,
                "route": route,
                "reason_codes": reason_codes,
                "metadata": metadata,
            }
        )

    shadow_sorted = sorted(
        annotated,
        key=lambda item: (
            -float((item.get("metadata") or {}).get("calibration_probability") or 0.0),
            -float(item.get("score") or 0.0),
            _normalize_str(item.get("candidate_id")),
        ),
    )
    base_top_k = [_normalize_str(item.get("candidate_id")) for item in annotated[:5]]
    base_selected_candidate_id = base_top_k[0] if base_top_k else None
    shadow_top_k = [_normalize_str(item.get("candidate_id")) for item in shadow_sorted[:5]]
    shadow_selected_candidate_id = shadow_top_k[0] if shadow_top_k else None
    probability_by_id = {
        _normalize_str(item.get("candidate_id")): float(
            (item.get("metadata") or {}).get("calibration_probability") or 0.0
        )
        for item in annotated
        if _normalize_str(item.get("candidate_id"))
    }
    shadow_selected_probability = (
        float(probability_by_id.get(shadow_selected_candidate_id) or 0.0)
        if shadow_selected_candidate_id
        else 0.0
    )
    base_selected_probability = (
        float(probability_by_id.get(base_selected_candidate_id) or 0.0)
        if base_selected_candidate_id
        else 0.0
    )
    second_probability = 0.0
    if len(shadow_top_k) > 1:
        second_probability = float(probability_by_id.get(shadow_top_k[1]) or 0.0)
    live_override_used = False
    live_override_reason = "live_disabled"
    if live:
        if shadow_selected_candidate_id is None:
            live_override_reason = "no_shadow_candidate"
        elif shadow_selected_candidate_id == base_selected_candidate_id:
            live_override_reason = "same_as_deterministic"
        elif shadow_selected_probability < min_probability:
            live_override_reason = "top_probability_below_threshold"
        elif (shadow_selected_probability - second_probability) < min_margin:
            live_override_reason = "top_margin_below_threshold"
        else:
            live_override_used = True
            live_override_reason = "applied"
    if not live_override_used:
        shadow_rank_by_id = {
            _normalize_str(item.get("candidate_id")): index
            for index, item in enumerate(shadow_sorted)
            if _normalize_str(item.get("candidate_id"))
        }
        for item in annotated:
            candidate_id = _normalize_str(item.get("candidate_id"))
            metadata = dict(item.get("metadata") or {})
            metadata["shadow_rank"] = shadow_rank_by_id.get(candidate_id, len(annotated))
            item["metadata"] = metadata
    output_candidates = shadow_sorted if live_override_used else annotated
    return {
        "candidates": output_candidates[: max(1, limit or len(output_candidates))],
        "summary": {
            "mode": "live" if live_override_used else "shadow",
            "model_version": _normalize_str(model.get("model_version"))
            or "chat-routing-calibrator-v1",
            "live_requested": live,
            "live_override_used": live_override_used,
            "live_override_reason": live_override_reason,
            "base_selected_candidate_id": base_selected_candidate_id,
            "base_selected_confidence": round(base_selected_probability, 6)
            if base_selected_candidate_id
            else None,
            "shadow_selected_candidate_id": shadow_selected_candidate_id,
            "shadow_selected_confidence": round(shadow_selected_probability, 6)
            if shadow_selected_candidate_id
            else None,
            "shadow_probability_margin": round(shadow_selected_probability - second_probability, 6)
            if shadow_selected_candidate_id
            else None,
            "min_probability": round(float(min_probability), 6),
            "min_margin": round(float(min_margin), 6),
            "base_top_k_candidates": base_top_k,
            "shadow_top_k_candidates": shadow_top_k,
            "probability_by_candidate_id": {
                candidate_id: round(probability, 6)
                for candidate_id, probability in sorted(probability_by_id.items())
            },
        },
    }
