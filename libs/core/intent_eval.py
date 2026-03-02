from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


@dataclass(frozen=True)
class IntentEvalCase:
    case_id: str
    goal: str
    expected_intents: tuple[str, ...]
    expected_capabilities: tuple[str, ...]
    expected_capabilities_by_segment: tuple[tuple[str, ...], ...]


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for entry in value:
        text = str(entry or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _coerce_segment_capability_lists(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        return ()
    normalized: list[tuple[str, ...]] = []
    for entry in value:
        if isinstance(entry, list):
            normalized.append(tuple(_coerce_string_list(entry)))
            continue
        text = str(entry or "").strip()
        if text:
            normalized.append((text,))
    return tuple(normalized)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _score_sets(expected: set[str], predicted: set[str]) -> dict[str, float]:
    tp = len(expected.intersection(predicted))
    fp = len(predicted.difference(expected))
    fn = len(expected.difference(predicted))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_intent_eval_cases(path: Path) -> list[IntentEvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("intent_eval_cases_invalid_root")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("intent_eval_cases_missing_cases")
    cases: list[IntentEvalCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            continue
        case_id = str(raw_case.get("id") or f"case_{index + 1}").strip()
        goal = str(raw_case.get("goal") or "").strip()
        expected_intents = tuple(_coerce_string_list(raw_case.get("expected_intents")))
        expected_capabilities = tuple(_coerce_string_list(raw_case.get("expected_capabilities")))
        expected_capabilities_by_segment = _coerce_segment_capability_lists(
            raw_case.get("expected_capabilities_by_segment")
        )
        if not case_id or not goal:
            continue
        cases.append(
            IntentEvalCase(
                case_id=case_id,
                goal=goal,
                expected_intents=expected_intents,
                expected_capabilities=expected_capabilities,
                expected_capabilities_by_segment=expected_capabilities_by_segment,
            )
        )
    if not cases:
        raise ValueError("intent_eval_cases_empty")
    return cases


def evaluate_intent_case(
    case: IntentEvalCase,
    graph: Mapping[str, Any] | None,
    *,
    top_k: int,
    allowed_capability_ids: set[str] | None = None,
) -> dict[str, Any]:
    raw_segments = graph.get("segments") if isinstance(graph, Mapping) else []
    segments = raw_segments if isinstance(raw_segments, list) else []
    predicted_intents: list[str] = []
    predicted_caps_by_segment: list[list[str]] = []
    for entry in segments:
        if not isinstance(entry, Mapping):
            continue
        intent = str(entry.get("intent") or "").strip()
        if intent:
            predicted_intents.append(intent)
        suggested = _coerce_string_list(entry.get("suggested_capabilities"))
        predicted_caps_by_segment.append(suggested)

    predicted_intents_unique = set(_unique(predicted_intents))
    expected_intents_unique = set(case.expected_intents)
    intent_scores = _score_sets(expected_intents_unique, predicted_intents_unique)

    ordered_hits = 0
    for index, expected_intent in enumerate(case.expected_intents):
        if index < len(predicted_intents) and predicted_intents[index] == expected_intent:
            ordered_hits += 1
    ordered_recall = _safe_div(ordered_hits, len(case.expected_intents))

    predicted_caps = _unique(
        [cap for segment_caps in predicted_caps_by_segment for cap in segment_caps]
    )
    predicted_caps_unique = set(predicted_caps)
    expected_caps_unique = set(case.expected_capabilities)
    capability_scores = _score_sets(expected_caps_unique, predicted_caps_unique)

    segment_slots = 0
    segment_hits = 0
    per_segment_hits: list[bool] = []
    for index, expected_candidates in enumerate(case.expected_capabilities_by_segment):
        expected_set = set(expected_candidates)
        if not expected_set:
            continue
        segment_slots += 1
        predicted_topk = (
            predicted_caps_by_segment[index][: max(1, top_k)]
            if index < len(predicted_caps_by_segment)
            else []
        )
        hit = bool(expected_set.intersection(predicted_topk))
        if hit:
            segment_hits += 1
        per_segment_hits.append(hit)
    topk_hit_rate = _safe_div(segment_hits, segment_slots) if segment_slots > 0 else 0.0

    invalid_capabilities: list[str] = []
    if allowed_capability_ids is not None:
        invalid_capabilities = sorted(
            capability_id
            for capability_id in predicted_caps_unique
            if capability_id not in allowed_capability_ids
        )

    return {
        "case_id": case.case_id,
        "goal": case.goal,
        "expected_intents": list(case.expected_intents),
        "predicted_intents": predicted_intents,
        "expected_capabilities": list(case.expected_capabilities),
        "predicted_capabilities": predicted_caps,
        "intent": {
            "precision": intent_scores["precision"],
            "recall": intent_scores["recall"],
            "f1": intent_scores["f1"],
            "ordered_recall": ordered_recall,
            "ordered_hits": ordered_hits,
            "ordered_total": len(case.expected_intents),
            "tp": int(intent_scores["tp"]),
            "fp": int(intent_scores["fp"]),
            "fn": int(intent_scores["fn"]),
        },
        "capabilities": {
            "precision": capability_scores["precision"],
            "recall": capability_scores["recall"],
            "f1": capability_scores["f1"],
            "tp": int(capability_scores["tp"]),
            "fp": int(capability_scores["fp"]),
            "fn": int(capability_scores["fn"]),
            "invalid": invalid_capabilities,
        },
        "segment_topk": {
            "top_k": max(1, top_k),
            "slots": segment_slots,
            "hits": segment_hits,
            "hit_rate": topk_hit_rate,
            "hits_by_segment": per_segment_hits,
        },
    }


def evaluate_intent_cases(
    cases: list[IntentEvalCase],
    *,
    decompose_goal: Callable[[str], Mapping[str, Any]],
    top_k: int = 3,
    allowed_capability_ids: set[str] | None = None,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k_must_be_positive")
    case_results: list[dict[str, Any]] = []
    intent_tp = intent_fp = intent_fn = 0
    capability_tp = capability_fp = capability_fn = 0
    segment_slots = segment_hits = 0
    ordered_intent_hits = 0
    ordered_intent_total = 0
    invalid_capabilities_total = 0
    predicted_capabilities_total = 0

    for case in cases:
        graph = decompose_goal(case.goal)
        result = evaluate_intent_case(
            case,
            graph,
            top_k=top_k,
            allowed_capability_ids=allowed_capability_ids,
        )
        case_results.append(result)

        intent_tp += int(result["intent"]["tp"])
        intent_fp += int(result["intent"]["fp"])
        intent_fn += int(result["intent"]["fn"])
        capability_tp += int(result["capabilities"]["tp"])
        capability_fp += int(result["capabilities"]["fp"])
        capability_fn += int(result["capabilities"]["fn"])
        segment_slots += int(result["segment_topk"]["slots"])
        segment_hits += int(result["segment_topk"]["hits"])
        ordered_intent_total += int(result["intent"]["ordered_total"])
        ordered_intent_hits += int(result["intent"]["ordered_hits"])
        invalid_capabilities_total += len(result["capabilities"]["invalid"])
        predicted_capabilities_total += len(result["predicted_capabilities"])

    intent_precision = _safe_div(intent_tp, intent_tp + intent_fp)
    intent_recall = _safe_div(intent_tp, intent_tp + intent_fn)
    intent_f1 = (
        _safe_div(2 * intent_precision * intent_recall, intent_precision + intent_recall)
        if (intent_precision + intent_recall) > 0
        else 0.0
    )
    capability_precision = _safe_div(capability_tp, capability_tp + capability_fp)
    capability_recall = _safe_div(capability_tp, capability_tp + capability_fn)
    capability_f1 = (
        _safe_div(
            2 * capability_precision * capability_recall,
            capability_precision + capability_recall,
        )
        if (capability_precision + capability_recall) > 0
        else 0.0
    )
    segment_hit_rate = _safe_div(segment_hits, segment_slots)

    summary = {
        "case_count": len(case_results),
        "intent_micro": {
            "precision": intent_precision,
            "recall": intent_recall,
            "f1": intent_f1,
            "tp": intent_tp,
            "fp": intent_fp,
            "fn": intent_fn,
        },
        "capability_micro": {
            "precision": capability_precision,
            "recall": capability_recall,
            "f1": capability_f1,
            "tp": capability_tp,
            "fp": capability_fp,
            "fn": capability_fn,
        },
        "segment_topk": {
            "hit_rate": segment_hit_rate,
            "hits": segment_hits,
            "slots": segment_slots,
            "top_k": max(1, top_k),
        },
        "invalid_capabilities": {
            "count": invalid_capabilities_total,
            "rate": _safe_div(invalid_capabilities_total, predicted_capabilities_total),
            "predicted_total": predicted_capabilities_total,
        },
        "intent_ordered_proxy": {
            "hit_rate": _safe_div(ordered_intent_hits, ordered_intent_total),
            "hits": ordered_intent_hits,
            "total": ordered_intent_total,
        },
    }

    return {
        "summary": summary,
        "cases": case_results,
    }
