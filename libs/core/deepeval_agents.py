from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import chat_boundary_eval, chat_clarification_eval, feedback_eval, intent_eval
from .deepeval_support import DeepEvalCase, DeepEvalMetricSpec, dumps_json


@dataclass(frozen=True)
class DeepEvalBundle:
    surface: str
    dataset_id: str
    cases: tuple[DeepEvalCase, ...]
    deterministic_summary: Mapping[str, Any]
    metric_specs: tuple[DeepEvalMetricSpec, ...]


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _top_capability_ids(evidence: Any) -> list[str]:
    if evidence is None:
        return []
    capability_ids: list[str] = []
    for capability in getattr(evidence, "top_capabilities", []) or []:
        capability_id = _normalize_string(getattr(capability, "capability_id", None))
        if capability_id and capability_id not in capability_ids:
            capability_ids.append(capability_id)
    return capability_ids


def _context_lines(lines: Sequence[str | None]) -> tuple[str, ...]:
    return tuple(line for line in (_normalize_string(line) for line in lines) if line)


def chat_metric_specs() -> tuple[DeepEvalMetricSpec, ...]:
    return (
        DeepEvalMetricSpec(
            name="chat_decision_quality",
            criteria=(
                "Judge whether ACTUAL_OUTPUT makes the correct front-door orchestration decision "
                "for the INPUT and CONTEXT. Reward correct choices among respond, ask for "
                "clarification, continue an in-flight clarification, submit a job, or run a "
                "workflow. Penalize unnecessary clarification, missed execution escalation, and "
                "decision drift away from the expected action."
            ),
            threshold=0.5,
        ),
        DeepEvalMetricSpec(
            name="chat_clarification_behavior",
            criteria=(
                "Judge whether ACTUAL_OUTPUT handles clarification state correctly for the INPUT "
                "and CONTEXT. Reward resolving the active field, advancing the queue only when the "
                "active field is resolved, preserving the current execution frame, and restarting "
                "only when the user actually changes intent. Penalize clarification loops."
            ),
            threshold=0.5,
        ),
    )


def planner_metric_specs() -> tuple[DeepEvalMetricSpec, ...]:
    return (
        DeepEvalMetricSpec(
            name="planner_capability_fit",
            criteria=(
                "Judge whether ACTUAL_OUTPUT is relevant to the user goal in INPUT and whether it "
                "uses capabilities that fit the expected planner boundary behavior in EXPECTED_OUTPUT "
                "and CONTEXT. Penalize wrong capability families, irrelevant capability choices, and "
                "instruction drift."
            ),
            threshold=0.5,
        ),
        DeepEvalMetricSpec(
            name="planner_missing_input_handling",
            criteria=(
                "Judge whether ACTUAL_OUTPUT correctly handles missing required inputs and "
                "clarification state relative to EXPECTED_OUTPUT and CONTEXT. Reward accurate "
                "missing-input detection, correct clarification mode, and plan readiness. Penalize "
                "spurious missing inputs, omitted blocking inputs, and contract-invalid behavior."
            ),
            threshold=0.5,
        ),
    )


def build_chat_gold_bundle(
    *,
    boundary_gold: Path,
    clarification_gold: Path,
) -> DeepEvalBundle:
    boundary_cases = chat_boundary_eval.load_chat_boundary_eval_cases(boundary_gold)
    boundary_report = chat_boundary_eval.evaluate_chat_boundary_cases(
        boundary_cases,
        predict=chat_boundary_eval.predict_chat_boundary_from_evidence,
    )
    boundary_case_by_id = {case.case_id: case for case in boundary_cases}
    deep_eval_cases: list[DeepEvalCase] = []

    for raw_result in boundary_report.get("cases", []):
        if not isinstance(raw_result, Mapping):
            continue
        case_id = _normalize_string(raw_result.get("case_id"))
        case = boundary_case_by_id.get(case_id)
        if case is None:
            continue
        predicted_decision = _normalize_string(raw_result.get("predicted_decision"))
        active_family = _normalize_string(raw_result.get("active_family")) or None
        evidence = case.evidence
        deep_eval_cases.append(
            DeepEvalCase(
                case_id=case.case_id,
                surface="chat_boundary",
                source_dataset_id=boundary_gold.stem,
                input_text=(
                    f"User turn: {case.content}\n"
                    f"Candidate goal: {case.candidate_goal or case.content}"
                ),
                actual_output=dumps_json(
                    {
                        "decision": predicted_decision,
                        "pending_clarification": case.pending_clarification,
                        "workflow_target_available": case.workflow_target_available,
                        "active_family": active_family,
                        "family_alignment": raw_result.get("family_alignment"),
                        "top_capability_ids": _top_capability_ids(evidence),
                    }
                ),
                expected_output=dumps_json(
                    {
                        "decision": case.expected_decision.value,
                        "active_family": active_family,
                    }
                ),
                context=_context_lines(
                    (
                        "Case type: chat boundary gold replay.",
                        f"Pending clarification: {case.pending_clarification}",
                        f"Workflow target available: {case.workflow_target_available}",
                        f"Top capability ids: {', '.join(_top_capability_ids(evidence)) or 'none'}",
                        f"Family alignment: {raw_result.get('family_alignment') or 'unknown'}",
                        case.notes,
                    )
                ),
                local_scores={
                    "decision_match": bool(raw_result.get("match")),
                    "family_scope_match": raw_result.get("family_alignment") != "drift"
                    if active_family
                    else None,
                },
                local_pass=bool(raw_result.get("match")),
                metadata={
                    "expected_decision": case.expected_decision.value,
                    "predicted_decision": predicted_decision,
                    "candidate_goal": case.candidate_goal,
                    "notes": case.notes,
                },
            )
        )

    clarification_cases = chat_clarification_eval.load_chat_clarification_eval_cases(
        clarification_gold
    )
    clarification_report = chat_clarification_eval.evaluate_chat_clarification_cases(
        clarification_cases
    )
    clarification_case_by_id = {case.case_id: case for case in clarification_cases}
    for raw_result in clarification_report.get("cases", []):
        if not isinstance(raw_result, Mapping):
            continue
        case_id = _normalize_string(raw_result.get("case_id"))
        case = clarification_case_by_id.get(case_id)
        if case is None:
            continue
        deep_eval_cases.append(
            DeepEvalCase(
                case_id=case.case_id,
                surface="chat_clarification",
                source_dataset_id=clarification_gold.stem,
                input_text=f"User clarification turn: {case.content}",
                actual_output=dumps_json(
                    {
                        "restarted": bool(raw_result.get("predicted_restarted")),
                        "resolved_active_field": bool(
                            raw_result.get("predicted_resolved_active_field")
                        ),
                        "queue_advanced": bool(raw_result.get("predicted_queue_advanced")),
                        "resolved_fields": list(raw_result.get("resolved_fields") or []),
                        "active_field_before": raw_result.get("active_field_before"),
                        "active_field_after": raw_result.get("active_field_after"),
                    }
                ),
                expected_output=dumps_json(
                    {
                        "restarted": case.expected_restarted,
                        "resolved_active_field": case.expected_resolved_active_field,
                        "queue_advanced": case.expected_queue_advanced,
                        "active_field_before": case.active_field_before,
                        "active_field_after": case.active_field_after,
                    }
                ),
                context=_context_lines(
                    (
                        "Case type: chat clarification gold replay.",
                        f"Resolved fields: {', '.join(case.resolved_fields) or 'none'}",
                        case.notes,
                    )
                ),
                local_scores={
                    "restart_decision_match": bool(raw_result.get("restart_match")),
                    "resolved_active_field_match": bool(
                        raw_result.get("resolved_active_field_match")
                    ),
                    "queue_advance_match": bool(raw_result.get("queue_advanced_match")),
                    "overall_match": bool(raw_result.get("overall_match")),
                },
                local_pass=bool(raw_result.get("overall_match")),
                metadata={
                    "active_field_before": case.active_field_before,
                    "active_field_after": case.active_field_after,
                    "resolved_fields": list(case.resolved_fields),
                    "notes": case.notes,
                },
            )
        )

    return DeepEvalBundle(
        surface="chat",
        dataset_id=f"{boundary_gold.stem}+{clarification_gold.stem}",
        cases=tuple(deep_eval_cases),
        deterministic_summary={
            "boundary": boundary_report.get("summary", {}),
            "clarification": clarification_report.get("summary", {}),
        },
        metric_specs=chat_metric_specs(),
    )


def build_planner_gold_bundle(
    *,
    normalization_gold: Path,
    normalize_goal_intent: Callable[[str, Mapping[str, Any] | None], Mapping[str, Any]],
) -> DeepEvalBundle:
    def _avg(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    cases = intent_eval.load_intent_eval_cases(normalization_gold)
    deep_eval_cases: list[DeepEvalCase] = []
    aggregate_profile_matches: list[float] = []
    aggregate_capability_alignment: list[float] = []
    aggregate_missing_match: list[float] = []
    aggregate_requires_clarification: list[float] = []
    aggregate_clarification_mode: list[float] = []
    aggregate_disagreement_reason: list[float] = []
    missing_tp = 0
    missing_fp = 0
    missing_fn = 0

    for case in cases:
        envelope = normalize_goal_intent(case.goal, case.intent_context)
        normalization = intent_eval.evaluate_intent_normalization_case(case, envelope)
        profile_match = normalization["profile_intent"]["match"]
        capability_alignment = bool(normalization["capability_alignment"]["matched"])
        missing_precision = float(normalization["missing_inputs"]["precision"] or 0.0)
        missing_recall = float(normalization["missing_inputs"]["recall"] or 0.0)
        missing_inputs_match = set(normalization["missing_inputs"]["expected"]) == set(
            normalization["missing_inputs"]["actual"]
        )
        clarification_match = normalization["clarification"]["requires_clarification_match"]
        mode_match = normalization["clarification"]["mode_match"]
        disagreement_reason_match = normalization["disagreement"]["reason_match"]

        if profile_match is not None:
            aggregate_profile_matches.append(1.0 if profile_match else 0.0)
        aggregate_capability_alignment.append(1.0 if capability_alignment else 0.0)
        aggregate_missing_match.append(1.0 if missing_inputs_match else 0.0)
        missing_tp += int(normalization["missing_inputs"]["tp"])
        missing_fp += int(normalization["missing_inputs"]["fp"])
        missing_fn += int(normalization["missing_inputs"]["fn"])
        if clarification_match is not None:
            aggregate_requires_clarification.append(1.0 if clarification_match else 0.0)
        if mode_match is not None:
            aggregate_clarification_mode.append(1.0 if mode_match else 0.0)
        if disagreement_reason_match is not None:
            aggregate_disagreement_reason.append(1.0 if disagreement_reason_match else 0.0)

        primary_score_components = [
            1.0 if capability_alignment else 0.0,
            1.0 if missing_inputs_match else 0.0,
        ]
        if profile_match is not None:
            primary_score_components.append(1.0 if profile_match else 0.0)
        if clarification_match is not None:
            primary_score_components.append(1.0 if clarification_match else 0.0)
        if mode_match is not None:
            primary_score_components.append(1.0 if mode_match else 0.0)
        if disagreement_reason_match is not None:
            primary_score_components.append(1.0 if disagreement_reason_match else 0.0)
        local_pass = (
            capability_alignment
            and missing_inputs_match
            and (profile_match is not False)
            and (clarification_match is not False)
            and (mode_match is not False)
            and (disagreement_reason_match is not False)
        )
        deep_eval_cases.append(
            DeepEvalCase(
                case_id=case.case_id,
                surface="planner_boundary",
                source_dataset_id=normalization_gold.stem,
                input_text=f"Planner goal: {case.goal}",
                actual_output=dumps_json(
                    {
                        "profile_intent": normalization["profile_intent"]["actual"],
                        "predicted_capabilities": normalization["capability_alignment"][
                            "predicted_capabilities"
                        ],
                        "top_capability": normalization["capability_alignment"][
                            "top_capability"
                        ],
                        "missing_inputs": normalization["missing_inputs"]["actual"],
                        "requires_clarification": normalization["clarification"][
                            "actual_requires_clarification"
                        ],
                        "clarification_mode": normalization["clarification"]["actual_mode"],
                        "disagreement_reason": normalization["disagreement"]["actual_reason"],
                    }
                ),
                expected_output=dumps_json(
                    {
                        "profile_intent": case.expected_profile_intent,
                        "expected_capabilities": list(case.expected_capabilities),
                        "missing_inputs": list(case.expected_missing_inputs),
                        "requires_clarification": case.expected_requires_clarification,
                        "clarification_mode": case.expected_clarification_mode,
                        "disagreement_reason": case.expected_disagreement_reason,
                    }
                ),
                context=_context_lines(
                    (
                        "Case type: planner normalization gold replay.",
                        f"Expected intents: {', '.join(case.expected_intents) or 'none'}",
                        (
                            "Expected capabilities by segment: "
                            + dumps_json(list(case.expected_capabilities_by_segment))
                        ),
                        (
                            "Intent context: " + dumps_json(dict(case.intent_context or {}))
                            if case.intent_context
                            else None
                        ),
                    )
                ),
                local_scores={
                    "profile_intent_match": profile_match,
                    "capability_alignment": capability_alignment,
                    "missing_inputs_match": missing_inputs_match,
                    "missing_inputs_precision": missing_precision,
                    "missing_inputs_recall": missing_recall,
                    "requires_clarification_match": clarification_match,
                    "clarification_mode_match": mode_match,
                    "disagreement_reason_match": disagreement_reason_match,
                },
                local_pass=local_pass,
                metadata={
                    "expected_capabilities": list(case.expected_capabilities),
                    "expected_missing_inputs": list(case.expected_missing_inputs),
                    "expected_profile_intent": case.expected_profile_intent,
                    "primary_local_score": round(_avg(primary_score_components), 4),
                },
            )
        )

    return DeepEvalBundle(
        surface="planner",
        dataset_id=normalization_gold.stem,
        cases=tuple(deep_eval_cases),
        deterministic_summary={
            "profile_intent_accuracy": round(_avg(aggregate_profile_matches), 4),
            "capability_alignment_rate": round(_avg(aggregate_capability_alignment), 4),
            "missing_inputs_match_rate": round(_avg(aggregate_missing_match), 4),
            "missing_inputs_precision": round(
                missing_tp / (missing_tp + missing_fp) if (missing_tp + missing_fp) else 0.0,
                4,
            ),
            "missing_inputs_recall": round(
                missing_tp / (missing_tp + missing_fn) if (missing_tp + missing_fn) else 0.0,
                4,
            ),
            "requires_clarification_accuracy": round(
                _avg(aggregate_requires_clarification), 4
            ),
            "clarification_mode_accuracy": round(_avg(aggregate_clarification_mode), 4),
            "disagreement_reason_accuracy": round(_avg(aggregate_disagreement_reason), 4),
        },
        metric_specs=planner_metric_specs(),
    )


def build_chat_feedback_bundle(
    *,
    examples: Sequence[Mapping[str, Any]],
    dataset_id: str = "feedback_chat_message",
) -> DeepEvalBundle:
    rows = feedback_eval.build_feedback_eval_rows(examples)
    deep_eval_cases: list[DeepEvalCase] = []
    for row in rows:
        if _normalize_string(row.get("target_type")) != "chat_message":
            continue
        snapshot = dict(row.get("snapshot") or {})
        user_turn = _normalize_string(
            snapshot.get("user_message_content")
            or snapshot.get("content")
            or snapshot.get("candidate_goal")
        )
        if not user_turn:
            continue
        sentiment = _normalize_string(row.get("sentiment")).lower() or "none"
        sentiment_score = 1.0 if sentiment == "positive" else 0.5 if sentiment == "partial" else 0.0
        actual_output = {
            "boundary_decision": row.get("boundary_decision"),
            "clarification_active_field_before": row.get(
                "clarification_mapping_active_field_before"
            ),
            "clarification_active_field_after": row.get(
                "clarification_mapping_active_field_after"
            ),
            "clarification_restarted": row.get("clarification_mapping_restarted"),
            "clarification_queue_advanced": row.get("clarification_mapping_queue_advanced"),
            "clarification_resolved_active_field": row.get(
                "clarification_mapping_resolved_active_field"
            ),
        }
        expected_output = {
            "feedback_sentiment": sentiment,
            "reason_codes": list(row.get("reason_codes") or []),
            "comment": row.get("comment"),
            "target_behavior": (
                "preserve the observed boundary behavior"
                if sentiment == "positive"
                else "avoid the failure reasons listed by operator feedback"
            ),
        }
        deep_eval_cases.append(
            DeepEvalCase(
                case_id=_normalize_string(row.get("feedback_id"))
                or _normalize_string(row.get("target_id"))
                or f"feedback_chat_{len(deep_eval_cases) + 1}",
                surface="chat_feedback_replay",
                source_dataset_id=dataset_id,
                input_text=f"User turn: {user_turn}",
                actual_output=dumps_json(actual_output),
                expected_output=dumps_json(expected_output),
                context=_context_lines(
                    (
                        "Case type: staging chat feedback replay.",
                        f"Clarification question: {row.get('clarification_current_question')}",
                        f"Boundary top family: {row.get('boundary_evidence', {}).get('top_families') if isinstance(row.get('boundary_evidence'), Mapping) else ''}",
                    )
                ),
                local_scores={"feedback_sentiment_score": sentiment_score},
                local_pass=sentiment == "positive",
                metadata={
                    "reason_codes": list(row.get("reason_codes") or []),
                    "comment": row.get("comment"),
                    "sentiment": sentiment,
                },
            )
        )

    return DeepEvalBundle(
        surface="chat",
        dataset_id=dataset_id,
        cases=tuple(deep_eval_cases),
        deterministic_summary={"feedback_row_count": len(deep_eval_cases)},
        metric_specs=chat_metric_specs(),
    )


def build_planner_feedback_bundle(
    *,
    examples: Sequence[Mapping[str, Any]],
    dataset_id: str = "feedback_plan",
) -> DeepEvalBundle:
    deep_eval_cases: list[DeepEvalCase] = []
    for example in examples:
        feedback = dict(example.get("feedback") or {})
        if _normalize_string(feedback.get("target_type")) != "plan":
            continue
        snapshot = dict(example.get("snapshot") or feedback.get("snapshot") or {})
        sentiment = _normalize_string(feedback.get("sentiment")).lower() or "none"
        sentiment_score = 1.0 if sentiment == "positive" else 0.5 if sentiment == "partial" else 0.0
        plan_id = _normalize_string(feedback.get("target_id")) or _normalize_string(
            snapshot.get("plan_id")
        )
        tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), list) else []
        actual_output = {
            "planner_version": snapshot.get("planner_version"),
            "tasks_summary": snapshot.get("tasks_summary"),
            "task_count": snapshot.get("task_count"),
            "tasks": tasks,
            "policy_decision": snapshot.get("policy_decision"),
        }
        expected_output = {
            "feedback_sentiment": sentiment,
            "reason_codes": list(feedback.get("reason_codes") or []),
            "comment": feedback.get("comment"),
            "target_behavior": (
                "preserve the observed plan behavior"
                if sentiment == "positive"
                else "avoid the plan defects called out by operator feedback"
            ),
        }
        deep_eval_cases.append(
            DeepEvalCase(
                case_id=plan_id or f"feedback_plan_{len(deep_eval_cases) + 1}",
                surface="planner_feedback_replay",
                source_dataset_id=dataset_id,
                input_text=(
                    f"Plan replay for plan_id={plan_id or 'unknown'} "
                    f"with tasks_summary={_normalize_string(snapshot.get('tasks_summary')) or 'n/a'}"
                ),
                actual_output=dumps_json(actual_output),
                expected_output=dumps_json(expected_output),
                context=_context_lines(
                    (
                        "Case type: staging planner feedback replay.",
                        f"Planner version: {snapshot.get('planner_version')}",
                        f"Task count: {snapshot.get('task_count')}",
                    )
                ),
                local_scores={"feedback_sentiment_score": sentiment_score},
                local_pass=sentiment == "positive",
                metadata={
                    "reason_codes": list(feedback.get("reason_codes") or []),
                    "comment": feedback.get("comment"),
                    "sentiment": sentiment,
                },
            )
        )

    return DeepEvalBundle(
        surface="planner",
        dataset_id=dataset_id,
        cases=tuple(deep_eval_cases),
        deterministic_summary={"feedback_row_count": len(deep_eval_cases)},
        metric_specs=planner_metric_specs(),
    )
