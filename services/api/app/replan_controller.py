from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from libs.core import models


@dataclass(frozen=True)
class RecoveryDecision:
    strategy: models.ReplanStrategy
    strategy_reason: str
    should_replan: bool = False
    replan_reason: str | None = None
    require_adaptive: bool = False
    context: dict[str, Any] = field(default_factory=dict)


def decide_manual_replan() -> RecoveryDecision:
    return RecoveryDecision(
        strategy=models.ReplanStrategy.full_replan,
        strategy_reason="manual_replan_requested",
        should_replan=True,
        replan_reason="manual",
        require_adaptive=False,
    )


def decide_task_failure_recovery(
    *,
    planning_mode: models.PlanningMode,
    has_pending_replan: bool,
    replans_used: int,
    max_replans: int,
    error_message: str | None,
    classification: Mapping[str, Any] | None,
    attempt_number: int,
    max_attempts: int,
    intent_mismatch_context: Mapping[str, Any] | None = None,
    retry_context: Mapping[str, Any] | None = None,
    checkpoint_context: Mapping[str, Any] | None = None,
) -> RecoveryDecision:
    normalized_error = str(error_message or "").strip().lower()
    details = classification if isinstance(classification, Mapping) else {}
    category = str(details.get("category") or "").strip().lower()
    retryable = bool(details.get("retryable"))
    adaptive_enabled = planning_mode == models.PlanningMode.adaptive
    exhausted = max_attempts > 0 and attempt_number >= max_attempts
    checkpoint = dict(checkpoint_context) if isinstance(checkpoint_context, Mapping) else {}
    checkpoint_lineage = (
        dict(checkpoint.get("checkpoint_lineage"))
        if isinstance(checkpoint.get("checkpoint_lineage"), Mapping)
        else {}
    )
    checkpoint_available = bool(checkpoint_lineage.get("checkpoint_id")) or bool(
        checkpoint_lineage.get("checkpoint_key")
    )
    resume_supported = bool(checkpoint.get("resume_supported", checkpoint_available))
    try:
        max_checkpoint_replays = max(0, int(checkpoint.get("max_checkpoint_replays", 0) or 0))
    except (TypeError, ValueError):
        max_checkpoint_replays = 0
    try:
        replay_count = max(0, int(checkpoint_lineage.get("replay_count", 0) or 0))
    except (TypeError, ValueError):
        replay_count = 0
    checkpoint_budget_available = (
        resume_supported
        and checkpoint_available
        and max_checkpoint_replays > 0
        and replay_count < max_checkpoint_replays
    )

    if has_pending_replan:
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason="pending_replan_already_requested",
        )

    if intent_mismatch_context:
        if not adaptive_enabled:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.no_replan,
                strategy_reason="adaptive_mode_required_for_intent_mismatch_repair",
            )
        if replans_used >= max_replans:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.no_replan,
                strategy_reason="max_replans_exhausted",
            )
        return RecoveryDecision(
            strategy=models.ReplanStrategy.switch_capability,
            strategy_reason="contract_or_intent_mismatch",
            should_replan=True,
            replan_reason="intent_mismatch_auto_repair",
            require_adaptive=True,
            context=dict(intent_mismatch_context),
        )

    if (
        "clarification" in normalized_error
        or "intent_clarification_required" in normalized_error
        or "missing input" in normalized_error
        or "missing_input" in normalized_error
        or "workflow_inputs_invalid" in normalized_error
    ):
        return RecoveryDecision(
            strategy=models.ReplanStrategy.pause_for_human,
            strategy_reason="missing_or_ambiguous_user_input",
        )

    if category == "policy":
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason="policy_blocked",
        )

    if retryable and checkpoint_budget_available:
        return RecoveryDecision(
            strategy=models.ReplanStrategy.retry_same_step,
            strategy_reason=(
                "checkpoint_resume_after_retry_budget_exhausted"
                if exhausted
                else "checkpoint_resume_available"
            ),
            context=checkpoint,
        )

    if retryable and max_attempts > 0 and attempt_number < max_attempts:
        return RecoveryDecision(
            strategy=models.ReplanStrategy.retry_same_step,
            strategy_reason="retry_budget_remaining",
        )

    if retryable and exhausted:
        if not adaptive_enabled:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.no_replan,
                strategy_reason="adaptive_mode_required_for_retry_exhausted_repair",
            )
        if replans_used >= max_replans:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.no_replan,
                strategy_reason="max_replans_exhausted",
            )
        return RecoveryDecision(
            strategy=models.ReplanStrategy.patch_suffix,
            strategy_reason="retry_budget_exhausted_with_reusable_prefix",
            should_replan=True,
            replan_reason="retry_exhausted_auto_repair",
            require_adaptive=True,
            context={
                **dict(retry_context or {}),
                **(
                    {"checkpoint_lineage": checkpoint_lineage}
                    if checkpoint_lineage
                    else {}
                ),
            },
        )

    return RecoveryDecision(
        strategy=models.ReplanStrategy.no_replan,
        strategy_reason="no_applicable_recovery_strategy",
    )


def decide_task_evaluator_recovery(
    *,
    planning_mode: models.PlanningMode,
    has_pending_replan: bool,
    replans_used: int,
    max_replans: int,
    requested_rework_count: int,
    max_reworks: int,
    evaluator_signal: Mapping[str, Any] | None = None,
    requested_rework: bool = False,
    min_confidence: float = 0.6,
    replan_confidence_floor: float = 0.35,
    schema_invalid_strategy: models.ReplanStrategy = models.ReplanStrategy.rework_step,
) -> RecoveryDecision:
    signal = evaluator_signal if isinstance(evaluator_signal, Mapping) else {}
    adaptive_enabled = planning_mode == models.PlanningMode.adaptive
    confidence = signal.get("confidence")
    if isinstance(confidence, bool):
        confidence = None
    if isinstance(confidence, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence)))
    else:
        confidence = None
    schema_valid = signal.get("schema_valid")
    output_valid = signal.get("output_valid")
    schema_invalid = (
        (isinstance(schema_valid, bool) and not schema_valid)
        or (isinstance(output_valid, bool) and not output_valid)
    )
    low_confidence = confidence is not None and confidence < max(0.0, min(1.0, float(min_confidence)))
    severe_low_confidence = confidence is not None and confidence < max(
        0.0,
        min(float(min_confidence), float(replan_confidence_floor)),
    )
    rework_budget_available = max_reworks <= 0 or requested_rework_count <= max_reworks
    replan_budget_available = adaptive_enabled and replans_used < max_replans

    if has_pending_replan:
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason="pending_replan_already_requested",
        )

    if schema_invalid:
        if schema_invalid_strategy == models.ReplanStrategy.patch_suffix and replan_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.patch_suffix,
                strategy_reason="schema_invalid_policy_requires_suffix_replan",
                should_replan=True,
                replan_reason="evaluator_schema_invalid_auto_repair",
                require_adaptive=True,
            )
        if rework_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.rework_step,
                strategy_reason="schema_invalid_needs_rework",
            )
        if replan_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.patch_suffix,
                strategy_reason="schema_invalid_rework_budget_exhausted",
                should_replan=True,
                replan_reason="evaluator_schema_invalid_auto_repair",
                require_adaptive=True,
            )
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason=(
                "max_replans_exhausted"
                if adaptive_enabled and replans_used >= max_replans
                else "rework_budget_exhausted"
            ),
        )

    if severe_low_confidence and replan_budget_available:
        return RecoveryDecision(
            strategy=models.ReplanStrategy.patch_suffix,
            strategy_reason="low_confidence_below_replan_floor",
            should_replan=True,
            replan_reason="evaluator_low_confidence_auto_repair",
            require_adaptive=True,
        )

    if low_confidence:
        if rework_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.rework_step,
                strategy_reason="low_confidence_needs_rework",
            )
        if replan_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.patch_suffix,
                strategy_reason="low_confidence_rework_budget_exhausted",
                should_replan=True,
                replan_reason="evaluator_low_confidence_auto_repair",
                require_adaptive=True,
            )
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason=(
                "max_replans_exhausted"
                if adaptive_enabled and replans_used >= max_replans
                else "rework_budget_exhausted"
            ),
        )

    if requested_rework:
        if rework_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.rework_step,
                strategy_reason="critic_requested_rework",
            )
        if replan_budget_available:
            return RecoveryDecision(
                strategy=models.ReplanStrategy.patch_suffix,
                strategy_reason="critic_rework_budget_exhausted",
                should_replan=True,
                replan_reason="evaluator_rework_exhausted_auto_repair",
                require_adaptive=True,
            )
        return RecoveryDecision(
            strategy=models.ReplanStrategy.no_replan,
            strategy_reason=(
                "max_replans_exhausted"
                if adaptive_enabled and replans_used >= max_replans
                else "rework_budget_exhausted"
            ),
        )

    return RecoveryDecision(
        strategy=models.ReplanStrategy.no_replan,
        strategy_reason="accepted_with_sufficient_confidence",
    )
