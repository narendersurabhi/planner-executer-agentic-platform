from __future__ import annotations

from libs.core import models
from services.api.app import replan_controller


def test_decide_task_failure_recovery_returns_retry_same_step_when_retry_budget_remains() -> None:
    decision = replan_controller.decide_task_failure_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        error_message="service unavailable: upstream temporary 503",
        classification={
            "category": "transient",
            "retryable": True,
        },
        attempt_number=1,
        max_attempts=3,
    )

    assert decision.strategy == models.ReplanStrategy.retry_same_step
    assert decision.strategy_reason == "retry_budget_remaining"
    assert decision.should_replan is False


def test_decide_task_failure_recovery_prefers_checkpoint_resume_when_retry_budget_is_exhausted() -> None:
    decision = replan_controller.decide_task_failure_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        error_message="service unavailable: upstream temporary 503",
        classification={
            "category": "transient",
            "retryable": True,
        },
        attempt_number=3,
        max_attempts=3,
        checkpoint_context={
            "resume_supported": True,
            "max_checkpoint_replays": 1,
            "checkpoint_lineage": {
                "checkpoint_id": "checkpoint-1",
                "checkpoint_key": "after-input",
                "replay_count": 0,
            },
        },
    )

    assert decision.strategy == models.ReplanStrategy.retry_same_step
    assert decision.strategy_reason == "checkpoint_resume_after_retry_budget_exhausted"
    assert decision.should_replan is False
    assert decision.context["checkpoint_lineage"]["checkpoint_id"] == "checkpoint-1"


def test_decide_task_failure_recovery_replans_when_checkpoint_budget_is_exhausted() -> None:
    decision = replan_controller.decide_task_failure_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        error_message="service unavailable: upstream temporary 503",
        classification={
            "category": "transient",
            "retryable": True,
        },
        attempt_number=3,
        max_attempts=3,
        retry_context={"failed_task_id": "task-1"},
        checkpoint_context={
            "resume_supported": True,
            "max_checkpoint_replays": 1,
            "checkpoint_lineage": {
                "checkpoint_id": "checkpoint-1",
                "checkpoint_key": "after-input",
                "replay_count": 1,
            },
        },
    )

    assert decision.strategy == models.ReplanStrategy.patch_suffix
    assert decision.strategy_reason == "retry_budget_exhausted_with_reusable_prefix"
    assert decision.should_replan is True
    assert decision.context["checkpoint_lineage"]["checkpoint_id"] == "checkpoint-1"


def test_decide_task_failure_recovery_returns_pause_for_human_for_missing_input() -> None:
    decision = replan_controller.decide_task_failure_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        error_message="missing_input:path",
        classification={
            "category": "contract",
            "retryable": False,
        },
        attempt_number=1,
        max_attempts=3,
    )

    assert decision.strategy == models.ReplanStrategy.pause_for_human
    assert decision.strategy_reason == "missing_or_ambiguous_user_input"
    assert decision.should_replan is False


def test_decide_manual_replan_returns_full_replan() -> None:
    decision = replan_controller.decide_manual_replan()

    assert decision.strategy == models.ReplanStrategy.full_replan
    assert decision.strategy_reason == "manual_replan_requested"
    assert decision.should_replan is True
    assert decision.replan_reason == "manual"


def test_decide_task_evaluator_recovery_returns_rework_for_low_confidence() -> None:
    decision = replan_controller.decide_task_evaluator_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        requested_rework_count=1,
        max_reworks=2,
        evaluator_signal={"confidence": 0.42},
        requested_rework=False,
        min_confidence=0.6,
        replan_confidence_floor=0.35,
    )

    assert decision.strategy == models.ReplanStrategy.rework_step
    assert decision.strategy_reason == "low_confidence_needs_rework"
    assert decision.should_replan is False


def test_decide_task_evaluator_recovery_returns_patch_suffix_for_schema_invalid_replan_policy() -> None:
    decision = replan_controller.decide_task_evaluator_recovery(
        planning_mode=models.PlanningMode.adaptive,
        has_pending_replan=False,
        replans_used=0,
        max_replans=2,
        requested_rework_count=1,
        max_reworks=2,
        evaluator_signal={"schema_valid": False},
        requested_rework=True,
        min_confidence=0.6,
        replan_confidence_floor=0.35,
        schema_invalid_strategy=models.ReplanStrategy.patch_suffix,
    )

    assert decision.strategy == models.ReplanStrategy.patch_suffix
    assert decision.strategy_reason == "schema_invalid_policy_requires_suffix_replan"
    assert decision.should_replan is True
    assert decision.replan_reason == "evaluator_schema_invalid_auto_repair"
