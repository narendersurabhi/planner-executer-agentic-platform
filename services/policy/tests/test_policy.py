from datetime import datetime

from libs.core import models, policy_engine


def test_policy_allow_in_dev(tmp_path):
    config = tmp_path / "policy.yaml"
    config.write_text("policy: {}")
    engine = policy_engine.PolicyEngine("dev", str(config))
    task = models.Task(
        id="task",
        job_id="job",
        plan_id="plan",
        name="task",
        description="",
        instruction="",
        acceptance_criteria=[],
        expected_output_schema_ref="TaskResult",
        status=models.TaskStatus.pending,
        deps=[],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=2,
        assigned_to=None,
        tool_requests=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        critic_required=True,
    )
    decision = engine.evaluate_task(task, tool_http_fetch_enabled=True)
    assert decision.decision == models.PolicyDecisionType.allow


def test_policy_deny_in_prod(tmp_path):
    config = tmp_path / "policy.yaml"
    config.write_text("policy: { allowlist: [math_eval] }")
    engine = policy_engine.PolicyEngine("prod", str(config))
    task = models.Task(
        id="task",
        job_id="job",
        plan_id="plan",
        name="task",
        description="",
        instruction="",
        acceptance_criteria=[],
        expected_output_schema_ref="TaskResult",
        status=models.TaskStatus.pending,
        deps=[],
        attempts=0,
        max_attempts=3,
        rework_count=0,
        max_reworks=2,
        assigned_to=None,
        tool_requests=["http_fetch"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        critic_required=True,
    )
    decision = engine.evaluate_task(task, tool_http_fetch_enabled=False)
    assert decision.decision == models.PolicyDecisionType.deny
