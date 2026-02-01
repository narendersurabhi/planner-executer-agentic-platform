from datetime import datetime

from libs.core import models, orchestrator


def _task(task_id: str, deps: list[str], status: models.TaskStatus) -> models.Task:
    return models.Task(
        id=task_id,
        job_id="job",
        plan_id="plan",
        name=task_id,
        description="",
        instruction="",
        acceptance_criteria=[],
        expected_output_schema_ref="TaskResult",
        status=status,
        deps=deps,
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


def test_ready_tasks_dependency_release():
    tasks = [
        _task("a", [], models.TaskStatus.completed),
        _task("b", ["a"], models.TaskStatus.pending),
        _task("c", ["b"], models.TaskStatus.pending),
    ]
    ready = orchestrator.ready_tasks(tasks)
    assert ready == ["b"]


def test_idempotent_event():
    seen: dict[str, str] = {}
    assert orchestrator.apply_idempotent_event(seen, "evt")
    assert not orchestrator.apply_idempotent_event(seen, "evt")
