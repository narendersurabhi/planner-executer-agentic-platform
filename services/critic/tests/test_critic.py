from datetime import datetime

from libs.core import models
from services.critic.app.main import evaluate_task


def test_critic_accepts_output():
    result = models.TaskResult(
        task_id="task",
        status=models.TaskStatus.completed,
        outputs={"ok": True},
        artifacts=[],
        tool_calls=[],
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )
    critic = evaluate_task(result)
    assert critic.decision == "accepted"


def test_critic_requests_rework():
    result = models.TaskResult(
        task_id="task",
        status=models.TaskStatus.completed,
        outputs={},
        artifacts=[],
        tool_calls=[],
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )
    critic = evaluate_task(result)
    assert critic.decision == "rework"
