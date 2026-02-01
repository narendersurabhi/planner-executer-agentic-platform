from datetime import datetime

from libs.core import models
from services.planner.app.main import rule_based_plan


def test_rule_based_plan_schema():
    job = models.Job(
        id="job",
        goal="demo",
        context_json={},
        status=models.JobStatus.queued,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        priority=1,
        metadata={},
    )
    plan = rule_based_plan(job, [])
    assert plan.planner_version
    assert len(plan.tasks) == 3
