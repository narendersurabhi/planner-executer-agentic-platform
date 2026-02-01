JOB_STREAM = "jobs.events"
PLAN_STREAM = "plans.events"
TASK_STREAM = "tasks.events"
CRITIC_STREAM = "critic.events"
POLICY_STREAM = "policy.events"

JOB_EVENTS = ["job.created", "job.canceled"]
PLAN_EVENTS = ["plan.created", "plan.approved", "plan.rejected"]
TASK_EVENTS = [
    "task.ready",
    "task.started",
    "task.heartbeat",
    "task.completed",
    "task.failed",
    "task.blocked",
    "task.canceled",
]
CRITIC_EVENTS = ["task.accepted", "task.rework_requested"]
POLICY_EVENTS = ["policy.decision_made"]
