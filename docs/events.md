# Events

Redis Streams:
- jobs.events: job.created, job.canceled
- plans.events: plan.created, plan.approved, plan.rejected
- tasks.events: task.ready, task.started, task.heartbeat, task.completed, task.failed, task.blocked, task.canceled
- critic.events: task.accepted, task.rework_requested
- policy.events: policy.decision_made
