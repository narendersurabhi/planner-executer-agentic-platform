from __future__ import annotations

from typing import Dict, Set

from .models import JobStatus, TaskStatus

JOB_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.queued: {JobStatus.planning, JobStatus.canceled},
    JobStatus.planning: {JobStatus.running, JobStatus.failed, JobStatus.canceled},
    JobStatus.running: {JobStatus.succeeded, JobStatus.failed, JobStatus.canceled},
    JobStatus.succeeded: set(),
    JobStatus.failed: {JobStatus.planning},
    JobStatus.canceled: {JobStatus.planning},
}

TASK_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.pending: {TaskStatus.ready, TaskStatus.canceled},
    TaskStatus.ready: {TaskStatus.running, TaskStatus.blocked, TaskStatus.canceled},
    TaskStatus.running: {
        TaskStatus.completed,
        TaskStatus.failed,
        TaskStatus.canceled,
        TaskStatus.rework_requested,
    },
    TaskStatus.completed: {TaskStatus.accepted, TaskStatus.rework_requested},
    TaskStatus.accepted: set(),
    TaskStatus.rework_requested: {TaskStatus.ready, TaskStatus.failed},
    TaskStatus.blocked: {TaskStatus.ready, TaskStatus.failed, TaskStatus.canceled},
    TaskStatus.failed: set(),
    TaskStatus.canceled: set(),
}


def validate_job_transition(current: JobStatus, new: JobStatus) -> bool:
    return new in JOB_TRANSITIONS.get(current, set())


def validate_task_transition(current: TaskStatus, new: TaskStatus) -> bool:
    return new in TASK_TRANSITIONS.get(current, set())
