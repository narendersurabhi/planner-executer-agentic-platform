from __future__ import annotations

from typing import Dict, List

from .models import Task, TaskStatus


def ready_tasks(tasks: List[Task]) -> List[str]:
    completed = {
        task.id for task in tasks if task.status in {TaskStatus.completed, TaskStatus.accepted}
    }
    ready = []
    for task in tasks:
        if task.status != TaskStatus.pending:
            continue
        if all(dep in completed for dep in task.deps):
            ready.append(task.id)
    return ready


def apply_idempotent_event(events_seen: Dict[str, str], event_id: str) -> bool:
    if event_id in events_seen:
        return False
    events_seen[event_id] = event_id
    return True
