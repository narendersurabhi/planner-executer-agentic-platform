#!/usr/bin/env python3
"""
Set the first task for a given job to 'ready' and emit a `task.ready` event.

Usage:
  PYTHONPATH=. DATABASE_URL=<your-db-url> REDIS_URL=<your-redis-url> \
    python3 scripts/set_first_task_ready.py <job_id>
"""

import os
import sys
import uuid
from datetime import datetime

import redis

# Import the project's DB/session and models
from services.api.app.database import SessionLocal
from services.api.app.models import TaskRecord
from libs.core import models as core_models, events as core_events

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TASK_STREAM = core_events.TASK_STREAM  # "tasks.events"


def set_first_task_ready(job_id: str) -> None:
    now = datetime.utcnow()
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    with SessionLocal() as db:
        # Pick the first task by created_at for the job
        tasks = (
            db.query(TaskRecord)
            .filter(TaskRecord.job_id == job_id)
            .order_by(TaskRecord.created_at.asc())
            .all()
        )
        if not tasks:
            print(f"No tasks found for job {job_id}")
            return

        task = tasks[0]
        print(f"Found first task: id={task.id} name={task.name} status={task.status}")

        if task.status == core_models.TaskStatus.ready.value:
            print("Task is already ready. No change.")
            return

        # increment attempts (mirrors API behavior) and set ready
        task.attempts = (task.attempts or 0) + 1
        task.status = core_models.TaskStatus.ready.value
        task.updated_at = now
        db.commit()

        # Build payload similar to API's _task_payload_from_record
        payload = {
            "task_id": task.id,
            "id": task.id,
            "job_id": task.job_id,
            "plan_id": task.plan_id,
            "name": task.name,
            "description": task.description,
            "instruction": task.instruction,
            "acceptance_criteria": task.acceptance_criteria or [],
            "expected_output_schema_ref": task.expected_output_schema_ref,
            "status": task.status,
            "deps": task.deps or [],
            "attempts": task.attempts or 0,
            "max_attempts": task.max_attempts or 0,
            "rework_count": task.rework_count or 0,
            "max_reworks": task.max_reworks or 0,
            "assigned_to": task.assigned_to,
            "tool_requests": task.tool_requests or [],
            "critic_required": bool(task.critic_required),
            "created_at": (task.created_at.isoformat() if getattr(task, 'created_at', None) else None),
            "updated_at": (task.updated_at.isoformat() if getattr(task, 'updated_at', None) else None),
            "correlation_id": os.getenv("CORRELATION_ID", str(uuid.uuid4())),
        }

        # Create envelope and push to Redis stream
        envelope = core_models.EventEnvelope(
            type="task.ready",
            version="1",
            occurred_at=datetime.utcnow(),
            correlation_id=payload["correlation_id"],
            job_id=payload["job_id"],
            task_id=payload["task_id"],
            payload=payload,
        )
        # xadd uses the stream name defined in libs/core/events.py
        r.xadd(TASK_STREAM, {"data": envelope.model_dump_json()})
        print(f"Task {task.id} set to ready and task.ready event emitted on stream '{TASK_STREAM}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/set_first_task_ready.py <job_id>")
        sys.exit(2)
    set_first_task_ready(sys.argv[1])
