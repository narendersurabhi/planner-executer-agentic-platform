from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import redis

from libs.core import events, logging as core_logging, models

core_logging.configure_logging("critic")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CRITIC_ENABLED = os.getenv("CRITIC_ENABLED", "false").lower() == "true"
CRITIC_MAX_REWORKS = int(os.getenv("CRITIC_MAX_REWORKS", "2"))

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def evaluate_task(result: models.TaskResult) -> models.CriticResult:
    reasons = []
    feedback = None
    decision = "accepted"
    if result.outputs == {}:
        decision = "rework"
        reasons.append("No outputs provided")
        feedback = "Provide structured outputs"
    return models.CriticResult(
        task_id=result.task_id,
        decision=decision,
        reasons=reasons,
        feedback=feedback,
        checked_at=datetime.utcnow(),
    )


def run() -> None:
    if not CRITIC_ENABLED:
        return
    group = "critic"
    consumer = str(uuid.uuid4())
    try:
        redis_client.xgroup_create(events.TASK_STREAM, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass
    while True:
        messages = redis_client.xreadgroup(
            group, consumer, {events.TASK_STREAM: ">"}, count=1, block=1000
        )
        for _, entries in messages:
            for message_id, data in entries:
                payload = json.loads(data["data"])
                if payload.get("type") != "task.completed":
                    redis_client.xack(events.TASK_STREAM, group, message_id)
                    continue
                result = models.TaskResult(**payload.get("payload", {}))
                critic_result = evaluate_task(result)
                event_type = (
                    "task.accepted"
                    if critic_result.decision == "accepted"
                    else "task.rework_requested"
                )
                event = models.EventEnvelope(
                    type=event_type,
                    version="1",
                    occurred_at=datetime.utcnow(),
                    correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                    job_id=payload.get("job_id"),
                    task_id=result.task_id,
                    payload=critic_result.model_dump(),
                )
                redis_client.xadd(events.CRITIC_STREAM, {"data": event.model_dump_json()})
                redis_client.xack(events.TASK_STREAM, group, message_id)


if __name__ == "__main__":
    run()
