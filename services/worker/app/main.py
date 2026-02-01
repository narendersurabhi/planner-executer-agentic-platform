from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime

import redis

from libs.core import events, logging as core_logging, models, tool_registry

core_logging.configure_logging("worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TOOL_HTTP_FETCH_ENABLED = os.getenv("TOOL_HTTP_FETCH_ENABLED", "false").lower() == "true"

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def execute_task(task_payload: dict) -> models.TaskResult:
    task_id = task_payload.get("task_id")
    tool_requests = task_payload.get("tool_requests", [])
    registry = tool_registry.default_registry(TOOL_HTTP_FETCH_ENABLED)
    tool_calls = []
    started_at = datetime.utcnow()
    outputs = {}
    for tool_name in tool_requests:
        call = registry.execute(
            tool_name,
            payload={"text": task_payload.get("instruction", "")},
            idempotency_key=str(uuid.uuid4()),
            trace_id=task_payload.get("correlation_id", ""),
        )
        tool_calls.append(call)
        outputs[tool_name] = call.output_or_error
    finished_at = datetime.utcnow()
    return models.TaskResult(
        task_id=task_id,
        status=models.TaskStatus.completed,
        outputs=outputs,
        artifacts=[],
        tool_calls=tool_calls,
        started_at=started_at,
        finished_at=finished_at,
        error=None,
    )


def run() -> None:
    group = "workers"
    consumer = str(uuid.uuid4())
    try:
        redis_client.xgroup_create(events.TASK_STREAM, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass
    while True:
        messages = redis_client.xreadgroup(group, consumer, {events.TASK_STREAM: ">"}, count=1, block=1000)
        for _, entries in messages:
            for message_id, data in entries:
                payload = json.loads(data["data"])
                if payload.get("type") != "task.ready":
                    redis_client.xack(events.TASK_STREAM, group, message_id)
                    continue
                task_payload = payload.get("payload", {})
                result = execute_task(task_payload)
                event = models.EventEnvelope(
                    type="task.completed",
                    version="1",
                    occurred_at=datetime.utcnow(),
                    correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                    job_id=payload.get("job_id"),
                    task_id=task_payload.get("task_id"),
                    payload=result.model_dump(),
                )
                redis_client.xadd(events.TASK_STREAM, {"data": event.model_dump_json()})
                redis_client.xack(events.TASK_STREAM, group, message_id)
                time.sleep(0.1)


if __name__ == "__main__":
    run()
