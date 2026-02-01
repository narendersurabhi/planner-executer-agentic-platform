from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import redis

from libs.core import events, logging as core_logging, models, policy_engine

core_logging.configure_logging("policy")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
POLICY_GATE_ENABLED = os.getenv("POLICY_GATE_ENABLED", "false").lower() == "true"
POLICY_MODE = os.getenv("POLICY_MODE", "dev")
TOOL_HTTP_FETCH_ENABLED = os.getenv("TOOL_HTTP_FETCH_ENABLED", "false").lower() == "true"
POLICY_CONFIG_PATH = os.getenv("POLICY_CONFIG_PATH", "config/policy.yaml")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
engine = policy_engine.PolicyEngine(POLICY_MODE, POLICY_CONFIG_PATH)


def run() -> None:
    if not POLICY_GATE_ENABLED:
        return
    group = "policy"
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
                task = models.Task(**payload.get("payload", {}))
                decision = engine.evaluate_task(task, TOOL_HTTP_FETCH_ENABLED)
                event = models.EventEnvelope(
                    type="policy.decision_made",
                    version="1",
                    occurred_at=datetime.utcnow(),
                    correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
                    job_id=task.job_id,
                    task_id=task.id,
                    payload=decision.model_dump(),
                )
                redis_client.xadd(events.POLICY_STREAM, {"data": event.model_dump_json()})
                redis_client.xack(events.TASK_STREAM, group, message_id)


if __name__ == "__main__":
    run()
