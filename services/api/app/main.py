from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Generator, List

import redis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from sqlalchemy.orm import Session

from libs.core import events, logging as core_logging, models, state_machine
from .database import SessionLocal
from .models import JobRecord, PlanRecord, TaskRecord

core_logging.configure_logging("api")

app = FastAPI(title="Agentic Planner Executor API")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

jobs_created_total = Counter("jobs_created_total", "Jobs created")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _emit_event(event_type: str, payload: dict[str, Any]) -> None:
    envelope = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.utcnow(),
        correlation_id=payload.get("correlation_id", str(uuid.uuid4())),
        job_id=payload.get("job_id"),
        task_id=payload.get("task_id"),
        payload=payload,
    )
    stream = _stream_for_event(event_type)
    redis_client.xadd(stream, {"data": envelope.model_dump_json()})


def _stream_for_event(event_type: str) -> str:
    if event_type.startswith("job"):
        return events.JOB_STREAM
    if event_type.startswith("plan"):
        return events.PLAN_STREAM
    if event_type.startswith("task"):
        return events.TASK_STREAM
    if event_type.startswith("policy"):
        return events.POLICY_STREAM
    return events.TASK_STREAM


@app.post("/jobs", response_model=models.Job)
def create_job(job: models.JobCreate, db: Session = Depends(get_db)) -> models.Job:
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    record = JobRecord(
        id=job_id,
        goal=job.goal,
        context_json=job.context_json,
        status=models.JobStatus.queued.value,
        created_at=now,
        updated_at=now,
        priority=job.priority,
        metadata={"idempotency_key": job.idempotency_key} if job.idempotency_key else {},
    )
    db.add(record)
    db.commit()
    jobs_created_total.inc()
    _emit_event(
        "job.created",
        {"job_id": job_id, "goal": job.goal, "correlation_id": str(uuid.uuid4())},
    )
    return models.Job(**record.__dict__)


@app.get("/jobs", response_model=List[models.Job])
def list_jobs(db: Session = Depends(get_db)) -> List[models.Job]:
    jobs = db.query(JobRecord).all()
    return [models.Job(**job.__dict__) for job in jobs]


@app.get("/jobs/{job_id}", response_model=models.Job)
def get_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return models.Job(**job.__dict__)


@app.get("/jobs/{job_id}/plan", response_model=models.Plan)
def get_plan(job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    plan = db.query(PlanRecord).filter(PlanRecord.job_id == job_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return models.Plan(**plan.__dict__)


@app.get("/jobs/{job_id}/tasks", response_model=List[models.Task])
def get_tasks(job_id: str, db: Session = Depends(get_db)) -> List[models.Task]:
    tasks = db.query(TaskRecord).filter(TaskRecord.job_id == job_id).all()
    return [models.Task(**task.__dict__) for task in tasks]


@app.get("/tasks/{task_id}", response_model=models.Task)
def get_task(task_id: str, db: Session = Depends(get_db)) -> models.Task:
    task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return models.Task(**task.__dict__)


@app.post("/jobs/{job_id}/cancel", response_model=models.Job)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> models.Job:
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not state_machine.validate_job_transition(models.JobStatus(job.status), models.JobStatus.canceled):
        raise HTTPException(status_code=400, detail="Invalid state transition")
    job.status = models.JobStatus.canceled.value
    job.updated_at = datetime.utcnow()
    db.commit()
    _emit_event("job.canceled", {"job_id": job_id, "correlation_id": str(uuid.uuid4())})
    return models.Job(**job.__dict__)


@app.get("/events/stream")
def stream_events(request: Request, once: bool = False):
    def event_generator():
        if once:
            yield "data: {}\n\n"
            return
        last_ids = {
            events.JOB_STREAM: "0-0",
            events.PLAN_STREAM: "0-0",
            events.TASK_STREAM: "0-0",
            events.CRITIC_STREAM: "0-0",
            events.POLICY_STREAM: "0-0",
        }
        while True:
            if request.client is None:
                break
            results = redis_client.xread(last_ids, block=1000, count=10)
            for stream_name, messages in results:
                for message_id, data in messages:
                    last_ids[stream_name] = message_id
                    payload = data.get("data")
                    yield f"data: {payload}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/plans", response_model=models.Plan)
def create_plan(plan: models.PlanCreate, job_id: str, db: Session = Depends(get_db)) -> models.Plan:
    plan_id = str(uuid.uuid4())
    now = datetime.utcnow()
    record = PlanRecord(
        id=plan_id,
        job_id=job_id,
        planner_version=plan.planner_version,
        created_at=now,
        tasks_summary=plan.tasks_summary,
        dag_edges=plan.dag_edges,
        policy_decision={},
    )
    db.add(record)
    for task in plan.tasks:
        task_id = str(uuid.uuid4())
        task_record = TaskRecord(
            id=task_id,
            job_id=job_id,
            plan_id=plan_id,
            name=task.name,
            description=task.description,
            instruction=task.instruction,
            acceptance_criteria=task.acceptance_criteria,
            expected_output_schema_ref=task.expected_output_schema_ref,
            status=models.TaskStatus.pending.value,
            deps=task.deps,
            attempts=0,
            max_attempts=3,
            rework_count=0,
            max_reworks=2,
            assigned_to=None,
            tool_requests=task.tool_requests,
            created_at=now,
            updated_at=now,
            critic_required=1 if task.critic_required else 0,
        )
        db.add(task_record)
    db.commit()
    _emit_event(
        "plan.created",
        {"job_id": job_id, "plan_id": plan_id, "correlation_id": str(uuid.uuid4())},
    )
    return models.Plan(**record.__dict__)
