from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    goal: Mapped[str] = mapped_column(String)
    context_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    plan: Mapped["PlanRecord"] = relationship("PlanRecord", back_populates="job", uselist=False)


class PlanRecord(Base):
    __tablename__ = "plans"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    planner_version: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    tasks_summary: Mapped[str] = mapped_column(String)
    dag_edges: Mapped[List[List[str]]] = mapped_column(JSON, default=list)
    policy_decision: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    job: Mapped[JobRecord] = relationship("JobRecord", back_populates="plan")
    tasks: Mapped[List["TaskRecord"]] = relationship("TaskRecord", back_populates="plan")


class TaskRecord(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id"))
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    instruction: Mapped[str] = mapped_column(String)
    acceptance_criteria: Mapped[List[str]] = mapped_column(JSON, default=list)
    expected_output_schema_ref: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    deps: Mapped[List[str]] = mapped_column(JSON, default=list)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    rework_count: Mapped[int] = mapped_column(Integer, default=0)
    max_reworks: Mapped[int] = mapped_column(Integer, default=2)
    assigned_to: Mapped[str | None] = mapped_column(String, nullable=True)
    intent: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_requests: Mapped[List[str]] = mapped_column(JSON, default=list)
    tool_inputs: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    critic_required: Mapped[bool] = mapped_column(Integer, default=1)

    plan: Mapped[PlanRecord] = relationship("PlanRecord", back_populates="tasks")
