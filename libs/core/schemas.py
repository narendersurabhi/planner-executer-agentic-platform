from __future__ import annotations

from pathlib import Path
from typing import Dict, Type

from pydantic import BaseModel

from . import models

SCHEMA_TARGETS: Dict[str, Type[BaseModel]] = {
    "Job": models.Job,
    "Plan": models.Plan,
    "Task": models.Task,
    "ToolSpec": models.ToolSpec,
    "ToolCall": models.ToolCall,
    "TaskResult": models.TaskResult,
    "CriticResult": models.CriticResult,
    "PolicyDecision": models.PolicyDecision,
    "EventEnvelope": models.EventEnvelope,
    "JobCreate": models.JobCreate,
    "PlanCreate": models.PlanCreate,
}


def export_schemas(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_TARGETS.items():
        schema_path = target_dir / f"{name}.json"
        schema_path.write_text(model.model_json_schema_json(indent=2))
