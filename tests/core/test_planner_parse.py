from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_planner_module():
    if "redis" not in sys.modules:
        redis_stub = types.ModuleType("redis")

        class _FakeRedisClient:
            @classmethod
            def from_url(cls, *_args, **_kwargs):
                return object()

        redis_stub.Redis = _FakeRedisClient
        redis_stub.ResponseError = Exception
        sys.modules["redis"] = redis_stub

    root = Path(__file__).resolve().parents[2]
    module_path = root / "services" / "planner" / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("planner_main_test_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_plan_json() -> dict:
    return {
        "planner_version": "1.0.0",
        "tasks_summary": "sample",
        "dag_edges": [],
        "tasks": [
            {
                "name": "TaskA",
                "description": "desc",
                "instruction": "instr",
                "acceptance_criteria": ["ok"],
                "expected_output_schema_ref": "schemas/task-output",
                "deps": [],
                "tool_requests": ["llm_generate"],
                "tool_inputs": {"llm_generate": {"text": "hello"}},
                "critic_required": False,
            }
        ],
    }


def test_parse_llm_plan_accepts_double_encoded_json() -> None:
    planner_main = _load_planner_module()
    plan = _sample_plan_json()
    content = json.dumps(json.dumps(plan))
    parsed = planner_main._parse_llm_plan(content)
    assert parsed is not None
    assert parsed.planner_version == "1.0.0"
    assert parsed.tasks[0].name == "TaskA"


def test_parse_llm_plan_accepts_markdown_json() -> None:
    planner_main = _load_planner_module()
    plan = _sample_plan_json()
    content = "```json\n" + json.dumps(plan, indent=2) + "\n```"
    parsed = planner_main._parse_llm_plan(content)
    assert parsed is not None
    assert parsed.tasks_summary == "sample"
