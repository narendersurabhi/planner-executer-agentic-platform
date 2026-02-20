from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

from libs.core import models


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


def test_parse_llm_plan_accepts_plan_wrapped_object() -> None:
    planner_main = _load_planner_module()
    plan = _sample_plan_json()
    content = json.dumps({"plan": plan})
    parsed = planner_main._parse_llm_plan(content)
    assert parsed is not None
    assert parsed.tasks_summary == "sample"


class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls = 0

    def generate(self, _prompt: str) -> _FakeLLMResponse:
        self.calls += 1
        if not self._outputs:
            raise RuntimeError("no_more_outputs")
        return _FakeLLMResponse(self._outputs.pop(0))


def test_llm_plan_retries_with_repair_prompt_when_initial_parse_fails() -> None:
    planner_main = _load_planner_module()
    plan = _sample_plan_json()
    provider = _FakeProvider(
        outputs=[
            '{"planner_version":"1.0.0","tasks_summary":"broken"',  # malformed json
            json.dumps(plan),
        ]
    )
    job = models.Job(
        id="job-1",
        goal="Generate a sample plan",
        context_json={},
        status=models.JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        priority=1,
        metadata={},
    )
    tools = [
        models.ToolSpec(
            name="llm_generate",
            description="Generate text",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            output_schema={"type": "object"},
            tool_intent=models.ToolIntent.generate,
        )
    ]

    parsed = planner_main.llm_plan(job, tools, provider)

    assert parsed.tasks_summary == "sample"
    assert provider.calls == 2
