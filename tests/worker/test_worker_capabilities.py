from __future__ import annotations

from services.worker.app import main as worker_main
from libs.core import capability_registry


class _DummyRegistry:
    def get(self, _name: str):
        raise AssertionError("registry.get should not be called for capability execution")


def _capability_spec() -> capability_registry.CapabilitySpec:
    return capability_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="mcp",
                server_id="github_remote",
                tool_name="github_repo_list",
                timeout_s=30,
            ),
        ),
        enabled=True,
    )


def _local_document_validate_capability_spec() -> capability_registry.CapabilitySpec:
    return capability_registry.CapabilitySpec(
        capability_id="document.spec.validate",
        description="Validate DocumentSpec",
        risk_tier="read_only",
        idempotency="read",
        adapters=(
            capability_registry.CapabilityAdapterSpec(
                type="tool",
                server_id="local_worker",
                tool_name="document_spec_validate",
                timeout_s=30,
            ),
        ),
        enabled=True,
    )


def test_execute_task_runs_capability_request(monkeypatch) -> None:
    monkeypatch.setattr(worker_main.tool_registry, "default_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(
        worker_main,
        "_resolve_enabled_capability_request",
        lambda name: _capability_spec() if name == "github.repo.list" else None,
    )
    monkeypatch.setattr(
        worker_main.mcp_gateway,
        "invoke_capability",
        lambda capability_id, arguments, **_: {
            "capability_id": capability_id,
            "arguments": arguments,
            "repos": ["awe", "platform"],
        },
    )
    task_payload = {
        "task_id": "task-1",
        "job_id": "job-1",
        "correlation_id": "trace-1",
        "tool_requests": ["github.repo.list"],
        "instruction": "List my repositories",
        "context": {"job_context": {"topic": "agentic"}},
        "tool_inputs": {"github.repo.list": {"query": "agentic"}},
    }
    result = worker_main.execute_task(task_payload)
    assert result.status == worker_main.models.TaskStatus.completed
    assert result.outputs["github.repo.list"]["repos"] == ["awe", "platform"]


def test_execute_task_marks_capability_timeout(monkeypatch) -> None:
    monkeypatch.setattr(worker_main.tool_registry, "default_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(
        worker_main,
        "_resolve_enabled_capability_request",
        lambda name: _capability_spec() if name == "github.repo.list" else None,
    )

    def _raise_timeout(_capability_id: str, _arguments: dict, **_: dict):
        raise RuntimeError("mcp_sdk_timeout:phase=initialize;mcp_call_timed_out_after_10.0s")

    monkeypatch.setattr(worker_main.mcp_gateway, "invoke_capability", _raise_timeout)
    task_payload = {
        "task_id": "task-2",
        "job_id": "job-1",
        "correlation_id": "trace-2",
        "tool_requests": ["github.repo.list"],
        "instruction": "List my repositories",
        "context": {},
        "tool_inputs": {"github.repo.list": {"query": "agentic"}},
    }
    result = worker_main.execute_task(task_payload)
    assert result.status == worker_main.models.TaskStatus.failed
    tool_error = result.outputs["github.repo.list"]["error"]
    assert "tool_call_timed_out" in tool_error


def test_capability_native_tool_hydrates_memory_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        worker_main,
        "_resolve_enabled_capability_request",
        lambda name: _local_document_validate_capability_spec()
        if name == "document.spec.validate"
        else None,
    )
    monkeypatch.setattr(worker_main, "MEMORY_WRITE_ENABLED", False)
    monkeypatch.setattr(
        worker_main,
        "_load_memory_inputs",
        lambda tool, task_payload, trace_id: {
            "task_outputs": [
                {"document_spec": {"blocks": [{"type": "paragraph", "text": "hello"}]}}
            ]
        },
    )

    def _invoke_capability(capability_id: str, arguments: dict, **kwargs):
        execute_tool = kwargs.get("execute_tool")
        assert callable(execute_tool)
        return execute_tool("document_spec_validate", arguments)

    monkeypatch.setattr(worker_main.mcp_gateway, "invoke_capability", _invoke_capability)
    task_payload = {
        "task_id": "task-3",
        "job_id": "job-1",
        "correlation_id": "trace-3",
        "tool_requests": ["document.spec.validate"],
        "instruction": "Validate the spec",
        "context": {},
        "tool_inputs": {"document.spec.validate": {"strict": True}},
    }
    result = worker_main.execute_task(task_payload)
    assert result.status == worker_main.models.TaskStatus.completed
    output = result.outputs["document.spec.validate"]
    assert output["valid"] is True
