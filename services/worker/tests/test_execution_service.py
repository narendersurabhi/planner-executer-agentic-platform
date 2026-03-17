from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime

from libs.core import capability_registry, execution_contracts, intent_contract, models
from services.worker.app import execution_service


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.spec = models.ToolSpec(
            name=name,
            description="Fake tool",
            input_schema={},
            output_schema={},
            tool_intent=models.ToolIntent.generate,
        )


class _FakeToolRuntime:
    def __init__(self) -> None:
        self.registry = self
        self.tool = _FakeTool("llm_generate")
        self.executed_payloads: list[dict[str, object]] = []

    def get_tool(self, tool_name: str) -> _FakeTool:
        assert tool_name == "llm_generate"
        return self.tool

    def execute_tool(
        self,
        tool_name: str,
        *,
        payload: dict[str, object],
        idempotency_key: str,
        trace_id: str,
        max_output_bytes: int,
    ) -> models.ToolCall:
        self.executed_payloads.append(dict(payload))
        return models.ToolCall(
            tool_name=tool_name,
            input=dict(payload),
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            status="completed",
            output_or_error={"text": "hello world"},
        )

    def evaluate_allowlist(
        self,
        tool_name: str,
        *,
        context: dict[str, object] | None = None,
        tool_spec: models.ToolSpec | None = None,
    ) -> object:
        return type(
            "AllowDecision",
            (),
            {"allowed": True, "violated": False, "reason": "", "mode": "enforce"},
        )()


class _FakeCapabilityRuntime:
    def resolve_enabled_capability(self, capability_id: str) -> None:
        assert capability_id == "llm_generate"
        return None


def test_execute_task_request_runs_native_tool(monkeypatch) -> None:
    tool_runtime = _FakeToolRuntime()
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Write hello",
        source_payload={"task_id": "task-1", "instruction": "Write hello"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="llm_generate",
                resolved_inputs={"text": "hello"},
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )
    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=tool_runtime,
            capability_runtime=_FakeCapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="generate",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["llm_generate"]["text"] == "hello world"
    assert tool_runtime.executed_payloads[0]["text"] == "hello"
    assert tool_runtime.executed_payloads[0]["_registry"] is tool_runtime.registry


def test_execute_task_request_runs_capability_tool(monkeypatch) -> None:
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Check repo",
        source_payload={"task_id": "task-1", "instruction": "Check repo"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="github.repo.list",
                resolved_inputs={"owner": "narendersurabhi", "repo": "scientific-agent-lab"},
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )
    spec = capability_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
    )

    class _CapabilityRuntime:
        def resolve_enabled_capability(self, capability_id: str):
            assert capability_id == "github.repo.list"
            return spec

        def evaluate_allowlist(self, capability_id: str):
            assert capability_id == "github.repo.list"
            return capability_registry.CapabilityAllowDecision(
                allowed=True,
                reason="allowed",
                mode="enforce",
                violated=False,
            )

        def execute_capability(self, **kwargs):
            return models.ToolCall(
                tool_name="github.repo.list",
                input=dict(kwargs["payload"]),
                idempotency_key=kwargs["idempotency_key"],
                trace_id=kwargs["trace_id"],
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                status="completed",
                output_or_error={"items": [{"name": "scientific-agent-lab"}]},
            )

    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=_FakeToolRuntime(),
            capability_runtime=_CapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="io",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["github.repo.list"]["items"][0]["name"] == "scientific-agent-lab"


def test_execute_task_request_uses_bound_native_tool_name(monkeypatch) -> None:
    tool_runtime = _FakeToolRuntime()
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Write hello",
        source_payload={"task_id": "task-1", "instruction": "Write hello"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="generate_text",
                resolved_inputs={"text": "hello"},
                capability_binding=execution_contracts.CapabilityBinding(
                    request_id="generate_text",
                    tool_name="llm_generate",
                    adapter_type="tool",
                ),
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )

    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=tool_runtime,
            capability_runtime=_FakeCapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="generate",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["generate_text"]["text"] == "hello world"
    assert tool_runtime.executed_payloads[0]["text"] == "hello"


def test_execute_task_request_uses_bound_capability_id(monkeypatch) -> None:
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Check repo",
        source_payload={"task_id": "task-1", "instruction": "Check repo"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="check_repo",
                resolved_inputs={"owner": "narendersurabhi", "repo": "scientific-agent-lab"},
                capability_binding=execution_contracts.CapabilityBinding(
                    request_id="check_repo",
                    capability_id="github.repo.list",
                    tool_name="github.repo.list",
                    adapter_type="mcp",
                    server_id="github_local",
                ),
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )
    spec = capability_registry.CapabilitySpec(
        capability_id="github.repo.list",
        description="List repos",
        risk_tier="read_only",
        idempotency="read",
    )

    class _CapabilityRuntime:
        def resolve_enabled_capability(self, capability_id: str):
            assert capability_id == "github.repo.list"
            return spec

        def evaluate_allowlist(self, capability_id: str):
            assert capability_id == "github.repo.list"
            return capability_registry.CapabilityAllowDecision(
                allowed=True,
                reason="allowed",
                mode="enforce",
                violated=False,
            )

        def execute_capability(self, **kwargs):
            return models.ToolCall(
                tool_name="github.repo.list",
                input=dict(kwargs["payload"]),
                idempotency_key=kwargs["idempotency_key"],
                trace_id=kwargs["trace_id"],
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                status="completed",
                output_or_error={"items": [{"name": "scientific-agent-lab"}]},
            )

    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=_FakeToolRuntime(),
            capability_runtime=_CapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="io",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["check_repo"]["items"][0]["name"] == "scientific-agent-lab"


def test_execute_task_request_skips_when_execution_gate_is_false(monkeypatch) -> None:
    tool_runtime = _FakeToolRuntime()
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Write hello",
        context={"approved": False},
        source_payload={"task_id": "task-1", "instruction": "Write hello"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="llm_generate",
                resolved_inputs={"text": "hello"},
                execution_gate={"expression": "context.approved == true"},
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )
    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=tool_runtime,
            capability_runtime=_FakeCapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="generate",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["llm_generate"]["skipped"] is True
    assert tool_runtime.executed_payloads == []


def test_execute_task_request_runs_when_execution_gate_is_negated(monkeypatch) -> None:
    tool_runtime = _FakeToolRuntime()
    request = execution_contracts.TaskExecutionRequest(
        task_id="task-1",
        job_id="job-1",
        run_id="run-1",
        trace_id="trace-1",
        instruction="Write hello",
        context={"approved": False},
        source_payload={"task_id": "task-1", "instruction": "Write hello"},
        requests=[
            execution_contracts.TaskExecutionStep(
                request_id="llm_generate",
                resolved_inputs={"text": "hello"},
                execution_gate={"expression": "context.approved == true", "negate": True},
            )
        ],
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "start_span",
        lambda *args, **kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        execution_service.core_tracing,
        "set_span_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        execution_service.core_logging,
        "log_event",
        lambda *args, **kwargs: None,
    )
    result = execution_service.execute_task_request(
        request,
        context=execution_service.WorkerExecutionContext(
            tool_runtime=tool_runtime,
            capability_runtime=_FakeCapabilityRuntime(),
            logger=object(),
            config=execution_service.WorkerExecutionConfig(
                llm_provider_name="mock",
                openai_model="",
                prompt_version="test",
                policy_version="test",
                tool_version="test",
                output_size_cap=1024,
            ),
        ),
        callbacks=execution_service.WorkerExecutionCallbacks(
            task_intent_inference=lambda _request: intent_contract.TaskIntentInference(
                intent="generate",
                source="unit_test",
                confidence=1.0,
            ),
            intent_segment=lambda _request: None,
            capability_intent_mismatch=lambda _intent, _spec: None,
            enforce_capability_input_contract=lambda spec, payload: (payload, None, []),
            build_tool_payload=lambda tool_name, instruction, context, task_payload, tool_inputs: dict(
                tool_inputs.get(tool_name, {})
            ),
            intent_mismatch=lambda _intent, _tool_intent, _tool_name: None,
            load_memory_inputs=lambda _tool, _task_payload, _trace_id: {},
            apply_memory_defaults=lambda _tool_name, payload: payload,
            missing_memory_only_inputs=lambda _tool_name, _payload: [],
            persist_memory_outputs=lambda _tool, _task_payload, _call, _trace_id: None,
            sync_output_artifact=lambda _output, _task_id, _tool_name, _trace_id: None,
            auto_persist_semantic_facts=lambda **kwargs: None,
            validate_expected_output=lambda _task_payload, _outputs: None,
            build_task_run_scorecard=lambda **kwargs: {
                "total_latency_ms": 0,
                "failure_stage": "",
            },
        ),
    )

    assert result.status == models.TaskStatus.completed
    assert result.outputs["llm_generate"]["text"] == "hello world"
