from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from libs.core import (
    capability_registry,
    execution_contracts,
    intent_contract,
    logging as core_logging,
    models,
    tracing as core_tracing,
)
from services.worker.app import capability_runtime_adapter, tool_runtime_adapter


@dataclass(frozen=True)
class WorkerExecutionConfig:
    llm_provider_name: str
    openai_model: str
    prompt_version: str
    policy_version: str
    tool_version: str
    output_size_cap: int


@dataclass(frozen=True)
class WorkerExecutionContext:
    tool_runtime: tool_runtime_adapter.WorkerToolRuntime
    capability_runtime: capability_runtime_adapter.WorkerCapabilityRuntime
    logger: Any
    config: WorkerExecutionConfig


@dataclass(frozen=True)
class WorkerExecutionCallbacks:
    task_intent_inference: Callable[
        [execution_contracts.TaskExecutionRequest],
        intent_contract.TaskIntentInference,
    ]
    intent_segment: Callable[
        [execution_contracts.TaskExecutionRequest],
        Mapping[str, Any] | None,
    ]
    capability_intent_mismatch: Callable[
        [str, capability_registry.CapabilitySpec],
        str | None,
    ]
    enforce_capability_input_contract: Callable[
        [capability_registry.CapabilitySpec, dict[str, Any]],
        tuple[dict[str, Any], str | None, list[str]],
    ]
    build_tool_payload: Callable[
        [str, str, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]],
        dict[str, Any],
    ]
    intent_mismatch: Callable[[str, models.ToolIntent, str], str | None]
    load_memory_inputs: Callable[[Any, dict[str, Any], str], dict[str, Any]]
    apply_memory_defaults: Callable[[str, dict[str, Any]], dict[str, Any]]
    missing_memory_only_inputs: Callable[[str, dict[str, Any]], list[str]]
    persist_memory_outputs: Callable[[Any, dict[str, Any], models.ToolCall, str], None]
    sync_output_artifact: Callable[[Mapping[str, Any], str | None, str, str], None]
    auto_persist_semantic_facts: Callable[..., None]
    validate_expected_output: Callable[[dict[str, Any], dict[str, Any]], str | None]
    build_task_run_scorecard: Callable[..., dict[str, Any]]


def execute_task_request(
    request: execution_contracts.TaskExecutionRequest,
    *,
    context: WorkerExecutionContext,
    callbacks: WorkerExecutionCallbacks,
) -> models.TaskResult:
    task_payload = dict(request.source_payload)
    task_id = request.task_id
    trace_id = request.trace_id
    run_id = request.run_id or trace_id or str(uuid.uuid4())
    job_id = request.job_id
    tool_requests = request.tool_requests
    task_intent_inference = callbacks.task_intent_inference(request)
    task_intent = task_intent_inference.intent
    task_intent_segment = callbacks.intent_segment(request)
    task_attempt = request.attempts
    task_max_attempts = max(request.max_attempts, request.attempts)
    started_at = datetime.now(UTC)
    outputs: dict[str, Any] = {}
    tool_calls: list[models.ToolCall] = []
    artifacts: list[dict[str, Any]] = []
    tool_error: str | None = None

    core_logging.log_event(
        context.logger,
        "task_intent_inferred",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "task_intent": task_intent,
            "intent_source": task_intent_inference.source,
            "intent_confidence": round(float(task_intent_inference.confidence), 3),
            "trace_id": trace_id,
        },
    )
    with core_tracing.start_span(
        "worker.execute_task",
        attributes={
            "task.id": str(task_id or ""),
            "job.id": job_id,
            "trace.id": trace_id,
            "run.id": run_id,
            "task.tool_request_count": len(tool_requests),
            "task.intent": task_intent,
            "task.intent_source": task_intent_inference.source,
            "task.intent_confidence": float(task_intent_inference.confidence),
            "task.attempt": task_attempt,
            "task.max_attempts": task_max_attempts,
            "model.provider": context.config.llm_provider_name,
            "model.name": context.config.openai_model,
            "prompt.version": context.config.prompt_version,
            "policy.version": context.config.policy_version,
            "tool.version": context.config.tool_version,
        },
    ) as task_span:
        for tool_index, step in enumerate(request.requests):
            request_id = step.request_id
            gate_decision = _evaluate_execution_gate(step.execution_gate, request.context)
            if gate_decision is not None and not gate_decision["allowed"]:
                outputs[request_id] = {
                    "skipped": True,
                    "reason": gate_decision["reason"],
                    "expression": gate_decision["expression"],
                }
                continue
            binding = step.capability_binding
            bound_capability_id = (
                binding.capability_id if binding and binding.capability_id else None
            )
            native_execution_name = (
                binding.tool_name if binding and binding.tool_name else request_id
            )
            capability_spec = None
            if bound_capability_id:
                capability_spec = context.capability_runtime.resolve_enabled_capability(
                    bound_capability_id
                )
            elif binding is None:
                capability_spec = context.capability_runtime.resolve_enabled_capability(
                    request_id
                )
            execution_name = (
                capability_spec.capability_id if capability_spec is not None else native_execution_name
            )
            with core_tracing.start_span(
                "worker.execute_tool",
                attributes={
                    "task.id": str(task_id or ""),
                    "job.id": job_id,
                    "trace.id": trace_id,
                    "run.id": run_id,
                    "tool.name": execution_name,
                    "tool.request_id": request_id,
                    "tool.sequence": tool_index + 1,
                    "task.attempt": task_attempt,
                    "task.max_attempts": task_max_attempts,
                    "model.provider": context.config.llm_provider_name,
                    "model.name": context.config.openai_model,
                    "prompt.version": context.config.prompt_version,
                    "policy.version": context.config.policy_version,
                    "tool.version": context.config.tool_version,
                },
            ) as tool_span:
                if capability_spec is not None:
                    tool_error = _execute_capability_tool(
                        request=request,
                        request_id=request_id,
                        tool_name=capability_spec.capability_id,
                        tool_index=tool_index,
                        task_payload=task_payload,
                        run_id=run_id,
                        trace_id=trace_id,
                        job_id=job_id,
                        task_id=task_id,
                        task_intent=task_intent,
                        task_intent_segment=task_intent_segment,
                        task_intent_inference=task_intent_inference,
                        task_attempt=task_attempt,
                        task_max_attempts=task_max_attempts,
                        started_at=started_at,
                        outputs=outputs,
                        tool_calls=tool_calls,
                        tool_span=tool_span,
                        capability_spec=capability_spec,
                        context=context,
                        callbacks=callbacks,
                    )
                    if tool_error:
                        break
                    continue
                tool_error = _execute_native_tool(
                    request=request,
                    request_id=request_id,
                    tool_name=execution_name,
                    tool_index=tool_index,
                    task_payload=task_payload,
                    run_id=run_id,
                    trace_id=trace_id,
                    job_id=job_id,
                    task_id=task_id,
                    task_intent=task_intent,
                    task_intent_segment=task_intent_segment,
                    task_intent_inference=task_intent_inference,
                    task_attempt=task_attempt,
                    task_max_attempts=task_max_attempts,
                    started_at=started_at,
                    outputs=outputs,
                    tool_calls=tool_calls,
                    tool_span=tool_span,
                    context=context,
                    callbacks=callbacks,
                )
                if tool_error:
                    break
        finished_at = datetime.now(UTC)
        validation_error = callbacks.validate_expected_output(task_payload, outputs)
        status = (
            models.TaskStatus.failed
            if tool_error or validation_error
            else models.TaskStatus.completed
        )
        run_scorecard = callbacks.build_task_run_scorecard(
            run_id=run_id,
            trace_id=trace_id,
            job_id=job_id,
            task_id=str(task_id or ""),
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            tool_calls=tool_calls,
            outputs=outputs,
            task_attempt=task_attempt,
            task_max_attempts=task_max_attempts,
            failure_error=validation_error or tool_error,
        )
        artifacts.append({"type": "run_scorecard", "summary": run_scorecard})
        core_logging.log_event(context.logger, "task_run_scorecard", run_scorecard)
        core_tracing.set_span_attributes(
            task_span,
            {
                "task.status": status.value,
                "task.error": validation_error or tool_error or "",
                "task.tool_calls": len(tool_calls),
                "task.total_latency_ms": run_scorecard.get("total_latency_ms", 0),
                "task.failure_stage": run_scorecard.get("failure_stage", ""),
            },
        )
        if tool_error:
            outputs["tool_error"] = {"error": tool_error}
        if validation_error:
            outputs["validation_error"] = {"error": validation_error}
        return models.TaskResult(
            task_id=task_id,
            status=status,
            outputs=outputs,
            artifacts=artifacts,
            tool_calls=tool_calls,
            started_at=started_at,
            finished_at=finished_at,
            error=validation_error or tool_error,
        )


def _evaluate_execution_gate(
    gate: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(gate, Mapping):
        return None
    expression = str(gate.get("expression") or "").strip()
    negate = bool(gate.get("negate"))
    if not expression:
        return None
    try:
        value = _evaluate_context_expression(expression, context or {})
    except Exception as exc:  # noqa: BLE001
        return {
            "allowed": False,
            "expression": expression,
            "reason": f"execution_gate_invalid:{exc}",
        }
    allowed = bool(value)
    if negate:
        allowed = not allowed
    return {
        "allowed": allowed,
        "expression": expression,
        "reason": "execution_gate_condition_false" if not negate else "execution_gate_condition_negated_false",
    }


def _evaluate_context_expression(expression: str, context: Mapping[str, Any]) -> Any:
    for operator in ("==", "!="):
        if operator in expression:
            left, right = expression.split(operator, 1)
            left_value = _resolve_context_operand(left.strip(), context)
            right_value = _parse_expression_literal(right.strip(), context)
            return left_value == right_value if operator == "==" else left_value != right_value
    return _resolve_context_operand(expression.strip(), context)


def _resolve_context_operand(token: str, context: Mapping[str, Any]) -> Any:
    normalized = token.strip()
    if not normalized:
        return None
    if not normalized.startswith("context."):
        raise ValueError("unsupported_operand")
    value: Any = context
    for segment in normalized.split(".")[1:]:
        key = segment.strip()
        if not key:
            raise ValueError("empty_context_segment")
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _parse_expression_literal(token: str, context: Mapping[str, Any]) -> Any:
    normalized = token.strip()
    lowered = normalized.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if normalized.startswith(("context.",)):
        return _resolve_context_operand(normalized, context)
    if normalized.startswith(("'", '"')) and normalized.endswith(("'", '"')) and len(normalized) >= 2:
        return normalized[1:-1]
    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except ValueError:
        return normalized


def _execute_capability_tool(
    *,
    request: execution_contracts.TaskExecutionRequest,
    request_id: str,
    tool_name: str,
    tool_index: int,
    task_payload: dict[str, Any],
    run_id: str,
    trace_id: str,
    job_id: str,
    task_id: str,
    task_intent: str,
    task_intent_segment: Mapping[str, Any] | None,
    task_intent_inference: intent_contract.TaskIntentInference,
    task_attempt: int,
    task_max_attempts: int,
    started_at: datetime,
    outputs: dict[str, Any],
    tool_calls: list[models.ToolCall],
    tool_span: Any,
    capability_spec: capability_registry.CapabilitySpec,
    context: WorkerExecutionContext,
    callbacks: WorkerExecutionCallbacks,
) -> str | None:
    capability_allow_decision = context.capability_runtime.evaluate_allowlist(
        capability_spec.capability_id
    )
    if capability_allow_decision.violated and capability_allow_decision.mode == "dry_run":
        core_logging.log_event(
            context.logger,
            "capability_governance_violation_dry_run",
            {
                "run_id": run_id,
                "job_id": job_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "capability_id": capability_spec.capability_id,
                "reason": capability_allow_decision.reason,
                "mode": capability_allow_decision.mode,
                "trace_id": trace_id,
            },
        )
    if not capability_allow_decision.allowed:
        tool_error = (
            "policy.capability_not_allowed:"
            f"{capability_spec.capability_id}:"
            f"{capability_allow_decision.reason}"
        )
        outputs[request_id] = {
            "error": tool_error,
            "error_code": "policy.capability_not_allowed",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload={},
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
                error_code="policy.capability_not_allowed",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": tool_error,
                "tool.error_code": "policy.capability_not_allowed",
                "capability.id": capability_spec.capability_id,
                "capability.governance_mode": capability_allow_decision.mode,
                "capability.allowlist_reason": capability_allow_decision.reason,
            },
        )
        return tool_error
    capability_mismatch = callbacks.capability_intent_mismatch(task_intent, capability_spec)
    if capability_mismatch:
        tool_error = f"contract.intent_mismatch:{capability_mismatch}"
        outputs[request_id] = {
            "error": tool_error,
            "error_code": "contract.intent_mismatch",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload={},
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
                error_code="contract.intent_mismatch",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": tool_error,
                "tool.error_code": "contract.intent_mismatch",
                "capability.id": capability_spec.capability_id,
            },
        )
        return tool_error
    payload = callbacks.build_tool_payload(
        tool_name,
        request.instruction,
        request.context,
        task_payload,
        _request_tool_inputs(request, request_id, tool_name),
    )
    payload, capability_payload_error, dropped_payload_keys = (
        callbacks.enforce_capability_input_contract(
            capability_spec,
            payload,
        )
    )
    if dropped_payload_keys:
        core_logging.log_event(
            context.logger,
            "capability_payload_pruned",
            {
                "run_id": run_id,
                "job_id": job_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "capability_id": capability_spec.capability_id,
                "dropped_keys": dropped_payload_keys,
                "trace_id": trace_id,
            },
        )
    if capability_payload_error:
        outputs[request_id] = {
            "error": capability_payload_error,
            "error_code": "contract.input_invalid",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload=payload,
                trace_id=trace_id,
                started_at=started_at,
                error=capability_payload_error,
                error_code="contract.input_invalid",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": capability_payload_error,
                "tool.error_code": "contract.input_invalid",
                "capability.id": capability_spec.capability_id,
            },
        )
        return capability_payload_error
    segment_contract_error = intent_contract.validate_intent_segment_contract(
        segment=task_intent_segment,
        task_intent=task_intent,
        tool_name=tool_name,
        payload=payload,
        capability_id=capability_spec.capability_id,
        capability_risk_tier=capability_spec.risk_tier,
    )
    if segment_contract_error:
        tool_error = f"contract.intent_mismatch:{segment_contract_error}"
        outputs[request_id] = {
            "error": tool_error,
            "error_code": "contract.intent_mismatch",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload=payload,
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
                error_code="contract.intent_mismatch",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": tool_error,
                "tool.error_code": "contract.intent_mismatch",
                "capability.id": capability_spec.capability_id,
            },
        )
        return tool_error
    idempotency_key = str(uuid.uuid4())
    tool_started_at = time.monotonic()
    core_logging.log_event(
        context.logger,
        "tool_call_started",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "tool_intent": f"capability:{capability_spec.idempotency}",
            "tool_timeout_s": capability_spec.adapters[0].timeout_s
            if capability_spec.adapters and capability_spec.adapters[0].timeout_s is not None
            else None,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "model_provider": context.config.llm_provider_name,
            "model_name": context.config.openai_model,
            "prompt_version": context.config.prompt_version,
            "policy_version": context.config.policy_version,
            "tool_version": context.config.tool_version,
            "payload_keys": sorted(payload.keys()),
            "capability_id": capability_spec.capability_id,
            "task_intent": task_intent,
            "intent_source": task_intent_inference.source,
            "intent_confidence": round(float(task_intent_inference.confidence), 3),
        },
    )
    call = context.capability_runtime.execute_capability(
        capability_id=capability_spec.capability_id,
        payload=payload,
        trace_id=trace_id,
        idempotency_key=idempotency_key,
        task_payload=task_payload,
        tool_runtime=context.tool_runtime,
    )
    duration_ms = int((time.monotonic() - tool_started_at) * 1000)
    core_logging.log_event(
        context.logger,
        "tool_call_finished",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "status": call.status,
            "duration_ms": duration_ms,
            "error_code": call.output_or_error.get("error_code"),
            "error": call.output_or_error.get("error"),
            "capability_id": capability_spec.capability_id,
        },
    )
    core_tracing.set_span_attributes(
        tool_span,
        {
            "tool.status": call.status,
            "tool.idempotency_key": idempotency_key,
            "tool.error": call.output_or_error.get("error", ""),
            "tool.error_code": call.output_or_error.get("error_code", ""),
            "tool.duration_ms": duration_ms,
            "tool.is_capability": True,
            "capability.id": capability_spec.capability_id,
        },
    )
    tool_calls.append(call)
    outputs[request_id] = call.output_or_error
    if call.status == "completed":
        callbacks.sync_output_artifact(call.output_or_error, task_id, tool_name, trace_id)
        if isinstance(call.output_or_error, Mapping):
            callbacks.auto_persist_semantic_facts(
                tool_name=tool_name,
                task_payload=task_payload,
                output=call.output_or_error,
                trace_id=trace_id,
                run_id=run_id,
            )
        return None
    tool_error = str(call.output_or_error.get("error", "capability_failed"))
    normalized_error = _normalize_timeout_error(tool_error)
    if normalized_error != tool_error:
        tool_error = normalized_error
        call.output_or_error["error"] = tool_error
        outputs[request_id] = call.output_or_error
    core_logging.log_event(
        context.logger,
        "tool_call_failed",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "error": tool_error,
            "error_code": call.output_or_error.get("error_code"),
            "duration_ms": duration_ms,
            "capability_id": capability_spec.capability_id,
        },
    )
    return tool_error


def _execute_native_tool(
    *,
    request: execution_contracts.TaskExecutionRequest,
    request_id: str,
    tool_name: str,
    tool_index: int,
    task_payload: dict[str, Any],
    run_id: str,
    trace_id: str,
    job_id: str,
    task_id: str,
    task_intent: str,
    task_intent_segment: Mapping[str, Any] | None,
    task_intent_inference: intent_contract.TaskIntentInference,
    task_attempt: int,
    task_max_attempts: int,
    started_at: datetime,
    outputs: dict[str, Any],
    tool_calls: list[models.ToolCall],
    tool_span: Any,
    context: WorkerExecutionContext,
    callbacks: WorkerExecutionCallbacks,
) -> str | None:
    try:
        tool = context.tool_runtime.get_tool(tool_name)
    except KeyError:
        tool_error = f"contract.tool_not_found:unknown_tool:{tool_name}"
        outputs[request_id] = {"error": tool_error}
        core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
        return tool_error
    governance_context = {
        "job_id": job_id,
        "tenant_id": task_payload.get("tenant_id"),
        "org_id": task_payload.get("org_id"),
        "job_type": task_payload.get("job_type"),
        "context": request.context,
        "job_context": request.context.get("job_context")
        if isinstance(request.context, dict)
        else None,
    }
    allow_decision = context.tool_runtime.evaluate_allowlist(
        tool_name,
        context=governance_context,
        tool_spec=tool.spec,
    )
    if allow_decision.violated and allow_decision.mode == "dry_run":
        core_logging.log_event(
            context.logger,
            "tool_governance_violation_dry_run",
            {
                "run_id": run_id,
                "job_id": job_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "reason": allow_decision.reason,
                "mode": allow_decision.mode,
                "trace_id": trace_id,
            },
        )
    if not allow_decision.allowed:
        tool_error = f"policy.tool_not_allowed:{tool_name}:{allow_decision.reason}"
        outputs[request_id] = {
            "error": tool_error,
            "error_code": "policy.tool_not_allowed",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload={},
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
                error_code="policy.tool_not_allowed",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": tool_error,
                "tool.error_code": "policy.tool_not_allowed",
                "tool.allowlist_reason": allow_decision.reason,
                "tool.governance_mode": allow_decision.mode,
            },
        )
        return tool_error
    mismatch = callbacks.intent_mismatch(task_intent, tool.spec.tool_intent, tool_name)
    if mismatch:
        tool_error = f"contract.intent_mismatch:{mismatch}"
        outputs[request_id] = {"error": tool_error}
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload={},
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
            )
        )
        core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
        return tool_error
    payload = callbacks.build_tool_payload(
        tool_name,
        request.instruction,
        request.context,
        task_payload,
        _request_tool_inputs(request, request_id, tool_name),
    )
    memory_payload = callbacks.load_memory_inputs(tool, task_payload, trace_id)
    if memory_payload:
        payload.setdefault("memory", {}).update(memory_payload)
        payload = callbacks.apply_memory_defaults(tool.spec.name, payload)
        missing = callbacks.missing_memory_only_inputs(tool.spec.name, payload)
        if missing:
            tool_error = (
                f"contract.input_missing:memory_only_inputs_missing:{','.join(missing)}"
            )
            outputs[request_id] = {"error": tool_error}
            tool_calls.append(
                _failed_tool_call(
                    tool_name=tool_name,
                    input_payload=payload,
                    trace_id=trace_id,
                    started_at=started_at,
                    error=tool_error,
                )
            )
            core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
            return tool_error
    segment_contract_error = intent_contract.validate_intent_segment_contract(
        segment=task_intent_segment,
        task_intent=task_intent,
        tool_name=tool_name,
        payload=payload,
        capability_id=None,
        capability_risk_tier=None,
    )
    if segment_contract_error:
        tool_error = f"contract.intent_mismatch:{segment_contract_error}"
        outputs[request_id] = {
            "error": tool_error,
            "error_code": "contract.intent_mismatch",
        }
        tool_calls.append(
            _failed_tool_call(
                tool_name=tool_name,
                input_payload=payload,
                trace_id=trace_id,
                started_at=started_at,
                error=tool_error,
                error_code="contract.intent_mismatch",
            )
        )
        core_tracing.set_span_attributes(
            tool_span,
            {
                "tool.error": tool_error,
                "tool.error_code": "contract.intent_mismatch",
            },
        )
        return tool_error
    payload = dict(payload)
    payload["_registry"] = context.tool_runtime.registry
    idempotency_key = str(uuid.uuid4())
    tool_started_at = time.monotonic()
    core_logging.log_event(
        context.logger,
        "tool_call_started",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "tool_intent": tool.spec.tool_intent.value,
            "tool_timeout_s": tool.spec.timeout_s,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "model_provider": context.config.llm_provider_name,
            "model_name": context.config.openai_model,
            "prompt_version": context.config.prompt_version,
            "policy_version": context.config.policy_version,
            "tool_version": context.config.tool_version,
            "payload_keys": sorted(payload.keys()),
            "task_intent": task_intent,
            "intent_source": task_intent_inference.source,
            "intent_confidence": round(float(task_intent_inference.confidence), 3),
        },
    )
    call = context.tool_runtime.execute_tool(
        tool_name,
        payload=payload,
        idempotency_key=idempotency_key,
        trace_id=trace_id,
        max_output_bytes=context.config.output_size_cap,
    )
    duration_ms = int((time.monotonic() - tool_started_at) * 1000)
    core_logging.log_event(
        context.logger,
        "tool_call_finished",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "status": call.status,
            "duration_ms": duration_ms,
            "error_code": call.output_or_error.get("error_code"),
            "error": call.output_or_error.get("error"),
        },
    )
    core_tracing.set_span_attributes(
        tool_span,
        {
            "tool.status": call.status,
            "tool.idempotency_key": idempotency_key,
            "tool.error": call.output_or_error.get("error", ""),
            "tool.error_code": call.output_or_error.get("error_code", ""),
            "tool.duration_ms": duration_ms,
        },
    )
    tool_calls.append(call)
    outputs[request_id] = call.output_or_error
    if call.status == "completed":
        callbacks.sync_output_artifact(call.output_or_error, task_id, tool_name, trace_id)
        callbacks.persist_memory_outputs(tool, task_payload, call, trace_id)
        if isinstance(call.output_or_error, Mapping):
            callbacks.auto_persist_semantic_facts(
                tool_name=tool_name,
                task_payload=task_payload,
                output=call.output_or_error,
                trace_id=trace_id,
                run_id=run_id,
            )
        return None
    tool_error = str(call.output_or_error.get("error", "tool_failed"))
    error_code = call.output_or_error.get("error_code")
    if isinstance(error_code, str) and error_code and not tool_error.startswith(f"{error_code}:"):
        tool_error = f"{error_code}:{tool_error}"
        call.output_or_error["error"] = tool_error
        outputs[request_id] = call.output_or_error
    normalized_error = _normalize_timeout_error(tool_error)
    if normalized_error != tool_error:
        tool_error = normalized_error
        call.output_or_error["error"] = tool_error
        outputs[request_id] = call.output_or_error
    core_logging.log_event(
        context.logger,
        "tool_call_failed",
        {
            "run_id": run_id,
            "job_id": job_id,
            "task_id": task_id,
            "tool_sequence": tool_index + 1,
            "task_attempt": task_attempt,
            "task_max_attempts": task_max_attempts,
            "tool_name": tool_name,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "error": tool_error,
            "error_code": call.output_or_error.get("error_code"),
            "duration_ms": duration_ms,
        },
    )
    return tool_error


def _request_tool_inputs(
    request: execution_contracts.TaskExecutionRequest,
    request_id: str,
    tool_name: str,
) -> dict[str, dict[str, Any]]:
    tool_inputs = dict(request.tool_inputs)
    request_inputs = dict(tool_inputs.get(request_id) or {})
    if tool_name and tool_name != request_id:
        tool_inputs[tool_name] = request_inputs
    return tool_inputs


def _failed_tool_call(
    *,
    tool_name: str,
    input_payload: dict[str, Any],
    trace_id: str,
    started_at: datetime,
    error: str,
    error_code: str | None = None,
) -> models.ToolCall:
    output_or_error = {"error": error}
    if error_code:
        output_or_error["error_code"] = error_code
    return models.ToolCall(
        tool_name=tool_name,
        input=input_payload,
        idempotency_key=str(uuid.uuid4()),
        trace_id=trace_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        status="failed",
        output_or_error=output_or_error,
    )


def _normalize_timeout_error(error: str) -> str:
    lowered = error.lower()
    if (
        "timed out" in lowered
        or "timeout" in lowered
        or "mcp_sdk_timeout:" in lowered
    ) and not error.startswith("tool_call_timed_out"):
        return f"tool_call_timed_out:{error}"
    return error
