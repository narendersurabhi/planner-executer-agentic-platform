from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from libs.core import intent_contract, workflow_contracts

EXECUTION_BINDINGS_KEY = "__capability_bindings__"
EXECUTION_GATE_KEY = "__execution_gate__"


class CapabilityBinding(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    capability_id: str | None = None
    tool_name: str | None = None
    adapter_type: str | None = None
    server_id: str | None = None


class TaskExecutionStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    resolved_inputs: dict[str, Any] = Field(default_factory=dict)
    capability_binding: CapabilityBinding | None = None
    execution_gate: dict[str, Any] | None = None


class TaskExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = ""
    job_id: str = ""
    run_id: str = ""
    trace_id: str = ""
    name: str = ""
    description: str = ""
    instruction: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 1
    max_attempts: int = 1
    intent: str | None = None
    intent_source: str | None = None
    intent_confidence: float | None = None
    intent_segment: workflow_contracts.IntentGraphSegment | None = None
    dependency_artifacts: dict[str, Any] = Field(default_factory=dict)
    retry_policy: str | None = None
    source_payload: dict[str, Any] = Field(default_factory=dict, exclude=True)
    requests: list[TaskExecutionStep] = Field(default_factory=list)

    @property
    def tool_requests(self) -> list[str]:
        return [request.request_id for request in self.requests]

    @property
    def tool_inputs(self) -> dict[str, dict[str, Any]]:
        return {
            request.request_id: dict(request.resolved_inputs)
            for request in self.requests
        }


class TaskDispatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = ""
    id: str = ""
    job_id: str = ""
    plan_id: str = ""
    name: str = ""
    description: str = ""
    instruction: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    expected_output_schema_ref: str = ""
    status: str = ""
    deps: list[str] = Field(default_factory=list)
    attempts: int = 1
    max_attempts: int = 1
    rework_count: int = 0
    max_reworks: int = 0
    assigned_to: str | None = None
    tool_requests: list[str] = Field(default_factory=list)
    tool_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    capability_bindings: dict[str, CapabilityBinding] = Field(default_factory=dict)
    execution_gates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    critic_required: bool = True
    intent: str | None = None
    intent_source: str | None = None
    intent_confidence: float | None = None
    intent_segment: workflow_contracts.IntentGraphSegment | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = ""
    trace_id: str = ""
    run_id: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tool_inputs_resolved: bool = False
    tool_inputs_validation: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


def build_task_execution_request(
    value: Mapping[str, Any] | TaskExecutionRequest | None,
    *,
    default_max_attempts: int = 1,
) -> TaskExecutionRequest:
    if isinstance(value, TaskExecutionRequest):
        return value
    payload = dict(value) if isinstance(value, Mapping) else {}
    trace_id = _string_value(payload.get("correlation_id")) or _string_value(
        payload.get("trace_id")
    )
    run_id = _string_value(payload.get("run_id")) or trace_id
    request_ids = _request_ids(payload.get("tool_requests"))
    tool_inputs = _tool_inputs(payload.get("tool_inputs"))
    bindings = _binding_index(
        normalize_capability_bindings(
            payload,
            request_ids=request_ids,
        )
    )
    execution_gates = _execution_gates(
        payload,
        request_ids=request_ids,
    )
    attempts = _attempt_count(payload.get("attempts"))
    max_attempts = max(
        attempts,
        _max_attempts(payload.get("max_attempts"), default_max_attempts),
    )
    return TaskExecutionRequest(
        task_id=_string_value(payload.get("task_id")),
        job_id=_string_value(payload.get("job_id")),
        run_id=run_id,
        trace_id=trace_id,
        name=_string_value(payload.get("name")),
        description=_string_value(payload.get("description")),
        instruction=_string_value(payload.get("instruction")),
        context=dict(payload.get("context")) if isinstance(payload.get("context"), Mapping) else {},
        attempts=attempts,
        max_attempts=max_attempts,
        intent=_normalized_intent(payload),
        intent_source=_string_value(payload.get("intent_source")) or None,
        intent_confidence=_intent_confidence(payload.get("intent_confidence")),
        intent_segment=_intent_segment(payload),
        dependency_artifacts=(
            dict(payload.get("dependency_artifacts"))
            if isinstance(payload.get("dependency_artifacts"), Mapping)
            else {}
        ),
        retry_policy=_string_value(payload.get("retry_policy")) or None,
        source_payload=payload,
        requests=[
            TaskExecutionStep(
                request_id=request_id,
                resolved_inputs=dict(tool_inputs.get(request_id) or {}),
                capability_binding=bindings.get(request_id),
                execution_gate=dict(execution_gates.get(request_id) or {}) or None,
            )
            for request_id in request_ids
        ],
    )


def build_task_dispatch_payload(
    value: Mapping[str, Any] | TaskDispatchPayload | None,
    *,
    default_max_attempts: int = 1,
) -> TaskDispatchPayload:
    if isinstance(value, TaskDispatchPayload):
        return value
    payload = dict(value) if isinstance(value, Mapping) else {}
    execution_request = build_task_execution_request(
        payload,
        default_max_attempts=default_max_attempts,
    )
    capability_bindings = normalize_capability_bindings(
        payload,
        request_ids=execution_request.tool_requests,
    )
    execution_gates = normalize_execution_gates(
        payload,
        request_ids=execution_request.tool_requests,
    )
    correlation_id = _string_value(payload.get("correlation_id")) or execution_request.trace_id
    trace_id = _string_value(payload.get("trace_id")) or correlation_id or execution_request.trace_id
    normalized = dict(payload)
    normalized.update(
        {
            "task_id": execution_request.task_id,
            "id": _string_value(payload.get("id")) or execution_request.task_id,
            "job_id": execution_request.job_id,
            "plan_id": _string_value(payload.get("plan_id")),
            "name": _string_value(payload.get("name")),
            "description": _string_value(payload.get("description")),
            "instruction": execution_request.instruction,
            "acceptance_criteria": _string_list(payload.get("acceptance_criteria")),
            "expected_output_schema_ref": _string_value(
                payload.get("expected_output_schema_ref")
            ),
            "status": _string_value(payload.get("status")),
            "deps": _string_list(payload.get("deps")),
            "attempts": execution_request.attempts,
            "max_attempts": execution_request.max_attempts,
            "rework_count": _int_or_default(payload.get("rework_count"), 0),
            "max_reworks": _int_or_default(payload.get("max_reworks"), 0),
            "assigned_to": _string_value(payload.get("assigned_to")) or None,
            "tool_requests": execution_request.tool_requests,
            "tool_inputs": execution_request.tool_inputs,
            "capability_bindings": capability_bindings,
            "execution_gates": execution_gates,
            "critic_required": _bool_value(payload.get("critic_required"), default=True),
            "intent": execution_request.intent,
            "intent_source": execution_request.intent_source,
            "intent_confidence": execution_request.intent_confidence,
            "intent_segment": execution_request.intent_segment,
            "context": execution_request.context,
            "correlation_id": correlation_id,
            "trace_id": trace_id,
            "run_id": execution_request.run_id,
            "created_at": _datetime_or_none(payload.get("created_at")),
            "updated_at": _datetime_or_none(payload.get("updated_at")),
            "tool_inputs_resolved": bool(payload.get("tool_inputs_resolved")),
            "tool_inputs_validation": _string_dict(payload.get("tool_inputs_validation")),
            "error": _string_value(payload.get("error")) or None,
        }
    )
    return TaskDispatchPayload.model_validate(normalized)


def dump_task_dispatch_payload(
    value: Mapping[str, Any] | TaskDispatchPayload | None,
    *,
    default_max_attempts: int = 1,
) -> dict[str, Any]:
    payload = build_task_dispatch_payload(
        value,
        default_max_attempts=default_max_attempts,
    )
    dumped = payload.model_dump(mode="json", exclude_none=True)
    if not dumped.get("tool_inputs_validation"):
        dumped.pop("tool_inputs_validation", None)
    if not dumped.get("tool_inputs_resolved"):
        dumped.pop("tool_inputs_resolved", None)
    if not dumped.get("capability_bindings"):
        dumped.pop("capability_bindings", None)
    if not dumped.get("execution_gates"):
        dumped.pop("execution_gates", None)
    return dumped


def normalize_capability_bindings(
    value: Any,
    *,
    request_ids: Sequence[str] | None = None,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_request_ids = _request_ids(request_ids or [])
    bindings = _binding_index(_raw_capability_bindings(value))
    for request_id in normalized_request_ids:
        if request_id in bindings:
            continue
        bindings[request_id] = _synthesized_binding(
            request_id,
            capabilities.get(request_id) if isinstance(capabilities, Mapping) else None,
        )
    if normalized_request_ids:
        ordered_ids = normalized_request_ids
    else:
        ordered_ids = list(bindings.keys())
    dumped: dict[str, dict[str, Any]] = {}
    for request_id in ordered_ids:
        binding = bindings.get(request_id)
        if binding is None:
            continue
        dumped[request_id] = binding.model_dump(mode="json", exclude_none=True)
    return dumped


def embed_capability_bindings(
    tool_inputs: Mapping[str, Any] | None,
    capability_bindings: Any,
    *,
    request_ids: Sequence[str] | None = None,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = strip_execution_metadata_from_tool_inputs(tool_inputs)
    bindings = normalize_capability_bindings(
        {"capability_bindings": capability_bindings},
        request_ids=request_ids,
        capabilities=capabilities,
    )
    if bindings:
        merged[EXECUTION_BINDINGS_KEY] = bindings
    return merged


def normalize_execution_gates(
    value: Any,
    *,
    request_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_request_ids = _request_ids(request_ids or [])
    gates = _execution_gates(value, request_ids=normalized_request_ids)
    if normalized_request_ids:
        return {
            request_id: gate
            for request_id in normalized_request_ids
            if isinstance((gate := gates.get(request_id)), Mapping)
        }
    return gates


def embed_execution_gate(
    tool_inputs: Mapping[str, Any] | None,
    execution_gate: Mapping[str, Any] | None,
    *,
    request_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    merged = strip_execution_metadata_from_tool_inputs(tool_inputs)
    if not isinstance(execution_gate, Mapping):
        return merged
    normalized_request_ids = _request_ids(request_ids or [])
    embedded = {
        request_id: dict(execution_gate)
        for request_id in normalized_request_ids
        if request_id
    }
    if embedded:
        merged[EXECUTION_GATE_KEY] = embedded
    return merged


def strip_execution_metadata_from_tool_inputs(tool_inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tool_inputs, Mapping):
        return {}
    stripped: dict[str, Any] = {}
    for key, value in tool_inputs.items():
        normalized_key = _string_value(key)
        if not normalized_key or normalized_key in {EXECUTION_BINDINGS_KEY, EXECUTION_GATE_KEY}:
            continue
        stripped[normalized_key] = dict(value) if isinstance(value, Mapping) else value
    return stripped


def _string_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    items: list[str] = []
    for item in value:
        normalized = _string_value(item)
        if normalized:
            items.append(normalized)
    return items


def _request_ids(value: Any) -> list[str]:
    return _string_list(value)


def _tool_inputs(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for request_id, payload in value.items():
        key = _string_value(request_id)
        if not key or key in {EXECUTION_BINDINGS_KEY, EXECUTION_GATE_KEY}:
            continue
        normalized[key] = dict(payload) if isinstance(payload, Mapping) else {}
    return normalized


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        dict_key = _string_value(key)
        dict_value = _string_value(item)
        if dict_key and dict_value:
            normalized[dict_key] = dict_value
    return normalized


def _attempt_count(value: Any) -> int:
    return max(1, _int_or_default(value, 1))


def _max_attempts(value: Any, default_max_attempts: int) -> int:
    fallback = max(1, default_max_attempts)
    if value is None:
        return fallback
    parsed = _int_or_default(value, fallback)
    if parsed <= 0:
        return fallback
    return max(1, parsed)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _normalized_intent(payload: Mapping[str, Any]) -> str | None:
    return intent_contract.normalize_task_intent(
        payload.get("intent")
    ) or intent_contract.normalize_task_intent(payload.get("task_intent"))


def _intent_confidence(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _datetime_or_none(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


def _intent_segment(payload: Mapping[str, Any]) -> workflow_contracts.IntentGraphSegment | None:
    segment = payload.get("intent_segment")
    if not isinstance(segment, Mapping):
        profile = payload.get("intent_profile")
        if isinstance(profile, Mapping):
            candidate = profile.get("segment")
            segment = candidate if isinstance(candidate, Mapping) else None
    if not isinstance(segment, Mapping):
        return None
    try:
        return workflow_contracts.IntentGraphSegment.model_validate(segment)
    except ValidationError:
        return None


def _binding_index(value: Any) -> dict[str, CapabilityBinding]:
    index: dict[str, CapabilityBinding] = {}
    if isinstance(value, Mapping):
        for request_id, binding_payload in value.items():
            if not isinstance(binding_payload, Mapping):
                continue
            binding = _binding_from_mapping(binding_payload, fallback_request_id=_string_value(request_id))
            if binding is not None:
                index[binding.request_id] = binding
        return index
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for binding_payload in value:
            if not isinstance(binding_payload, Mapping):
                continue
            binding = _binding_from_mapping(binding_payload)
            if binding is not None:
                index[binding.request_id] = binding
    return index


def _raw_capability_bindings(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    explicit = value.get("capability_bindings") or value.get("execution_bindings")
    if explicit is not None:
        return explicit
    tool_inputs = value.get("tool_inputs")
    if isinstance(tool_inputs, Mapping):
        embedded = tool_inputs.get(EXECUTION_BINDINGS_KEY)
        if embedded is not None:
            return embedded
    if any(
        isinstance(binding_payload, Mapping)
        and any(
            key in binding_payload
            for key in (
                "request_id",
                "capability_id",
                "tool_name",
                "adapter_type",
                "server_id",
                "type",
            )
        )
        for binding_payload in value.values()
    ):
        return value
    return None


def _execution_gates(
    value: Any,
    *,
    request_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    explicit = _raw_execution_gates(value)
    if not isinstance(explicit, Mapping):
        return {}
    normalized_request_ids = _request_ids(request_ids or [])
    gates: dict[str, dict[str, Any]] = {}
    for request_id, gate_payload in explicit.items():
        key = _string_value(request_id)
        if not key or not isinstance(gate_payload, Mapping):
            continue
        gates[key] = dict(gate_payload)
    if normalized_request_ids:
        return {request_id: gates[request_id] for request_id in normalized_request_ids if request_id in gates}
    return gates


def _raw_execution_gates(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    explicit = value.get("execution_gates") or value.get("execution_gate")
    if explicit is not None:
        return explicit
    tool_inputs = value.get("tool_inputs")
    if isinstance(tool_inputs, Mapping):
        embedded = tool_inputs.get(EXECUTION_GATE_KEY)
        if embedded is not None:
            return embedded
    return None


def _binding_from_mapping(
    value: Mapping[str, Any],
    *,
    fallback_request_id: str = "",
) -> CapabilityBinding | None:
    request_id = _string_value(value.get("request_id")) or fallback_request_id
    if not request_id:
        return None
    return CapabilityBinding(
        request_id=request_id,
        capability_id=_string_value(value.get("capability_id")) or None,
        tool_name=_string_value(value.get("tool_name")) or None,
        adapter_type=(
            _string_value(value.get("adapter_type"))
            or _string_value(value.get("type"))
            or None
        ),
        server_id=_string_value(value.get("server_id")) or None,
    )


def _synthesized_binding(request_id: str, capability: Any = None) -> CapabilityBinding:
    capability_id = _string_value(getattr(capability, "capability_id", None))
    adapters = getattr(capability, "adapters", ()) if capability is not None else ()
    adapter = None
    if isinstance(adapters, Sequence):
        for candidate in adapters:
            if getattr(candidate, "enabled", True):
                adapter = candidate
                break
    if capability_id:
        return CapabilityBinding(
            request_id=request_id,
            capability_id=capability_id,
            tool_name=_string_value(getattr(adapter, "tool_name", None)) or request_id,
            adapter_type=_string_value(getattr(adapter, "type", None)) or "capability",
            server_id=_string_value(getattr(adapter, "server_id", None)) or None,
        )
    return CapabilityBinding(
        request_id=request_id,
        tool_name=request_id,
        adapter_type="tool",
    )
