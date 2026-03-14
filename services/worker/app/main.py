from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Mapping
from datetime import datetime
from pathlib import Path

import redis
from jsonschema import Draft202012Validator
from prometheus_client import start_http_server

from libs.core import (
    capability_registry,
    document_store,
    events,
    intent_contract,
    llm_provider,
    logging as core_logging,
    mcp_gateway,
    models,
    payload_resolver,
    tool_registry,
    tracing as core_tracing,
)
from libs.core.memory_client import MemoryClient
from libs.framework.tool_runtime import classify_tool_error
from services.worker.app.memory_semantics import (
    apply_memory_defaults,
    missing_memory_only_inputs,
    select_memory_payload,
    stable_memory_keys,
)

core_logging.configure_logging("worker")
LOGGER = core_logging.get_logger("worker")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
core_tracing.configure_tracing("worker", endpoint=OTEL_EXPORTER_OTLP_ENDPOINT)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TOOL_HTTP_FETCH_ENABLED = os.getenv("TOOL_HTTP_FETCH_ENABLED", "false").lower() == "true"
WORKER_MODE = os.getenv("WORKER_MODE", "rule_based")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
OPENAI_MAX_OUTPUT_TOKENS = os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
SCHEMA_REGISTRY_PATH = os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas")
SCHEMA_VALIDATION_STRICT = os.getenv("SCHEMA_VALIDATION_STRICT", "false").lower() == "true"
OPENAI_TIMEOUT_S = os.getenv("OPENAI_TIMEOUT_S")
OPENAI_MAX_RETRIES = os.getenv("OPENAI_MAX_RETRIES")
TOOL_OUTPUT_SIZE_CAP = os.getenv("TOOL_OUTPUT_SIZE_CAP", "50000")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "unknown")
POLICY_VERSION = os.getenv("POLICY_VERSION", "unknown")
TOOL_VERSION = os.getenv("TOOL_VERSION", "unknown")
CAPABILITY_INPUT_VALIDATION_ENABLED = (
    os.getenv("CAPABILITY_INPUT_VALIDATION_ENABLED", "true").lower() == "true"
)
CAPABILITY_ENFORCE_SCHEMA_PROPERTIES = (
    os.getenv("CAPABILITY_ENFORCE_SCHEMA_PROPERTIES", "false").lower() == "true"
)

LLM_ENABLED = WORKER_MODE == "llm"

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

DEFAULT_ALLOWED_BLOCK_TYPES = [
    "text",
    "paragraph",
    "heading",
    "bullets",
    "spacer",
    "optional_paragraph",
    "repeat",
]

def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


OUTPUT_SIZE_CAP = _parse_optional_int(TOOL_OUTPUT_SIZE_CAP) or 50000
MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://api:8000")
MEMORY_READ_ENABLED = os.getenv("MEMORY_READ_ENABLED", "true").lower() == "true"
MEMORY_WRITE_ENABLED = os.getenv("MEMORY_WRITE_ENABLED", "true").lower() == "true"
SEMANTIC_MEMORY_AUTO_WRITE_ENABLED = (
    os.getenv("SEMANTIC_MEMORY_AUTO_WRITE_ENABLED", "true").lower() == "true"
)
SEMANTIC_MEMORY_AUTO_WRITE_TOOL_PATTERNS = [
    value.strip().lower()
    for value in os.getenv(
        "SEMANTIC_MEMORY_AUTO_WRITE_TOOLS",
        "llm.text.generate,document.spec.generate,document.spec.improve,"
        "document.spec.generate_iterative,llm_generate,llm_generate_document_spec,"
        "llm_improve_document_spec,llm_iterative_improve_document_spec,openapi.iterative.improve",
    ).split(",")
    if value.strip()
]
SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS = (
    _parse_optional_int(os.getenv("SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS", "3")) or 3
)
SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS = max(1, min(SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS, 8))
SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS = (
    _parse_optional_int(os.getenv("SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS", "280")) or 280
)
SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS = max(80, min(SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS, 1200))
SEMANTIC_MEMORY_AUTO_WRITE_NAMESPACE = (
    os.getenv("SEMANTIC_MEMORY_AUTO_WRITE_NAMESPACE", "runtime").strip() or "runtime"
)
MEMORY_CLIENT = MemoryClient(MEMORY_API_URL)
WORKER_GROUP = os.getenv("WORKER_GROUP", "workers")
WORKER_CONSUMER = os.getenv("WORKER_CONSUMER", str(uuid.uuid4()))
WORKER_READ_COUNT = _parse_optional_int(os.getenv("WORKER_READ_COUNT", "1")) or 1
WORKER_BLOCK_MS = _parse_optional_int(os.getenv("WORKER_BLOCK_MS", "1000")) or 1000
WORKER_RECOVER_PENDING = os.getenv("WORKER_RECOVER_PENDING", "true").lower() == "true"
WORKER_RECOVER_IDLE_MS = _parse_optional_int(os.getenv("WORKER_RECOVER_IDLE_MS", "60000")) or 60000
WORKER_RECOVER_INTERVAL_S = (
    _parse_optional_float(os.getenv("WORKER_RECOVER_INTERVAL_S", "10")) or 10.0
)
WORKER_RECOVER_COUNT = _parse_optional_int(os.getenv("WORKER_RECOVER_COUNT", "10")) or 10
WORKER_RETRY_ENABLED = os.getenv("WORKER_RETRY_ENABLED", "true").lower() == "true"
WORKER_RETRY_POLICY = os.getenv("WORKER_RETRY_POLICY", "transient").strip().lower()
WORKER_DEFAULT_MAX_ATTEMPTS = (
    _parse_optional_int(os.getenv("WORKER_DEFAULT_MAX_ATTEMPTS", "3")) or 3
)
WORKER_DLQ_ENABLED = os.getenv("WORKER_DLQ_ENABLED", "true").lower() == "true"
METRICS_PORT = _parse_optional_int(os.getenv("WORKER_METRICS_PORT", "9102")) or 9102
_SEMANTIC_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+|\n+")
_SEMANTIC_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_METRICS_SERVER_STARTED = False


LLM_PROVIDER_INSTANCE = None
if LLM_ENABLED:
    LLM_PROVIDER_INSTANCE = llm_provider.resolve_provider(
        LLM_PROVIDER,
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_BASE_URL,
        temperature=_parse_optional_float(OPENAI_TEMPERATURE),
        max_output_tokens=_parse_optional_int(OPENAI_MAX_OUTPUT_TOKENS),
        timeout_s=_parse_optional_float(OPENAI_TIMEOUT_S),
        max_retries=_parse_optional_int(OPENAI_MAX_RETRIES),
    )


def _capability_mode() -> str:
    return capability_registry.resolve_capability_mode()


def _resolve_enabled_capability_request(
    capability_id: str,
) -> capability_registry.CapabilitySpec | None:
    mode = _capability_mode()
    if mode == "disabled":
        return None
    try:
        registry = capability_registry.load_capability_registry()
    except Exception as exc:  # noqa: BLE001
        core_logging.log_event(
            LOGGER,
            "capability_registry_load_failed",
            {"capability_id": capability_id, "mode": mode, "error": str(exc)},
        )
        return None
    spec = registry.get(capability_id)
    if spec is None or not spec.enabled:
        return None
    if mode == "dry_run":
        core_logging.log_event(
            LOGGER,
            "capability_mode_dry_run",
            {"capability_id": capability_id},
        )
    return spec


def _execute_capability_request(
    *,
    capability_id: str,
    payload: dict[str, Any],
    trace_id: str,
    idempotency_key: str,
    registry: tool_registry.ToolRegistry,
    task_payload: dict[str, Any] | None = None,
) -> models.ToolCall:
    started_at = datetime.utcnow()
    task_payload = task_payload if isinstance(task_payload, dict) else {}
    task_id = task_payload.get("task_id")

    def _execute_native_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            tool = registry.get(tool_name)
        except KeyError as exc:
            raise tool_registry.ToolExecutionError(f"unknown_tool:{tool_name}") from exc
        tool_payload = dict(arguments)
        memory_payload = _load_memory_inputs(tool, task_payload, trace_id)
        if memory_payload:
            existing_memory = tool_payload.get("memory")
            merged_memory = dict(existing_memory) if isinstance(existing_memory, dict) else {}
            merged_memory.update(memory_payload)
            tool_payload["memory"] = merged_memory
        tool_payload = apply_memory_defaults(tool.spec.name, tool_payload)
        missing = missing_memory_only_inputs(tool.spec.name, tool_payload)
        if missing:
            raise tool_registry.ToolExecutionError(
                f"contract.input_missing:memory_only_inputs_missing:{','.join(missing)}"
            )
        tool_payload["_registry"] = registry
        call = registry.execute(
            tool_name,
            payload=tool_payload,
            idempotency_key=str(uuid.uuid4()),
            trace_id=trace_id,
            max_output_bytes=OUTPUT_SIZE_CAP,
        )
        if call.status != "completed":
            output = call.output_or_error if isinstance(call.output_or_error, dict) else {}
            error_text = output.get("error", "capability_native_tool_failed")
            raise tool_registry.ToolExecutionError(str(error_text))
        _persist_memory_outputs(tool, task_payload, call, trace_id)
        output = call.output_or_error
        if isinstance(output, dict):
            return output
        return {"result": output}

    try:
        result = mcp_gateway.invoke_capability(
            capability_id,
            payload,
            execute_tool=_execute_native_tool,
        )
        output = result if isinstance(result, dict) else {"result": result}
        status = "completed"
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        output = {"error": error_text, "error_code": classify_tool_error(error_text)}
        status = "failed"
    finished_at = datetime.utcnow()
    return models.ToolCall(
        tool_name=capability_id,
        input=payload,
        idempotency_key=idempotency_key,
        trace_id=trace_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        output_or_error=output,
    )


def execute_task(task_payload: dict) -> models.TaskResult:
    task_id = task_payload.get("task_id")
    trace_id = str(task_payload.get("correlation_id", ""))
    run_id = str(task_payload.get("run_id") or trace_id or uuid.uuid4())
    job_id = str(task_payload.get("job_id") or "")
    tool_requests = task_payload.get("tool_requests", [])
    context = task_payload.get("context", {})
    tool_inputs = task_payload.get("tool_inputs", {})
    registry = tool_registry.default_registry(
        TOOL_HTTP_FETCH_ENABLED,
        llm_enabled=LLM_ENABLED,
        llm_provider=LLM_PROVIDER_INSTANCE,
        service_name="worker",
    )
    tool_calls = []
    started_at = datetime.utcnow()
    outputs = {}
    tool_error = None
    task_intent_inference = _infer_task_intent_inference(task_payload)
    task_intent = task_intent_inference.intent
    task_intent_segment = _intent_segment_from_payload(task_payload)
    task_attempt, task_max_attempts = _task_attempt_limits(task_payload)
    artifacts: list[dict[str, Any]] = []
    core_logging.log_event(
        LOGGER,
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
            "model.provider": LLM_PROVIDER,
            "model.name": OPENAI_MODEL,
            "prompt.version": PROMPT_VERSION,
            "policy.version": POLICY_VERSION,
            "tool.version": TOOL_VERSION,
        },
    ) as task_span:
        for tool_index, tool_name in enumerate(tool_requests):
            with core_tracing.start_span(
                "worker.execute_tool",
                attributes={
                    "task.id": str(task_id or ""),
                    "job.id": job_id,
                    "trace.id": trace_id,
                    "run.id": run_id,
                    "tool.name": tool_name,
                    "tool.sequence": tool_index + 1,
                    "task.attempt": task_attempt,
                    "task.max_attempts": task_max_attempts,
                    "model.provider": LLM_PROVIDER,
                    "model.name": OPENAI_MODEL,
                    "prompt.version": PROMPT_VERSION,
                    "policy.version": POLICY_VERSION,
                    "tool.version": TOOL_VERSION,
                },
            ) as tool_span:
                capability_spec = _resolve_enabled_capability_request(tool_name)
                if capability_spec is not None:
                    capability_allow_decision = capability_registry.evaluate_capability_allowlist(
                        capability_spec.capability_id,
                        "worker",
                    )
                    if (
                        capability_allow_decision.violated
                        and capability_allow_decision.mode == "dry_run"
                    ):
                        core_logging.log_event(
                            LOGGER,
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
                        outputs[tool_name] = {
                            "error": tool_error,
                            "error_code": "policy.capability_not_allowed",
                        }
                        tool_calls.append(
                            models.ToolCall(
                                tool_name=tool_name,
                                input={},
                                idempotency_key=str(uuid.uuid4()),
                                trace_id=trace_id,
                                started_at=started_at,
                                finished_at=datetime.utcnow(),
                                status="failed",
                                output_or_error={
                                    "error": tool_error,
                                    "error_code": "policy.capability_not_allowed",
                                },
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
                        break
                    capability_mismatch = _capability_intent_mismatch(task_intent, capability_spec)
                    if capability_mismatch:
                        tool_error = f"contract.intent_mismatch:{capability_mismatch}"
                        outputs[tool_name] = {
                            "error": tool_error,
                            "error_code": "contract.intent_mismatch",
                        }
                        tool_calls.append(
                            models.ToolCall(
                                tool_name=tool_name,
                                input={},
                                idempotency_key=str(uuid.uuid4()),
                                trace_id=trace_id,
                                started_at=started_at,
                                finished_at=datetime.utcnow(),
                                status="failed",
                                output_or_error={
                                    "error": tool_error,
                                    "error_code": "contract.intent_mismatch",
                                },
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
                        break
                    payload = _tool_payload(
                        tool_name,
                        task_payload.get("instruction", ""),
                        context,
                        task_payload,
                        tool_inputs,
                    )
                    payload, capability_payload_error, dropped_payload_keys = (
                        _enforce_capability_input_contract(
                            capability_spec,
                            payload,
                        )
                    )
                    if dropped_payload_keys:
                        core_logging.log_event(
                            LOGGER,
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
                        tool_error = capability_payload_error
                        outputs[tool_name] = {
                            "error": tool_error,
                            "error_code": "contract.input_invalid",
                        }
                        tool_calls.append(
                            models.ToolCall(
                                tool_name=tool_name,
                                input=payload,
                                idempotency_key=str(uuid.uuid4()),
                                trace_id=trace_id,
                                started_at=started_at,
                                finished_at=datetime.utcnow(),
                                status="failed",
                                output_or_error={
                                    "error": tool_error,
                                    "error_code": "contract.input_invalid",
                                },
                            )
                        )
                        core_tracing.set_span_attributes(
                            tool_span,
                            {
                                "tool.error": tool_error,
                                "tool.error_code": "contract.input_invalid",
                                "capability.id": capability_spec.capability_id,
                            },
                        )
                        break
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
                        outputs[tool_name] = {
                            "error": tool_error,
                            "error_code": "contract.intent_mismatch",
                        }
                        tool_calls.append(
                            models.ToolCall(
                                tool_name=tool_name,
                                input=payload,
                                idempotency_key=str(uuid.uuid4()),
                                trace_id=trace_id,
                                started_at=started_at,
                                finished_at=datetime.utcnow(),
                                status="failed",
                                output_or_error={
                                    "error": tool_error,
                                    "error_code": "contract.intent_mismatch",
                                },
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
                        break
                    idempotency_key = str(uuid.uuid4())
                    tool_started_at = time.monotonic()
                    core_logging.log_event(
                        LOGGER,
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
                            if capability_spec.adapters
                            and capability_spec.adapters[0].timeout_s is not None
                            else None,
                            "trace_id": trace_id,
                            "idempotency_key": idempotency_key,
                            "model_provider": LLM_PROVIDER,
                            "model_name": OPENAI_MODEL,
                            "prompt_version": PROMPT_VERSION,
                            "policy_version": POLICY_VERSION,
                            "tool_version": TOOL_VERSION,
                            "payload_keys": sorted(payload.keys()),
                            "capability_id": capability_spec.capability_id,
                            "task_intent": task_intent,
                            "intent_source": task_intent_inference.source,
                            "intent_confidence": round(
                                float(task_intent_inference.confidence), 3
                            ),
                        },
                    )
                    call = _execute_capability_request(
                        capability_id=capability_spec.capability_id,
                        payload=payload,
                        trace_id=trace_id,
                        idempotency_key=idempotency_key,
                        registry=registry,
                        task_payload=task_payload,
                    )
                    core_logging.log_event(
                        LOGGER,
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
                            "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
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
                            "tool.duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                            "tool.is_capability": True,
                            "capability.id": capability_spec.capability_id,
                        },
                    )
                    tool_calls.append(call)
                    outputs[tool_name] = call.output_or_error
                    if call.status == "completed":
                        _sync_output_artifact(call.output_or_error, task_id, tool_name, trace_id)
                        if isinstance(call.output_or_error, Mapping):
                            _auto_persist_semantic_facts(
                                tool_name=tool_name,
                                task_payload=task_payload,
                                output=call.output_or_error,
                                trace_id=trace_id,
                                run_id=run_id,
                            )
                        continue
                    tool_error = call.output_or_error.get("error", "capability_failed")
                    if isinstance(tool_error, str):
                        lowered = tool_error.lower()
                        if (
                            "timed out" in lowered
                            or "timeout" in lowered
                            or "mcp_sdk_timeout:" in lowered
                        ):
                            if not tool_error.startswith("tool_call_timed_out"):
                                tool_error = f"tool_call_timed_out:{tool_error}"
                                call.output_or_error["error"] = tool_error
                                outputs[tool_name] = call.output_or_error
                    if isinstance(tool_error, str):
                        core_logging.log_event(
                            LOGGER,
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
                                "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                                "capability_id": capability_spec.capability_id,
                            },
                        )
                    break
                try:
                    tool = registry.get(tool_name)
                except KeyError:
                    tool_error = f"contract.tool_not_found:unknown_tool:{tool_name}"
                    outputs[tool_name] = {"error": tool_error}
                    core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
                    break
                governance_context = {
                    "job_id": job_id,
                    "tenant_id": task_payload.get("tenant_id"),
                    "org_id": task_payload.get("org_id"),
                    "job_type": task_payload.get("job_type"),
                    "context": context,
                    "job_context": context.get("job_context")
                    if isinstance(context, dict)
                    else None,
                }
                allow_decision = tool_registry.evaluate_tool_allowlist(
                    tool_name,
                    "worker",
                    context=governance_context,
                    tool_spec=tool.spec,
                )
                if allow_decision.violated and allow_decision.mode == "dry_run":
                    core_logging.log_event(
                        LOGGER,
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
                    outputs[tool_name] = {"error": tool_error, "error_code": "policy.tool_not_allowed"}
                    tool_calls.append(
                        models.ToolCall(
                            tool_name=tool_name,
                            input={},
                            idempotency_key=str(uuid.uuid4()),
                            trace_id=trace_id,
                            started_at=started_at,
                            finished_at=datetime.utcnow(),
                            status="failed",
                            output_or_error={
                                "error": tool_error,
                                "error_code": "policy.tool_not_allowed",
                            },
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
                    break
                mismatch = _intent_mismatch(task_intent, tool.spec.tool_intent, tool_name)
                if mismatch:
                    tool_error = f"contract.intent_mismatch:{mismatch}"
                    outputs[tool_name] = {"error": tool_error}
                    tool_calls.append(
                        models.ToolCall(
                            tool_name=tool_name,
                            input={},
                            idempotency_key=str(uuid.uuid4()),
                            trace_id=trace_id,
                            started_at=started_at,
                            finished_at=datetime.utcnow(),
                            status="failed",
                            output_or_error={"error": tool_error},
                        )
                    )
                    core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
                    break
                payload = _tool_payload(
                    tool_name,
                    task_payload.get("instruction", ""),
                    context,
                    task_payload,
                    tool_inputs,
                )
                memory_payload = _load_memory_inputs(tool, task_payload, trace_id)
                if memory_payload:
                    payload.setdefault("memory", {}).update(memory_payload)
                    payload = apply_memory_defaults(tool.spec.name, payload)
                    missing = missing_memory_only_inputs(tool.spec.name, payload)
                    if missing:
                        tool_error = (
                            f"contract.input_missing:memory_only_inputs_missing:{','.join(missing)}"
                        )
                        outputs[tool_name] = {"error": tool_error}
                        tool_calls.append(
                            models.ToolCall(
                                tool_name=tool_name,
                                input=payload,
                                idempotency_key=str(uuid.uuid4()),
                                trace_id=trace_id,
                                started_at=started_at,
                                finished_at=datetime.utcnow(),
                                status="failed",
                                output_or_error={"error": tool_error},
                            )
                        )
                        core_tracing.set_span_attributes(tool_span, {"tool.error": tool_error})
                        break
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
                    outputs[tool_name] = {
                        "error": tool_error,
                        "error_code": "contract.intent_mismatch",
                    }
                    tool_calls.append(
                        models.ToolCall(
                            tool_name=tool_name,
                            input=payload,
                            idempotency_key=str(uuid.uuid4()),
                            trace_id=trace_id,
                            started_at=started_at,
                            finished_at=datetime.utcnow(),
                            status="failed",
                            output_or_error={
                                "error": tool_error,
                                "error_code": "contract.intent_mismatch",
                            },
                        )
                    )
                    core_tracing.set_span_attributes(
                        tool_span,
                        {
                            "tool.error": tool_error,
                            "tool.error_code": "contract.intent_mismatch",
                        },
                    )
                    break
                payload["_registry"] = registry
                idempotency_key = str(uuid.uuid4())
                tool_started_at = time.monotonic()
                core_logging.log_event(
                    LOGGER,
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
                        "model_provider": LLM_PROVIDER,
                        "model_name": OPENAI_MODEL,
                        "prompt_version": PROMPT_VERSION,
                        "policy_version": POLICY_VERSION,
                        "tool_version": TOOL_VERSION,
                        "payload_keys": sorted(payload.keys()),
                        "task_intent": task_intent,
                        "intent_source": task_intent_inference.source,
                        "intent_confidence": round(float(task_intent_inference.confidence), 3),
                    },
                )
                call = registry.execute(
                    tool_name,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    trace_id=trace_id,
                    max_output_bytes=OUTPUT_SIZE_CAP,
                )
                core_logging.log_event(
                    LOGGER,
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
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
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
                        "tool.duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                    },
                )
                tool_calls.append(call)
                outputs[tool_name] = call.output_or_error
                if call.status == "completed":
                    _sync_output_artifact(call.output_or_error, task_id, tool_name, trace_id)
                    _persist_memory_outputs(tool, task_payload, call, trace_id)
                    if isinstance(call.output_or_error, Mapping):
                        _auto_persist_semantic_facts(
                            tool_name=tool_name,
                            task_payload=task_payload,
                            output=call.output_or_error,
                            trace_id=trace_id,
                            run_id=run_id,
                        )
                if call.status != "completed":
                    tool_error = call.output_or_error.get("error", "tool_failed")
                    error_code = call.output_or_error.get("error_code")
                    if isinstance(error_code, str) and error_code:
                        if isinstance(tool_error, str):
                            if not tool_error.startswith(f"{error_code}:"):
                                tool_error = f"{error_code}:{tool_error}"
                                call.output_or_error["error"] = tool_error
                                outputs[tool_name] = call.output_or_error
                    if isinstance(tool_error, str):
                        lowered = tool_error.lower()
                        if (
                            "timed out" in lowered
                            or "timeout" in lowered
                            or "mcp_sdk_timeout:" in lowered
                        ):
                            if not tool_error.startswith("tool_call_timed_out"):
                                tool_error = f"tool_call_timed_out:{tool_error}"
                                call.output_or_error["error"] = tool_error
                                outputs[tool_name] = call.output_or_error
                    if isinstance(tool_error, str):
                        core_logging.log_event(
                            LOGGER,
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
                                "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                            },
                        )
                    break
        finished_at = datetime.utcnow()
        validation_error = _validate_expected_output(task_payload, outputs)
        status = (
            models.TaskStatus.failed
            if tool_error or validation_error
            else models.TaskStatus.completed
        )
        run_scorecard = _build_task_run_scorecard(
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
        core_logging.log_event(LOGGER, "task_run_scorecard", run_scorecard)
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


def _build_task_run_scorecard(
    *,
    run_id: str,
    trace_id: str,
    job_id: str,
    task_id: str,
    status: models.TaskStatus,
    started_at: datetime,
    finished_at: datetime,
    tool_calls: list[models.ToolCall],
    outputs: dict[str, Any],
    task_attempt: int,
    task_max_attempts: int,
    failure_error: str | None,
) -> dict[str, Any]:
    total_latency_ms = int(max(0.0, (finished_at - started_at).total_seconds()) * 1000)
    tool_failures = [call for call in tool_calls if call.status != "completed"]
    scorecard: dict[str, Any] = {
        "run_id": run_id,
        "trace_id": trace_id,
        "job_id": job_id,
        "task_id": task_id,
        "success": status == models.TaskStatus.completed,
        "task_status": status.value,
        "failure_stage": _failure_stage_for_error(failure_error),
        "total_latency_ms": total_latency_ms,
        "tool_calls_count": len(tool_calls),
        "tool_failures_count": len(tool_failures),
        "task_attempt": task_attempt,
        "task_max_attempts": task_max_attempts,
        "model_provider": LLM_PROVIDER,
        "model_name": OPENAI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "policy_version": POLICY_VERSION,
        "tool_version": TOOL_VERSION,
        "policy_hits": _count_policy_hits(tool_calls, outputs),
    }
    token_metrics = _aggregate_token_metrics(tool_calls)
    scorecard.update(token_metrics)
    return scorecard


def _failure_stage_for_error(error: str | None) -> str:
    if not isinstance(error, str) or not error:
        return ""
    lowered = error.lower()
    if lowered.startswith("contract."):
        return "contract"
    if "validation" in lowered or "schema_" in lowered or "invalid_json" in lowered:
        return "validation"
    if "timeout" in lowered:
        return "timeout"
    if "mcp_" in lowered:
        return "mcp"
    if lowered.startswith("policy.") or lowered.startswith("guardrail_blocked"):
        return "guardrail"
    if lowered.startswith("tool_") or lowered.startswith("runtime."):
        return "tool_call"
    return "task_execution"


def _count_policy_hits(tool_calls: list[models.ToolCall], outputs: dict[str, Any]) -> int:
    hits = 0
    for call in tool_calls:
        output = call.output_or_error if isinstance(call.output_or_error, dict) else {}
        error_text = output.get("error")
        error_code = output.get("error_code")
        if isinstance(error_code, str) and error_code.startswith("policy."):
            hits += 1
            continue
        if isinstance(error_text, str) and "guardrail" in error_text.lower():
            hits += 1
    tool_error = outputs.get("tool_error")
    if isinstance(tool_error, dict):
        error_text = tool_error.get("error")
        if isinstance(error_text, str) and "guardrail" in error_text.lower():
            hits += 1
    return hits


def _aggregate_token_metrics(tool_calls: list[models.ToolCall]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    cost_usd_est = 0.0
    for call in tool_calls:
        output = call.output_or_error if isinstance(call.output_or_error, dict) else {}
        usage = output.get("usage")
        usage_dict = usage if isinstance(usage, dict) else output
        prompt_tokens += _int_metric(usage_dict.get("prompt_tokens"))
        completion_tokens += _int_metric(usage_dict.get("completion_tokens"))
        total_tokens += _int_metric(usage_dict.get("total_tokens"))
        cost_usd_est += _float_metric(usage_dict.get("cost_usd"))
    return {
        "tokens_in": prompt_tokens,
        "tokens_out": completion_tokens,
        "tokens_total": total_tokens,
        "cost_usd_est": round(cost_usd_est, 6),
    }


def _int_metric(value: Any) -> int:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value)))
        except ValueError:
            return 0
    return 0


def _float_metric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _infer_task_intent(task_payload: dict) -> str:
    return _infer_task_intent_inference(task_payload).intent


def _intent_segment_from_payload(task_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    segment = task_payload.get("intent_segment")
    if isinstance(segment, Mapping):
        return segment
    profile = task_payload.get("intent_profile")
    if isinstance(profile, Mapping):
        maybe_segment = profile.get("segment")
        if isinstance(maybe_segment, Mapping):
            return maybe_segment
    return None


def _infer_task_intent_inference(task_payload: dict) -> intent_contract.TaskIntentInference:
    explicit_intent = intent_contract.normalize_task_intent(
        task_payload.get("intent")
    ) or intent_contract.normalize_task_intent(task_payload.get("task_intent"))
    source = task_payload.get("intent_source")
    confidence = task_payload.get("intent_confidence")
    if (
        explicit_intent
        and isinstance(source, str)
        and source.strip()
        and isinstance(confidence, (int, float))
    ):
        return intent_contract.TaskIntentInference(
            intent=explicit_intent,
            source=source.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
        )
    return intent_contract.infer_task_intent_for_payload_with_metadata(task_payload)


def _intent_mismatch(
    task_intent: str, tool_intent: models.ToolIntent, tool_name: str
) -> str | None:
    return intent_contract.validate_tool_intent_compatibility(
        task_intent,
        tool_intent,
        tool_name,
    )


def _capability_intent_mismatch(
    task_intent: str,
    capability_spec: capability_registry.CapabilitySpec,
) -> str | None:
    hints = (
        capability_spec.planner_hints
        if isinstance(capability_spec.planner_hints, dict)
        else {}
    )
    raw_allowed = hints.get("task_intents")
    if not isinstance(raw_allowed, list) or not raw_allowed:
        return None
    allowed = {
        normalized
        for item in raw_allowed
        for normalized in [intent_contract.normalize_task_intent(item)]
        if normalized
    }
    if not allowed:
        return None
    if task_intent in allowed:
        return None
    return (
        f"task_intent_mismatch:{capability_spec.capability_id}:{task_intent}:"
        f"allowed={','.join(sorted(allowed))}"
    )


def _load_memory_inputs(tool, task_payload: dict, trace_id: str) -> dict:
    if not MEMORY_READ_ENABLED:
        return {}
    memory_reads = getattr(tool.spec, "memory_reads", None) or []
    if not memory_reads:
        return {}
    job_id = task_payload.get("job_id")
    if not job_id:
        return {}
    memory_payload: dict[str, list] = {}
    for name in memory_reads:
        entries = MEMORY_CLIENT.read(name=name, job_id=job_id)
        enriched_entries: list[dict] = []
        for entry in entries:
            payload = entry.get("payload", {})
            if not isinstance(payload, dict):
                continue
            enriched = dict(payload)
            if entry.get("key"):
                enriched["_memory_key"] = entry.get("key")
            if entry.get("updated_at"):
                enriched["_memory_updated_at"] = entry.get("updated_at")
            enriched_entries.append(enriched)
        memory_payload[name] = enriched_entries
        core_logging.log_event(
            LOGGER,
            "memory_read",
            {
                "task_id": task_payload.get("task_id"),
                "tool_name": tool.spec.name,
                "memory_name": name,
                "count": len(entries),
                "trace_id": trace_id,
            },
        )
    return memory_payload


def _persist_memory_outputs(tool, task_payload: dict, call: models.ToolCall, trace_id: str) -> None:
    if not MEMORY_WRITE_ENABLED:
        return
    memory_writes = getattr(tool.spec, "memory_writes", None) or []
    if not memory_writes:
        return
    job_id = task_payload.get("job_id")
    if not job_id:
        return
    task_id = task_payload.get("task_id") or task_payload.get("id")
    for name in memory_writes:
        selected_payload = select_memory_payload(tool.spec.name, call.output_or_error)
        if not selected_payload:
            continue
        entry = {
            "name": name,
            "job_id": job_id,
            "key": f"{tool.spec.name}:{task_id}" if task_id else tool.spec.name,
            "payload": {"source_tool": tool.spec.name, **selected_payload},
            "metadata": {"trace_id": trace_id},
        }
        written = MEMORY_CLIENT.write(entry)
        stable_keys = stable_memory_keys(tool.spec.name, selected_payload)
        if stable_keys and name == "task_outputs":
            for stable_key in stable_keys:
                stable_entry = {
                    "name": name,
                    "job_id": job_id,
                    "key": stable_key,
                    "payload": {"source_tool": tool.spec.name, **selected_payload},
                    "metadata": {"trace_id": trace_id, "alias": True, "alias_key": stable_key},
                }
                MEMORY_CLIENT.write(stable_entry)
        core_logging.log_event(
            LOGGER,
            "memory_write",
            {
                "task_id": task_id,
                "tool_name": tool.spec.name,
                "memory_name": name,
                "status": "ok" if written else "failed",
                "trace_id": trace_id,
            },
        )


def _sync_output_artifact(
    output: Mapping[str, Any], task_id: str | None, tool_name: str, trace_id: str
) -> None:
    if not isinstance(output, Mapping):
        return
    path_value = output.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return
    try:
        key = document_store.upload_artifact(path_value)
    except Exception as exc:  # noqa: BLE001
        core_logging.log_event(
            LOGGER,
            "artifact_sync_failed",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "path": path_value,
                "error": str(exc),
                "trace_id": trace_id,
            },
        )
        return
    if key:
        core_logging.log_event(
            LOGGER,
            "artifact_synced",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "path": path_value,
                "object_key": key,
                "trace_id": trace_id,
            },
        )


def _semantic_should_capture(tool_name: str) -> bool:
    if not SEMANTIC_MEMORY_AUTO_WRITE_ENABLED:
        return False
    normalized = (tool_name or "").strip().lower()
    if not normalized:
        return False
    patterns = SEMANTIC_MEMORY_AUTO_WRITE_TOOL_PATTERNS
    if not patterns:
        return True
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _semantic_tokens(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return set(_SEMANTIC_TOKEN_RE.findall(text.lower()))


def _semantic_split_facts(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    candidates: list[str] = []
    for part in _SEMANTIC_SENTENCE_SPLIT_RE.split(text):
        sentence = " ".join(part.strip().split())
        if len(sentence) < 30:
            continue
        if sentence.startswith("{") or sentence.startswith("["):
            continue
        if sentence.count("/") > 3:
            # likely file path / URL blob, not a semantic fact sentence
            continue
        if len(sentence) > SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS:
            sentence = sentence[:SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS].strip()
        candidates.append(sentence)
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
        if len(deduped) >= SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS:
            break
    return deduped


def _semantic_collect_candidate_texts(output: Mapping[str, Any]) -> list[str]:
    if not isinstance(output, Mapping):
        return []
    preferred_keys = [
        "fact",
        "facts",
        "summary",
        "alignment_summary",
        "text",
        "content",
        "objective",
        "analysis",
        "reasoning",
    ]
    texts: list[str] = []
    for key in preferred_keys:
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
    if texts:
        return texts
    # Fallback: shallow scan for string leaves in unknown output shapes.
    for value in output.values():
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if len(texts) >= 6:
                break
        elif isinstance(value, Mapping):
            for nested in value.values():
                if isinstance(nested, str) and nested.strip():
                    texts.append(nested.strip())
                    if len(texts) >= 6:
                        break
            if len(texts) >= 6:
                break
    return texts


def _semantic_confidence_from_output(output: Mapping[str, Any]) -> float:
    if not isinstance(output, Mapping):
        return 0.75
    confidence = output.get("confidence")
    if isinstance(confidence, (int, float)):
        return min(1.0, max(0.0, float(confidence)))
    alignment_score = output.get("alignment_score")
    if isinstance(alignment_score, (int, float)):
        score = float(alignment_score)
        if score > 1.0:
            score = score / 100.0
        return min(1.0, max(0.0, score))
    return 0.75


def _auto_persist_semantic_facts(
    *,
    tool_name: str,
    task_payload: Mapping[str, Any],
    output: Mapping[str, Any],
    trace_id: str,
    run_id: str,
) -> None:
    if not _semantic_should_capture(tool_name):
        return
    job_id = str(task_payload.get("job_id") or "").strip()
    user_id = str(task_payload.get("user_id") or "").strip() or None
    task_id = str(task_payload.get("task_id") or "").strip()
    task_name = str(task_payload.get("name") or tool_name).strip() or tool_name
    subject = task_name
    namespace = str(task_payload.get("intent") or SEMANTIC_MEMORY_AUTO_WRITE_NAMESPACE).strip()
    if not namespace:
        namespace = SEMANTIC_MEMORY_AUTO_WRITE_NAMESPACE
    confidence = _semantic_confidence_from_output(output)
    candidate_texts = _semantic_collect_candidate_texts(output)
    facts: list[str] = []
    for text in candidate_texts:
        facts.extend(_semantic_split_facts(text))
        if len(facts) >= SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS:
            break
    deduped_facts: list[str] = []
    seen_fact_tokens: set[str] = set()
    for fact in facts:
        token_fingerprint = " ".join(sorted(_semantic_tokens(fact))[:20])
        if token_fingerprint in seen_fact_tokens:
            continue
        seen_fact_tokens.add(token_fingerprint)
        deduped_facts.append(fact)
        if len(deduped_facts) >= SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS:
            break
    if not deduped_facts:
        return
    for fact in deduped_facts:
        digest = hashlib.sha1(f"{namespace}|{subject}|{fact}".encode("utf-8")).hexdigest()[:12]
        key = (
            f"{re.sub(r'[^a-z0-9]+', '_', namespace.lower()).strip('_') or 'runtime'}:"
            f"{re.sub(r'[^a-z0-9]+', '_', subject.lower()).strip('_') or 'task'}:{digest}"
        )
        write_payload: dict[str, Any] = {
            "fact": fact,
            "subject": subject,
            "namespace": namespace,
            "key": key,
            "confidence": confidence,
            "source": "worker_auto",
            "source_ref": f"tool:{tool_name}",
            "metadata": {
                "trace_id": trace_id,
                "run_id": run_id,
                "job_id": job_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "auto_distilled": True,
            },
        }
        if user_id:
            write_payload["user_id"] = user_id
        if job_id:
            write_payload["job_id"] = job_id
        written = MEMORY_CLIENT.semantic_write(write_payload)
        core_logging.log_event(
            LOGGER,
            "semantic_memory_auto_write",
            {
                "task_id": task_id,
                "tool_name": tool_name,
                "job_id": job_id,
                "key": key,
                "status": "ok" if isinstance(written, dict) else "failed",
                "trace_id": trace_id,
            },
        )


def _tool_payload(
    tool_name: str,
    instruction: str,
    context: dict,
    task_payload: dict,
    tool_inputs: dict,
) -> dict:
    return payload_resolver.resolve_tool_payload(
        tool_name,
        instruction,
        context if isinstance(context, dict) else {},
        task_payload if isinstance(task_payload, dict) else {},
        tool_inputs if isinstance(tool_inputs, dict) else {},
    )


def _merge_payload_from_task(payload: dict, task_payload: dict, instruction: str) -> dict:
    merged = dict(payload)
    if not isinstance(task_payload, dict):
        task_payload = {}
    for key in (
        "data",
        "schema_ref",
        "template_id",
        "template_path",
        "output_path",
        "path",
        "content",
        "render_context",
    ):
        if key not in merged and key in task_payload:
            merged[key] = task_payload.get(key)
    if "schema_ref" not in merged:
        value = _extract_schema_ref(instruction)
        if value:
            merged["schema_ref"] = value
    if "template_id" not in merged:
        value = _extract_template_id(instruction)
        if value:
            merged["template_id"] = value
    if "template_path" not in merged:
        value = _extract_template_path(instruction)
        if value:
            merged["template_path"] = value
    if "output_path" not in merged:
        value = _extract_output_path(instruction)
        if value:
            merged["output_path"] = value
    if "path" not in merged and isinstance(task_payload.get("path"), str):
        merged["path"] = task_payload.get("path")
    if "content" not in merged and isinstance(task_payload.get("content"), str):
        merged["content"] = task_payload.get("content")
    return merged


def _fill_payload_from_context(payload: dict, context: dict) -> dict:
    filled = dict(payload)
    if "document_spec" not in filled:
        doc = _extract_document_spec_from_context(context)
        if isinstance(doc, dict):
            filled["document_spec"] = doc
    if "data" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["data"] = doc
    if "errors" not in filled:
        errors = _extract_validation_errors_from_context(context)
        if isinstance(errors, list):
            filled["errors"] = errors
    if "validation_report" not in filled:
        report = _extract_validation_report_from_context(context)
        if isinstance(report, dict):
            filled["validation_report"] = report
    if "original_spec" not in filled:
        doc = _extract_json_from_context(context)
        if isinstance(doc, dict):
            filled["original_spec"] = doc
    if "content" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["content"] = text
    if "text" not in filled:
        text = _extract_text_from_context(context)
        if isinstance(text, str):
            filled["text"] = text
    return filled


def _extract_json_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            json_candidate = _extract_json_from_outputs(output)
            if isinstance(json_candidate, dict):
                return json_candidate
    return None


def _extract_document_spec_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            doc = _extract_document_spec_from_outputs(output)
            if isinstance(doc, dict):
                return doc
    return None


def _extract_validation_errors_from_context(context: dict) -> list[dict] | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                errors = entry.get("errors")
                if isinstance(errors, list):
                    return errors
    return None


def _extract_validation_report_from_context(context: dict) -> dict | None:
    if not isinstance(context, dict):
        return None
    groups = [context.get("dependencies_by_name"), context.get("dependencies")]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            entry = output.get("document_spec_validate")
            if isinstance(entry, dict):
                return entry
    return None


def _extract_json_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    spec_output = outputs.get("llm_generate_document_spec")
    if isinstance(spec_output, dict):
        document_spec = spec_output.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            json_text = _extract_json(text)
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    return None
    llm_repair = outputs.get("llm_repair_document_spec")
    if not isinstance(llm_repair, dict):
        llm_repair = outputs.get("llm_repair_json")
    if isinstance(llm_repair, dict):
        document_spec = llm_repair.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    llm_improve = outputs.get("llm_improve_document_spec")
    if isinstance(llm_improve, dict):
        document_spec = llm_improve.get("document_spec")
        if isinstance(document_spec, dict):
            return document_spec
    json_transform = outputs.get("json_transform")
    if isinstance(json_transform, dict):
        result = json_transform.get("result")
        if isinstance(result, dict):
            return result
    payload = outputs.get("result")
    if isinstance(payload, dict):
        return payload
    return None


def _extract_document_spec_from_outputs(outputs: dict) -> dict | None:
    if not isinstance(outputs, dict):
        return None
    for key in (
        "llm_generate_document_spec",
        "llm_repair_document_spec",
        "llm_repair_json",
        "llm_improve_document_spec",
    ):
        candidate = outputs.get(key)
        if isinstance(candidate, dict):
            document_spec = candidate.get("document_spec")
            if isinstance(document_spec, dict):
                return document_spec
    direct = outputs.get("document_spec")
    if isinstance(direct, dict):
        return direct
    return None


def _extract_template_id(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("template_id")
            if isinstance(value, str) and value:
                return value
            output = data.get("output")
            if isinstance(output, dict):
                value = output.get("template_id")
                if isinstance(value, str) and value:
                    return value
    value = _extract_key_from_text(instruction, "template_id")
    if value:
        return value
    return None


def _extract_template_path(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("template_path")
            if isinstance(value, str) and value:
                return value
    value = _extract_key_from_text(instruction, "template_path")
    if value:
        return value
    return None


def _extract_output_path(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("output_path")
            if isinstance(value, str) and value:
                return value
    value = _extract_key_from_text(instruction, "output_path")
    if value:
        return value
    return None


def _extract_schema_ref(instruction: str) -> str | None:
    if not instruction:
        return None
    json_text = _extract_json(instruction)
    if json_text:
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            value = data.get("schema_ref")
            if isinstance(value, str) and value:
                return value
            output = data.get("output")
            if isinstance(output, dict):
                value = output.get("schema_ref")
                if isinstance(value, str) and value:
                    return value
    value = _extract_key_from_text(instruction, "schema_ref")
    if value:
        return value
    return None


def _extract_key_from_text(text: str, key: str) -> str | None:
    import re

    patterns = [
        rf"{key}\\s*[:=]\\s*\"([^\"]+)\"",
        rf"{key}\\s*[:=]\\s*'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def _extract_template_id_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("template_id")
    return value if isinstance(value, str) and value else None


def _extract_template_path_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("template_path")
    return value if isinstance(value, str) and value else None


def _extract_output_path_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("output_path")
    return value if isinstance(value, str) and value else None


def _extract_schema_ref_from_task(task_payload: dict) -> str | None:
    value = task_payload.get("schema_ref")
    return value if isinstance(value, str) and value else None


def _extract_text_from_context(context: dict) -> str | None:
    if not isinstance(context, dict):
        return None
    for key in ("dependencies_by_name", "dependencies"):
        group = context.get(key)
        if not isinstance(group, dict):
            continue
        for output in group.values():
            if not isinstance(output, dict):
                continue
            text = _extract_text_from_outputs(output)
            if text:
                return text
    return None


def _extract_text_from_outputs(outputs: dict) -> str | None:
    if not isinstance(outputs, dict):
        return None
    for tool_key in ("llm_generate", "text_summarize", "json_transform"):
        entry = outputs.get(tool_key)
        if not isinstance(entry, dict):
            continue
        if tool_key == "llm_generate":
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
        if tool_key == "text_summarize":
            summary = entry.get("summary")
            if isinstance(summary, str) and summary:
                return summary
        if tool_key == "json_transform":
            result = entry.get("result")
            if isinstance(result, dict):
                text = result.get("text")
                if isinstance(text, str) and text:
                    return text
            elif isinstance(result, str) and result:
                return result
    for entry in outputs.values():
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
    return None


def _load_schema_from_ref(schema_ref: str) -> dict[str, Any] | None:
    schema_path = _resolve_schema_path(schema_ref)
    if schema_path is None or not schema_path.exists():
        return None
    try:
        parsed = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _prune_capability_payload_by_schema(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not CAPABILITY_ENFORCE_SCHEMA_PROPERTIES:
        return dict(payload), []
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return dict(payload), []
    allowed_keys = set(properties.keys())
    pruned = {key: value for key, value in payload.items() if key in allowed_keys}
    dropped = sorted(key for key in payload.keys() if key not in allowed_keys)
    return pruned, dropped


def _enforce_capability_input_contract(
    capability: capability_registry.CapabilitySpec,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str | None, list[str]]:
    normalized_payload = dict(payload) if isinstance(payload, dict) else {}
    if not CAPABILITY_INPUT_VALIDATION_ENABLED or not capability.input_schema_ref:
        return normalized_payload, None, []
    schema = _load_schema_from_ref(capability.input_schema_ref)
    if schema is None:
        return (
            normalized_payload,
            f"contract.input_schema_not_found:{capability.input_schema_ref}",
            [],
        )
    normalized_payload, dropped = _prune_capability_payload_by_schema(normalized_payload, schema)
    validation_errors = payload_resolver.validate_tool_inputs(
        {capability.capability_id: normalized_payload},
        {capability.capability_id: schema},
    )
    error = validation_errors.get(capability.capability_id)
    if error:
        return normalized_payload, f"contract.input_invalid:{error}", dropped
    return normalized_payload, None, dropped


def _validate_expected_output(task_payload: dict, outputs: dict) -> str | None:
    schema_ref = task_payload.get("expected_output_schema_ref")
    tool_requests = task_payload.get("tool_requests", [])
    if "llm_generate" not in tool_requests:
        return None
    if not schema_ref:
        return None
    schema_path = _resolve_schema_path(schema_ref)
    if not schema_path:
        return None
    if not schema_path.exists():
        if SCHEMA_VALIDATION_STRICT:
            return f"schema_not_found:{schema_path}"
        return None
    text = _extract_llm_text(outputs)
    if not text:
        return "missing_llm_output"
    json_text = _extract_json(text)
    if not json_text:
        return "missing_json_payload"
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return f"invalid_json:{exc}"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"invalid_schema:{exc}"
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.path)
    if errors:
        messages = "; ".join(
            f"{'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors[:5]
        )
        return f"schema_validation_failed:{messages}"
    return None


def _resolve_schema_path(schema_ref: str) -> Path | None:
    if not schema_ref:
        return None
    name = schema_ref
    if schema_ref.startswith("schema/"):
        name = schema_ref.split("/", 1)[1]
    if not name:
        return None
    if not name.endswith(".json"):
        name = f"{name}.json"
    registry_path = Path(SCHEMA_REGISTRY_PATH) / name
    if registry_path.exists():
        return registry_path
    template_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    return template_dir / name


def _extract_llm_text(outputs: dict) -> str:
    llm_output = outputs.get("llm_generate")
    if isinstance(llm_output, dict):
        text = llm_output.get("text")
        if isinstance(text, str):
            return text
    return ""


def _extract_json(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) > 1:
            content = parts[1]
        content = content.lstrip()
        if content.startswith("json"):
            content = content[4:].lstrip()
    first_obj = content.find("{")
    first_arr = content.find("[")
    if first_obj == -1 and first_arr == -1:
        return ""
    if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
        start = first_obj
        end = content.rfind("}")
    else:
        start = first_arr
        end = content.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return ""
    return content[start : end + 1]


def _extract_instruction_payload(instruction: str) -> dict:
    if not instruction:
        return {}
    json_text = _extract_json(instruction)
    if not json_text:
        return {}
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _task_attempt_limits(task_payload: Mapping[str, Any]) -> tuple[int, int]:
    attempts_raw = task_payload.get("attempts")
    max_attempts_raw = task_payload.get("max_attempts")
    try:
        attempts = int(attempts_raw) if attempts_raw is not None else 1
    except (TypeError, ValueError):
        attempts = 1
    attempts = max(attempts, 1)
    if max_attempts_raw is None:
        max_attempts = WORKER_DEFAULT_MAX_ATTEMPTS
    else:
        try:
            max_attempts = int(max_attempts_raw)
        except (TypeError, ValueError):
            max_attempts = WORKER_DEFAULT_MAX_ATTEMPTS
        if max_attempts <= 0:
            max_attempts = WORKER_DEFAULT_MAX_ATTEMPTS
    max_attempts = max(max_attempts, attempts)
    return attempts, max_attempts


def _is_transient_error(error: str | None) -> bool:
    if not error:
        return False
    value = error.lower()
    transient_tokens = (
        "timed out",
        "timeout",
        "connection refused",
        "remote end closed",
        "session terminated",
        "temporary",
        "service unavailable",
        "too many requests",
        "rate limit",
        "502",
        "503",
        "504",
    )
    return any(token in value for token in transient_tokens)


def _is_contract_error(error: str | None) -> bool:
    if not error:
        return False
    value = error.strip().lower()
    head = value.split(":", 1)[0]
    # Prefer explicit machine codes; keep a tiny legacy prefix map for backward compatibility.
    explicit_contract_codes = {
        "contract.input_invalid",
        "contract.output_invalid",
        "contract.schema_not_found",
        "contract.schema_invalid",
        "contract.tool_not_found",
        "contract.input_missing",
        "contract.intent_mismatch",
    }
    if head in explicit_contract_codes:
        return True
    legacy_contract_prefixes = {
        "unknown_tool",
        "memory_only_inputs_missing",
        "tool_intent_mismatch",
    }
    return head in legacy_contract_prefixes


def _should_retry_task(task_payload: Mapping[str, Any], error: str | None) -> bool:
    if not WORKER_RETRY_ENABLED:
        return False
    attempts, max_attempts = _task_attempt_limits(task_payload)
    if attempts >= max_attempts:
        return False
    if _is_contract_error(error):
        return False
    if WORKER_RETRY_POLICY == "none":
        return False
    if WORKER_RETRY_POLICY == "any":
        return True
    return _is_transient_error(error)


def _emit_task_event(
    event_type: str, envelope: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_id = envelope.get("task_id") if isinstance(envelope.get("task_id"), str) else None
    event = models.EventEnvelope(
        type=event_type,
        version="1",
        occurred_at=datetime.utcnow(),
        correlation_id=str(envelope.get("correlation_id") or str(uuid.uuid4())),
        job_id=envelope.get("job_id") if isinstance(envelope.get("job_id"), str) else None,
        task_id=task_id,
        payload=dict(payload),
    )
    redis_client.xadd(events.TASK_STREAM, {"data": event.model_dump_json()})


def _emit_task_dlq_event(
    message_id: str, envelope: Mapping[str, Any], task_payload: Mapping[str, Any], error: str
) -> None:
    if not WORKER_DLQ_ENABLED:
        return
    dlq_payload = {
        "message_id": message_id,
        "failed_at": datetime.utcnow().isoformat(),
        "error": error,
        "worker_consumer": WORKER_CONSUMER,
        "job_id": envelope.get("job_id"),
        "task_id": task_payload.get("task_id"),
        "envelope": dict(envelope),
        "task_payload": dict(task_payload),
    }
    redis_client.xadd(events.TASK_DLQ_STREAM, {"data": json.dumps(dlq_payload, ensure_ascii=True)})


def _queue_task_retry(
    task_payload: Mapping[str, Any], envelope: Mapping[str, Any], error: str
) -> None:
    attempts, max_attempts = _task_attempt_limits(task_payload)
    retry_payload = dict(task_payload)
    run_id = str(
        retry_payload.get("run_id")
        or envelope.get("correlation_id")
        or retry_payload.get("correlation_id")
        or uuid.uuid4()
    )
    retry_payload["run_id"] = run_id
    retry_payload["attempts"] = attempts + 1
    retry_payload["max_attempts"] = max_attempts
    retry_payload["status"] = models.TaskStatus.ready.value
    retry_payload["last_error"] = error
    _emit_task_event(
        "task.heartbeat",
        envelope,
        {
            "task_id": retry_payload.get("task_id"),
            "status": "retrying",
            "attempts": attempts,
            "max_attempts": max_attempts,
            "error": error,
            "worker_consumer": WORKER_CONSUMER,
            "run_id": run_id,
        },
    )
    _emit_task_event("task.ready", envelope, retry_payload)


def _claim_stale_messages(group: str, consumer: str) -> list[tuple[str, Mapping[str, str]]]:
    try:
        res = redis_client.xautoclaim(
            events.TASK_STREAM,
            group,
            consumer,
            min_idle_time=WORKER_RECOVER_IDLE_MS,
            start_id="0-0",
            count=WORKER_RECOVER_COUNT,
        )
    except Exception:  # noqa: BLE001
        core_logging.log_event(
            LOGGER,
            "worker_claim_failed",
            {"group": group, "consumer": consumer, "stream": events.TASK_STREAM},
        )
        return []
    if not res or len(res) < 2:
        return []
    messages = res[1]
    if messages:
        core_logging.log_event(
            LOGGER,
            "worker_claimed_messages",
            {
                "group": group,
                "consumer": consumer,
                "stream": events.TASK_STREAM,
                "count": len(messages),
            },
        )
    return messages


def _process_task_ready_message(message_id: str, envelope: Mapping[str, Any]) -> None:
    payload = envelope.get("payload")
    task_payload = dict(payload) if isinstance(payload, dict) else {}
    if not task_payload:
        return
    run_id = str(
        task_payload.get("run_id")
        or envelope.get("correlation_id")
        or task_payload.get("correlation_id")
        or uuid.uuid4()
    )
    task_payload["run_id"] = run_id
    attempts, max_attempts = _task_attempt_limits(task_payload)
    _emit_task_event(
        "task.started",
        envelope,
        {
            "task_id": task_payload.get("task_id"),
            "attempts": attempts,
            "max_attempts": max_attempts,
            "worker_consumer": WORKER_CONSUMER,
            "run_id": run_id,
        },
    )
    try:
        result = execute_task(task_payload)
    except Exception as exc:  # noqa: BLE001
        error = f"worker_execution_error:{exc}"
        core_logging.log_event(
            LOGGER,
            "worker_task_execution_error",
            {
                "task_id": task_payload.get("task_id"),
                "message_id": message_id,
                "error": error,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "run_id": run_id,
            },
        )
        if _should_retry_task(task_payload, error):
            _queue_task_retry(task_payload, envelope, error)
            return
        failed_payload = {
            "task_id": str(task_payload.get("task_id") or ""),
            "status": models.TaskStatus.failed.value,
            "outputs": {"tool_error": {"error": error}},
            "artifacts": [],
            "tool_calls": [],
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "error": error,
            "run_id": run_id,
        }
        _emit_task_event("task.failed", envelope, failed_payload)
        _emit_task_dlq_event(message_id, envelope, task_payload, error)
        return

    result_error = result.error
    if not result_error:
        maybe_tool_error = (
            result.outputs.get("tool_error") if isinstance(result.outputs, dict) else {}
        )
        if isinstance(maybe_tool_error, dict):
            value = maybe_tool_error.get("error")
            if isinstance(value, str) and value:
                result_error = value

    if result.status == models.TaskStatus.failed and _should_retry_task(task_payload, result_error):
        if isinstance(result_error, str) and result_error.startswith("tool_call_timed_out"):
            _emit_task_event(
                "task.heartbeat",
                envelope,
                {
                    "task_id": task_payload.get("task_id"),
                    "status": "timed_out",
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                    "error": result_error,
                    "worker_consumer": WORKER_CONSUMER,
                    "run_id": run_id,
                },
            )
        _queue_task_retry(task_payload, envelope, result_error or "task_failed")
        return

    event_type = "task.failed" if result.status == models.TaskStatus.failed else "task.completed"
    result_payload = result.model_dump(mode="json")
    result_payload["run_id"] = run_id
    _emit_task_event(event_type, envelope, result_payload)
    if result.status == models.TaskStatus.failed:
        _emit_task_dlq_event(
            message_id,
            envelope,
            task_payload,
            result_error or "task_failed",
        )


def run() -> None:
    global _METRICS_SERVER_STARTED
    if not _METRICS_SERVER_STARTED:
        start_http_server(METRICS_PORT)
        _METRICS_SERVER_STARTED = True
    group = WORKER_GROUP
    consumer = WORKER_CONSUMER
    try:
        redis_client.xgroup_create(events.TASK_STREAM, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass
    last_recovery = 0.0
    while True:
        reclaimed_messages: list[tuple[str, Mapping[str, str]]] = []
        now = time.time()
        if WORKER_RECOVER_PENDING and now - last_recovery >= WORKER_RECOVER_INTERVAL_S:
            reclaimed_messages = _claim_stale_messages(group, consumer)
            last_recovery = now
        if reclaimed_messages:
            entries = reclaimed_messages
        else:
            messages = redis_client.xreadgroup(
                group,
                consumer,
                {events.TASK_STREAM: ">"},
                count=WORKER_READ_COUNT,
                block=WORKER_BLOCK_MS,
            )
            entries = []
            for _, stream_entries in messages:
                entries.extend(stream_entries)
        for message_id, data in entries:
            try:
                raw = data.get("data")
                payload = json.loads(raw) if isinstance(raw, str) else {}
            except json.JSONDecodeError:
                payload = {}
            try:
                if payload.get("type") == "task.ready":
                    _process_task_ready_message(message_id, payload)
            finally:
                redis_client.xack(events.TASK_STREAM, group, message_id)
        time.sleep(0.05)


if __name__ == "__main__":
    run()
