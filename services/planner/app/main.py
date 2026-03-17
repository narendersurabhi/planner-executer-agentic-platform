from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from time import perf_counter
from typing import Any, List, Mapping, Sequence

from prometheus_client import start_http_server
from pydantic import BaseModel

from libs.core import (
    capability_search,
    capability_registry,
    execution_contracts,
    intent_contract,
    job_projection,
    llm_provider,
    logging as core_logging,
    models,
    planner_contracts,
    prompts,
    workflow_contracts,
)
try:
    from . import bootstrap, planner_service, runtime_service
except ImportError:  # pragma: no cover - compatibility for direct module loading in tests
    from services.planner.app import bootstrap, planner_service, runtime_service

core_logging.configure_logging("planner")

_PLANNER_BOOTSTRAP: bootstrap.PlannerBootstrap | None = None

DEFAULT_ALLOWED_BLOCK_TYPES = [
    "text",
    "paragraph",
    "heading",
    "bullets",
    "spacer",
    "optional_paragraph",
    "repeat",
]


def _log_entry_exit(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = core_logging.get_logger("planner")
            safe_args = _safe_log_value(args)
            safe_kwargs = _safe_log_value(kwargs)
            logger.info("enter", function=name, args=safe_args, kwargs=safe_kwargs)
            started = perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = int((perf_counter() - started) * 1000)
                safe_result = _safe_log_value(result)
                logger.info(
                    "exit",
                    function=name,
                    status="ok",
                    duration_ms=duration_ms,
                    result=safe_result,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                duration_ms = int((perf_counter() - started) * 1000)
                logger.error(
                    "exit",
                    function=name,
                    status="error",
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                raise

        return wrapper

    return decorator


def _json_fallback(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_log_value(value: object, max_len: int = 10000) -> object:
    try:
        serialized = json.dumps(value, default=_json_fallback, ensure_ascii=True)
    except Exception:
        try:
            serialized = json.dumps(str(value), ensure_ascii=True)
        except Exception:
            return "<unserializable>"
    if len(serialized) > max_len:
        return f"{serialized[:max_len]}...[truncated]"
    try:
        return json.loads(serialized)
    except Exception:
        return serialized


@_log_entry_exit("rule_based_plan")
def rule_based_plan(job: models.Job, tools: List[models.ToolSpec]) -> models.PlanCreate:
    request = planner_contracts.build_plan_request(
        job,
        tools=tools,
        capabilities={},
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return planner_service.rule_based_plan(request)


@_log_entry_exit("llm_plan")
def llm_plan(
    job: models.Job, tools: List[models.ToolSpec], provider: llm_provider.LLMProvider
) -> models.PlanCreate:
    config = _planner_service_config()
    request = planner_service.build_plan_request(
        job,
        tools,
        config=config,
        runtime=_planner_service_runtime(),
        include_semantic_hints=True,
    )
    return planner_service.llm_plan(
        request,
        provider,
        config=config,
        runtime=_planner_service_runtime(),
    )


@_log_entry_exit("plan_job")
def plan_job(job: models.Job) -> models.PlanCreate:
    execution = runtime_service.resolve_execution_context(_planner_runtime_config())
    return planner_service.plan_job(
        job,
        execution.tool_specs,
        provider=execution.provider,
        config=_planner_service_config(),
        runtime=_planner_service_runtime(),
    )


def _planner_bootstrap() -> bootstrap.PlannerBootstrap:
    global _PLANNER_BOOTSTRAP
    if _PLANNER_BOOTSTRAP is None:
        _PLANNER_BOOTSTRAP = bootstrap.build_bootstrap_from_env(
            os.environ,
            redis_client_factory=runtime_service.build_redis_client,
        )
    return _PLANNER_BOOTSTRAP


def _planner_service_config() -> planner_service.PlannerServiceConfig:
    return _planner_bootstrap().service_config


def _planner_runtime_config() -> runtime_service.PlannerRuntimeConfig:
    return _planner_bootstrap().runtime_config


def _planner_max_dependency_depth() -> int | None:
    return _planner_service_config().max_dependency_depth


def _planner_redis_client() -> object:
    return _planner_bootstrap().redis_client


def _planner_service_runtime() -> planner_service.PlannerServiceRuntime:
    return planner_service.PlannerServiceRuntime(
        load_capabilities=_planner_capabilities,
        build_semantic_capability_hints=lambda job, capabilities, limit: _planner_semantic_capability_hints(
            job,
            capabilities,
            limit=limit,
        ),
        parse_llm_plan=_parse_llm_plan,
        ensure_llm_tool=_ensure_llm_tool,
        ensure_task_intents=_ensure_task_intents_for_request,
        ensure_job_inputs=_ensure_job_inputs_for_request,
        ensure_default_value_markers=_ensure_default_value_markers_for_request,
        ensure_renderer_required_inputs=_ensure_renderer_required_inputs,
        ensure_tool_input_dependencies=_ensure_tool_input_dependencies,
        ensure_renderer_output_extensions=_ensure_renderer_output_extensions,
        ensure_execution_bindings=_ensure_execution_bindings,
        apply_max_depth=_apply_max_depth_limit,
    )


@_log_entry_exit("_job_from_event_payload")
def _job_from_event_payload(event: dict) -> models.Job | None:
    data = event.get("payload")
    if not isinstance(data, dict):
        return None
    try:
        return models.Job(**data)
    except Exception:
        job_id = data.get("id") or data.get("job_id")
        goal = data.get("goal")
        if not job_id or not goal:
            return None
        now = datetime.now(UTC)
        return models.Job(
            id=job_id,
            goal=goal,
            context_json=data.get("context_json", {}),
            status=_parse_job_status(data.get("status")),
            created_at=_parse_datetime(data.get("created_at"), now),
            updated_at=_parse_datetime(data.get("updated_at"), now),
            priority=data.get("priority", 0),
            metadata=data.get("metadata", {}),
        )


def _parse_job_status(value: object) -> models.JobStatus:
    if isinstance(value, models.JobStatus):
        return value
    if isinstance(value, str):
        try:
            return models.JobStatus(value)
        except ValueError:
            return models.JobStatus.queued
    return models.JobStatus.queued


def _parse_datetime(value: object, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def _llm_prompt(job: models.Job, tools: List[models.ToolSpec]) -> str:
    request = planner_service.build_plan_request(
        job,
        tools,
        config=_planner_service_config(),
        runtime=_planner_service_runtime(),
        include_semantic_hints=True,
    )
    return planner_service.build_llm_prompt(request)


def _llm_plan_repair_prompt(
    *, original_prompt: str, raw_output: str, tools: List[models.ToolSpec]
) -> str:
    request = planner_contracts.PlanRequest(
        job_id="repair",
        goal="",
        tools=list(tools),
        capabilities=[
            planner_contracts.PlanRequestCapability(
                capability_id=capability_id,
                description=capability.description,
                risk_tier=capability.risk_tier,
                idempotency=capability.idempotency,
                group=capability.group,
                subgroup=capability.subgroup,
                input_schema_ref=capability.input_schema_ref,
                output_schema_ref=capability.output_schema_ref,
                planner_hints=(
                    dict(capability.planner_hints)
                    if isinstance(capability.planner_hints, dict)
                    else {}
                ),
                adapters=[
                    planner_contracts.PlanRequestCapabilityAdapter(
                        type=adapter.type,
                        server_id=adapter.server_id,
                        tool_name=adapter.tool_name,
                    )
                    for adapter in capability.adapters
                    if adapter.enabled
                ],
            )
            for capability_id, capability in _planner_capabilities().items()
        ],
    )
    return planner_service.build_llm_repair_prompt(original_prompt, raw_output, request)


def _document_spec_prompt(job: models.Job, allowed_block_types: List[str]) -> str:
    return prompts.document_spec_prompt(job.model_dump(mode="json"), allowed_block_types)


@_log_entry_exit("_parse_llm_plan")
def _parse_llm_plan(content: str) -> models.PlanCreate | None:
    try:
        data = llm_provider.parse_json_object(content)
    except llm_provider.LLMProviderError:
        return None
    candidates = []
    inner = data.get("plan")
    if isinstance(inner, dict):
        candidates.append(inner)
    candidates.append(data)
    for candidate in candidates:
        try:
            return models.PlanCreate.model_validate(candidate)
        except Exception:
            continue
    return None


def _ensure_llm_tool(plan: models.PlanCreate) -> models.PlanCreate:
    updated_tasks = []
    for task in plan.tasks:
        if not task.tool_requests:
            updated_tasks.append(task.model_copy(update={"tool_requests": ["llm_generate"]}))
        else:
            updated_tasks.append(task)
    return plan.model_copy(update={"tasks": updated_tasks})


def _ensure_task_intents(
    plan: models.PlanCreate,
    tools: List[models.ToolSpec],
    goal_text: str = "",
    goal_intent_sequence: list[str] | None = None,
) -> models.PlanCreate:
    logger = core_logging.get_logger("planner")
    tool_map = {tool.name: tool for tool in tools}
    updated_tasks: list[models.TaskCreate] = []
    changed = False
    for index, task in enumerate(plan.tasks):
        unique_tool_intents = {
            tool_map[tool_name].tool_intent
            for tool_name in (task.tool_requests or [])
            if tool_name in tool_map
        }
        inference = intent_contract.infer_task_intent_for_task_with_metadata(
            explicit_intent=task.intent,
            description=task.description,
            instruction=task.instruction,
            acceptance_criteria=task.acceptance_criteria,
            goal_text=goal_text,
        )
        inferred = inference.intent
        intent_source = inference.source
        intent_confidence = float(inference.confidence)
        # If text inference falls back to generate, prefer known tool intent where unambiguous.
        if inferred == models.ToolIntent.generate.value and not task.intent:
            if len(unique_tool_intents) == 1:
                inferred = next(iter(unique_tool_intents)).value
                intent_source = "tool_intent"
                intent_confidence = 0.9
            elif goal_intent_sequence and index < len(goal_intent_sequence):
                hinted = intent_contract.normalize_task_intent(goal_intent_sequence[index])
                if hinted:
                    inferred = hinted
                    intent_source = "goal_graph"
                    intent_confidence = max(intent_confidence, 0.88)
        # LLM can emit explicit task intents that conflict with a task's only tool intent.
        # Auto-repair to the single tool intent so plan validation does not fail hard.
        if len(unique_tool_intents) == 1:
            has_mismatch = any(
                bool(
                    intent_contract.validate_tool_intent_compatibility(
                        inferred,
                        tool_map[tool_name].tool_intent,
                        tool_name,
                    )
                )
                for tool_name in (task.tool_requests or [])
                if tool_name in tool_map
            )
            if has_mismatch:
                inferred = next(iter(unique_tool_intents)).value
                intent_source = "tool_intent_repair"
                intent_confidence = max(intent_confidence, 0.93)
        try:
            normalized = models.ToolIntent(inferred)
        except ValueError:
            normalized = models.ToolIntent.generate
        logger.info(
            "task_intent_inferred",
            task_name=task.name,
            task_intent=normalized.value,
            intent_source=intent_source,
            intent_confidence=round(max(0.0, min(1.0, intent_confidence)), 3),
            has_explicit_intent=bool(task.intent),
        )
        if task.intent != normalized:
            updated_tasks.append(task.model_copy(update={"intent": normalized}))
            changed = True
        else:
            updated_tasks.append(task)
    if not changed:
        return plan
    return plan.model_copy(update={"tasks": updated_tasks})


def _ensure_task_intents_for_request(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
) -> models.PlanCreate:
    return _ensure_task_intents(
        plan,
        request.tools,
        goal_text=request.goal,
        goal_intent_sequence=planner_contracts.goal_intent_sequence(request),
    )


def _job_goal_intent_graph(job: models.Job) -> workflow_contracts.IntentGraph | None:
    if not isinstance(job.metadata, dict):
        return None
    graph = workflow_contracts.parse_intent_graph(job.metadata.get("goal_intent_graph"))
    if graph is None or not graph.segments:
        return None
    return graph


def _job_goal_intent_sequence(job: models.Job) -> list[str]:
    graph = _job_goal_intent_graph(job)
    if not graph:
        return []
    sequence: list[str] = []
    for segment in graph.segments:
        normalized = intent_contract.normalize_task_intent(segment.intent)
        if normalized:
            sequence.append(normalized)
    return sequence


def _job_goal_intent_segments(job: models.Job) -> list[dict[str, Any]]:
    graph = _job_goal_intent_graph(job)
    if not graph:
        return []
    segments: list[dict[str, Any]] = []
    for segment in graph.segments:
        segments.append(segment.model_dump(mode="json", exclude_none=True))
    return segments


def _select_goal_intent_segment_for_task(
    *,
    task: models.TaskCreate,
    task_index: int,
    task_intent: str,
    goal_intent_segments: Sequence[dict[str, Any]],
    total_tasks: int,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    return planner_service.select_goal_intent_segment_for_task(
        task=task,
        task_index=task_index,
        task_intent=task_intent,
        goal_intent_segments=list(goal_intent_segments),
        total_tasks=total_tasks,
        capabilities=capabilities or _planner_capabilities(),
    )


def _job_intent_mismatch_recovery(job: models.Job) -> dict[str, Any] | None:
    if not isinstance(job.metadata, dict):
        return None
    raw = job.metadata.get("intent_mismatch_recovery")
    if not isinstance(raw, dict):
        return None
    return raw


def _job_intent_mismatch_recovery_block(job: models.Job) -> str:
    recovery = _job_intent_mismatch_recovery(job)
    if not recovery:
        return ""
    return _format_intent_mismatch_recovery_block(recovery)


def _format_intent_mismatch_recovery_block(recovery: Mapping[str, Any]) -> str:
    payload = json.dumps(recovery, ensure_ascii=False, indent=2, default=_json_fallback)
    guardrails = (
        "Intent mismatch auto-repair context:\n"
        f"{payload}\n"
        "Recovery rules:\n"
        "- Do not repeat the failing task/tool intent mismatch.\n"
        "- Ensure each task.intent matches the selected tool/capability intent policy.\n"
        "- If allowed_task_intents are provided, set task intent to one of them.\n"
        "- Keep dependencies valid while minimally modifying the prior plan shape.\n"
    )
    return guardrails


def _ensure_job_inputs(
    plan: models.PlanCreate, job: models.Job, tools: List[models.ToolSpec]
) -> models.PlanCreate:
    request = planner_contracts.build_plan_request(
        job,
        tools=tools,
        capabilities={},
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return _ensure_job_inputs_for_request(plan, request)


def _ensure_job_inputs_for_request(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
) -> models.PlanCreate:
    tool_map = {tool.name: tool for tool in request.tools}
    updated_tasks = []
    for task in plan.tasks:
        tool_inputs = dict(task.tool_inputs) if isinstance(task.tool_inputs, dict) else {}
        changed = False
        for tool_name in task.tool_requests or []:
            tool = tool_map.get(tool_name)
            payload = tool_inputs.get(tool_name)
            payload = dict(payload) if isinstance(payload, dict) else {}
            projected_inputs = job_projection.project_explicit_inputs_for_tool(
                tool_name,
                request.job_payload,
                default_goal=request.goal,
            )
            if projected_inputs:
                if "job" in payload:
                    payload.pop("job", None)
                    changed = True
                for key, value in projected_inputs.items():
                    if key in payload:
                        continue
                    payload[key] = value
                    changed = True
                if changed:
                    tool_inputs[tool_name] = payload
                continue
            if tool is None:
                continue
            schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
            if not _schema_requires_key(schema, "job"):
                continue
            existing_job = payload.get("job")
            projected_job_payload = job_projection.project_job_payload_for_tool(
                tool_name,
                request.job_payload,
                default_goal=request.goal,
            )
            if isinstance(existing_job, dict):
                # Resolve planner placeholders so downstream contract checks can inspect actual context.
                marker_keys = {str(key) for key in existing_job.keys()}
                if "$default" in marker_keys and bool(existing_job.get("$default")):
                    payload["job"] = projected_job_payload
                elif marker_keys and marker_keys.issubset({"$default", "$from"}):
                    payload["job"] = projected_job_payload
                else:
                    continue
            elif existing_job is not None:
                continue
            payload["job"] = projected_job_payload
            tool_inputs[tool_name] = payload
            changed = True
        if changed:
            updated_tasks.append(task.model_copy(update={"tool_inputs": tool_inputs}))
        else:
            updated_tasks.append(task)
    return plan.model_copy(update={"tasks": updated_tasks})


def _ensure_default_value_markers(plan: models.PlanCreate, job: models.Job) -> models.PlanCreate:
    request = planner_contracts.build_plan_request(
        job,
        tools=[],
        capabilities={},
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return _ensure_default_value_markers_for_request(plan, request)


def _ensure_default_value_markers_for_request(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
) -> models.PlanCreate:
    job_context = request.job_context if isinstance(request.job_context, dict) else {}
    if not job_context:
        return plan
    tasks = [task.model_copy(deep=True) for task in plan.tasks]
    changed = False
    for task in tasks:
        if not isinstance(task.tool_inputs, dict):
            continue
        tool_inputs = dict(task.tool_inputs)
        task_changed = False
        for request_id, payload_raw in list(tool_inputs.items()):
            if not isinstance(payload_raw, dict):
                continue
            payload = dict(payload_raw)
            payload_changed = False
            for key, value in list(payload.items()):
                if not (isinstance(value, dict) and value.get("$default") is True):
                    continue
                replacement = job_context.get(key)
                if isinstance(replacement, str):
                    stripped = replacement.strip()
                    if stripped:
                        payload[key] = stripped
                        payload_changed = True
                        continue
                if isinstance(replacement, (int, float, bool)):
                    payload[key] = replacement
                    payload_changed = True
                    continue
                # No usable default in context: drop marker and allow later fillers/validators.
                payload.pop(key, None)
                payload_changed = True
            if payload_changed:
                tool_inputs[request_id] = payload
                task_changed = True
        if task_changed:
            task.tool_inputs = tool_inputs
            changed = True
    if not changed:
        return plan
    return plan.model_copy(update={"tasks": tasks})


def _ensure_renderer_output_extensions(plan: models.PlanCreate) -> models.PlanCreate:
    derive_request_ids, renderer_extension_by_request = _planner_renderer_hints()
    if not derive_request_ids or not renderer_extension_by_request:
        return plan
    task_by_name = {task.name: task for task in plan.tasks}
    derive_ext_requests: dict[str, set[str]] = {}

    for task in plan.tasks:
        if not task.tool_requests:
            continue
        tool_inputs = task.tool_inputs if isinstance(task.tool_inputs, dict) else {}
        for request_id in task.tool_requests:
            required_ext = renderer_extension_by_request.get(request_id)
            if not required_ext:
                continue
            payload = tool_inputs.get(request_id)
            if not isinstance(payload, dict):
                continue
            derive_task_name = _derive_task_name_from_path_reference(
                payload.get("path"), derive_request_ids
            )
            if not derive_task_name:
                continue
            derive_task = task_by_name.get(derive_task_name)
            if not derive_task or not any(
                tool in derive_request_ids for tool in (derive_task.tool_requests or [])
            ):
                continue
            derive_ext_requests.setdefault(derive_task_name, set()).add(required_ext)

    if not derive_ext_requests:
        return plan

    updated_tasks: list[models.TaskCreate] = []
    changed = False
    for task in plan.tasks:
        requested = derive_ext_requests.get(task.name)
        if not requested or len(requested) != 1:
            updated_tasks.append(task)
            continue
        derive_request_id = _derive_request_id_for_task(task, derive_request_ids)
        if not derive_request_id:
            updated_tasks.append(task)
            continue
        extension = next(iter(requested))
        tool_inputs = dict(task.tool_inputs) if isinstance(task.tool_inputs, dict) else {}
        payload = tool_inputs.get(derive_request_id)
        payload_dict = dict(payload) if isinstance(payload, dict) else {}
        task_changed = False
        if _normalize_extension_hint(payload_dict.get("output_extension")) != extension:
            payload_dict["output_extension"] = extension
            task_changed = True
        document_type = payload_dict.get("document_type")
        if not isinstance(document_type, str) or not document_type.strip():
            payload_dict["document_type"] = _infer_document_type_for_derive_task(
                task,
                task_by_name=task_by_name,
                all_tasks=plan.tasks,
            )
            task_changed = True
        if not task_changed:
            updated_tasks.append(task)
            continue
        tool_inputs[derive_request_id] = payload_dict
        updated_tasks.append(task.model_copy(update={"tool_inputs": tool_inputs}))
        changed = True
    if not changed:
        return plan
    return plan.model_copy(update={"tasks": updated_tasks})


def _ensure_execution_bindings(plan: models.PlanCreate) -> models.PlanCreate:
    capabilities = _planner_capabilities()
    updated_tasks: list[models.TaskCreate] = []
    changed = False
    for task in plan.tasks:
        normalized_bindings = execution_contracts.normalize_capability_bindings(
            task.capability_bindings,
            request_ids=task.tool_requests,
            capabilities=capabilities,
        )
        if normalized_bindings != (task.capability_bindings or {}):
            updated_tasks.append(
                task.model_copy(update={"capability_bindings": normalized_bindings})
            )
            changed = True
        else:
            updated_tasks.append(task)
    if not changed:
        return plan
    return plan.model_copy(update={"tasks": updated_tasks})


def _infer_document_type_for_derive_task(
    task: models.TaskCreate,
    *,
    task_by_name: Mapping[str, models.TaskCreate],
    all_tasks: Sequence[models.TaskCreate],
) -> str:
    return "document"


def _planner_renderer_hints() -> tuple[set[str], dict[str, str]]:
    capabilities = _planner_capabilities()
    derive_request_ids: set[str] = set()
    renderer_extension_by_request: dict[str, str] = {}
    for capability_id, spec in capabilities.items():
        hints = spec.planner_hints if isinstance(spec.planner_hints, dict) else {}
        if bool(hints.get("derives_output_path")):
            derive_request_ids.add(capability_id)
        required_ext = hints.get("required_output_extension")
        normalized_ext = _normalize_extension_hint(required_ext)
        if normalized_ext:
            renderer_extension_by_request[capability_id] = normalized_ext
    return derive_request_ids, renderer_extension_by_request


def _derive_task_name_from_path_reference(
    path_value: Any, derive_request_ids: set[str]
) -> str | None:
    if not isinstance(path_value, dict):
        return None
    path_spec = path_value.get("$from")
    segments: list[Any]
    if isinstance(path_spec, list):
        segments = list(path_spec)
    elif isinstance(path_spec, str):
        raw = path_spec.strip()
        if not raw:
            return None
        segments = [segment for segment in raw.split(".") if segment]
    else:
        return None
    if len(segments) < 4:
        return None
    if segments[0] not in ("dependencies_by_name", "dependencies"):
        return None
    source_tool = str(segments[2])
    output_field = str(segments[3])
    if source_tool not in derive_request_ids:
        return None
    if output_field != "path":
        return None
    return str(segments[1])


def _reference_task_name_from_from_spec(from_spec: Any) -> str | None:
    segments: list[Any]
    if isinstance(from_spec, list):
        segments = list(from_spec)
    elif isinstance(from_spec, str):
        raw = from_spec.strip()
        if not raw:
            return None
        segments = [segment for segment in raw.split(".") if segment]
    else:
        return None
    if len(segments) < 2:
        return None
    if segments[0] not in ("dependencies_by_name", "dependencies"):
        return None
    task_name = str(segments[1]).strip()
    return task_name or None


def _collect_referenced_task_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        if "$from" in value:
            ref_name = _reference_task_name_from_from_spec(value.get("$from"))
            if ref_name:
                names.add(ref_name)
        for child in value.values():
            names.update(_collect_referenced_task_names(child))
        return names
    if isinstance(value, list):
        for item in value:
            names.update(_collect_referenced_task_names(item))
    return names


def _ensure_tool_input_dependencies(plan: models.PlanCreate) -> models.PlanCreate:
    task_names = {task.name for task in plan.tasks}
    tasks = [task.model_copy(deep=True) for task in plan.tasks]
    changed = False
    for task in tasks:
        referenced = _collect_referenced_task_names(task.tool_inputs)
        if not referenced:
            continue
        deps = list(task.deps or [])
        dep_set = set(deps)
        for dep_name in sorted(referenced):
            if dep_name == task.name or dep_name not in task_names or dep_name in dep_set:
                continue
            deps.append(dep_name)
            dep_set.add(dep_name)
            changed = True
        if deps != list(task.deps or []):
            task.deps = deps
    if not changed:
        return plan

    edges: list[list[str]] = []
    existing_edges: set[tuple[str, str]] = set()
    for edge in plan.dag_edges or []:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        from_node = str(edge[0])
        to_node = str(edge[1])
        pair = (from_node, to_node)
        if pair in existing_edges:
            continue
        existing_edges.add(pair)
        edges.append([from_node, to_node])
    for task in tasks:
        for dep_name in task.deps or []:
            pair = (dep_name, task.name)
            if pair in existing_edges:
                continue
            existing_edges.add(pair)
            edges.append([dep_name, task.name])
    return plan.model_copy(update={"tasks": tasks, "dag_edges": edges})


def _ensure_renderer_required_inputs(plan: models.PlanCreate) -> models.PlanCreate:
    tasks = [task.model_copy(deep=True) for task in plan.tasks]
    task_by_name = {task.name: task for task in tasks}
    changed = False
    renderer_requests = {
        "docx_generate_from_spec",
        "pdf_generate_from_spec",
        "document.docx.generate",
        "document.pdf.generate",
    }
    document_spec_consumer_requests = {
        "document_spec_validate",
        "document.spec.validate",
    }
    document_spec_producers = {
        "document.spec.generate",
        "document_spec_generate",
        "llm_generate_document_spec",
        "llm_repair_document_spec",
        "llm_improve_document_spec",
        "llm_iterative_improve_document_spec",
        "document.spec.repair",
    }
    def _ref_task_exists(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        ref_task = _reference_task_name_from_from_spec(value.get("$from"))
        if not ref_task:
            return False
        return ref_task in task_by_name

    def _find_source_task_name(
        current_task: models.TaskCreate,
        index: int,
        producer_requests: set[str],
    ) -> tuple[str | None, str | None]:
        deps = [str(dep) for dep in (current_task.deps or [])]
        for dep_name in deps:
            dep_task = task_by_name.get(dep_name)
            if not dep_task:
                continue
            for req in dep_task.tool_requests or []:
                req_name = str(req)
                if req_name in producer_requests:
                    return dep_task.name, req_name
        for dep_task in tasks[:index]:
            for req in dep_task.tool_requests or []:
                req_name = str(req)
                if req_name in producer_requests:
                    return dep_task.name, req_name
        for dep_task in tasks:
            if dep_task.name == current_task.name:
                continue
            for req in dep_task.tool_requests or []:
                req_name = str(req)
                if req_name in producer_requests:
                    return dep_task.name, req_name
        return None, None

    for index, task in enumerate(tasks):
        request_id = next(
            (
                req
                for req in (task.tool_requests or [])
                if str(req) in renderer_requests
                or str(req) in document_spec_consumer_requests
            ),
            None,
        )
        if request_id is None:
            continue
        request_id = str(request_id)
        tool_inputs = dict(task.tool_inputs) if isinstance(task.tool_inputs, dict) else {}
        payload_raw = tool_inputs.get(request_id)
        payload = dict(payload_raw) if isinstance(payload_raw, dict) else {}

        has_valid_document_spec_ref = _ref_task_exists(payload.get("document_spec"))
        if ("document_spec" not in payload) or (
            isinstance(payload.get("document_spec"), dict) and not has_valid_document_spec_ref
        ):
            source_task_name, source_request_id = _find_source_task_name(
                task, index, document_spec_producers
            )
            if source_task_name and source_request_id:
                payload["document_spec"] = {
                    "$from": (
                        f"dependencies_by_name.{source_task_name}."
                        f"{source_request_id}.document_spec"
                    )
                }
                changed = True

        if request_id in renderer_requests:
            has_valid_path_ref = _ref_task_exists(payload.get("path"))
            has_valid_output_path_ref = _ref_task_exists(payload.get("output_path"))
            if isinstance(payload.get("path"), dict) and not has_valid_path_ref:
                payload.pop("path", None)
                changed = True
            if isinstance(payload.get("output_path"), dict) and not has_valid_output_path_ref:
                payload.pop("output_path", None)
                changed = True

        tool_inputs[request_id] = payload
        task.tool_inputs = tool_inputs

    if not changed:
        return plan
    return plan.model_copy(update={"tasks": tasks})


def _derive_request_id_for_task(
    task: models.TaskCreate, derive_request_ids: set[str]
) -> str | None:
    for request_id in task.tool_requests or []:
        if request_id in derive_request_ids:
            return request_id
    return None


def _normalize_extension_hint(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    if normalized.startswith("."):
        normalized = normalized[1:]
    if normalized == "markdown":
        normalized = "md"
    return normalized


def _validate_plan(
    plan: models.PlanCreate, tools: List[models.ToolSpec], job: models.Job
) -> tuple[bool, str]:
    request = planner_contracts.build_plan_request(
        job,
        tools=tools,
        capabilities=_planner_capabilities(),
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return _validate_plan_request(plan, request)


def _validate_plan_request(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
) -> tuple[bool, str]:
    return planner_service.validate_plan_request(
        plan,
        request,
        schema_registry_path=_planner_service_config().schema_registry_path,
    )


def _resolve_task_intent_for_validation(
    task: models.TaskCreate,
    tool_map: dict[str, models.ToolSpec],
    goal_text: str = "",
) -> str:
    return planner_service.resolve_task_intent_for_validation(
        task,
        tool_map,
        goal_text=goal_text,
    )


def _capability_intent_mismatch(
    task_intent: str,
    capability: capability_registry.CapabilitySpec,
    capability_id: str,
) -> str | None:
    return planner_service.capability_intent_mismatch(
        task_intent,
        capability,
        capability_id,
    )


def _build_validation_payload(
    task: models.TaskCreate,
    tool: models.ToolSpec,
    job: models.Job,
    raw_tool_inputs: dict,
) -> dict:
    request = planner_contracts.build_plan_request(
        job,
        tools=[tool],
        capabilities={},
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return _build_validation_payload_for_request(task, tool, request, raw_tool_inputs)


def _build_validation_payload_for_request(
    task: models.TaskCreate,
    tool: models.ToolSpec,
    request: planner_contracts.PlanRequest,
    raw_tool_inputs: dict,
) -> dict:
    return planner_service.build_validation_payload(
        task,
        tool,
        request,
        raw_tool_inputs,
    )


def _dependency_fill_defaults() -> dict[str, object]:
    return planner_service.dependency_fill_defaults()


def _planner_capabilities() -> dict[str, capability_registry.CapabilitySpec]:
    mode = capability_registry.resolve_capability_mode()
    if mode == "disabled":
        return {}
    try:
        registry = capability_registry.load_capability_registry()
    except Exception as exc:  # noqa: BLE001
        core_logging.get_logger("planner").warning(
            "capability_registry_load_failed",
            mode=mode,
            error=str(exc),
        )
        return {}
    return dict(sorted(registry.enabled_capabilities().items()))


def _emit_plan_event(
    event_type: str,
    *,
    job_id: str | None,
    correlation_id: str | None,
    payload: dict[str, Any],
) -> None:
    runtime_service.emit_plan_event(
        _planner_redis_client(),
        event_type,
        job_id=job_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def _planner_semantic_capability_hints(
    job: models.Job,
    capabilities: Mapping[str, capability_registry.CapabilitySpec],
    *,
    correlation_id: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    entries = capability_search.build_capability_search_entries(capabilities)
    started = perf_counter()
    matches = capability_search.search_capabilities(
        query=job.goal,
        capability_entries=entries,
        limit=limit,
    )
    latency_ms = round((perf_counter() - started) * 1000.0, 3)
    _emit_plan_event(
        "plan.capability_search",
        job_id=job.id,
        correlation_id=correlation_id,
        payload={
            "job_id": job.id,
            "goal": job.goal,
            "request_source": "planner_prompt",
            "limit": limit,
            "latency_ms": latency_ms,
            "results": matches,
            "result_count": len(matches),
        },
    )
    return matches


def _emit_planner_capability_selection_event(
    job: models.Job,
    plan: models.PlanCreate,
    *,
    correlation_id: str | None = None,
) -> None:
    selected_capabilities: list[str] = []
    for task in plan.tasks:
        for tool_name in task.tool_requests or []:
            normalized = str(tool_name or "").strip()
            if normalized and normalized not in selected_capabilities:
                selected_capabilities.append(normalized)
    _emit_plan_event(
        "plan.capability_selection",
        job_id=job.id,
        correlation_id=correlation_id,
        payload={
            "job_id": job.id,
            "goal": job.goal,
            "planner_version": plan.planner_version,
            "selected_capabilities": selected_capabilities,
            "task_count": len(plan.tasks),
        },
    )


def _validate_capability_inputs(
    capability: capability_registry.CapabilitySpec | planner_contracts.PlanRequestCapability,
    task: models.TaskCreate,
    raw_tool_inputs: dict[str, Any],
    job_or_request: models.Job | planner_contracts.PlanRequest,
) -> str | None:
    request = (
        job_or_request
        if isinstance(job_or_request, planner_contracts.PlanRequest)
        else planner_contracts.build_plan_request(
            job_or_request,
            tools=[],
            capabilities={},
            semantic_capability_hints=[],
            max_dependency_depth=_planner_max_dependency_depth(),
        )
    )
    return planner_service.validate_capability_inputs(
        capability,
        task,
        raw_tool_inputs,
        request,
        schema_registry_path=_planner_service_config().schema_registry_path,
    )


def _synthesize_github_repo_query(
    *,
    raw_tool_inputs: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
    job_context: Mapping[str, Any] | None,
) -> str | None:
    return planner_service.synthesize_github_repo_query(
        raw_tool_inputs=raw_tool_inputs,
        payload=payload,
        job_context=job_context,
    )


def _load_schema_from_ref(schema_ref: str) -> dict[str, Any] | None:
    return planner_service.load_schema_from_ref(
        schema_ref,
        schema_registry_path=_planner_service_config().schema_registry_path,
    )


def _build_capability_validation_payload(
    task: models.TaskCreate,
    raw_tool_inputs: dict[str, Any],
    job: models.Job,
) -> dict[str, Any]:
    request = planner_contracts.build_plan_request(
        job,
        tools=[],
        capabilities={},
        semantic_capability_hints=[],
        max_dependency_depth=_planner_max_dependency_depth(),
    )
    return _build_capability_validation_payload_for_request(task, raw_tool_inputs, request)


def _build_capability_validation_payload_for_request(
    task: models.TaskCreate,
    raw_tool_inputs: dict[str, Any],
    request: planner_contracts.PlanRequest,
) -> dict[str, Any]:
    return planner_service.build_capability_validation_payload(
        task,
        raw_tool_inputs,
        request,
    )


def _schema_requires_key(schema: dict, key: str) -> bool:
    return planner_service.schema_requires_key(schema, key)


def _parse_optional_float(value: str | None) -> float | None:
    return bootstrap.parse_optional_float(value)


def _parse_optional_int(value: str | None) -> int | None:
    return bootstrap.parse_optional_int(value)


def _apply_max_depth(plan: models.PlanCreate) -> models.PlanCreate:
    return _apply_max_depth_limit(plan, _planner_max_dependency_depth())


def _apply_max_depth_limit(
    plan: models.PlanCreate,
    max_depth: int | None,
) -> models.PlanCreate:
    if not max_depth or max_depth <= 0:
        return plan
    tasks = [task.model_copy(deep=True) for task in plan.tasks]
    for _ in range(len(tasks) + 1):
        depths = _task_depths(tasks, max_depth)
        current_max = max(depths.values(), default=0)
        if current_max <= max_depth:
            break
        changed = False
        for task in tasks:
            if not task.deps:
                continue
            allowed = [dep for dep in task.deps if depths.get(dep, 1) < max_depth]
            if allowed != task.deps:
                task.deps = allowed
                changed = True
        if not changed:
            for task in tasks:
                if depths.get(task.name, 1) > max_depth and task.deps:
                    task.deps = []
                    changed = True
            if not changed:
                break
    return plan.model_copy(update={"tasks": tasks})


def _task_depths(tasks: List[models.TaskCreate], limit: int) -> dict[str, int]:
    by_name = {task.name: task for task in tasks}
    depth_cache: dict[str, int] = {}
    visiting: set[str] = set()

    def depth_for(name: str) -> int:
        if name in depth_cache:
            return depth_cache[name]
        if name in visiting:
            return limit + 1
        visiting.add(name)
        task = by_name.get(name)
        if not task:
            visiting.remove(name)
            return 1
        max_depth = 1
        for dep in task.deps:
            max_depth = max(max_depth, depth_for(dep) + 1)
            if max_depth > limit:
                break
        visiting.remove(name)
        depth_cache[name] = max_depth
        return max_depth

    for task in tasks:
        depth_for(task.name)
    return depth_cache


def run() -> None:
    planner_bootstrap = _planner_bootstrap()
    planner_bootstrap.ensure_metrics_started(start_http_server)
    runtime_service.run_loop(
        _planner_redis_client(),
        config=_planner_runtime_config(),
        callbacks=runtime_service.PlannerRuntimeCallbacks(
            parse_job_event=_job_from_event_payload,
            plan_job=plan_job,
        ),
    )


if __name__ == "__main__":
    run()
