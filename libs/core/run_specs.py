from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from libs.core import capability_registry, execution_contracts, models, planner_contracts


def parse_run_spec(value: Any) -> models.RunSpec | None:
    if isinstance(value, models.RunSpec):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return models.RunSpec.model_validate(dict(value))
    except ValidationError:
        return None


def plan_to_run_spec(
    plan: models.PlanCreate,
    *,
    kind: models.RunKind = models.RunKind.studio,
    metadata: Mapping[str, Any] | None = None,
) -> models.RunSpec:
    enabled_capabilities = _enabled_capabilities()
    used_step_ids: set[str] = set()
    step_id_by_task_name = {
        task.name: _step_id_for_name(task.name, index=index, used_ids=used_step_ids)
        for index, task in enumerate(plan.tasks)
    }
    steps: list[models.StepSpec] = []
    capability_requests: list[models.CapabilityRequestSpec] = []
    for task in plan.tasks:
        source_request_ids = list(task.capability_requests or task.tool_requests or [])
        planner_request_field = (
            "capability_requests" if task.capability_requests else "tool_requests"
        )
        compiled = planner_contracts.compile_task_request_payloads(
            capability_requests=getattr(task, "capability_requests", None),
            tool_requests=task.tool_requests,
            tool_inputs=task.tool_inputs,
            capability_bindings=task.capability_bindings,
            capabilities=enabled_capabilities,
        )
        execution_request_id, capability_id, routing_hints = _task_request(task, compiled=compiled)
        planner_request_id = source_request_ids[0] if source_request_ids else capability_id
        if planner_request_id:
            routing_hints["planner_request_field"] = planner_request_field
        execution_gate = execution_contracts.normalize_execution_gates(
            {"tool_inputs": task.tool_inputs},
            request_ids=source_request_ids,
        )
        execution_gate = planner_contracts.rewrite_request_keyed_mapping(
            execution_gate,
            compiled.request_id_rewrites,
        ).get(execution_request_id)
        step = models.StepSpec(
            step_id=step_id_by_task_name[task.name],
            name=task.name,
            description=task.description,
            instruction=task.instruction,
            intent=task.intent,
            capability_request=models.CapabilityRequestSpec(
                request_id=planner_request_id,
                capability_id=capability_id,
                execution_request_id=execution_request_id,
            ),
            input_bindings=_request_inputs(compiled.tool_inputs, execution_request_id),
            execution_gate=dict(execution_gate) if isinstance(execution_gate, Mapping) else None,
            expected_output_schema_ref=task.expected_output_schema_ref,
            retry_policy=models.StepRetryPolicy(),
            acceptance_policy=models.StepAcceptancePolicy(
                acceptance_criteria=list(task.acceptance_criteria),
                critic_required=bool(task.critic_required),
            ),
            depends_on=[
                step_id_by_task_name[dependency]
                for dependency in task.deps
                if dependency in step_id_by_task_name
            ],
            routing_hints=routing_hints,
        )
        steps.append(step)
        capability_requests.append(step.capability_request)
    return models.RunSpec(
        kind=kind,
        planner_version=plan.planner_version,
        tasks_summary=plan.tasks_summary,
        steps=steps,
        dag_edges=[
            [step_id_by_task_name[source], step_id_by_task_name[target]]
            for source, target in plan.dag_edges
            if source in step_id_by_task_name and target in step_id_by_task_name
        ],
        capability_requests=capability_requests,
        metadata=dict(metadata or {}),
    )


def run_spec_to_plan(run_spec: models.RunSpec) -> models.PlanCreate:
    step_by_id: dict[str, models.StepSpec] = {}
    for step in run_spec.steps:
        if step.step_id in step_by_id:
            raise ValueError(f"RunSpec contains duplicate step_id '{step.step_id}'.")
        step_by_id[step.step_id] = step
    task_name_counts: dict[str, int] = {}
    for step in run_spec.steps:
        task_name_counts[step.name] = task_name_counts.get(step.name, 0) + 1
    duplicate_names = sorted(name for name, count in task_name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(
            "RunSpec cannot be translated into PlanCreate because task names must be unique: "
            + ", ".join(duplicate_names)
        )

    tasks: list[models.TaskCreate] = []
    for step in run_spec.steps:
        request_id = str(step.capability_request.request_id or "").strip()
        capability_id = str(step.capability_request.capability_id or "").strip() or request_id
        execution_request_id = (
            str(step.capability_request.execution_request_id or "").strip() or request_id
        )
        if not request_id or not capability_id or not execution_request_id:
            raise ValueError(
                f"RunSpec step '{step.step_id}' is missing capability request identifiers."
            )
        restored_request_id, restored_request_field = _restored_planner_request(step)
        tool_inputs = execution_contracts.embed_execution_gate(
            {restored_request_id: dict(step.input_bindings or {})},
            dict(step.execution_gate) if isinstance(step.execution_gate, Mapping) else None,
            request_ids=[restored_request_id],
        )
        capability_bindings = (
            _step_capability_bindings(
                step,
                request_id=restored_request_id,
                capability_id=capability_id,
            )
            if restored_request_field == "capability_requests"
            else {}
        )
        tasks.append(
            models.TaskCreate(
                name=step.name,
                description=step.description,
                instruction=step.instruction or f"Use capability {capability_id}.",
                acceptance_criteria=list(step.acceptance_policy.acceptance_criteria),
                expected_output_schema_ref=step.expected_output_schema_ref,
                intent=step.intent,
                deps=[
                    step_by_id[dependency_id].name
                    for dependency_id in step.depends_on
                    if dependency_id in step_by_id
                ],
                capability_requests=(
                    [restored_request_id]
                    if restored_request_field == "capability_requests"
                    else []
                ),
                tool_requests=(
                    [restored_request_id] if restored_request_field == "tool_requests" else []
                ),
                tool_inputs=tool_inputs,
                capability_bindings=capability_bindings,
                critic_required=bool(step.acceptance_policy.critic_required),
            )
        )

    dag_edges = [
        [step_by_id[source].name, step_by_id[target].name]
        for source, target in _run_spec_edges(run_spec)
        if source in step_by_id and target in step_by_id
    ]
    return models.PlanCreate(
        planner_version=run_spec.planner_version or "run_spec_adapter_v1",
        tasks_summary=run_spec.tasks_summary
        or f"Execute canonical run spec with {len(tasks)} step(s).",
        dag_edges=dag_edges,
        tasks=tasks,
    )


def _task_request(
    task: models.TaskCreate,
    *,
    compiled: planner_contracts.CompiledTaskRequests,
) -> tuple[str, str, dict[str, Any]]:
    request_ids = list(compiled.request_ids)
    if len(request_ids) != 1:
        raise ValueError(
            f"Task '{task.name}' must have exactly one capability request to convert into StepSpec."
        )
    request_id = request_ids[0]
    raw_binding = (
        compiled.capability_bindings.get(request_id)
        if isinstance(compiled.capability_bindings, Mapping)
        else None
    )
    capability_id = request_id
    routing_hints: dict[str, Any] = {}
    if isinstance(raw_binding, Mapping):
        bound_capability_id = str(raw_binding.get("capability_id") or "").strip()
        if bound_capability_id:
            capability_id = bound_capability_id
        routing_hints = {
            key: value
            for key, value in dict(raw_binding).items()
            if key not in {"request_id", "capability_id"} and value is not None
        }
    return request_id, capability_id, routing_hints


def _request_inputs(tool_inputs: Mapping[str, Any] | None, request_id: str) -> dict[str, Any]:
    stripped = execution_contracts.strip_execution_metadata_from_tool_inputs(tool_inputs)
    request_inputs = stripped.get(request_id)
    return dict(request_inputs) if isinstance(request_inputs, Mapping) else {}


def _step_capability_bindings(
    step: models.StepSpec,
    *,
    request_id: str,
    capability_id: str,
) -> dict[str, dict[str, Any]]:
    binding = {
        key: value
        for key, value in dict(step.routing_hints or {}).items()
        if key not in {"planner_request_id", "planner_request_field"} and value is not None
    }
    if capability_id != request_id or binding:
        binding["capability_id"] = capability_id
        return {request_id: binding}
    return {}


def _restored_planner_request(step: models.StepSpec) -> tuple[str, str]:
    request_id = str(step.capability_request.request_id or "").strip()
    routing_hints = dict(step.routing_hints or {})
    planner_request_field = str(routing_hints.get("planner_request_field") or "").strip()
    restored_request_id = request_id
    if planner_request_field in {"capability_requests", "tool_requests"}:
        return restored_request_id, planner_request_field
    if str(step.capability_request.execution_request_id or "").strip() != request_id:
        return restored_request_id, "capability_requests"
    return restored_request_id, "tool_requests"


def _enabled_capabilities() -> Mapping[str, Any]:
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return {}
    return registry.enabled_capabilities()


def _run_spec_edges(run_spec: models.RunSpec) -> list[tuple[str, str]]:
    explicit_edges: list[tuple[str, str]] = []
    for raw_edge in run_spec.dag_edges:
        if len(raw_edge) != 2:
            continue
        source = str(raw_edge[0]).strip()
        target = str(raw_edge[1]).strip()
        if source and target:
            explicit_edges.append((source, target))
    if explicit_edges:
        return explicit_edges
    return [
        (dependency_id, step.step_id)
        for step in run_spec.steps
        for dependency_id in step.depends_on
        if dependency_id
    ]


def _step_id_for_name(name: str, *, index: int, used_ids: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    candidate = slug or f"step_{index + 1}"
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used_ids:
        suffix += 1
    deduped = f"{candidate}_{suffix}"
    used_ids.add(deduped)
    return deduped
