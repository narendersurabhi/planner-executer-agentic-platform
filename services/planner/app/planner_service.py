from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel

from libs.core import (
    intent_contract,
    job_projection,
    llm_provider,
    logging as core_logging,
    models,
    payload_resolver,
    planner_contracts,
    tool_registry,
)


@dataclass(frozen=True)
class PlannerServiceConfig:
    mode: str = "rule_based"
    max_dependency_depth: int | None = None
    semantic_hint_limit: int = 10
    schema_registry_path: str = "/app/schemas"


@dataclass(frozen=True)
class PlannerServiceRuntime:
    load_capabilities: Callable[[], Mapping[str, Any]]
    build_semantic_capability_hints: Callable[
        [models.Job, Mapping[str, Any], int], list[dict[str, Any]]
    ]
    parse_llm_plan: Callable[[str], models.PlanCreate | None]
    ensure_llm_tool: Callable[[models.PlanCreate], models.PlanCreate]
    ensure_task_intents: Callable[
        [models.PlanCreate, planner_contracts.PlanRequest], models.PlanCreate
    ]
    ensure_job_inputs: Callable[[models.PlanCreate, planner_contracts.PlanRequest], models.PlanCreate]
    ensure_default_value_markers: Callable[
        [models.PlanCreate, planner_contracts.PlanRequest], models.PlanCreate
    ]
    ensure_renderer_required_inputs: Callable[[models.PlanCreate], models.PlanCreate]
    ensure_tool_input_dependencies: Callable[[models.PlanCreate], models.PlanCreate]
    ensure_renderer_output_extensions: Callable[[models.PlanCreate], models.PlanCreate]
    ensure_execution_bindings: Callable[[models.PlanCreate], models.PlanCreate]
    apply_max_depth: Callable[[models.PlanCreate, int | None], models.PlanCreate]


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


def _format_intent_mismatch_recovery_block(recovery: Mapping[str, Any] | None) -> str:
    if not isinstance(recovery, Mapping) or not recovery:
        return ""
    payload = json.dumps(recovery, ensure_ascii=False, indent=2, default=_json_fallback)
    return (
        "Intent mismatch auto-repair context:\n"
        f"{payload}\n"
        "Recovery rules:\n"
        "- Do not repeat the failing task/tool intent mismatch.\n"
        "- Ensure each task.intent matches the selected tool/capability intent policy.\n"
        "- If allowed_task_intents are provided, set task intent to one of them.\n"
        "- Keep dependencies valid while minimally modifying the prior plan shape.\n"
    )


def build_plan_request(
    job: models.Job,
    tools: list[models.ToolSpec],
    *,
    config: PlannerServiceConfig,
    runtime: PlannerServiceRuntime,
    include_semantic_hints: bool | None = None,
) -> planner_contracts.PlanRequest:
    capabilities = runtime.load_capabilities()
    use_semantic_hints = config.mode == "llm" if include_semantic_hints is None else include_semantic_hints
    semantic_hints: list[dict[str, Any]] = []
    if use_semantic_hints and config.semantic_hint_limit > 0:
        semantic_hints = runtime.build_semantic_capability_hints(
            job,
            capabilities,
            config.semantic_hint_limit,
        )
    return planner_contracts.build_plan_request(
        job,
        tools=tools,
        capabilities=capabilities,
        semantic_capability_hints=semantic_hints,
        max_dependency_depth=config.max_dependency_depth,
    )


def rule_based_plan(_: planner_contracts.PlanRequest) -> models.PlanCreate:
    checklist_task = models.TaskCreate(
        name="create_checklist",
        description="Create implementation checklist",
        instruction="Generate a detailed checklist for adding a new tool.",
        acceptance_criteria=["Checklist has at least 5 items"],
        expected_output_schema_ref="TaskResult",
        intent=models.ToolIntent.generate,
        deps=[],
        tool_requests=[],
        critic_required=True,
    )
    write_task = models.TaskCreate(
        name="write_tools_doc",
        description="Write tools.md artifact",
        instruction="Write a tools.md draft file as an artifact.",
        acceptance_criteria=["Artifact path returned"],
        expected_output_schema_ref="TaskResult",
        intent=models.ToolIntent.render,
        deps=["create_checklist"],
        tool_requests=["file_write_artifact"],
        critic_required=True,
    )
    summarize_task = models.TaskCreate(
        name="summarize_artifact",
        description="Summarize the artifact",
        instruction="Summarize the tools.md artifact.",
        acceptance_criteria=["Summary provided"],
        expected_output_schema_ref="TaskResult",
        intent=models.ToolIntent.transform,
        deps=["write_tools_doc"],
        tool_requests=["text_summarize"],
        critic_required=False,
    )
    return models.PlanCreate(
        planner_version="rule_based_v1",
        tasks_summary="Checklist, write artifact, summarize",
        dag_edges=[
            ["create_checklist", "write_tools_doc"],
            ["write_tools_doc", "summarize_artifact"],
        ],
        tasks=[checklist_task, write_task, summarize_task],
    )


def build_llm_prompt(request: planner_contracts.PlanRequest) -> str:
    capabilities = planner_contracts.capability_map(request)
    allowed_names = sorted({tool.name for tool in request.tools} | set(capabilities.keys()))
    tool_names = ", ".join(allowed_names)
    tool_catalog = [
        {
            "type": "tool",
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "usage_guidance": tool.usage_guidance,
            "risk_level": tool.risk_level.value
            if isinstance(tool.risk_level, Enum)
            else tool.risk_level,
            "tool_intent": tool.tool_intent.value
            if isinstance(tool.tool_intent, Enum)
            else tool.tool_intent,
        }
        for tool in request.tools
    ]
    capability_catalog = [
        {
            "type": "capability",
            "name": capability.capability_id,
            "description": capability.description,
            "risk_tier": capability.risk_tier,
            "idempotency": capability.idempotency,
            "group": capability.group,
            "subgroup": capability.subgroup,
            "input_schema_ref": capability.input_schema_ref,
            "output_schema_ref": capability.output_schema_ref,
            "adapters": [
                {
                    "type": adapter.type,
                    "server_id": adapter.server_id,
                    "tool_name": adapter.tool_name,
                }
                for adapter in capability.adapters
            ],
        }
        for capability in capabilities.values()
    ]
    tool_catalog_json = json.dumps(
        tool_catalog + capability_catalog,
        ensure_ascii=False,
        indent=2,
        default=_json_fallback,
    )
    depth_hint = ""
    if request.max_dependency_depth:
        depth_hint = f"Max dependency chain depth: {request.max_dependency_depth}.\n"
    intent_graph_block = ""
    if request.goal_intent_graph is not None:
        intent_graph_json = json.dumps(
            request.goal_intent_graph.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
            default=_json_fallback,
        )
        intent_graph_block = (
            "Goal intent decomposition graph (ordered hints for planning):\n"
            f"{intent_graph_json}\n"
            "Prefer preserving this segment order in tasks/dependencies.\n"
        )
    intent_repair_block = _format_intent_mismatch_recovery_block(
        planner_contracts.intent_mismatch_recovery(request)
    )
    semantic_capability_block = ""
    if request.semantic_capability_hints:
        semantic_capability_block = (
            "Most relevant capabilities for this goal from local semantic search:\n"
            f"{json.dumps(request.semantic_capability_hints, ensure_ascii=False, indent=2, default=_json_fallback)}\n"
            "Prefer these capabilities when they fit the goal and required inputs.\n"
        )
    job_json = json.dumps(request.job_payload, ensure_ascii=False, indent=2, default=_json_fallback)
    return (
        "You are a planner. Return ONLY valid JSON for a PlanCreate object (no prose).\n"
        "Required top-level fields: planner_version, tasks_summary, dag_edges, tasks.\n"
        "Schema rules:\n"
        '- dag_edges must be an array of 2-element string arrays, e.g. [["A","B"],["B","C"]].\n'
        "- acceptance_criteria must be an array of strings, not a single string.\n"
        "Each task must include: name, description, instruction, acceptance_criteria, "
        "expected_output_schema_ref, intent, deps, tool_requests, tool_inputs, critic_required.\n"
        "Example:\n"
        "{\n"
        '  "planner_version": "1.0.0",\n'
        '  "tasks_summary": "...",\n'
        '  "dag_edges": [["TaskA","TaskB"],["TaskB","TaskC"]],\n'
        '  "tasks": [\n'
        "    {\n"
        '      "name": "TaskA",\n'
        '      "description": "...",\n'
        '      "instruction": "...",\n'
        '      "acceptance_criteria": ["..."],\n'
        '      "expected_output_schema_ref": "schemas/example",\n'
        '      "intent": "generate",\n'
        '      "deps": [],\n'
        '      "tool_requests": ["llm_generate"],\n'
        '      "tool_inputs": {"llm_generate": {"text": "..."}},\n'
        '      "critic_required": false\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Rules:\n"
        "1) Use only tool names from the allowed list.\n"
        "1a) Every task must set intent to one of: transform, generate, validate, render, io.\n"
        "2) deps must reference task names that appear in this plan.\n"
        "3) If a tool requires structured JSON, add a prior task to generate that JSON and set "
        "expected_output_schema_ref to that schema.\n"
        "4) If a tool requires specific inputs, put them in task.tool_inputs "
        "(a dict keyed by tool name). Do NOT embed JSON in instruction text.\n"
        "5) Do NOT use placeholder strings like ${Task.output} in tool_inputs. "
        "When a later task needs dependency output, either omit the field and rely on deps context "
        "injection OR use explicit reference objects like "
        '{"$from":"dependencies_by_name.TaskA.tool_name.field"} (or add "$default"). '
        "You may still include other inputs like strict, allowed_block_types, or path.\n"
        "6) Prefer the generic validation + rendering pipeline:\n"
        "   - Generate a DocumentSpec JSON.\n"
        "   - Validate with document_spec_validate.\n"
        "   - Render with docx_generate_from_spec or pdf_generate_from_spec.\n"
        "   - Do not add a separate output-path derivation task unless the path itself is needed downstream.\n"
        "   Use specialized tools only if explicitly requested.\n"
        "7) If unsure, use llm_generate.\n"
        "8) Keep output compact. Do NOT copy or embed large raw text from Job JSON "
        "(especially long context fields like job_description) into tasks, instructions, "
        "acceptance criteria, or tool_inputs.\n"
        "9) For tool_inputs include only minimal scalar params. Omit large/context fields "
        "(e.g., job, memory, document_spec) "
        "and rely on runtime dependency/context injection.\n"
        "10) Keep each task instruction concise (one short paragraph) and keep acceptance "
        "criteria short bullets.\n"
        "\n"
        f"{depth_hint}"
        f"{intent_graph_block}"
        f"{intent_repair_block}"
        f"Allowed tool names: {tool_names}\n"
        f"{semantic_capability_block}"
        f"Tool catalog (JSON): {tool_catalog_json}\n"
        f"Goal: {request.goal}\n"
        f"Job (JSON): {job_json}\n"
    )


def build_llm_repair_prompt(
    original_prompt: str,
    raw_output: str,
    request: planner_contracts.PlanRequest,
) -> str:
    capabilities = planner_contracts.capability_map(request)
    allowed_names = sorted({tool.name for tool in request.tools} | set(capabilities.keys()))
    tool_names = ", ".join(allowed_names)
    return (
        "You are fixing a malformed planner response.\n"
        "Return ONLY one valid JSON object for PlanCreate.\n"
        "Do not include markdown, comments, or prose.\n"
        "Required top-level fields: planner_version, tasks_summary, dag_edges, tasks.\n"
        "Each task must include: name, description, instruction, acceptance_criteria, "
        "expected_output_schema_ref, intent, deps, tool_requests, tool_inputs, critic_required.\n"
        "Rules:\n"
        "- acceptance_criteria must be string array.\n"
        "- dag_edges must be array of 2-string arrays.\n"
        "- each task must include intent in {transform, generate, validate, render, io}.\n"
        "- Use only allowed tool names.\n"
        "- If a field is missing, add a safe default value.\n"
        f"Allowed tool names: {tool_names}\n\n"
        f"Original planner prompt (for context):\n{original_prompt}\n\n"
        f"Malformed planner output to repair:\n{raw_output}\n"
    )


def resolve_task_intent_for_validation(
    task: models.TaskCreate,
    tool_map: Mapping[str, models.ToolSpec],
    *,
    goal_text: str = "",
) -> str:
    inference = intent_contract.infer_task_intent_for_task_with_metadata(
        explicit_intent=task.intent,
        description=task.description,
        instruction=task.instruction,
        acceptance_criteria=task.acceptance_criteria,
        goal_text=goal_text,
    )
    inferred = inference.intent
    if inferred == models.ToolIntent.generate.value and not task.intent:
        unique_tool_intents = {
            tool_map[tool_name].tool_intent
            for tool_name in (task.tool_requests or [])
            if tool_name in tool_map
        }
        if len(unique_tool_intents) == 1:
            inferred = next(iter(unique_tool_intents)).value
    return inferred


def select_goal_intent_segment_for_task(
    *,
    task: models.TaskCreate,
    task_index: int,
    task_intent: str,
    goal_intent_segments: list[dict[str, Any]],
    total_tasks: int,
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not goal_intent_segments:
        return None
    capability_map = dict(capabilities or {})
    has_suggested_capabilities = any(
        isinstance(segment.get("suggested_capabilities"), list)
        and bool(segment.get("suggested_capabilities"))
        for segment in goal_intent_segments
    )
    task_requests = {
        str(name).strip().lower() for name in (task.tool_requests or []) if str(name).strip()
    }
    if task_requests:
        for segment in goal_intent_segments:
            suggested = segment.get("suggested_capabilities")
            if not isinstance(suggested, list):
                continue
            suggested_ids = set()
            for item in suggested:
                capability_id = str(item).strip().lower()
                if not capability_id:
                    continue
                suggested_ids.add(capability_id)
                capability = capability_map.get(capability_id)
                if capability is None:
                    continue
                for adapter in capability.adapters:
                    tool_name = str(adapter.tool_name or "").strip().lower()
                    if tool_name:
                        suggested_ids.add(tool_name)
            if suggested_ids & task_requests:
                return segment
    if has_suggested_capabilities:
        if len(goal_intent_segments) == 1:
            only_segment = goal_intent_segments[0]
            if intent_contract.normalize_task_intent(only_segment.get("intent")) == task_intent:
                return only_segment
        return None
    for segment in goal_intent_segments:
        if intent_contract.normalize_task_intent(segment.get("intent")) == task_intent:
            return segment
    if (
        not has_suggested_capabilities
        and len(goal_intent_segments) == total_tasks
        and task_index < len(goal_intent_segments)
    ):
        return goal_intent_segments[task_index]
    return None


def capability_intent_mismatch(
    task_intent: str,
    capability: Any,
    capability_id: str,
) -> str | None:
    hints = capability.planner_hints if isinstance(capability.planner_hints, dict) else {}
    raw_allowed = hints.get("task_intents")
    if not isinstance(raw_allowed, list) or not raw_allowed:
        return None
    allowed = {
        normalized
        for item in raw_allowed
        for normalized in [intent_contract.normalize_task_intent(item)]
        if normalized
    }
    if not allowed or task_intent in allowed:
        return None
    return f"task_intent_mismatch:{capability_id}:{task_intent}:allowed={','.join(sorted(allowed))}"


def dependency_fill_defaults() -> dict[str, object]:
    return {
        "content": "__dependency__",
        "text": "__dependency__",
        "prompt": "__dependency__",
        "data": {},
        "document_spec": {},
        "validation_report": {},
        "errors": [],
        "original_spec": {},
        "openapi_spec": {},
    }


def schema_requires_key(schema: dict[str, Any], key: str) -> bool:
    if not isinstance(schema, dict):
        return False
    required = schema.get("required")
    if isinstance(required, list) and key in required:
        return True
    for keyword in ("allOf", "anyOf", "oneOf"):
        entries = schema.get(keyword)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and schema_requires_key(entry, key):
                return True
    return False


def synthesize_github_repo_query(
    *,
    raw_tool_inputs: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
    job_context: Mapping[str, Any] | None,
) -> str | None:
    def _read_str(source: Mapping[str, Any] | None, *keys: str) -> str | None:
        if not isinstance(source, Mapping):
            return None
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    owner = (
        _read_str(raw_tool_inputs, "owner", "repo_owner")
        or _read_str(payload, "owner", "repo_owner")
        or _read_str(job_context, "owner", "repo_owner")
    )
    repo = (
        _read_str(raw_tool_inputs, "repo", "repo_name")
        or _read_str(payload, "repo", "repo_name")
        or _read_str(job_context, "repo", "repo_name")
    )
    if owner and repo:
        return f"repo:{repo} owner:{owner}"
    return (
        _read_str(raw_tool_inputs, "query", "github_query")
        or _read_str(payload, "query", "github_query")
        or _read_str(job_context, "query", "github_query")
    )


def load_schema_from_ref(schema_ref: str, *, schema_registry_path: str) -> dict[str, Any] | None:
    candidate = Path(schema_ref)
    if not candidate.is_absolute():
        candidate = Path(schema_registry_path) / (
            schema_ref if schema_ref.endswith(".json") else f"{schema_ref}.json"
        )
    if not candidate.exists():
        return None
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _fill_payload_context_defaults(
    payload: dict[str, Any],
    *,
    job_context: Mapping[str, Any],
) -> None:
    for key in (
        "instruction",
        "markdown_text",
        "topic",
        "audience",
        "tone",
        "today",
        "date",
        "target_pages",
        "page_count",
        "target_role_name",
        "role_name",
        "company_name",
        "company",
        "candidate_name",
        "first_name",
        "last_name",
        "job_description",
        "output_dir",
        "document_type",
    ):
        if key in payload:
            continue
        value = job_context.get(key)
        if isinstance(value, str) and value.strip():
            payload[key] = value
            continue
        if isinstance(value, (int, float, bool)):
            payload[key] = value


def build_validation_payload(
    task: models.TaskCreate,
    tool: models.ToolSpec,
    request: planner_contracts.PlanRequest,
    raw_tool_inputs: dict[str, Any],
) -> dict[str, Any]:
    payload = payload_resolver.normalize_reference_payload_for_validation(
        dict(raw_tool_inputs),
        dependency_defaults=dependency_fill_defaults(),
    )
    projected_inputs = job_projection.project_explicit_inputs_for_tool(
        tool.name,
        request.job_payload,
        default_goal=request.goal,
    )
    if projected_inputs:
        payload.pop("job", None)
        for key, value in projected_inputs.items():
            payload.setdefault(key, value)
    payload.setdefault("tool_inputs", dict(raw_tool_inputs))
    if "instruction" not in payload and isinstance(task.instruction, str) and task.instruction.strip():
        payload["instruction"] = task.instruction.strip()
    schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
    if schema_requires_key(schema, "job"):
        payload.setdefault(
            "job",
            job_projection.project_job_payload_for_tool(
                tool.name,
                request.job_payload,
                default_goal=request.goal,
            ),
        )
    if task.deps:
        for key, default_value in dependency_fill_defaults().items():
            payload.setdefault(key, default_value)
    if isinstance(tool.memory_reads, list) and tool.memory_reads:
        payload.setdefault("memory", {})
    _fill_payload_context_defaults(
        payload,
        job_context=request.job_context if isinstance(request.job_context, dict) else {},
    )
    if "today" not in payload and "date" not in payload:
        payload["today"] = datetime.now(UTC).date().isoformat()
    return payload


def build_capability_validation_payload(
    task: models.TaskCreate,
    raw_tool_inputs: dict[str, Any],
    request: planner_contracts.PlanRequest,
) -> dict[str, Any]:
    payload = payload_resolver.normalize_reference_payload_for_validation(
        dict(raw_tool_inputs),
        dependency_defaults=dependency_fill_defaults(),
    )
    projected_inputs = {}
    if task.tool_requests:
        for request_id in task.tool_requests:
            projected_inputs = job_projection.project_explicit_inputs_for_tool(
                request_id,
                request.job_payload,
                default_goal=request.goal,
            )
            if projected_inputs:
                break
    if projected_inputs:
        payload.pop("job", None)
        for key, value in projected_inputs.items():
            payload.setdefault(key, value)
    payload.setdefault("tool_inputs", dict(raw_tool_inputs))
    if task.deps:
        for key, default_value in dependency_fill_defaults().items():
            payload.setdefault(key, default_value)
    job_context = request.job_context if isinstance(request.job_context, dict) else {}
    _fill_payload_context_defaults(payload, job_context=job_context)
    if task.tool_requests and "github.repo.list" in task.tool_requests:
        github_query = synthesize_github_repo_query(
            raw_tool_inputs=raw_tool_inputs,
            payload=payload,
            job_context=job_context,
        )
        if github_query:
            payload["query"] = github_query
    if "today" not in payload and "date" not in payload:
        payload["today"] = datetime.now(UTC).date().isoformat()
    return payload


def validate_capability_inputs(
    capability: Any,
    task: models.TaskCreate,
    raw_tool_inputs: dict[str, Any],
    request: planner_contracts.PlanRequest,
    *,
    schema_registry_path: str,
) -> str | None:
    if not capability.input_schema_ref:
        return None
    schema = load_schema_from_ref(
        capability.input_schema_ref,
        schema_registry_path=schema_registry_path,
    )
    if schema is None:
        return f"capability_schema_not_found:{capability.input_schema_ref}"
    payload = build_capability_validation_payload(task, raw_tool_inputs, request)
    if capability.capability_id == "github.repo.list":
        job_context = request.job_context if isinstance(request.job_context, dict) else {}
        github_query = synthesize_github_repo_query(
            raw_tool_inputs=raw_tool_inputs,
            payload=payload,
            job_context=job_context,
        )
        if github_query:
            payload["query"] = github_query
    errors = payload_resolver.validate_tool_inputs(
        {capability.capability_id: payload},
        {capability.capability_id: schema},
    )
    return errors.get(capability.capability_id)


def validate_plan_request(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
    *,
    schema_registry_path: str,
) -> tuple[bool, str]:
    tool_map = {tool.name: tool for tool in request.tools}
    tool_schemas = {tool.name: tool.input_schema or {} for tool in request.tools}
    capabilities = planner_contracts.capability_map(request)
    goal_intent_segments = planner_contracts.goal_intent_segments(request)
    for task_index, task in enumerate(plan.tasks):
        if not task.tool_requests:
            continue
        task_intent = resolve_task_intent_for_validation(
            task,
            tool_map,
            goal_text=request.goal,
        )
        if not task_intent:
            return False, f"missing_task_intent:{task.name}"
        goal_intent_segment = select_goal_intent_segment_for_task(
            task=task,
            task_index=task_index,
            task_intent=task_intent,
            goal_intent_segments=goal_intent_segments,
            total_tasks=len(plan.tasks),
            capabilities=capabilities,
        )
        for tool_name in task.tool_requests:
            tool = tool_map.get(tool_name)
            capability = capabilities.get(tool_name)
            raw_tool_inputs: dict[str, Any] = {}
            if isinstance(task.tool_inputs, dict) and tool_name in task.tool_inputs:
                entry = task.tool_inputs.get(tool_name)
                if not isinstance(entry, dict):
                    return False, f"tool_inputs_invalid:{tool_name}:{task.name}:payload_not_object"
                raw_tool_inputs = dict(entry)
            segment_payload: dict[str, Any] = raw_tool_inputs
            if tool is not None:
                segment_payload = build_validation_payload(task, tool, request, raw_tool_inputs)
            elif capability is not None:
                segment_payload = build_capability_validation_payload(
                    task,
                    raw_tool_inputs,
                    request,
                )
            segment_contract_error = intent_contract.validate_intent_segment_contract(
                segment=goal_intent_segment,
                task_intent=task_intent,
                tool_name=tool_name,
                payload=segment_payload,
                capability_id=tool_name if capability is not None else None,
                capability_risk_tier=capability.risk_tier if capability is not None else None,
            )
            if segment_contract_error:
                return (
                    False,
                    f"intent_segment_invalid:{tool_name}:{task.name}:{segment_contract_error}",
                )
            if tool is None and capability is None:
                return False, f"unknown_tool_or_capability:{tool_name}"
            if tool is None and capability is not None:
                mismatch = capability_intent_mismatch(task_intent, capability, tool_name)
                if mismatch:
                    return False, f"capability_intent_invalid:{tool_name}:{task.name}:{mismatch}"
                validation_error = validate_capability_inputs(
                    capability,
                    task,
                    raw_tool_inputs,
                    request,
                    schema_registry_path=schema_registry_path,
                )
                if validation_error:
                    return (
                        False,
                        f"capability_inputs_invalid:{tool_name}:{task.name}:{validation_error}",
                    )
                continue
            allow_decision = tool_registry.evaluate_tool_allowlist(
                tool_name,
                "planner",
                context=planner_contracts.governance_context(request),
                tool_spec=tool,
            )
            if not allow_decision.allowed:
                return False, f"tool_not_allowed:{tool_name}:{allow_decision.reason}"
            if allow_decision.violated and allow_decision.mode == "dry_run":
                core_logging.get_logger("planner").warning(
                    "tool_governance_violation_dry_run",
                    tool_name=tool_name,
                    mode=allow_decision.mode,
                    reason=allow_decision.reason,
                    task_name=task.name,
                    job_id=request.job_id,
                )
            mismatch = intent_contract.validate_tool_intent_compatibility(
                task_intent,
                tool.tool_intent,
                tool_name,
            )
            if mismatch:
                return False, f"{mismatch}:task={task.name}"
            validation_payload = build_validation_payload(
                task,
                tool,
                request,
                raw_tool_inputs,
            )
            validation_errors = payload_resolver.validate_tool_inputs(
                {tool_name: validation_payload},
                tool_schemas,
            )
            message = validation_errors.get(tool_name)
            if message:
                return False, f"tool_inputs_invalid:{tool_name}:{task.name}:{message}"
    return True, "ok"


def postprocess_llm_plan(
    plan: models.PlanCreate,
    request: planner_contracts.PlanRequest,
    *,
    runtime: PlannerServiceRuntime,
) -> models.PlanCreate:
    candidate = runtime.ensure_llm_tool(plan)
    candidate = runtime.ensure_task_intents(candidate, request)
    candidate = runtime.ensure_job_inputs(candidate, request)
    candidate = runtime.ensure_default_value_markers(candidate, request)
    candidate = runtime.ensure_renderer_required_inputs(candidate)
    candidate = runtime.ensure_tool_input_dependencies(candidate)
    candidate = runtime.ensure_renderer_output_extensions(candidate)
    candidate = runtime.ensure_execution_bindings(candidate)
    return runtime.apply_max_depth(candidate, request.max_dependency_depth)


def llm_plan(
    request: planner_contracts.PlanRequest,
    provider: llm_provider.LLMProvider,
    *,
    config: PlannerServiceConfig,
    runtime: PlannerServiceRuntime,
) -> models.PlanCreate:
    logger = core_logging.get_logger("planner")
    prompt = build_llm_prompt(request)
    response = provider.generate_request(
        llm_provider.LLMRequest(
            prompt=prompt,
            metadata={
                "component": "planner",
                "operation": "plan_generation",
                "job_id": request.job_id,
                "goal_len": len(request.goal or ""),
                "tool_count": len(request.tools),
            },
        )
    )
    candidate = runtime.parse_llm_plan(response.content)
    if not candidate:
        logger.warning("llm_plan_parse_retry", reason="initial_parse_failed")
        repair_prompt = build_llm_repair_prompt(prompt, response.content, request)
        repaired = provider.generate_request(
            llm_provider.LLMRequest(
                prompt=repair_prompt,
                metadata={
                    "component": "planner",
                    "operation": "plan_generation_repair",
                    "job_id": request.job_id,
                    "goal_len": len(request.goal or ""),
                    "tool_count": len(request.tools),
                },
            )
        )
        candidate = runtime.parse_llm_plan(repaired.content)
    if not candidate:
        raise ValueError("Invalid plan generated: parse_failed")
    candidate = postprocess_llm_plan(candidate, request, runtime=runtime)
    logger.info("llm_plan_candidate", plan=candidate.model_dump())
    valid, reason = validate_plan_request(
        candidate,
        request,
        schema_registry_path=config.schema_registry_path,
    )
    if not valid:
        logger.warning("llm_plan_invalid", reason=reason)
        raise ValueError(f"Invalid plan generated: {reason}")
    return candidate


def plan_job(
    job: models.Job,
    tools: list[models.ToolSpec],
    *,
    provider: llm_provider.LLMProvider | None,
    config: PlannerServiceConfig,
    runtime: PlannerServiceRuntime,
) -> models.PlanCreate:
    request = build_plan_request(
        job,
        tools,
        config=config,
        runtime=runtime,
    )
    if config.mode == "llm":
        if provider is None:
            raise ValueError("LLM planner mode requires a provider")
        return llm_plan(request, provider, config=config, runtime=runtime)
    return rule_based_plan(request)
