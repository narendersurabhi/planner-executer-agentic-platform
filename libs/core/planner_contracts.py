from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from libs.core import capability_registry, intent_contract, models, workflow_contracts


RENDER_PATH_MODE_AUTO = "auto"
RENDER_PATH_MODE_EXPLICIT = "explicit"

_RENDER_REQUEST_IDS = frozenset(
    {
        "docx_render_from_spec",
        "docx_generate_from_spec",
        "pdf_render_from_spec",
        "pdf_generate_from_spec",
        "document.docx.render",
        "document.docx.generate",
        "document.pdf.render",
        "document.pdf.generate",
    }
)
_RENDER_PATH_KEYS = ("path", "output_path", "filename", "file_name", "output_filename")


class PlanRequestCapabilityAdapter(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    server_id: str | None = None
    tool_name: str | None = None


class PlanRequestCapability(BaseModel):
    model_config = ConfigDict(extra="allow")

    capability_id: str
    description: str = ""
    risk_tier: str = "low"
    idempotency: str = "unknown"
    group: str | None = None
    subgroup: str | None = None
    input_schema_ref: str | None = None
    output_schema_ref: str | None = None
    aliases: list[str] = Field(default_factory=list)
    planner_hints: dict[str, Any] = Field(default_factory=dict)
    adapters: list[PlanRequestCapabilityAdapter] = Field(default_factory=list)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    goal: str
    job_context: dict[str, Any] = Field(default_factory=dict)
    job_metadata: dict[str, Any] = Field(default_factory=dict)
    job_payload: dict[str, Any] = Field(default_factory=dict)
    tools: list[models.ToolSpec] = Field(default_factory=list)
    capabilities: list[PlanRequestCapability] = Field(default_factory=list)
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | None = None
    goal_intent_graph: workflow_contracts.IntentGraph | None = None
    semantic_capability_hints: list[dict[str, Any]] = Field(default_factory=list)
    max_dependency_depth: int | None = None
    render_path_mode: str = RENDER_PATH_MODE_EXPLICIT


def normalize_render_path_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == RENDER_PATH_MODE_AUTO:
        return RENDER_PATH_MODE_AUTO
    return RENDER_PATH_MODE_EXPLICIT


def render_path_mode_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return RENDER_PATH_MODE_EXPLICIT
    return normalize_render_path_mode(metadata.get("render_path_mode"))


def render_path_mode(request: PlanRequest) -> str:
    return normalize_render_path_mode(getattr(request, "render_path_mode", None))


def is_render_request_id(request_id: str | None) -> bool:
    normalized = str(request_id or "").strip()
    return normalized in _RENDER_REQUEST_IDS


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _mapping_has_explicit_render_path(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(_non_empty_string(payload.get(key)) for key in _RENDER_PATH_KEYS)


def _job_context_has_explicit_render_path(job_context: Mapping[str, Any] | None) -> bool:
    if not isinstance(job_context, Mapping):
        return False
    return any(_non_empty_string(job_context.get(key)) for key in _RENDER_PATH_KEYS)


def _job_context_has_explicit_render_path_key(
    job_context: Mapping[str, Any] | None,
    key: str,
) -> bool:
    if not isinstance(job_context, Mapping):
        return False
    normalized = str(key or "").strip()
    if normalized == "path":
        return bool(
            _non_empty_string(job_context.get("path"))
            or _non_empty_string(job_context.get("output_path"))
        )
    if normalized == "output_path":
        return bool(
            _non_empty_string(job_context.get("output_path"))
            or _non_empty_string(job_context.get("path"))
        )
    return _non_empty_string(job_context.get(normalized)) is not None


def _reference_segments(path_spec: Any) -> list[str] | None:
    segments: list[str] = []
    if isinstance(path_spec, (list, tuple)):
        for item in path_spec:
            normalized = _non_empty_string(item)
            if normalized:
                segments.append(normalized)
    elif isinstance(path_spec, str):
        raw = path_spec.strip()
        if not raw:
            return None
        if raw.startswith("$."):
            raw = raw[2:]
        if raw.startswith("/"):
            segments = [
                item.replace("~1", "/").replace("~0", "~")
                for item in raw.split("/")[1:]
                if item
            ]
        else:
            segments = [item for item in raw.split(".") if item]
    else:
        return None
    return segments or None


def _reference_has_default_path(value: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, Mapping):
        return False
    return _non_empty_string(value.get("$default")) is not None


def validate_render_path_requirement(
    *,
    request_id: str,
    raw_payload: Mapping[str, Any] | None,
    resolved_payload: Mapping[str, Any] | None = None,
    job_context: Mapping[str, Any] | None = None,
    render_path_mode: str | None = None,
) -> str | None:
    if not is_render_request_id(request_id):
        return None
    if normalize_render_path_mode(render_path_mode) != RENDER_PATH_MODE_EXPLICIT:
        return None

    if _mapping_has_explicit_render_path(raw_payload):
        return None

    if isinstance(raw_payload, Mapping):
        for key in _RENDER_PATH_KEYS:
            raw_value = raw_payload.get(key)
            if not isinstance(raw_value, Mapping):
                continue
            segments = _reference_segments(raw_value.get("$from"))
            if not segments:
                continue
            if segments[0] == "job_context":
                if len(segments) >= 2 and segments[1] in _RENDER_PATH_KEYS:
                    if _job_context_has_explicit_render_path_key(
                        job_context,
                        segments[1],
                    ) or _reference_has_default_path(raw_value):
                        return None
                return f"render_path_explicit_required:{request_id}"
            return f"render_path_derived_not_allowed:{request_id}"

    if _job_context_has_explicit_render_path(job_context):
        return None

    if _mapping_has_explicit_render_path(resolved_payload):
        return f"render_path_explicit_required:{request_id}"

    return f"render_path_explicit_required:{request_id}"


def _candidate_capabilities_from_graph(
    graph: workflow_contracts.IntentGraph | None,
) -> dict[str, list[str]]:
    if graph is None:
        return {}
    candidates: dict[str, list[str]] = {}
    for segment in graph.segments:
        deduped: list[str] = []
        for capability_id in segment.suggested_capabilities:
            normalized = str(capability_id or "").strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        if deduped:
            candidates[segment.id] = deduped
    return candidates


def normalized_intent_envelope_for_job(
    job: models.Job,
) -> workflow_contracts.NormalizedIntentEnvelope | None:
    metadata = job.metadata if isinstance(job.metadata, Mapping) else {}
    envelope = workflow_contracts.parse_normalized_intent_envelope(
        metadata.get("normalized_intent_envelope")
    )
    if envelope is not None:
        return envelope
    profile = workflow_contracts.parse_goal_intent_profile(metadata.get("goal_intent_profile"))
    graph = workflow_contracts.parse_intent_graph(metadata.get("goal_intent_graph"))
    if profile is None and graph is None:
        return None
    graph = graph or workflow_contracts.IntentGraph()
    profile = profile or workflow_contracts.GoalIntentProfile()
    return workflow_contracts.NormalizedIntentEnvelope(
        goal=str(job.goal or "").strip(),
        profile=profile,
        graph=graph,
        candidate_capabilities=_candidate_capabilities_from_graph(graph),
        clarification=workflow_contracts.ClarificationState(
            needs_clarification=bool(profile.needs_clarification),
            requires_blocking_clarification=bool(profile.requires_blocking_clarification),
            missing_inputs=list(profile.missing_slots),
            questions=list(profile.questions),
            blocking_slots=list(profile.blocking_slots),
            slot_values=dict(profile.slot_values),
            clarification_mode=profile.clarification_mode,
        ),
    )


def build_plan_request(
    job: models.Job,
    *,
    tools: Sequence[models.ToolSpec],
    capabilities: Mapping[str, capability_registry.CapabilitySpec]
    | Sequence[capability_registry.CapabilitySpec]
    | None = None,
    semantic_capability_hints: Sequence[Mapping[str, Any]] | None = None,
    max_dependency_depth: int | None = None,
) -> PlanRequest:
    capability_values: Sequence[capability_registry.CapabilitySpec]
    if isinstance(capabilities, Mapping):
        capability_values = list(capabilities.values())
    elif capabilities is None:
        capability_values = []
    else:
        capability_values = list(capabilities)
    job_metadata = job.metadata if isinstance(job.metadata, dict) else {}
    normalized_intent_envelope = normalized_intent_envelope_for_job(job)
    goal_intent_graph = None
    if normalized_intent_envelope is not None and normalized_intent_envelope.graph.segments:
        goal_intent_graph = normalized_intent_envelope.graph
    if goal_intent_graph is None:
        goal_intent_graph = workflow_contracts.parse_intent_graph(job_metadata.get("goal_intent_graph"))
    return PlanRequest(
        job_id=job.id,
        goal=job.goal,
        job_context=job.context_json if isinstance(job.context_json, dict) else {},
        job_metadata=dict(job_metadata),
        job_payload=job.model_dump(mode="json"),
        tools=list(tools),
        capabilities=[
            PlanRequestCapability(
                capability_id=spec.capability_id,
                description=spec.description,
                risk_tier=spec.risk_tier,
                idempotency=spec.idempotency,
                group=spec.group,
                subgroup=spec.subgroup,
                input_schema_ref=spec.input_schema_ref,
                output_schema_ref=spec.output_schema_ref,
                aliases=list(spec.aliases),
                planner_hints=(
                    dict(spec.planner_hints) if isinstance(spec.planner_hints, dict) else {}
                ),
                adapters=[
                    PlanRequestCapabilityAdapter(
                        type=adapter.type,
                        server_id=adapter.server_id,
                        tool_name=adapter.tool_name,
                    )
                    for adapter in spec.adapters
                    if adapter.enabled
                ],
            )
            for spec in capability_values
        ],
        normalized_intent_envelope=normalized_intent_envelope,
        goal_intent_graph=goal_intent_graph,
        semantic_capability_hints=[
            dict(item) for item in semantic_capability_hints or [] if isinstance(item, Mapping)
        ],
        max_dependency_depth=max_dependency_depth,
        render_path_mode=render_path_mode_from_metadata(job_metadata),
    )


def capability_map(request: PlanRequest) -> dict[str, PlanRequestCapability]:
    mapped: dict[str, PlanRequestCapability] = {}
    for capability in request.capabilities:
        mapped[capability.capability_id] = capability
        for alias in capability.aliases:
            normalized = str(alias or "").strip()
            if normalized:
                mapped[normalized] = capability
    return mapped


def normalized_intent_envelope(
    request: PlanRequest,
) -> workflow_contracts.NormalizedIntentEnvelope | None:
    return request.normalized_intent_envelope


def normalized_intent_graph(
    request: PlanRequest,
) -> workflow_contracts.IntentGraph | None:
    envelope = normalized_intent_envelope(request)
    if envelope is not None and envelope.graph.segments:
        return envelope.graph
    return request.goal_intent_graph


def goal_intent_sequence(request: PlanRequest) -> list[str]:
    graph = normalized_intent_graph(request)
    if graph is None:
        return []
    sequence: list[str] = []
    for segment in graph.segments:
        normalized = intent_contract.normalize_task_intent(segment.intent)
        if normalized:
            sequence.append(normalized)
    return sequence


def goal_intent_segments(request: PlanRequest) -> list[dict[str, Any]]:
    graph = normalized_intent_graph(request)
    if graph is None:
        return []
    return [segment.model_dump(mode="json", exclude_none=True) for segment in graph.segments]


def intent_mismatch_recovery(request: PlanRequest) -> dict[str, Any] | None:
    raw = request.job_metadata.get("intent_mismatch_recovery")
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def governance_context(request: PlanRequest) -> dict[str, Any]:
    metadata = request.job_metadata if isinstance(request.job_metadata, dict) else {}
    return {
        "job_id": request.job_id,
        "job_type": metadata.get("job_type"),
        "tenant_id": metadata.get("tenant_id"),
        "org_id": metadata.get("org_id"),
        "job_context": request.job_context if isinstance(request.job_context, dict) else {},
    }
