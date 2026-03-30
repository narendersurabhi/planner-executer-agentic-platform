from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from libs.core import capability_registry, intent_contract, models, workflow_contracts


RENDER_PATH_MODE_AUTO = "auto"
RENDER_PATH_MODE_EXPLICIT = "explicit"
PLANNER_CAPABILITY_LANGUAGE_MODE_ENFORCE = "enforce"
PLANNER_CAPABILITY_LANGUAGE_MODE_COMPAT = "compat"

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
_PLANNER_CONTEXT_NOISE_TOKENS = {
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "go ahead",
    "sounds good",
    "looks good",
    "use default",
}
_PLANNER_INTERACTION_SUMMARY_LIMIT = 4
_PLANNER_CAPABILITY_CANDIDATE_LIMIT = 10


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
    planner_tools: list[models.ToolSpec] = Field(default_factory=list)
    capabilities: list[PlanRequestCapability] = Field(default_factory=list)
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | None = None
    goal_intent_graph: workflow_contracts.IntentGraph | None = None
    semantic_capability_hints: list[dict[str, Any]] = Field(default_factory=list)
    max_dependency_depth: int | None = None
    render_path_mode: str = RENDER_PATH_MODE_EXPLICIT


@dataclass(frozen=True)
class CompiledTaskRequests:
    request_ids: list[str]
    request_id_rewrites: dict[str, str]
    tool_inputs: dict[str, Any]
    capability_bindings: dict[str, dict[str, Any]]


def resolve_planner_capability_language_mode() -> str:
    raw = (
        os.getenv(
            "PLANNER_CAPABILITY_LANGUAGE_MODE",
            PLANNER_CAPABILITY_LANGUAGE_MODE_ENFORCE,
        )
        .strip()
        .lower()
    )
    if raw == PLANNER_CAPABILITY_LANGUAGE_MODE_COMPAT:
        return PLANNER_CAPABILITY_LANGUAGE_MODE_COMPAT
    return PLANNER_CAPABILITY_LANGUAGE_MODE_ENFORCE


def planner_capability_language_enforced() -> bool:
    return resolve_planner_capability_language_mode() == PLANNER_CAPABILITY_LANGUAGE_MODE_ENFORCE


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


def project_planner_job_context(job: models.Job) -> dict[str, Any]:
    projected = dict(job.context_json) if isinstance(job.context_json, Mapping) else {}
    for key in (
        "user_profile",
        "interaction_summaries_ref",
        "interaction_summaries_meta",
    ):
        projected.pop(key, None)

    interaction_summaries = _planner_interaction_summaries(projected)
    if interaction_summaries:
        projected["interaction_summaries"] = interaction_summaries[:_PLANNER_INTERACTION_SUMMARY_LIMIT]
    else:
        projected.pop("interaction_summaries", None)

    normalized_intent_envelope = normalized_intent_envelope_for_job(job)
    capability_candidates = _planner_capability_candidates(projected, normalized_intent_envelope)
    if capability_candidates:
        projected["capability_candidates"] = capability_candidates[:_PLANNER_CAPABILITY_CANDIDATE_LIMIT]
    else:
        projected.pop("capability_candidates", None)

    missing_inputs = _planner_missing_inputs(projected, normalized_intent_envelope)
    if missing_inputs:
        projected["missing_inputs"] = missing_inputs
    else:
        projected.pop("missing_inputs", None)
    return projected


def build_plan_request(
    job: models.Job,
    *,
    tools: Sequence[models.ToolSpec],
    planner_tools: Sequence[models.ToolSpec] | None = None,
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
        job_context=project_planner_job_context(job),
        job_metadata=dict(job_metadata),
        job_payload=job.model_dump(mode="json"),
        tools=list(tools),
        planner_tools=list(planner_tools or []),
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


def _planner_interaction_summaries(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("interaction_summaries")
    if not isinstance(raw, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        summary = dict(item)
        if _planner_interaction_summary_is_noisy(summary):
            continue
        summaries.append(summary)
    return summaries


def _planner_interaction_summary_is_noisy(item: Mapping[str, Any]) -> bool:
    text = _planner_interaction_summary_text(item)
    if not text:
        return True
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not normalized:
        return True
    if normalized in _PLANNER_CONTEXT_NOISE_TOKENS:
        return True
    tokens = [token for token in normalized.split() if token]
    return bool(
        tokens
        and all(token in {"yes", "yeah", "yep", "ok", "okay", "thanks", "thank", "you"} for token in tokens)
    )


def _planner_interaction_summary_text(item: Mapping[str, Any]) -> str:
    fields: list[str] = []
    for key in ("action", "facts", "evidence", "speculation"):
        value = item.get(key)
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, list):
            fields.extend(str(entry) for entry in value if isinstance(entry, str))
    return " ".join(fields).strip().lower()


def _planner_capability_candidates(
    context: Mapping[str, Any],
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | None,
) -> list[str]:
    capability_ids: list[str] = []
    seen: set[str] = set()
    raw = context.get("capability_candidates")
    if isinstance(raw, list):
        for capability_id in raw:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
    if normalized_intent_envelope is not None:
        for capability_list in normalized_intent_envelope.candidate_capabilities.values():
            for capability_id in capability_list:
                normalized = capability_registry.canonicalize_capability_id(capability_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    capability_ids.append(normalized)
        for segment in normalized_intent_envelope.graph.segments:
            for capability_id in segment.suggested_capabilities:
                normalized = capability_registry.canonicalize_capability_id(capability_id)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    capability_ids.append(normalized)
    return capability_ids


def _planner_missing_inputs(
    context: Mapping[str, Any],
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | None,
) -> list[str]:
    if normalized_intent_envelope is None:
        existing = context.get("missing_inputs")
        if isinstance(existing, list):
            return [
                normalized
                for normalized in (
                    intent_contract.normalize_required_input_key(item) for item in existing
                )
                if normalized
            ]
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for value in normalized_intent_envelope.profile.missing_slots:
        normalized = intent_contract.normalize_required_input_key(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    for value in normalized_intent_envelope.clarification.missing_inputs:
        normalized = intent_contract.normalize_required_input_key(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)
    unresolved: list[str] = []
    for key in candidates:
        if not _planner_context_has_required_input(context, key):
            unresolved.append(key)
    return unresolved


def _planner_context_has_required_input(context: Mapping[str, Any], key: str) -> bool:
    normalized = intent_contract.normalize_required_input_key(key)
    if not normalized:
        return False
    aliases: dict[str, tuple[str, ...]] = {
        "path": ("output_path", "filename", "file_name", "output_filename"),
        "output_path": ("path", "filename", "file_name", "output_filename"),
        "filename": ("path", "output_path", "file_name", "output_filename"),
        "query": ("path", "source"),
        "instruction": ("goal_details", "goal"),
        "topic": ("main_topic", "title", "subject"),
        "main_topic": ("topic", "title", "subject"),
        "target_repo": ("repo", "repo_name", "repo_full_name", "query"),
    }
    for candidate in (normalized,) + aliases.get(normalized, ()):
        if _planner_mapping_has_non_empty_value(context, candidate):
            return True
        workflow = context.get("workflow")
        if isinstance(workflow, Mapping):
            workflow_inputs = workflow.get("inputs")
            if isinstance(workflow_inputs, Mapping) and _planner_mapping_has_non_empty_value(
                workflow_inputs,
                candidate,
            ):
                return True
    return False


def _planner_mapping_has_non_empty_value(mapping: Mapping[str, Any], key: str) -> bool:
    if key not in mapping:
        return False
    value = mapping.get(key)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def capability_map(request: PlanRequest) -> dict[str, PlanRequestCapability]:
    mapped: dict[str, PlanRequestCapability] = {}
    for capability in request.capabilities:
        mapped[capability.capability_id] = capability
        for alias in capability.aliases:
            normalized = str(alias or "").strip()
            if normalized:
                mapped[normalized] = capability
    return mapped


def _capability_values(
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None,
) -> list[Any]:
    if isinstance(capabilities, PlanRequest):
        return list(capabilities.capabilities)
    if isinstance(capabilities, Mapping):
        return list(capabilities.values())
    if isinstance(capabilities, Sequence) and not isinstance(capabilities, (str, bytes, bytearray)):
        return list(capabilities)
    return []


def _full_capability_values(
    full_capabilities: capability_registry.CapabilityRegistry
    | Mapping[str, Any]
    | Sequence[Any]
    | PlanRequest
    | None,
) -> list[Any]:
    if isinstance(full_capabilities, capability_registry.CapabilityRegistry):
        return list(full_capabilities.capabilities.values())
    if full_capabilities is None:
        try:
            return list(capability_registry.load_capability_registry().capabilities.values())
        except Exception:  # noqa: BLE001
            return []
    return _capability_values(full_capabilities)


def _register_unique_mapping(
    index: dict[str, str],
    ambiguous: set[str],
    key: str,
    capability_id: str,
) -> None:
    if not key:
        return
    existing = index.get(key)
    if existing and existing != capability_id:
        ambiguous.add(key)
        index.pop(key, None)
        return
    if key not in ambiguous:
        index[key] = capability_id


def _capability_index(
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None,
) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for capability in _capability_values(capabilities):
        capability_id = str(getattr(capability, "capability_id", "") or "").strip()
        if capability_id and capability_id not in indexed:
            indexed[capability_id] = capability
    return indexed


def canonicalize_planner_request_id(
    request_id: str,
    *,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
) -> str:
    candidate = str(request_id or "").strip()
    if not candidate:
        return ""
    capability_values = _capability_values(capabilities)
    if not capability_values:
        return candidate

    exact_lookup: dict[str, str] = {}
    lower_lookup: dict[str, str] = {}
    adapter_lookup: dict[str, str] = {}
    adapter_lower_lookup: dict[str, str] = {}
    ambiguous_adapter_keys: set[str] = set()
    ambiguous_adapter_lower_keys: set[str] = set()

    for capability in capability_values:
        capability_id = str(getattr(capability, "capability_id", "") or "").strip()
        if not capability_id:
            continue
        exact_lookup.setdefault(capability_id, capability_id)
        lower_lookup.setdefault(capability_id.lower(), capability_id)
        for alias in getattr(capability, "aliases", ()) or ():
            normalized_alias = str(alias or "").strip()
            if not normalized_alias:
                continue
            exact_lookup.setdefault(normalized_alias, capability_id)
            lower_lookup.setdefault(normalized_alias.lower(), capability_id)
        for adapter in getattr(capability, "adapters", ()) or ():
            if getattr(adapter, "enabled", True) is False:
                continue
            tool_name = str(getattr(adapter, "tool_name", "") or "").strip()
            if not tool_name:
                continue
            _register_unique_mapping(
                adapter_lookup,
                ambiguous_adapter_keys,
                tool_name,
                capability_id,
            )
            _register_unique_mapping(
                adapter_lower_lookup,
                ambiguous_adapter_lower_keys,
                tool_name.lower(),
                capability_id,
            )

    return (
        exact_lookup.get(candidate)
        or lower_lookup.get(candidate.lower())
        or adapter_lookup.get(candidate)
        or adapter_lower_lookup.get(candidate.lower())
        or candidate
    )


def canonicalize_planner_request_ids(
    request_ids: Sequence[str] | None,
    *,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
) -> tuple[list[str], dict[str, str]]:
    canonicalized: list[str] = []
    rewrites: dict[str, str] = {}
    seen: set[str] = set()
    for request_id in request_ids or []:
        raw = str(request_id or "").strip()
        if not raw:
            continue
        canonical = canonicalize_planner_request_id(raw, capabilities=capabilities)
        if canonical and canonical not in seen:
            canonicalized.append(canonical)
            seen.add(canonical)
        if canonical and canonical != raw:
            rewrites[raw] = canonical
    return canonicalized, rewrites


def planner_task_request_ids(task: Any) -> list[str]:
    request_ids = getattr(task, "capability_requests", None) or getattr(task, "tool_requests", None) or []
    normalized: list[str] = []
    seen: set[str] = set()
    for request_id in request_ids:
        raw = str(request_id or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        normalized.append(raw)
    return normalized


def validate_planner_request_language(
    request_id: str,
    *,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
    full_capabilities: capability_registry.CapabilityRegistry
    | Mapping[str, Any]
    | Sequence[Any]
    | PlanRequest
    | None = None,
    runtime_tool_names: Sequence[str] | None = None,
    mode: str | None = None,
) -> str | None:
    normalized_request_id = str(request_id or "").strip()
    if not normalized_request_id:
        return None
    resolved_mode = str(mode or resolve_planner_capability_language_mode()).strip().lower()
    if resolved_mode == PLANNER_CAPABILITY_LANGUAGE_MODE_COMPAT:
        return None

    allowed_capabilities = _capability_index(capabilities)
    full_capability_values = _full_capability_values(
        full_capabilities if full_capabilities is not None else capabilities
    )
    resolved_capability = resolve_planner_capability(
        normalized_request_id,
        capabilities=full_capability_values,
    )
    if resolved_capability is not None:
        canonical_id = str(getattr(resolved_capability, "capability_id", "") or "").strip()
        if not canonical_id:
            return None
        if getattr(resolved_capability, "enabled", True) is False:
            return f"planner_request_capability_disabled:{canonical_id}"
        if allowed_capabilities and canonical_id not in allowed_capabilities:
            return f"planner_request_capability_not_allowed:{canonical_id}"
        if normalized_request_id != canonical_id:
            return (
                f"planner_request_language_invalid:{normalized_request_id}:"
                f"use_capability_id:{canonical_id}"
            )
        return None

    runtime_tool_set = {
        str(tool_name or "").strip()
        for tool_name in (runtime_tool_names or [])
        if str(tool_name or "").strip()
    }
    if normalized_request_id in runtime_tool_set:
        return None
    return f"planner_request_unknown_capability:{normalized_request_id}"


def resolve_planner_capability(
    capability_id: str,
    *,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
) -> Any | None:
    canonical = canonicalize_planner_request_id(capability_id, capabilities=capabilities)
    if not canonical:
        return None
    return _capability_index(capabilities).get(canonical)


def resolve_primary_adapter(capability: Any) -> Any | None:
    adapters = getattr(capability, "adapters", ()) if capability is not None else ()
    if not isinstance(adapters, Sequence):
        return None
    for adapter in adapters:
        if getattr(adapter, "enabled", True) is False:
            continue
        tool_name = str(getattr(adapter, "tool_name", "") or "").strip()
        if tool_name:
            return adapter
    return None


def compile_capability_request(
    request_id: str,
    *,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
) -> dict[str, Any]:
    canonical_request_id = canonicalize_planner_request_id(request_id, capabilities=capabilities)
    capability = resolve_planner_capability(canonical_request_id, capabilities=capabilities)
    if capability is None:
        return {
            "request_id": canonical_request_id or str(request_id or "").strip(),
            "capability_id": None,
            "tool_name": canonical_request_id or str(request_id or "").strip(),
            "adapter_type": None,
            "server_id": None,
        }
    adapter = resolve_primary_adapter(capability)
    if adapter is None:
        return {
            "request_id": capability.capability_id,
            "capability_id": capability.capability_id,
            "tool_name": None,
            "adapter_type": None,
            "server_id": None,
        }
    tool_name = str(getattr(adapter, "tool_name", "") or "").strip() or capability.capability_id
    return {
        "request_id": tool_name,
        "capability_id": capability.capability_id,
        "tool_name": tool_name,
        "adapter_type": str(getattr(adapter, "type", "") or "").strip() or None,
        "server_id": str(getattr(adapter, "server_id", "") or "").strip() or None,
    }


def rewrite_request_keyed_mapping(
    value: Mapping[str, Any] | None,
    rewrites: Mapping[str, str] | None,
) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    rewrite_map = dict(rewrites or {})
    for request_id, payload in (value or {}).items():
        raw_request_id = str(request_id or "").strip()
        if not raw_request_id:
            continue
        target_request_id = rewrite_map.get(raw_request_id, raw_request_id)
        if target_request_id in rewritten:
            continue
        rewritten[target_request_id] = payload
    return rewritten


def compile_task_request_payloads(
    *,
    capability_requests: Sequence[str] | None = None,
    tool_requests: Sequence[str] | None = None,
    tool_inputs: Mapping[str, Any] | None = None,
    capability_bindings: Mapping[str, Any] | None = None,
    capabilities: Mapping[str, Any] | Sequence[Any] | PlanRequest | None = None,
) -> CompiledTaskRequests:
    source_request_ids = [
        str(request_id).strip()
        for request_id in (capability_requests or tool_requests or [])
        if str(request_id).strip()
    ]
    compiled_request_ids: list[str] = []
    compiled_bindings: dict[str, dict[str, Any]] = {}
    request_id_rewrites: dict[str, str] = {}
    seen: set[str] = set()
    for request_id in source_request_ids:
        compiled = compile_capability_request(request_id, capabilities=capabilities)
        compiled_request_id = str(compiled.get("request_id") or "").strip()
        if not compiled_request_id:
            continue
        if compiled_request_id not in seen:
            compiled_request_ids.append(compiled_request_id)
            seen.add(compiled_request_id)
        if compiled_request_id != request_id:
            request_id_rewrites[request_id] = compiled_request_id
        capability_id = str(compiled.get("capability_id") or "").strip()
        if capability_id:
            compiled_bindings[compiled_request_id] = {
                key: value
                for key, value in {
                    "request_id": compiled_request_id,
                    "capability_id": capability_id,
                    "tool_name": compiled.get("tool_name"),
                    "adapter_type": compiled.get("adapter_type"),
                    "server_id": compiled.get("server_id"),
                }.items()
                if value is not None and value != ""
            }

    rewritten_inputs = rewrite_request_keyed_mapping(tool_inputs, request_id_rewrites)
    rewritten_bindings = rewrite_request_keyed_mapping(capability_bindings, request_id_rewrites)
    rewritten_bindings.update(compiled_bindings)
    if compiled_request_ids:
        rewritten_bindings = {
            request_id: binding
            for request_id in compiled_request_ids
            if isinstance((binding := rewritten_bindings.get(request_id)), Mapping)
        }

    return CompiledTaskRequests(
        request_ids=compiled_request_ids or source_request_ids,
        request_id_rewrites=request_id_rewrites,
        tool_inputs=rewritten_inputs,
        capability_bindings={key: dict(value) for key, value in rewritten_bindings.items()},
    )


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
