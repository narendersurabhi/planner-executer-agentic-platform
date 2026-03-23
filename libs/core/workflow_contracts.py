from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import capability_registry


def _canonicalize_capability_id(value: Any) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    return capability_registry.canonicalize_capability_id(candidate)


def _canonicalize_capability_id_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        canonical = _canonicalize_capability_id(item)
        if canonical and canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


class IntentGraphSlots(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity: str | None = None
    artifact_type: str | None = None
    output_format: str | None = None
    risk_level: str | None = None
    must_have_inputs: list[str] = Field(default_factory=list)


class IntentGraphSegment(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    intent: str
    objective: str = ""
    objective_facts: list[str] = Field(default_factory=list)
    source: str | None = None
    confidence: float | None = None
    depends_on: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    suggested_capabilities: list[str] = Field(default_factory=list)
    suggested_capability_rankings: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_facts: list[str] = Field(default_factory=list)
    slots: IntentGraphSlots = Field(default_factory=IntentGraphSlots)

    @field_validator("suggested_capabilities", mode="before")
    @classmethod
    def _normalize_suggested_capabilities(cls, value: Any) -> list[str] | Any:
        if not isinstance(value, list):
            return value
        return _canonicalize_capability_id_list(value)

    @field_validator("suggested_capability_rankings", mode="before")
    @classmethod
    def _normalize_suggested_capability_rankings(cls, value: Any) -> list[dict[str, Any]] | Any:
        if not isinstance(value, list):
            return value
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, Mapping):
                continue
            entry = dict(item)
            raw_id = entry.get("id")
            if raw_id is None:
                normalized.append(entry)
                continue
            canonical = _canonicalize_capability_id(raw_id)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            entry["id"] = canonical
            normalized.append(entry)
        return normalized


class IntentGraphSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    segment_count: int | None = None
    intent_order: list[str] = Field(default_factory=list)
    fact_candidates: int | None = None
    fact_supported: int | None = None
    fact_stripped: int | None = None
    fact_support_rate: float | None = None
    has_interaction_summaries: bool | None = None
    capability_suggestions_total: int | None = None
    capability_suggestions_matched: int | None = None
    capability_suggestions_selected: int | None = None
    capability_suggestions_autofilled: int | None = None
    capability_top_k: int | None = None
    schema_version: str | None = None
    capability_match_rate: float | None = None
    memory_hints_used: int | None = None
    memory_retrieval_enabled: bool | None = None
    semantic_capability_hints_used: int | None = None
    interaction_summary_compaction: dict[str, Any] | None = None


class IntentGraph(BaseModel):
    model_config = ConfigDict(extra="allow")

    segments: list[IntentGraphSegment] = Field(default_factory=list)
    summary: IntentGraphSummary = Field(default_factory=IntentGraphSummary)
    overall_confidence: float | None = None
    source: str | None = None


class GoalIntentProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    intent: str = ""
    source: str | None = None
    confidence: float | None = None
    risk_level: str | None = None
    threshold: float | None = None
    low_confidence: bool = False
    needs_clarification: bool = False
    requires_blocking_clarification: bool = False
    questions: list[str] = Field(default_factory=list)
    blocking_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    slot_values: dict[str, Any] = Field(default_factory=dict)
    clarification_mode: str | None = None


class ClarificationState(BaseModel):
    model_config = ConfigDict(extra="allow")

    needs_clarification: bool = False
    requires_blocking_clarification: bool = False
    missing_inputs: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    blocking_slots: list[str] = Field(default_factory=list)
    slot_values: dict[str, Any] = Field(default_factory=dict)
    clarification_mode: str | None = None


class NormalizationTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    assessment_source: str | None = None
    assessment_mode: str | None = None
    assessment_model: str | None = None
    assessment_fallback_used: bool | None = None
    decomposition_source: str | None = None
    decomposition_mode: str | None = None
    decomposition_model: str | None = None
    decomposition_fallback_used: bool | None = None


class NormalizedIntentEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "intent_envelope_v1"
    goal: str = ""
    profile: GoalIntentProfile = Field(default_factory=GoalIntentProfile)
    graph: IntentGraph = Field(default_factory=IntentGraph)
    candidate_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    clarification: ClarificationState = Field(default_factory=ClarificationState)
    trace: NormalizationTrace = Field(default_factory=NormalizationTrace)

    @field_validator("candidate_capabilities", mode="before")
    @classmethod
    def _normalize_candidate_capabilities(
        cls,
        value: Any,
    ) -> dict[str, list[str]] | Any:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[str, list[str]] = {}
        for segment_id, capability_ids in value.items():
            normalized_ids = _canonicalize_capability_id_list(capability_ids)
            if normalized_ids:
                normalized[str(segment_id)] = normalized_ids
        return normalized


def parse_intent_graph(value: Any) -> IntentGraph | None:
    if isinstance(value, IntentGraph):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return IntentGraph.model_validate(value)
    except ValidationError:
        return None


def dump_intent_graph(value: IntentGraph | Mapping[str, Any] | None) -> dict[str, Any] | None:
    graph = parse_intent_graph(value)
    if graph is None:
        return None
    return graph.model_dump(mode="json", exclude_none=True)


def parse_goal_intent_profile(value: Any) -> GoalIntentProfile | None:
    if isinstance(value, GoalIntentProfile):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return GoalIntentProfile.model_validate(value)
    except ValidationError:
        return None


def dump_goal_intent_profile(
    value: GoalIntentProfile | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    profile = parse_goal_intent_profile(value)
    if profile is None:
        return None
    return profile.model_dump(mode="json", exclude_none=True)


def parse_normalized_intent_envelope(value: Any) -> NormalizedIntentEnvelope | None:
    if isinstance(value, NormalizedIntentEnvelope):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        return NormalizedIntentEnvelope.model_validate(value)
    except ValidationError:
        return None


def dump_normalized_intent_envelope(
    value: NormalizedIntentEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    envelope = parse_normalized_intent_envelope(value)
    if envelope is None:
        return None
    return envelope.model_dump(mode="json", exclude_none=True)
