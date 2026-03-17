from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
