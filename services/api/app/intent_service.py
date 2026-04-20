from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.orm import Session

from libs.core import intent_contract, workflow_contracts


@dataclass(frozen=True)
class IntentDecomposeConfig:
    enabled: bool
    mode: str
    capability_top_k: int
    memory_retrieval_enabled: bool
    memory_retrieval_limit: int


@dataclass(frozen=True)
class IntentDecomposeRuntime:
    provider: Any
    heuristic_decompose: Callable[[str], dict[str, Any]]
    capability_entries: Callable[[], list[dict[str, str]]]
    capability_ids: Callable[[], set[str]]
    normalize_user_id: Callable[[str | None], str]
    retrieve_workflow_hints: Callable[[Session | None, str, str, int], list[dict[str, Any]]]
    semantic_goal_capability_hints: Callable[[str, list[dict[str, str]], int], list[dict[str, Any]]]
    llm_decompose: Callable[..., dict[str, Any]]
    annotate_graph_summary_defaults: Callable[[dict[str, Any]], dict[str, Any]]
    apply_supported_fact_filter: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]]
    record_metrics: Callable[[dict[str, Any], str, bool], None]
    on_llm_failure: Callable[[Exception], None]


@dataclass(frozen=True)
class GoalIntentConfig:
    min_confidence: float
    min_confidence_by_intent: dict[str, float]
    min_confidence_by_risk: dict[str, float]
    clarification_blocking_slots: set[str]


@dataclass(frozen=True)
class GoalIntentRuntime:
    infer_task_intent: Callable[[str], Any]
    record_metrics: Callable[[workflow_contracts.GoalIntentProfile], None] | None = None


@dataclass(frozen=True)
class IntentNormalizeConfig:
    include_decomposition: bool
    assessment_mode: str
    assessment_model: str
    decomposition_mode: str
    decomposition_model: str


@dataclass(frozen=True)
class IntentNormalizeRuntime:
    assess_goal_intent: Callable[[str], workflow_contracts.GoalIntentProfile]
    decompose_goal_intent: Callable[..., workflow_contracts.IntentGraph]
    capability_required_inputs: Callable[[str], list[str]]
    assess_goal_intent_heuristic: Callable[[str], workflow_contracts.GoalIntentProfile] | None = None


def normalize_goal_intent(
    goal: str,
    *,
    db: Session | None = None,
    user_id: str | None = None,
    interaction_summaries: list[dict[str, Any]] | None = None,
    intent_context: dict[str, Any] | None = None,
    config: IntentNormalizeConfig,
    runtime: IntentNormalizeRuntime,
) -> workflow_contracts.NormalizedIntentEnvelope:
    normalized_intent_context = dict(intent_context or {})
    context_interaction_summaries = normalized_intent_context.get("interaction_summaries")
    if interaction_summaries is None and isinstance(context_interaction_summaries, list):
        interaction_summaries = [
            dict(item) for item in context_interaction_summaries if isinstance(item, dict)
        ]
    initial_profile = runtime.assess_goal_intent(goal)
    heuristic_profile = (
        runtime.assess_goal_intent_heuristic(goal)
        if runtime.assess_goal_intent_heuristic is not None
        else initial_profile
    )
    context_slot_values = (
        dict(normalized_intent_context.get("intent_slot_values"))
        if isinstance(normalized_intent_context.get("intent_slot_values"), dict)
        else {}
    )
    context_slot_provenance = (
        dict(normalized_intent_context.get("intent_slot_provenance"))
        if isinstance(normalized_intent_context.get("intent_slot_provenance"), dict)
        else {}
    )
    if context_slot_values:
        merged_slot_values = dict(initial_profile.slot_values or {})
        updated = False
        for key, value in context_slot_values.items():
            if key in merged_slot_values and str(merged_slot_values.get(key) or "").strip():
                continue
            if isinstance(value, str):
                if not value.strip():
                    continue
                merged_slot_values[key] = value.strip()
            elif value is not None:
                merged_slot_values[key] = value
            updated = True
        if updated:
            blocking_slots = list(initial_profile.blocking_slots or [])
            missing_slots = [
                slot_name
                for slot_name in blocking_slots
                if not str(merged_slot_values.get(slot_name) or "").strip()
            ]
            if (
                initial_profile.low_confidence
                and "intent_action" in blocking_slots
                and "intent_action" not in missing_slots
            ):
                missing_slots.append("intent_action")
            initial_profile = initial_profile.model_copy(
                update={
                    "slot_values": merged_slot_values,
                    "missing_slots": missing_slots,
                    "needs_clarification": bool(missing_slots),
                    "requires_blocking_clarification": bool(missing_slots),
                    "questions": [slot_question(slot_name, goal) for slot_name in missing_slots],
                }
            )
    graph = workflow_contracts.IntentGraph()
    if config.include_decomposition:
        graph = runtime.decompose_goal_intent(
            goal,
            db=db,
            user_id=user_id,
            interaction_summaries=interaction_summaries,
        )
    candidate_capabilities = _candidate_capabilities_by_segment(graph)
    clarification = intent_contract.derive_envelope_clarification(
        goal=goal,
        profile=initial_profile.model_dump(mode="json", exclude_none=True),
        heuristic_profile=heuristic_profile.model_dump(mode="json", exclude_none=True),
        graph=graph.model_dump(mode="json", exclude_none=True),
        context_capability_candidates=(
            normalized_intent_context.get("capability_candidates")
            if isinstance(normalized_intent_context.get("capability_candidates"), list)
            else []
        ),
        candidate_required_inputs_by_segment={
            segment_id: _segment_required_inputs_for_candidates(
                capability_ids,
                required_input_lookup=runtime.capability_required_inputs,
            )
            for segment_id, capability_ids in candidate_capabilities.items()
        },
    )
    profile = initial_profile.model_copy(
        update={
            "needs_clarification": bool(clarification.get("needs_clarification")),
            "requires_blocking_clarification": bool(
                clarification.get("requires_blocking_clarification")
            ),
            "questions": list(clarification.get("questions") or []),
            "blocking_slots": list(clarification.get("blocking_slots") or []),
            "missing_slots": list(clarification.get("missing_inputs") or []),
            "slot_values": dict(clarification.get("slot_values") or dict(initial_profile.slot_values)),
            "clarification_mode": str(
                clarification.get("clarification_mode") or initial_profile.clarification_mode or ""
            )
            or None,
        }
    )
    return workflow_contracts.NormalizedIntentEnvelope(
        goal=goal,
        profile=profile,
        graph=graph,
        candidate_capabilities=candidate_capabilities,
        clarification=workflow_contracts.ClarificationState(
            needs_clarification=bool(clarification.get("needs_clarification")),
            requires_blocking_clarification=bool(
                clarification.get("requires_blocking_clarification")
            ),
            missing_inputs=list(clarification.get("missing_inputs") or []),
            questions=list(clarification.get("questions") or []),
            blocking_slots=list(clarification.get("blocking_slots") or []),
            slot_values=dict(clarification.get("slot_values") or {}),
            clarification_mode=str(clarification.get("clarification_mode") or "") or None,
        ),
        trace=workflow_contracts.NormalizationTrace(
            assessment_source=str(profile.source or "").strip() or None,
            assessment_mode=config.assessment_mode or None,
            assessment_model=config.assessment_model or None,
            assessment_fallback_used=_fallback_used(
                config.assessment_mode,
                profile.source,
            ),
            decomposition_source=str(graph.source or "").strip() or None,
            decomposition_mode=config.decomposition_mode or None,
            decomposition_model=config.decomposition_model or None,
            decomposition_fallback_used=(
                _fallback_used(config.decomposition_mode, graph.source)
                if config.include_decomposition
                else False
            ),
            heuristic_assessment_source=str(heuristic_profile.source or "").strip() or None,
            heuristic_assessment_intent=str(heuristic_profile.intent or "").strip() or None,
            context_projection="intent",
            context_slot_keys=sorted(context_slot_values.keys()),
            context_slot_provenance=context_slot_provenance,
            disagreement=dict(clarification.get("disagreement") or {}),
        ),
    )


def decompose_goal_intent(
    goal: str,
    *,
    db: Session | None = None,
    user_id: str | None = None,
    interaction_summaries: list[dict[str, Any]] | None = None,
    config: IntentDecomposeConfig,
    runtime: IntentDecomposeRuntime,
) -> workflow_contracts.IntentGraph:
    fallback_graph = runtime.heuristic_decompose(goal)
    if "source" not in fallback_graph:
        fallback_graph = {**fallback_graph, "source": "heuristic"}
    allowed_capability_catalog = runtime.capability_entries()
    allowed_capability_ids = {
        str(entry.get("id") or "").strip()
        for entry in allowed_capability_catalog
        if str(entry.get("id") or "").strip()
    }
    if not allowed_capability_ids:
        allowed_capability_ids = runtime.capability_ids()
    normalized_user_id = runtime.normalize_user_id(user_id)
    workflow_hints = runtime.retrieve_workflow_hints(
        db,
        goal,
        normalized_user_id,
        config.memory_retrieval_limit,
    )
    semantic_goal_capabilities = runtime.semantic_goal_capability_hints(
        goal,
        allowed_capability_catalog,
        max(4, config.capability_top_k * 2),
    )
    has_interaction_summaries = bool(interaction_summaries)
    result = "heuristic"
    graph = fallback_graph
    if not config.enabled:
        result = "disabled"
    elif config.mode == "heuristic":
        result = "heuristic"
    elif runtime.provider is None:
        result = "provider_unavailable"
    else:
        try:
            graph = runtime.llm_decompose(
                goal=goal,
                provider=runtime.provider,
                fallback_graph=fallback_graph,
                allowed_capability_ids=allowed_capability_ids,
                allowed_capability_catalog=allowed_capability_catalog,
                capability_top_k=config.capability_top_k,
                interaction_summaries=interaction_summaries,
                workflow_hints=workflow_hints,
                semantic_goal_capabilities=semantic_goal_capabilities,
            )
            result = "llm"
        except Exception as exc:  # noqa: BLE001
            runtime.on_llm_failure(exc)
            graph = fallback_graph
            result = "llm_failed_fallback"
    graph = runtime.annotate_graph_summary_defaults(graph)
    summary_raw = graph.get("summary")
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    summary["memory_hints_used"] = len(workflow_hints)
    summary["memory_retrieval_enabled"] = bool(config.memory_retrieval_enabled)
    summary["semantic_capability_hints_used"] = len(semantic_goal_capabilities)
    graph = {**graph, "summary": summary}
    if interaction_summaries:
        graph = runtime.apply_supported_fact_filter(graph, interaction_summaries)
    runtime.record_metrics(graph, result, has_interaction_summaries)
    return workflow_contracts.IntentGraph.model_validate(graph)


def assess_goal_intent(
    goal: str,
    *,
    config: GoalIntentConfig,
    runtime: GoalIntentRuntime,
) -> workflow_contracts.GoalIntentProfile:
    inference = runtime.infer_task_intent(goal)
    intent = str(getattr(inference, "intent", "") or "")
    source = str(getattr(inference, "source", "") or "")
    confidence = round(float(getattr(inference, "confidence", 0.0) or 0.0), 3)
    risk_level = _infer_goal_risk_level(goal, intent)
    threshold = resolve_intent_confidence_threshold(
        intent,
        risk_level,
        config=config,
    )
    slot_values = _extract_goal_slot_signals(goal, intent, risk_level)
    blocking_slots = blocking_clarification_slots(
        intent,
        risk_level,
        config=config,
    )
    missing_slots = [
        slot_name
        for slot_name in blocking_slots
        if not str(slot_values.get(slot_name) or "").strip()
    ]
    low_confidence = confidence < threshold
    if low_confidence and "intent_action" in blocking_slots and "intent_action" not in missing_slots:
        missing_slots.append("intent_action")
    questions = [slot_question(slot_name, goal) for slot_name in missing_slots]
    profile = workflow_contracts.GoalIntentProfile(
        intent=intent,
        source=source,
        confidence=confidence,
        risk_level=risk_level,
        threshold=threshold,
        low_confidence=low_confidence,
        needs_clarification=bool(missing_slots),
        requires_blocking_clarification=bool(missing_slots),
        questions=questions,
        blocking_slots=blocking_slots,
        missing_slots=missing_slots,
        slot_values=slot_values,
        clarification_mode="targeted_slot_filling",
    )
    if runtime.record_metrics is not None:
        runtime.record_metrics(profile)
    return profile


def resolve_intent_confidence_threshold(
    intent: str,
    risk_level: str,
    *,
    config: GoalIntentConfig,
) -> float:
    base = max(0.0, min(1.0, float(config.min_confidence)))
    intent_override = config.min_confidence_by_intent.get(intent)
    risk_override = config.min_confidence_by_risk.get(risk_level)
    candidates = [base]
    if isinstance(intent_override, float):
        candidates.append(intent_override)
    if isinstance(risk_override, float):
        candidates.append(risk_override)
    return round(max(candidates), 3)


def blocking_clarification_slots(
    intent: str,
    risk_level: str,
    *,
    config: GoalIntentConfig,
) -> list[str]:
    slots: list[str] = []
    if "intent_action" in config.clarification_blocking_slots:
        slots.append("intent_action")
    if intent in {"render", "generate"} and "output_format" in config.clarification_blocking_slots:
        slots.append("output_format")
    if intent == "io" and "target_system" in config.clarification_blocking_slots:
        slots.append("target_system")
    if (
        risk_level == "high_risk_write"
        and "safety_constraints" in config.clarification_blocking_slots
    ):
        slots.append("safety_constraints")
    return slots


def slot_question(slot: str, goal: str) -> str:
    return intent_contract.required_input_question(slot, goal)


def _infer_goal_risk_level(goal: str, intent: str) -> str:
    lowered = str(goal or "").lower()
    high_risk_tokens = (
        "delete",
        "destroy",
        "remove",
        "drop",
        "shutdown",
        "infra",
        "prod",
        "production",
        "payment",
        "billing",
        "credential",
        "secret",
    )
    if any(token in lowered for token in high_risk_tokens):
        return "high_risk_write"
    bounded_tokens = (
        "create",
        "update",
        "write",
        "publish",
        "push",
        "commit",
        "open pr",
        "pull request",
        "render",
        "generate",
    )
    if intent in {"render", "transform", "generate"} and any(
        token in lowered for token in bounded_tokens
    ):
        return "bounded_write"
    if intent == "io" and any(token in lowered for token in ("write", "upload", "save", "push")):
        return "bounded_write"
    return "read_only"


def _candidate_capabilities_by_segment(
    graph: workflow_contracts.IntentGraph,
) -> dict[str, list[str]]:
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


def _segment_required_inputs_for_candidates(
    capability_ids: list[str],
    *,
    required_input_lookup: Callable[[str], list[str]],
) -> list[str]:
    for capability_id in capability_ids:
        required_inputs: list[str] = []
        for raw_input in required_input_lookup(capability_id):
            normalized = intent_contract.normalize_required_input_key(raw_input)
            if normalized and normalized not in required_inputs:
                required_inputs.append(normalized)
        if required_inputs:
            return required_inputs
    return []


def _fallback_used(mode: str, source: str | None) -> bool:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode in {"", "heuristic", "disabled"}:
        return False
    return str(source or "").strip().lower() != "llm"


def _extract_goal_slot_signals(goal: str, intent: str, risk_level: str) -> dict[str, Any]:
    lowered = str(goal or "").lower()
    output_format = ""
    for token, normalized in (
        ("word document", "docx"),
        ("microsoft word", "docx"),
        ("word doc", "docx"),
        ("pdf", "pdf"),
        ("docx", "docx"),
        ("markdown", "md"),
        (".md", "md"),
        ("json", "json"),
        ("csv", "csv"),
        ("xlsx", "xlsx"),
        ("excel", "xlsx"),
        ("html", "html"),
        ("text", "txt"),
        ("txt", "txt"),
    ):
        if token in lowered:
            output_format = normalized
            break
    target_system = ""
    target_candidates = (
        "github",
        "gitlab",
        "jira",
        "slack",
        "notion",
        "confluence",
        "filesystem",
        "workspace",
        "artifacts",
        "gmail",
    )
    for token in target_candidates:
        if token in lowered:
            target_system = token
            break
    safety_constraints = ""
    if any(
        token in lowered
        for token in (
            "read only",
            "read-only",
            "no write",
            "without write",
            "do not delete",
            "safe mode",
            "dry run",
        )
    ):
        safety_constraints = "present"
    return {
        "intent_action": intent or "",
        "output_format": output_format,
        "target_system": target_system,
        "safety_constraints": safety_constraints,
        "risk_level": risk_level,
    }
