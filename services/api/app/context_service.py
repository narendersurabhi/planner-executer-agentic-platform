from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Sequence

from sqlalchemy.orm import Session

from libs.core import capability_registry, workflow_contracts

from . import memory_profile_service

_WORKFLOW_CONTEXT_KEYS = (
    "workflow",
    "workflow_inputs",
    "workflow_context_json",
    "workflow_run_metadata",
    "workflow_trigger_id",
    "workflow_version_id",
    "workflow_definition_id",
    "workflow_ref",
)

_INTERACTION_SUMMARY_STAGE_LIMITS: dict[str, int] = {
    "envelope": 8,
    "intent": 6,
    "chat_route": 3,
    "chat_submit": 6,
    "planner": 4,
    "execution": 6,
    "workflow_runtime": 8,
    "preflight": 8,
}

_CAPABILITY_CANDIDATE_STAGE_LIMITS: dict[str, int] = {
    "intent": 8,
    "chat_route": 8,
    "planner": 10,
}

_INTENT_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "instruction": ("instruction", "goal_details", "goal"),
    "topic": ("topic", "main_topic", "title", "subject", "document_title"),
    "audience": ("audience", "target_role_name", "role_name"),
    "tone": ("tone",),
    "query": ("query", "github_query"),
    "path": ("path", "output_path", "filename", "file_name", "output_filename"),
    "output_format": ("output_format", "format"),
    "target_system": ("target_system",),
    "safety_constraints": ("safety_constraints",),
    "intent_action": ("intent_action",),
    "risk_level": ("risk_level",),
    "length": ("length", "target_pages", "page_count", "word_count", "max_words"),
    "markdown_text": ("markdown_text", "content", "text"),
    "target_repo": ("target_repo", "repo", "repo_name", "repo_full_name"),
}

_CONTEXT_NOISE_TOKENS: set[str] = {
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

_COMMON_STOPWORDS: set[str] = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "have",
    "what",
    "when",
    "where",
    "which",
    "about",
    "only",
    "give",
    "just",
    "need",
    "want",
    "make",
}


def collect_context_sources(
    **sources: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    for name, value in sources.items():
        if isinstance(value, Mapping):
            collected[name] = dict(value)
    return collected


def normalize_context_sources(
    sources: Mapping[str, Mapping[str, Any]] | None,
    *,
    user_id: str | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    merged_context: dict[str, Any] = {}
    sources_used: list[str] = []
    for source_name, payload in (sources or {}).items():
        if not isinstance(payload, Mapping) or not payload:
            continue
        merged_context = _merge_context_maps(merged_context, payload)
        sources_used.append(str(source_name))

    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        normalized_user_id = str(
            merged_context.get("user_id") or merged_context.get("semantic_user_id") or ""
        ).strip()
    if normalized_user_id:
        merged_context["user_id"] = normalized_user_id
        merged_context["semantic_user_id"] = normalized_user_id
        if "authenticated_user" not in sources_used and user_id:
            sources_used.append("authenticated_user")
    return merged_context, sources_used, normalized_user_id


def build_context_envelope(
    *,
    db: Session | None,
    goal: str,
    session_metadata: Mapping[str, Any] | None = None,
    context_sources: Mapping[str, Mapping[str, Any]] | None = None,
    user_id: str | None = None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> workflow_contracts.ContextEnvelope:
    merged_context, sources_used, normalized_user_id = normalize_context_sources(
        context_sources,
        user_id=user_id,
    )
    parsed_normalized = workflow_contracts.parse_normalized_intent_envelope(
        normalized_intent_envelope
    )
    filtered_context, dropped_inputs = drop_noisy_context_items(
        goal=goal,
        context=merged_context,
        stage="envelope",
    )

    profile_payload: dict[str, Any] = {}
    if normalized_user_id and db is not None:
        try:
            profile = memory_profile_service.load_user_profile(db, normalized_user_id)
            profile_payload = profile.model_dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001
            profile_payload = {}
        if profile_payload and "user_profile" not in sources_used:
            sources_used.append("user_profile")

    if parsed_normalized is not None and "normalized_intent_envelope" not in sources_used:
        sources_used.append("normalized_intent_envelope")

    interaction_summaries = _interaction_summaries_from_context(filtered_context)
    if interaction_summaries and "interaction_summaries" not in sources_used:
        sources_used.append("interaction_summaries")
    capability_candidates = _rank_capability_candidates(
        goal=goal,
        capability_candidates=_capability_candidates_from_envelope(parsed_normalized),
        missing_inputs=derive_missing_inputs(
            context=filtered_context,
            normalized_intent_envelope=parsed_normalized,
        ),
    )

    return workflow_contracts.ContextEnvelope(
        goal=str(goal or "").strip(),
        context_json=filtered_context,
        user_scope={"user_id": normalized_user_id} if normalized_user_id else {},
        session_scope=_session_scope_from_metadata(session_metadata),
        workflow_scope=_workflow_scope_from_context(filtered_context),
        normalized_intent_envelope=(
            workflow_contracts.dump_normalized_intent_envelope(parsed_normalized)
            if parsed_normalized is not None
            else None
        ),
        profile=profile_payload,
        interaction_summaries=interaction_summaries,
        capability_candidates=capability_candidates,
        runtime_metadata=dict(runtime_metadata or {}),
        missing_inputs=derive_missing_inputs(
            context=filtered_context,
            normalized_intent_envelope=parsed_normalized,
        ),
        dropped_inputs=dropped_inputs,
        trace=workflow_contracts.ContextEnvelopeTrace(
            sources_used=sources_used,
            profile_loaded=bool(profile_payload),
            normalized_intent_source=(
                str(parsed_normalized.profile.source or "").strip() or None
                if parsed_normalized is not None
                else None
            ),
        ),
    )


def build_chat_context_envelope(
    *,
    db: Session,
    goal: str,
    session_metadata: Mapping[str, Any] | None,
    session_context: Mapping[str, Any] | None,
    turn_context: Mapping[str, Any] | None,
    user_id: str | None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> workflow_contracts.ContextEnvelope:
    return build_context_envelope(
        db=db,
        goal=goal,
        session_metadata=session_metadata,
        context_sources=collect_context_sources(
            session_context=session_context,
            turn_context=turn_context,
        ),
        user_id=user_id,
        normalized_intent_envelope=normalized_intent_envelope,
        runtime_metadata=runtime_metadata,
    )


def build_workflow_runtime_context_envelope(
    *,
    db: Session | None,
    goal: str,
    version_context: Mapping[str, Any] | None,
    trigger_context: Mapping[str, Any] | None,
    request_context: Mapping[str, Any] | None,
    trigger_inputs: Mapping[str, Any] | None,
    explicit_inputs: Mapping[str, Any] | None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> tuple[workflow_contracts.ContextEnvelope, dict[str, Any]]:
    envelope = build_context_envelope(
        db=db,
        goal=goal,
        context_sources=collect_context_sources(
            version_context=version_context,
            trigger_context=trigger_context,
            request_context=request_context,
        ),
        normalized_intent_envelope=normalized_intent_envelope,
        runtime_metadata=runtime_metadata,
    )
    runtime_inputs, _sources_used, _normalized_user_id = normalize_context_sources(
        collect_context_sources(
            trigger_inputs=trigger_inputs,
            explicit_inputs=explicit_inputs,
        )
    )
    return envelope, runtime_inputs


def build_preflight_context_envelope(
    *,
    db: Session | None,
    goal: str,
    provided_job_context: Mapping[str, Any] | None,
    persisted_job_context: Mapping[str, Any] | None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> workflow_contracts.ContextEnvelope:
    context_sources = (
        collect_context_sources(provided_job_context=provided_job_context)
        if isinstance(provided_job_context, Mapping) and provided_job_context
        else collect_context_sources(persisted_job_context=persisted_job_context)
    )
    return build_context_envelope(
        db=db,
        goal=goal,
        context_sources=context_sources,
        normalized_intent_envelope=normalized_intent_envelope,
        runtime_metadata=runtime_metadata,
    )


def chat_route_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    context = budget_context_for_stage(parsed, stage="chat_route")
    if parsed.profile:
        context["user_profile"] = dict(parsed.profile)
    if parsed.capability_candidates:
        limit = _CAPABILITY_CANDIDATE_STAGE_LIMITS.get("chat_route", 0)
        ranked = list(parsed.capability_candidates)
        if limit > 0:
            ranked = ranked[:limit]
        if ranked:
            context["capability_candidates"] = ranked
    if parsed.missing_inputs:
        context["missing_inputs"] = list(parsed.missing_inputs)
    return context


def intent_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    base_context = budget_context_for_stage(parsed, stage="intent")
    context = _merge_clarification_slot_ledger(base_context, parsed.session_scope)
    if parsed.profile:
        context["user_profile"] = dict(parsed.profile)
    if parsed.capability_candidates:
        limit = _CAPABILITY_CANDIDATE_STAGE_LIMITS.get("intent", 0)
        ranked = list(parsed.capability_candidates)
        if limit > 0:
            ranked = ranked[:limit]
        if ranked:
            context["capability_candidates"] = ranked
    if parsed.missing_inputs:
        context["missing_inputs"] = list(parsed.missing_inputs)
    intent_slot_values, intent_slot_provenance = _intent_slot_values(parsed, base_context)
    if intent_slot_values:
        context["intent_slot_values"] = intent_slot_values
        context["intent_slot_provenance"] = intent_slot_provenance
        for key, value in intent_slot_values.items():
            context.setdefault(key, value)
    return context


def workflow_runtime_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    return budget_context_for_stage(envelope, stage="workflow_runtime")


def planner_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    context = budget_context_for_stage(parsed, stage="planner")
    context.pop("user_profile", None)
    if parsed.capability_candidates:
        limit = _CAPABILITY_CANDIDATE_STAGE_LIMITS.get("planner", 0)
        ranked = list(parsed.capability_candidates)
        if limit > 0:
            ranked = ranked[:limit]
        if ranked:
            context["capability_candidates"] = ranked
    if parsed.missing_inputs:
        context["missing_inputs"] = list(parsed.missing_inputs)
    return context


def execution_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = budget_context_for_stage(envelope, stage="execution")
    context.pop("user_profile", None)
    context.pop("capability_candidates", None)
    context.pop("missing_inputs", None)
    return context


def preflight_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    return budget_context_for_stage(envelope, stage="preflight")


def chat_submit_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    context = budget_context_for_stage(parsed, stage="chat_submit")
    return _merge_clarification_slot_ledger(context, parsed.session_scope)


def rank_context_items(
    *,
    goal: str,
    interaction_summaries: list[dict[str, Any]] | None = None,
    capability_candidates: list[str] | None = None,
    missing_inputs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "interaction_summaries": _rank_interaction_summaries(
            goal=goal,
            interaction_summaries=interaction_summaries or [],
        ),
        "capability_candidates": _rank_capability_candidates(
            goal=goal,
            capability_candidates=capability_candidates or [],
            missing_inputs=missing_inputs or [],
        ),
    }


def drop_noisy_context_items(
    *,
    goal: str,
    context: Mapping[str, Any] | None,
    stage: str,
) -> tuple[dict[str, Any], list[str]]:
    payload = dict(context) if isinstance(context, Mapping) else {}
    dropped: list[str] = []
    interaction_summaries = _interaction_summaries_from_context(payload)
    if interaction_summaries:
        ranked = _rank_interaction_summaries(goal=goal, interaction_summaries=interaction_summaries)
        if len(ranked) < len(interaction_summaries):
            dropped.append("interaction_summaries:noise")
        limit = _INTERACTION_SUMMARY_STAGE_LIMITS.get(stage, _INTERACTION_SUMMARY_STAGE_LIMITS["envelope"])
        if len(ranked) > limit:
            ranked = ranked[:limit]
            dropped.append("interaction_summaries:budget")
        if ranked:
            payload["interaction_summaries"] = ranked
        else:
            payload.pop("interaction_summaries", None)
    return payload, list(dict.fromkeys(dropped))


def derive_missing_inputs(
    *,
    context: Mapping[str, Any] | None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None,
) -> list[str]:
    parsed = workflow_contracts.parse_normalized_intent_envelope(normalized_intent_envelope)
    missing_inputs = _missing_inputs_from_envelope(parsed)
    if not missing_inputs:
        return []
    context_map = dict(context) if isinstance(context, Mapping) else {}
    unresolved: list[str] = []
    for key in missing_inputs:
        if not _context_has_required_input(context_map, key):
            unresolved.append(key)
    return unresolved


def budget_context_for_stage(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    context, _dropped = drop_noisy_context_items(
        goal=parsed.goal,
        context=parsed.context_json,
        stage=stage,
    )
    if stage != "chat_route":
        for key in (
            "interaction_summaries_ref",
            "interaction_summaries_meta",
        ):
            context.pop(key, None)
    if stage in {"chat_submit", "planner", "execution"}:
        context.pop("user_profile", None)
    if stage == "execution":
        context.pop("capability_candidates", None)
        context.pop("missing_inputs", None)
    return context


def update_chat_context_envelope(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any],
    *,
    goal: str | None = None,
    context_json: Mapping[str, Any] | None = None,
    normalized_intent_envelope: workflow_contracts.NormalizedIntentEnvelope | Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> workflow_contracts.ContextEnvelope:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        raise ValueError("context_envelope_required")

    updated_context = dict(parsed.context_json or {})
    if isinstance(context_json, Mapping):
        updated_context = _merge_context_maps(updated_context, context_json)
    updated_context, dropped_inputs = drop_noisy_context_items(
        goal=str(goal or parsed.goal or "").strip(),
        context=updated_context,
        stage="envelope",
    )

    parsed_normalized = workflow_contracts.parse_normalized_intent_envelope(
        normalized_intent_envelope or parsed.normalized_intent_envelope
    )
    merged_runtime_metadata = dict(parsed.runtime_metadata or {})
    if isinstance(runtime_metadata, Mapping):
        merged_runtime_metadata.update(dict(runtime_metadata))

    return parsed.model_copy(
        update={
            "goal": str(goal or parsed.goal or "").strip(),
            "context_json": updated_context,
            "workflow_scope": _workflow_scope_from_context(updated_context),
            "normalized_intent_envelope": (
                workflow_contracts.dump_normalized_intent_envelope(parsed_normalized)
                if parsed_normalized is not None
                else None
            ),
            "capability_candidates": _rank_capability_candidates(
                goal=str(goal or parsed.goal or "").strip(),
                capability_candidates=_capability_candidates_from_envelope(parsed_normalized),
                missing_inputs=derive_missing_inputs(
                    context=updated_context,
                    normalized_intent_envelope=parsed_normalized,
                ),
            ),
            "missing_inputs": derive_missing_inputs(
                context=updated_context,
                normalized_intent_envelope=parsed_normalized,
            ),
            "interaction_summaries": _interaction_summaries_from_context(updated_context),
            "runtime_metadata": merged_runtime_metadata,
            "dropped_inputs": dropped_inputs,
        }
    )


def _session_scope_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    session_scope: dict[str, Any] = {}
    if not isinstance(metadata, Mapping):
        return session_scope
    for key in ("draft_goal", "active_job_id", "active_workflow_run_id"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                session_scope[key] = value.strip()
            continue
        session_scope[key] = value
    pending_clarification = metadata.get("pending_clarification")
    if isinstance(pending_clarification, Mapping):
        pending_scope = {
            "questions": [
                str(question).strip()
                for question in pending_clarification.get("questions", [])
                if isinstance(question, str) and question.strip()
            ]
        }
        for key in (
            "state_version",
            "active_family",
            "active_segment_id",
            "active_capability_id",
            "original_goal",
            "current_question",
            "current_question_field",
            "auto_path_allowed",
        ):
            value = pending_clarification.get(key)
            if isinstance(value, str):
                if value.strip():
                    pending_scope[key] = value.strip()
                continue
            if value is not None:
                pending_scope[key] = value
        for key in (
            "pending_questions",
            "pending_fields",
            "required_fields",
            "answered_fields",
            "candidate_capabilities",
            "question_history",
            "answer_history",
        ):
            value = pending_clarification.get(key)
            if not isinstance(value, list):
                continue
            normalized_values = []
            for item in value:
                if isinstance(item, str):
                    normalized = item.strip()
                    if normalized:
                        normalized_values.append(normalized)
                    continue
                if item is not None:
                    normalized_values.append(item)
            if normalized_values:
                pending_scope[key] = normalized_values
        for key in ("known_slot_values", "resolved_slots", "slot_provenance"):
            value = pending_clarification.get(key)
            if isinstance(value, Mapping) and value:
                pending_scope[key] = dict(value)
        session_scope["pending_clarification"] = pending_scope
    pending_workflow_input = metadata.get("pending_workflow_input")
    if isinstance(pending_workflow_input, Mapping):
        session_scope["pending_workflow_input"] = dict(pending_workflow_input)
    return session_scope


def _workflow_scope_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    workflow_scope: dict[str, Any] = {}
    for key in _WORKFLOW_CONTEXT_KEYS:
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            workflow_scope[key] = dict(value)
        else:
            workflow_scope[key] = value
    return workflow_scope


def _interaction_summaries_from_context(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = context.get("interaction_summaries")
    if not isinstance(raw, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            summaries.append(dict(item))
    return summaries


def _rank_interaction_summaries(
    *,
    goal: str,
    interaction_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    goal_tokens = _goal_tokens(goal)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, item in enumerate(interaction_summaries):
        if _interaction_summary_is_noisy(item):
            continue
        haystack = _interaction_summary_text(item)
        lexical_score = sum(1 for token in goal_tokens if token in haystack)
        recency_score = float(index) / max(1, len(interaction_summaries))
        score = float(lexical_score * 4) + recency_score
        scored.append((score, index, dict(item)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item for _score, _index, item in scored]


def _rank_capability_candidates(
    *,
    goal: str,
    capability_candidates: list[str],
    missing_inputs: list[str],
) -> list[str]:
    goal_tokens = _goal_tokens(goal)
    missing = {str(value or "").strip().lower() for value in missing_inputs if str(value or "").strip()}
    scored: list[tuple[int, int, str]] = []
    for index, capability_id in enumerate(capability_candidates):
        normalized = capability_registry.canonicalize_capability_id(capability_id)
        if not normalized:
            continue
        score = 0
        parts = [part for part in normalized.lower().split(".") if part]
        if any(token in parts for token in goal_tokens):
            score += 2
        if "path" in missing and any(token in parts for token in ("render", "filename", "filesystem")):
            score += 1
        if "query" in missing and any(token in parts for token in ("search", "list", "github", "filesystem")):
            score += 1
        scored.append((score, -index, normalized))
    scored.sort(reverse=True)
    deduped: list[str] = []
    for _score, _neg_index, normalized in scored:
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _context_has_required_input(context: Mapping[str, Any], key: str) -> bool:
    normalized = str(key or "").strip().lower()
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
    candidates = (normalized,) + aliases.get(normalized, ())
    for candidate in candidates:
        if _mapping_has_non_empty_value(context, candidate):
            return True
        workflow = context.get("workflow")
        if isinstance(workflow, Mapping):
            workflow_inputs = workflow.get("inputs")
            if isinstance(workflow_inputs, Mapping) and _mapping_has_non_empty_value(workflow_inputs, candidate):
                return True
    return False


def _intent_slot_values(
    envelope: workflow_contracts.ContextEnvelope,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    slot_values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    normalized_fields = _clarification_normalized_fields(context)
    normalized_intent = workflow_contracts.parse_normalized_intent_envelope(
        envelope.normalized_intent_envelope
    )
    workflow_inputs = _workflow_inputs_from_context(context)
    clarification_slot_values, clarification_slot_provenance = _clarification_slot_ledger(
        envelope.session_scope,
        context,
    )
    clarification_slot_values = (
        {
            **(
                dict(normalized_intent.clarification.slot_values)
                if normalized_intent is not None
                else {}
            ),
            **clarification_slot_values,
        }
    )
    inferred_slot_values = (
        dict(normalized_intent.profile.slot_values) if normalized_intent is not None else {}
    )
    for canonical_key, aliases in _INTENT_SLOT_ALIASES.items():
        explicit_value = _first_non_empty_value(context, workflow_inputs, aliases)
        if explicit_value is not None:
            slot_values[canonical_key] = explicit_value
            provenance[canonical_key] = (
                "clarification_normalized"
                if canonical_key in normalized_fields and canonical_key in context
                else "explicit"
            )
            continue
        clarification_value = _first_non_empty_value(clarification_slot_values, None, aliases)
        if clarification_value is not None:
            slot_values[canonical_key] = clarification_value
            canonical_provenance = _first_non_empty_value(
                clarification_slot_provenance,
                None,
                aliases,
            )
            if isinstance(canonical_provenance, str) and canonical_provenance.strip():
                provenance[canonical_key] = canonical_provenance.strip()
            else:
                provenance[canonical_key] = (
                    "clarification_normalized"
                    if canonical_key in normalized_fields
                    else "inferred"
                )
            continue
        inferred_value = _first_non_empty_value(inferred_slot_values, None, aliases)
        if inferred_value is not None:
            slot_values[canonical_key] = inferred_value
            provenance[canonical_key] = "inferred"
    return slot_values, provenance


def _clarification_slot_ledger(
    session_scope: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    slot_values: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    pending_scope = (
        session_scope.get("pending_clarification")
        if isinstance(session_scope, Mapping)
        and isinstance(session_scope.get("pending_clarification"), Mapping)
        else None
    )
    if isinstance(pending_scope, Mapping):
        for key in ("known_slot_values", "resolved_slots"):
            raw_value = pending_scope.get(key)
            if isinstance(raw_value, Mapping):
                for raw_key, raw_slot in raw_value.items():
                    normalized_key = str(raw_key or "").strip()
                    if normalized_key and raw_slot is not None:
                        slot_values[normalized_key] = raw_slot
        raw_provenance = pending_scope.get("slot_provenance")
        if isinstance(raw_provenance, Mapping):
            for raw_key, raw_value in raw_provenance.items():
                normalized_key = str(raw_key or "").strip()
                normalized_value = str(raw_value or "").strip()
                if normalized_key and normalized_value:
                    provenance[normalized_key] = normalized_value

    if isinstance(context, Mapping):
        raw_context_slots = context.get("clarification_resolved_slots")
        if isinstance(raw_context_slots, Mapping):
            for raw_key, raw_slot in raw_context_slots.items():
                normalized_key = str(raw_key or "").strip()
                if normalized_key and raw_slot is not None:
                    slot_values.setdefault(normalized_key, raw_slot)
        raw_context_provenance = context.get("clarification_slot_provenance")
        if isinstance(raw_context_provenance, Mapping):
            for raw_key, raw_value in raw_context_provenance.items():
                normalized_key = str(raw_key or "").strip()
                normalized_value = str(raw_value or "").strip()
                if normalized_key and normalized_value:
                    provenance.setdefault(normalized_key, normalized_value)

    return slot_values, provenance


def _merge_clarification_slot_ledger(
    context: Mapping[str, Any] | None,
    session_scope: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(context) if isinstance(context, Mapping) else {}
    slot_values, provenance = _clarification_slot_ledger(session_scope, context)
    if not slot_values:
        return merged

    merged["clarification_resolved_slots"] = dict(slot_values)
    if provenance:
        merged["clarification_slot_provenance"] = dict(provenance)
    for key, value in slot_values.items():
        if not _mapping_has_non_empty_value(merged, key):
            merged[key] = value
    return merged


def _clarification_normalized_fields(context: Mapping[str, Any]) -> set[str]:
    raw = context.get("clarification_normalization")
    if not isinstance(raw, Mapping):
        return set()
    normalized: set[str] = set()
    for value in raw.get("fields", []):
        key = str(value or "").strip()
        if key:
            normalized.add(key)
    return normalized


def _workflow_inputs_from_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    workflow = context.get("workflow")
    if not isinstance(workflow, Mapping):
        return {}
    inputs = workflow.get("inputs")
    if not isinstance(inputs, Mapping):
        return {}
    return inputs


def _first_non_empty_value(
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any] | None,
    aliases: Sequence[str],
) -> Any:
    for mapping in (primary, secondary):
        if not isinstance(mapping, Mapping):
            continue
        for alias in aliases:
            if alias not in mapping:
                continue
            value = mapping.get(alias)
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
                continue
            if isinstance(value, Mapping):
                if value:
                    return dict(value)
                continue
            if isinstance(value, list):
                if value:
                    return list(value)
                continue
            if value is not None:
                return value
    return None


def _mapping_has_non_empty_value(mapping: Mapping[str, Any], key: str) -> bool:
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


def _goal_tokens(goal: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", str(goal or "").lower())
        if token not in _COMMON_STOPWORDS
    }
    return tokens


def _interaction_summary_text(item: Mapping[str, Any]) -> str:
    fields: list[str] = []
    for key in ("action", "facts", "evidence", "speculation"):
        value = item.get(key)
        if isinstance(value, str):
            fields.append(value)
        elif isinstance(value, list):
            fields.extend(str(entry) for entry in value if isinstance(entry, str))
    return " ".join(fields).strip().lower()


def _interaction_summary_is_noisy(item: Mapping[str, Any]) -> bool:
    text = _interaction_summary_text(item)
    if not text:
        return True
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if not normalized:
        return True
    if normalized in _CONTEXT_NOISE_TOKENS:
        return True
    tokens = [token for token in normalized.split() if token]
    if tokens and all(token in {"yes", "yeah", "yep", "ok", "okay", "thanks", "thank", "you"} for token in tokens):
        return True
    if len(tokens) <= 2 and normalized in _CONTEXT_NOISE_TOKENS:
        return True
    return False


def _capability_candidates_from_envelope(
    envelope: workflow_contracts.NormalizedIntentEnvelope | None,
) -> list[str]:
    if envelope is None:
        return []
    capability_ids: list[str] = []
    seen: set[str] = set()
    for capability_list in envelope.candidate_capabilities.values():
        for capability_id in capability_list:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
    for segment in envelope.graph.segments:
        for capability_id in segment.suggested_capabilities:
            normalized = capability_registry.canonicalize_capability_id(capability_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                capability_ids.append(normalized)
    return capability_ids


def _missing_inputs_from_envelope(
    envelope: workflow_contracts.NormalizedIntentEnvelope | None,
) -> list[str]:
    if envelope is None:
        return []
    missing: list[str] = []
    seen: set[str] = set()
    for value in envelope.profile.missing_slots:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            missing.append(normalized)
    for value in envelope.clarification.missing_inputs:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            missing.append(normalized)
    return missing


def _merge_context_maps(
    session_context: Mapping[str, Any] | None,
    turn_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    left = dict(session_context) if isinstance(session_context, Mapping) else {}
    right = dict(turn_context) if isinstance(turn_context, Mapping) else {}
    merged = {**left, **right}
    for key in ("workflow_inputs", "workflow_context_json", "workflow_run_metadata"):
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
            merged[key] = {
                **(dict(left_value) if isinstance(left_value, Mapping) else {}),
                **(dict(right_value) if isinstance(right_value, Mapping) else {}),
            }
    left_ref = left.get("workflow_ref")
    right_ref = right.get("workflow_ref")
    if isinstance(left_ref, Mapping) or isinstance(right_ref, Mapping):
        merged["workflow_ref"] = {
            **(dict(left_ref) if isinstance(left_ref, Mapping) else {}),
            **(dict(right_ref) if isinstance(right_ref, Mapping) else {}),
        }
    return merged
