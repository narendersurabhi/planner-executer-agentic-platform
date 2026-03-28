from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    merged_context = _merge_context_maps(session_context, turn_context)
    normalized_user_id = str(user_id or "").strip()
    sources_used: list[str] = []
    if isinstance(session_context, Mapping) and session_context:
        sources_used.append("session_context")
    if isinstance(turn_context, Mapping) and turn_context:
        sources_used.append("turn_context")
    if normalized_user_id:
        merged_context["user_id"] = normalized_user_id
        merged_context["semantic_user_id"] = normalized_user_id
        sources_used.append("authenticated_user")

    profile_payload: dict[str, Any] = {}
    if normalized_user_id:
        try:
            profile = memory_profile_service.load_user_profile(db, normalized_user_id)
            profile_payload = profile.model_dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001
            profile_payload = {}
        if profile_payload:
            sources_used.append("user_profile")

    parsed_normalized = workflow_contracts.parse_normalized_intent_envelope(
        normalized_intent_envelope
    )
    if parsed_normalized is not None:
        sources_used.append("normalized_intent_envelope")

    interaction_summaries = _interaction_summaries_from_context(merged_context)
    if interaction_summaries:
        sources_used.append("interaction_summaries")

    return workflow_contracts.ContextEnvelope(
        goal=str(goal or "").strip(),
        context_json=merged_context,
        user_scope={"user_id": normalized_user_id} if normalized_user_id else {},
        session_scope=_session_scope_from_metadata(session_metadata),
        workflow_scope=_workflow_scope_from_context(merged_context),
        normalized_intent_envelope=(
            workflow_contracts.dump_normalized_intent_envelope(parsed_normalized)
            if parsed_normalized is not None
            else None
        ),
        profile=profile_payload,
        interaction_summaries=interaction_summaries,
        capability_candidates=_capability_candidates_from_envelope(parsed_normalized),
        runtime_metadata=dict(runtime_metadata or {}),
        missing_inputs=_missing_inputs_from_envelope(parsed_normalized),
        dropped_inputs=[],
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


def chat_route_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    context = dict(parsed.context_json or {})
    if parsed.profile:
        context["user_profile"] = dict(parsed.profile)
    return context


def chat_submit_context_view(
    envelope: workflow_contracts.ContextEnvelope | Mapping[str, Any] | None,
) -> dict[str, Any]:
    parsed = workflow_contracts.parse_context_envelope(envelope)
    if parsed is None:
        return {}
    return dict(parsed.context_json or {})


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
            "capability_candidates": _capability_candidates_from_envelope(parsed_normalized),
            "missing_inputs": _missing_inputs_from_envelope(parsed_normalized),
            "interaction_summaries": _interaction_summaries_from_context(updated_context),
            "runtime_metadata": merged_runtime_metadata,
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
        session_scope["pending_clarification"] = {
            "questions": [
                str(question).strip()
                for question in pending_clarification.get("questions", [])
                if isinstance(question, str) and question.strip()
            ]
        }
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
