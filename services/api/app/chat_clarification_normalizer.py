from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from libs.core import intent_contract
from libs.core.llm_provider import LLMProvider, LLMProviderError, LLMRequest

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "instruction": ("instruction",),
    "topic": ("topic", "main_topic", "document_title", "title", "subject"),
    "audience": ("audience", "target_role_name", "role_name"),
    "tone": ("tone",),
    "query": ("query", "github_query"),
    "markdown_text": ("markdown_text", "content", "text"),
    "path": ("path", "output_path", "filename", "file_name", "output_filename"),
}
_FIELD_QUESTION_OVERRIDES = {
    "instruction": "What should this specifically cover?",
    "topic": "What is the main topic or title?",
    "audience": "Who is the target audience?",
    "tone": "What tone should it use (for example practical, formal, conversational, or executive)?",
    "query": "What exact search query should I use?",
}
_FIELD_EXAMPLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "tone": ("practical", "concise", "technical", "formal", "conversational", "executive"),
}


@dataclass(frozen=True)
class CapabilityNormalizationContract:
    capability_id: str
    description: str
    required_inputs: tuple[str, ...] = ()
    collectible_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    existing_fields: dict[str, str] = field(default_factory=dict)
    field_descriptions: dict[str, str] = field(default_factory=dict)
    field_examples: dict[str, tuple[str, ...]] = field(default_factory=dict)


def build_capability_normalization_contracts(
    *,
    goal: str,
    merged_context: Mapping[str, Any] | None,
    capability_contracts: Sequence[Mapping[str, Any]],
) -> list[CapabilityNormalizationContract]:
    contracts: list[CapabilityNormalizationContract] = []
    for raw_contract in capability_contracts:
        if not isinstance(raw_contract, Mapping):
            continue
        capability_id = str(raw_contract.get("capability_id") or "").strip()
        if not capability_id:
            continue
        collectible_fields = _normalized_field_list(
            raw_contract.get("chat_collectible_fields")
        )
        if not collectible_fields:
            continue
        required_inputs = _normalized_field_list(raw_contract.get("required_inputs"))
        chat_required_fields = _normalized_field_list(raw_contract.get("chat_required_fields"))
        field_descriptions = _normalized_field_map(raw_contract.get("field_descriptions"))
        field_examples = _normalized_field_examples(raw_contract.get("field_examples"))
        existing_fields = _existing_fields_for_context(
            goal=goal,
            merged_context=merged_context,
            field_names=collectible_fields,
        )
        required_fields = tuple(
            field
            for field in collectible_fields
            if field in required_inputs or field in chat_required_fields
        )
        missing_fields = tuple(
            field
            for field in collectible_fields
            if not existing_fields.get(field)
        )
        if not missing_fields:
            continue
        contracts.append(
            CapabilityNormalizationContract(
                capability_id=capability_id,
                description=str(raw_contract.get("description") or "").strip(),
                required_inputs=required_inputs,
                collectible_fields=collectible_fields,
                required_fields=required_fields,
                missing_fields=missing_fields,
                existing_fields=existing_fields,
                field_descriptions=field_descriptions,
                field_examples=field_examples,
            )
        )
    return contracts


def normalize_contract_fields_with_llm(
    *,
    contract: CapabilityNormalizationContract,
    provider: LLMProvider | None,
    confidence_threshold: float,
    goal: str,
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
    preferred_field: str | None = None,
    latest_answer: str | None = None,
) -> tuple[dict[str, str], list[str], dict[str, float]]:
    if not contract.missing_fields:
        return {}, [], {}
    if provider is None:
        return {}, list(contract.required_fields), {}
    normalized_preferred_field = intent_contract.normalize_required_input_key(preferred_field)
    prioritized_missing_fields_list: list[str] = []
    ordered_candidates = (
        ([normalized_preferred_field] if normalized_preferred_field in contract.missing_fields else [])
        + [field for field in contract.missing_fields if field != normalized_preferred_field]
    )
    for field in ordered_candidates:
        field_name = intent_contract.normalize_required_input_key(field)
        if field_name and field_name not in prioritized_missing_fields_list:
            prioritized_missing_fields_list.append(field_name)
    prioritized_missing_fields = tuple(prioritized_missing_fields_list)
    effective_missing_fields = prioritized_missing_fields or contract.missing_fields

    payload = {
        "goal_with_clarifications": goal,
        "latest_answer": str(latest_answer or "").strip(),
        "conversation_history": [
            {
                "role": str(item.get("role") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
            for item in (conversation_history or [])
            if isinstance(item, Mapping)
            and str(item.get("role") or "").strip()
            and str(item.get("content") or "").strip()
        ],
        "capability_id": contract.capability_id,
        "capability_description": contract.description,
        "required_inputs": list(contract.required_inputs),
        "existing_fields": {key: value for key, value in contract.existing_fields.items() if value},
        "missing_fields": list(effective_missing_fields),
        "preferred_field": (
            normalized_preferred_field
            if normalized_preferred_field in effective_missing_fields
            else None
        ),
        "field_descriptions": contract.field_descriptions,
        "field_examples": {
            key: list(values)
            for key, values in contract.field_examples.items()
            if values
        },
        "response_schema": {
            "normalized_slots": {field: "string" for field in effective_missing_fields},
            "field_confidence": {field: "0..1" for field in effective_missing_fields},
            "unresolved_fields": list(effective_missing_fields),
        },
    }
    try:
        parsed = provider.generate_request_json_object(
            LLMRequest(
                prompt=json.dumps(payload, ensure_ascii=True),
                system_prompt=(
                    "Normalize chat clarification text into canonical capability input fields. "
                    "Return JSON only. Use only the missing field names provided. "
                    "Do not invent fields. "
                    "If preferred_field is present, treat the latest_answer as primarily answering that field. "
                    "You may fill other missing fields only when they are clearly supported by the latest answer or recent conversation history. "
                    "If a value is unclear, leave it empty and include the field in unresolved_fields."
                ),
                metadata={"component": "chat_clarification_normalizer"},
            )
        )
    except (LLMProviderError, ValueError):
        return {}, list(contract.required_fields), {}

    raw_slots = parsed.get("normalized_slots")
    slot_map = dict(raw_slots) if isinstance(raw_slots, Mapping) else {}
    raw_confidence = parsed.get("field_confidence")
    confidence_map = dict(raw_confidence) if isinstance(raw_confidence, Mapping) else {}
    raw_unresolved = parsed.get("unresolved_fields")
    unresolved = {
        str(field).strip()
        for field in raw_unresolved
        if isinstance(field, str) and str(field).strip() in contract.missing_fields
        and str(field).strip() in contract.required_fields
    }

    updates: dict[str, str] = {}
    accepted_confidence: dict[str, float] = {}
    for field_name in effective_missing_fields:
        raw_value = slot_map.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            if field_name in contract.required_fields:
                unresolved.add(field_name)
            continue
        confidence = _coerce_confidence(confidence_map.get(field_name))
        if confidence < confidence_threshold:
            if field_name in contract.required_fields:
                unresolved.add(field_name)
            continue
        value = _normalize_field_value(field_name, raw_value.strip())
        if not value:
            if field_name in contract.required_fields:
                unresolved.add(field_name)
            continue
        updates[field_name] = value
        accepted_confidence[field_name] = confidence
        unresolved.discard(field_name)

    for field_name in contract.required_fields:
        if field_name not in contract.missing_fields:
            continue
        if field_name not in updates and field_name not in unresolved:
            unresolved.add(field_name)

    return updates, sorted(unresolved), accepted_confidence


def clarification_question_for_field(field: str, *, goal: str = "") -> str:
    normalized = intent_contract.normalize_required_input_key(field)
    if normalized in _FIELD_QUESTION_OVERRIDES:
        return _FIELD_QUESTION_OVERRIDES[normalized]
    return intent_contract.required_input_question(normalized, goal)


def _existing_fields_for_context(
    *,
    goal: str,
    merged_context: Mapping[str, Any] | None,
    field_names: Sequence[str],
) -> dict[str, str]:
    context = dict(merged_context) if isinstance(merged_context, Mapping) else {}
    nested_job = context.get("job_context")
    job_context = dict(nested_job) if isinstance(nested_job, Mapping) else {}
    clarification_slots = _clarification_slot_values_from_context(context)
    existing: dict[str, str] = {}
    for field_name in field_names:
        aliases = _FIELD_ALIASES.get(field_name, (field_name,))
        value = _first_non_empty_str(*(context.get(alias) for alias in aliases))
        if not value:
            value = _first_non_empty_str(*(job_context.get(alias) for alias in aliases))
        if not value:
            value = _first_non_empty_str(*(clarification_slots.get(alias) for alias in aliases))
        if not value and field_name == "instruction":
            value = _first_non_empty_str(goal)
        if value:
            existing[field_name] = _normalize_field_value(field_name, value)
    return existing


def _clarification_slot_values_from_context(
    context: Mapping[str, Any],
) -> dict[str, str]:
    slot_values: dict[str, str] = {}
    pending = context.get("pending_clarification")
    if isinstance(pending, Mapping):
        for key in ("known_slot_values", "resolved_slots"):
            raw_slots = pending.get(key)
            if isinstance(raw_slots, Mapping):
                for raw_key, raw_value in raw_slots.items():
                    normalized_key = str(raw_key or "").strip()
                    normalized_value = str(raw_value or "").strip()
                    if normalized_key and normalized_value:
                        slot_values.setdefault(normalized_key, normalized_value)
    raw_slots = context.get("clarification_resolved_slots")
    if isinstance(raw_slots, Mapping):
        for raw_key, raw_value in raw_slots.items():
            normalized_key = str(raw_key or "").strip()
            normalized_value = str(raw_value or "").strip()
            if normalized_key and normalized_value:
                slot_values.setdefault(normalized_key, normalized_value)
    return slot_values


def _normalized_field_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    normalized: list[str] = []
    for item in value:
        key = intent_contract.normalize_required_input_key(item)
        if key and key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def _normalized_field_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = intent_contract.normalize_required_input_key(raw_key)
        if not key:
            continue
        if isinstance(raw_value, str) and raw_value.strip():
            normalized[key] = raw_value.strip()
    return normalized


def _normalized_field_examples(value: Any) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    if isinstance(value, Mapping):
        for raw_key, raw_examples in value.items():
            key = intent_contract.normalize_required_input_key(raw_key)
            if not key:
                continue
            normalized_values = _normalize_example_list(raw_examples)
            if normalized_values:
                normalized[key] = normalized_values
    for key, examples in _FIELD_EXAMPLE_FALLBACKS.items():
        normalized.setdefault(key, examples)
    return normalized


def _normalize_example_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return tuple(normalized)


def _normalize_field_value(field: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    if field == "tone":
        return normalized.lower()
    if field == "path":
        return normalized.strip("\"'")
    return normalized


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 0.0
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return 0.0
    return 0.0
