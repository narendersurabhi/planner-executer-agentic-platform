from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from libs.core import capability_registry, intent_contract
from libs.core.llm_provider import LLMProvider, LLMProviderError, LLMRequest

_BOOTSTRAP_FORMAT_TOKENS = {"docx": "docx", "pdf": "pdf"}


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
    field_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    field_questions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatClarificationFieldMetadata:
    field_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    field_questions: dict[str, str] = field(default_factory=dict)
    field_examples: dict[str, tuple[str, ...]] = field(default_factory=dict)
    output_format_tokens: dict[str, str] = field(default_factory=dict)


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
        field_aliases = _normalized_field_aliases(raw_contract.get("field_aliases"))
        field_questions = _normalized_field_map(raw_contract.get("field_questions"))
        existing_fields = _existing_fields_for_context(
            goal=goal,
            merged_context=merged_context,
            field_names=collectible_fields,
            field_aliases=field_aliases,
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
                field_aliases=field_aliases,
                field_questions=field_questions,
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
    heuristic_updates = heuristic_field_updates_for_answer(
        preferred_field=preferred_field,
        latest_answer=latest_answer,
        allowed_fields=contract.missing_fields,
    )
    heuristic_confidence = {field: 1.0 for field in heuristic_updates}
    if provider is None:
        unresolved = [
            field
            for field in contract.required_fields
            if field in contract.missing_fields and field not in heuristic_updates
        ]
        return heuristic_updates, unresolved, heuristic_confidence
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
    for field_name, value in heuristic_updates.items():
        updates[field_name] = value
        accepted_confidence[field_name] = heuristic_confidence[field_name]
        unresolved.discard(field_name)
    for field_name in effective_missing_fields:
        if field_name in updates:
            continue
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
    registry_question = _chat_field_metadata().field_questions.get(normalized)
    if registry_question:
        return registry_question
    return intent_contract.required_input_question(normalized, goal)


def normalize_clarification_field_key(value: Any) -> str:
    normalized = intent_contract.normalize_required_input_key(value)
    if not normalized:
        return ""
    for field, aliases in _chat_field_metadata().field_aliases.items():
        if normalized == field or normalized in aliases:
            return field
    return normalized


def normalize_output_format_token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _normalize_output_format_token(value)


def heuristic_field_updates_for_answer(
    *,
    preferred_field: str | None,
    latest_answer: str | None,
    allowed_fields: Sequence[str] = (),
) -> dict[str, str]:
    normalized_preferred = intent_contract.normalize_required_input_key(preferred_field)
    answer = str(latest_answer or "").strip()
    allowed = {
        normalized
        for field in allowed_fields
        if (normalized := intent_contract.normalize_required_input_key(field))
    }
    if not answer or not normalized_preferred:
        return {}
    if allowed and normalized_preferred not in allowed:
        return {}
    if normalized_preferred == "instruction" and _looks_like_substantive_instruction(answer):
        return {"instruction": answer}
    if normalized_preferred == "goal" and _looks_like_substantive_instruction(answer):
        return {"goal": answer}
    if normalized_preferred == "workspace_path":
        return {"workspace_path": _normalize_workspace_path_answer(answer)}
    if normalized_preferred == "output_format":
        normalized_format = _normalize_output_format_token(answer)
        if normalized_format:
            return {"output_format": normalized_format}
    return {}


def _normalize_workspace_path_answer(answer: str) -> str:
    normalized = " ".join(str(answer or "").strip().split())
    if not normalized:
        return ""
    lowered = normalized.lower()
    explicit_path = re.search(
        r"\bworkspace(?:/[A-Za-z0-9._-]+)+\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if explicit_path:
        return explicit_path.group(0).rstrip("/")
    if lowered in {
        "workspace",
        "workspace folder",
        "the workspace",
        "in workspace",
        "in the workspace",
        "the workspace folder",
    }:
        return "workspace"
    if "workspace" in lowered and "folder" in lowered and "/" not in normalized:
        return "workspace"
    return normalized


def _existing_fields_for_context(
    *,
    goal: str,
    merged_context: Mapping[str, Any] | None,
    field_names: Sequence[str],
    field_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    context = dict(merged_context) if isinstance(merged_context, Mapping) else {}
    nested_job = context.get("job_context")
    job_context = dict(nested_job) if isinstance(nested_job, Mapping) else {}
    clarification_slots = _clarification_slot_values_from_context(context)
    registry_aliases = _chat_field_metadata().field_aliases
    contract_aliases = {
        intent_contract.normalize_required_input_key(key): tuple(value)
        for key, value in (field_aliases or {}).items()
        if intent_contract.normalize_required_input_key(key)
    }
    existing: dict[str, str] = {}
    for field_name in field_names:
        aliases = _unique_field_aliases(
            (field_name,),
            registry_aliases.get(field_name, ()),
            contract_aliases.get(field_name, ()),
        )
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


def _normalized_field_aliases(value: Any) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    if not isinstance(value, Mapping):
        return normalized
    for raw_key, raw_aliases in value.items():
        key = intent_contract.normalize_required_input_key(raw_key)
        if not key:
            continue
        aliases = _normalize_example_list(raw_aliases)
        if aliases:
            normalized[key] = aliases
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
    for key, examples in _chat_field_metadata().field_examples.items():
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
        if "." not in normalized and "/" not in normalized and _normalize_output_format_token(normalized):
            return ""
        return normalized.strip("\"'")
    return normalized


def _normalize_output_format_token(value: str) -> str:
    normalized = str(value or "").strip().strip("\"'").lower()
    if not normalized:
        return ""
    tokens = dict(_BOOTSTRAP_FORMAT_TOKENS)
    tokens.update(_chat_field_metadata().output_format_tokens)
    direct = tokens.get(normalized)
    if direct:
        return direct
    for token in re.findall(r"[a-z0-9]+", normalized):
        mapped = tokens.get(token)
        if mapped:
            return mapped
    return ""


def _chat_field_metadata() -> ChatClarificationFieldMetadata:
    try:
        registry = capability_registry.load_capability_registry()
    except Exception:  # noqa: BLE001
        return ChatClarificationFieldMetadata(output_format_tokens=dict(_BOOTSTRAP_FORMAT_TOKENS))

    aliases: dict[str, list[str]] = {}
    questions: dict[str, str] = {}
    examples: dict[str, list[str]] = {}
    format_tokens: dict[str, str] = {}

    def _append_alias(field: str, alias: str) -> None:
        normalized_field = intent_contract.normalize_required_input_key(field)
        normalized_alias = intent_contract.normalize_required_input_key(alias)
        if not normalized_field or not normalized_alias:
            return
        values = aliases.setdefault(normalized_field, [])
        if normalized_field not in values:
            values.append(normalized_field)
        if normalized_alias not in values:
            values.append(normalized_alias)

    def _append_example(field: str, example: str) -> None:
        normalized_field = intent_contract.normalize_required_input_key(field)
        normalized_example = str(example or "").strip()
        if not normalized_field or not normalized_example:
            return
        values = examples.setdefault(normalized_field, [])
        if normalized_example not in values:
            values.append(normalized_example)

    for spec in registry.enabled_capabilities().values():
        hints = spec.planner_hints if isinstance(spec.planner_hints, Mapping) else {}
        raw_aliases = hints.get("chat_field_aliases")
        if isinstance(raw_aliases, Mapping):
            for raw_field, raw_values in raw_aliases.items():
                field = intent_contract.normalize_required_input_key(raw_field)
                if not field:
                    continue
                _append_alias(field, field)
                for alias in _normalize_example_list(raw_values):
                    _append_alias(field, alias)

        raw_questions = hints.get("chat_field_questions")
        if isinstance(raw_questions, Mapping):
            for raw_field, raw_question in raw_questions.items():
                field = intent_contract.normalize_required_input_key(raw_field)
                question = str(raw_question or "").strip()
                if field and question:
                    questions.setdefault(field, question)

        raw_examples = hints.get("chat_field_examples")
        if isinstance(raw_examples, Mapping):
            for raw_field, raw_values in raw_examples.items():
                for example in _normalize_example_list(raw_values):
                    _append_example(str(raw_field), example)

        raw_tokens = hints.get("chat_output_format_tokens")
        if isinstance(raw_tokens, Mapping):
            for raw_token, raw_value in raw_tokens.items():
                token = str(raw_token or "").strip().strip("\"'").lower()
                value = str(raw_value or "").strip().strip("\"'").lower()
                if token and value:
                    format_tokens[token] = value

        required_extension = str(hints.get("required_output_extension") or "").strip().lower()
        if required_extension:
            format_tokens.setdefault(required_extension, required_extension)

    for key, value in _BOOTSTRAP_FORMAT_TOKENS.items():
        format_tokens.setdefault(key, value)

    return ChatClarificationFieldMetadata(
        field_aliases={key: tuple(values) for key, values in aliases.items()},
        field_questions=questions,
        field_examples={key: tuple(values) for key, values in examples.items()},
        output_format_tokens=format_tokens,
    )


def _unique_field_aliases(*collections: Sequence[str]) -> tuple[str, ...]:
    aliases: list[str] = []
    for collection in collections:
        if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
            continue
        for raw_alias in collection:
            alias = intent_contract.normalize_required_input_key(raw_alias)
            if alias and alias not in aliases:
                aliases.append(alias)
    return tuple(aliases)


def _looks_like_substantive_instruction(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    if _normalize_output_format_token(normalized):
        return False
    if len(normalized.split()) >= 4:
        return True
    lowered = normalized.lower()
    return any(token in lowered for token in ("cover ", "about ", "include ", "explain ", "how "))


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
