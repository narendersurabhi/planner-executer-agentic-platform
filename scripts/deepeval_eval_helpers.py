from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from libs.core import capability_registry, intent_contract, workflow_contracts
from services.api.app import intent_service


def load_required_inputs_lookup() -> Any:
    registry = capability_registry.load_capability_registry()
    required_inputs_by_capability: dict[str, list[str]] = {}
    for capability_id, spec in registry.enabled_capabilities().items():
        required_inputs_by_capability[capability_id] = (
            capability_registry.planner_collectible_inputs_for_capability(
                capability_id,
                registry=registry,
            )
        )

    def _lookup(capability_id: str) -> list[str]:
        canonical = registry.canonicalize_id(capability_id) or str(capability_id or "").strip()
        return list(required_inputs_by_capability.get(canonical, ()))

    return _lookup


def goal_intent_assess_config() -> intent_service.GoalIntentConfig:
    return intent_service.GoalIntentConfig(
        min_confidence=0.70,
        min_confidence_by_intent={},
        min_confidence_by_risk={},
        clarification_blocking_slots={
            "intent_action",
            "output_format",
            "target_system",
            "safety_constraints",
        },
    )


def build_heuristic_goal_normalizer() -> Any:
    required_inputs_lookup = load_required_inputs_lookup()

    def _assess(goal: str) -> workflow_contracts.GoalIntentProfile:
        return intent_service.assess_goal_intent(
            goal,
            config=goal_intent_assess_config(),
            runtime=intent_service.GoalIntentRuntime(
                infer_task_intent=intent_contract.infer_task_intent_from_goal_with_metadata,
                record_metrics=None,
            ),
        )

    def _decompose(goal: str, **_: Any) -> workflow_contracts.IntentGraph:
        return workflow_contracts.IntentGraph.model_validate(
            intent_contract.decompose_goal_intent(goal)
        )

    def _normalize(goal: str, intent_context: Mapping[str, Any] | None) -> dict[str, Any]:
        envelope = intent_service.normalize_goal_intent(
            goal,
            intent_context=dict(intent_context or {}),
            config=intent_service.IntentNormalizeConfig(
                include_decomposition=True,
                assessment_mode="heuristic",
                assessment_model="",
                decomposition_mode="heuristic",
                decomposition_model="",
            ),
            runtime=intent_service.IntentNormalizeRuntime(
                assess_goal_intent=_assess,
                assess_goal_intent_heuristic=_assess,
                decompose_goal_intent=_decompose,
                capability_required_inputs=required_inputs_lookup,
            ),
        )
        return envelope.model_dump(mode="json", exclude_none=True)

    return _normalize


def request_feedback_examples(
    *,
    base_url: str,
    bearer_token: str | None,
    timeout_s: float,
    target_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    query = urlencode(
        [
            ("target_type", target_type),
            ("sentiment", "positive"),
            ("sentiment", "partial"),
            ("sentiment", "negative"),
            ("limit", str(max(1, limit))),
            ("format", "jsonl"),
        ]
    )
    headers = {"Accept": "application/x-ndjson"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        f"{base_url.rstrip('/')}/feedback/examples?{query}",
        headers=headers,
        method="GET",
    )
    with urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    examples: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            examples.append(item)
    return examples
