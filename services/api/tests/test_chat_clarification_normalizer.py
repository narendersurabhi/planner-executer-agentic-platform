import json

from services.api.app import chat_clarification_normalizer


def test_normalize_contract_fields_with_llm_prefers_active_field() -> None:
    class _Provider:
        def generate_request_json_object(self, request):
            payload = json.loads(request.prompt)
            assert payload["preferred_field"] == "tone"
            assert payload["latest_answer"] == "practical"
            assert payload["missing_fields"][0] == "tone"
            assert "preferred_field is present" in request.system_prompt
            return {
                "normalized_slots": {
                    "tone": "Practical",
                    "audience": "Senior software engineers",
                },
                "field_confidence": {
                    "tone": 0.98,
                    "audience": 0.93,
                },
                "unresolved_fields": [],
            }

    contract = chat_clarification_normalizer.CapabilityNormalizationContract(
        capability_id="document.spec.generate",
        description="Generate a DocumentSpec.",
        required_inputs=("audience", "tone"),
        collectible_fields=("audience", "tone"),
        required_fields=("audience", "tone"),
        missing_fields=("audience", "tone"),
    )

    updates, unresolved, confidence = (
        chat_clarification_normalizer.normalize_contract_fields_with_llm(
            contract=contract,
            provider=_Provider(),
            confidence_threshold=0.8,
            goal="Create a deployment report",
            conversation_history=[
                {"role": "assistant", "content": "What tone should it use?"},
                {"role": "user", "content": "practical"},
            ],
            preferred_field="tone",
            latest_answer="practical",
        )
    )

    assert updates == {
        "tone": "practical",
        "audience": "Senior software engineers",
    }
    assert unresolved == []
    assert confidence == {
        "tone": 0.98,
        "audience": 0.93,
    }
