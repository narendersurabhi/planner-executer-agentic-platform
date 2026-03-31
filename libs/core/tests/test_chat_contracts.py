from libs.core import chat_contracts


def test_clarification_mapping_contracts_round_trip() -> None:
    request = chat_contracts.ClarificationMappingRequest(
        original_goal="Create a Kubernetes deployment guide",
        latest_answer="any",
        pending_state=chat_contracts.ClarificationPendingState(
            original_goal="Create a Kubernetes deployment guide",
            pending_fields=["path", "tone"],
            pending_questions=[
                "What output path or filename should be used?",
                "What tone should it use?",
            ],
            known_slot_values={"output_format": "docx"},
            candidate_capabilities=["document.spec.generate", "document.docx.render"],
        ),
    )

    result = chat_contracts.ClarificationMappingResult(
        resolved_fields=[
            chat_contracts.ClarificationResolvedField(
                field="path",
                value="auto",
                confidence=0.93,
                source="llm_mapper",
            )
        ],
        remaining_fields=["tone"],
        confidence_by_field={"path": 0.93},
        auto_path_allowed=True,
    )

    request_dump = request.model_dump(mode="json")
    result_dump = result.model_dump(mode="json")

    assert request_dump["pending_state"]["pending_fields"] == ["path", "tone"]
    assert request_dump["pending_state"]["known_slot_values"]["output_format"] == "docx"
    assert result_dump["resolved_fields"][0]["field"] == "path"
    assert result_dump["auto_path_allowed"] is True
    assert result_dump["remaining_fields"] == ["tone"]


def test_clarification_state_syncs_execution_frame_and_slot_ledger() -> None:
    state = chat_contracts.ClarificationState(
        original_goal="Create a Kubernetes deployment guide",
        active_segment_id="s1",
        active_capability_id="document.pdf.generate",
        pending_fields=["path"],
        pending_questions=["What output path or filename should be used?"],
        known_slot_values={"tone": "practical"},
        slot_provenance={"tone": "clarification_normalized"},
    )

    dumped = state.model_dump(mode="json")

    assert dumped["execution_frame"]["mode"] == "clarification"
    assert dumped["execution_frame"]["original_goal"] == "Create a Kubernetes deployment guide"
    assert dumped["active_capability_id"] == "document.pdf.render"
    assert dumped["execution_frame"]["active_capability_id"] == "document.pdf.render"
    assert dumped["resolved_slots"] == {"tone": "practical"}
    assert dumped["answered_fields"] == ["tone"]
    assert dumped["slot_provenance"]["tone"] == "clarification_normalized"
    assert dumped["current_question"] == "What output path or filename should be used?"
    assert dumped["current_question_field"] == "path"
    assert dumped["questions"] == ["What output path or filename should be used?"]
