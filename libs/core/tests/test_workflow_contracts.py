from libs.core import workflow_contracts


def test_parse_intent_graph_accepts_minimal_segment_graph() -> None:
    graph = workflow_contracts.parse_intent_graph(
        {
            "segments": [{"id": "s1", "intent": "render"}],
            "summary": {"semantic_capability_hints_used": 2},
            "source": "heuristic",
        }
    )

    assert graph is not None
    assert graph.segments[0].id == "s1"
    assert graph.segments[0].objective == ""
    assert graph.summary.semantic_capability_hints_used == 2
    assert graph.source == "heuristic"


def test_dump_intent_graph_returns_json_mapping() -> None:
    dumped = workflow_contracts.dump_intent_graph(
        workflow_contracts.IntentGraph.model_validate(
            {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "generate",
                        "slots": {"must_have_inputs": ["instruction"]},
                    }
                ],
                "summary": {"segment_count": 1},
            }
        )
    )

    assert dumped == {
        "segments": [
            {
                "id": "s1",
                "intent": "generate",
                "objective": "",
                "objective_facts": [],
                "depends_on": [],
                "required_inputs": [],
                "suggested_capabilities": [],
                "suggested_capability_rankings": [],
                "unsupported_facts": [],
                "slots": {"must_have_inputs": ["instruction"]},
            }
        ],
        "summary": {"segment_count": 1, "intent_order": []},
    }


def test_parse_goal_intent_profile_accepts_profile_mapping() -> None:
    profile = workflow_contracts.parse_goal_intent_profile(
        {
            "intent": "render",
            "confidence": 0.84,
            "risk_level": "bounded_write",
            "missing_slots": ["output_format"],
            "slot_values": {"intent_action": "render"},
        }
    )

    assert profile is not None
    assert profile.intent == "render"
    assert profile.missing_slots == ["output_format"]
    assert profile.slot_values["intent_action"] == "render"


def test_dump_goal_intent_profile_returns_json_mapping() -> None:
    dumped = workflow_contracts.dump_goal_intent_profile(
        workflow_contracts.GoalIntentProfile(
            intent="io",
            source="heuristic",
            confidence=0.72,
            risk_level="read_only",
            missing_slots=["target_system"],
            slot_values={"intent_action": "io"},
        )
    )

    assert dumped == {
        "intent": "io",
        "source": "heuristic",
        "confidence": 0.72,
        "risk_level": "read_only",
        "low_confidence": False,
        "needs_clarification": False,
        "requires_blocking_clarification": False,
        "questions": [],
        "blocking_slots": [],
        "missing_slots": ["target_system"],
        "slot_values": {"intent_action": "io"},
    }
