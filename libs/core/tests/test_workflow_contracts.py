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


def test_parse_normalized_intent_envelope_accepts_mapping() -> None:
    envelope = workflow_contracts.parse_normalized_intent_envelope(
        {
            "goal": "Generate a report",
            "profile": {"intent": "generate", "source": "heuristic"},
            "graph": {"segments": [{"id": "s1", "intent": "generate"}]},
            "candidate_capabilities": {"s1": ["document.spec.generate"]},
            "clarification": {"needs_clarification": False},
            "trace": {"assessment_mode": "hybrid"},
        }
    )

    assert envelope is not None
    assert envelope.goal == "Generate a report"
    assert envelope.profile.intent == "generate"
    assert envelope.candidate_capabilities["s1"] == ["document.spec.generate"]
    assert envelope.trace.assessment_mode == "hybrid"


def test_parse_normalized_intent_envelope_canonicalizes_legacy_render_capabilities() -> None:
    envelope = workflow_contracts.parse_normalized_intent_envelope(
        {
            "goal": "Render the document spec as PDF",
            "profile": {"intent": "render", "source": "heuristic"},
            "graph": {
                "segments": [
                    {
                        "id": "s1",
                        "intent": "render",
                        "suggested_capabilities": ["document.pdf.generate"],
                        "suggested_capability_rankings": [{"id": "document.pdf.generate"}],
                    }
                ]
            },
            "candidate_capabilities": {"s1": ["document.pdf.generate"]},
        }
    )

    assert envelope is not None
    assert envelope.graph.segments[0].suggested_capabilities == ["document.pdf.render"]
    assert envelope.graph.segments[0].suggested_capability_rankings[0]["id"] == "document.pdf.render"
    assert envelope.candidate_capabilities["s1"] == ["document.pdf.render"]


def test_dump_normalized_intent_envelope_returns_json_mapping() -> None:
    dumped = workflow_contracts.dump_normalized_intent_envelope(
        workflow_contracts.NormalizedIntentEnvelope(
            goal="Generate a report",
            profile=workflow_contracts.GoalIntentProfile(
                intent="generate",
                source="heuristic",
            ),
            graph=workflow_contracts.IntentGraph(
                segments=[workflow_contracts.IntentGraphSegment(id="s1", intent="generate")]
            ),
            candidate_capabilities={"s1": ["document.spec.generate"]},
            clarification=workflow_contracts.ClarificationState(
                needs_clarification=False,
                missing_inputs=[],
            ),
            trace=workflow_contracts.NormalizationTrace(
                assessment_source="heuristic",
                assessment_mode="hybrid",
                assessment_fallback_used=True,
            ),
        )
    )

    assert dumped == {
        "schema_version": "intent_envelope_v1",
        "goal": "Generate a report",
        "profile": {
            "intent": "generate",
            "source": "heuristic",
            "low_confidence": False,
            "needs_clarification": False,
            "requires_blocking_clarification": False,
            "questions": [],
            "blocking_slots": [],
            "missing_slots": [],
            "slot_values": {},
        },
        "graph": {
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
                    "slots": {"must_have_inputs": []},
                }
            ],
            "summary": {"intent_order": []},
        },
        "candidate_capabilities": {"s1": ["document.spec.generate"]},
        "clarification": {
            "needs_clarification": False,
            "requires_blocking_clarification": False,
            "missing_inputs": [],
            "questions": [],
            "blocking_slots": [],
            "slot_values": {},
        },
        "trace": {
            "assessment_source": "heuristic",
            "assessment_mode": "hybrid",
            "assessment_fallback_used": True,
        },
    }
