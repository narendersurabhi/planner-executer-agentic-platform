from libs.core import intent_tuning


def test_build_intent_tuning_rows_flattens_candidate_export() -> None:
    rows = intent_tuning.build_intent_tuning_rows(
        [
            {
                "feedback": {
                    "id": "fb_1",
                    "target_id": "job_1",
                    "target_type": "intent_assessment",
                    "sentiment": "negative",
                    "reason_codes": ["wrong_goal"],
                    "comment": "Wrong intent.",
                },
                "dimensions": {
                    "intent_assessment_intent": "io",
                    "intent_assessment_source": "llm",
                    "intent_top_capability": "document.spec.generate",
                    "intent_top_family": "document",
                },
                "linked_ids": {"job_id": "job_1", "session_id": "session_1"},
                "review_label": "likely_wrong_intent_interpretation",
                "review_score": 140,
                "tuning_focus": "assessment_prompt_and_capability_evidence",
                "suggested_case_id": "create_release_report_fb1",
                "observed_case": {
                    "goal": "Create a release report",
                    "profile_intent": "io",
                    "profile_source": "llm",
                    "graph_intents": ["generate"],
                    "candidate_capabilities": ["document.spec.generate"],
                    "missing_inputs": ["intent_action"],
                    "clarification_mode": "intent_disagreement",
                    "disagreement_reason": "graph_intent_conflict",
                },
                "gold_case_stub": {
                    "id": "create_release_report_fb1",
                    "goal": "Create a release report",
                },
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["feedback_id"] == "fb_1"
    assert row["review_label"] == "likely_wrong_intent_interpretation"
    assert row["goal"] == "Create a release report"
    assert row["candidate_capabilities"] == ["document.spec.generate"]
    assert row["gold_case_stub"]["id"] == "create_release_report_fb1"


def test_build_intent_tuning_gold_bundle_collects_case_stubs() -> None:
    payload = intent_tuning.build_intent_tuning_gold_bundle(
        [
            {
                "suggested_case_id": "create_release_report_fb1",
                "gold_case_stub": {
                    "goal": "Create a release report",
                    "expected_intents": [],
                    "_review_label": "likely_wrong_intent_interpretation",
                },
            }
        ],
        description="Intent tuning bundle",
    )

    assert payload["version"] == 1
    assert payload["description"] == "Intent tuning bundle"
    assert payload["cases"][0]["id"] == "create_release_report_fb1"
    assert payload["cases"][0]["goal"] == "Create a release report"


def test_dumps_intent_tuning_gold_yaml_includes_case_ids() -> None:
    rendered = intent_tuning.dumps_intent_tuning_gold_yaml(
        [
            {
                "suggested_case_id": "create_release_report_fb1",
                "gold_case_stub": {
                    "goal": "Create a release report",
                    "expected_intents": [],
                },
            }
        ]
    )

    assert "cases:" in rendered
    assert "id: create_release_report_fb1" in rendered
    assert "goal: Create a release report" in rendered
