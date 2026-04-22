from libs.core import chat_routing_feedback


def test_build_feedback_rows_extracts_query_route_and_training_signals() -> None:
    rows = chat_routing_feedback.build_feedback_rows(
        [
            {
                "feedback": {
                    "id": "fb-1",
                    "session_id": "session-1",
                    "message_id": "msg-1",
                    "sentiment": "positive",
                    "reason_codes": [],
                },
                "snapshot": {
                    "user_message_content": "Run release readiness workflow",
                    "metadata": {
                        "routing_decision": {
                            "route": "run_workflow",
                            "selected_candidate_id": "workflow:release",
                            "top_k_candidates": ["workflow:release", "generic:submit_job"],
                        }
                    },
                },
                "dimensions": {
                    "routing_decision_route": "run_workflow",
                    "routing_selected_candidate_id": "workflow:release",
                    "routing_selected_candidate_type": "workflow",
                    "routing_fallback_used": "no",
                    "routing_fallback_reason": None,
                    "routing_execution_started": "yes",
                    "routing_execution_succeeded": "yes",
                },
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "Run release readiness workflow"
    assert row["decision_route"] == "run_workflow"
    assert row["selected_candidate_id"] == "workflow:release"
    assert row["selected_positive_ids"] == ["workflow:release"]
    assert row["hard_negative_ids"] == ["generic:submit_job"]
