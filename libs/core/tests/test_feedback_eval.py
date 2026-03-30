import json

from libs.core import feedback_eval


def test_feedback_eval_builds_rows_and_jsonl() -> None:
    examples = [
        {
            "feedback": {
                "id": "fb-1",
                "target_type": "plan",
                "target_id": "plan-1",
                "sentiment": "negative",
                "reason_codes": ["missing_step"],
                "comment": "Missing review step.",
            },
            "snapshot": {
                "plan_id": "plan-1",
                "task_count": 2,
                "metadata": {
                    "boundary_decision": {
                        "decision": "execution_request",
                        "evidence": {"top_families": [{"family": "documents"}]},
                    }
                },
            },
            "dimensions": {"planner_version": "planner_v2", "llm_model": "gpt-test"},
            "linked_ids": {"job_id": "job-1", "plan_id": "plan-1"},
        }
    ]

    rows = feedback_eval.build_feedback_eval_rows(examples)

    assert rows == [
        {
            "feedback_id": "fb-1",
            "target_type": "plan",
            "target_id": "plan-1",
            "sentiment": "negative",
            "reason_codes": ["missing_step"],
            "comment": "Missing review step.",
            "snapshot": {
                "plan_id": "plan-1",
                "task_count": 2,
                "metadata": {
                    "boundary_decision": {
                        "decision": "execution_request",
                        "evidence": {"top_families": [{"family": "documents"}]},
                    }
                },
            },
            "dimensions": {"planner_version": "planner_v2", "llm_model": "gpt-test"},
            "linked_ids": {"job_id": "job-1", "plan_id": "plan-1"},
            "boundary_decision": "execution_request",
            "boundary_evidence": {"top_families": [{"family": "documents"}]},
            "clarification_active_family": None,
            "clarification_slot_loss_state": None,
            "clarification_family_alignment": None,
            "clarification_answer_count": 0,
            "clarification_resolved_slot_count": 0,
        }
    ]

    payload = feedback_eval.dumps_feedback_eval_rows_jsonl(examples)
    parsed = [json.loads(line) for line in payload.splitlines() if line.strip()]
    assert parsed == rows
