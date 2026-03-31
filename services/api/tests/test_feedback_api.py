import os
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app import main  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from services.api.app.models import (  # noqa: E402
    ChatMessageRecord,
    ChatSessionRecord,
    JobRecord,
    PlanRecord,
    TaskRecord,
)


Base.metadata.create_all(bind=engine)

client = TestClient(main.app)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _create_plan_for_job(job_id: str, *, planner_version: str = "test_feedback_v1") -> tuple[str, str]:
    plan_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    now = _utcnow()
    with SessionLocal() as db:
        plan = PlanRecord(
            id=plan_id,
            job_id=job_id,
            planner_version=planner_version,
            created_at=now,
            tasks_summary="Test feedback plan",
            dag_edges=[],
            policy_decision={},
        )
        task = TaskRecord(
            id=task_id,
            job_id=job_id,
            plan_id=plan_id,
            name="WriteSummary",
            description="Write a short summary.",
            instruction="Write a short summary.",
            acceptance_criteria=["Summary exists"],
            expected_output_schema_ref="TaskResult",
            status="pending",
            deps=[],
            attempts=0,
            max_attempts=3,
            rework_count=0,
            max_reworks=2,
            assigned_to=None,
            intent="transform",
            tool_requests=["text_summarize"],
            tool_inputs={},
            created_at=now,
            updated_at=now,
            critic_required=True,
        )
        db.add(plan)
        db.add(task)
        db.commit()
    return plan_id, task_id


def _update_job(job_id: str, *, metadata: dict[str, Any] | None = None, status: str | None = None) -> None:
    now = _utcnow()
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        assert job is not None
        merged = dict(job.metadata_json or {})
        if metadata:
            merged.update(metadata)
        job.metadata_json = merged
        if status is not None:
            job.status = status
        job.updated_at = now
        db.commit()


def _update_job_intent_metadata(
    job_id: str,
    *,
    intent: str,
    source: str,
    clarification_mode: str,
    disagreement_reason: str | None = None,
    candidate_capability: str | None = None,
    missing_inputs: list[str] | None = None,
) -> None:
    normalized_intent_envelope: dict[str, Any] = {
        "goal": "intent feedback goal",
        "profile": {
            "intent": intent,
            "source": source,
            "needs_clarification": bool(missing_inputs),
            "requires_blocking_clarification": bool(missing_inputs),
            "missing_slots": list(missing_inputs or []),
            "blocking_slots": list(missing_inputs or []),
            "slot_values": {"intent_action": intent},
            "clarification_mode": clarification_mode,
        },
        "graph": {
            "segments": [
                {
                    "id": "s1",
                    "intent": intent,
                    "objective": "Intent review test",
                    "suggested_capabilities": [candidate_capability] if candidate_capability else [],
                }
            ]
        },
        "candidate_capabilities": {
            "s1": [candidate_capability] if candidate_capability else []
        },
        "clarification": {
            "needs_clarification": bool(missing_inputs),
            "requires_blocking_clarification": bool(missing_inputs),
            "missing_inputs": list(missing_inputs or []),
            "questions": ["Clarify intent"] if missing_inputs else [],
            "blocking_slots": list(missing_inputs or []),
            "slot_values": {"intent_action": intent},
            "clarification_mode": clarification_mode,
        },
        "trace": {
            "assessment_source": source,
            "context_projection": "intent",
            "disagreement": (
                {"reason_code": disagreement_reason} if disagreement_reason is not None else {}
            ),
        },
    }
    _update_job(
        job_id,
        metadata={
            "goal_intent_profile": dict(normalized_intent_envelope["profile"]),
            "normalized_intent_envelope": normalized_intent_envelope,
        },
    )


def _update_task(task_id: str, *, status: str | None = None, rework_count: int | None = None) -> None:
    now = _utcnow()
    with SessionLocal() as db:
        task = db.query(TaskRecord).filter(TaskRecord.id == task_id).first()
        assert task is not None
        if status is not None:
            task.status = status
        if rework_count is not None:
            task.rework_count = rework_count
        task.updated_at = now
        db.commit()


def _attach_chat_session_to_job(
    job_id: str,
    *,
    clarification_questions: list[str] | None = None,
) -> str:
    session_id = str(uuid.uuid4())
    now = _utcnow()
    with SessionLocal() as db:
        session = ChatSessionRecord(
            id=session_id,
            title="Feedback summary session",
            metadata_json={},
            created_at=now,
            updated_at=now,
        )
        db.add(session)
        if clarification_questions:
            db.add(
                ChatMessageRecord(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content="\n".join(clarification_questions),
                    metadata_json={},
                    action_json={
                        "type": "ask_clarification",
                        "clarification_questions": clarification_questions,
                    },
                    job_id=None,
                    created_at=now,
                )
            )
        db.add(
            ChatMessageRecord(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=f"Started job {job_id}",
                metadata_json={},
                action_json={"type": "submit_job"},
                job_id=job_id,
                created_at=now,
            )
        )
        db.commit()
    return session_id


def _create_chat_message_with_boundary_decision(
    *,
    content: str = "I can help with that.",
    action_type: str = "respond",
    boundary_decision: dict[str, Any] | None = None,
    session_metadata: dict[str, Any] | None = None,
    message_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    now = _utcnow()
    with SessionLocal() as db:
        db.add(
            ChatSessionRecord(
                id=session_id,
                title="Boundary feedback session",
                metadata_json=dict(session_metadata or {}),
                created_at=now,
                updated_at=now,
            )
        )
        db.add(
            ChatMessageRecord(
                id=message_id,
                session_id=session_id,
                role="assistant",
                content=content,
                metadata_json={
                    "boundary_decision": dict(boundary_decision or {}),
                    **dict(message_metadata or {}),
                },
                action_json={"type": action_type},
                job_id=None,
                created_at=now,
            )
        )
        db.commit()
    return session_id, message_id


def _metric_value(metrics_text: str, metric_name: str, labels: dict[str, str] | None = None) -> float:
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if labels:
            if not line.startswith(f"{metric_name}{{"):
                continue
            if any(f'{key}="{value}"' not in line for key, value in labels.items()):
                continue
            return float(line.rsplit(" ", 1)[-1])
        if re.match(rf"^{re.escape(metric_name)}\s+[-+0-9.eE]+$", line):
            return float(line.rsplit(" ", 1)[-1])
    return 0.0


def test_submit_chat_message_feedback_and_list_for_session() -> None:
    session = client.post("/chat/sessions", json={"title": "Feedback session"}).json()
    turn = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"content": "help", "context_json": {}, "priority": 0},
    ).json()
    assistant_message = turn["assistant_message"]

    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-chat"},
        json={
            "target_type": "chat_message",
            "target_id": assistant_message["id"],
            "sentiment": "positive",
            "reason_codes": [],
            "comment": "Helpful answer.",
            "metadata": {"surface": "chat"},
        },
    )

    assert feedback_response.status_code == 200
    body = feedback_response.json()
    assert body["target_type"] == "chat_message"
    assert body["message_id"] == assistant_message["id"]
    assert body["session_id"] == session["id"]
    assert body["snapshot"]["content"] == assistant_message["content"]
    assert body["metadata"]["dimensions"]["target_type"] == "chat_message"
    assert body["metadata"]["dimensions"]["assistant_action_type"] == "respond"
    assert body["metadata"]["dimensions"]["has_comment"] is True
    assert body["metadata"]["dimensions"]["reason_count"] == 0

    list_response = client.get(f"/chat/sessions/{session['id']}/feedback")
    assert list_response.status_code == 200
    list_body = list_response.json()
    assert list_body["summary"]["total"] >= 1
    assert any(item["id"] == body["id"] for item in list_body["items"])


def test_chat_message_feedback_carries_boundary_dimensions_into_summary_export_and_metrics() -> None:
    before_metrics = client.get("/metrics")
    assert before_metrics.status_code == 200
    boundary_feedback_before = _metric_value(
        before_metrics.text,
        "chat_boundary_feedback_total",
        {"decision": "chat_reply", "sentiment": "negative"},
    )

    session_id, message_id = _create_chat_message_with_boundary_decision(
        boundary_decision={
            "decision": "chat_reply",
            "reason_code": "conversation_first",
            "evidence": {
                "conversation_mode_hint": "execution_oriented",
                "top_families": [
                    {"family": "documents", "score": 1.7, "capability_ids": ["document.spec.generate"]}
                ],
            },
        }
    )

    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-boundary"},
        json={
            "target_type": "chat_message",
            "target_id": message_id,
            "sentiment": "negative",
            "reason_codes": ["missed_request"],
            "comment": "This should have started a job.",
        },
    )

    assert feedback_response.status_code == 200
    body = feedback_response.json()
    assert body["metadata"]["dimensions"]["boundary_decision"] == "chat_reply"
    assert body["metadata"]["dimensions"]["boundary_reason_code"] == "conversation_first"
    assert body["metadata"]["dimensions"]["boundary_conversation_mode_hint"] == "execution_oriented"
    assert body["metadata"]["dimensions"]["boundary_top_family"] == "documents"

    summary_response = client.get(
        "/feedback/summary",
        params={"target_type": "chat_message"},
    )
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert any(
        bucket["key"] == "chat_reply" and bucket["total"] >= 1
        for bucket in summary["boundary_decisions"]
    )
    assert any(
        bucket["key"] == "conversation_first" and bucket["total"] >= 1
        for bucket in summary["boundary_reason_codes"]
    )
    assert any(
        bucket["key"] == "documents" and bucket["total"] >= 1
        for bucket in summary["boundary_top_families"]
    )

    export_response = client.get(
        "/feedback/examples",
        params=[("target_type", "chat_message"), ("sentiment", "negative")],
    )
    assert export_response.status_code == 200
    export = export_response.json()
    exported = next(item for item in export["items"] if item["feedback"]["id"] == body["id"])
    assert exported["dimensions"]["boundary_decision"] == "chat_reply"
    assert exported["dimensions"]["boundary_top_family"] == "documents"
    assert exported["snapshot"]["metadata"]["boundary_decision"]["reason_code"] == "conversation_first"

    after_metrics = client.get("/metrics")
    assert after_metrics.status_code == 200
    assert (
        _metric_value(
            after_metrics.text,
            "chat_boundary_feedback_total",
            {"decision": "chat_reply", "sentiment": "negative"},
        )
        >= boundary_feedback_before + 1
    )

    list_response = client.get(f"/chat/sessions/{session_id}/feedback")
    assert list_response.status_code == 200
    assert any(item["id"] == body["id"] for item in list_response.json()["items"])


def test_chat_boundary_review_queue_orders_likely_misroutes() -> None:
    _session_a, message_a = _create_chat_message_with_boundary_decision(
        content="I can explain that in chat.",
        action_type="respond",
        boundary_decision={
            "decision": "chat_reply",
            "reason_code": "conversation_first",
            "evidence": {
                "conversation_mode_hint": "execution_oriented",
                "top_families": [{"family": "documents", "score": 1.7}],
            },
        },
    )
    _session_b, message_b = _create_chat_message_with_boundary_decision(
        content="I started a job for that request.",
        action_type="submit_job",
        boundary_decision={
            "decision": "execution_request",
            "reason_code": "semantic_capability_evidence",
            "evidence": {
                "conversation_mode_hint": "execution_oriented",
                "top_families": [{"family": "github", "score": 1.2}],
            },
        },
    )

    first = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "review-chat-reply"},
        json={
            "target_type": "chat_message",
            "target_id": message_a,
            "sentiment": "negative",
            "reason_codes": ["missed_request"],
            "comment": "This should have become a job.",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "review-exec"},
        json={
            "target_type": "chat_message",
            "target_id": message_b,
            "sentiment": "partial",
            "reason_codes": ["incorrect"],
            "comment": "This should probably have stayed in chat.",
        },
    )
    assert second.status_code == 200

    response = client.get("/feedback/chat-boundary/review")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    labels = [item["review_label"] for item in body["items"]]
    assert "likely_false_chat_reply" in labels
    assert "likely_false_execution_request" in labels
    filtered_chat_reply = client.get(
        "/feedback/chat-boundary/review",
        params={"review_label": "likely_false_chat_reply", "boundary_decision": "chat_reply"},
    )
    assert filtered_chat_reply.status_code == 200
    filtered_chat_reply_body = filtered_chat_reply.json()
    assert filtered_chat_reply_body["total"] >= 1
    assert filtered_chat_reply_body["items"][0]["review_label"] == "likely_false_chat_reply"

    filtered = client.get(
        "/feedback/chat-boundary/review",
        params={"boundary_decision": "chat_reply", "review_label": "likely_false_chat_reply"},
    )
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total"] >= 1
    assert all(item["dimensions"]["boundary_decision"] == "chat_reply" for item in filtered_body["items"])
    assert all(item["review_label"] == "likely_false_chat_reply" for item in filtered_body["items"])


def test_chat_clarification_feedback_captures_slot_loss_and_family_drift() -> None:
    before_metrics = client.get("/metrics")
    assert before_metrics.status_code == 200
    slot_loss_before = _metric_value(
        before_metrics.text,
        "chat_clarification_slot_loss_feedback_total",
        {"slot_loss_state": "resolved_field_still_pending", "sentiment": "negative"},
    )
    family_drift_before = _metric_value(
        before_metrics.text,
        "chat_clarification_family_alignment_feedback_total",
        {"alignment": "drift", "sentiment": "negative"},
    )
    mapping_before = _metric_value(
        before_metrics.text,
        "chat_clarification_mapping_feedback_total",
        {
            "resolved_active_field": "yes",
            "queue_advanced": "yes",
            "restarted": "no",
            "sentiment": "negative",
        },
    )

    _session_id, message_id = _create_chat_message_with_boundary_decision(
        content="Let me ask one more question.",
        action_type="ask_clarification",
        boundary_decision={
            "decision": "chat_reply",
            "reason_code": "conversation_first",
            "evidence": {
                "conversation_mode_hint": "clarification_answer",
                "active_family": "documents",
                "top_families": [{"family": "github", "score": 1.1}],
            },
        },
        message_metadata={
            "clarification_mapping": {
                "active_field_before": "path",
                "active_field_after": "tone",
                "resolved_fields": ["path"],
                "resolved_active_field": True,
                "queue_advanced": True,
                "restarted": False,
            }
        },
        session_metadata={
            "pending_clarification": {
                "original_goal": "create a Kubernetes deployment guide",
                "active_family": "documents",
                "active_capability_id": "document.docx.render",
                "pending_fields": ["path"],
                "required_fields": ["path"],
                "current_question": "What filename should I use?",
                "current_question_field": "path",
                "known_slot_values": {
                    "topic": "Kubernetes deployment guide",
                    "audience": "Senior Software Engineers",
                },
                "resolved_slots": {
                    "topic": "Kubernetes deployment guide",
                    "audience": "Senior Software Engineers",
                },
                "answered_fields": ["topic", "audience", "path"],
                "question_history": ["What filename should I use?"],
                "answer_history": [
                    "Senior Software Engineers",
                    "save it as medic.docx in the workspace",
                ],
                "slot_provenance": {
                    "topic": "clarification_normalized",
                    "audience": "clarification_normalized",
                },
            }
        },
    )

    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-clarification"},
        json={
            "target_type": "chat_message",
            "target_id": message_id,
            "sentiment": "negative",
            "reason_codes": ["missed_request"],
            "comment": "It lost the filename and drifted away from documents.",
        },
    )
    assert feedback_response.status_code == 200
    body = feedback_response.json()
    assert body["metadata"]["dimensions"]["clarification_active_family"] == "documents"
    assert body["metadata"]["dimensions"]["clarification_current_question"] == (
        "What filename should I use?"
    )
    assert body["metadata"]["dimensions"]["clarification_current_question_field"] == "path"
    assert body["metadata"]["dimensions"]["clarification_slot_loss_state"] == "resolved_field_still_pending"
    assert body["metadata"]["dimensions"]["clarification_family_alignment"] == "drift"
    assert body["metadata"]["dimensions"]["clarification_answer_count"] == 2
    assert body["metadata"]["dimensions"]["clarification_resolved_slot_count"] == 2
    assert body["metadata"]["dimensions"]["clarification_mapping_active_field_before"] == "path"
    assert body["metadata"]["dimensions"]["clarification_mapping_active_field_after"] == "tone"
    assert body["metadata"]["dimensions"]["clarification_mapping_resolved_active_field"] == "yes"
    assert body["metadata"]["dimensions"]["clarification_mapping_queue_advanced"] == "yes"
    assert body["metadata"]["dimensions"]["clarification_mapping_restarted"] == "no"

    summary_response = client.get(
        "/feedback/summary",
        params={"target_type": "chat_message"},
    )
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert any(
        bucket["key"] == "documents" and bucket["total"] >= 1
        for bucket in summary["clarification_active_families"]
    )
    assert any(
        bucket["key"] == "resolved_field_still_pending" and bucket["total"] >= 1
        for bucket in summary["clarification_slot_loss_states"]
    )
    assert any(
        bucket["key"] == "drift" and bucket["total"] >= 1
        for bucket in summary["clarification_family_alignments"]
    )
    assert any(
        bucket["key"] == "yes" and bucket["total"] >= 1
        for bucket in summary["clarification_mapping_resolved_active_field_states"]
    )
    assert any(
        bucket["key"] == "yes" and bucket["total"] >= 1
        for bucket in summary["clarification_mapping_queue_advancement_states"]
    )
    assert any(
        bucket["key"] == "no" and bucket["total"] >= 1
        for bucket in summary["clarification_mapping_restart_states"]
    )
    assert summary["metrics"]["clarification_slot_loss_feedback_rate"] > 0.0
    assert summary["metrics"]["clarification_family_drift_feedback_rate"] > 0.0
    assert summary["metrics"]["clarification_mapping_resolved_active_field_feedback_rate"] > 0.0
    assert summary["metrics"]["clarification_mapping_queue_advanced_feedback_rate"] > 0.0

    after_metrics = client.get("/metrics")
    assert after_metrics.status_code == 200
    assert (
        _metric_value(
            after_metrics.text,
            "chat_clarification_slot_loss_feedback_total",
            {"slot_loss_state": "resolved_field_still_pending", "sentiment": "negative"},
        )
        >= slot_loss_before + 1
    )
    assert (
        _metric_value(
            after_metrics.text,
            "chat_clarification_family_alignment_feedback_total",
            {"alignment": "drift", "sentiment": "negative"},
        )
        >= family_drift_before + 1
    )
    assert (
        _metric_value(
            after_metrics.text,
            "chat_clarification_mapping_feedback_total",
            {
                "resolved_active_field": "yes",
                "queue_advanced": "yes",
                "restarted": "no",
                "sentiment": "negative",
            },
        )
        >= mapping_before + 1
    )


def test_chat_clarification_review_queue_returns_slot_loss_items() -> None:
    _session_id, message_id = _create_chat_message_with_boundary_decision(
        content="I need one more detail before I can continue.",
        action_type="ask_clarification",
        boundary_decision={
            "decision": "continue_pending",
            "reason_code": "pending_clarification_state_preservation",
            "evidence": {
                "conversation_mode_hint": "clarification_answer",
                "active_family": "documents",
                "top_families": [{"family": "documents", "score": 1.4}],
            },
        },
        session_metadata={
            "pending_clarification": {
                "original_goal": "create a Kubernetes deployment guide",
                "active_family": "documents",
                "active_capability_id": "document.docx.render",
                "pending_fields": ["path"],
                "required_fields": ["path"],
                "current_question": "What filename should I use?",
                "current_question_field": "path",
                "known_slot_values": {"topic": "Kubernetes deployment guide"},
                "resolved_slots": {"topic": "Kubernetes deployment guide"},
                "answered_fields": ["path"],
                "question_history": ["What filename should I use?"],
                "answer_history": ["save it as medic.docx in the workspace"],
            }
        },
    )
    created = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-clarification-review"},
        json={
            "target_type": "chat_message",
            "target_id": message_id,
            "sentiment": "negative",
            "reason_codes": ["missed_request"],
            "comment": "The clarification loop is still dropping the filename.",
        },
    )
    assert created.status_code == 200

    response = client.get("/feedback/chat-clarification/review")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert any(item["review_label"] == "likely_slot_loss" for item in body["items"])

    filtered = client.get(
        "/feedback/chat-clarification/review",
        params={"review_label": "likely_slot_loss", "active_family": "documents"},
    )
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total"] >= 1
    assert all(item["review_label"] == "likely_slot_loss" for item in filtered_body["items"])
    assert all(
        item["dimensions"]["clarification_active_family"] == "documents"
        for item in filtered_body["items"]
    )
    assert any(
        item["dimensions"].get("clarification_current_question_field") == "path"
        for item in filtered_body["items"]
    )


def test_submit_intent_and_plan_feedback_updates_same_actor_record() -> None:
    job_response = client.post(
        "/jobs",
        json={"goal": "Create a short document summary", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    plan_id, _task_id = _create_plan_for_job(job["id"])

    first = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-plan"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "negative",
            "reason_codes": ["missing_step"],
            "comment": "Needs one more review step.",
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-plan"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "positive",
            "reason_codes": [],
            "comment": "Looks good now.",
        },
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["id"] == first.json()["id"]
    assert second_body["sentiment"] == "positive"
    assert second_body["snapshot"]["plan_id"] == plan_id
    assert second_body["snapshot"]["task_count"] == 1

    filtered = client.get(
        f"/feedback?target_type=plan&target_id={plan_id}&actor_key=tester-plan"
    )
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["summary"]["total"] == 1
    assert filtered_body["items"][0]["comment"] == "Looks good now."


def test_intent_feedback_summary_and_examples_include_intent_dimensions() -> None:
    job_response = client.post(
        "/jobs",
        json={"goal": "Create a release report", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    _update_job_intent_metadata(
        job["id"],
        intent="io",
        source="llm",
        clarification_mode="intent_disagreement",
        disagreement_reason="capability_intent_conflict",
        candidate_capability="document.spec.generate",
        missing_inputs=["intent_action"],
    )

    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-dims-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_scope"],
            "comment": "This should have been treated as generation.",
        },
    )
    assert feedback_response.status_code == 200

    summary_response = client.get(
        "/feedback/summary",
        params={"target_type": "intent_assessment"},
    )
    assert summary_response.status_code == 200
    summary_body = summary_response.json()
    assert any(bucket["key"] == "io" for bucket in summary_body["intent_assessment_intents"])
    assert any(bucket["key"] == "llm" for bucket in summary_body["intent_assessment_sources"])
    assert any(
        bucket["key"] == "intent_disagreement"
        for bucket in summary_body["intent_clarification_modes"]
    )
    assert any(
        bucket["key"] == "capability_intent_conflict"
        for bucket in summary_body["intent_disagreement_reasons"]
    )

    export_response = client.get(
        "/feedback/examples",
        params={
            "target_type": "intent_assessment",
            "reason_code": "wrong_scope",
        },
    )
    assert export_response.status_code == 200
    body = export_response.json()
    assert body["total"] >= 1
    example = body["items"][0]
    assert example["dimensions"]["intent_assessment_intent"] == "io"
    assert example["dimensions"]["intent_assessment_source"] == "llm"
    assert example["dimensions"]["intent_clarification_mode"] == "intent_disagreement"
    assert example["dimensions"]["intent_disagreement_reason"] == "capability_intent_conflict"
    assert example["dimensions"]["intent_top_capability"] == "document.spec.generate"
    assert example["dimensions"]["intent_top_family"] == "document"


def test_job_feedback_endpoint_returns_intent_plan_and_outcome_feedback() -> None:
    job_response = client.post(
        "/jobs",
        json={"goal": "Plan a report", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    plan_id, _task_id = _create_plan_for_job(job["id"])

    intent_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-job"},
        json={
            "target_type": "intent_assessment",
            "target_id": job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_scope"],
            "comment": "It missed the output constraints.",
        },
    )
    assert intent_feedback.status_code == 200
    assert intent_feedback.json()["snapshot"]["normalized_intent_envelope"]["goal"] == "Plan a report"

    plan_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-job"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "positive",
            "reason_codes": [],
            "comment": "Plan structure is good.",
        },
    )
    assert plan_feedback.status_code == 200

    outcome_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-job"},
        json={
            "target_type": "job_outcome",
            "target_id": job["id"],
            "sentiment": "partial",
            "score": 3,
            "reason_codes": ["did_not_finish"],
            "comment": "Good direction, but incomplete.",
        },
    )
    assert outcome_feedback.status_code == 200
    assert outcome_feedback.json()["snapshot"]["job_id"] == job["id"]

    response = client.get(f"/jobs/{job['id']}/feedback")
    assert response.status_code == 200
    body = response.json()
    target_types = {item["target_type"] for item in body["items"]}
    assert "intent_assessment" in target_types
    assert "plan" in target_types
    assert "job_outcome" in target_types


def test_submit_feedback_returns_not_found_for_unknown_target() -> None:
    response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": "tester-missing"},
        json={
            "target_type": "chat_message",
            "target_id": "missing-message",
            "sentiment": "negative",
            "reason_codes": ["incorrect"],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "feedback_target_not_found"


def test_feedback_summary_filters_breaks_down_dimensions_and_correlates() -> None:
    suffix = uuid.uuid4().hex[:8]
    workflow_source = f"summary-source-{suffix}"
    llm_model = f"summary-model-{suffix}"
    planner_version = f"summary-planner-{suffix}"

    job_response = client.post(
        "/jobs",
        json={"goal": "Assemble an incident write-up", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    plan_id, task_id = _create_plan_for_job(job["id"], planner_version=planner_version)
    _update_job(
        job["id"],
        metadata={
            "workflow_source": workflow_source,
            "llm_provider": "openai",
            "llm_model": llm_model,
            "replan_count": 2,
            "plan_error": "planner_validation_failed",
        },
        status="failed",
    )
    _update_task(task_id, status="failed", rework_count=1)
    _attach_chat_session_to_job(
        job["id"],
        clarification_questions=["What level of detail should the write-up include?"],
    )

    intent_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"summary-intent-{suffix}"},
        json={
            "target_type": "intent_assessment",
            "target_id": job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_scope"],
            "comment": "The request scope was too broad.",
        },
    )
    assert intent_feedback.status_code == 200
    assert intent_feedback.json()["metadata"]["dimensions"]["workflow_source"] == workflow_source
    assert intent_feedback.json()["metadata"]["dimensions"]["llm_model"] == llm_model
    assert intent_feedback.json()["metadata"]["dimensions"]["planner_version"] == planner_version
    assert intent_feedback.json()["metadata"]["dimensions"]["job_status_at_feedback"] == "failed"

    plan_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"summary-plan-{suffix}"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "negative",
            "reason_codes": ["missing_step"],
            "comment": "The review step is missing.",
        },
    )
    assert plan_feedback.status_code == 200

    outcome_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"summary-outcome-{suffix}"},
        json={
            "target_type": "job_outcome",
            "target_id": job["id"],
            "sentiment": "partial",
            "score": 3,
            "reason_codes": ["did_not_finish"],
            "comment": "The draft never completed.",
        },
    )
    assert outcome_feedback.status_code == 200

    response = client.get(
        "/feedback/summary",
        params={
            "workflow_source": workflow_source,
            "llm_model": llm_model,
            "planner_version": planner_version,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["sentiment_counts"]["negative"] == 2
    assert body["sentiment_counts"]["partial"] == 1
    assert any(
        bucket["key"] == "intent_assessment" and bucket["total"] == 1
        for bucket in body["target_type_counts"]
    )
    assert any(bucket["key"] == workflow_source and bucket["total"] == 3 for bucket in body["workflow_sources"])
    assert any(bucket["key"] == llm_model and bucket["total"] == 3 for bucket in body["llm_models"])
    assert any(
        bucket["key"] == planner_version and bucket["total"] == 3
        for bucket in body["planner_versions"]
    )
    assert any(bucket["key"] == "failed" and bucket["total"] == 3 for bucket in body["job_statuses"])
    assert set(reason["reason_code"] for reason in body["negative_reasons"]) >= {
        "wrong_scope",
        "missing_step",
        "did_not_finish",
    }
    assert body["metrics"]["intent_agreement_rate"] == 0.0
    assert body["metrics"]["plan_approval_rate"] == 0.0
    assert body["metrics"]["job_outcome_positive_rate"] == 0.0
    assert body["correlates"]["job_count"] == 1
    assert body["correlates"]["replan_count"] == 2
    assert body["correlates"]["retry_count"] == 1
    assert body["correlates"]["failed_task_count"] == 1
    assert body["correlates"]["plan_failure_count"] == 1
    assert body["correlates"]["clarification_turn_count"] == 1
    assert any(
        bucket["key"] == "failed" and bucket["total"] == 1
        for bucket in body["correlates"]["terminal_statuses"]
    )

    plan_only = client.get(
        "/feedback/summary",
        params={
            "workflow_source": workflow_source,
            "target_type": "plan",
        },
    )
    assert plan_only.status_code == 200
    assert plan_only.json()["total"] == 1


def test_feedback_examples_export_supports_filters_and_jsonl() -> None:
    suffix = uuid.uuid4().hex[:8]
    workflow_source = f"examples-source-{suffix}"
    llm_model = f"examples-model-{suffix}"
    planner_version = f"examples-planner-{suffix}"

    job_response = client.post(
        "/jobs",
        json={"goal": "Draft a troubleshooting guide", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    plan_id, _task_id = _create_plan_for_job(job["id"], planner_version=planner_version)
    _update_job(
        job["id"],
        metadata={
            "workflow_source": workflow_source,
            "llm_provider": "openai",
            "llm_model": llm_model,
        },
        status="failed",
    )

    negative_plan = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"examples-plan-{suffix}"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "negative",
            "reason_codes": ["missing_step"],
            "comment": "A remediation step is missing.",
        },
    )
    assert negative_plan.status_code == 200

    partial_outcome = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"examples-outcome-{suffix}"},
        json={
            "target_type": "job_outcome",
            "target_id": job["id"],
            "sentiment": "partial",
            "reason_codes": ["did_not_finish"],
            "comment": "The guide stopped midway.",
        },
    )
    assert partial_outcome.status_code == 200

    positive_intent = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"examples-intent-{suffix}"},
        json={
            "target_type": "intent_assessment",
            "target_id": job["id"],
            "sentiment": "positive",
            "reason_codes": [],
            "comment": "Intent looked right.",
        },
    )
    assert positive_intent.status_code == 200

    export_response = client.get(
        "/feedback/examples",
        params={
            "llm_model": llm_model,
            "planner_version": planner_version,
        },
    )
    assert export_response.status_code == 200
    body = export_response.json()
    assert body["total"] == 2
    assert {item["feedback"]["sentiment"] for item in body["items"]} == {"negative", "partial"}
    assert all(item["dimensions"]["llm_model"] == llm_model for item in body["items"])
    assert all(item["dimensions"]["planner_version"] == planner_version for item in body["items"])
    assert all(item["linked_ids"]["job_id"] == job["id"] for item in body["items"])
    assert any(item["feedback"]["target_type"] == "plan" for item in body["items"])
    assert any(item["feedback"]["target_type"] == "job_outcome" for item in body["items"])

    filtered_reason = client.get(
        "/feedback/examples",
        params={
            "llm_model": llm_model,
            "planner_version": planner_version,
            "reason_code": "did_not_finish",
        },
    )
    assert filtered_reason.status_code == 200
    filtered_body = filtered_reason.json()
    assert filtered_body["total"] == 1
    assert filtered_body["items"][0]["feedback"]["target_type"] == "job_outcome"


def test_intent_review_queue_returns_labeled_items_and_filters() -> None:
    wrong_job = client.post(
        "/jobs",
        json={"goal": "Create a release report", "context_json": {}, "priority": 0},
    ).json()
    _update_job_intent_metadata(
        wrong_job["id"],
        intent="io",
        source="llm",
        clarification_mode="intent_disagreement",
        disagreement_reason="graph_intent_conflict",
        candidate_capability="document.spec.generate",
        missing_inputs=["intent_action"],
    )
    wrong_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-review-wrong-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": wrong_job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_goal"],
            "comment": "This intent was classified incorrectly.",
        },
    )
    assert wrong_feedback.status_code == 200

    unnecessary_job = client.post(
        "/jobs",
        json={"goal": "List GitHub issues", "context_json": {}, "priority": 0},
    ).json()
    _update_job_intent_metadata(
        unnecessary_job["id"],
        intent="io",
        source="heuristic",
        clarification_mode="targeted_slot_filling",
        candidate_capability="github.issue.search",
        missing_inputs=[],
    )
    unnecessary_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-review-unnecessary-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": unnecessary_job["id"],
            "sentiment": "partial",
            "reason_codes": ["asked_unnecessary_clarification"],
            "comment": "The system asked an unnecessary question.",
        },
    )
    assert unnecessary_feedback.status_code == 200

    response = client.get("/feedback/intent/review")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    labels = [item["review_label"] for item in body["items"]]
    assert "likely_wrong_intent_interpretation" in labels
    assert "likely_unnecessary_intent_clarification" in labels

    filtered = client.get(
        "/feedback/intent/review",
        params={
            "review_label": "likely_wrong_intent_interpretation",
            "intent_source": "llm",
        },
    )
    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert filtered_body["total"] >= 1
    assert all(
        item["review_label"] == "likely_wrong_intent_interpretation"
        for item in filtered_body["items"]
    )
    assert all(
        item["dimensions"]["intent_assessment_source"] == "llm"
        for item in filtered_body["items"]
    )


def test_intent_tuning_report_summarizes_failure_slices() -> None:
    wrong_job = client.post(
        "/jobs",
        json={"goal": "Create a release report", "context_json": {}, "priority": 0},
    ).json()
    _update_job_intent_metadata(
        wrong_job["id"],
        intent="io",
        source="llm",
        clarification_mode="intent_disagreement",
        disagreement_reason="graph_intent_conflict",
        candidate_capability="document.spec.generate",
        missing_inputs=["intent_action"],
    )
    wrong_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-tuning-wrong-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": wrong_job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_scope"],
            "comment": "The system chose the wrong intent.",
        },
    )
    assert wrong_feedback.status_code == 200

    missing_slot_job = client.post(
        "/jobs",
        json={"goal": "Render the release report", "context_json": {}, "priority": 0},
    ).json()
    _update_job_intent_metadata(
        missing_slot_job["id"],
        intent="render",
        source="heuristic",
        clarification_mode="capability_required_inputs",
        candidate_capability="document.pdf.render",
        missing_inputs=["path", "output_format"],
    )
    missing_slot_feedback = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-tuning-missing-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": missing_slot_job["id"],
            "sentiment": "partial",
            "reason_codes": ["missed_constraint"],
            "comment": "The system missed a required path/detail.",
        },
    )
    assert missing_slot_feedback.status_code == 200

    response = client.get("/feedback/intent/tuning-report")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 2
    assert any(
        bucket["key"] == "likely_wrong_intent_interpretation"
        for bucket in body["review_labels"]
    )
    assert any(
        bucket["key"] == "likely_missing_constraint_or_slot"
        for bucket in body["review_labels"]
    )
    assert any(
        bucket["key"] == "assessment_prompt_and_capability_evidence"
        for bucket in body["tuning_focuses"]
    )
    assert any(bucket["key"] == "document.spec.generate" for bucket in body["intent_top_capabilities"])
    assert any(bucket["key"] == "document" for bucket in body["intent_top_families"])
    assert any(bucket["key"] == "2" for bucket in body["missing_input_counts"])
    assert set(reason["reason_code"] for reason in body["negative_reasons"]) >= {
        "wrong_scope",
        "missed_constraint",
    }


def test_intent_tuning_candidates_export_observed_case_and_gold_stub() -> None:
    job = client.post(
        "/jobs",
        json={"goal": "Create a compliance report", "context_json": {}, "priority": 0},
    ).json()
    _update_job_intent_metadata(
        job["id"],
        intent="io",
        source="llm",
        clarification_mode="intent_disagreement",
        disagreement_reason="graph_intent_conflict",
        candidate_capability="document.spec.generate",
        missing_inputs=["intent_action"],
    )
    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"intent-tuning-candidate-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "intent_assessment",
            "target_id": job["id"],
            "sentiment": "negative",
            "reason_codes": ["wrong_goal"],
            "comment": "This should have been treated as report generation.",
        },
    )
    assert feedback_response.status_code == 200

    response = client.get(
        "/feedback/intent/tuning-candidates",
        params={"review_label": "likely_wrong_intent_interpretation"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    candidate = next(
        item for item in body["items"] if item["feedback"]["target_id"] == job["id"]
    )
    assert candidate["review_label"] == "likely_wrong_intent_interpretation"
    assert candidate["tuning_focus"] == "assessment_prompt_and_capability_evidence"
    assert candidate["suggested_case_id"]
    assert candidate["observed_case"]["goal"] == "Create a compliance report"
    assert candidate["observed_case"]["profile_intent"] == "io"
    assert candidate["observed_case"]["candidate_capabilities"] == ["document.spec.generate"]
    assert candidate["gold_case_stub"]["goal"] == "Create a compliance report"
    assert candidate["gold_case_stub"]["expected_requires_clarification"] is True
    assert candidate["gold_case_stub"]["_review_label"] == "likely_wrong_intent_interpretation"


def test_feedback_metrics_increment_for_submit_summary_and_export() -> None:
    job_response = client.post(
        "/jobs",
        json={"goal": "Review a deployment plan", "context_json": {}, "priority": 0},
    )
    assert job_response.status_code == 200
    job = job_response.json()
    plan_id, _task_id = _create_plan_for_job(job["id"])

    before_metrics = client.get("/metrics")
    assert before_metrics.status_code == 200
    before_text = before_metrics.text
    submitted_before = _metric_value(
        before_text,
        "feedback_submitted_total",
        {"target_type": "plan", "sentiment": "negative"},
    )
    reason_before = _metric_value(
        before_text,
        "feedback_reason_total",
        {"target_type": "plan", "reason_code": "missing_step"},
    )
    summary_before = _metric_value(before_text, "feedback_summary_requests_total")
    export_before = _metric_value(before_text, "feedback_examples_export_total")

    feedback_response = client.post(
        "/feedback",
        headers={"X-Feedback-Actor-Id": f"metrics-plan-{uuid.uuid4().hex[:8]}"},
        json={
            "target_type": "plan",
            "target_id": plan_id,
            "sentiment": "negative",
            "reason_codes": ["missing_step"],
            "comment": "Metrics test negative plan feedback.",
        },
    )
    assert feedback_response.status_code == 200

    summary_response = client.get("/feedback/summary")
    assert summary_response.status_code == 200
    export_response = client.get("/feedback/examples", params={"target_type": "plan"})
    assert export_response.status_code == 200

    after_metrics = client.get("/metrics")
    assert after_metrics.status_code == 200
    after_text = after_metrics.text
    assert (
        _metric_value(
            after_text,
            "feedback_submitted_total",
            {"target_type": "plan", "sentiment": "negative"},
        )
        >= submitted_before + 1
    )
    assert (
        _metric_value(
            after_text,
            "feedback_reason_total",
            {"target_type": "plan", "reason_code": "missing_step"},
        )
        >= reason_before + 1
    )
    assert _metric_value(after_text, "feedback_summary_requests_total") >= summary_before + 1
    assert _metric_value(after_text, "feedback_examples_export_total") >= export_before + 1
