from __future__ import annotations

from services.worker.app import main as worker_main


def test_task_attempt_limits_defaults(monkeypatch) -> None:
    monkeypatch.setattr(worker_main, "WORKER_DEFAULT_MAX_ATTEMPTS", 3)
    attempts, max_attempts = worker_main._task_attempt_limits({})
    assert attempts == 1
    assert max_attempts == 3


def test_task_attempt_limits_respects_task_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr(worker_main, "WORKER_DEFAULT_MAX_ATTEMPTS", 3)
    attempts, max_attempts = worker_main._task_attempt_limits({"attempts": 1, "max_attempts": 1})
    assert attempts == 1
    assert max_attempts == 1


def test_should_retry_transient_error(monkeypatch) -> None:
    monkeypatch.setattr(worker_main, "WORKER_RETRY_ENABLED", True)
    monkeypatch.setattr(worker_main, "WORKER_RETRY_POLICY", "transient")
    monkeypatch.setattr(worker_main, "WORKER_DEFAULT_MAX_ATTEMPTS", 3)
    task_payload = {"attempts": 1, "max_attempts": 3}
    assert worker_main._should_retry_task(task_payload, "coder_http_error:timed out") is True
    assert worker_main._should_retry_task(task_payload, "tool_intent_mismatch:foo") is False
    assert (
        worker_main._should_retry_task(
            task_payload,
            "mcp_tool_error:Error executing tool tailor_resume: "
            "tailored_resume_missing_fields:experience,education,certifications",
        )
        is False
    )


def test_should_retry_any_policy(monkeypatch) -> None:
    monkeypatch.setattr(worker_main, "WORKER_RETRY_ENABLED", True)
    monkeypatch.setattr(worker_main, "WORKER_RETRY_POLICY", "any")
    monkeypatch.setattr(worker_main, "WORKER_DEFAULT_MAX_ATTEMPTS", 3)
    task_payload = {"attempts": 1, "max_attempts": 3}
    assert worker_main._should_retry_task(task_payload, "non_transient") is True
    assert (
        worker_main._should_retry_task(task_payload, "unknown_tool:llm_generate_document_spec")
        is False
    )
    assert (
        worker_main._should_retry_task(
            task_payload,
            "contract.output_invalid:mcp_tool_error:Error executing tool tailor_resume:"
            " tailored_resume_missing_fields:experience",
        )
        is False
    )


def test_queue_task_retry_emits_ready_with_incremented_attempt(monkeypatch) -> None:
    emitted: list[tuple[str, dict]] = []

    def _fake_emit(event_type: str, envelope: dict, payload: dict) -> None:
        assert envelope["job_id"] == "j-1"
        emitted.append((event_type, payload))

    monkeypatch.setattr(worker_main, "_emit_task_event", _fake_emit)
    monkeypatch.setattr(worker_main, "WORKER_DEFAULT_MAX_ATTEMPTS", 3)
    task_payload = {"task_id": "t-1", "attempts": 1, "max_attempts": 3}
    envelope = {"job_id": "j-1", "correlation_id": "c-1", "task_id": "t-1"}

    worker_main._queue_task_retry(task_payload, envelope, "timed out")

    assert emitted[0][0] == "task.heartbeat"
    assert emitted[1][0] == "task.ready"
    assert emitted[1][1]["attempts"] == 2
    assert emitted[1][1]["max_attempts"] == 3
    assert emitted[1][1]["last_error"] == "timed out"
