from services.worker.app import main


def test_semantic_should_capture_matches_pattern(monkeypatch):
    monkeypatch.setattr(main, "SEMANTIC_MEMORY_AUTO_WRITE_ENABLED", True)
    monkeypatch.setattr(main, "SEMANTIC_MEMORY_AUTO_WRITE_TOOL_PATTERNS", ["llm.*", "document.spec.*"])
    assert main._semantic_should_capture("llm.text.generate") is True
    assert main._semantic_should_capture("document.spec.generate") is True
    assert main._semantic_should_capture("memory.read") is False


def test_auto_persist_semantic_facts_writes_distilled_entries(monkeypatch):
    writes = []

    class _MemoryClient:
        def semantic_write(self, payload):
            writes.append(payload)
            return {"entry": {"id": "ok"}}

    monkeypatch.setattr(main, "MEMORY_CLIENT", _MemoryClient())
    monkeypatch.setattr(main, "SEMANTIC_MEMORY_AUTO_WRITE_ENABLED", True)
    monkeypatch.setattr(main, "SEMANTIC_MEMORY_AUTO_WRITE_TOOL_PATTERNS", ["llm_generate"])
    monkeypatch.setattr(main, "SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS", 2)

    output = {
        "summary": (
            "Agent workflows should keep semantic memory for stable facts and preferences. "
            "Retries should use bounded backoff to avoid cascading failures in MCP calls."
        )
    }
    task_payload = {
        "job_id": "job-1",
        "task_id": "task-1",
        "name": "Improve_DocumentSpec",
        "intent": "transform",
        "user_id": "user-1",
    }

    main._auto_persist_semantic_facts(
        tool_name="llm_generate",
        task_payload=task_payload,
        output=output,
        trace_id="trace-1",
        run_id="run-1",
    )

    assert len(writes) >= 1
    first = writes[0]
    assert first["fact"]
    assert first["subject"] == "Improve_DocumentSpec"
    assert first["namespace"] == "transform"
    assert first["user_id"] == "user-1"
    assert first["metadata"]["auto_distilled"] is True
