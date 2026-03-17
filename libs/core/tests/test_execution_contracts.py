from libs.core import execution_contracts


def test_build_task_execution_request_normalizes_payload() -> None:
    payload = {
        "task_id": "task-1",
        "job_id": "job-1",
        "correlation_id": "trace-1",
        "instruction": "Render the final document",
        "context": {"job_context": {"title": "Quarterly Review"}},
        "tool_requests": ["docx_generate_from_spec", "github.repo.list"],
        "tool_inputs": {
            "docx_generate_from_spec": {"path": "artifacts/report.docx"},
            "github.repo.list": {"owner": "narendersurabhi", "repo": "scientific-agent-lab"},
        },
        "intent": "Render",
        "attempts": 0,
        "max_attempts": "bad-value",
        "intent_segment": {
            "id": "s1",
            "intent": "render",
            "objective": "Render a DOCX artifact",
            "slots": {"must_have_inputs": ["document_spec", "path"]},
        },
    }

    request = execution_contracts.build_task_execution_request(
        payload,
        default_max_attempts=4,
    )

    assert request.task_id == "task-1"
    assert request.job_id == "job-1"
    assert request.trace_id == "trace-1"
    assert request.run_id == "trace-1"
    assert request.intent == "render"
    assert request.attempts == 1
    assert request.max_attempts == 4
    assert request.tool_requests == ["docx_generate_from_spec", "github.repo.list"]
    assert request.tool_inputs["docx_generate_from_spec"]["path"] == "artifacts/report.docx"
    assert request.requests[1].resolved_inputs == {
        "owner": "narendersurabhi",
        "repo": "scientific-agent-lab",
    }
    assert request.intent_segment is not None
    assert request.intent_segment.intent == "render"


def test_build_task_execution_request_reads_capability_bindings() -> None:
    request = execution_contracts.build_task_execution_request(
        {
            "task_id": "task-1",
            "tool_requests": ["github.repo.list"],
            "tool_inputs": {"github.repo.list": {}},
            "capability_bindings": {
                "github.repo.list": {
                    "capability_id": "github.repo.list",
                    "tool_name": "github.repo.list",
                    "adapter_type": "mcp",
                    "server_id": "github_local",
                }
            },
        }
    )

    binding = request.requests[0].capability_binding
    assert binding is not None
    assert binding.request_id == "github.repo.list"
    assert binding.capability_id == "github.repo.list"
    assert binding.adapter_type == "mcp"
    assert binding.server_id == "github_local"


def test_build_task_dispatch_payload_normalizes_and_roundtrips_execution_fields() -> None:
    payload = execution_contracts.build_task_dispatch_payload(
        {
            "task_id": "task-1",
            "job_id": "job-1",
            "plan_id": "plan-1",
            "correlation_id": "corr-1",
            "instruction": "Render the final report",
            "tool_requests": ["docx_generate_from_spec"],
            "tool_inputs": {
                "docx_generate_from_spec": {"path": "artifacts/report.docx"}
            },
            "attempts": 0,
            "max_attempts": 0,
            "critic_required": 0,
            "tool_inputs_resolved": True,
            "tool_inputs_validation": {"docx_generate_from_spec": "missing document_spec"},
        },
        default_max_attempts=3,
    )

    assert payload.task_id == "task-1"
    assert payload.id == "task-1"
    assert payload.trace_id == "corr-1"
    assert payload.correlation_id == "corr-1"
    assert payload.attempts == 1
    assert payload.max_attempts == 3
    assert payload.tool_requests == ["docx_generate_from_spec"]
    assert payload.tool_inputs["docx_generate_from_spec"]["path"] == "artifacts/report.docx"
    assert payload.critic_required is False
    assert payload.tool_inputs_resolved is True
    request = execution_contracts.build_task_execution_request(payload.model_dump(mode="json"))
    assert request.trace_id == "corr-1"
    assert request.tool_inputs["docx_generate_from_spec"]["path"] == "artifacts/report.docx"


def test_embed_capability_bindings_preserves_plain_tool_inputs() -> None:
    embedded = execution_contracts.embed_capability_bindings(
        {
            "github.repo.list": {"owner": "narendersurabhi", "repo": "scientific-agent-lab"}
        },
        {
            "github.repo.list": {
                "request_id": "github.repo.list",
                "capability_id": "github.repo.list",
                "tool_name": "github.repo.list",
                "adapter_type": "mcp",
            }
        },
        request_ids=["github.repo.list"],
    )

    assert embedded["github.repo.list"]["repo"] == "scientific-agent-lab"
    assert execution_contracts.EXECUTION_BINDINGS_KEY in embedded
    stripped = execution_contracts.strip_execution_metadata_from_tool_inputs(embedded)
    assert stripped == {
        "github.repo.list": {
            "owner": "narendersurabhi",
            "repo": "scientific-agent-lab",
        }
    }


def test_build_task_dispatch_payload_reads_embedded_capability_bindings() -> None:
    payload = execution_contracts.build_task_dispatch_payload(
        {
            "task_id": "task-1",
            "tool_requests": ["github.repo.list"],
            "tool_inputs": execution_contracts.embed_capability_bindings(
                {"github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}},
                {
                    "github.repo.list": {
                        "request_id": "github.repo.list",
                        "capability_id": "github.repo.list",
                        "tool_name": "github.repo.list",
                        "adapter_type": "mcp",
                        "server_id": "github_local",
                    }
                },
                request_ids=["github.repo.list"],
            ),
        }
    )

    assert payload.tool_inputs == {
        "github.repo.list": {"owner": "narendersurabhi", "repo": "demo"}
    }
    assert payload.capability_bindings["github.repo.list"].server_id == "github_local"


def test_embed_execution_gate_roundtrips_through_dispatch_payload() -> None:
    payload = execution_contracts.build_task_dispatch_payload(
        {
            "task_id": "task-1",
            "tool_requests": ["llm_generate"],
            "tool_inputs": execution_contracts.embed_execution_gate(
                {"llm_generate": {"text": "hello"}},
                {"expression": "context.approved == true"},
                request_ids=["llm_generate"],
            ),
        }
    )

    assert payload.tool_inputs == {"llm_generate": {"text": "hello"}}
    assert payload.execution_gates == {
        "llm_generate": {"expression": "context.approved == true"}
    }
    request = execution_contracts.build_task_execution_request(payload.model_dump(mode="json"))
    assert request.requests[0].execution_gate == {
        "expression": "context.approved == true"
    }
