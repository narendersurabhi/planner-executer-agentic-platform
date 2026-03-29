import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["ORCHESTRATOR_ENABLED"] = "false"
os.environ["JOB_RECOVERY_ENABLED"] = "false"
os.environ["POLICY_GATE_ENABLED"] = "false"
os.environ["CAPABILITY_MODE"] = "enabled"

from services.api.app import context_service, memory_profile_service  # noqa: E402
from services.api.app.database import Base, SessionLocal, engine  # noqa: E402
from libs.core import workflow_contracts  # noqa: E402


Base.metadata.create_all(bind=engine)


def test_build_chat_context_envelope_loads_profile_and_projects_views() -> None:
    db = SessionLocal()
    try:
        memory_profile_service.write_user_profile(
            db,
            user_id="alice",
            payload={
                "preferences": {
                    "preferred_output_format": "markdown",
                    "response_verbosity": "concise",
                }
            },
        )
        normalized = workflow_contracts.NormalizedIntentEnvelope(
            goal="Search GitHub issues",
            profile=workflow_contracts.GoalIntentProfile(
                intent="io",
                source="test",
                confidence=0.95,
                risk_level="read_only",
                threshold=0.7,
                low_confidence=False,
                needs_clarification=True,
                requires_blocking_clarification=True,
                questions=["What GitHub issues should I search for?"],
                blocking_slots=["query"],
                missing_slots=["query"],
                slot_values={"intent_action": "io", "risk_level": "read_only"},
            ),
            graph=workflow_contracts.IntentGraph(
                segments=[
                    workflow_contracts.IntentGraphSegment(
                        id="s1",
                        intent="io",
                        objective="Search GitHub issues",
                        required_inputs=["query"],
                        suggested_capabilities=["github.issue.search"],
                    )
                ]
            ),
            candidate_capabilities={"s1": ["github.issue.search"]},
            clarification=workflow_contracts.ClarificationState(
                needs_clarification=True,
                requires_blocking_clarification=True,
                missing_inputs=["query"],
                questions=["What GitHub issues should I search for?"],
                blocking_slots=["query"],
                slot_values={"intent_action": "io", "risk_level": "read_only"},
            ),
        )
        envelope = context_service.build_chat_context_envelope(
            db=db,
            goal="Search GitHub issues",
            session_metadata={
                "draft_goal": "Search GitHub issues",
                "pending_clarification": {"questions": ["What GitHub issues should I search for?"]},
            },
            session_context={
                "workflow_inputs": {"repo": "planner-executer-agentic-platform"},
                "interaction_summaries": [{"facts": ["uses github"], "action": "search"}],
            },
            turn_context={"query_hint": "authentication failures"},
            user_id="alice",
            normalized_intent_envelope=normalized,
            runtime_metadata={"stage": "chat"},
        )
    finally:
        db.close()

    assert envelope.user_scope["user_id"] == "alice"
    assert envelope.profile["preferences"]["preferred_output_format"] == "markdown"
    assert envelope.capability_candidates == ["github.issue.search"]
    assert envelope.missing_inputs == ["query"]
    assert envelope.session_scope["draft_goal"] == "Search GitHub issues"
    assert envelope.workflow_scope["workflow_inputs"]["repo"] == "planner-executer-agentic-platform"
    assert envelope.trace.profile_loaded is True
    assert "user_profile" in envelope.trace.sources_used

    route_view = context_service.chat_route_context_view(envelope)
    submit_view = context_service.chat_submit_context_view(envelope)
    assert route_view["user_profile"]["preferences"]["response_verbosity"] == "concise"
    assert route_view["capability_candidates"] == ["github.issue.search"]
    assert route_view["missing_inputs"] == ["query"]
    assert "user_profile" not in submit_view
    assert submit_view["query_hint"] == "authentication failures"


def test_build_workflow_runtime_context_envelope_merges_context_and_inputs() -> None:
    envelope, runtime_inputs = context_service.build_workflow_runtime_context_envelope(
        db=None,
        goal="Run workflow",
        version_context={"topic": "Release notes", "workflow_inputs": {"topic": "Version topic"}},
        trigger_context={"channel": "chat"},
        request_context={"topic": "Request topic", "workflow_context_json": {"priority": "high"}},
        trigger_inputs={"repo": "repo-a"},
        explicit_inputs={"repo": "repo-b", "topic": "Input topic"},
        runtime_metadata={"surface": "workflow_runtime"},
    )

    runtime_view = context_service.workflow_runtime_context_view(envelope)
    assert runtime_view["topic"] == "Request topic"
    assert runtime_view["channel"] == "chat"
    assert runtime_view["workflow_context_json"]["priority"] == "high"
    assert runtime_inputs == {"repo": "repo-b", "topic": "Input topic"}
    assert envelope.trace.sources_used[:3] == ["version_context", "trigger_context", "request_context"]


def test_build_preflight_context_envelope_prefers_provided_job_context() -> None:
    envelope = context_service.build_preflight_context_envelope(
        db=None,
        goal="Render report",
        provided_job_context={"path": "documents/provided.docx"},
        persisted_job_context={"path": "documents/persisted.docx", "topic": "Persisted"},
        runtime_metadata={"surface": "preflight"},
    )

    preflight_view = context_service.preflight_context_view(envelope)
    assert preflight_view["path"] == "documents/provided.docx"
    assert "topic" not in preflight_view
    assert envelope.trace.sources_used == ["provided_job_context"]


def test_chat_route_context_view_ranks_and_budgets_interaction_summaries() -> None:
    normalized = workflow_contracts.NormalizedIntentEnvelope(
        goal="Search GitHub issues",
        profile=workflow_contracts.GoalIntentProfile(
            intent="io",
            source="test",
            confidence=0.9,
            risk_level="read_only",
            missing_slots=["query"],
        ),
        graph=workflow_contracts.IntentGraph(
            segments=[
                workflow_contracts.IntentGraphSegment(
                    id="s1",
                    intent="io",
                    objective="Search GitHub issues",
                    suggested_capabilities=[
                        "filesystem.workspace.list",
                        "github.issue.search",
                        "memory.read",
                    ],
                )
            ]
        ),
        candidate_capabilities={
            "s1": [
                "filesystem.workspace.list",
                "github.issue.search",
                "memory.read",
            ]
        },
        clarification=workflow_contracts.ClarificationState(
            missing_inputs=["query"],
        ),
    )
    envelope = context_service.build_chat_context_envelope(
        db=None,
        goal="Search GitHub issues",
        session_metadata=None,
        session_context={
            "interaction_summaries": [
                {"facts": ["thanks"], "action": "thanks"},
                {"facts": ["github issues for auth"], "action": "search github auth issues"},
                {"facts": ["workspace files"], "action": "list workspace"},
                {"facts": ["github login failures"], "action": "search github login"},
                {"facts": ["github auth tokens"], "action": "search github auth tokens"},
            ]
        },
        turn_context=None,
        user_id=None,
        normalized_intent_envelope=normalized,
    )

    route_view = context_service.chat_route_context_view(envelope)

    assert route_view["capability_candidates"][0] == "github.issue.search"
    assert len(route_view["interaction_summaries"]) == 3
    assert all(item["action"] != "thanks" for item in route_view["interaction_summaries"])
    assert "interaction_summaries:noise" in envelope.dropped_inputs


def test_derive_missing_inputs_uses_aliases_from_context() -> None:
    normalized = workflow_contracts.NormalizedIntentEnvelope(
        goal="List workspace files",
        profile=workflow_contracts.GoalIntentProfile(missing_slots=["query"]),
        clarification=workflow_contracts.ClarificationState(missing_inputs=["query"]),
    )

    missing = context_service.derive_missing_inputs(
        context={"path": "reports"},
        normalized_intent_envelope=normalized,
    )

    assert missing == []


def test_chat_submit_context_view_drops_interaction_summary_refs() -> None:
    envelope = workflow_contracts.ContextEnvelope(
        goal="Search GitHub issues",
        context_json={
            "query": "auth failures",
            "interaction_summaries": [{"facts": ["auth"], "action": "search github auth issues"}],
            "interaction_summaries_ref": {"memory_name": "interaction_summaries_compact"},
            "interaction_summaries_meta": {"count": 1},
        },
    )

    submit_view = context_service.chat_submit_context_view(envelope)

    assert submit_view["query"] == "auth failures"
    assert "interaction_summaries_ref" not in submit_view
    assert "interaction_summaries_meta" not in submit_view


def test_planner_and_execution_context_views_use_stage_specific_projection() -> None:
    envelope = workflow_contracts.ContextEnvelope(
        goal="Search GitHub issues",
        context_json={
            "query": "auth failures",
            "user_profile": {"preferences": {"response_verbosity": "concise"}},
            "interaction_summaries": [
                {"facts": ["thanks"], "action": "thanks"},
                {"facts": ["github auth failures"], "action": "search github auth issues"},
            ],
            "interaction_summaries_ref": {"memory_name": "interaction_summaries_compact"},
            "interaction_summaries_meta": {"count": 2},
            "capability_candidates": ["memory.read"],
            "missing_inputs": ["query"],
        },
        profile={"preferences": {"response_verbosity": "concise"}},
        capability_candidates=["github.issue.search", "memory.read"],
        missing_inputs=["query"],
    )

    planner_view = context_service.planner_context_view(envelope)
    execution_view = context_service.execution_context_view(envelope)

    assert planner_view["query"] == "auth failures"
    assert planner_view["capability_candidates"] == ["github.issue.search", "memory.read"]
    assert planner_view["missing_inputs"] == ["query"]
    assert "user_profile" not in planner_view
    assert "interaction_summaries_ref" not in planner_view
    assert len(planner_view["interaction_summaries"]) == 1

    assert execution_view["query"] == "auth failures"
    assert "capability_candidates" not in execution_view
    assert "missing_inputs" not in execution_view
    assert "user_profile" not in execution_view
    assert "interaction_summaries_ref" not in execution_view


def test_intent_context_view_merges_slot_values_with_provenance() -> None:
    normalized = workflow_contracts.NormalizedIntentEnvelope(
        goal="Generate a deployment report",
        profile=workflow_contracts.GoalIntentProfile(
            intent="generate",
            source="test",
            confidence=0.95,
            risk_level="read_only",
            slot_values={
                "intent_action": "generate",
                "topic": "Deployment report",
            },
        ),
        clarification=workflow_contracts.ClarificationState(
            slot_values={
                "tone": "practical",
                "topic": "Deployment report",
            },
        ),
    )
    envelope = workflow_contracts.ContextEnvelope(
        goal="Generate a deployment report",
        context_json={
            "tone": "executive",
            "clarification_normalization": {
                "source": "chat_clarification_normalizer",
                "fields": ["tone"],
            },
        },
        normalized_intent_envelope=workflow_contracts.dump_normalized_intent_envelope(normalized),
        profile={"preferences": {"response_verbosity": "concise"}},
        capability_candidates=["document.spec.generate"],
        missing_inputs=["output_format"],
    )

    intent_view = context_service.intent_context_view(envelope)

    assert intent_view["user_profile"]["preferences"]["response_verbosity"] == "concise"
    assert intent_view["capability_candidates"] == ["document.spec.generate"]
    assert intent_view["missing_inputs"] == ["output_format"]
    assert intent_view["intent_slot_values"]["tone"] == "executive"
    assert intent_view["intent_slot_values"]["topic"] == "Deployment report"
    assert intent_view["intent_slot_values"]["intent_action"] == "generate"
    assert intent_view["intent_slot_provenance"]["tone"] == "clarification_normalized"
    assert intent_view["intent_slot_provenance"]["topic"] == "inferred"
    assert intent_view["intent_slot_provenance"]["intent_action"] == "inferred"
