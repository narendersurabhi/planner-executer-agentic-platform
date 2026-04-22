import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from services.api.app.database import Base  # noqa: E402
from services.api.app.models import (  # noqa: E402
    AgentDefinitionRecord,
    AgentDefinitionVersionRecord,
)


def test_agent_definition_record_declares_phase_zero_table_shape() -> None:
    table = Base.metadata.tables["agent_definitions"]

    assert AgentDefinitionRecord.__tablename__ == "agent_definitions"
    assert table.c["id"].primary_key is True
    assert "name" in table.c
    assert "agent_capability_id" in table.c
    assert "instructions" in table.c
    assert "default_constraints" in table.c
    assert "model_config" in table.c
    assert "allowed_capability_ids" in table.c
    assert "memory_policy" in table.c
    assert "guardrail_policy" in table.c
    assert "workspace_policy" in table.c
    assert "metadata" in table.c
    assert "enabled" in table.c
    assert "user_id" in table.c
    assert "created_at" in table.c
    assert "updated_at" in table.c


def test_agent_definition_version_record_declares_phase_five_table_shape() -> None:
    table = Base.metadata.tables["agent_definition_versions"]

    assert AgentDefinitionVersionRecord.__tablename__ == "agent_definition_versions"
    assert table.c["id"].primary_key is True
    assert "agent_definition_id" in table.c
    assert "version_number" in table.c
    assert "agent_capability_id" in table.c
    assert "instructions" in table.c
    assert "default_constraints" in table.c
    assert "model_config" in table.c
    assert "allowed_capability_ids" in table.c
    assert "definition_metadata" in table.c
    assert "version_metadata" in table.c
    assert "published_by" in table.c
    assert "version_note" in table.c
    assert "definition_created_at" in table.c
    assert "definition_updated_at" in table.c
    assert "created_at" in table.c
