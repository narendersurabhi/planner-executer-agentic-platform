from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from libs.core import models


def test_agent_definition_contract_accepts_model_config_wire_field() -> None:
    definition = models.AgentDefinitionCreate(
        name="Codegen autonomous",
        agent_capability_id="codegen.autonomous",
        instructions="Implement the requested code change.",
        default_max_steps=6,
        model_config={"provider": "openai", "model": "gpt-5.4"},
    )

    assert definition.llm_config == {"provider": "openai", "model": "gpt-5.4"}

    dumped = definition.model_dump(mode="json", by_alias=True)

    assert dumped["model_config"] == {"provider": "openai", "model": "gpt-5.4"}
    assert "llm_config" not in dumped


def test_agent_definition_update_rejects_non_positive_default_max_steps() -> None:
    with pytest.raises(ValidationError):
        models.AgentDefinitionUpdate(default_max_steps=0)


def test_agent_definition_snapshot_uses_model_config_alias() -> None:
    captured_at = datetime.now(UTC)

    snapshot = models.AgentDefinitionSnapshot(
        agent_definition_id="agent-1",
        agent_definition_version_id="version-1",
        agent_definition_version_number=2,
        name="Codegen autonomous",
        agent_capability_id="codegen.autonomous",
        instructions="Implement the requested code change.",
        model_config={"provider": "openai"},
        captured_at=captured_at,
    )

    dumped = snapshot.model_dump(mode="json", by_alias=True)

    assert dumped["agent_definition_id"] == "agent-1"
    assert dumped["agent_definition_version_id"] == "version-1"
    assert dumped["agent_definition_version_number"] == 2
    assert dumped["model_config"] == {"provider": "openai"}
    assert dumped["captured_at"] == captured_at.isoformat().replace("+00:00", "Z")


def test_agent_definition_version_uses_model_config_alias() -> None:
    created_at = datetime.now(UTC)

    version = models.AgentDefinitionVersion(
        id="version-1",
        agent_definition_id="agent-1",
        version_number=1,
        name="Codegen autonomous",
        agent_capability_id="codegen.autonomous",
        instructions="Implement the requested code change.",
        model_config={"provider": "openai"},
        version_metadata={"label": "published"},
        created_at=created_at,
    )

    dumped = version.model_dump(mode="json", by_alias=True)

    assert dumped["id"] == "version-1"
    assert dumped["agent_definition_id"] == "agent-1"
    assert dumped["version_number"] == 1
    assert dumped["model_config"] == {"provider": "openai"}
    assert dumped["version_metadata"] == {"label": "published"}
    assert dumped["created_at"] == created_at.isoformat().replace("+00:00", "Z")
