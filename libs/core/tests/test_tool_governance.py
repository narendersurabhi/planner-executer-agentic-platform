from pathlib import Path

from libs.core import tool_governance
from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool, ToolRegistry


def _register_tool(registry: ToolRegistry, name: str) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name=name,
                description=f"{name} tool",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload: payload,
        )
    )


def test_filter_registry_tools_applies_service_allow_rules(monkeypatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "tool_governance.yaml"
    cfg_path.write_text(
        """
tool_governance:
  mode: enforce
  services:
    worker:
      allow:
        - text_summarize
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TOOL_GOVERNANCE_ENABLED", "true")
    monkeypatch.setenv("TOOL_GOVERNANCE_MODE", "enforce")
    monkeypatch.setenv("TOOL_GOVERNANCE_CONFIG_PATH", str(cfg_path))

    registry = ToolRegistry()
    _register_tool(registry, "text_summarize")
    _register_tool(registry, "sleep")

    tool_governance.filter_registry_tools(registry, "worker")

    assert set(registry._tools.keys()) == {"text_summarize"}
