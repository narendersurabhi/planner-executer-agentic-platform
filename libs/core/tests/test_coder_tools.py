from libs.core.llm_provider import LLMProvider, LLMResponse
from libs.framework.tool_runtime import ToolExecutionError
from libs.tools import coder_tools
from pathlib import Path


class _PlanProvider(LLMProvider):
    def generate(self, prompt: str) -> LLMResponse:
        del prompt
        return LLMResponse(
            content='{"steps":[{"title":"Create app","files":["src/app.py"]}]}'
        )


def test_coding_agent_generate_writes_workspace_files() -> None:
    writes: list[dict] = []

    def post_mcp_tool_call(service_url: str, tool_name: str, arguments: dict) -> dict:
        del service_url, tool_name, arguments
        return {"files": [{"path": "src/app.py", "content": "print(1)"}]}

    def write_workspace_text_file(payload: dict, default_filename: str) -> dict:
        del default_filename
        writes.append(payload)
        return {"path": payload["path"]}

    out = coder_tools.coding_agent_generate(
        {"goal": "build app", "workspace_path": "workspace"},
        post_mcp_tool_call=post_mcp_tool_call,
        write_workspace_text_file=write_workspace_text_file,
    )
    assert out["written_paths"] == ["workspace/src/app.py"]
    assert writes[0]["path"] == "workspace/src/app.py"


def test_coding_agent_autonomous_retries_invalid_json_once() -> None:
    calls: list[dict] = []
    writes: list[dict] = []

    def post_mcp_tool_call(service_url: str, tool_name: str, arguments: dict) -> dict:
        del service_url, tool_name
        calls.append(arguments)
        if len(calls) == 1:
            raise ToolExecutionError("coder_invalid_json:bad_response")
        return {"files": [{"path": "src/app.py", "content": "print(1)"}]}

    def write_workspace_text_file(payload: dict, default_filename: str) -> dict:
        del default_filename
        writes.append(payload)
        return {"path": payload["path"]}

    out = coder_tools.coding_agent_autonomous(
        {"goal": "build app", "workspace_path": "workspace", "max_steps": 1},
        _PlanProvider(),
        post_mcp_tool_call=post_mcp_tool_call,
        write_workspace_text_file=write_workspace_text_file,
        extract_json=lambda text: text,
    )

    assert out["steps_completed"] == 1
    assert len(calls) == 2
    assert "constraints" in calls[1]
    assert "STRICT JSON ONLY" in calls[1]["constraints"]
    # plan file + plan update + generated file
    assert len(writes) >= 3


def test_build_plan_prompt_includes_constraints() -> None:
    prompt = coder_tools.build_plan_prompt("Ship feature", "Use tests", 6)
    assert "Use tests" in prompt
    assert '"steps"' in prompt


def test_coding_agent_publish_pr_normalizes_same_branch_as_base(tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello", encoding="utf-8")

    def safe_workspace_path(workspace_path: str, relative_path: str) -> Path:
        del relative_path
        assert workspace_path == "workspace"
        return workspace

    def invoke_capability(name: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append((name, payload))
        return {"ok": True}

    out = coder_tools.coding_agent_publish_pr(
        {
            "owner": "narendersurabhi",
            "repo": "scientific-agent-lab",
            "branch": "main",
            "base": "main",
            "workspace_path": "workspace",
            "include_globs": ["README.md"],
        },
        safe_workspace_path=safe_workspace_path,
        invoke_capability=invoke_capability,
    )

    expected_branch = "codex/scientific-agent-lab"
    assert out["branch"] == expected_branch
    assert out["base"] == "main"
    assert calls[0] == (
        "github.branch.create",
        {
            "owner": "narendersurabhi",
            "repo": "scientific-agent-lab",
            "branch": expected_branch,
            "from_branch": "main",
        },
    )
    assert calls[1][0] == "github.files.push"
    assert calls[1][1]["branch"] == expected_branch
    assert calls[2][0] == "github.pull_request.create"
    assert calls[2][1]["head"] == expected_branch
    assert calls[2][1]["base"] == "main"
