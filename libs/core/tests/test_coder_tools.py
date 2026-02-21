from libs.core.llm_provider import LLMProvider, LLMResponse
from libs.framework.tool_runtime import ToolExecutionError
from libs.tools import coder_tools


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
