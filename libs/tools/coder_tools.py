from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Callable

from libs.core.llm_provider import LLMProvider
from libs.framework.tool_runtime import ToolExecutionError


def post_json(url: str, payload: dict[str, Any], *, timeout_s: int) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise ToolExecutionError(f"coder_http_error:{detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise ToolExecutionError(f"coder_http_error:{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"coder_invalid_json:{exc}") from exc


def coding_agent_generate(
    payload: dict[str, Any],
    *,
    post_mcp_tool_call: Callable[[str, str, dict[str, Any]], dict[str, Any]],
    write_workspace_text_file: Callable[[dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ToolExecutionError("Missing goal")
    files = payload.get("files")
    if files is not None and not isinstance(files, list):
        raise ToolExecutionError("files must be a list of strings")
    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, str):
        raise ToolExecutionError("constraints must be a string")
    workspace_path = payload.get("workspace_path")
    if workspace_path is not None and not isinstance(workspace_path, str):
        raise ToolExecutionError("workspace_path must be a string")

    coder_url = os.getenv("CODER_API_URL", "http://coder:8000").rstrip("/")
    request_payload: dict[str, Any] = {"goal": goal}
    if files:
        request_payload["files"] = files
    if constraints:
        request_payload["constraints"] = constraints

    response = post_mcp_tool_call(
        coder_url,
        "generate_code",
        request_payload,
    )
    if not isinstance(response, dict):
        raise ToolExecutionError("coder_response_invalid")
    file_entries = response.get("files")
    if not isinstance(file_entries, list) or not file_entries:
        raise ToolExecutionError("coder_response_missing_files")

    written_paths: list[str] = []
    for entry in file_entries:
        if not isinstance(entry, dict):
            raise ToolExecutionError("coder_file_entry_invalid")
        path = entry.get("path")
        content = entry.get("content")
        if not isinstance(path, str) or not path.strip():
            raise ToolExecutionError("coder_file_entry_missing_path")
        if path.startswith("/") or path.startswith(".."):
            raise ToolExecutionError("coder_file_entry_invalid_path")
        if not isinstance(content, str):
            raise ToolExecutionError("coder_file_entry_missing_content")
        target_path = str(Path(workspace_path) / path) if workspace_path else path
        write_workspace_text_file(
            {"path": target_path, "content": content}, default_filename="output.txt"
        )
        written_paths.append(target_path)

    return {"files": file_entries, "written_paths": written_paths}


def build_plan_prompt(goal: str, constraints: str | None, max_steps: int) -> str:
    constraints_block = f"Constraints:\n{constraints}\n" if constraints else ""
    return (
        "You are a coding planner. Return ONLY JSON (no prose, no markdown).\n"
        "Output schema:\n"
        "{\n"
        '  "steps": [\n'
        '    {"title": "Short step title", "files": ["relative/path.ext"]}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        f"- Provide between 3 and {max_steps} steps.\n"
        "- Each step must list only the files it should create or modify.\n"
        "- File paths must be relative, no leading slash, no '..' segments.\n"
        "- Keep each step small: 5 files max per step.\n"
        "- Avoid large or binary assets; prefer small placeholder text or simple SVGs.\n"
        "- Do not include repos/, repositories/, or repo-name prefixes in file paths.\n"
        "- Keep the file list minimal but complete (include README, tests, Dockerfile if needed).\n"
        f"{constraints_block}"
        f"Goal:\n{goal}\n"
    )


def render_plan_markdown(goal: str, steps: list[dict[str, Any]], statuses: list[bool]) -> str:
    lines = ["# IMPLEMENTATION_PLAN", "", f"Goal: {goal}", "", "## Steps"]
    for idx, step in enumerate(steps, start=1):
        status = "x" if statuses[idx - 1] else " "
        title = step.get("title", f"Step {idx}")
        files = step.get("files", [])
        files_text = ", ".join(files) if isinstance(files, list) else ""
        lines.append(f"- [{status}] Step {idx}: {title} (files: {files_text})")
    lines.append("")
    return "\n".join(lines)


def coding_agent_autonomous(
    payload: dict[str, Any],
    provider: LLMProvider,
    *,
    post_mcp_tool_call: Callable[[str, str, dict[str, Any]], dict[str, Any]],
    write_workspace_text_file: Callable[[dict[str, Any], str], dict[str, Any]],
    extract_json: Callable[[str], str],
) -> dict[str, Any]:
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ToolExecutionError("Missing goal")
    workspace_path = payload.get("workspace_path")
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        raise ToolExecutionError("Missing workspace_path")
    constraints = payload.get("constraints")
    if constraints is not None and not isinstance(constraints, str):
        raise ToolExecutionError("constraints must be a string")
    max_steps = payload.get("max_steps") or 6
    if not isinstance(max_steps, int):
        max_steps = 6
    max_steps = max(3, min(12, max_steps))

    plan_prompt = build_plan_prompt(goal, constraints, max_steps)
    response = provider.generate(plan_prompt)
    plan_json = extract_json(response.content)
    if not plan_json:
        raise ToolExecutionError("plan_json_missing")
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"plan_json_invalid:{exc}") from exc
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ToolExecutionError("plan_steps_missing")
    steps = steps[:max_steps]
    statuses = [False for _ in steps]

    plan_path = f"{workspace_path.rstrip('/')}/IMPLEMENTATION_PLAN.md"
    plan_md = render_plan_markdown(goal, steps, statuses)
    write_workspace_text_file({"path": plan_path, "content": plan_md}, "IMPLEMENTATION_PLAN.md")

    coder_url = os.getenv("CODER_API_URL", "http://coder:8000").rstrip("/")
    written_paths: list[str] = []

    for idx, step in enumerate(steps, start=1):
        title = step.get("title", f"Step {idx}")
        files = step.get("files")
        if not isinstance(files, list) or not files:
            raise ToolExecutionError(f"plan_step_missing_files:{idx}")
        step_goal = (
            f"{goal}\n\nCurrent step {idx}: {title}\nOnly implement the files listed for this step."
        )
        request_payload: dict[str, Any] = {"goal": step_goal, "files": files}
        if constraints:
            request_payload["constraints"] = constraints
        attempts = 0
        step_response: dict[str, Any] | None = None
        last_error: str | None = None
        while attempts < 2 and step_response is None:
            attempts += 1
            try:
                step_response = post_mcp_tool_call(
                    coder_url,
                    "generate_code",
                    request_payload,
                )
            except ToolExecutionError as exc:
                last_error = str(exc)
                if "invalid_json" not in last_error:
                    raise
                tighten = (
                    "STRICT JSON ONLY. Return a single JSON object. "
                    "Escape newlines as \\n and avoid unescaped quotes. "
                    "Keep output minimal. Avoid large strings or binary data."
                )
                if request_payload.get("constraints"):
                    request_payload["constraints"] = f"{request_payload['constraints']}\n{tighten}"
                else:
                    request_payload["constraints"] = tighten
                step_response = None
        if step_response is None:
            raise ToolExecutionError(last_error or "coder_invalid_json")
        if not isinstance(step_response, dict):
            raise ToolExecutionError("coder_response_invalid")
        file_entries = step_response.get("files")
        if not isinstance(file_entries, list) or not file_entries:
            raise ToolExecutionError("coder_response_missing_files")
        for entry in file_entries:
            if not isinstance(entry, dict):
                raise ToolExecutionError("coder_file_entry_invalid")
            path = entry.get("path")
            content = entry.get("content")
            if not isinstance(path, str) or not path.strip():
                raise ToolExecutionError("coder_file_entry_missing_path")
            if path.startswith("/") or path.startswith(".."):
                raise ToolExecutionError("coder_file_entry_invalid_path")
            if not isinstance(content, str):
                raise ToolExecutionError("coder_file_entry_missing_content")
            target_path = str(Path(workspace_path) / path)
            write_workspace_text_file(
                {"path": target_path, "content": content}, default_filename="output.txt"
            )
            written_paths.append(target_path)
        statuses[idx - 1] = True
        plan_md = render_plan_markdown(goal, steps, statuses)
        write_workspace_text_file(
            {"path": plan_path, "content": plan_md}, "IMPLEMENTATION_PLAN.md"
        )

    return {
        "plan_path": plan_path,
        "steps_total": len(steps),
        "steps_completed": len(steps),
        "written_paths": written_paths,
    }
