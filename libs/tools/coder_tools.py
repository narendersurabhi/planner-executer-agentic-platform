from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
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


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out or list(default)


def _matches_any(path: str, patterns: list[str]) -> bool:
    pure = PurePosixPath(path)
    for pattern in patterns:
        if pure.match(pattern) or fnmatch.fnmatch(path, pattern):
            return True
    return False


def _collect_workspace_files_for_push(
    root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    files: list[dict[str, str]] = []
    skipped: dict[str, list[str]] = {
        "excluded": [],
        "too_large": [],
        "non_utf8": [],
    }
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _matches_any(rel, exclude_globs):
            skipped["excluded"].append(rel)
            continue
        if include_globs and not _matches_any(rel, include_globs):
            skipped["excluded"].append(rel)
            continue
        file_bytes = path.stat().st_size
        if file_bytes > max_file_bytes:
            skipped["too_large"].append(rel)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped["non_utf8"].append(rel)
            continue
        encoded_len = len(content.encode("utf-8"))
        if total_bytes + encoded_len > max_total_bytes:
            break
        files.append({"path": rel, "content": content})
        total_bytes += encoded_len
        if len(files) >= max_files:
            break
    return files, skipped


def coding_agent_publish_pr(
    payload: dict[str, Any],
    *,
    safe_workspace_path: Callable[[str, str], Path],
    invoke_capability: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    owner = payload.get("owner")
    repo = payload.get("repo")
    branch = payload.get("branch")
    base = payload.get("base", "main")
    workspace_path = payload.get("workspace_path")
    if not isinstance(owner, str) or not owner.strip():
        raise ToolExecutionError("Missing owner")
    if not isinstance(repo, str) or not repo.strip():
        raise ToolExecutionError("Missing repo")
    if not isinstance(branch, str) or not branch.strip():
        raise ToolExecutionError("Missing branch")
    if not isinstance(base, str) or not base.strip():
        raise ToolExecutionError("Missing base")
    if not isinstance(workspace_path, str) or not workspace_path.strip():
        raise ToolExecutionError("Missing workspace_path")

    include_globs = _as_str_list(payload.get("include_globs"), ["**/*"])
    exclude_globs = _as_str_list(
        payload.get("exclude_globs"),
        [".git/**", "**/.git/**", "IMPLEMENTATION_PLAN.md"],
    )
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    max_file_bytes = payload.get("max_file_bytes", 200_000)
    if not isinstance(max_file_bytes, int) or max_file_bytes < 1:
        max_file_bytes = 200_000
    max_total_bytes = payload.get("max_total_bytes", 2_000_000)
    if not isinstance(max_total_bytes, int) or max_total_bytes < 1:
        max_total_bytes = 2_000_000

    root = safe_workspace_path(workspace_path, "")
    if not root.exists() or not root.is_dir():
        raise ToolExecutionError("workspace_path not found or not a directory")

    files, skipped = _collect_workspace_files_for_push(
        root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )
    if not files:
        raise ToolExecutionError("No files selected for push")

    branch_create_result: dict[str, Any] = {}
    try:
        branch_create_result = invoke_capability(
            "github.branch.create",
            {"owner": owner, "repo": repo, "branch": branch, "from_branch": base},
        )
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if (
            "already exists" not in message
            and "reference already exists" not in message
            and "name already exists" not in message
        ):
            raise
        branch_create_result = {"status": "exists"}

    commit_message = payload.get("message")
    if not isinstance(commit_message, str) or not commit_message.strip():
        commit_message = f"chore(codegen): apply autonomous changes for {branch}"

    push_result = invoke_capability(
        "github.files.push",
        {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "files": files,
            "message": commit_message,
        },
    )

    pr_title = payload.get("title")
    if not isinstance(pr_title, str) or not pr_title.strip():
        pr_title = f"[codegen] {branch}"
    pr_body = payload.get("body")
    if not isinstance(pr_body, str):
        pr_body = (
            "Automated PR created by codegen.publish_pr.\n\n"
            f"- Branch: `{branch}`\n"
            f"- Base: `{base}`\n"
            f"- Files pushed: {len(files)}"
        )
    pr_head = payload.get("head")
    if not isinstance(pr_head, str) or not pr_head.strip():
        pr_head = branch

    pr_payload: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "title": pr_title,
        "head": pr_head,
        "base": base,
        "body": pr_body,
    }
    if isinstance(payload.get("draft"), bool):
        pr_payload["draft"] = payload.get("draft")
    if isinstance(payload.get("maintainer_can_modify"), bool):
        pr_payload["maintainer_can_modify"] = payload.get("maintainer_can_modify")
    pr_result = invoke_capability("github.pull_request.create", pr_payload)

    return {
        "branch": branch,
        "base": base,
        "selected_files": len(files),
        "selected_paths_preview": [entry["path"] for entry in files[:20]],
        "skipped": skipped,
        "branch_create": branch_create_result,
        "push_result": push_result,
        "pull_request": pr_result,
    }
