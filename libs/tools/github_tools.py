from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from subprocess import run
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from libs.core.models import RiskLevel, ToolIntent, ToolSpec


def register_github_tools(registry) -> None:
    from libs.core.tool_registry import Tool
    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_create",
                description="Create a GitHub repository (user or org)",
                usage_guidance=(
                    "Create a repository using the GitHub API. Provide name and optionally "
                    "description, private, org, auto_init, default_branch."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "private": {"type": "boolean"},
                        "org": {"type": "string"},
                        "auto_init": {"type": "boolean"},
                        "default_branch": {"type": "string"},
                    },
                    "required": ["name"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "full_name": {"type": "string"},
                        "html_url": {"type": "string"},
                        "clone_url": {"type": "string"},
                    },
                    "required": ["full_name", "html_url", "clone_url"],
                },
                timeout_s=15,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_create,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_update",
                description="Update repo metadata (description, visibility, default branch)",
                usage_guidance=(
                    "Update repository settings. Provide owner and repo, plus any fields to update."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "homepage": {"type": "string"},
                        "private": {"type": "boolean"},
                        "default_branch": {"type": "string"},
                    },
                    "required": ["owner", "repo"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"full_name": {"type": "string"}, "html_url": {"type": "string"}},
                    "required": ["full_name", "html_url"],
                },
                timeout_s=15,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_update,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_list",
                description="List repositories for the authenticated user or org",
                usage_guidance=(
                    "List repositories. Provide optional org to list organization repos."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "org": {"type": "string"},
                        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"repos": {"type": "array", "items": {"type": "object"}}},
                    "required": ["repos"],
                },
                timeout_s=15,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_list,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_branch_list",
                description="List branches for a repository",
                usage_guidance="Provide owner and repo. Optionally provide per_page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["owner", "repo"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"branches": {"type": "array", "items": {"type": "object"}}},
                    "required": ["branches"],
                },
                timeout_s=15,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_branch_list,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_file_write",
                description="Create or update a file in a GitHub repository",
                usage_guidance=(
                    "Write file content to a repo via the GitHub contents API. "
                    "Provide owner, repo, path, content, and commit message. "
                    "Branch defaults to main."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 1},
                        "message": {"type": "string", "minLength": 1},
                        "branch": {"type": "string"},
                        "sha": {"type": "string"},
                    },
                    "required": ["owner", "repo", "path", "content", "message"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "commit_sha": {"type": "string"}},
                    "required": ["path", "commit_sha"],
                },
                timeout_s=20,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_file_write,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_pr_create",
                description="Create a pull request",
                usage_guidance=(
                    "Create a PR. Provide owner, repo, title, head, base, and optional body."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "title": {"type": "string", "minLength": 1},
                        "head": {"type": "string", "minLength": 1},
                        "base": {"type": "string", "minLength": 1},
                        "body": {"type": "string"},
                    },
                    "required": ["owner", "repo", "title", "head", "base"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"html_url": {"type": "string"}, "number": {"type": "integer"}},
                    "required": ["html_url", "number"],
                },
                timeout_s=20,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_pr_create,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_push",
                description="Push a local workspace folder to a GitHub repository",
                usage_guidance=(
                    "Push local workspace files to GitHub. Provide owner, repo, path (workspace "
                    "relative), and optional branch and message. Uses GITHUB_TOKEN."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "branch": {"type": "string"},
                        "message": {"type": "string"},
                        "author_name": {"type": "string"},
                        "author_email": {"type": "string"},
                    },
                    "required": ["owner", "repo", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"branch": {"type": "string"}, "remote": {"type": "string"}},
                    "required": ["branch", "remote"],
                },
                timeout_s=60,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_push,
        )
    )
    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_clone_or_pull",
                description="Clone a GitHub repository into the workspace or pull if it already exists",
                usage_guidance=(
                    "Clone or update a repo under the workspace. Provide owner, repo, path "
                    "(workspace relative), and optional branch. Uses GITHUB_TOKEN. "
                    "Optional: pull_strategy ('rebase', 'merge', 'ff-only') and force_reset=true "
                    "to hard reset to origin/<branch>."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "branch": {"type": "string"},
                        "pull_strategy": {"type": "string"},
                        "force_reset": {"type": "boolean"},
                    },
                    "required": ["owner", "repo", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "action": {"type": "string"},
                        "branch": {"type": "string"},
                    },
                    "required": ["path", "action", "branch"],
                },
                timeout_s=120,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_clone_or_pull,
        )
    )
    registry.register(
        Tool(
            spec=ToolSpec(
                name="github_repo_push_pr",
                description="Create a branch from base, push workspace changes, and open a pull request",
                usage_guidance=(
                    "Provide owner, repo, path (workspace relative), base branch, and new branch name. "
                    "Optionally provide PR title/body, commit message, and author identity."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "minLength": 1},
                        "repo": {"type": "string", "minLength": 1},
                        "path": {"type": "string", "minLength": 1},
                        "base": {"type": "string"},
                        "branch": {"type": "string", "minLength": 1},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "message": {"type": "string"},
                        "author_name": {"type": "string"},
                        "author_email": {"type": "string"},
                    },
                    "required": ["owner", "repo", "path", "branch"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "branch": {"type": "string"},
                        "base": {"type": "string"},
                        "remote": {"type": "string"},
                        "pr_url": {"type": "string"},
                        "pr_number": {"type": "integer"},
                    },
                    "required": ["branch", "base", "remote", "pr_url", "pr_number"],
                },
                timeout_s=120,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.io,
            ),
            handler=_github_repo_push_pr,
        )
    )


def _github_repo_create(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing repo name")
    description = payload.get("description")
    if isinstance(description, str):
        description = description.strip()
        if len(description) > 350:
            description = description[:350]
    body = {
        "name": name.strip(),
        "description": description,
        "private": payload.get("private", True),
        "auto_init": payload.get("auto_init", False),
    }
    default_branch = payload.get("default_branch")
    if isinstance(default_branch, str) and default_branch.strip():
        body["default_branch"] = default_branch.strip()
    org = payload.get("org")
    if isinstance(org, str) and org.strip():
        data = _github_request("POST", f"/orgs/{org.strip()}/repos", body=body)
    else:
        data = _github_request("POST", "/user/repos", body=body)
    return {
        "full_name": data.get("full_name", ""),
        "html_url": data.get("html_url", ""),
        "clone_url": data.get("clone_url", ""),
    }


def _github_repo_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    body: Dict[str, Any] = {}
    for key in ("description", "homepage", "private", "default_branch"):
        if key in payload:
            body[key] = payload.get(key)
    if isinstance(body.get("description"), str):
        desc = body["description"].strip()
        if len(desc) > 350:
            desc = desc[:350]
        body["description"] = desc
    if not body:
        raise ValueError("No update fields provided")
    data = _github_request("PATCH", f"/repos/{owner}/{repo}", body=body)
    return {"full_name": data.get("full_name", ""), "html_url": data.get("html_url", "")}


def _github_repo_list(payload: Dict[str, Any]) -> Dict[str, Any]:
    org = payload.get("org")
    if isinstance(org, str):
        normalized = org.strip().lower()
        if normalized in {"", "authenticated-user", "me", "self", "personal", "user"}:
            org = None
    per_page = _clamp_per_page(payload.get("per_page"))
    params = {"per_page": per_page}
    if isinstance(org, str) and org.strip():
        data = _github_request("GET", f"/orgs/{org.strip()}/repos", params=params)
    else:
        data = _github_request("GET", "/user/repos", params=params)
    repos: list[dict[str, Any]] = []
    for repo in data or []:
        if not isinstance(repo, dict):
            continue
        repos.append(
            {
                "name": repo.get("name", ""),
                "full_name": repo.get("full_name", ""),
                "html_url": repo.get("html_url", ""),
                "private": repo.get("private", False),
                "default_branch": repo.get("default_branch", ""),
            }
        )
    return {"repos": repos}


def _github_branch_list(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    per_page = _clamp_per_page(payload.get("per_page"))
    data = _github_request("GET", f"/repos/{owner}/{repo}/branches", params={"per_page": per_page})
    return {"branches": data}


def _github_file_write(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    path = _required(payload, "path")
    content = _required(payload, "content")
    message = _required(payload, "message")
    branch = payload.get("branch") or "main"
    sha = payload.get("sha")
    if not sha:
        try:
            existing = _github_request(
                "GET", f"/repos/{owner}/{repo}/contents/{path}", params={"ref": branch}
            )
            if isinstance(existing, dict):
                sha = existing.get("sha")
        except ValueError:
            sha = None
    body = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    data = _github_request("PUT", f"/repos/{owner}/{repo}/contents/{path}", body=body)
    commit = data.get("commit") or {}
    return {"path": data.get("content", {}).get("path", path), "commit_sha": commit.get("sha", "")}


def _github_pr_create(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    title = _required(payload, "title")
    head = _required(payload, "head")
    base = _required(payload, "base")
    body = payload.get("body")
    data = _github_request(
        "POST",
        f"/repos/{owner}/{repo}/pulls",
        body={"title": title, "head": head, "base": base, "body": body},
    )
    return {"html_url": data.get("html_url", ""), "number": data.get("number", 0)}


def _github_repo_push(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    rel_path = _required(payload, "path")
    branch = payload.get("branch") or "main"
    message = payload.get("message") or "Sync repository"
    author_name = payload.get("author_name") or "agentic-bot"
    author_email = payload.get("author_email") or "agentic-bot@users.noreply.github.com"
    token = _github_token()
    root = _workspace_root()
    if not root.exists():
        raise ValueError("Workspace root not found")
    local_path = (root / rel_path).resolve()
    if not local_path.exists():
        raise ValueError("Local path not found")
    if root not in local_path.parents and local_path != root:
        raise ValueError("Path must be within workspace")
    remote = f"https://{token}@github.com/{owner}/{repo}.git"

    _run_git(["git", "init"], cwd=local_path)
    _run_git(["git", "config", "user.name", author_name], cwd=local_path)
    _run_git(["git", "config", "user.email", author_email], cwd=local_path)
    _run_git(["git", "add", "-A"], cwd=local_path)
    _run_git(["git", "commit", "-m", message, "--allow-empty"], cwd=local_path)
    _run_git(["git", "branch", "-M", branch], cwd=local_path)
    _run_git(["git", "remote", "remove", "origin"], cwd=local_path, allow_fail=True)
    _run_git(["git", "remote", "add", "origin", remote], cwd=local_path)
    _run_git(["git", "push", "-u", "origin", branch], cwd=local_path)
    return {"branch": branch, "remote": f"{owner}/{repo}"}


def _github_repo_clone_or_pull(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    rel_path = _required(payload, "path")
    branch = payload.get("branch") or "main"
    pull_strategy = payload.get("pull_strategy") or "rebase"
    force_reset = bool(payload.get("force_reset", False))
    token = _github_token()
    root = _workspace_root()
    if not root.exists():
        raise ValueError("Workspace root not found")
    local_path = (root / rel_path).resolve()
    if root not in local_path.parents and local_path != root:
        raise ValueError("Path must be within workspace")
    remote = f"https://{token}@github.com/{owner}/{repo}.git"

    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["git", "clone", "--branch", branch, remote, str(local_path)], cwd=root)
        return {"path": str(local_path), "action": "cloned", "branch": branch}

    if not (local_path / ".git").exists():
        raise ValueError("Target path exists but is not a git repository")

    _run_git(["git", "fetch", "origin", branch], cwd=local_path)
    branch_check = run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=str(local_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if branch_check.returncode == 0:
        _run_git(["git", "checkout", branch], cwd=local_path)
    else:
        _run_git(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=local_path)
    if force_reset:
        _run_git(["git", "reset", "--hard", f"origin/{branch}"], cwd=local_path)
    else:
        if pull_strategy not in {"rebase", "merge", "ff-only"}:
            raise ValueError("pull_strategy must be one of: rebase, merge, ff-only")
        if pull_strategy == "rebase":
            _run_git(["git", "pull", "--rebase", "origin", branch], cwd=local_path)
        elif pull_strategy == "ff-only":
            _run_git(["git", "pull", "--ff-only", "origin", branch], cwd=local_path)
        else:
            _run_git(["git", "pull", "--no-rebase", "origin", branch], cwd=local_path)
    return {"path": str(local_path), "action": "pulled", "branch": branch}


def _github_repo_push_pr(payload: Dict[str, Any]) -> Dict[str, Any]:
    owner = _required(payload, "owner")
    repo = _required(payload, "repo")
    rel_path = _required(payload, "path")
    branch = _required(payload, "branch")
    base = payload.get("base") or "main"
    title = payload.get("title") or f"Update {repo}"
    body = payload.get("body") or ""
    message = payload.get("message") or "Update from workspace"
    author_name = payload.get("author_name") or "agentic-bot"
    author_email = payload.get("author_email") or "agentic-bot@users.noreply.github.com"
    token = _github_token()
    root = _workspace_root()
    if not root.exists():
        raise ValueError("Workspace root not found")
    local_path = (root / rel_path).resolve()
    if not local_path.exists():
        raise ValueError("Local path not found")
    if root not in local_path.parents and local_path != root:
        raise ValueError("Path must be within workspace")
    remote = f"https://{token}@github.com/{owner}/{repo}.git"

    if not (local_path / ".git").exists():
        raise ValueError("Target path is not a git repository; clone first")

    _run_git(["git", "remote", "remove", "origin"], cwd=local_path, allow_fail=True)
    _run_git(["git", "remote", "add", "origin", remote], cwd=local_path)
    _run_git(["git", "fetch", "origin", base], cwd=local_path)
    _run_git(["git", "checkout", "-B", branch, f"origin/{base}"], cwd=local_path)
    _run_git(["git", "config", "user.name", author_name], cwd=local_path)
    _run_git(["git", "config", "user.email", author_email], cwd=local_path)
    _run_git(["git", "add", "-A"], cwd=local_path)
    _run_git(["git", "commit", "-m", message, "--allow-empty"], cwd=local_path)
    _run_git(["git", "push", "-u", "origin", branch], cwd=local_path)

    pr = _github_request(
        "POST",
        f"/repos/{owner}/{repo}/pulls",
        body={"title": title, "head": branch, "base": base, "body": body},
    )
    return {
        "branch": branch,
        "base": base,
        "remote": f"{owner}/{repo}",
        "pr_url": pr.get("html_url", ""),
        "pr_number": pr.get("number", 0),
    }


def _github_request(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    token = _github_token()
    base_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    url = f"{base_url}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "agentic-platform",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise ValueError(f"GitHub API error: {detail}") from exc
    except URLError as exc:
        raise ValueError(f"GitHub API connection failed: {exc}") from exc
    if not payload:
        return {}
    return json.loads(payload)


def _github_token() -> str:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GITHUB_TOKEN")
    return token


def _workspace_root() -> Path:
    workspace_dir = os.getenv("WORKSPACE_DIR")
    if workspace_dir:
        return Path(workspace_dir).resolve()
    return Path.cwd().resolve()


def _required(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {key}")
    return value.strip()


def _clamp_per_page(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 30
    return max(1, min(parsed, 100))


def _run_git(args: list[str], cwd: Path, allow_fail: bool = False) -> None:
    result = run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0 and not allow_fail:
        raise ValueError(result.stderr.strip() or "Git command failed")
