from __future__ import annotations

from typing import Any, Callable

from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool


PayloadHandler = Callable[[dict[str, Any]], dict[str, Any]]


def register_llm_text_tool(
    registry,
    *,
    timeout_s: int,
    handler: PayloadHandler,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate",
                description="Generate text with an LLM",
                usage_guidance=(
                    "Use for open-ended text generation or reasoning. "
                    "Provide the prompt in 'text' (preferred) or 'prompt'. "
                    "Returns the raw completion in the 'text' field."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "prompt": {"type": "string", "minLength": 1},
                    },
                    "anyOf": [{"required": ["text"]}, {"required": ["prompt"]}],
                },
                output_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler,
        )
    )


def register_coding_agent_tools(
    registry,
    *,
    timeout_s: int,
    handler_generate: PayloadHandler,
    handler_autonomous: PayloadHandler,
    handler_publish_pr: PayloadHandler | None = None,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="coding_agent_generate",
                description="Generate code files using the coding agent service",
                usage_guidance=(
                    "Use to generate code for a repo or feature. Provide 'goal' and optional "
                    "'files' (list of relative paths), 'constraints', and 'workspace_path'. "
                    "The tool calls the coding agent service and writes files to the workspace."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "minLength": 1},
                        "files": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "string"},
                        "workspace_path": {"type": "string"},
                    },
                    "required": ["goal"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["path", "content"],
                            },
                        },
                        "written_paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["files", "written_paths"],
                },
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_generate,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="coding_agent_autonomous",
                description="Autonomously plan and implement a codebase in steps using the coding agent",
                usage_guidance=(
                    "Provide 'goal' and 'workspace_path'. The tool creates "
                    "IMPLEMENTATION_PLAN.md, then implements each step and updates status "
                    "in the plan file until complete."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "minLength": 1},
                        "workspace_path": {"type": "string", "minLength": 1},
                        "constraints": {"type": "string"},
                        "max_steps": {"type": "integer", "minimum": 1, "maximum": 12},
                    },
                    "required": ["goal", "workspace_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "plan_path": {"type": "string"},
                        "steps_total": {"type": "integer"},
                        "steps_completed": {"type": "integer"},
                        "written_paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "plan_path",
                        "steps_total",
                        "steps_completed",
                        "written_paths",
                    ],
                },
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_autonomous,
        )
    )

    if handler_publish_pr is not None:
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="coding_agent_publish_pr",
                    description=(
                        "Publish workspace codegen changes to GitHub via MCP (create branch, "
                        "push files, create PR)"
                    ),
                    usage_guidance=(
                        "Provide owner, repo, branch, base, and workspace_path. Optional: "
                        "title, body, message, include_globs, exclude_globs, max_files, "
                        "max_file_bytes, max_total_bytes, draft."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string", "minLength": 1},
                            "repo": {"type": "string", "minLength": 1},
                            "branch": {"type": "string", "minLength": 1},
                            "base": {"type": "string", "minLength": 1},
                            "workspace_path": {"type": "string", "minLength": 1},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "message": {"type": "string"},
                            "head": {"type": "string"},
                            "draft": {"type": "boolean"},
                            "maintainer_can_modify": {"type": "boolean"},
                            "include_globs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "exclude_globs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "max_files": {"type": "integer", "minimum": 1, "maximum": 2000},
                            "max_file_bytes": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5_000_000,
                            },
                            "max_total_bytes": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20_000_000,
                            },
                        },
                        "required": ["owner", "repo", "branch", "base", "workspace_path"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "branch": {"type": "string"},
                            "base": {"type": "string"},
                            "selected_files": {"type": "integer"},
                            "selected_paths_preview": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "skipped": {"type": "object"},
                            "branch_create": {"type": "object"},
                            "push_result": {"type": "object"},
                            "pull_request": {"type": "object"},
                        },
                        "required": [
                            "branch",
                            "base",
                            "selected_files",
                            "selected_paths_preview",
                            "skipped",
                            "branch_create",
                            "push_result",
                            "pull_request",
                        ],
                    },
                    timeout_s=timeout_s,
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.io,
                ),
                handler=handler_publish_pr,
            )
        )



