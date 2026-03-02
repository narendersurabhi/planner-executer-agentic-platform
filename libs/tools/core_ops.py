from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from libs.core.models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import Tool


PayloadHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class CoreOpsHandlers:
    math_eval: PayloadHandler
    text_summarize: PayloadHandler
    file_write_artifact: PayloadHandler
    file_write_text: PayloadHandler
    file_write_code: PayloadHandler
    file_read_text: PayloadHandler
    list_files: PayloadHandler
    workspace_write_text: PayloadHandler
    workspace_write_code: PayloadHandler
    workspace_read_text: PayloadHandler
    workspace_list_files: PayloadHandler
    artifact_mkdir: PayloadHandler
    workspace_mkdir: PayloadHandler
    artifact_delete: PayloadHandler
    workspace_delete: PayloadHandler
    artifact_rename: PayloadHandler
    workspace_rename: PayloadHandler
    artifact_copy: PayloadHandler
    workspace_copy: PayloadHandler
    artifact_move: PayloadHandler
    derive_output_path: PayloadHandler
    derive_output_filename: PayloadHandler
    run_tests: PayloadHandler
    search_text: PayloadHandler
    memory_read: PayloadHandler
    memory_write: PayloadHandler
    memory_semantic_write: PayloadHandler
    memory_semantic_search: PayloadHandler
    docx_render: PayloadHandler
    sleep: PayloadHandler
    http_fetch: PayloadHandler


def register_core_ops_tools(
    registry,
    *,
    handlers: CoreOpsHandlers,
    http_fetch_enabled: bool,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="math_eval",
                description="Evaluate a safe math expression",
                usage_guidance=(
                    "Use for deterministic math when you can provide a concrete expression. "
                    "Pass the expression as a string in the 'expr' field (example: '14*12')."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"expr": {"type": "string", "minLength": 1}},
                    "required": ["expr"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"value": {"type": "number"}},
                    "required": ["value"],
                },
                timeout_s=3,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=handlers.math_eval,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="text_summarize",
                description="Summarize text by truncation",
                usage_guidance="Use to shorten an existing text. Provide input in the 'text' field.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string", "minLength": 1}},
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=handlers.text_summarize,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_artifact",
                description="Write artifact content to shared volume",
                usage_guidance=(
                    "Use to write text to /shared/artifacts. Provide 'content' (required) "
                    "and optional 'path' (relative filename). Defaults to artifact.txt."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "document_type": {"type": "string"},
                    },
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.file_write_artifact,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_text",
                description="Write text content to a file under /shared/artifacts",
                usage_guidance=(
                    "Use to write text to /shared/artifacts. Provide 'content' (required) "
                    "and 'path' (required, include the filename)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.file_write_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_write_code",
                description="Write code content to a file under /shared/artifacts",
                usage_guidance=(
                    "Use to write code files to /shared/artifacts. Provide 'content' and 'path' "
                    "(required, include the filename). The tool creates missing directories and "
                    "expects a code file extension (e.g., .py, .js, .ts, .html, .css)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.file_write_code,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="file_read_text",
                description="Read text content from a file under /shared/artifacts",
                usage_guidance=(
                    "Use to read a text file from /shared/artifacts. Provide the 'path' "
                    "relative to /shared/artifacts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "minLength": 1}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.file_read_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="list_files",
                description="List files under /shared/artifacts",
                usage_guidance=(
                    "Use to list files under /shared/artifacts. Provide optional 'path' (relative "
                    "subdirectory), 'recursive' (bool), and 'max_files' (int)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_files": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["path", "type"],
                            },
                        }
                    },
                    "required": ["entries"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.list_files,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_write_text",
                description="Write text content to a file under the workspace",
                usage_guidance=(
                    "Use to write text to the workspace. Provide 'content' (required) "
                    "and 'path' (required, include the filename)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_write_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_write_code",
                description="Write code content to a file under the workspace",
                usage_guidance=(
                    "Use to write code files in the workspace. Provide 'content' and 'path' "
                    "(required, include the filename). The tool creates missing directories and "
                    "expects a code file extension (e.g., .py, .js, .ts, .html, .css)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "minLength": 1},
                    },
                    "required": ["content", "path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_write_code,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_read_text",
                description="Read text content from a file under the workspace",
                usage_guidance=(
                    "Use to read a text file from the workspace. Provide the 'path' "
                    "relative to the workspace root."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string", "minLength": 1}},
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_read_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_list_files",
                description="List files under the workspace",
                usage_guidance=(
                    "Use to list files under the workspace. Provide optional 'path' (relative "
                    "subdirectory), 'recursive' (bool), and 'max_files' (int)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "max_files": {"type": "integer", "minimum": 1, "maximum": 2000},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["path", "type"],
                            },
                        }
                    },
                    "required": ["entries"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_list_files,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_move",
                description="Move a file from /shared/artifacts into the workspace",
                usage_guidance=(
                    "Use to move an artifact into the workspace. Provide 'source_path' (relative "
                    "to /shared/artifacts) and 'destination_path' (relative to the workspace). "
                    "Set 'overwrite' true to replace an existing destination."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.artifact_move,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_mkdir",
                description="Create a directory under /shared/artifacts",
                usage_guidance=(
                    "Use to create directories under /shared/artifacts. Provide 'path' (required). "
                    "Optional: 'parents' (default true), 'exist_ok' (default true)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "parents": {"type": "boolean"},
                        "exist_ok": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.artifact_mkdir,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_mkdir",
                description="Create a directory under the workspace",
                usage_guidance=(
                    "Use to create directories under the workspace. Provide 'path' (required). "
                    "Optional: 'parents' (default true), 'exist_ok' (default true)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "parents": {"type": "boolean"},
                        "exist_ok": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_mkdir,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_delete",
                description="Delete a file or directory under /shared/artifacts",
                usage_guidance=(
                    "Use to delete files or directories under /shared/artifacts. Provide 'path' "
                    "(required). For non-empty directories set 'recursive' true. Optional: "
                    "'missing_ok' (default false)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "recursive": {"type": "boolean"},
                        "missing_ok": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "deleted": {"type": "boolean"},
                    },
                    "required": ["path", "deleted"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.artifact_delete,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_delete",
                description="Delete a file or directory under the workspace",
                usage_guidance=(
                    "Use to delete files or directories under the workspace. Provide 'path' "
                    "(required). For non-empty directories set 'recursive' true. Optional: "
                    "'missing_ok' (default false)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "recursive": {"type": "boolean"},
                        "missing_ok": {"type": "boolean"},
                    },
                    "required": ["path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "deleted": {"type": "boolean"},
                    },
                    "required": ["path", "deleted"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_delete,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_rename",
                description="Rename or move a file/directory within /shared/artifacts",
                usage_guidance=(
                    "Use to rename or move files/directories within /shared/artifacts. Provide "
                    "'source_path' and 'destination_path' (required). Optional: 'overwrite' (default false)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.artifact_rename,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_rename",
                description="Rename or move a file/directory within the workspace",
                usage_guidance=(
                    "Use to rename or move files/directories within the workspace. Provide "
                    "'source_path' and 'destination_path' (required). Optional: 'overwrite' (default false)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_rename,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="artifact_copy",
                description="Copy a file/directory within /shared/artifacts",
                usage_guidance=(
                    "Use to copy files/directories within /shared/artifacts. Provide 'source_path' "
                    "and 'destination_path' (required). Optional: 'overwrite' (default false), "
                    "'recursive' (default true for directories)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=15,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.artifact_copy,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="workspace_copy",
                description="Copy a file/directory within the workspace",
                usage_guidance=(
                    "Use to copy files/directories within the workspace. Provide 'source_path' "
                    "and 'destination_path' (required). Optional: 'overwrite' (default false), "
                    "'recursive' (default true for directories)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_path": {"type": "string", "minLength": 1},
                        "destination_path": {"type": "string", "minLength": 1},
                        "overwrite": {"type": "boolean"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["source_path", "destination_path"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=15,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.workspace_copy,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="derive_output_path",
                description="Derive a generic filesystem-safe output path for generated artifacts",
                usage_guidance=(
                    "Use for capability-driven document rendering where naming should be generic and deterministic. "
                    "Provide topic and output_dir. "
                    "document_type is optional metadata and defaults to 'document'. "
                    "Optionally provide today/date (YYYY-MM-DD). "
                    "Optionally provide output_extension (or file_extension/extension/format); "
                    "if omitted, document_type can be used as extension when it is a known format, "
                    "otherwise default extension is docx."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "minLength": 1},
                        "today": {"type": "string", "minLength": 4},
                        "date": {"type": "string", "minLength": 4},
                        "output_dir": {"type": "string", "minLength": 1},
                        "document_type": {"type": "string", "minLength": 1},
                        "output_extension": {"type": "string"},
                        "file_extension": {"type": "string"},
                        "extension": {"type": "string"},
                        "format": {"type": "string"},
                    },
                    "required": ["topic", "output_dir"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "document_type": {"type": "string"},
                        "output_extension": {"type": "string"},
                    },
                    "required": ["path"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.derive_output_path,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="derive_output_filename",
                description="Derive a filesystem-safe output path for generated documents",
                usage_guidance=(
                    "Use to create a safe output path for render tools. "
                    "Provide target_role_name (or role_name or topic) and date/today (YYYY-MM-DD) "
                    "to get role_date naming. If date/today is omitted, "
                    "the tool defaults to the current UTC date. "
                    "Optionally provide 'output_dir' (default: documents). "
                    "Optionally provide output_extension (or file_extension/extension/format) "
                    "to control the file extension (default: docx)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "target_role_name": {"type": "string", "minLength": 1},
                        "role_name": {"type": "string", "minLength": 1},
                        "topic": {"type": "string", "minLength": 1},
                        "candidate_name": {"type": "string", "minLength": 1},
                        "first_name": {"type": "string", "minLength": 1},
                        "last_name": {"type": "string", "minLength": 1},
                        "company_name": {"type": "string", "minLength": 1},
                        "company": {"type": "string", "minLength": 1},
                        "job_description": {"type": "string", "minLength": 1},
                        "date": {"type": "string", "minLength": 4},
                        "today": {"type": "string", "minLength": 4},
                        "output_dir": {"type": "string"},
                        "document_type": {"type": "string"},
                        "output_extension": {"type": "string"},
                        "file_extension": {"type": "string"},
                        "extension": {"type": "string"},
                        "format": {"type": "string"},
                    },
                    "allOf": [
                        {
                            "anyOf": [
                                {
                                    "anyOf": [
                                        {"required": ["target_role_name"]},
                                        {"required": ["role_name"]},
                                        {"required": ["topic"]},
                                        {"required": ["job_description"]},
                                    ]
                                },
                                {
                                    "allOf": [
                                        {
                                            "anyOf": [
                                                {"required": ["candidate_name"]},
                                                {"required": ["first_name", "last_name"]},
                                            ]
                                        },
                                        {
                                            "anyOf": [
                                                {"required": ["company_name"]},
                                                {"required": ["company"]},
                                            ]
                                        },
                                        {
                                            "anyOf": [
                                                {"required": ["target_role_name"]},
                                                {"required": ["role_name"]},
                                                {"required": ["topic"]},
                                                {"required": ["job_description"]},
                                            ]
                                        },
                                    ]
                                },
                                {
                                    "required": ["memory"]
                                },
                            ]
                        }
                    ],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=2,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=handlers.derive_output_filename,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="run_tests",
                description="Run tests within /shared/artifacts using an allowlisted command",
                usage_guidance=(
                    "Use to run tests in /shared/artifacts. Provide 'command' and optional 'args' "
                    "and 'cwd' (relative). Only allowlisted commands are permitted."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "minLength": 1},
                        "args": {"type": "array", "items": {"type": "string"}},
                        "cwd": {"type": "string"},
                    },
                    "required": ["command"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "exit_code": {"type": "integer"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                    },
                    "required": ["exit_code", "stdout", "stderr"],
                },
                timeout_s=30,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.validate,
            ),
            handler=handlers.run_tests,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="search_text",
                description="Search for text in files under /shared/artifacts",
                usage_guidance=(
                    "Use to find text in files under /shared/artifacts. Provide 'query' (required), "
                    "optional 'path', 'glob', 'case_sensitive', 'regex', 'context_lines', and 'max_matches'."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "path": {"type": "string"},
                        "glob": {"type": "string"},
                        "case_sensitive": {"type": "boolean"},
                        "regex": {"type": "boolean"},
                        "context_lines": {"type": "integer", "minimum": 0, "maximum": 5},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "matches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "text": {"type": "string"},
                                    "context": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["path", "line", "text"],
                            },
                        }
                    },
                    "required": ["matches"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.search_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="memory_read",
                description="Read entries from memory store",
                usage_guidance=(
                    "Use to resolve memory pointers before downstream tool calls. "
                    "Provide 'name' and optional filters: scope, key, job_id, user_id, project_id, limit."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "scope": {
                            "type": "string",
                            "enum": ["request", "session", "user", "project", "global"],
                        },
                        "key": {"type": "string"},
                        "job_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "project_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        "include_expired": {"type": "boolean"},
                    },
                    "required": ["name"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entries": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                    },
                    "required": ["entries", "count"],
                },
                timeout_s=10,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.memory_read,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="memory_write",
                description="Write an entry to memory store",
                usage_guidance=(
                    "Use to persist structured outputs for reuse across tasks and jobs. "
                    "Provide name and payload; optional scope, key, job_id, user_id, project_id, ttl_seconds, metadata."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "payload": {"type": "object"},
                        "scope": {
                            "type": "string",
                            "enum": ["request", "session", "user", "project", "global"],
                        },
                        "key": {"type": "string"},
                        "job_id": {"type": "string"},
                        "user_id": {"type": "string"},
                        "project_id": {"type": "string"},
                        "ttl_seconds": {"type": "integer", "minimum": 1},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "payload"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"entry": {"type": "object"}},
                    "required": ["entry"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.memory_write,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="memory_semantic_write",
                description="Write distilled semantic facts to semantic memory",
                usage_guidance=(
                    "Use to persist stable facts for later reasoning. "
                    "Provide 'fact' and optional subject/namespace/keywords/aliases/confidence. "
                    "User scope defaults to SEMANTIC_MEMORY_DEFAULT_USER_ID when user_id is omitted."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string", "minLength": 1},
                        "subject": {"type": "string"},
                        "namespace": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "source": {"type": "string"},
                        "source_ref": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "key": {"type": "string"},
                        "user_id": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["fact"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entry": {"type": "object"},
                        "semantic_record": {"type": "object"},
                    },
                    "required": ["entry"],
                },
                timeout_s=10,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.memory_semantic_write,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="memory_semantic_search",
                description="Search semantic memory by natural-language query",
                usage_guidance=(
                    "Use to retrieve relevant facts for reasoning/context. "
                    "Provide 'query' and optional namespace/subject filters and score threshold."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "namespace": {"type": "string"},
                        "subject": {"type": "string"},
                        "key": {"type": "string"},
                        "user_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                        "min_score": {"type": "number", "minimum": 0},
                        "include_payload": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "matches": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                    },
                    "required": ["matches", "count"],
                },
                timeout_s=10,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.memory_semantic_search,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="docx_render",
                description="Render a DOCX file from structured JSON and a DOCX template",
                usage_guidance=(
                    "Provide data (object), plus either template_id (resolved under /shared/templates) "
                    "or template_path. Optionally include schema_ref to validate data against a schema "
                    "from the registry before rendering. Optionally set output_path for the rendered DOCX."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "schema_ref": {"type": "string"},
                        "template_id": {"type": "string"},
                        "template_path": {"type": "string"},
                        "output_path": {"type": "string"},
                    },
                    "required": ["data", "output_path"],
                    "anyOf": [
                        {"required": ["template_id"]},
                        {"required": ["template_path"]},
                    ],
                },
                output_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                timeout_s=20,
                risk_level=RiskLevel.medium,
                tool_intent=ToolIntent.render,
            ),
            handler=handlers.docx_render,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="sleep",
                description="Sleep for a number of seconds",
                usage_guidance="Use only for testing delays. Provide seconds as a number.",
                input_schema={
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"slept": {"type": "number"}},
                    "required": ["slept"],
                },
                timeout_s=10,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.io,
            ),
            handler=handlers.sleep,
        )
    )

    if http_fetch_enabled:
        registry.register(
            Tool(
                spec=ToolSpec(
                    name="http_fetch",
                    description="Fetch HTTP content",
                    usage_guidance=(
                        "Use to fetch public HTTP(S) URLs. The host must be in TOOL_HTTP_FETCH_ALLOWLIST."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {"url": {"type": "string", "minLength": 1}},
                        "required": ["url"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {"body": {"type": "string"}},
                        "required": ["body"],
                    },
                    timeout_s=10,
                    risk_level=RiskLevel.high,
                    tool_intent=ToolIntent.io,
                ),
                handler=handlers.http_fetch,
            )
        )
