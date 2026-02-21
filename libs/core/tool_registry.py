from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import time
from subprocess import CompletedProcess, run
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from .llm_provider import LLMProvider
from . import prompts, tracing as core_tracing
from .models import RiskLevel, ToolIntent, ToolSpec
from libs.framework.tool_runtime import (
    Tool,
    ToolExecutionError,
    ToolRegistry,
    classify_tool_error as _classify_tool_error,
    validate_schema as _validate_schema,
)
from libs.tools.docx_generate_from_spec import register_docx_tools
from libs.tools.document_spec_validate import register_document_spec_tools
from libs.tools.resume_doc_spec_validate import register_resume_doc_spec_tools
from libs.tools.resume_doc_spec_to_document_spec import register_resume_doc_spec_convert_tools
from libs.tools.coverletter_doc_spec_to_document_spec import (
    register_coverletter_doc_spec_convert_tools,
)
from libs.tools.cover_letter_generate_ats_docx import register_cover_letter_generate_tools
from libs.tools.document_spec_iterative import register_document_spec_iterative_tools
from libs.tools.document_spec_llm import (
    llm_generate_document_spec as _llm_generate_document_spec_external,
    llm_improve_document_spec as _llm_improve_document_spec_external,
    register_document_spec_llm_tools,
)
from libs.tools.core_ops import CoreOpsHandlers, register_core_ops_tools
from libs.tools.llm_tool_groups import (
    register_coding_agent_tools,
    register_llm_text_tool,
    register_resume_llm_tools,
    register_tailor_mcp_tools,
)
from libs.tools import mcp_client
from libs.tools import resume_llm
from libs.tools import coder_tools
from libs.tools.github_tools import register_github_tools
from libs.tools.openapi_iterative import register_openapi_iterative_tools

LOGGER = logging.getLogger(__name__)


def _extract_mcp_error_phase(error_text: str) -> str | None:
    return mcp_client.extract_mcp_error_phase(error_text)


def _safe_artifact_path(path: str, default_name: str) -> Path:
    base_dir = Path("/shared/artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside /shared/artifacts")
    return resolved


def _workspace_root() -> Path:
    env_root = os.getenv("WORKSPACE_DIR")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def _safe_workspace_path(path: str, default_name: str) -> Path:
    base_dir = _workspace_root()
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(path or default_name)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid path outside workspace")
    return resolved


def _resolve_schema_path(schema_ref: str) -> Path:
    registry_dir = Path(os.getenv("SCHEMA_REGISTRY_PATH", "/app/schemas"))
    template_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    name = schema_ref
    if schema_ref.startswith("schema/"):
        name = schema_ref.split("/", 1)[1]
    if not name.endswith(".json"):
        name = f"{name}.json"
    candidate = registry_dir / name
    if candidate.exists():
        return candidate
    return template_dir / name


def _validate_schema_from_registry(schema_ref: str, payload: Dict[str, Any]) -> None:
    schema_path = _resolve_schema_path(schema_ref)
    if not schema_path.exists():
        raise ToolExecutionError(f"schema_not_found:{schema_path}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError(f"invalid_schema:{exc}") from exc
    _validate_schema(schema, payload, "input")


def _parse_http_allowlist() -> List[str]:
    raw = os.getenv("TOOL_HTTP_FETCH_ALLOWLIST", "")
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _resolve_llm_timeout_s(provider: Optional[LLMProvider]) -> int:
    if provider is not None and hasattr(provider, "timeout_s"):
        try:
            return max(1, int(math.ceil(getattr(provider, "timeout_s"))))
        except (TypeError, ValueError):
            pass
    env_timeout = os.getenv("OPENAI_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _resolve_coding_agent_timeout_s() -> int:
    env_timeout = os.getenv("CODING_AGENT_TIMEOUT_S")
    if env_timeout:
        try:
            return max(1, int(math.ceil(float(env_timeout))))
        except ValueError:
            return 30
    return 30


def _resolve_coder_http_timeout_s() -> int:
    for key in ("CODER_HTTP_TIMEOUT_S", "CODING_AGENT_TIMEOUT_S"):
        env_timeout = os.getenv(key)
        if env_timeout:
            try:
                return max(1, int(math.ceil(float(env_timeout))))
            except ValueError:
                return 30
    return 30


def _resolve_mcp_timeout_s() -> float:
    return mcp_client.resolve_mcp_timeout_s()


def _resolve_mcp_outer_timeout_headroom_s() -> float:
    return mcp_client.resolve_mcp_outer_timeout_headroom_s()


def _resolve_mcp_tool_timeout_s() -> int:
    return mcp_client.resolve_mcp_tool_timeout_s()


def _resolve_mcp_max_retries() -> int:
    return mcp_client.resolve_mcp_max_retries()


def _resolve_mcp_retry_sleep_s() -> float:
    return mcp_client.resolve_mcp_retry_sleep_s()


def _resolve_mcp_first_attempt_reserve_s(timeout_s: float) -> float:
    return mcp_client.resolve_mcp_first_attempt_reserve_s(timeout_s)


def _resolve_mcp_transport_timeout_s(timeout_s: float) -> float:
    return mcp_client.resolve_mcp_transport_timeout_s(timeout_s)


def _resolve_mcp_isolation_mode() -> str:
    return mcp_client.resolve_mcp_isolation_mode()


def _streamable_http_client_kwargs(
    client_factory: Callable[..., Any], timeout_s: float
) -> Dict[str, Any]:
    return mcp_client.streamable_http_client_kwargs(client_factory, timeout_s)


def _is_retryable_mcp_error(message: str) -> bool:
    return mcp_client.is_retryable_mcp_error(message)


def _host_allowed(host: str, allowlist: List[str]) -> bool:
    if not allowlist:
        return False
    if "*" in allowlist:
        return True
    for entry in allowlist:
        if entry.startswith("*."):
            suffix = entry[1:]
            if host.endswith(suffix):
                return True
        elif entry.startswith("."):
            if host.endswith(entry):
                return True
        elif host == entry:
            return True
    return False


def default_registry(
    http_fetch_enabled: bool = False,
    llm_enabled: bool = False,
    llm_provider: Optional[LLMProvider] = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            spec=ToolSpec(
                name="json_transform",
                description="Wrap or reshape already-available JSON input",
                usage_guidance=(
                    "Use only when you already have structured JSON and need to pass it forward. "
                    "Provide the JSON as the 'input' field. This tool does NOT generate new content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"input": {}},
                    "required": ["input"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"result": {}},
                    "required": ["result"],
                },
                timeout_s=5,
                risk_level=RiskLevel.low,
                tool_intent=ToolIntent.transform,
            ),
            handler=lambda payload: {"result": payload.get("input")},
        )
    )

    register_core_ops_tools(
        registry,
        handlers=CoreOpsHandlers(
            math_eval=_math_eval,
            text_summarize=_text_summarize,
            file_write_artifact=lambda payload: _write_text_file(payload, default_filename="artifact.txt"),
            file_write_text=lambda payload: _write_text_file(payload, default_filename="output.txt"),
            file_write_code=_file_write_code,
            file_read_text=_file_read_text,
            list_files=_list_files,
            workspace_write_text=lambda payload: _write_workspace_text_file(
                payload, default_filename="output.txt"
            ),
            workspace_write_code=_workspace_write_code,
            workspace_read_text=_workspace_read_text,
            workspace_list_files=_list_workspace_files,
            artifact_move=_artifact_move_to_workspace,
            derive_output_filename=_derive_output_filename,
            run_tests=_run_tests,
            search_text=_search_text,
            docx_render=_docx_render,
            sleep=_sleep,
            http_fetch=_http_fetch,
        ),
        http_fetch_enabled=http_fetch_enabled,
    )

    register_docx_tools(registry)
    register_document_spec_tools(registry)
    register_resume_doc_spec_tools(registry)
    register_resume_doc_spec_convert_tools(registry)
    register_coverletter_doc_spec_convert_tools(registry)
    register_cover_letter_generate_tools(registry)
    register_github_tools(registry)

    if llm_enabled:
        if llm_provider is None:
            raise ValueError("llm_enabled requires a llm_provider instance")
        llm_timeout_s = _resolve_llm_timeout_s(llm_provider)
        coding_agent_timeout_s = _resolve_coding_agent_timeout_s()
        mcp_timeout_s = _resolve_mcp_tool_timeout_s()
        llm_iterative_timeout_s = _resolve_llm_iterative_tool_timeout_s(llm_provider)

        register_llm_text_tool(
            registry,
            timeout_s=llm_timeout_s,
            handler=lambda payload, provider=llm_provider: _llm_generate(payload, provider),
        )
        register_coding_agent_tools(
            registry,
            timeout_s=coding_agent_timeout_s,
            handler_generate=_coding_agent_generate,
            handler_autonomous=lambda payload, provider=llm_provider: _coding_agent_autonomous(
                payload, provider
            ),
        )
        register_tailor_mcp_tools(
            registry,
            timeout_s=mcp_timeout_s,
            handler_iterative_improve=lambda payload, provider=llm_provider: (
                _llm_iterative_improve_tailored_resume_text(payload, provider)
            ),
            handler_tailor=lambda payload, provider=llm_provider: _llm_tailor_resume_text(
                payload, provider
            ),
            handler_improve=lambda payload, provider=llm_provider: (
                _llm_improve_tailored_resume_text(payload, provider)
            ),
        )
        register_resume_llm_tools(
            registry,
            timeout_s=llm_timeout_s,
            handler_generate_resume_doc_spec_from_text=lambda payload, provider=llm_provider: (
                _llm_generate_resume_doc_spec_from_text(payload, provider)
            ),
            handler_generate_coverletter_doc_spec_from_text=lambda payload, provider=llm_provider: (
                _llm_generate_coverletter_doc_spec_from_text(payload, provider)
            ),
            handler_generate_cover_letter_from_resume=lambda payload, provider=llm_provider: (
                _llm_generate_cover_letter_from_resume(payload, provider)
            ),
            handler_generate_resume_doc_spec=lambda payload, provider=llm_provider: (
                _llm_generate_resume_doc_spec(payload, provider)
            ),
        )
        register_document_spec_llm_tools(
            registry,
            llm_provider,
            timeout_s=llm_timeout_s,
            sanitize_document_spec=_sanitize_document_spec,
        )
        register_document_spec_iterative_tools(
            registry,
            llm_provider,
            timeout_s=llm_iterative_timeout_s,
            generate_document_spec=_llm_generate_document_spec,
            improve_document_spec=_llm_improve_document_spec,
            sanitize_document_spec=_sanitize_document_spec,
        )
        register_openapi_iterative_tools(
            registry,
            llm_provider,
            timeout_s=llm_iterative_timeout_s,
        )

    return registry


def _math_eval(payload: Dict[str, Any]) -> Dict[str, Any]:
    expr = payload.get("expr", "0")
    allowed = {"sqrt": math.sqrt, "pow": pow}
    value = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
    return {"value": value}


def _text_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text", "")
    summary = text[:200]
    return {"summary": summary}


def _write_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_artifact_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _write_workspace_text_file(payload: Dict[str, Any], default_filename: str) -> Dict[str, Any]:
    path = payload.get("path", "")
    content = payload.get("content", "")
    if path and path.endswith("/"):
        raise ToolExecutionError("Missing file name in path")
    candidate = _safe_workspace_path(path, default_filename)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return {"path": str(candidate)}


def _file_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_text_file(payload, default_filename="output.txt")


def _workspace_write_code(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing file name in path")
    _ensure_code_extension(path)
    return _write_workspace_text_file(payload, default_filename="output.txt")


def _ensure_code_extension(path: str) -> None:
    code_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".json",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".sh",
        ".sql",
        ".txt",
    }
    suffix = Path(path).suffix.lower()
    if not suffix or suffix not in code_extensions:
        raise ToolExecutionError("Unsupported code file extension")


def _file_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_artifact_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}


def _workspace_read_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    if not path:
        raise ToolExecutionError("Missing path")
    candidate = _safe_workspace_path(path, "output.txt")
    if not candidate.exists():
        raise ToolExecutionError("File not found")
    return {"content": candidate.read_text(encoding="utf-8")}


def _list_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"entries": []}
    if root.is_file():
        return {"entries": [{"path": str(root), "type": "file"}]}
    entries = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for entry in iterator:
        entry_type = "dir" if entry.is_dir() else "file"
        entries.append({"path": str(entry), "type": entry_type})
        if len(entries) >= max_files:
            break
    return {"entries": entries}


def _list_workspace_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload.get("path", "")
    recursive = bool(payload.get("recursive", False))
    max_files = payload.get("max_files", 200)
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 200
    root = _safe_workspace_path(path, "")
    if not root.exists():
        return {"entries": []}
    if root.is_file():
        return {"entries": [{"path": str(root), "type": "file"}]}
    entries = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for entry in iterator:
        entry_type = "dir" if entry.is_dir() else "file"
        entries.append({"path": str(entry), "type": entry_type})
        if len(entries) >= max_files:
            break
    return {"entries": entries}


def _artifact_move_to_workspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_path = payload.get("source_path", "")
    destination_path = payload.get("destination_path", "")
    overwrite = bool(payload.get("overwrite", False))
    if not source_path:
        raise ToolExecutionError("Missing source_path")
    if not destination_path:
        raise ToolExecutionError("Missing destination_path")
    if destination_path.endswith("/"):
        raise ToolExecutionError("Missing file name in destination_path")
    source = _safe_artifact_path(source_path, "")
    if not source.exists():
        raise ToolExecutionError("Source file not found")
    destination = _safe_workspace_path(destination_path, "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise ToolExecutionError("Destination already exists")
    shutil.move(str(source), str(destination))
    return {"path": str(destination)}


def _derive_output_filename(payload: Dict[str, Any]) -> Dict[str, Any]:
    def pick_str(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    memory_context = _select_job_context_from_memory(payload.get("memory"))
    nested_context = memory_context.get("context_json")
    if not isinstance(nested_context, dict):
        nested_context = {}

    role_name = pick_str(
        payload.get("target_role_name"),
        payload.get("role_name"),
        payload.get("topic"),
        memory_context.get("target_role_name"),
        memory_context.get("role_name"),
        memory_context.get("topic"),
        nested_context.get("target_role_name"),
        nested_context.get("role_name"),
        nested_context.get("topic"),
    )
    company_name = pick_str(
        payload.get("company_name"),
        payload.get("company"),
        memory_context.get("company_name"),
        memory_context.get("company"),
        nested_context.get("company_name"),
        nested_context.get("company"),
    )
    candidate_name = pick_str(
        payload.get("candidate_name"),
        memory_context.get("candidate_name"),
        nested_context.get("candidate_name"),
    )
    first_name = pick_str(
        payload.get("first_name"),
        memory_context.get("first_name"),
        nested_context.get("first_name"),
    )
    last_name = pick_str(
        payload.get("last_name"),
        memory_context.get("last_name"),
        nested_context.get("last_name"),
    )
    job_description = pick_str(
        payload.get("job_description"),
        memory_context.get("job_description"),
        nested_context.get("job_description"),
    )
    candidate_resume = pick_str(
        payload.get("candidate_resume"),
        memory_context.get("candidate_resume"),
        nested_context.get("candidate_resume"),
    )
    tailored_text = pick_str(
        payload.get("tailored_text"),
        memory_context.get("tailored_text"),
        nested_context.get("tailored_text"),
    )
    date_value = pick_str(
        payload.get("date"),
        payload.get("today"),
        memory_context.get("date"),
        memory_context.get("today"),
        nested_context.get("date"),
        nested_context.get("today"),
    )
    output_dir = (
        pick_str(
            payload.get("output_dir"),
            memory_context.get("output_dir"),
            nested_context.get("output_dir"),
        )
        or "resumes"
    )
    document_type = pick_str(
        payload.get("document_type"),
        memory_context.get("document_type"),
        nested_context.get("document_type"),
    )
    normalized_doc_type = document_type.lower().replace("-", "_")
    is_cover_letter = normalized_doc_type in {"cover_letter", "coverletter"}
    if not isinstance(output_dir, str):
        output_dir = "resumes"
    output_dir = output_dir.strip().strip("/")
    if not output_dir:
        output_dir = "resumes"
    if output_dir.startswith("/") or ".." in Path(output_dir).parts:
        raise ToolExecutionError("Invalid output_dir")

    def clean_label(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        # Keep output human-readable while stripping filesystem-unsafe characters.
        cleaned = re.sub(r'[<>:"/\\\\|?*]', " ", value)
        cleaned = re.sub(r"[,_;:]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned

    def slugify(value: str, pattern: str) -> str:
        cleaned = re.sub(pattern, "_", value.lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned

    if (not isinstance(role_name, str) or not role_name.strip()) and isinstance(
        job_description, str
    ):
        role_name = _derive_role_name_from_jd(job_description)
    if (not isinstance(company_name, str) or not company_name.strip()) and isinstance(
        job_description, str
    ):
        company_name = _derive_company_name_from_jd(job_description)

    if (not isinstance(candidate_name, str) or not candidate_name.strip()) and not (
        isinstance(first_name, str)
        and first_name.strip()
        and isinstance(last_name, str)
        and last_name.strip()
    ):
        candidate_name = _derive_candidate_name_from_texts(
            candidate_resume, tailored_text, job_description
        )

    if (
        (
            (not isinstance(first_name, str) or not first_name.strip())
            or (not isinstance(last_name, str) or not last_name.strip())
        )
        and isinstance(candidate_name, str)
        and candidate_name.strip()
    ):
        tokens = [token for token in candidate_name.split() if token]
        if len(tokens) >= 2:
            if not isinstance(first_name, str) or not first_name.strip():
                first_name = tokens[0]
            if not isinstance(last_name, str) or not last_name.strip():
                last_name = tokens[-1]

    role_label = clean_label(role_name)
    company_label = clean_label(company_name)
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        name_parts = [clean_label(first_name), clean_label(last_name)]
        candidate_label = " ".join([part for part in name_parts if part]).strip()
    else:
        candidate_label = clean_label(candidate_name)

    if candidate_label and role_label and company_label:
        doc_label = "Cover Letter" if is_cover_letter else "Resume"
        filename = f"{candidate_label} {doc_label} - {role_label} - {company_label}.docx"
        return {
            "path": f"{output_dir}/{filename}",
            "document_type": "cover_letter" if is_cover_letter else "resume",
        }

    if not role_label:
        raise ToolExecutionError("Missing target_role_name")
    if not isinstance(date_value, str) or not date_value.strip():
        raise ToolExecutionError("Missing date")

    role_slug = slugify(role_label or str(role_name), r"[^a-z0-9]+") or "document"
    date_slug = slugify(date_value, r"[^0-9]+")
    if not date_slug:
        raise ToolExecutionError("Invalid date")
    if is_cover_letter:
        filename = f"cover_letter_{role_slug}_{date_slug}.docx"
    else:
        filename = f"{role_slug}_{date_slug}.docx"
    return {
        "path": f"{output_dir}/{filename}",
        "document_type": "cover_letter" if is_cover_letter else "resume",
    }


def _derive_role_name_from_jd(job_description: str) -> str:
    return resume_llm.derive_role_name_from_jd(job_description)


def _derive_company_name_from_jd(job_description: str) -> str:
    return resume_llm.derive_company_name_from_jd(job_description)


def _derive_candidate_name_from_texts(*texts: Any) -> str:
    return resume_llm.derive_candidate_name_from_texts(*texts)


def _run_tests(payload: Dict[str, Any]) -> Dict[str, Any]:
    command = payload.get("command", "")
    args = payload.get("args") or []
    cwd = payload.get("cwd", "")
    if not isinstance(args, list):
        args = []
    allowlist = {"pytest", "python"}
    if command not in allowlist:
        raise ToolExecutionError("Command not allowed")
    cmd = [command, *args]
    if command == "python":
        if len(args) < 2 or args[0] != "-m" or args[1] != "pytest":
            raise ToolExecutionError("Only python -m pytest is allowed")
    run_cwd = _safe_artifact_path(cwd, "")
    if not run_cwd.exists():
        raise ToolExecutionError("Working directory not found")
    result: CompletedProcess[str] = run(
        cmd, cwd=str(run_cwd), capture_output=True, text=True, check=False
    )
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _search_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload.get("query", "")
    if not isinstance(query, str) or not query:
        raise ToolExecutionError("Missing query")
    path = payload.get("path", "")
    glob = payload.get("glob", "")
    case_sensitive = bool(payload.get("case_sensitive", False))
    use_regex = bool(payload.get("regex", False))
    context_lines = payload.get("context_lines", 0)
    if not isinstance(context_lines, int) or context_lines < 0:
        context_lines = 0
    max_matches = payload.get("max_matches", 200)
    if not isinstance(max_matches, int) or max_matches < 1:
        max_matches = 200
    root = _safe_artifact_path(path, "")
    if not root.exists():
        return {"matches": []}
    pattern = glob or "**/*"
    matches = []
    needle = query if case_sensitive else query.lower()
    regex = None
    if use_regex:
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(query, flags=flags)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid regex: {exc}") from exc
    for file_path in root.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()
                for idx, line in enumerate(lines, start=1):
                    hay = line if case_sensitive else line.lower()
                    matched = False
                    if regex is not None:
                        matched = regex.search(line) is not None
                    else:
                        matched = needle in hay
                    if matched:
                        start = max(0, idx - 1 - context_lines)
                        end = min(len(lines), idx - 1 + context_lines + 1)
                        context = [item.rstrip("\n") for item in lines[start:end]]
                        entry = {"path": str(file_path), "line": idx, "text": line.rstrip("\n")}
                        if context_lines:
                            entry["context"] = context
                        matches.append(entry)
                        if len(matches) >= max_matches:
                            return {"matches": matches}
        except OSError:
            continue
    return {"matches": matches}


def _resolve_template_path(template_path: str, template_id: str) -> Path:
    base_dir = Path(os.getenv("DOCX_TEMPLATE_DIR", "/shared/templates"))
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(template_path or "")
    if not template_path:
        candidate = Path(f"{template_id}.docx")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ToolExecutionError("Invalid template path outside template directory")
    return resolved


def _docx_render(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from docxtpl import DocxTemplate
    except Exception as exc:  # noqa: BLE001
        raise ToolExecutionError("docxtpl is not installed") from exc
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ToolExecutionError("data must be an object")
    schema_ref = payload.get("schema_ref", "")
    template_id = payload.get("template_id", "")
    template_path = payload.get("template_path", "")
    output_path = payload.get("output_path", "")
    if not output_path:
        raise ToolExecutionError("Missing output_path")
    if not schema_ref and template_id:
        schema_ref = template_id
    if schema_ref:
        _validate_schema_from_registry(schema_ref, data)
    if not template_id and not template_path:
        raise ToolExecutionError("Missing template_id or template_path")
    template_file = _resolve_template_path(template_path, template_id)
    if not template_file.exists():
        raise ToolExecutionError("DOCX template not found")
    candidate = _safe_artifact_path(output_path, f"docx_{template_id or 'output'}.docx")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    doc = DocxTemplate(str(template_file))
    doc.render(data)
    doc.save(str(candidate))
    return {"path": str(candidate)}


def _sleep(payload: Dict[str, Any]) -> Dict[str, Any]:
    seconds = float(payload.get("seconds", 0))
    time.sleep(seconds)
    return {"slept": seconds}


def _http_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request

    url = payload.get("url", "")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError("Unsupported URL scheme")
    host = parsed.hostname or ""
    allowlist = _parse_http_allowlist()
    if not _host_allowed(host, allowlist):
        raise ToolExecutionError("URL host not in allowlist")
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8")
    return {"body": body}


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return coder_tools.post_json(url, payload, timeout_s=_resolve_coder_http_timeout_s())


def _post_mcp_tool_call(
    service_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    return mcp_client.post_mcp_tool_call(
        service_url,
        tool_name,
        arguments,
        call_mcp_tool_sdk=_call_mcp_tool_sdk,
        classify_tool_error=_classify_tool_error,
        logger=LOGGER,
        tracing_module=core_tracing,
    )


def _call_mcp_tool_sdk(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        tracing_module=core_tracing,
        logger=LOGGER,
    )


def _call_mcp_tool_sdk_inproc(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk_inproc(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        tracing_module=core_tracing,
    )


def _mcp_process_entry(
    queue: Any, mcp_url: str, tool_name: str, arguments: Dict[str, Any], timeout_s: float
) -> None:
    mcp_client.mcp_process_entry(queue, mcp_url, tool_name, arguments, timeout_s)


def _call_mcp_tool_sdk_process(
    mcp_url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    return mcp_client.call_mcp_tool_sdk_process(
        mcp_url,
        tool_name,
        arguments,
        timeout_s,
        logger=LOGGER,
        tracing_module=core_tracing,
    )


def _flatten_exception_messages(exc: BaseException) -> List[str]:
    return mcp_client.flatten_exception_messages(exc)


def _extract_mcp_sdk_result(result: Any) -> Dict[str, Any]:
    return mcp_client.extract_mcp_sdk_result(result)


def _normalize_mcp_structured_result(structured: Dict[str, Any]) -> Dict[str, Any]:
    return mcp_client.normalize_mcp_structured_result(structured)


def _extract_mcp_error_detail(result: Any) -> str:
    return mcp_client.extract_mcp_error_detail(result)


def _coding_agent_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    return coder_tools.coding_agent_generate(
        payload,
        post_mcp_tool_call=_post_mcp_tool_call,
        write_workspace_text_file=_write_workspace_text_file,
    )


def _build_plan_prompt(goal: str, constraints: Optional[str], max_steps: int) -> str:
    return coder_tools.build_plan_prompt(goal, constraints, max_steps)


def _render_plan_markdown(goal: str, steps: list[dict[str, Any]], statuses: list[bool]) -> str:
    return coder_tools.render_plan_markdown(goal, steps, statuses)


def _coding_agent_autonomous(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return coder_tools.coding_agent_autonomous(
        payload,
        provider,
        post_mcp_tool_call=_post_mcp_tool_call,
        write_workspace_text_file=_write_workspace_text_file,
        extract_json=_extract_json,
    )


def _llm_generate(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    prompt = payload.get("text") or payload.get("prompt") or ""
    response = provider.generate(prompt)
    return {"text": response.content}


def _llm_generate_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return _llm_generate_document_spec_external(
        payload,
        provider,
        sanitize_document_spec=_sanitize_document_spec,
    )


def _sanitize_document_spec(document_spec: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(document_spec)
    blocks = sanitized.get("blocks")
    if isinstance(blocks, list):
        sanitized["blocks"] = _sanitize_document_blocks(blocks)
    return sanitized


def _sanitize_document_blocks(blocks: List[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block = dict(raw)
        block_type = block.get("type")

        if block_type == "spacer":
            # Spacer maps to an empty paragraph in DOCX; skip to avoid blank lines.
            continue

        if block_type in {"text", "paragraph", "heading", "optional_paragraph"}:
            text = block.get("text")
            if isinstance(text, str) and not text.strip():
                continue

        if block_type == "bullets":
            items = block.get("items")
            if isinstance(items, list):
                filtered_items = []
                for item in items:
                    if isinstance(item, str) and not item.strip():
                        continue
                    filtered_items.append(item)
                if not filtered_items:
                    continue
                block["items"] = filtered_items

        if block_type == "repeat":
            template = block.get("template")
            if isinstance(template, list):
                cleaned_template = _sanitize_document_blocks(template)
                if not cleaned_template:
                    continue
                block["template"] = cleaned_template

        cleaned.append(block)
    return cleaned


def _resolve_llm_iterative_tool_timeout_s(provider: LLMProvider) -> int:
    base = _resolve_llm_timeout_s(provider)
    return min(900, max(60, base * 3))


def _llm_tailor_resume_text(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return resume_llm.llm_tailor_resume_text(
        payload,
        provider,
        post_mcp_tool_call=_post_mcp_tool_call,
    )


def _select_job_context_from_memory(memory: Any) -> Dict[str, Any]:
    return resume_llm.select_job_context_from_memory(memory)


def _merge_resume_job_context(
    job: Dict[str, Any], memory_context: Dict[str, Any]
) -> Dict[str, Any]:
    return resume_llm.merge_resume_job_context(job, memory_context)


def _build_resume_job_payload(job: Dict[str, Any] | None, memory: Any) -> Dict[str, Any]:
    return resume_llm.build_resume_job_payload(job, memory)


def _llm_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_improve_tailored_resume_text(
        payload,
        provider,
        post_mcp_tool_call=_post_mcp_tool_call,
    )


def _llm_iterative_improve_tailored_resume_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_iterative_improve_tailored_resume_text(
        payload,
        provider,
        post_mcp_tool_call=_post_mcp_tool_call,
    )


def _parse_target_pages(value: Any) -> int | None:
    return resume_llm.parse_target_pages(value)


def _resolve_target_pages(payload: Dict[str, Any], job: Any) -> int | None:
    return resume_llm.resolve_target_pages(payload, job)


def _apply_resume_target_pages_policy(
    resume_doc_spec: Dict[str, Any], target_pages: int | None
) -> None:
    resume_llm.apply_resume_target_pages_policy(resume_doc_spec, target_pages)


def _llm_generate_resume_doc_spec_from_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_generate_resume_doc_spec_from_text(payload, provider)


def _normalize_skills_definition_separators(resume_doc_spec: Dict[str, Any]) -> None:
    resume_llm.normalize_skills_definition_separators(resume_doc_spec)


def _llm_generate_cover_letter_from_resume(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_generate_cover_letter_from_resume(payload, provider)


def _llm_generate_coverletter_doc_spec_from_text(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_generate_coverletter_doc_spec_from_text(payload, provider)


def _build_coverletter_doc_spec(cover_letter: Dict[str, Any], date_text: str) -> Dict[str, Any]:
    return resume_llm.build_coverletter_doc_spec(cover_letter, date_text)


def _split_cover_letter_paragraphs(body_text: str) -> List[str]:
    return resume_llm.split_cover_letter_paragraphs(body_text)


def _validate_coverletter_doc_spec(spec: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    return resume_llm.validate_coverletter_doc_spec(spec, strict)


def _llm_generate_resume_doc_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return resume_llm.llm_generate_resume_doc_spec(payload, provider)


def _ensure_required_resume_sections(content: Any) -> None:
    resume_llm.ensure_required_resume_sections(content)


def _has_required_headings(text: str, headings: list[str]) -> bool:
    return resume_llm.has_required_headings(text, headings)


def _fill_missing_dates_from_text(resume_doc_spec: Dict[str, Any], tailored_text: str) -> None:
    resume_llm.fill_missing_dates_from_text(resume_doc_spec, tailored_text)


def _ensure_certifications_section_content(
    resume_doc_spec: Dict[str, Any],
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> None:
    resume_llm.ensure_certifications_section_content(
        resume_doc_spec,
        tailored_resume=tailored_resume,
        tailored_text=tailored_text,
        candidate_resume_text=candidate_resume_text,
    )


def _find_section_heading_block_index(
    content: List[Dict[str, Any]],
    headings: set[str],
    start: int = 0,
) -> int | None:
    return resume_llm.find_section_heading_block_index(content, headings, start)


def _find_next_section_heading_block_index(content: List[Dict[str, Any]], start: int) -> int:
    return resume_llm.find_next_section_heading_block_index(content, start)


def _extract_non_empty_cert_items(blocks: List[Any]) -> list[str]:
    return resume_llm.extract_non_empty_cert_items(blocks)


def _collect_fallback_certification_lines(
    tailored_resume: Dict[str, Any] | None,
    tailored_text: str | None,
    candidate_resume_text: str | None,
) -> list[str]:
    return resume_llm.collect_fallback_certification_lines(
        tailored_resume,
        tailored_text,
        candidate_resume_text,
    )


def _certification_lines_from_tailored_resume(tailored_resume: Dict[str, Any]) -> list[str]:
    return resume_llm.certification_lines_from_tailored_resume(tailored_resume)


def _certification_lines_from_text(text: str) -> list[str]:
    return resume_llm.certification_lines_from_text(text)


def _cert_section_is_removable(blocks: List[Any]) -> bool:
    return resume_llm.cert_section_is_removable(blocks)


def _extract_candidate_resume_text_from_job(job_payload: Any) -> str | None:
    return resume_llm.extract_candidate_resume_text_from_job(job_payload)


def _extract_dates_from_section(text: str, start_heading: str, end_heading: str) -> list[str]:
    return resume_llm.extract_dates_from_section(text, start_heading, end_heading)


def _find_heading_index(lines: list[str], heading: str, start: int = 0) -> int | None:
    return resume_llm.find_heading_index(lines, heading, start)


def _is_missing_value(value: Any) -> bool:
    return resume_llm.is_missing_value(value)


def _llm_generate_tailored_resume_content(
    payload: Dict[str, Any], provider: LLMProvider
) -> Dict[str, Any]:
    return resume_llm.llm_generate_tailored_resume_content(payload, provider)


def _extract_json(text: str) -> str:
    return resume_llm.extract_json(text)


def _llm_improve_document_spec(payload: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    return _llm_improve_document_spec_external(
        payload,
        provider,
        sanitize_document_spec=_sanitize_document_spec,
    )
