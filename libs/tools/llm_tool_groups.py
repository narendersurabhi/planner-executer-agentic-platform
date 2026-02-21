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


def register_tailor_mcp_tools(
    registry,
    *,
    timeout_s: int,
    handler_iterative_improve: PayloadHandler,
    handler_tailor: PayloadHandler,
    handler_improve: PayloadHandler,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_iterative_improve_tailored_resume_text",
                description=(
                    "Iteratively improve tailored resume text until alignment threshold or max iterations"
                ),
                usage_guidance=(
                    "Provide tailored_resume (preferred). Optionally provide job, min_alignment_score "
                    "(0-100), and max_iterations. Returns the best tailored_resume plus alignment stats."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_text": {"type": "string"},
                        "tailored_resume": {"type": "object"},
                        "job": {"type": "object"},
                        "min_alignment_score": {"type": "number"},
                        "max_iterations": {"type": "integer"},
                    },
                    "required": [],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "tailored_resume": {"type": "object"},
                        "alignment_score": {"type": "number"},
                        "alignment_summary": {"type": "string"},
                        "alignment_feedback": {
                            "type": "object",
                            "properties": {
                                "top_gaps": {"type": "array", "items": {"type": "string"}},
                                "must_fix_before_95": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "missing_evidence": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "recommended_edits": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                        "iterations": {"type": "integer"},
                        "reached_threshold": {"type": "boolean"},
                        "history": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": [
                        "tailored_resume",
                        "alignment_score",
                        "alignment_summary",
                        "iterations",
                        "reached_threshold",
                    ],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_iterative_improve,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_tailor_resume_text",
                description="Tailor a resume to a job description and return structured JSON content",
                usage_guidance=(
                    "Provide a full job object in 'job'. The job should include context_json "
                    "with the job description, candidate resume, target role name, and seniority "
                    "level. The tool returns JSON resume content."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"job": {"type": "object"}},
                    "required": ["job"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"tailored_resume": {"type": "object"}},
                    "required": ["tailored_resume"],
                },
                memory_reads=["job_context"],
                memory_writes=["task_outputs"],
                examples=[
                    {
                        "task": {
                            "name": "TailorResumeText",
                            "tool_requests": ["llm_tailor_resume_text"],
                            "tool_inputs": {
                                "llm_tailor_resume_text": {
                                    "job": {
                                        "id": "job-id",
                                        "goal": "Tailor resume for target role",
                                        "context_json": {
                                            "job_description": "Paste JD here",
                                            "candidate_resume": "Paste resume here",
                                            "target_role_name": "Target role",
                                            "seniority_level": "Senior",
                                        },
                                    }
                                }
                            },
                        }
                    }
                ],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_tailor,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_improve_tailored_resume_text",
                description="Review and improve tailored resume text while preserving truthfulness",
                usage_guidance=(
                    "Provide tailored_resume (preferred) and job context (optional). "
                    "Returns improved tailored_resume plus alignment score and summary."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_text": {"type": "string"},
                        "tailored_resume": {"type": "object"},
                        "job": {"type": "object"},
                    },
                    "required": [],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "tailored_resume": {"type": "object"},
                        "alignment_score": {"type": "number"},
                        "alignment_summary": {"type": "string"},
                        "alignment_feedback": {
                            "type": "object",
                            "properties": {
                                "top_gaps": {"type": "array", "items": {"type": "string"}},
                                "must_fix_before_95": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "missing_evidence": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "recommended_edits": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["tailored_resume", "alignment_score", "alignment_summary"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.transform,
            ),
            handler=handler_improve,
        )
    )


def register_resume_llm_tools(
    registry,
    *,
    timeout_s: int,
    handler_generate_resume_doc_spec_from_text: PayloadHandler,
    handler_generate_coverletter_doc_spec_from_text: PayloadHandler,
    handler_generate_cover_letter_from_resume: PayloadHandler,
    handler_generate_resume_doc_spec: PayloadHandler,
) -> None:
    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate_resume_doc_spec_from_text",
                description="Generate a ResumeDocSpec JSON from tailored resume text",
                usage_guidance=(
                    "Provide tailored_resume (preferred) or tailored_text. Optionally provide "
                    "job context in 'job'. Optionally provide target_pages (1 or 2) to "
                    "bias output density. Generates and validates resume_doc_spec "
                    "(resume_doc_spec_validate, strict=true) before returning it."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_text": {"type": "string"},
                        "tailored_resume": {"type": "object"},
                        "job": {"type": "object"},
                        "target_pages": {"type": "integer", "enum": [1, 2]},
                    },
                    "required": [],
                },
                output_schema={
                    "type": "object",
                    "properties": {"resume_doc_spec": {"type": "object"}},
                    "required": ["resume_doc_spec"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                examples=[
                    {
                        "task": {
                            "name": "BuildResumeDocSpecFromText",
                            "tool_requests": ["llm_generate_resume_doc_spec_from_text"],
                            "deps": ["TailorResumeText"],
                            "tool_inputs": {
                                "llm_generate_resume_doc_spec_from_text": {
                                    "tailored_text": {
                                        "$ref": "tasks.TailorResumeText.output.text"
                                    },
                                    "job": {"id": "job-id", "goal": "Tailor resume"},
                                }
                            },
                        }
                    }
                ],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_generate_resume_doc_spec_from_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate_coverletter_doc_spec_from_text",
                description="Generate a CoverLetterDocSpec JSON from tailored resume text and job context",
                usage_guidance=(
                    "Provide tailored_resume (preferred) or tailored_text, plus job context in 'job'. "
                    "Generates and validates coverletter_doc_spec (strict) before returning it."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_text": {"type": "string"},
                        "tailored_resume": {"type": "object"},
                        "job": {"type": "object"},
                        "today_pretty": {"type": "string"},
                        "today": {"type": "string"},
                    },
                    "required": [],
                },
                output_schema={
                    "type": "object",
                    "properties": {"coverletter_doc_spec": {"type": "object"}},
                    "required": ["coverletter_doc_spec"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_generate_coverletter_doc_spec_from_text,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate_cover_letter_from_resume",
                description="Generate a structured cover_letter JSON from tailored resume and job description",
                usage_guidance=(
                    "Provide tailored_resume (preferred) or tailored_text. Optionally provide "
                    "job context in 'job'. Returns cover_letter JSON suitable for "
                    "cover_letter_generate_ats_docx."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "tailored_text": {"type": "string"},
                        "tailored_resume": {"type": "object"},
                        "job": {"type": "object"},
                    },
                    "required": [],
                },
                output_schema={
                    "type": "object",
                    "properties": {"cover_letter": {"type": "object"}},
                    "required": ["cover_letter"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_generate_cover_letter_from_resume,
        )
    )

    registry.register(
        Tool(
            spec=ToolSpec(
                name="llm_generate_resume_doc_spec",
                description="Generate a ResumeDocSpec JSON using an LLM",
                usage_guidance=(
                    "Use for resume or CV generation when a ResumeDocSpec is needed. "
                    "Provide job context in 'job'. You MUST provide 'tailored_resume' from a "
                    "prior task. The output is a 'resume_doc_spec' object "
                    "matching the required schema and style (header, sections, roles, education, "
                    "certifications, and styles). Optionally provide target_pages (1 or 2) "
                    "to control output density."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "object"},
                        "tailored_resume": {"type": ["object", "string"]},
                        "target_pages": {"type": "integer", "enum": [1, 2]},
                    },
                    "required": ["job", "tailored_resume"],
                },
                output_schema={
                    "type": "object",
                    "properties": {"resume_doc_spec": {"type": "object"}},
                    "required": ["resume_doc_spec"],
                },
                memory_reads=["job_context", "task_outputs"],
                memory_writes=["task_outputs"],
                examples=[
                    {
                        "task": {
                            "name": "BuildResumeDocSpec",
                            "tool_requests": ["llm_generate_resume_doc_spec"],
                            "deps": ["TailorResumeContent"],
                            "tool_inputs": {
                                "llm_generate_resume_doc_spec": {
                                    "job": {"id": "job-id", "goal": "Tailor resume"},
                                    "tailored_resume": {
                                        "$ref": "tasks.TailorResumeContent.output.tailored_resume"
                                    },
                                }
                            },
                        }
                    }
                ],
                timeout_s=timeout_s,
                risk_level=RiskLevel.high,
                tool_intent=ToolIntent.generate,
            ),
            handler=handler_generate_resume_doc_spec,
        )
    )
