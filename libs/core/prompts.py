from __future__ import annotations

import json
from typing import Any


def document_spec_prompt(job: dict[str, Any], allowed_block_types: list[str]) -> str:
    job_json = json.dumps(job, ensure_ascii=False, indent=2, default=str)
    allowed_json = json.dumps(allowed_block_types, ensure_ascii=False)
    return (
        "You are generating a DocumentSpec JSON object only. No prose, no markdown.\n"
        "DocumentSpec must include: blocks (array), optional tokens (object), optional theme (object).\n"
        f"Allowed block types: {allowed_json}\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object."
    )


def document_spec_improve_prompt(
    document_spec: dict[str, Any],
    validation_report: dict[str, Any],
    allowed_block_types: list[str] | None = None,
) -> str:
    spec_json = json.dumps(document_spec, ensure_ascii=False, indent=2, default=str)
    report_json = json.dumps(validation_report, ensure_ascii=False, indent=2, default=str)
    allowed_json = (
        json.dumps(allowed_block_types, ensure_ascii=False)
        if isinstance(allowed_block_types, list)
        else "null"
    )
    return (
        "You are improving a DocumentSpec JSON object based on a validation report. "
        "Return ONLY the improved JSON object.\n"
        "Goals:\n"
        "- Fix all validation errors.\n"
        "- Address warnings when possible without changing intent.\n"
        "- Keep content concise and preserve the original structure and meaning.\n"
        f"Allowed block types: {allowed_json}\n"
        f"Validation report: {report_json}\n"
        f"Original DocumentSpec: {spec_json}\n"
        "Return ONLY the improved JSON object."
    )


def resume_doc_spec_prompt(job: dict[str, Any], tailored_resume: Any | None = None) -> str:
    job_json = json.dumps(job, ensure_ascii=False, indent=2, default=str)
    tailored_json = (
        json.dumps(tailored_resume, ensure_ascii=False, indent=2, default=str)
        if tailored_resume is not None
        else "null"
    )
    return (
        "You are generating a ResumeDocSpec JSON object only. No prose, no markdown.\n"
        "Follow the exact schema and style shown below.\n"
        "Required top-level keys: schema_version, doc_type, title, page, defaults, content, styles.\n"
        "Use the same structure and field names as the example. You may change text values to fit the job.\n"
        "If tailored resume content is provided, use it as the source of truth and do not invent new content.\n"
        "Example (style and structure):\n"
        "{\n"
        "  \"schema_version\": \"1.0\",\n"
        "  \"doc_type\": \"resume\",\n"
        "  \"title\": \"Full Name - Resume\",\n"
        "  \"page\": {\n"
        "    \"size\": \"LETTER\",\n"
        "    \"margins_in\": {\"top\": 0.5, \"right\": 0.5, \"bottom\": 0.5, \"left\": 0.5}\n"
        "  },\n"
        "  \"defaults\": {\"font_family\": \"Calibri\", \"font_size_pt\": 11, \"line_spacing\": 1.05},\n"
        "  \"content\": [\n"
        "    {\n"
        "      \"type\": \"header\",\n"
        "      \"align\": \"left\",\n"
        "      \"blocks\": [\n"
        "        {\"type\": \"text\", \"style\": \"name\", \"text\": \"Full Name\"},\n"
        "        {\"type\": \"text\", \"style\": \"title\", \"text\": \"Role\"},\n"
        "        {\"type\": \"text\", \"style\": \"contact\", \"text\": \"Location | Phone | Email\"},\n"
        "        {\"type\": \"text\", \"style\": \"contact\", \"text\": \"LinkedIn | GitHub\"}\n"
        "      ]\n"
        "    },\n"
        "    {\"type\": \"section_heading\", \"text\": \"SUMMARY\"},\n"
        "    {\"type\": \"paragraph\", \"text\": \"...\"},\n"
        "    {\"type\": \"section_heading\", \"text\": \"SKILLS\"},\n"
        "    {\"type\": \"definition_list\", \"items\": [{\"term\": \"...\", \"definition\": \"...\"}]},\n"
        "    {\"type\": \"section_heading\", \"text\": \"EXPERIENCE\"},\n"
        "    {\"type\": \"role\", \"company\": \"...\", \"location\": \"...\", \"title\": \"...\", \"dates\": \"...\", \"bullets\": [\"...\"]},\n"
        "    {\"type\": \"section_heading\", \"text\": \"EDUCATION\"},\n"
        "    {\"type\": \"education\", \"degree\": \"...\", \"school\": \"...\", \"location\": \"...\", \"dates\": \"...\"},\n"
        "    {\"type\": \"section_heading\", \"text\": \"CERTIFICATIONS\"},\n"
        "    {\"type\": \"bullets\", \"items\": [\"...\"]}\n"
        "  ],\n"
        "  \"styles\": {\n"
        "    \"name\": {\"bold\": true, \"size_pt\": 16},\n"
        "    \"title\": {\"bold\": true, \"size_pt\": 12},\n"
        "    \"contact\": {\"size_pt\": 10},\n"
        "    \"section_heading\": {\"bold\": true, \"all_caps\": true, \"space_before_pt\": 10, \"space_after_pt\": 4}\n"
        "  }\n"
        "}\n"
        f"Tailored resume content (JSON): {tailored_json}\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object."
    )


def tailored_resume_content_prompt(job: dict[str, Any]) -> str:
    job_json = json.dumps(job, ensure_ascii=False, indent=2, default=str)
    return (
        "You are generating tailored resume content as JSON only. No prose, no markdown.\n"
        "Create a JSON object with these top-level keys:\n"
        "- summary: string\n"
        "- skills: array of {term: string, definition: string}\n"
        "- experience: array of {company, location, title, dates, bullets: array of strings}\n"
        "- education: array of {degree, school, location, dates}\n"
        "- certifications: array of strings\n"
        "Use the job goal and job.context_json for candidate profile and target role.\n"
        "Focus on tailoring: emphasize relevant skills, impact, and keywords for the target role.\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object."
    )
