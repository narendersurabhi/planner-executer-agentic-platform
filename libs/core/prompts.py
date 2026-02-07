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

def resume_tailoring_prompt(job: dict[str, Any]) -> str:
    job_json = json.dumps(job, ensure_ascii=False, indent=2, default=str)
    return (
        "You are an expert resume writer and a hiring manager for the target role. "
        "Your goal is to tailor the candidate resume to the job description with ATS friendly "
        "language while staying credible and defensible.\n\n"
        "Global constraints:\n"
        "- Output ONLY plain text.\n"
        # "- Do not use any of these characters anywhere: < > [ ] \" { } \\\n"
        "- No em dash or en dash. Use simple hyphen or words like to instead.\n"
        "- Never invent experience. Only use skills, projects, and outcomes supported by the candidate resume.\n"
        "- Prefer concrete nouns and clear engineering or business outcomes. Avoid buzzword stacking.\n"
        "- Use the job description keywords naturally. Do not paste or quote the job description.\n\n"
        "Metrics policy:\n"
        "- If a metric exists in the input, keep it.\n"
        "- If not, include a metric only if it is clearly defensible from context, and use modest ranges.\n"
        "- Avoid fake precision. Use round numbers and ranges.\n"
        "- Do not attach platform wide metrics to a single feature unless scope is explicit.\n\n"
        "Number formatting guidance:\n"
        "- Use standard symbols when helpful for readability, including percent sign, <=, >=, and parentheses.\n"
        "- Keep formatting consistent across the resume.\n"
        "- Prefer simple ranges like 250-400 ms and 50-150 RPS.\n"
        "- Use p95 and p99 for latency when available.\n\n"
        "Bullet formula must be used for every bullet:\n"
        "- Action verb + what you built or owned + how tech and approach + impact metric or outcome.\n"
        "- One sentence per bullet.\n"
        "- 22 to 35 words per bullet.\n"
        "- Start with a strong past tense verb and avoid repeating the same verb in adjacent bullets.\n"
        "- Prefer concrete nouns like services, APIs, pipelines, orchestration, routing, caching, indexes, dashboards, release gates.\n"
        "- Mention tech only when it improves relevance or clarity, not as a tool list.\n\n"
        "Task:\n"
        "Tailor the candidate resume to the job description and generate a professional Summary, "
        "grouped Skills, and role specific Experience bullets.\n\n"
        "Output requirements:\n"
        "SECTION 1 SUMMARY\n"
        "- Write 3 to 4 lines.\n"
        "- Make it match the target role and level.\n"
        "- Include 2 to 3 proof points from the resume that are most relevant to the JD.\n"
        "- Keep it skimmable and outcome focused.\n\n"
        "SECTION 2 SKILLS\n"
        "- Provide grouped skills ordered by relevance to the JD.\n"
        "- Within each group, list the most relevant skills first.\n"
        "- Keep each group on one line.\n"
        "- Do not list skills not present in the resume.\n"
        "- Use 5 to 7 groups max.\n\n"
        "SECTION 3 EXPERIENCE\n"
        "For each role in the candidate resume:\n"
        "- Output the role header as: Company - Title - Location - Dates\n"
        "- Include every role from the candidate resume in the same order.\n"
        "- Do not drop older roles.\n"
        "- If any header field is missing in the resume, write Unknown for that field.\n"
        "- Most recent role: 6 to 10 bullets.\n"
        "- Older roles: 2 to 3 bullets each.\n"
        "- Each bullet must be one sentence, 22 to 35 words.\n"
        "- Use the required bullet formula.\n"
        "- Order bullets by relevance to the JD.\n\n"
        "SECTION 4 CREDIBILITY GATE\n"
        "- Identify bullets that might trigger skepticism due to ambiguity, scope mismatch, or aggressive metrics.\n"
        "- Rewrite those bullets to be more defensible while keeping them strong.\n"
        "- Return only the rewritten bullets.\n\n"
        "SECTION 5 ONE PAGE SELECTION\n"
        "- Return the best 6 to 8 bullets for the most recent role, maximizing match to the JD.\n"
        "- Do not repeat the same metric more than twice.\n"
        "- Ensure at least 3 bullets mention performance, reliability, quality, cost, or operational outcomes when applicable.\n"
        "- Return only the final bullet list.\n\n"
        "SECTION 6 EDUCATION\n"
        "- List each education entry on one line.\n"
        "- Format: Degree - School - Location - Dates.\n"
        "- Only include entries present in the candidate resume.\n\n"
        "SECTION 7 CERTIFICATIONS\n"
        "- List each certification on one line.\n"
        "- Format: Certification - Issuer - Year.\n"
        "- Only include entries present in the candidate resume.\n\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY plain text following the section headings exactly.\n"
    )


def resume_doc_spec_from_text_prompt(tailored_text: str, job: dict[str, Any] | None = None) -> str:
    job_json = (
        json.dumps(job, ensure_ascii=False, indent=2, default=str)
        if isinstance(job, dict)
        else "null"
    )
    return (
        "You are generating a ResumeDocSpec JSON object only. No prose, no markdown.\n"
        "Follow the exact schema and style shown below.\n"
        "Required top-level keys: schema_version, doc_type, title, page, defaults, content, styles.\n"
        "Use the same structure and field names as the example. You may change text values to fit the input.\n"
        "Parse the tailored resume text into Summary, Skills, Experience, Education, Certifications.\n"
        "Do not invent content beyond the tailored text.\n"
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
        f"Tailored resume text:\n{tailored_text}\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object."
    )


def resume_tailoring_improve_prompt(
    tailored_text: str, job: dict[str, Any] | None = None
) -> str:
    job_json = (
        json.dumps(job, ensure_ascii=False, indent=2, default=str)
        if isinstance(job, dict)
        else "null"
    )
    return (
        "You are a senior resume reviewer. Improve the tailored resume text for clarity, "
        "impact, and credibility while keeping the content truthful and aligned with the job.\n"
        "Rules:\n"
        "- Output ONLY JSON (no prose).\n"
        "- Preserve the same section headings and ordering in the improved text.\n"
        "- Do not introduce new experience or skills not supported by the input.\n"
        "- Keep bullets action-oriented and outcome-focused.\n"
        "- If you adjust metrics, keep them defensible and use modest ranges.\n"
        "- Keep each bullet as one sentence, 22 to 35 words.\n"
        "Return JSON with keys:\n"
        "- tailored_text: string (full improved resume text with SECTION headings)\n"
        "- alignment_score: number (0 to 100)\n"
        "- alignment_summary: string (1 to 3 sentences on JD alignment)\n"
        "Input tailored resume text:\n"
        f"{tailored_text}\n\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object.\n"
    )
