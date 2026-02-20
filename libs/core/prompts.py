from __future__ import annotations

import json
from typing import Any


def document_spec_prompt(job: dict[str, Any], allowed_block_types: list[str]) -> str:
    job_json = json.dumps(job, ensure_ascii=False, indent=2, default=str)
    allowed_json = json.dumps(allowed_block_types, ensure_ascii=False)
    return (
        "You are generating a DocumentSpec JSON object only. No prose, no markdown.\n"
        "DocumentSpec must include: blocks (array), optional tokens (object), optional theme (object).\n"
        "Do not emit spacer blocks.\n"
        "Do not emit empty text/paragraph/heading blocks.\n"
        "Do not include empty string items in bullets.\n"
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
        "Use tailored_resume.header to populate name, title, and contact lines in the header blocks.\n"
        "If tailored_resume.experience roles include groups, render each group as:\n"
        '- {"type": "paragraph", "style": "role_group_heading", "text": "Group Heading"}\n'
        '- followed by {"type": "bullets", "items": ["..."]}\n'
        "If a role has groups, you may still include top-level bullets before groups if needed.\n"
        "Always include a 'bullets' array on role blocks (use [] if all bullets are grouped).\n"
        "If tailored_resume includes projects/open_source items, include them as an optional section:\n"
        '- {"type": "section_heading", "text": "OPEN SOURCE"} (or "PROJECTS")\n'
        '- then {"type": "bullets", "items": ["Project: short description | URL", ...]}\n'
        "For CERTIFICATIONS, preserve and render any credential URLs when available\n"
        '(for example Credly public URLs) appended as: "Name - Issuer (Year) | https://...".\n'
        'For SKILLS definition_list items, separate skills with commas (", "), not semicolons.\n'
        "If target_pages is provided in job or job.context_json (supported values: 1 or 2), "
        "optimize content density to that page budget.\n"
        "For target_pages=1: keep content concise (summary 2-3 lines, skills up to 5 groups, "
        "most recent role 4-5 bullets, older roles 1-2 bullets).\n"
        "For target_pages=2: allow fuller detail (summary 3-4 lines, skills up to 7 groups, "
        "most recent role up to 8 bullets, older roles up to 3-4 bullets).\n"
        "Do not use placeholders like [Phone] or [Email]. If header fields are missing, return:\n"
        "{\n"
        '  "error": "missing_required_fields",\n'
        '  "missing_fields": ["header.name", "header.phone"]\n'
        "}\n"
        "Example (style and structure):\n"
        "{\n"
        '  "schema_version": "1.0",\n'
        '  "doc_type": "resume",\n'
        '  "title": "Full Name - Resume",\n'
        '  "page": {\n'
        '    "size": "LETTER",\n'
        '    "margins_in": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5}\n'
        "  },\n"
        '  "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.05},\n'
        '  "content": [\n'
        "    {\n"
        '      "type": "header",\n'
        '      "align": "left",\n'
        '      "blocks": [\n'
        '        {"type": "text", "style": "name", "text": "Full Name"},\n'
        '        {"type": "text", "style": "title", "text": "Role"},\n'
        '        {"type": "text", "style": "contact", "text": "Location | Phone | Email"},\n'
        '        {"type": "text", "style": "contact", "text": "LinkedIn | GitHub"}\n'
        "      ]\n"
        "    },\n"
        '    {"type": "section_heading", "text": "SUMMARY"},\n'
        '    {"type": "paragraph", "text": "..."},\n'
        '    {"type": "section_heading", "text": "SKILLS"},\n'
        '    {"type": "definition_list", "items": [{"term": "...", "definition": "..."}]},\n'
        '    {"type": "section_heading", "text": "EXPERIENCE"},\n'
        '    {"type": "role", "company": "...", "location": "...", "title": "...", "dates": "...", "bullets": ["..."]},\n'
        '    {"type": "paragraph", "style": "role_group_heading", "text": "Decisioning and Scoring Platforms"},\n'
        '    {"type": "bullets", "items": ["..."]},\n'
        '    {"type": "section_heading", "text": "EDUCATION"},\n'
        '    {"type": "education", "degree": "...", "school": "...", "location": "...", "dates": "..."},\n'
        '    {"type": "section_heading", "text": "CERTIFICATIONS"},\n'
        '    {"type": "bullets", "items": ["..."]},\n'
        '    {"type": "section_heading", "text": "OPEN SOURCE"},\n'
        '    {"type": "bullets", "items": ["Project Name: short description | https://..."]}\n'
        "  ],\n"
        '  "styles": {\n'
        '    "name": {"bold": true, "size_pt": 16},\n'
        '    "title": {"bold": true, "size_pt": 12},\n'
        '    "contact": {"size_pt": 10},\n'
        '    "section_heading": {"bold": true, "all_caps": true, "space_before_pt": 10, "space_after_pt": 4},\n'
        '    "role_group_heading": {"bold": true, "italic": true, "space_before_pt": 6, "space_after_pt": 2}\n'
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
        "- experience: array of {company, location, title, dates, bullets: array of strings, "
        "optional groups: array of {heading: string, bullets: array of strings}}\n"
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
        "- Output ONLY JSON (no prose, no markdown).\n"
        "- Never invent experience. Only use skills, projects, and outcomes supported by the candidate resume.\n"
        "- Prefer concrete nouns and clear engineering or business outcomes. Avoid buzzword stacking.\n"
        "- Use job description keywords naturally. Do not paste or quote the job description.\n\n"
        "- Do NOT use en-dashes or em-dashes. Use a simple hyphen '-'.\n\n"
        "Metrics and credibility policy:\n"
        "- If a metric exists in the input, keep it.\n"
        "- If not, include a metric only if it is clearly defensible from context, and use modest ranges.\n"
        "- Avoid fake precision. Use round numbers and ranges.\n"
        "- Do not attach platform wide metrics to a single feature unless scope is explicit.\n"
        "- If a statement could be interpreted as inflated, add scope context such as internal vs external or batch vs realtime.\n\n"
        "Bullet writing rules:\n"
        "- Every bullet must follow this formula:\n"
        "  Action verb + what you built or owned + how tech and approach + impact metric or outcome.\n"
        "- One sentence per bullet.\n"
        "- 22 to 35 words per bullet.\n"
        "- Start with a strong past tense verb and avoid repeating the same verb in adjacent bullets.\n\n"
        "Summary rules:\n"
        "- Write 3 to 4 lines.\n"
        "- Match the target role and level.\n"
        "- Include 2 to 3 proof points from the resume that are most relevant to the JD.\n"
        "- Keep it skimmable and outcome focused.\n\n"
        "SUMMARY years of experience guidance:\n"
        "- Decide whether to include years of experience in the summary using these rules.\n"
        "- Include years if the job description explicitly asks for a minimum years requirement or if years strengthens the initial recruiter scan.\n"
        "- Omit years if the summary would become crowded or if scope and impact already make seniority obvious.\n"
        '- If included, mention years only once in the first line and keep it short, for example "8 plus years" or "10 plus years".\n'
        '- Do not include total career years if it conflicts with the role focus. Prefer role relevant years, for example "8 plus years building production ML and GenAI" rather than total years.\n'
        "- Never invent years. Use only what is supported by the candidate resume dates and stated experience.\n\n"
        "Skills rules:\n"
        "- Provide grouped skills ordered by relevance to the JD.\n"
        "- Within each group, list the most relevant skills first.\n"
        "- Use 5 to 7 groups max.\n"
        "- Do not list skills not present in the candidate resume.\n\n"
        "Experience rules:\n"
        "- Include every role from the candidate resume in the same order.\n"
        "- Most recent role: 6 to 8 bullets.\n"
        "- Older roles: 2 to 3 bullets each.\n"
        "- Order bullets by relevance to the JD.\n\n"
        "- If the candidate resume uses subheadings within a role (for example grouping projects or domains), "
        "preserve those as experience groups.\n"
        "- When groups are used, keep each group heading short and add 2 to 4 bullets under each group.\n\n"
        "Education and certifications rules:\n"
        "- Only include entries present in the candidate resume.\n"
        "- If a certification includes a public credential URL in the source resume "
        "(for example a Credly badge URL), preserve it exactly in the certification object.\n"
        "- If none are present, return empty arrays.\n\n"
        "JSON OUTPUT SCHEMA\n"
        "Return a single JSON object with these top level keys:\n"
        "1) schema_version: string\n"
        "2) header: object\n"
        "3) summary: string\n"
        "4) skills: array\n"
        "5) experience: array\n"
        "6) education: array\n"
        "7) certifications: array\n\n"
        "Detailed structure:\n"
        'schema_version: "1.0"\n\n'
        "header:\n"
        "- name: string\n"
        "- title: string\n"
        "- location: string\n"
        "- phone: string\n"
        "- email: string\n"
        "- links: object with optional keys: linkedin, github, other\n\n"
        "summary:\n"
        "- string with newline characters for line breaks\n\n"
        "skills:\n"
        "- array of skill_group objects ordered by relevance\n"
        "- each skill_group has:\n"
        "  - group_name: string\n"
        "  - items: array of strings ordered by relevance within the group\n\n"
        "experience:\n"
        "- array of role objects in the same order as the candidate resume\n"
        "- each role has:\n"
        "  - company: string\n"
        "  - title: string\n"
        "  - location: string\n"
        "  - dates: string\n"
        "  - bullets: array of strings (use when there are no subheadings)\n"
        "  - groups: optional array of group objects when the resume uses subheadings\n"
        "    - heading: string\n"
        "    - bullets: array of strings\n\n"
        "education:\n"
        "- array of education objects\n"
        "- each has: degree, school, location, dates\n\n"
        "certifications:\n"
        "- array of certification objects\n"
        "- each has: name, issuer, year\n"
        "- optional: url (public credential URL when available, e.g., Credly)\n"
        "- when url is present, keep it as a plain URL string (not markdown link syntax)\n\n"
        "Required field policy:\n"
        "- Extract header and experience fields from the candidate resume. Do not invent them.\n"
        "- Do not use placeholders like Unknown.\n"
        "- If header.title is missing in the candidate resume, set it from target_role_name.\n"
        "- If both are missing, use the most recent role title.\n"
        "- Do not return missing_required_fields for header.title when a fallback is available.\n"
        "- If any required fields are missing in the candidate resume, return the error JSON below.\n\n"
        "If any required fields are missing, return ONLY this JSON object:\n"
        "{\n"
        '  "error": "missing_required_fields",\n'
        '  "missing_fields": ["header.name", "header.email", "experience[0].company", "experience[0].title"]\n'
        "}\n\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object.\n"
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
        "If the text contains an OPEN SOURCE/OPEN SOURCE (SELECTED)/PROJECTS section, include it in output.\n"
        "Map it as:\n"
        '- {"type": "section_heading", "text": "OPEN SOURCE"} (or "PROJECTS")\n'
        '- then {"type": "bullets", "items": ["Project Name: short description | URL", ...]}\n'
        "In CERTIFICATIONS, preserve any URLs present in the source text\n"
        "(for example Credly badge public URLs) and include them in certification bullets.\n"
        'For SKILLS definition_list items, separate skills with commas (", "), not semicolons.\n'
        "If target_pages is provided in job or job.context_json (supported values: 1 or 2), "
        "optimize content density to that page budget.\n"
        "For target_pages=1: keep content concise (summary 2-3 lines, skills up to 5 groups, "
        "most recent role 4-5 bullets, older roles 1-2 bullets).\n"
        "For target_pages=2: allow fuller detail (summary 3-4 lines, skills up to 7 groups, "
        "most recent role up to 8 bullets, older roles up to 3-4 bullets).\n"
        "Within EXPERIENCE, treat standalone subheadings under a company (lines that are not role headers or bullets)\n"
        "as group headings and render them as:\n"
        '- {"type": "paragraph", "style": "role_group_heading", "text": "Group Heading"}\n'
        '- followed by {"type": "bullets", "items": ["..."]}\n'
        "Always include a 'bullets' array on role blocks (use [] if all bullets are grouped).\n"
        "Do not invent content beyond the tailored text.\n"
        "Example (style and structure):\n"
        "{\n"
        '  "schema_version": "1.0",\n'
        '  "doc_type": "resume",\n'
        '  "title": "Full Name - Resume",\n'
        '  "page": {\n'
        '    "size": "LETTER",\n'
        '    "margins_in": {"top": 0.5, "right": 0.5, "bottom": 0.5, "left": 0.5}\n'
        "  },\n"
        '  "defaults": {"font_family": "Calibri", "font_size_pt": 11, "line_spacing": 1.05},\n'
        '  "content": [\n'
        "    {\n"
        '      "type": "header",\n'
        '      "align": "left",\n'
        '      "blocks": [\n'
        '        {"type": "text", "style": "name", "text": "Full Name"},\n'
        '        {"type": "text", "style": "title", "text": "Role"},\n'
        '        {"type": "text", "style": "contact", "text": "Location | Phone | Email"},\n'
        '        {"type": "text", "style": "contact", "text": "LinkedIn | GitHub"}\n'
        "      ]\n"
        "    },\n"
        '    {"type": "section_heading", "text": "SUMMARY"},\n'
        '    {"type": "paragraph", "text": "..."},\n'
        '    {"type": "section_heading", "text": "SKILLS"},\n'
        '    {"type": "definition_list", "items": [{"term": "...", "definition": "..."}]},\n'
        '    {"type": "section_heading", "text": "EXPERIENCE"},\n'
        '    {"type": "role", "company": "...", "location": "...", "title": "...", "dates": "...", "bullets": ["..."]},\n'
        '    {"type": "paragraph", "style": "role_group_heading", "text": "Decisioning and Scoring Platforms"},\n'
        '    {"type": "bullets", "items": ["..."]},\n'
        '    {"type": "section_heading", "text": "EDUCATION"},\n'
        '    {"type": "education", "degree": "...", "school": "...", "location": "...", "dates": "..."},\n'
        '    {"type": "section_heading", "text": "CERTIFICATIONS"},\n'
        '    {"type": "bullets", "items": ["..."]},\n'
        '    {"type": "section_heading", "text": "OPEN SOURCE"},\n'
        '    {"type": "bullets", "items": ["Project Name: short description | https://..."]}\n'
        "  ],\n"
        '  "styles": {\n'
        '    "name": {"bold": true, "size_pt": 16},\n'
        '    "title": {"bold": true, "size_pt": 12},\n'
        '    "contact": {"size_pt": 10},\n'
        '    "section_heading": {"bold": true, "all_caps": true, "space_before_pt": 10, "space_after_pt": 4},\n'
        '    "role_group_heading": {"bold": true, "italic": true, "space_before_pt": 6, "space_after_pt": 2}\n'
        "  }\n"
        "}\n"
        f"Tailored resume text:\n{tailored_text}\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object."
    )


def resume_tailoring_improve_prompt(
    tailored_resume: dict[str, Any],
    job: dict[str, Any] | None = None,
    *,
    evaluator_feedback: str | None = None,
    previous_alignment_score: float | None = None,
    min_required_changes: int | None = None,
) -> str:
    job_json = (
        json.dumps(job, ensure_ascii=False, indent=2, default=str)
        if isinstance(job, dict)
        else "null"
    )
    tailored_json = json.dumps(tailored_resume, ensure_ascii=False, indent=2, default=str)
    feedback_section = ""
    if isinstance(evaluator_feedback, str) and evaluator_feedback.strip():
        feedback_section = (
            "Iteration guidance from the independent evaluator:\n"
            f"- Previous evaluated alignment score: {previous_alignment_score if previous_alignment_score is not None else 'unknown'}\n"
            f"- Feedback to address: {evaluator_feedback.strip()}\n"
        )
    change_target = max(4, int(min_required_changes or 8))
    return (
        "You are a senior resume reviewer. Improve the tailored resume JSON for clarity, "
        "impact, and credibility while keeping the content truthful and aligned with the job.\n"
        "Rules:\n"
        "- Output ONLY JSON (no prose).\n"
        "- Preserve the same structure and required keys.\n"
        "- You may add new skills and experience bullets ONLY when they are clearly defendable from input evidence.\n"
        "- Defendable means directly grounded in existing roles, technologies, projects, metrics, or responsibilities already present in the provided resume context.\n"
        "- Never invent new employers, job titles, projects, durations, certifications, or domain claims not evidenced in input.\n"
        "- Keep bullets action-oriented and outcome-focused.\n"
        "- Preserve any experience groups (group headings and grouped bullets) if present.\n"
        "- Preserve certification URLs exactly when present in input (for example Credly badge URLs).\n"
        "- If you adjust metrics, keep them defensible and use modest ranges.\n"
        "- Keep each bullet as one sentence, 22 to 35 words.\n"
        "- Do NOT use en-dashes or em-dashes. Use a simple hyphen '-'.\n"
        f"- Make at least {change_target} major, high-impact edit(s) to the tailored_resume content.\n"
        "- Avoid micro wordsmithing only; perform substantial improvements across multiple sections.\n"
        "- In every iteration, target all key sections where possible: summary, skills, and experience.\n"
        "- Each edit must improve alignment to the target job (skills coverage, evidence quality, or impact clarity).\n"
        "Return JSON with keys:\n"
        "- tailored_resume: object (same schema as input, including header)\n"
        "- alignment_score: number (0 to 100)\n"
        "- alignment_summary: string (1 to 3 sentences on JD alignment)\n"
        "- applied_changes: array of short strings describing each major edit made\n"
        "- For each newly added skill or bullet, include an applied_changes entry describing the supporting evidence briefly.\n"
        f"{feedback_section}"
        "Input tailored resume JSON:\n"
        f"{tailored_json}\n\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object.\n"
    )


def resume_tailoring_evaluation_prompt(
    tailored_resume: dict[str, Any], job: dict[str, Any] | None = None
) -> str:
    job_json = (
        json.dumps(job, ensure_ascii=False, indent=2, default=str)
        if isinstance(job, dict)
        else "null"
    )
    tailored_json = json.dumps(tailored_resume, ensure_ascii=False, indent=2, default=str)
    return (
        "You are an independent resume evaluator. You only evaluate alignment; do not rewrite content.\n"
        "Score how well the provided tailored_resume matches the target job description.\n"
        "Return ONLY one JSON object with keys:\n"
        "- alignment_score: number from 0 to 100\n"
        "- alignment_summary: 1 to 3 direct sentences on strengths and gaps\n"
        "- top_gaps: array of 2 to 5 short strings (blunt gap statements)\n"
        "- must_fix_before_95: array of 1 to 4 short strings with imperative phrasing\n"
        "- missing_evidence: array of 1 to 4 short strings naming missing proof/metrics/examples\n"
        "- recommended_edits: array of 2 to 6 short strings with concrete resume edits\n"
        "Scoring rubric:\n"
        "- Role fit and seniority match\n"
        "- Required skills and platform/tool relevance\n"
        "- Production impact evidence with credible metrics\n"
        "- Breadth across architecture, implementation, and operations\n"
        "- Reward defendable inferred skills and bullets when evidence is explicit; penalize unsupported additions\n"
        "- Avoid inflated claims and unsupported keywords\n"
        "- Be blunt and specific; avoid vague wording like 'could be stronger'\n"
        "Do not include markdown or extra keys.\n"
        f"Tailored Resume (JSON):\n{tailored_json}\n\n"
        f"Job (JSON):\n{job_json}\n"
        "Return ONLY the JSON object.\n"
    )


def cover_letter_from_resume_prompt(
    tailored_resume: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> str:
    resume_json = json.dumps(tailored_resume, ensure_ascii=False, indent=2, default=str)
    job_json = (
        json.dumps(job, ensure_ascii=False, indent=2, default=str)
        if isinstance(job, dict)
        else "null"
    )
    return (
        "You are generating a cover letter JSON object only. No prose, no markdown.\n"
        "Create a concise, credible cover letter tailored to the job description using the provided tailored resume.\n"
        "Do not invent experience beyond the resume.\n"
        "Keep body to 4-5 short paragraphs, each 2-4 sentences.\n"
        "Use plain ATS-friendly language.\n"
        "Return ONLY one JSON object with keys:\n"
        "- full_name (string)\n"
        "- location (string, optional allowed empty)\n"
        "- phone (string, optional allowed empty)\n"
        "- email (string, optional allowed empty)\n"
        "- linkedin_url (string, optional allowed empty)\n"
        "- github_url (string, optional allowed empty)\n"
        "- company (string)\n"
        "- role_title (string)\n"
        "- hiring_manager (string, optional allowed empty)\n"
        "- recipient_line (string; use 'Hiring Team' if manager unknown)\n"
        "- salutation (string; default 'Dear Hiring Team,')\n"
        "- body (string; paragraphs separated by blank lines)\n"
        "- closing (string; default 'Sincerely,')\n"
        "Required quality rules:\n"
        "- Mention role_title and company in the first paragraph.\n"
        "- Include 2-3 concrete evidence points from the resume (scale, latency, reliability, impact).\n"
        "- Keep claims defensible and aligned with resume content.\n"
        "- Do not include markdown links; use plain text URLs.\n"
        "Tailored Resume (JSON):\n"
        f"{resume_json}\n\n"
        f"Job (JSON): {job_json}\n"
        "Return ONLY the JSON object.\n"
    )
