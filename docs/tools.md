# Tools

Tools are registered in libs/core/tool_registry.py with explicit schemas, usage guidance, and enforced timeouts.
The http_fetch tool is optional and guarded by TOOL_HTTP_FETCH_ENABLED and TOOL_HTTP_FETCH_ALLOWLIST.
File read/write tools only allow paths under /shared/artifacts unless using workspace_* tools.
Workspace tools operate under WORKSPACE_DIR (defaults to repo root inside the container).

Built-in tools:

- json_transform
- math_eval
- text_summarize
- llm_generate
- llm_generate_cover_letter_from_resume
- llm_generate_coverletter_doc_spec_from_text
- coding_agent_generate
- file_write_artifact
- file_write_text
- file_write_code
- file_read_text
- list_files
- workspace_write_text
- workspace_write_code
- workspace_read_text
- workspace_list_files
- artifact_move
- derive_output_filename
- run_tests
- search_text
- sleep
- http_fetch (optional)
- docx_render (requires output_path)
- docx_generate_from_spec
- document_spec_validate
- coverletter_doc_spec_to_document_spec

Docx render schema pattern:
- Keep a JSON schema per template (e.g., resume.json, cover_letter.json).
- Planner should generate JSON that matches the schema before calling docx_render.
- docx_render should use schema_ref/template_id that matches the template, and output_path is required.
- document_spec_validate can be used as a preflight check before rendering.

Notes on file write tools:
- file_write_artifact: quick text output; path optional (defaults to artifact.txt).
- file_write_text: write any text file; path required.
- file_write_code: write code files; path required and must include a code extension.

Notes on workspace tools:
- workspace_write_text: write any text file under the workspace; path required.
- workspace_write_code: write code files under the workspace; path required and must include a code extension.
- workspace_read_text: read text from the workspace; path required.
- workspace_list_files: list files and directories under the workspace.
- artifact_move: move a file from /shared/artifacts to the workspace.

Notes on derive tools:
- derive_output_filename:
  - Resume naming mode: with `candidate_name` (or `first_name` + `last_name`), `target_role_name`, and `company_name` (or `company`), returns `{"path":"resumes/Firstname Lastname Resume - Target Role - Company.docx"}`.
  - Cover letter naming mode: set `document_type="cover_letter"` with the same fields to get `{"path":"resumes/Firstname Lastname Cover Letter - Target Role - Company.docx"}`.
  - Derivation mode: if role/company/name are omitted, provide `job_description` and `candidate_resume` or `tailored_text`; tool will infer them when possible.
  - Fallback mode: role + date naming, returns `{"path":"resumes/<role>_<date>.docx"}`.
  - Writes `docx_path:latest` to memory.

Notes on coding agent:
- coding_agent_generate: calls the coder service to generate code files from a goal and writes them to the workspace.

Notes on resume tailoring service:
- llm_tailor_resume_text: now calls the resume tailoring service (RESUME_TAILOR_API_URL).
- llm_improve_tailored_resume_text and llm_iterative_improve_tailored_resume_text: now call the resume tailoring service.
- llm_generate_cover_letter_from_resume: generates a structured cover_letter JSON from tailored resume + JD for use with cover_letter_generate_ats_docx.
- llm_generate_coverletter_doc_spec_from_text: generates and strictly validates `coverletter_doc_spec` from tailored resume + JD.
- coverletter_doc_spec_to_document_spec: converts `coverletter_doc_spec` to `document_spec` so `docx_generate_from_spec` can render the cover letter DOCX.

Notes on GitHub tools (GITHUB_TOKEN required):
- github_repo_create: create a repo under user or org.
- github_repo_update: update description/homepage/visibility/default_branch.
- github_repo_list: list repos for user or org.
- github_branch_list: list repo branches.
- github_file_write: create/update a file via GitHub Contents API.
- github_pr_create: open a pull request.
- github_repo_push: push a local workspace folder to a GitHub repo using git.
