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

Docx render schema pattern:
- Keep a JSON schema per template (e.g., document.json).
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
  - Fallback mode: role + date naming, returns `{"path":"documents/<role>_<date>.docx"}`.
  - Writes `docx_path:latest` and `docx_path:document:latest` to memory.

Notes on coding agent:
- coding_agent_generate: calls the coder service to generate code files from a goal and writes them to the workspace.


Notes on GitHub tools (GITHUB_TOKEN required):
- github_repo_create: create a repo under user or org.
- github_repo_update: update description/homepage/visibility/default_branch.
- github_repo_list: list repos for user or org.
- github_branch_list: list repo branches.
- github_file_write: create/update a file via GitHub Contents API.
- github_pr_create: open a pull request.
- github_repo_push: push a local workspace folder to a GitHub repo using git.
