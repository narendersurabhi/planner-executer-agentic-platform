# Tools

Tools are registered in libs/core/tool_registry.py with explicit schemas, usage guidance, and enforced timeouts.
The http_fetch tool is optional and guarded by TOOL_HTTP_FETCH_ENABLED and TOOL_HTTP_FETCH_ALLOWLIST.
File read/write tools only allow paths under /shared/artifacts.

Built-in tools:

- json_transform
- math_eval
- text_summarize
- file_write_artifact
- file_write_text
- file_write_code
- file_read_text
- list_files
- run_tests
- search_text
- sleep
- http_fetch (optional)
- docx_render (requires output_path)
- docx_generate_from_spec
- document_spec_validate

Docx render schema pattern:
- Keep a JSON schema per template (e.g., resume.json, cover_letter.json).
- Planner should generate JSON that matches the schema before calling docx_render.
- docx_render should use schema_ref/template_id that matches the template, and output_path is required.
- document_spec_validate can be used as a preflight check before rendering.

Notes on file write tools:
- file_write_artifact: quick text output; path optional (defaults to artifact.txt).
- file_write_text: write any text file; path required.
- file_write_code: write code files; path required and must include a code extension.
