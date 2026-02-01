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
- file_read_text
- sleep
- http_fetch (optional)
