# docx_generate_from_spec

Generate an ATS-friendly single-column `.docx` document from a `DocumentSpec` JSON and write it under `/shared/artifacts`.
`document_spec` may be omitted when it is available in memory as `document_spec:latest` (preferred).
`path` may be omitted when it is available in memory as `docx_path:latest` (written by `derive_output_filename`).

## Input

```json
{
  "document_spec": {"theme": {}, "tokens": {}, "blocks": []},
  "path": "resume.docx",
  "render_context": {},
  "strict": true
}
```

- `document_spec` (required): JSON spec with `theme`, `tokens`, and `blocks`.
- `path` (required): relative `.docx` filename.
- `render_context` (optional): merged into `document_spec.tokens` (overrides).
- `strict` (optional, default true): unresolved placeholders raise an error.

## Example

```json
{
  "document_spec": {
    "theme": {
      "fonts": {"body": "Calibri"},
      "font_sizes": {"body": 11},
      "page_margins_in": {"top": 0.5, "bottom": 0.5, "left": 0.7, "right": 0.7}
    },
    "tokens": {
      "name": "Jane Doe",
      "headline": "Data Analyst",
      "experience": [
        {"company": "Acme", "bullets": ["Built dashboards"]}
      ]
    },
    "blocks": [
      {"type": "text", "text": "{{name}}", "style": "name"},
      {"type": "paragraph", "text": "{{headline}}"},
      {"type": "heading", "level": 1, "text": "EXPERIENCE"},
      {
        "type": "repeat",
        "items": "{{experience}}",
        "as": "r",
        "template": [
          {"type": "paragraph", "text": "{{r.company}}", "style": "body_bold"},
          {"type": "bullets", "items": "{{r.bullets}}"}
        ]
      }
    ]
  },
  "path": "resume.docx"
}
```

## Output

```json
{
  "path": "/shared/artifacts/resume.docx",
  "bytes_written": 12345
}
```
