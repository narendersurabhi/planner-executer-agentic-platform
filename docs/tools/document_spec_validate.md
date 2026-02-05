# document_spec_validate

Validate a DocumentSpec JSON for schema, supported blocks, and placeholder resolution before rendering.

Example payload:
```json
{
  "document_spec": {
    "doc_type": "resume",
    "version": "v1",
    "tokens": {
      "full_name": "Ada Lovelace",
      "summary": "Engineer and writer."
    },
    "blocks": [
      { "type": "heading", "level": 1, "text": "{{full_name}}" },
      { "type": "heading", "level": 2, "text": "SUMMARY" },
      { "type": "paragraph", "text": "{{summary}}" },
      {
        "type": "repeat",
        "items": "{{experience}}",
        "as": "r",
        "template": [
          { "type": "paragraph", "text": "{{r.company}}" },
          { "type": "bullets", "items": "{{r.bullets}}" }
        ]
      }
    ]
  },
  "render_context": {},
  "strict": true
}
```

Strict vs non-strict:
- strict=true: unresolved placeholders for tokens/global keys are errors.
- strict=false: unresolved placeholders become warnings.

Typical planner usage:
1) Call `document_spec_validate` on the generated DocumentSpec.
2) If valid, call `docx_generate_from_spec` to render the .docx.
3) If invalid, regenerate or fix the spec before rendering.
