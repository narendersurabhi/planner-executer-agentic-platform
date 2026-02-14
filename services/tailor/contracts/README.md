# Tailor Contracts

Versioned request/response JSON Schemas for the resume tailoring service.

- Current version: `v1`
- HTTP endpoints: `/tailor`, `/improve`, `/improve-iterative`
- MCP endpoint:
  - `/mcp/rpc` (streamable HTTP transport)

Tool names for MCP calls:
- `tailor_resume`
- `improve_resume`
- `improve_iterative`

Compatibility policy:
- Additive optional fields are allowed within a major version.
- Breaking changes require a new major folder (`v2/`).
