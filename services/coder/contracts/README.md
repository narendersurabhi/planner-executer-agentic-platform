# Coder Contracts

Versioned request/response JSON Schemas for the coder service.

- Current version: `v1`
- HTTP endpoint: `/generate`
- MCP endpoint:
  - `/mcp/rpc` (streamable HTTP transport)

Tool names for MCP calls:
- `generate_code`

Compatibility policy:
- Additive optional fields are allowed within a major version.
- Breaking changes require a new major folder (`v2/`).
