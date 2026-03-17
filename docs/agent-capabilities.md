# AI Agents as Capabilities

This document explains how to implement an AI agent as a capability in Agentic Workflow Studio.

It is written for engineers extending the platform, not end users.

The short version is:

- expose the agent as a normal typed capability
- keep the agent loop hidden behind that capability boundary
- choose either a native tool adapter or an MCP-backed service adapter
- validate inputs and outputs with schemas
- let governance, planner, Studio, worker, and chat interact with the capability through the same stable contract

The existing in-repo reference implementation is the coding-agent path:

- `codegen.generate`
- `codegen.autonomous`
- `codegen.publish_pr`

Those capabilities are registered in `config/capability_registry.yaml` and implemented through the tool path in `libs/tools/coder_tools.py`.

## 1. What "Agent as a Capability" Means

In this platform, a `capability` is the public execution contract.
An `agent` is an implementation strategy behind that contract.

That distinction matters.

Good boundary:

- capability input: explicit typed request
- capability output: explicit typed result
- agent internals: planning loop, tool use, retries, reasoning, memory, intermediate state

Bad boundary:

- planner or Studio knows the agent's hidden prompt format
- callers pass opaque blobs like `job` or `agent_state`
- the agent leaks implementation-specific steps into the capability contract

The platform already expects capabilities to be stable units with:

- `risk_tier`
- `idempotency`
- `group`
- `subgroup`
- `input_schema_ref`
- `output_schema_ref`
- one or more `adapters`

See:

- `libs/core/capability_registry.py`
- `config/capability_registry.yaml`

## 2. When an Agent Should Be a Capability

Use an agent as a capability when:

- the outside world should treat it as one bounded operation
- the agent has a clear request/response interface
- the internal loop is an implementation detail
- the result should be reusable in planner-led workflows, Studio workflows, and sometimes chat

Examples:

- `research.agent.run`
- `codegen.autonomous`
- `requirements.agent.expand`
- `openapi.agent.refine`
- `evaluation.agent.review`

Do not model it as a single capability when:

- the caller needs explicit control over each intermediate step
- each step has different risk or approval needs
- the internal stages should be visible in Workflow Studio as separate nodes
- the "agent" is really a workflow that should be authored as a DAG

Rule of thumb:

- one capability if the platform should see one unit of work
- a workflow if the platform should orchestrate the steps explicitly

## 3. Architecture Choices

There are two supported implementation models.

### 3.1 Native tool-backed capability

Use this when the agent runs inside the existing worker or API process boundary.

Pattern:

1. implement a tool handler in `libs/tools/...`
2. register the tool in the tool catalog
3. expose the tool through a capability adapter of type `tool`

This is how `codegen.autonomous` works.

Advantages:

- simplest path
- easiest local testing
- no extra network hop
- good for agents built from existing local tool/runtime primitives

Tradeoffs:

- tighter coupling to the main repo
- process/runtime limits are shared with worker or API
- harder to isolate dependencies if the agent has a unique runtime

### 3.2 MCP-backed capability

Use this when the agent should live in its own service or runtime.

Pattern:

1. run the agent as a separate service
2. expose its tool methods over MCP
3. add server config in `config/mcp_servers.yaml`
4. expose the capability through an adapter of type `mcp`

This is how many GitHub capabilities are wired.

Advantages:

- clean isolation
- service-specific dependencies
- easier operational boundaries
- simpler multi-language or external-service agent hosting

Tradeoffs:

- more deployment overhead
- network hop and auth management
- more operational failure modes

## 4. The Capability Contract

Every agent capability should start with a strict contract.

### 4.1 Capability registry entry

Capabilities are declared in `config/capability_registry.yaml`.

Representative shape:

```yaml
- id: research.agent.run
  description: Run a bounded research agent and return a structured report.
  risk_tier: read_only
  idempotency: read
  group: research
  subgroup: agent
  input_schema_ref: research_agent_run_capability_input
  output_schema_ref: research_agent_run_capability_output
  tags:
    - research
    - agent
    - llm
  enabled: true
  adapters:
    - type: tool
      server_id: local_worker
      tool_name: research_agent_run
      arg_map: {}
      enabled: true
```

Key fields:

- `id`
  Stable public identifier. Treat this as an API name, not a prompt label.

- `description`
  Short operator-facing meaning.

- `risk_tier`
  Important for governance and where the capability may run.

- `idempotency`
  Important for retry and operational safety.

- `input_schema_ref`
  The schema used by planner validation and worker runtime input enforcement.

- `output_schema_ref`
  The declared result contract. Use this when the output structure matters downstream.

- `adapters`
  The execution backend. This is what actually invokes the agent.

### 4.2 Input schema

Create a JSON schema in `schemas/`.

Example:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ResearchAgentRunCapabilityInput",
  "type": "object",
  "properties": {
    "goal": {"type": "string", "minLength": 1},
    "domain": {"type": "string"},
    "max_sources": {"type": "integer", "minimum": 1, "maximum": 20},
    "constraints": {"type": "string"}
  },
  "required": ["goal"],
  "additionalProperties": true
}
```

Design advice:

- make required fields explicit
- avoid hidden fallback inputs
- avoid passing a full `job` object
- prefer bounded parameters such as `max_steps`, `max_sources`, `workspace_path`
- make dangerous fields explicit rather than inferred

### 4.3 Output schema

Use an output schema when downstream tasks need predictable structure.

Example:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ResearchAgentRunCapabilityOutput",
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "url": {"type": "string"}
        },
        "required": ["title", "url"]
      }
    },
    "next_actions": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["summary", "sources", "next_actions"]
}
```

This matters when:

- Studio binds downstream inputs from agent output
- planner-generated tasks depend on named fields
- validation should fail early if the agent returns malformed structure

## 5. Native Tool Path

If your agent lives in the local runtime, this is the normal path.

### 5.1 Implement the handler

Add a handler in `libs/tools/`.

Minimal shape:

```python
from __future__ import annotations

from typing import Any

from libs.core.llm_provider import LLMProvider
from libs.framework.tool_runtime import ToolExecutionError


def research_agent_run(
    payload: dict[str, Any],
    provider: LLMProvider,
) -> dict[str, Any]:
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ToolExecutionError("missing_goal")

    max_sources = payload.get("max_sources", 5)
    if not isinstance(max_sources, int):
        raise ToolExecutionError("max_sources_must_be_integer")

    result = run_research_loop(
        goal=goal,
        max_sources=max_sources,
        provider=provider,
    )

    return {
        "summary": result.summary,
        "sources": result.sources,
        "next_actions": result.next_actions,
    }
```

Important:

- raise `ToolExecutionError` for contract or operational failures
- return a normal JSON-serializable dict
- do not return raw chain-of-thought or prompt internals
- keep the payload shape aligned with the schema

### 5.2 Register the tool spec

Tool registration lives in the tool catalog path:

- `libs/tools/llm_tool_groups.py`
- `libs/core/tool_catalog.py`
- `libs/core/tool_registry.py`
- `libs/core/tool_bootstrap.py`

You can follow the coding-agent example in `libs/tools/llm_tool_groups.py`.

Representative registration:

```python
registry.register(
    Tool(
        spec=ToolSpec(
            name="research_agent_run",
            description="Run a bounded research agent",
            usage_guidance=(
                "Provide 'goal' and optional 'domain', 'max_sources', and "
                "'constraints'. Returns a structured research summary."
            ),
            input_schema={...},
            output_schema={...},
            timeout_s=120,
            risk_level=RiskLevel.high,
            tool_intent=ToolIntent.generate,
        ),
        handler=handler_research_agent_run,
    )
)
```

### 5.3 Wire the handler into the default tool registry

If you use the built-in catalog path, update:

- `libs/core/tool_catalog.py`
  Add handler type(s) to `ToolCatalogHandlers`

- `libs/core/tool_registry.py`
  Add concrete `_research_agent_run(...)` wrapper and pass it through `_default_catalog_handlers()`

- `libs/tools/llm_tool_groups.py` or a dedicated registration module
  Register the tool spec

The final registry assembly happens through `tool_bootstrap.build_default_registry(...)`.

## 6. MCP Path

If your agent should live in its own service, use an MCP adapter.

### 6.1 Expose the agent tool from the service

The service should accept a structured argument payload and return a structured result.

Example conceptual tool:

- tool name: `run_research_agent`
- arguments: `{goal, domain, max_sources, constraints}`
- result: `{summary, sources, next_actions}`

### 6.2 Add the MCP server entry

In `config/mcp_servers.yaml`, define the backend server.

Representative shape:

```yaml
mcp_servers:
  research_agent:
    base_url: http://research-agent:8080
    route_paths:
      - /mcp
      - /mcp/rpc
      - /mcp/rpc/mcp
    timeout_s: 60
    max_retries: 1
    retry_sleep_s: 0.25
    bearer_env: RESEARCH_AGENT_TOKEN
    enabled: true
```

### 6.3 Point the capability adapter at the MCP tool

Example:

```yaml
- id: research.agent.run
  description: Run a bounded research agent and return a structured report.
  risk_tier: read_only
  idempotency: read
  group: research
  subgroup: agent
  input_schema_ref: research_agent_run_capability_input
  output_schema_ref: research_agent_run_capability_output
  enabled: true
  adapters:
    - type: mcp
      server_id: research_agent
      tool_name: run_research_agent
      route_paths:
        - /mcp
        - /mcp/rpc
        - /mcp/rpc/mcp
      timeout_s: 60
      max_retries: 1
      retry_sleep_s: 0.25
      arg_map: {}
      enabled: true
```

The gateway path is already implemented in `libs/core/mcp_gateway.py`.

## 7. How Runtime Execution Works

It is important to understand where the capability executes.

### 7.1 Worker runtime

Most nontrivial agent capabilities should run through worker execution.

The worker path:

1. resolves the capability from the registry
2. checks service-level capability allowlists
3. enforces the input contract
4. routes through `mcp_gateway.invoke_capability(...)`
5. either executes a native tool or an MCP adapter
6. records the result as a `ToolCall`

Relevant code:

- `services/worker/app/capability_runtime_adapter.py`
- `services/worker/app/main.py`
- `libs/core/mcp_gateway.py`

### 7.2 Direct chat execution

Direct chat execution is stricter.

The default direct-chat path in `services/api/app/chat_execution_service.py` is only for a small allowlisted set of safe capabilities.

If you want an agent capability to run directly from chat, it must usually be:

- `read_only`
- bounded
- fast enough
- safe to run without durable workflow state

Most agents should not be exposed this way.

If an agent:

- writes files
- opens PRs
- mutates repositories
- performs a long internal loop
- needs retries or artifacts

then it should run as a normal workflow or job capability, not direct chat.

## 8. Governance

Agent capabilities should be governed as capabilities first and tools second.

### 8.1 Capability governance

Configured through:

- `CAPABILITY_MODE`
- `CAPABILITY_GOVERNANCE_ENABLED`
- `CAPABILITY_GOVERNANCE_MODE`
- `ENABLED_CAPABILITIES`
- `DISABLED_CAPABILITIES`
- service-specific allowlists such as `WORKER_ENABLED_CAPABILITIES` and `API_ENABLED_CAPABILITIES`

The enforcement code lives in `libs/core/capability_registry.py`.

### 8.2 Tool governance

If the agent is backed by a native tool, the underlying tool can also be blocked by tool governance.

Configured through:

- `TOOL_GOVERNANCE_ENABLED`
- `TOOL_GOVERNANCE_MODE`
- `TOOL_GOVERNANCE_CONFIG_PATH`

And evaluated through the tool bootstrap and governance path.

### 8.3 Risk tier and idempotency

Set these carefully.

Examples:

- `read_only` + `read`
  For research or summarization agents that do not mutate state

- `bounded_write` + `safe_write`
  For agents that write within a constrained surface and are safe to retry with an idempotency key

- `high_risk` + `unsafe_write`
  For agents that create code changes, open PRs, or perform irreversible actions

These values affect whether the capability should ever be available for direct chat, what governance rules should look like, and how operators should treat retries.

## 9. Planner and Studio Integration

The whole point of the capability abstraction is that planner and Studio should consume the same boundary.

### 9.1 Planner-led jobs

Planner validation reads the capability schema reference and validates the composed payload before execution.

Relevant path:

- `services/planner/app/planner_service.py`

If your input schema is vague or hidden, planner quality will be poor.

### 9.2 Workflow Studio

Studio nodes are built around capability contracts.

That means your agent capability should expose:

- clean required inputs
- stable output fields
- predictable names

Bad Studio experience:

- one huge `config` blob
- output only in a generic `result` field
- hidden required behavior encoded in prompt text

Good Studio experience:

- `goal`
- `workspace_path`
- `constraints`
- `max_steps`
- output fields like `summary`, `written_paths`, `plan_path`, `sources`

### 9.3 Workflow-level inputs and variables

If the agent needs runtime values, map them through workflow inputs or variables rather than baking them into a hidden payload.

Examples:

- workflow input `repo_name` -> capability input `repo`
- workflow variable `analysis_date` -> capability input `reference_date`
- memory binding `user_profile.github_username` -> capability input `owner`

## 10. Memory and Secrets

### 10.1 Memory

If the agent needs reusable user or workflow context:

- prefer explicit input bindings from memory
- do not silently reach into unrelated global state

The capability runtime already supports memory-aware payload resolution.

Good pattern:

- caller binds `memory.read` output or memory input binding into the capability payload

Bad pattern:

- the agent internally guesses which memory record to load with no contract-level input

### 10.2 Secrets

If the agent needs a secret:

- persist a `secret_ref`, not the resolved secret value
- resolve it late in worker execution
- never store the concrete secret in workflow input history

This repo already supports late-bound workflow secret refs.

Good pattern:

- workflow binding kind `secret`
- worker resolves the ref immediately before live tool execution

Bad pattern:

- API resolves the secret and persists it into job context

## 11. Designing the Agent Itself

The internal loop is your choice, but it should be bounded.

Recommended design constraints:

- explicit step or iteration limit
- explicit timeout
- explicit tool budget
- structured intermediate state
- structured final output
- deterministic failure codes where possible

Good inputs:

- `max_steps`
- `max_sources`
- `timeout_s`
- `constraints`
- `workspace_path`

Good outputs:

- `steps_total`
- `steps_completed`
- `written_paths`
- `summary`
- `sources`
- `status`

Avoid:

- unconstrained autonomous loops
- opaque free-form output that downstream nodes cannot consume
- prompt-only contracts with no schema
- hidden writes outside the declared scope

## 12. Error Handling

Use clear machine-readable errors where possible.

Examples:

- `missing_goal`
- `workspace_path_required`
- `plan_steps_missing`
- `agent_backend_unavailable`
- `agent_invalid_output`
- `contract.input_invalid:...`

The runtime already classifies tool errors, so consistent strings help operators and tests.

If the agent returns malformed JSON or an invalid structure:

- fail with a bounded error
- do not silently coerce nonsense into a success payload

## 13. Testing Strategy

You need tests at multiple layers.

### 13.1 Unit test the tool or adapter

For a native tool:

- input validation
- bounded loop behavior
- error propagation
- output shape

Use the existing style in:

- `libs/core/tests/test_coder_tools.py`

### 13.2 Test capability registry loading

Validate:

- capability exists
- schema refs are correct
- allowlist behavior works

Relevant examples:

- `libs/core/tests/test_capability_registry.py`

### 13.3 Test worker capability execution

Validate:

- capability resolves
- adapter routes correctly
- output is recorded
- failure is surfaced cleanly

Relevant examples:

- `services/worker/tests/test_capability_runtime_adapter.py`
- `services/worker/tests/test_execution_service.py`

### 13.4 Test chat direct execution only if allowed

If you intentionally expose the capability to direct chat, add API-side tests for that path.

Relevant examples:

- `services/api/tests/test_chat_execution_service.py`

### 13.5 Test planner validation

If planner is expected to select or validate the capability, add planner-side tests for schema validation and intent/use patterns.

Relevant examples:

- `services/planner/tests/test_tool_usage_validation.py`

## 14. Example Implementation Checklist

Use this as the minimum implementation sequence.

1. Define the public capability name.
2. Create input schema in `schemas/`.
3. Create output schema if downstream structure matters.
4. Implement the native tool or MCP service.
5. Register the tool if using the native path.
6. Add the capability entry to `config/capability_registry.yaml`.
7. Set `risk_tier` and `idempotency` correctly.
8. Add governance config if needed.
9. Add unit tests for the tool or adapter.
10. Add worker capability runtime tests.
11. Add planner or chat tests if those paths are expected.
12. Verify the capability appears correctly in API capability surfaces and Studio.

## 15. Worked Example: Research Agent

Suppose you want `research.agent.run`.

Recommended public contract:

Inputs:

- `goal`
- `domain`
- `max_sources`
- `constraints`

Outputs:

- `summary`
- `sources`
- `next_actions`

Not recommended:

- `job`
- `raw_prompt`
- `scratchpad`
- `messages`
- `agent_state_blob`

Why:

- Studio should be able to configure it cleanly
- planner should validate it cleanly
- worker should enforce it cleanly
- downstream tasks should consume named fields, not parse prose

## 16. Common Mistakes

### 16.1 Using one capability for multiple different jobs

Bad:

- `agent.run` that sometimes researches, sometimes writes code, sometimes creates PRs

Better:

- `research.agent.run`
- `codegen.autonomous`
- `repo.review.run`

### 16.2 Hiding required inputs inside a `job` object

This makes planner, Studio, and validation worse.

Prefer explicit fields.

### 16.3 Returning prose only

If downstream steps need structure, return structure.

### 16.4 Exposing unsafe agents to direct chat

Direct chat should stay conservative.

### 16.5 Letting the agent decide its own write scope

If it writes files or external resources, declare the write scope in the input contract.

## 17. Recommended Repo Files to Touch

Native path usually touches:

- `config/capability_registry.yaml`
- `schemas/<your_schema>.json`
- `libs/tools/<your_agent>.py`
- `libs/tools/llm_tool_groups.py` or another tool registration module
- `libs/core/tool_catalog.py`
- `libs/core/tool_registry.py`
- relevant tests under `libs/core/tests/`, `services/worker/tests/`, `services/api/tests/`, or `services/planner/tests/`

MCP path usually touches:

- `config/capability_registry.yaml`
- `config/mcp_servers.yaml`
- `schemas/<your_schema>.json`
- the external MCP service repo or implementation
- relevant worker and integration tests

## 18. Recommended Design Standard

For this repo, the cleanest standard is:

- one capability per bounded agent operation
- explicit inputs only
- structured outputs only
- typed schemas
- native tool path for simple in-repo agents
- MCP path for isolated or service-backed agents
- worker execution by default
- direct chat only for clearly safe read-only agents

If you follow that standard, the same agent capability will fit:

- planner-led jobs
- Workflow Studio
- workflow triggers and runs
- worker execution
- governance
- observability

without creating a second orchestration model.

## Related Reading

- [tools.md](tools.md)
- [architecture.md](architecture.md)
- [api.md](api.md)
- [user-guide.md](user-guide.md)
