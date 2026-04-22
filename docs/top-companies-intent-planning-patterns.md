# How Top Companies Implement Intent Parsing, Planning, and Tool Calling

This document explains how major platform companies are publicly implementing
intent parsing, planning, and tool calling as of **March 21, 2026**, and what
that means for Agentic Workflow Studio.

It is not a ranking of vendors. It is a synthesis of recurring architecture
patterns visible in official public docs and product/engineering posts from:

- OpenAI
- Anthropic
- Microsoft
- Google Cloud
- Amazon Web Services
- GitHub

The goal is not to copy one vendor literally. The goal is to identify the
patterns that have clearly converged across the industry.

## 1. Executive Summary

Top companies are converging on the same core architecture:

1. normalize the user request into a structured task or intent
2. expose a bounded set of tools or functions with schemas
3. let the model iteratively call those tools
4. execute the tools in application code, not inside the model
5. feed tool results back into the model until completion
6. keep traces, policies, and validation around the whole loop

Where they differ is mostly in:

- how much of the loop the platform hosts for you
- how much control you get over orchestration
- how explicit their planner abstraction is
- how much governance and tracing is built in

The common lesson is clear:

- **schema-first tools**
- **bounded tool catalogs**
- **iterative tool loops**
- **application-owned execution**
- **traceable orchestration**
- **policy and safety gates**

## 2. The Common Industry Pattern

Across the major platforms, the architecture usually looks like this:

```text
User Goal
  -> Intent / task understanding
  -> Candidate tool or capability set
  -> Model chooses next action
  -> Application executes tool
  -> Tool result returned to model
  -> Repeat until done
  -> Final answer or compiled workflow
```

Important characteristics:

- tools are declared with schemas
- the model proposes tool calls, but the application executes them
- the loop is iterative rather than one-shot
- planning is increasingly implemented via native function/tool calling
- observability is treated as part of the product, not an afterthought

This is now the mainstream architecture.

## 3. OpenAI

### 3.1 Public pattern

OpenAI’s public direction is strongly model-native:

- the `Responses API` is positioned as the core primitive for agentic apps
- models can call built-in tools and external tools
- remote MCP servers are first-class
- long-running tasks are pushed into background execution
- traces and workflow-building primitives are becoming product features

OpenAI’s public material shows several important themes:

- built-in tools are part of the same API primitive as generation
- reasoning models can call tools while reasoning
- MCP is treated as a standard way to attach external capabilities
- long-horizon tasks should be handled asynchronously
- agent frameworks should preserve traceability

### 3.2 Architectural implications

OpenAI’s model is:

- model-native tool selection
- unified responses primitive
- hosted and external tool support
- background execution for slow tasks
- traces as a first-class developer need

That pushes teams toward:

- one canonical interaction loop
- fewer bespoke planner formats
- tighter integration of tool calling and reasoning

### 3.3 What matters for this repo

The useful lessons are:

- use one canonical planning/runtime artifact
- support external tool attachment through MCP cleanly
- support async execution for long-running planner or executor work
- keep tool-call traces durable

What not to copy blindly:

- do not collapse your internal planner/compiler discipline just because the
  model can call tools natively
- do not let tool calling replace validation

## 4. Anthropic

### 4.1 Public pattern

Anthropic’s public documentation is explicit about the application-level agent
loop:

- you define tools with names, descriptions, and JSON schemas
- the model emits `tool_use`
- your application executes the tool
- your application returns `tool_result`
- the model continues

Anthropic is unusually explicit about tool quality:

- tool descriptions should be very detailed
- tool choice can be forced or left automatic
- parallel tool use is a normal part of the loop
- message ordering rules around `tool_use` and `tool_result` are strict

### 4.2 Architectural implications

Anthropic’s pattern is:

- client-controlled loop
- schema-rich tool definitions
- model as planner/chooser
- application as executor
- strict sequencing contract

This is a very strong fit for systems that want:

- high execution control
- explicit safety boundaries
- replayable and debuggable loops

### 4.3 What matters for this repo

The useful lessons are:

- planner- and runtime-visible tools need richer descriptions, not just names
- schema quality and parameter descriptions materially affect routing quality
- tool-call/result ordering should be durable and explicit
- forcing tool use should be deliberate, not the default

This aligns well with:

- `capability` descriptions
- durable `invocations`
- structured planner support tools

## 5. Microsoft

### 5.1 Public pattern

Microsoft’s Semantic Kernel material is the clearest statement that the industry
has moved away from prompt-only planners.

Their current position is:

- old prompt planners are deprecated
- function calling is now the primary way to plan and execute
- the platform automates the back-and-forth function loop
- only explicitly exposed functions are serialized to the model

Semantic Kernel also emphasizes:

- automatic planning loops
- plugin/function registration
- schema serialization
- iteration until the model stops calling functions

### 5.2 Architectural implications

Microsoft’s pattern is:

- function calling replaces special planner abstractions in many cases
- plugin registration defines planner-visible action space
- orchestration is a managed loop, not an LLM one-shot
- exposure is explicit: helper methods stay private

### 5.3 What matters for this repo

The useful lessons are:

- do not maintain a separate fragile “planner DSL” if function/tool calling can
  directly drive step synthesis
- expose only planner-safe functions to the planner
- make the loop explicit and bounded

For this repo, that means:

- planner support tools should be curated
- runtime tools should stay private
- `RunSpec` should remain the compiler output

## 6. Google Cloud

### 6.1 Public pattern

Google’s Vertex AI documentation presents function calling as:

- declare tools or functions in schema form
- send them with the prompt
- receive structured function-call output
- provide the API output back to the model

Google also emphasizes:

- OpenAPI-compatible schemas
- function declarations as formal contracts
- ADK / Agent Engine patterns for memory and orchestration

### 6.2 Architectural implications

Google’s pattern is:

- strict schema declaration
- structured function-call output
- application-controlled fulfillment
- optional higher-level agent frameworks on top

This keeps the architecture modular:

- model chooses actions
- platform validates structure
- application owns execution and memory

### 6.3 What matters for this repo

The useful lessons are:

- capability contracts should stay schema-first
- model outputs should be validated as structured artifacts, not free text
- higher-level agent kits are wrappers around the same fundamental loop

For this repo, that supports:

- `RunSpec` compilation from structured planner output
- capability schema lookup during planning
- planner-only tool exposure

## 7. Amazon Bedrock

### 7.1 Public pattern

AWS Bedrock documents a more explicit orchestration layer than some other
vendors.

Publicly documented behavior includes:

- a default orchestration strategy based on `ReAct`
- prompt-template-based orchestration by default
- custom orchestration for more complex workflows
- custom orchestration implemented through Lambda
- orchestration traces that expose model input, output, rationale, and
  observations

### 7.2 Architectural implications

AWS’s pattern is:

- default agent loop for common cases
- explicit orchestration override for advanced cases
- orchestration treated as a configurable control-plane component
- traces exposed as formal runtime artifacts

This is attractive for enterprises because it separates:

- simple default orchestration
- custom orchestration logic
- traceability and auditability

### 7.3 What matters for this repo

The useful lessons are:

- keep a default planner/tool-calling strategy for most runs
- allow custom orchestration stages for special workflows
- treat orchestration traces as durable debugger data

This aligns closely with:

- `goal_intent_graph`
- planner traces
- `run_events`
- `step_attempts`
- `invocations`

## 8. GitHub

### 8.1 Public pattern

GitHub’s public agent-mode and Copilot material shows a more productized
developer workflow version of the same architecture:

- agent mode gets a list of available tools and MCP servers
- the model decides what to do next
- the system iteratively calls tools until the task is complete
- broader context is gathered on demand instead of front-loading everything
- governance and enterprise policies are visible product features

GitHub also publicly emphasizes:

- MCP as the way to attach external context and capabilities
- agent loops that gather repository context on demand
- higher-signal outcomes from targeted context retrieval
- policy controls for enterprise deployment

### 8.2 Architectural implications

GitHub’s pattern is:

- tool-calling with dynamic context expansion
- context retrieval on demand rather than maximal upfront context stuffing
- strong operational emphasis on logs, policies, and validation tools

### 8.3 What matters for this repo

The useful lessons are:

- capability search should be iterative and demand-driven
- planner should fetch more contract/context only when needed
- governance should be visible and enforceable
- debugger traces should explain why more context was fetched

## 9. What These Companies Have In Common

Despite different product shapes, the common implementation pattern is now
clear.

### 9.1 They all use structured tool contracts

Whether the vocabulary is:

- functions
- tools
- plugins
- MCP tools
- action groups

the shared pattern is:

- name
- description
- schema
- model-visible contract

### 9.2 They all keep execution outside the model

The model selects or proposes.

The application:

- executes tools
- validates arguments
- applies policy
- returns results

This is the core safety boundary.

### 9.3 They all use iterative loops

The dominant architecture is no longer:

- prompt once
- get a full plan
- trust it blindly

It is now:

- propose next action
- execute
- observe result
- continue

### 9.4 They all bound the tool space

No serious system gives the model the whole universe blindly.

They all use some combination of:

- explicit registration
- allowlists
- policy gates
- per-session or per-agent tool visibility
- MCP attachment controls

### 9.5 They all invest in traces

This shows up as:

- trace APIs
- orchestration traces
- chat histories with tool calls/results
- session logs
- debugger views

The industry has learned that agent systems are not operable without this.

### 9.6 They all separate defaults from advanced orchestration

There is usually:

- a default loop for normal cases
- a more programmable orchestration layer for advanced workflows

That is the right pattern for this repo too.

## 10. Where They Differ

### 10.1 OpenAI

Leans into:

- model-native tool reasoning
- built-in tools
- MCP integration
- background tasks

### 10.2 Anthropic

Leans into:

- explicit client-side loop discipline
- very strong tool-definition guidance
- strict tool-result sequencing

### 10.3 Microsoft

Leans into:

- function calling as the replacement for older planner abstractions
- plugin registration and automatic loops

### 10.4 Google

Leans into:

- schema formality
- function declarations
- agent frameworks on top of structured tool use

### 10.5 AWS

Leans into:

- explicit orchestration strategies
- default ReAct plus custom orchestration
- traceable orchestration runtime

### 10.6 GitHub

Leans into:

- MCP-connected context expansion
- iterative task completion
- policy-rich developer workflows

## 11. The Right Synthesis For This Repo

For Agentic Workflow Studio, the right architecture is not “pick one vendor”.

It is:

### 11.1 Use OpenAI / Anthropic / Google style schema-first tool contracts

Planner-visible actions should always have:

- stable ids
- detailed descriptions
- JSON-schema-like input contracts
- expected outputs

### 11.2 Use Microsoft’s function-calling-first planning direction

The planner should use tool calling as its primary planning mechanism instead of
relying on large opaque JSON plans with weak repair logic.

### 11.3 Use AWS’s explicit orchestration separation

Keep:

- default orchestration for common runs
- custom orchestration hooks for advanced flows
- durable orchestration traces

### 11.4 Use GitHub’s demand-driven context expansion

Do not dump every capability and every hint into the model at once.

Instead:

- retrieve top-k candidates
- fetch more contract detail only when needed
- preserve the retrieval trace

### 11.5 Keep the repo’s capability boundary

This repo is already opinionated that:

- capabilities are public
- tools are private

That is the correct abstraction.

The planner should reason about:

- `capability_id`
- schemas
- bindings
- `RunSpec`

not about:

- local tool handler names
- worker-only implementation details

## 12. Recommended Architecture Decision

The best architecture for this repo is:

1. use LLM or heuristic intent parsing only to produce normalized intent
   metadata and segment graphs
2. retrieve bounded capability candidates per segment
3. let the planner run a tool-calling loop over planner-only support tools
4. compile the result into canonical `RunSpec`
5. validate intent, capability contracts, and runtime conformance
6. execute only after durable compile succeeds

That is the intersection of what the top companies are doing and what this repo
already needs.

## 13. Anti-Patterns To Avoid

Top companies’ public docs also imply what not to do.

Avoid:

- letting the planner call runtime side-effecting tools directly
- trusting raw LLM plan JSON without normalization and validation
- exposing every internal helper to the planner
- treating intent parsing as runtime execution authority
- skipping traces for planner retrieval and tool-calling decisions
- using one-shot planning where an iterative loop is needed

## 14. Sources

Official sources used for this document:

- OpenAI, "New tools and features in the Responses API" (May 21, 2025):
  https://openai.com/index/new-tools-and-features-in-the-responses-api/
- OpenAI, "Agents SDK":
  https://developers.openai.com/api/docs/guides/agents-sdk
- Anthropic, "How to implement tool use":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Microsoft Learn, "Planning" in Semantic Kernel:
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning
- Microsoft Learn, "Function calling with chat completion":
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/
- Google Cloud, "Introduction to function calling":
  https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
- AWS, "Customize agent orchestration strategy":
  https://docs.aws.amazon.com/bedrock/latest/userguide/orch-strategy.html
- AWS, "Customize your Amazon Bedrock Agent's behavior with custom orchestration":
  https://docs.aws.amazon.com/bedrock/latest/userguide/agents-custom-orchestration.html
- AWS, "OrchestrationTrace":
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_OrchestrationTrace.html
- GitHub Blog, "Vibe coding with GitHub Copilot: Agent mode and MCP support rolling out to all VS Code users" (April 4, 2025; updated June 17, 2025):
  https://github.blog/news-insights/product-news/github-copilot-agent-mode-activated/
- GitHub Changelog, "Copilot code review now runs on an agentic architecture" (March 5, 2026):
  https://github.blog/changelog/2026-03-05-copilot-code-review-now-runs-on-an-agentic-architecture/
