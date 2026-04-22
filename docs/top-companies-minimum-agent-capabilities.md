# Minimum Agent Capabilities Top Companies Implement

This document synthesizes the minimum agent-platform capabilities that major
companies now publicly implement, based on official public documentation as of
**March 21, 2026**.

This is not a catalog of every feature every vendor offers. It is a distilled
answer to a narrower question:

> If you strip away product branding and vendor-specific packaging, what is the
> minimum capability set that serious agent platforms now implement?

The companies considered here are:

- OpenAI
- Anthropic
- Microsoft
- Google Cloud
- Amazon Web Services
- GitHub

## 1. Executive Summary

The minimum credible agent-platform capability set is now:

1. structured intent normalization
2. schema-based tool or function calling
3. retrieval over private files or enterprise knowledge
4. live web or external search grounding
5. sandboxed code or command execution
6. file read, edit, and artifact generation
7. external system actions through APIs, connectors, MCP, or OpenAPI
8. conversation state and run memory
9. human clarification, approvals, and policy gates
10. traces, logs, and step-level observability

Not every vendor exposes each capability with the same product name, but this
is the common minimum shape of modern agent systems.

If a platform is missing several of these, it is now closer to "chat with
tools" than to a production-grade agent platform.

## 2. The Capability Stack

The common stack now looks like this:

```text
User Request
  -> Intent normalization
  -> Retrieval / grounding
  -> Tool selection
  -> Action / execution
  -> File or artifact output
  -> Memory / run state
  -> Approval / policy / trace
```

The capabilities are not all equal.

Some are direct user-facing capabilities:

- search the web
- search files
- run code
- edit files
- call external systems

Some are platform capabilities:

- normalize intent
- preserve state
- request clarification
- enforce approvals
- capture traces

Top companies now implement both layers.

## 3. The Minimum Capability List

## 3.1 Structured Intent Normalization

Before planning or tool use, the platform needs a first-pass layer that turns a
raw request into a structured task, route, or intent artifact.

This usually includes:

- intent or task type
- extracted entities
- missing inputs
- candidate tools or routes
- clarification needed

This shows up publicly as:

- structured outputs and function calling in OpenAI
- tool-schema-guided routing in Anthropic
- function-calling-first planning in Microsoft
- response-schema and function-declaration patterns in Google Cloud
- routing workflows in AWS
- context-aware task interpretation in GitHub Copilot agent mode

Why it is minimum:

- without this, planning quality is unstable
- downstream validation becomes brittle
- clarification is too late

## 3.2 Schema-Based Tool Or Function Calling

All serious platforms expose actions to the model through bounded,
machine-readable contracts.

This takes the form of:

- function calling
- tool calling
- plugins
- action groups
- extensions
- MCP tools

What matters is not the label. What matters is:

- stable ids
- descriptions
- input schema
- bounded action space

Why it is minimum:

- the model needs a typed action surface
- application code must remain the executor
- safety and validation depend on explicit contracts

## 3.3 Retrieval Over Private Files Or Enterprise Knowledge

Top companies now treat private-data grounding as baseline, not advanced.

This usually appears as:

- file search
- vector store retrieval
- enterprise search indexes
- knowledge bases
- repo search

Why it is minimum:

- users expect the agent to work over their own documents and data
- enterprise use cases fail without grounded retrieval
- many tasks cannot be solved from model priors alone

## 3.4 Live Web Or External Search Grounding

Top companies also implement some form of current-information grounding.

This may appear as:

- web search
- Bing grounding
- website search extensions
- external knowledge connectors

Why it is minimum:

- agents are expected to answer current-information questions
- planning often depends on recent information
- sourcing and citations are now expected in many UX surfaces

This is distinct from private retrieval. A credible platform now needs both:

- private grounding
- live external grounding

## 3.5 Sandboxed Code Or Command Execution

A modern agent platform is increasingly expected to do more than call APIs. It
is expected to compute, transform, debug, and inspect with an execution
environment.

This appears as:

- code interpreter
- code execution
- bash or shell execution
- build / test task execution

Why it is minimum:

- computation and transformation are common agent tasks
- file manipulation and debugging depend on execution
- coding and analyst workflows increasingly require it

The important implementation detail is sandboxing. The model suggests the
action, but execution happens in a constrained environment.

## 3.6 File Read, Edit, And Artifact Generation

Top companies increasingly expose file-level operations rather than limiting
agents to text replies.

This includes:

- reading files
- editing files
- writing files
- generating artifacts such as reports, images, charts, or code outputs

Why it is minimum:

- many tasks end in an artifact, not a chat answer
- coding agents are expected to edit files directly
- analysis agents often need to generate files or visual outputs

In some platforms this is a separate text-editor or file tool. In others it is
bundled into code execution or the agent host environment. Either way, the
capability is now part of the minimum set.

## 3.7 External System Actions Through APIs, Connectors, MCP, Or OpenAPI

Agents need a way to do work outside their own sandbox.

The industry now converges on a few forms:

- OpenAPI-backed actions
- first-party connectors
- custom functions
- MCP servers
- action groups
- extensions

Why it is minimum:

- enterprise value comes from interacting with real systems
- private data and workflows live outside the model
- API action surfaces are the bridge from reasoning to execution

The exact transport differs by vendor, but the capability class is the same.

## 3.8 Conversation State And Run Memory

A platform needs state, even if it does not market that feature as "memory."

This can include:

- thread history
- uploaded files
- run state
- temporary execution context
- durable conversation storage
- longer-lived memory

Why it is minimum:

- multi-step tasks require continuity
- tool loops need state across turns or runs
- debugging and replay depend on durable history

Some vendors emphasize memory directly. Others emphasize threads, containers,
projects, or conversation state. The underlying need is the same.

## 3.9 Human Clarification, Approvals, And Policy Gates

Modern agent platforms do not assume the model can always proceed safely.

Common control points include:

- ask the user for missing inputs
- require approval before external actions
- restrict tool access
- apply allowlists / policy controls
- enforce enterprise governance

Why it is minimum:

- ambiguity is unavoidable
- high-impact tools need human control
- enterprise deployment requires governance

Without this layer, the platform is difficult to trust outside low-risk demos.

## 3.10 Traces, Logs, And Step-Level Observability

Top companies now treat observability as a core platform feature.

This usually includes:

- tool-call traces
- run logs
- orchestration traces
- step histories
- metrics
- debugger views

Why it is minimum:

- agent failures are otherwise impossible to diagnose
- prompt-only debugging does not scale
- production operations require replayable lineage

This is no longer optional infrastructure. It is part of the minimum product.

## 4. Company Signals Behind The Minimum Set

## 4.1 OpenAI

Public OpenAI materials show a platform centered on:

- structured outputs and function calling
- web search
- file search
- code interpreter
- connectors and remote MCP servers
- computer use
- conversation state and background execution

This strongly supports the minimum categories of:

- normalization
- tool calling
- private retrieval
- external search
- sandboxed execution
- external connectors
- state and approvals

## 4.2 Anthropic

Public Anthropic materials show:

- schema-defined tools
- web search
- code execution
- bash
- text editor
- MCP connector support
- explicit tool-result sequencing

This strongly supports the minimum categories of:

- bounded tool contracts
- live search
- sandboxed execution
- file editing
- MCP-connected external actions
- strong application-level control

## 4.3 Microsoft

Public Microsoft Foundry and Semantic Kernel materials show:

- function-calling-first orchestration
- Azure AI Search and file-search style grounding
- Bing and other knowledge tools
- Code Interpreter
- Browser Automation
- OpenAPI, MCP, Functions, and Logic Apps integration
- tracing and project/thread storage

This strongly supports the minimum categories of:

- bounded route and tool surfaces
- private retrieval
- external grounding
- sandboxed execution
- external actions
- state and observability

## 4.4 Google Cloud

Public Vertex AI materials show:

- controlled generation with response schemas
- function calling
- custom extensions via OpenAPI
- Code Interpreter extension
- Vertex AI Search extension

This strongly supports the minimum categories of:

- schema-constrained normalization
- function / tool calling
- private retrieval
- sandboxed execution
- external actions through structured API wrappers

## 4.5 Amazon Web Services

Public Bedrock materials show:

- routing and orchestration patterns
- action groups
- knowledge bases
- built-in user input handling
- code interpretation
- orchestration traces

This strongly supports the minimum categories of:

- front-door routing
- external actions
- private grounding
- clarification
- sandboxed execution
- traceability

## 4.6 GitHub

Public GitHub Copilot materials show:

- agent mode with codebase search
- file edits
- terminal and build task execution
- MCP servers
- GitHub and Playwright integrations
- toolset controls and policy

This strongly supports the minimum categories of:

- context-aware normalization
- private retrieval over repo context
- sandboxed or permissioned execution
- file editing
- external actions
- approval and policy controls

## 5. What Is Actually Minimum Versus Merely Common

To avoid overstating convergence, it helps to separate three buckets.

### 5.1 True minimum

These are the capabilities that now show up repeatedly enough to treat as
minimum platform requirements:

1. intent normalization
2. schema-based tool calling
3. private retrieval
4. external or live grounding
5. sandboxed execution
6. file operations and artifact generation
7. external action integrations
8. state or memory
9. approvals and clarification
10. observability

### 5.2 Very common, but not universal

These are common enough to matter, but not yet universal:

- browser automation
- full computer use
- multi-agent orchestration
- long-term personalized memory
- built-in deep research products

### 5.3 Differentiators, not minimum

These are meaningful product differentiators, but not part of the base minimum
set:

- visual workflow builders
- prebuilt vertical research providers
- marketplace-style public tool catalogs
- hosted fine-tuning loops for planners
- fully autonomous background cloud coding environments

## 6. Recommended Minimum Capability Set For This Repo

If this repo wants to match the minimum capability shape top companies now
implement, the target set should be:

1. strict intent normalization into a typed artifact
2. bounded capability selection through schemas
3. retrieval over uploaded files, workspace state, and knowledge stores
4. live external search with citations
5. sandboxed execution for code, transforms, and validation
6. file read, write, edit, and artifact output
7. external action integration through MCP and API-backed capabilities
8. durable run state and optional memory
9. explicit clarification and human approval gates
10. durable step, invocation, and run traces

This is the smallest set that still looks like a competitive modern agent
platform rather than a chat wrapper.

## 7. Anti-Patterns To Avoid

Avoid:

- treating tool calling alone as a full agent platform
- supporting private retrieval without live external grounding
- supporting actions without clarification or approval gates
- supporting execution without durable traces
- exposing tools without stable schemas and descriptions
- assuming chat history alone is sufficient state management

## 8. Sources

Official sources used for this document:

- OpenAI, "New tools and features in the Responses API" (May 21, 2025):
  https://openai.com/index/new-tools-and-features-in-the-responses-api/
- OpenAI, "Web search":
  https://platform.openai.com/docs/guides/tools-web-search
- OpenAI, "File search":
  https://platform.openai.com/docs/guides/tools-file-search/
- OpenAI, "Code Interpreter":
  https://platform.openai.com/docs/guides/tools-code-interpreter/
- OpenAI, "Connectors and MCP servers":
  https://platform.openai.com/docs/guides/tools-connectors-mcp/
- OpenAI, "Computer use":
  https://platform.openai.com/docs/guides/tools-computer-use
- Anthropic, "How to implement tool use":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Anthropic, "Web search tool":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool
- Anthropic, "Code execution tool":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool
- Anthropic, "Bash tool":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/bash-tool
- Anthropic, "Text editor tool":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool
- Anthropic, "MCP connector":
  https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
- Microsoft Learn, "Function calling with chat completion":
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/
- Microsoft Learn, "What are tools in Azure AI Foundry Agent Service":
  https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/overview
- Microsoft Learn, "What is Foundry Agent Service?":
  https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview
- Google Cloud, "Extensions overview":
  https://cloud.google.com/vertex-ai/generative-ai/docs/extensions/overview
- Google Cloud, "Create and run extensions":
  https://cloud.google.com/vertex-ai/generative-ai/docs/extensions/create-extension
- Google Cloud, "Code Interpreter extension":
  https://cloud.google.com/vertex-ai/generative-ai/docs/extensions/code-interpreter
- Google Cloud, "Vertex AI Search extension":
  https://cloud.google.com/vertex-ai/generative-ai/docs/extensions/vertex-ai-search
- AWS Prescriptive Guidance, "Workflow for routing":
  https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html
- Amazon Bedrock, "CreateAgentActionGroup":
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_CreateAgentActionGroup.html
- Amazon Bedrock, "Enable code interpretation in Amazon Bedrock":
  https://docs.aws.amazon.com/bedrock/latest/userguide/agents-enable-code-interpretation.html
- Amazon Bedrock, "Create and configure agent manually":
  https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html
- GitHub Blog, "Vibe coding with GitHub Copilot: Agent mode and MCP support rolling out to all VS Code users" (April 4, 2025; updated June 17, 2025):
  https://github.blog/news-insights/product-news/github-copilot-agent-mode-activated/
- GitHub Docs, "Model Context Protocol (MCP) and GitHub Copilot coding agent":
  https://docs.github.com/en/enterprise-cloud%40latest/copilot/concepts/coding-agent/mcp-and-coding-agent
- GitHub Docs, "Using the GitHub MCP Server":
  https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp/use-the-github-mcp-server
- GitHub Docs, "Configuring toolsets for the GitHub MCP Server":
  https://docs.github.com/copilot/how-tos/provide-context/use-mcp/configure-toolsets
- GitHub Changelog, "GitHub Copilot updates in Visual Studio Code February Release" (March 6, 2025):
  https://github.blog/changelog/2025-03-06-github-copilot-updates-in-visual-studio-code-february-release-v0-25-including-improvements-to-agent-mode-and-next-exit-suggestions-ga-of-custom-instructions-and-more/
