# How Top Companies Normalize User Requests into Structured Tasks or Intents

This document explains how major platform companies publicly implement the
"front door" of an agent system: taking an unstructured user request and
normalizing it into a structured task, intent, or routing artifact.

This is a narrower topic than planning or tool calling. It focuses on the
first-pass layer that decides what the user is asking for, what information is
missing, and which downstream workflow should handle the request.

As of **March 21, 2026**, the public implementation pattern across leading
platforms has converged more than it has diverged.

The companies examined here are:

- OpenAI
- Anthropic
- Microsoft
- Google Cloud
- Amazon Web Services
- GitHub

## 1. Executive Summary

Top companies do not treat request normalization as a free-form prompt that
returns arbitrary prose.

They usually implement it as:

1. a first-pass classifier or extractor
2. constrained by a schema, tool contract, or bounded action set
3. optionally grounded with available tools, plugins, or capability metadata
4. followed by clarification if required information is missing
5. then handed to a planner, router, or execution loop

The recurring industry pattern is:

```text
Raw user request
  -> preprocess and attach local context
  -> constrain possible intents / routes / fields
  -> extract structured task object
  -> validate
  -> clarify if confidence is low or inputs are missing
  -> hand off to planner / router / workflow
```

The key lesson is that normalization is not the same thing as planning.
Normalization creates a reliable typed artifact that planning can trust.

## 2. What "Normalization" Means in Practice

Across vendors, the normalization layer usually produces some variation of:

- `intent`
- `task_type`
- `domain`
- `entities`
- `constraints`
- `missing_inputs`
- `candidate_tools` or `candidate_capabilities`
- `clarification_needed`
- `confidence`

It may be called a classifier output, tool call, structured response, routing
decision, or planner pre-step. The shape differs, but the function is the
same: convert ambiguous natural language into a typed handoff artifact.

## 3. The Common Architecture Pattern

The clearest converged pattern looks like this:

```text
User Request
  -> Context Attachment
     - session state
     - workspace / repo / issue metadata
     - allowed tools or capabilities
  -> First-pass LLM normalization
     - classify intent
     - extract slots
     - detect ambiguity
  -> Deterministic validation
     - schema check
     - enum / taxonomy check
     - required-field check
  -> Clarification if needed
  -> Downstream routing
     - planner
     - tool loop
     - specialist workflow
     - direct response
```

Important properties:

- the LLM is not allowed to invent an unlimited task taxonomy
- the output is expected to conform to a schema or tool contract
- normalization is usually bounded by the visible tools, plugins, or routes
- low-confidence cases are escalated into clarification
- the result is logged because misrouting is an operational problem

## 4. OpenAI

### 4.1 Public pattern

OpenAI's public direction is to make structured outputs and tool/function
calling the native control surface for agentic systems.

That matters for normalization because the first-pass interpretation step can be
implemented as:

- a strict structured output
- a function call with JSON-schema parameters
- or a tool-selection step inside the Responses API / Agents stack

Public OpenAI materials emphasize:

- a unified responses primitive for agentic apps
- function tools defined by JSON schema
- strict schema adherence for function calling
- model-native tool reasoning
- built-in and external tools in the same loop

### 4.2 How this affects normalization

OpenAI's public architecture implies that request normalization is best treated
as schema-constrained extraction, not prose summarization.

In practice, teams using this model tend to:

- define an intent schema
- ask the model to return that schema directly
- or expose a normalization function/tool and let the model call it
- validate the output before routing downstream

This pushes normalization toward:

- typed fields over narrative explanation
- explicit missing-input detection
- bounded enums and nested objects
- one canonical structured handoff artifact

### 4.3 Takeaway

OpenAI's public implementation style suggests:

- use strict structured outputs for normalization
- keep the normalization object small and typed
- prefer one canonical request artifact over multiple ad hoc ones
- let later planning happen after validation, not during extraction

## 5. Anthropic

### 5.1 Public pattern

Anthropic's public tool-use documentation is explicit that tools are defined by
name, description, and `input_schema`, and that those definitions are inserted
into a special system prompt.

Anthropic also emphasizes:

- detailed tool descriptions
- valid JSON-schema inputs
- optional input examples
- strict ordering of `tool_use` and `tool_result`
- client-controlled execution loops

### 5.2 How this affects normalization

Anthropic's public approach implies that normalization should usually be
implemented as one of two things:

1. structured output against an explicit target format
2. a dedicated normalization or routing tool with a strict schema

Because Anthropic emphasizes rich tool descriptions and examples, the
normalization step is not just about output structure. It is also about making
the intent taxonomy legible to the model.

This favors:

- richer field descriptions
- examples for ambiguous inputs
- explicit "when to use this route" guidance
- clear separation between "classify" and "execute"

### 5.3 Takeaway

Anthropic's public pattern says that normalization quality depends heavily on:

- precise schema design
- excellent route / tool descriptions
- examples for edge cases
- strict application ownership of execution

## 6. Microsoft

### 6.1 Public pattern

Microsoft's Semantic Kernel documentation is explicit that older prompt-based
planners were deprecated and that function calling is now the primary planning
and execution mechanism.

Microsoft also emphasizes:

- automatic function-calling loops
- plugin registration
- serialization of only explicitly exposed functions
- model-visible function metadata as the planning surface

### 6.2 How this affects normalization

Microsoft's public position implies that intent normalization should not be a
detached, poetic planning prompt. It should be an explicit step that maps user
language into a bounded function or plugin surface.

That means:

- normalization is bounded by the registered functions
- hidden helper functions remain invisible
- the model selects among explicit options rather than inventing a route space

In practical terms, the first-pass intent object is often less about abstract
taxonomy and more about:

- which plugin namespace applies
- which function family is relevant
- what arguments are already available
- what clarification is needed before auto-calling continues

### 6.3 Takeaway

Microsoft's public implementation style suggests:

- expose only planner-safe routes to the model
- let the available function surface shape normalization
- avoid separate prompt-only route DSLs when function calling already gives a
  bounded typed interface

## 7. Google Cloud

### 7.1 Public pattern

Google Cloud's Vertex AI documentation emphasizes two complementary patterns:

- controlled generation with `response_schema`
- function declarations in OpenAPI-compatible schema form

Public docs repeatedly show the model returning structured data that conforms
to a declared schema, or returning a structured function call rather than
free-form text.

### 7.2 How this affects normalization

Google's public implementation style pushes normalization toward:

- response schemas for extraction and classification
- function declarations for route or action suggestion
- enums and required fields to narrow model freedom

This is especially well suited for:

- intent classification
- entity extraction
- route selection
- conversion of unstructured text into workflow-ready JSON

### 7.3 Takeaway

Google's public pattern suggests:

- use response schemas when you want a typed intent object
- use function declarations when normalization is already coupled to route
  selection
- rely on OpenAPI-like formality to reduce ambiguity

## 8. Amazon Web Services

### 8.1 Public pattern

AWS Prescriptive Guidance is unusually explicit about routing and first-pass
classification as a separate workflow.

AWS describes routing as a pattern where:

- a classifier or router agent uses an LLM to interpret the intent or category
  of a query
- the input is then routed to a specialized downstream task, tool, agent, or
  workflow

AWS also describes the input stage as including:

- preprocessing
- prompt templating
- goals identification

### 8.2 How this affects normalization

AWS's public guidance treats normalization as a front-door orchestration
concern, not just a prompt trick.

That leads to a clear architecture:

- first-pass router
- explicit downstream workflow selection
- specialized handlers after classification
- separate orchestration and execution concerns

This is a strong pattern for enterprise systems because it keeps:

- front-door classification
- downstream specialization
- and stateful execution

as distinct responsibilities.

### 8.3 Takeaway

AWS's public pattern suggests:

- make normalization its own workflow stage
- treat routing as an architectural primitive
- specialize downstream flows rather than overloading one giant agent

## 9. GitHub

### 9.1 Public pattern

GitHub's public Copilot agent-mode material shows a more productized version of
the same pattern.

Publicly, GitHub describes agent mode as:

- using issue bodies or prompts as starting context
- searching the workspace for relevant context
- combining the request with available tools or MCP servers
- asking an LLM what to do next
- iterating until the task is complete

### 9.2 How this affects normalization

GitHub's public material implies that normalization is not done in isolation.
It is context-aware and capability-aware.

In practice, that means the normalized task is influenced by:

- the repo or issue context
- the currently attached tools
- the accessible MCP servers
- the search results gathered on demand

This is a more dynamic normalization style than a static intent labeler.

The key idea is:

- normalize against the current environment
- not against an abstract universal taxonomy

### 9.3 Takeaway

GitHub's public pattern suggests:

- attach local context before normalization
- make normalization aware of the currently available capability surface
- log what context and searches influenced the route

## 10. What These Companies Have In Common

Despite different product surfaces, the common implementation pattern is now
clear.

### 10.1 Schema-first normalization

They do not leave first-pass interpretation unconstrained when reliability
matters.

Instead they use:

- JSON schema
- response schema
- function declarations
- tool schemas
- bounded route sets

### 10.2 Bounded route spaces

The model is usually not asked:

- "invent the perfect task ontology"

It is asked something more like:

- "choose or fill one of these known shapes"

### 10.3 Clarification before commitment

When critical fields are missing, the safest pattern is not hallucinated
completion. It is clarification.

The normalization layer therefore often includes:

- `missing_inputs`
- `clarification_needed`
- `confidence`

### 10.4 Capability-aware interpretation

Normalization is increasingly tied to the currently visible capability set:

- functions
- plugins
- tools
- MCP servers
- routes

This prevents the model from normalizing into a task that the system cannot
actually execute.

### 10.5 Logging and traces

Misclassification and misrouting are major operational failure modes.

The normalization layer is therefore increasingly treated as observable
infrastructure, not invisible prompt glue.

## 11. Where They Differ

### 11.1 OpenAI

Leans toward model-native structured outputs and function/tool calling.

### 11.2 Anthropic

Leans toward schema-rich tools, detailed descriptions, and explicit
application-controlled loops.

### 11.3 Microsoft

Leans toward function calling as the replacement for legacy prompt planners.

### 11.4 Google Cloud

Leans toward formal response schemas and OpenAPI-like function declarations.

### 11.5 AWS

Leans toward front-door routing as its own orchestration pattern.

### 11.6 GitHub

Leans toward context-aware normalization driven by live repo context and the
currently attached tool surface.

## 12. The Right Synthesis For This Repo

For this repo, the best synthesis is:

1. attach local request context first
2. run a first-pass LLM normalization step that returns a strict typed object
3. validate the output deterministically
4. ask a clarification question if required inputs are missing
5. retrieve bounded capability candidates for the normalized segments
6. hand the result to the planner

The normalization output should look more like this than a prose paragraph:

```json
{
  "primary_intent": "create_document",
  "segments": [
    {
      "intent": "generate_document_spec",
      "artifact_type": "document_spec",
      "entities": {
        "topic": "quarterly planning memo"
      },
      "constraints": {
        "audience": "executives",
        "tone": "concise"
      },
      "missing_inputs": [],
      "clarification_needed": false
    }
  ],
  "candidate_capabilities": [
    "llm_generate_document_spec"
  ],
  "confidence": 0.88
}
```

Important design decisions for this repo:

- normalization should happen before planner tool calling
- normalization should be capability-aware but not capability-executing
- the planner should receive explicit structured intent, not re-infer it from
  raw prose
- runtime should treat normalization as advisory only after validation has
  passed

## 13. Recommended Architecture Decision

The strongest architecture, based on public top-company patterns, is:

1. use an LLM to normalize the request into a strict schema
2. constrain that schema with enums, typed fields, and missing-input markers
3. include the currently allowed capability catalog in the normalization prompt
4. validate the result deterministically
5. clarify before planning when required data is missing or confidence is low
6. pass the normalized artifact into the planner and record the trace

This gives you:

- higher routing accuracy
- less planner hallucination
- fewer invalid plans
- better debuggability
- cleaner separation between parse, plan, and execute

## 14. Anti-Patterns To Avoid

Avoid:

- letting normalization return free-form prose that later code must parse again
- allowing the model to invent an unbounded taxonomy of intents
- mixing normalization and side-effecting execution in one step
- skipping clarification and silently inventing required fields
- normalizing without knowledge of the available capability set
- hiding normalization decisions from observability and debugger views

## 15. Sources

Official sources used for this document:

- OpenAI, "New tools and features in the Responses API" (May 21, 2025):
  https://openai.com/index/new-tools-and-features-in-the-responses-api/
- OpenAI, "Agents SDK":
  https://developers.openai.com/api/docs/guides/agents-sdk
- OpenAI, "Function calling":
  https://platform.openai.com/docs/guides/function-calling
- Anthropic, "How to implement tool use":
  https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use
- Anthropic, "Increase output consistency":
  https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/increase-consistency
- Microsoft Learn, "What are Planners in Semantic Kernel":
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/planning
- Microsoft Learn, "Function calling with chat completion":
  https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/
- Google Cloud, "Introduction to function calling":
  https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
- Google Cloud, "Controlled generation JSON output with predefined schema":
  https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-controlled-generation-response-schema-2
- AWS Prescriptive Guidance, "Workflow for routing":
  https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html
- AWS Prescriptive Guidance, "Basic reasoning agents":
  https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/basic-reasoning-agents.html
- GitHub Changelog, "VSCode Copilot agent mode in Codespaces" (April 11, 2025):
  https://github.blog/changelog/2025-04-11-vscode-copilot-agent-mode-in-codespaces/
- GitHub Changelog, "GitHub Copilot updates in Visual Studio Code February Release" (March 6, 2025):
  https://github.blog/changelog/2025-03-06-github-copilot-updates-in-visual-studio-code-february-release-v0-25-including-improvements-to-agent-mode-and-next-exit-suggestions-ga-of-custom-instructions-and-more/
- GitHub Blog, "Vibe coding with GitHub Copilot: Agent mode and MCP support rolling out to all VS Code users" (April 4, 2025; updated June 17, 2025):
  https://github.blog/news-insights/product-news/github-copilot-agent-mode-activated/
