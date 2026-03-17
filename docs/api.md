# API Guide

This document is the canonical API overview for Agentic Workflow Studio.
It explains the major resource groups, common payload shapes, and the current
endpoint surface exposed by the FastAPI control plane.

It is not intended to be a full generated schema reference. The source of truth
for the live API remains:

- [main.py](/Users/narendersurabhi/planner-executer-agentic-platform/services/api/app/main.py)
- shared request and response models in [models.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/models.py)
- chat contracts in [chat_contracts.py](/Users/narendersurabhi/planner-executer-agentic-platform/libs/core/chat_contracts.py)

## 1. Base URL and Conventions

Typical local API URL:

- `http://localhost:18000`

Common conventions:

- request and response bodies are JSON unless the endpoint is a file download or SSE stream
- identifiers such as `job_id`, `task_id`, `session_id`, `definition_id`, and `version_id` are server-generated UUID-like strings
- the API is a control-plane surface; asynchronous execution is coordinated through Redis-backed events and durable database state

## 2. Common State Models

### Job statuses

- `queued`
- `planning`
- `running`
- `succeeded`
- `failed`
- `canceled`

### Task statuses

- `pending`
- `ready`
- `running`
- `blocked`
- `completed`
- `accepted`
- `rework_requested`
- `failed`
- `canceled`

### Workflow trigger types

- `manual`
- `api`
- `webhook`
- `schedule`

## 3. Jobs, Plans, and Tasks

This is the core planner-led execution path.

### Create a job

`POST /jobs`

Minimal request body:

```json
{
  "goal": "Generate an implementation checklist and artifact summary",
  "context_json": {},
  "priority": 1
}
```

Response type:

- `Job`

### List and inspect jobs

- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/details`
- `GET /jobs/{job_id}/plan`
- `GET /jobs/{job_id}/tasks`
- `GET /jobs/{job_id}/task_results`
- `GET /tasks/{task_id}`
- `GET /jobs/{job_id}/debugger`
- `GET /jobs/{job_id}/events/outbox`

Use `details` when you want a combined view of job, plan, and task state.

### Job lifecycle operations

- `POST /jobs/{job_id}/replan`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/resume`
- `POST /jobs/{job_id}/retry`
- `POST /jobs/{job_id}/retry_failed`
- `POST /jobs/{job_id}/tasks/{task_id}/retry`
- `POST /jobs/{job_id}/clear`

### DLQ inspection

- `GET /jobs/{job_id}/tasks/dlq`

This is useful when worker execution has exhausted retries and the task was
written to the dead-letter stream.

## 4. Chat API

The chat API supports conversational turns, clarification, direct safe
capability execution, and job submission when needed.

### Create a chat session

`POST /chat/sessions`

Minimal request body:

```json
{
  "title": "Workspace assistant"
}
```

Response type:

- `ChatSession`

### Get a chat session

- `GET /chat/sessions/{session_id}`

### Send a chat turn

`POST /chat/sessions/{session_id}/messages`

Minimal request body:

```json
{
  "content": "List the workspace files",
  "context_json": {},
  "priority": 0
}
```

Response type:

- `ChatTurnResponse`

Important response behavior:

- chat may return a conversational assistant response only
- chat may return a direct `tool_call` action for safe read-only capabilities
- chat may ask for clarification
- chat may create and attach to a durable job

## 5. Workflow Studio API

This API family supports saving, publishing, triggering, and running
Studio-authored workflows without going through the planner.

### Workflow definitions

- `POST /workflows/definitions`
- `GET /workflows/definitions`
- `GET /workflows/definitions/{definition_id}`
- `PUT /workflows/definitions/{definition_id}`

Minimal create payload:

```json
{
  "title": "Document pipeline",
  "goal": "Generate and render a document",
  "draft": {},
  "context_json": {},
  "user_id": "narendersurabhi"
}
```

### Workflow versions

- `GET /workflows/definitions/{definition_id}/versions`
- `POST /workflows/definitions/{definition_id}/publish`

Publishing creates an immutable workflow version with compiled plan data.

### Workflow triggers

- `POST /workflows/definitions/{definition_id}/triggers`
- `GET /workflows/definitions/{definition_id}/triggers`
- `PUT /workflows/triggers/{trigger_id}`
- `POST /workflows/triggers/{trigger_id}/invoke`

Minimal trigger create payload:

```json
{
  "title": "Manual run",
  "trigger_type": "manual",
  "enabled": true,
  "config": {}
}
```

### Workflow runs

- `GET /workflows/definitions/{definition_id}/runs`
- `POST /workflows/versions/{version_id}/run`

Minimal run payload:

```json
{
  "inputs": {},
  "context_json": {},
  "priority": 0
}
```

Response type:

- `WorkflowRunResult`

This includes:

- workflow definition
- workflow version
- workflow run
- created job
- persisted plan

## 6. Memory API

Memory supports both structured records and semantic fact storage.

### Structured memory

- `POST /memory/write`
- `GET /memory/read`
- `GET /memory/specs`
- `DELETE /memory/delete`

Minimal structured memory write:

```json
{
  "name": "user_profile",
  "scope": "user",
  "user_id": "narendersurabhi",
  "key": "profile",
  "payload": {
    "full_name": "Narender Rao Surabhi"
  },
  "metadata": {
    "source": "manual"
  }
}
```

Example structured read:

```text
GET /memory/read?name=user_profile&scope=user&user_id=narendersurabhi&key=profile
```

### Semantic memory

- `POST /memory/semantic/write`
- `POST /memory/semantic/search`

Minimal semantic search request:

```json
{
  "query": "education and certifications",
  "namespace": "resume_profile",
  "user_id": "narendersurabhi"
}
```

## 7. Capabilities, Intent, and Composer

This group supports discovery, clarification, and Studio/Compose compilation.

### Capability discovery

- `GET /capabilities`
- `POST /capabilities/search`

Use this surface to inspect or search capability definitions available to the
platform.

### Intent services

- `POST /intent/clarify`
- `POST /intent/decompose`

These endpoints expose intent-assessment behavior used by planner and chat
adjacent flows.

### Composer and Studio compilation

- `POST /composer/compile`
- `POST /composer/recommend_capabilities`
- `POST /plans/preflight`
- `POST /plans`

Minimal compile payload:

```json
{
  "draft": {
    "goal": "Draft workflow",
    "nodes": [],
    "edges": []
  }
}
```

Use this family to:

- compile a Studio draft into a plan
- validate workflow-interface bindings
- preflight a candidate plan before execution
- create plans directly when needed

## 8. Downloads and Event Streaming

### Artifact download

- `GET /artifacts/download?path=<relative_path>`

Artifact downloads support:

- shared-filesystem access in filesystem mode
- object-store fallback in S3 mode

### Workspace download

- `GET /workspace/download?path=<relative_path>`

Workspace downloads are local/shared-storage only and do not fall back to S3.

### SSE event stream

- `GET /events/stream`

This endpoint is used by the UI for live updates.

## 9. Example Local Calls

Create a job:

```bash
curl -X POST http://localhost:18000/jobs \
  -H "Content-Type: application/json" \
  -d '{"goal":"Generate an implementation checklist and artifact summary","context_json":{},"priority":1}'
```

Create a chat session:

```bash
curl -X POST http://localhost:18000/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"title":"Workspace assistant"}'
```

Compile a workflow draft:

```bash
curl -X POST http://localhost:18000/composer/compile \
  -H "Content-Type: application/json" \
  -d '{"draft":{"goal":"Draft workflow","nodes":[],"edges":[]}}'
```

Read memory:

```bash
curl "http://localhost:18000/memory/read?name=user_profile&scope=user&user_id=narendersurabhi&key=profile"
```

## 10. Error Handling Notes

Typical error patterns:

- `404`
  - missing job, task, session, workflow definition, version, or trigger
- `400`
  - invalid payload shape
  - invalid semantic-memory request
  - invalid workflow interface or trigger input
- `409`
  - write conflict for structured memory in optimistic-update scenarios
- `422`
  - create-time clarification or validation failure for some job paths

For planner and workflow compilation failures, the API often returns structured
diagnostics instead of a single string error.

## 11. Related Documents

- [README.md](/Users/narendersurabhi/planner-executer-agentic-platform/README.md)
  High-level project overview and quick reference
- [architecture.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/architecture.md)
  Canonical system architecture overview
- [user-guide.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/user-guide.md)
  End-user walkthroughs and playbooks
- [semantic-memory.md](/Users/narendersurabhi/planner-executer-agentic-platform/docs/semantic-memory.md)
  Memory behavior and usage model
