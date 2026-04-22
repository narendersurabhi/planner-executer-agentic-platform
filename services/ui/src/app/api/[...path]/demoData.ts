const NOW = "2026-04-13T03:45:00.000Z";
const USER_ID = "narendersurabhi";

const adaptiveStatus = {
  active_plan_id: "plan-demo-main",
  pending_replan: false,
  pending_replan_reason: null,
  max_replans: 2,
  replans_used: 1,
  replans_remaining: 1,
  can_manual_replan: true,
  replan_block_reason: null,
};

const capabilityItems = [
  {
    id: "llm.text.generate",
    description: "Generate, summarize, or transform text with the configured LLM provider.",
    enabled: true,
    risk_tier: "medium",
    idempotency: "idempotent",
    group: "AI",
    subgroup: "LLM",
    tags: ["llm", "summarize", "draft", "reasoning"],
    input_schema_ref: "schemas/llm_text_generate_input.json",
    input_schema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Instruction for the model." },
        system_prompt: { type: "string", description: "Optional behavior guidance." },
      },
      required: ["prompt"],
    },
    output_schema_ref: "schemas/llm_text_generate_output.json",
    output_schema: {
      type: "object",
      properties: { text: { type: "string" } },
    },
    input_fields: [
      { path: "prompt", type: "string", required: true, description: "Instruction for the model." },
      { path: "system_prompt", type: "string", required: false, description: "Optional behavior guidance." },
    ],
    output_fields: [{ path: "text", type: "string", required: true }],
    required_inputs: ["prompt"],
    adapters: [{ type: "tool", server_id: "core", tool_name: "llm.text.generate" }],
    planner_hints: { best_for: ["summaries", "drafts", "classification"] },
  },
  {
    id: "rag.search",
    description: "Retrieve grounded context from indexed workspace and knowledge documents.",
    enabled: true,
    risk_tier: "low",
    idempotency: "idempotent",
    group: "Knowledge",
    subgroup: "Retrieval",
    tags: ["rag", "knowledge", "retrieval"],
    input_schema_ref: "schemas/rag_search_input.json",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        top_k: { type: "number" },
      },
      required: ["query"],
    },
    output_schema_ref: "schemas/rag_search_output.json",
    output_schema: {
      type: "object",
      properties: { matches: { type: "array" } },
    },
    input_fields: [
      { path: "query", type: "string", required: true },
      { path: "top_k", type: "number", required: false },
    ],
    output_fields: [{ path: "matches", type: "array", required: true }],
    required_inputs: ["query"],
    adapters: [{ type: "tool", server_id: "rag", tool_name: "rag.search" }],
    planner_hints: { best_for: ["knowledge lookup", "grounding"] },
  },
  {
    id: "data.extract.table",
    description: "Extract structured table data from unstructured text or documents.",
    enabled: true,
    risk_tier: "medium",
    idempotency: "idempotent",
    group: "Data",
    subgroup: "Extraction",
    tags: ["extract", "table", "json"],
    input_schema_ref: "schemas/data_extract_table_input.json",
    input_schema: {
      type: "object",
      properties: {
        source_text: { type: "string" },
        columns: { type: "array" },
      },
      required: ["source_text"],
    },
    output_schema_ref: "schemas/data_extract_table_output.json",
    output_schema: {
      type: "object",
      properties: { rows: { type: "array" } },
    },
    input_fields: [
      { path: "source_text", type: "string", required: true },
      { path: "columns", type: "array", required: false },
    ],
    output_fields: [{ path: "rows", type: "array", required: true }],
    required_inputs: ["source_text"],
    adapters: [{ type: "tool", server_id: "core", tool_name: "data.extract.table" }],
    planner_hints: { best_for: ["structured extraction"] },
  },
  {
    id: "document.docx.render",
    description: "Render a DOCX deliverable from markdown or a document specification.",
    enabled: true,
    risk_tier: "low",
    idempotency: "idempotent",
    group: "Document",
    subgroup: "Render",
    tags: ["docx", "document", "render"],
    input_schema_ref: "schemas/document_docx_render_input.json",
    input_schema: {
      type: "object",
      properties: {
        markdown: { type: "string" },
        output_path: { type: "string" },
      },
      required: ["markdown", "output_path"],
    },
    output_schema_ref: "schemas/document_docx_render_output.json",
    output_schema: {
      type: "object",
      properties: { path: { type: "string" } },
    },
    input_fields: [
      { path: "markdown", type: "string", required: true },
      { path: "output_path", type: "string", required: true },
    ],
    output_fields: [{ path: "path", type: "string", required: true }],
    required_inputs: ["markdown", "output_path"],
    adapters: [{ type: "tool", server_id: "render", tool_name: "document.docx.render" }],
    planner_hints: { best_for: ["client-ready documents"] },
  },
  {
    id: "document.pdf.render",
    description: "Render a PDF deliverable from markdown or a document specification.",
    enabled: true,
    risk_tier: "low",
    idempotency: "idempotent",
    group: "Document",
    subgroup: "Render",
    tags: ["pdf", "document", "render"],
    input_schema_ref: "schemas/document_pdf_render_input.json",
    input_schema: {
      type: "object",
      properties: {
        markdown: { type: "string" },
        output_path: { type: "string" },
      },
      required: ["markdown", "output_path"],
    },
    output_schema_ref: "schemas/document_pdf_render_output.json",
    output_schema: {
      type: "object",
      properties: { path: { type: "string" } },
    },
    input_fields: [
      { path: "markdown", type: "string", required: true },
      { path: "output_path", type: "string", required: true },
    ],
    output_fields: [{ path: "path", type: "string", required: true }],
    required_inputs: ["markdown", "output_path"],
    adapters: [{ type: "tool", server_id: "render", tool_name: "document.pdf.render" }],
    planner_hints: { best_for: ["PDF delivery"] },
  },
  {
    id: "notify.slack.message",
    description: "Send a workflow completion or approval notification to a Slack channel.",
    enabled: true,
    risk_tier: "medium",
    idempotency: "non_idempotent",
    group: "Operations",
    subgroup: "Notifications",
    tags: ["notify", "slack", "handoff"],
    input_schema_ref: "schemas/notify_slack_message_input.json",
    input_schema: {
      type: "object",
      properties: {
        channel: { type: "string" },
        message: { type: "string" },
      },
      required: ["channel", "message"],
    },
    output_schema_ref: "schemas/notify_slack_message_output.json",
    output_schema: {
      type: "object",
      properties: { sent: { type: "boolean" } },
    },
    input_fields: [
      { path: "channel", type: "string", required: true },
      { path: "message", type: "string", required: true },
    ],
    output_fields: [{ path: "sent", type: "boolean", required: true }],
    required_inputs: ["channel", "message"],
    adapters: [{ type: "tool", server_id: "ops", tool_name: "notify.slack.message" }],
    planner_hints: { best_for: ["handoffs", "notifications"] },
  },
];

const workflowDraft = {
  summary: "Client onboarding automation",
  goal: "Create a client onboarding packet, retrieve source policies, draft implementation notes, render a PDF, and notify the delivery channel.",
  contextJsonText: JSON.stringify(
    {
      workspace_user_id: USER_ID,
      client: "Acme Health",
      deliverable: "AI workflow implementation brief",
      audience: "product owners and implementation leads",
      output_path: "/shared/artifacts/acme-onboarding-brief.pdf",
    },
    null,
    2
  ),
  nodePositions: {
    "node-clarify": { x: 160, y: 180 },
    "node-retrieve": { x: 460, y: 90 },
    "node-draft": { x: 760, y: 180 },
    "node-render": { x: 1060, y: 180 },
    "node-notify": { x: 1360, y: 180 },
  },
  nodes: [
    {
      id: "node-clarify",
      taskName: "Clarify business request",
      capabilityId: "llm.text.generate",
      outputPath: "requirements",
      inputBindings: {
        prompt: {
          kind: "literal",
          value: "Summarize the buyer requirement, acceptance criteria, and implementation constraints.",
        },
      },
      outputs: [{ id: "out-requirements", name: "Requirements", path: "requirements" }],
      variables: [],
    },
    {
      id: "node-retrieve",
      taskName: "Retrieve delivery context",
      capabilityId: "rag.search",
      outputPath: "knowledge.matches",
      inputBindings: {
        query: { kind: "literal", value: "agentic AI workflow implementation checklist" },
        top_k: { kind: "literal", value: "5" },
      },
      outputs: [{ id: "out-knowledge", name: "Knowledge matches", path: "knowledge.matches" }],
      variables: [],
    },
    {
      id: "node-draft",
      taskName: "Draft implementation brief",
      capabilityId: "llm.text.generate",
      outputPath: "brief.markdown",
      inputBindings: {
        prompt: { kind: "step_output", sourceNodeId: "node-clarify", sourcePath: "requirements" },
        system_prompt: {
          kind: "literal",
          value: "Write a concise product-owner implementation brief with milestones and handoff notes.",
        },
      },
      outputs: [{ id: "out-brief", name: "Brief markdown", path: "brief.markdown" }],
      variables: [],
    },
    {
      id: "node-render",
      taskName: "Render delivery PDF",
      capabilityId: "document.pdf.render",
      outputPath: "pdf.path",
      inputBindings: {
        markdown: { kind: "step_output", sourceNodeId: "node-draft", sourcePath: "brief.markdown" },
        output_path: { kind: "context", path: "output_path" },
      },
      outputs: [{ id: "out-pdf", name: "PDF path", path: "pdf.path" }],
      variables: [],
    },
    {
      id: "node-notify",
      taskName: "Notify delivery channel",
      capabilityId: "notify.slack.message",
      outputPath: "notification",
      inputBindings: {
        channel: { kind: "literal", value: "#ai-delivery" },
        message: { kind: "step_output", sourceNodeId: "node-render", sourcePath: "pdf.path" },
      },
      outputs: [{ id: "out-notification", name: "Notification", path: "notification" }],
      variables: [],
    },
  ],
  edges: [
    { fromNodeId: "node-clarify", toNodeId: "node-retrieve" },
    { fromNodeId: "node-retrieve", toNodeId: "node-draft" },
    { fromNodeId: "node-draft", toNodeId: "node-render" },
    { fromNodeId: "node-render", toNodeId: "node-notify" },
  ],
  workflowInterface: {
    inputs: [
      {
        id: "input-client",
        key: "client",
        label: "Client",
        valueType: "string",
        required: true,
        description: "Customer or team receiving the workflow.",
        defaultValue: "Acme Health",
      },
      {
        id: "input-deliverable",
        key: "deliverable",
        label: "Deliverable",
        valueType: "string",
        required: true,
        description: "Main output to produce.",
        defaultValue: "AI workflow implementation brief",
      },
    ],
    variables: [
      {
        id: "var-channel",
        key: "delivery_channel",
        description: "Notification channel for completed deliverables.",
        binding: { kind: "literal", value: "#ai-delivery" },
      },
    ],
    outputs: [
      {
        id: "workflow-output-pdf",
        key: "delivery_pdf",
        label: "Delivery PDF",
        description: "Rendered PDF sent to the buyer.",
        binding: { kind: "step_output", sourceNodeId: "node-render", sourcePath: "pdf.path" },
      },
    ],
  },
  runtimeSettings: {
    executionMode: "adaptive",
    adaptivePolicy: { maxReplans: 2 },
  },
};

const workflowDefinition = {
  id: "wf-demo-onboarding",
  title: "Client onboarding automation",
  goal: workflowDraft.goal,
  context_json: {
    workspace_user_id: USER_ID,
    client: "Acme Health",
    deliverable: "AI workflow implementation brief",
    output_path: "/shared/artifacts/acme-onboarding-brief.pdf",
  },
  draft: workflowDraft,
  user_id: USER_ID,
  metadata: { source: "demo_data", status: "ready" },
  created_at: "2026-04-11T14:00:00.000Z",
  updated_at: NOW,
};

const workflowVersion = {
  id: "wfv-demo-onboarding-v3",
  definition_id: workflowDefinition.id,
  version_number: 3,
  title: workflowDefinition.title,
  goal: workflowDefinition.goal,
  context_json: workflowDefinition.context_json,
  draft: workflowDraft,
  compiled_plan: {
    planner_version: "demo-planner-v3",
    tasks_summary: "Clarify request -> retrieve context -> draft brief -> render PDF -> notify.",
  },
  run_spec: {
    kind: "agent_run",
    steps: workflowDraft.nodes.map((node) => ({
      step_id: node.id,
      name: node.taskName,
      capability_id: node.capabilityId,
    })),
  },
  user_id: USER_ID,
  metadata: { source: "demo_data" },
  created_at: NOW,
};

const workflowTrigger = {
  id: "trigger-demo-manual",
  definition_id: workflowDefinition.id,
  title: "Manual product-owner demo trigger",
  trigger_type: "manual",
  enabled: true,
  config: { version_mode: "latest_published" },
  user_id: USER_ID,
  metadata: { source: "demo_data" },
  created_at: NOW,
  updated_at: NOW,
};

const workflowRun = {
  id: "wfr-demo-001",
  definition_id: workflowDefinition.id,
  version_id: workflowVersion.id,
  trigger_id: workflowTrigger.id,
  title: "Client onboarding automation - Acme Health",
  goal: workflowDefinition.goal,
  requested_context_json: workflowDefinition.context_json,
  job_id: "job-demo-001",
  plan_id: "plan-demo-main",
  job_status: "running",
  job_error: null,
  latest_task_id: "task-demo-draft",
  latest_task_name: "Draft implementation brief",
  latest_task_error: null,
  user_id: USER_ID,
  planning_mode: "adaptive",
  current_revision_number: 2,
  adaptive_status: adaptiveStatus,
  metadata: { source: "demo_data", execution_mode: "adaptive" },
  created_at: "2026-04-13T03:20:00.000Z",
  updated_at: NOW,
};

const jobs = [
  {
    id: "job-demo-001",
    goal: workflowDefinition.goal,
    status: "running",
    created_at: "2026-04-13T03:20:00.000Z",
    updated_at: NOW,
    priority: 1,
    planning_mode: "adaptive",
    current_revision_number: 2,
    adaptive_status: adaptiveStatus,
    metadata: {
      workflow_definition_id: workflowDefinition.id,
      workflow_run_id: workflowRun.id,
      llm_provider: "gemini",
      llm_model: "gemini-2.5-flash",
      replan_reason: "retrieval evidence improved",
    },
    context_json: workflowDefinition.context_json,
  },
  {
    id: "job-demo-002",
    goal: "Generate a support triage workflow that classifies inbound tickets and drafts responses.",
    status: "succeeded",
    created_at: "2026-04-12T18:00:00.000Z",
    updated_at: "2026-04-12T18:18:00.000Z",
    priority: 0,
    planning_mode: "static",
    current_revision_number: 1,
    adaptive_status: {
      active_plan_id: "plan-demo-support",
      pending_replan: false,
      max_replans: 0,
      replans_used: 0,
      replans_remaining: 0,
      can_manual_replan: true,
      replan_block_reason: null,
    },
    metadata: {
      workflow_definition_id: "wf-demo-support",
      llm_provider: "openai",
      llm_model: "ft:gpt-4.1-mini:demo-support-triage",
    },
    context_json: { workspace_user_id: USER_ID, queue: "support-intake" },
  },
];

const tasks = [
  {
    id: "task-demo-clarify",
    name: "Clarify business request",
    status: "succeeded",
    deps: [],
    description: "Capture the buyer requirement and acceptance criteria.",
    instruction: "Summarize the request, constraints, and handoff criteria.",
    expected_output_schema_ref: "schemas/text_output.json",
    tool_requests: ["llm.text.generate"],
  },
  {
    id: "task-demo-retrieve",
    name: "Retrieve delivery context",
    status: "succeeded",
    deps: ["task-demo-clarify"],
    description: "Pull relevant implementation guidance from the knowledge base.",
    instruction: "Search indexed architecture and delivery documents.",
    expected_output_schema_ref: "schemas/rag_matches.json",
    tool_requests: ["rag.search"],
  },
  {
    id: "task-demo-draft",
    name: "Draft implementation brief",
    status: "running",
    deps: ["task-demo-retrieve"],
    description: "Write a product-owner implementation brief.",
    instruction: "Use retrieved context to draft milestones, assumptions, and acceptance criteria.",
    expected_output_schema_ref: "schemas/text_output.json",
    tool_requests: ["llm.text.generate"],
  },
  {
    id: "task-demo-render",
    name: "Render delivery PDF",
    status: "queued",
    deps: ["task-demo-draft"],
    description: "Render the implementation brief as a PDF deliverable.",
    instruction: "Render markdown into the configured PDF output path.",
    expected_output_schema_ref: "schemas/file_output.json",
    tool_requests: ["document.pdf.render"],
  },
];

const taskResults = {
  "task-demo-clarify": {
    task_id: "task-demo-clarify",
    status: "succeeded",
    outputs: {
      requirements: "Buyer needs a repeatable AI workflow that retrieves context, drafts a brief, renders a PDF, and notifies the delivery team.",
    },
    tool_calls: [
      {
        tool_name: "llm.text.generate",
        input: { prompt: "Summarize the buyer requirement." },
        status: "succeeded",
        output_or_error: { text: "Repeatable AI workflow with PDF handoff." },
        started_at: "2026-04-13T03:21:00.000Z",
        finished_at: "2026-04-13T03:21:08.000Z",
      },
    ],
    error: null,
  },
  "task-demo-retrieve": {
    task_id: "task-demo-retrieve",
    status: "succeeded",
    outputs: {
      matches: [
        { title: "Agentic AI delivery checklist", score: 0.92 },
        { title: "Planner-executor deployment notes", score: 0.87 },
      ],
    },
    tool_calls: [
      {
        tool_name: "rag.search",
        input: { query: "agentic AI workflow implementation checklist", top_k: 5 },
        status: "succeeded",
        output_or_error: { match_count: 2 },
      },
    ],
    error: null,
  },
};

const plan = {
  id: "plan-demo-main",
  job_id: "job-demo-001",
  planner_version: "demo-planner-v3",
  tasks_summary: "Clarify request, retrieve delivery context, draft implementation brief, render PDF, notify delivery channel.",
  dag_edges: [
    ["task-demo-clarify", "task-demo-retrieve"],
    ["task-demo-retrieve", "task-demo-draft"],
    ["task-demo-draft", "task-demo-render"],
  ],
};

const jobDetails = {
  job_id: "job-demo-001",
  job_status: "running",
  job_error: null,
  plan,
  tasks,
  task_results: taskResults,
  planning_mode: "adaptive",
  current_revision_number: 2,
  adaptive_status: adaptiveStatus,
  revision_history: [
    {
      revision_number: 1,
      plan_id: "plan-demo-original",
      trigger_reason: "initial_plan",
      created_at: "2026-04-13T03:20:20.000Z",
      superseded_at: "2026-04-13T03:29:00.000Z",
      active: false,
      task_count: 4,
    },
    {
      revision_number: 2,
      plan_id: "plan-demo-main",
      trigger_reason: "retrieval evidence improved",
      created_at: "2026-04-13T03:29:00.000Z",
      superseded_at: null,
      active: true,
      task_count: 4,
    },
  ],
  last_replan_reason: "retrieval evidence improved",
  recovery_metadata: { strategy: "adaptive_replan", previous_plan_id: "plan-demo-original" },
};

const memorySpecs = [
  {
    name: "user_profile",
    description: "Stable preferences and delivery details for this workspace user.",
    scope: "user",
    ttl_seconds: null,
  },
  {
    name: "delivery_preferences",
    description: "Reusable output, style, and handoff preferences.",
    scope: "user",
    ttl_seconds: null,
  },
  {
    name: "project_context",
    description: "Project-specific facts that should carry across workflow runs.",
    scope: "user",
    ttl_seconds: null,
  },
];

const userMemoryEntries = [
  {
    id: "memory-demo-profile",
    name: "user_profile",
    scope: "user",
    key: "product_owner_demo",
    user_id: USER_ID,
    payload: {
      audience: "product owners and implementation leads",
      preferred_terms: ["workflow builder", "knowledge base", "run monitor"],
      avoid_terms: ["DAG", "operator chat"],
    },
    metadata: { source: "demo_seed" },
    created_at: "2026-04-12T20:00:00.000Z",
    updated_at: NOW,
    expires_at: null,
  },
  {
    id: "memory-demo-delivery",
    name: "delivery_preferences",
    scope: "user",
    key: "handoff",
    user_id: USER_ID,
    payload: {
      format: "PDF brief plus implementation checklist",
      tone: "product-owner friendly",
      review_required: true,
    },
    metadata: { source: "demo_seed" },
    created_at: "2026-04-12T20:05:00.000Z",
    updated_at: NOW,
    expires_at: null,
  },
];

const jobMemoryEntries = {
  job_context: [
    {
      id: "memory-demo-job-context",
      name: "job_context",
      scope: "job",
      key: "workflow_goal",
      job_id: "job-demo-001",
      payload: {
        client: "Acme Health",
        objective: "Create onboarding implementation brief",
        source_tool: "workflow_studio",
      },
      metadata: { source: "demo_seed" },
      created_at: NOW,
      updated_at: NOW,
    },
  ],
  task_outputs: [
    {
      id: "memory-demo-task-output",
      name: "task_outputs",
      scope: "job",
      key: "task-demo-retrieve",
      job_id: "job-demo-001",
      payload: {
        source_tool: "rag.search",
        summary: "Retrieved implementation checklist and deployment notes.",
      },
      metadata: { task_id: "task-demo-retrieve" },
      created_at: NOW,
      updated_at: NOW,
    },
  ],
};

const ragDocuments = [
  {
    document_id: "doc-agentic-delivery-checklist",
    source_uri: "docs/agentic-delivery-checklist.md",
    namespace: "docs",
    tenant_id: null,
    user_id: USER_ID,
    workspace_id: "demo-workspace",
    chunk_count: 5,
    chunking_strategy: "markdown_headings",
    content_type: "text/markdown",
    filename: "agentic-delivery-checklist.md",
    path: "docs/agentic-delivery-checklist.md",
    repo: "planner-executer-agentic-platform",
    indexed_at: NOW,
    metadata: { audience: "product owners", status: "demo" },
  },
  {
    document_id: "doc-planner-executor-architecture",
    source_uri: "docs/attention-routing-architecture.md",
    namespace: "docs",
    tenant_id: null,
    user_id: USER_ID,
    workspace_id: "demo-workspace",
    chunk_count: 8,
    chunking_strategy: "markdown_headings",
    content_type: "text/markdown",
    filename: "attention-routing-architecture.md",
    path: "docs/attention-routing-architecture.md",
    repo: "planner-executer-agentic-platform",
    indexed_at: "2026-04-12T21:00:00.000Z",
    metadata: { topic: "architecture", status: "demo" },
  },
  {
    document_id: "doc-upwork-offer-playbook",
    source_uri: "docs/upwork-agentic-ai-offer.md",
    namespace: "docs",
    tenant_id: null,
    user_id: USER_ID,
    workspace_id: "demo-workspace",
    chunk_count: 4,
    chunking_strategy: "markdown_headings",
    content_type: "text/markdown",
    filename: "upwork-agentic-ai-offer.md",
    path: "docs/upwork-agentic-ai-offer.md",
    repo: "planner-executer-agentic-platform",
    indexed_at: "2026-04-13T02:20:00.000Z",
    metadata: { topic: "sales", status: "demo" },
  },
];

const chatSession = {
  id: "chat-demo-001",
  title: "Build an onboarding workflow",
  created_at: "2026-04-13T03:18:00.000Z",
  updated_at: NOW,
  metadata: { source: "demo_data" },
  active_job_id: "job-demo-001",
  messages: [
    {
      id: "msg-demo-user-1",
      session_id: "chat-demo-001",
      role: "user",
      content: "Create an onboarding workflow that drafts a product-owner brief, renders a PDF, and notifies the team.",
      created_at: "2026-04-13T03:18:12.000Z",
      metadata: {},
      action: null,
      job_id: null,
    },
    {
      id: "msg-demo-assistant-1",
      session_id: "chat-demo-001",
      role: "assistant",
      content: "I mapped that into a workflow-backed job with retrieval, drafting, PDF rendering, and a delivery notification step.",
      created_at: "2026-04-13T03:18:20.000Z",
      metadata: { confidence: 0.91 },
      action: {
        type: "submit_job",
        goal: workflowDefinition.goal,
        job_id: "job-demo-001",
        workflow_run_id: "wfr-demo-001",
        context_json: workflowDefinition.context_json,
      },
      job_id: "job-demo-001",
    },
  ],
};

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: { "x-demo-data": "true" },
  });
}

function empty(status = 204): Response {
  return new Response(null, { status, headers: { "x-demo-data": "true" } });
}

function eventStream(data: unknown): Response {
  return new Response(`data: ${JSON.stringify(data)}\n\n`, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-demo-data": "true",
    },
  });
}

async function readBody(request: Request): Promise<Record<string, unknown>> {
  try {
    const text = await request.text();
    if (!text) {
      return {};
    }
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function workflowRunResult() {
  return {
    workflow_definition: workflowDefinition,
    workflow_version: workflowVersion,
    workflow_run: workflowRun,
    job: jobs[0],
    plan: {
      id: "plan-demo-main",
      job_id: "job-demo-001",
      planner_version: "demo-planner-v3",
    },
  };
}

function workflowDefinitionResponse(body?: Record<string, unknown>) {
  return {
    ...workflowDefinition,
    title: typeof body?.title === "string" && body.title ? body.title : workflowDefinition.title,
    goal: typeof body?.goal === "string" && body.goal ? body.goal : workflowDefinition.goal,
    draft:
      body?.draft && typeof body.draft === "object" && !Array.isArray(body.draft)
        ? body.draft
        : workflowDefinition.draft,
    context_json:
      body?.context_json && typeof body.context_json === "object" && !Array.isArray(body.context_json)
        ? body.context_json
        : workflowDefinition.context_json,
    updated_at: new Date().toISOString(),
  };
}

function debuggerResponse(jobId: string) {
  return {
    job_id: jobId,
    job_status: "running",
    plan_id: "plan-demo-main",
    planning_mode: "adaptive",
    current_revision_number: 2,
    adaptive_status: adaptiveStatus,
    revision_history: jobDetails.revision_history,
    last_replan_reason: "retrieval evidence improved",
    recovery_metadata: jobDetails.recovery_metadata,
    generated_at: NOW,
    timeline_events_scanned: 14,
    tasks: tasks.map((task, index) => ({
      task,
      resolved_tool_inputs: {
        source: index === 1 ? "knowledge base" : "workflow context",
        user_id: USER_ID,
      },
      tool_inputs_validation: {},
      tool_inputs_resolved: true,
      context_keys: ["client", "deliverable", "output_path"],
      timeline: [
        {
          stream_id: `stream-${task.id}`,
          type: "task_status",
          occurred_at: NOW,
          job_id: jobId,
          task_id: task.id,
          status: task.status,
          attempts: task.status === "running" ? 1 : 0,
          max_attempts: 3,
          worker_consumer: "demo-worker-1",
          error: null,
        },
      ],
      latest_result: taskResults[task.id as keyof typeof taskResults] || {},
      error: {
        category: "none",
        code: "",
        retryable: false,
        message: "",
        hint: "",
      },
    })),
  };
}

function feedbackSummary() {
  const emptyBucket = { total: 0, positive: 0, negative: 0, neutral: 0, partial: 0 };
  return {
    total: 18,
    sentiment_counts: { total: 18, positive: 13, negative: 1, neutral: 2, partial: 2 },
    target_type_counts: [
      { key: "chat_message", total: 8, positive: 6, negative: 0, neutral: 1, partial: 1 },
      { key: "job_outcome", total: 10, positive: 7, negative: 1, neutral: 1, partial: 1 },
    ],
    negative_reasons: [{ reason_code: "too_generic", count: 1 }],
    workflow_sources: [{ key: "workflow_studio", total: 12, positive: 9, negative: 1, neutral: 1, partial: 1 }],
    llm_models: [{ key: "gemini-2.5-flash", total: 10, positive: 8, negative: 0, neutral: 1, partial: 1 }],
    planner_versions: [{ key: "demo-planner-v3", total: 12, positive: 9, negative: 1, neutral: 1, partial: 1 }],
    job_statuses: [{ key: "succeeded", total: 11, positive: 9, negative: 0, neutral: 1, partial: 1 }],
    assistant_action_types: [{ key: "submit_job", total: 7, positive: 6, negative: 0, neutral: 0, partial: 1 }],
    metrics: { avg_score: 0.88, demo_runs: 6 },
    correlates: {
      job_count: 6,
      replan_count: 1,
      retry_count: 0,
      failed_task_count: 0,
      plan_failure_count: 0,
      clarification_turn_count: 2,
      terminal_statuses: [{ key: "succeeded", ...emptyBucket, total: 5, positive: 4, partial: 1 }],
    },
  };
}

export async function demoResponse(
  request: Request,
  path: string[] | undefined
): Promise<Response | null> {
  const method = request.method.toUpperCase();
  const segments = Array.isArray(path) ? path : [];
  const pathname = segments.join("/");
  const url = new URL(request.url);

  if (method === "GET" && pathname === "events/stream") {
    return eventStream({
      type: "task.updated",
      job_id: "job-demo-001",
      payload: {
        job_id: "job-demo-001",
        task_id: "task-demo-draft",
        status: "running",
        message: "Demo workflow is drafting the implementation brief.",
      },
      occurred_at: NOW,
    });
  }

  if (method === "GET" && pathname === "capabilities") {
    return json({ mode: "demo", items: capabilityItems });
  }
  if (method === "POST" && pathname === "capabilities/search") {
    const body = await readBody(request);
    const query = typeof body.query === "string" ? body.query : "";
    return json({
      mode: "demo",
      query,
      items: capabilityItems.slice(0, Number(body.limit || 6)).map((item, index) => ({
        id: item.id,
        score: 0.96 - index * 0.05,
        reason: `Demo match for ${query || item.group}.`,
        source: "demo_catalog",
        description: item.description,
        group: item.group,
        subgroup: item.subgroup,
        tags: item.tags,
      })),
    });
  }

  if (method === "GET" && pathname === "jobs") {
    return json(jobs);
  }
  if (method === "GET" && segments[0] === "jobs" && segments[2] === "details") {
    return json({ ...jobDetails, job_id: segments[1] || "job-demo-001" });
  }
  if (method === "GET" && segments[0] === "jobs" && segments[2] === "debugger") {
    return json(debuggerResponse(segments[1] || "job-demo-001"));
  }
  if (method === "GET" && segments[0] === "jobs" && segments[2] === "tasks" && segments[3] === "dlq") {
    return json([]);
  }
  if (method === "POST" && pathname === "jobs") {
    return json(jobs[0], 201);
  }
  if (method === "POST" && segments[0] === "jobs") {
    return json(jobs.find((job) => job.id === segments[1]) || jobs[0]);
  }

  if (method === "GET" && segments[0] === "memory" && segments[1] === "specs") {
    return json(memorySpecs);
  }
  if (method === "GET" && segments[0] === "memory" && segments[1] === "read") {
    const name = url.searchParams.get("name") || "";
    if (name === "job_context" || name === "task_outputs") {
      return json(jobMemoryEntries[name]);
    }
    return json(userMemoryEntries.filter((entry) => !name || entry.name === name));
  }
  if (method === "POST" && segments[0] === "memory" && segments[1] === "write") {
    const body = await readBody(request);
    return json({
      id: "memory-demo-saved",
      name: typeof body.name === "string" ? body.name : "user_profile",
      scope: typeof body.scope === "string" ? body.scope : "user",
      key: typeof body.key === "string" ? body.key : "demo",
      user_id: typeof body.user_id === "string" ? body.user_id : USER_ID,
      payload: body.payload && typeof body.payload === "object" ? body.payload : {},
      metadata: body.metadata && typeof body.metadata === "object" ? body.metadata : {},
      created_at: NOW,
      updated_at: new Date().toISOString(),
      expires_at: null,
    });
  }
  if (method === "DELETE" && segments[0] === "memory" && segments[1] === "delete") {
    return json(userMemoryEntries[0]);
  }
  if (method === "POST" && segments[0] === "memory" && segments[1] === "semantic" && segments[2] === "search") {
    return json({
      matches: [
        {
          id: "semantic-demo-1",
          score: 0.91,
          payload: {
            fact: "Product-owner demos should use workflow and business-process language.",
            namespace: "positioning",
            subject: "screen_copy",
          },
        },
        {
          id: "semantic-demo-2",
          score: 0.84,
          payload: {
            fact: "The buyer prefers implementation milestones and handoff notes.",
            namespace: "delivery",
            subject: "upwork_offer",
          },
        },
      ],
    });
  }
  if (method === "POST" && segments[0] === "memory" && segments[1] === "semantic" && segments[2] === "write") {
    return json({ status: "stored", id: "semantic-demo-written" });
  }

  if (method === "GET" && segments[0] === "rag" && segments[1] === "documents" && segments[2] === "chunks") {
    const documentId = url.searchParams.get("document_id") || ragDocuments[0].document_id;
    const document = ragDocuments.find((item) => item.document_id === documentId) || ragDocuments[0];
    return json({
      collection_name: url.searchParams.get("collection_name") || "rag_default",
      document,
      chunks: [
        {
          chunk_id: `${document.document_id}#0`,
          document_id: document.document_id,
          source_uri: document.source_uri,
          text: "Agentic AI workflows need orchestration, state, tool boundaries, retrieval, and operator visibility.",
          chunk_index: 0,
          metadata: { heading: "Implementation checklist" },
        },
        {
          chunk_id: `${document.document_id}#1`,
          document_id: document.document_id,
          source_uri: document.source_uri,
          text: "Product-owner demos should show the workflow lifecycle: request, builder, readiness check, saved workflows, run monitor, knowledge, and context.",
          chunk_index: 1,
          metadata: { heading: "Demo guidance" },
        },
      ],
    });
  }
  if (method === "GET" && segments[0] === "rag" && segments[1] === "documents") {
    return json({
      collection_name: url.searchParams.get("collection_name") || "rag_default",
      truncated: false,
      scanned_point_count: 17,
      documents: ragDocuments,
    });
  }
  if (method === "POST" && segments[0] === "rag" && segments[1] === "index") {
    const body = await readBody(request);
    return json({
      status: "indexed",
      document_id: typeof body.document_id === "string" ? body.document_id : "doc-demo-new",
      chunk_count: 3,
      collection_name: typeof body.collection_name === "string" ? body.collection_name : "rag_default",
    });
  }
  if (method === "PUT" && segments[0] === "rag" && segments[1] === "documents") {
    const documentId = url.searchParams.get("document_id") || "doc-demo-new";
    return json({
      deleted: {
        collection_name: url.searchParams.get("collection_name") || "rag_default",
        document_id: documentId,
        deleted_chunk_count: 3,
      },
      indexed: { document_id: documentId, chunk_count: 3 },
    });
  }
  if (method === "DELETE" && segments[0] === "rag" && segments[1] === "documents") {
    return json({
      collection_name: url.searchParams.get("collection_name") || "rag_default",
      document_id: url.searchParams.get("document_id") || "doc-demo-new",
      deleted_chunk_count: 3,
    });
  }

  if (method === "GET" && pathname === "workflows/definitions") {
    return json([workflowDefinition]);
  }
  if (method === "GET" && segments[0] === "workflows" && segments[1] === "definitions" && segments.length === 3) {
    return json(workflowDefinition);
  }
  if (method === "POST" && pathname === "workflows/definitions") {
    return json(workflowDefinitionResponse(await readBody(request)), 201);
  }
  if (method === "PUT" && segments[0] === "workflows" && segments[1] === "definitions" && segments.length === 3) {
    return json(workflowDefinitionResponse(await readBody(request)));
  }
  if (method === "DELETE" && segments[0] === "workflows" && segments[1] === "definitions" && segments.length === 3) {
    return empty();
  }
  if (method === "GET" && segments[0] === "workflows" && segments[1] === "definitions" && segments[3] === "versions") {
    return json([workflowVersion]);
  }
  if (method === "GET" && segments[0] === "workflows" && segments[1] === "definitions" && segments[3] === "triggers") {
    return json([workflowTrigger]);
  }
  if (method === "POST" && segments[0] === "workflows" && segments[1] === "definitions" && segments[3] === "triggers") {
    return json(workflowTrigger, 201);
  }
  if (method === "GET" && segments[0] === "workflows" && segments[1] === "definitions" && segments[3] === "runs") {
    return json([workflowRun]);
  }
  if (method === "POST" && segments[0] === "workflows" && segments[1] === "definitions" && segments[3] === "publish") {
    return json(workflowVersion, 201);
  }
  if (method === "POST" && segments[0] === "workflows" && segments[1] === "versions" && segments[3] === "run") {
    return json(workflowRunResult(), 201);
  }
  if (method === "POST" && segments[0] === "workflows" && segments[1] === "triggers" && segments[3] === "invoke") {
    return json(workflowRunResult(), 201);
  }

  if (method === "POST" && pathname === "composer/compile") {
    return json({
      valid: true,
      diagnostics: { valid: true, errors: [], warnings: [] },
      plan,
      run_spec: workflowVersion.run_spec,
      preflight_errors: {},
    });
  }
  if (method === "POST" && pathname === "plans/preflight") {
    return json({ valid: true, errors: {}, diagnostics: [], checked_at: NOW });
  }
  if (method === "POST" && pathname === "composer/recommend_capabilities") {
    return json({ items: capabilityItems.slice(0, 4), query: "demo" });
  }
  if (method === "POST" && pathname === "intent/decompose") {
    return json({
      goal: workflowDefinition.goal,
      intents: [
        { id: "intent-1", label: "Retrieve context", confidence: 0.93 },
        { id: "intent-2", label: "Draft deliverable", confidence: 0.89 },
        { id: "intent-3", label: "Render PDF", confidence: 0.86 },
      ],
    });
  }
  if (method === "POST" && pathname === "intent/clarify") {
    return json({ needs_clarification: false, questions: [], normalized_goal: workflowDefinition.goal });
  }

  if (method === "POST" && pathname === "chat/sessions") {
    return json(chatSession, 201);
  }
  if (method === "GET" && segments[0] === "chat" && segments[1] === "sessions" && segments.length === 3) {
    return json(chatSession);
  }
  if (method === "POST" && segments[0] === "chat" && segments[1] === "sessions" && segments[3] === "messages") {
    const body = await readBody(request);
    const content = typeof body.content === "string" ? body.content : "Run the demo workflow.";
    const userMessage = {
      id: "msg-demo-user-live",
      session_id: chatSession.id,
      role: "user",
      content,
      created_at: new Date().toISOString(),
      metadata: {},
      action: null,
      job_id: null,
    };
    const assistantMessage = {
      id: "msg-demo-assistant-live",
      session_id: chatSession.id,
      role: "assistant",
      content: "Demo mode routed this request into the client onboarding automation workflow.",
      created_at: new Date().toISOString(),
      metadata: { demo: true },
      action: { type: "submit_job", goal: workflowDefinition.goal, job_id: "job-demo-001" },
      job_id: "job-demo-001",
    };
    return json({
      session: {
        ...chatSession,
        updated_at: new Date().toISOString(),
        active_job_id: "job-demo-001",
        messages: [...chatSession.messages, userMessage, assistantMessage],
      },
      user_message: userMessage,
      assistant_message: assistantMessage,
      job: jobs[0],
      workflow_run: workflowRun,
    });
  }

  if (method === "GET" && segments[0] === "runs" && segments.length === 2) {
    return json({
      id: segments[1],
      kind: "agent_run",
      title: workflowRun.title,
      goal: workflowRun.goal,
      requested_context_json: workflowRun.requested_context_json,
      status: "running",
      job_id: workflowRun.job_id,
      plan_id: workflowRun.plan_id,
      workflow_run_id: workflowRun.id,
      source_definition_id: workflowDefinition.id,
      source_version_id: workflowVersion.id,
      source_trigger_id: workflowTrigger.id,
      job_status: "running",
      latest_step_id: "task-demo-draft",
      latest_step_name: "Draft implementation brief",
      user_id: USER_ID,
      planning_mode: "adaptive",
      current_revision_number: 2,
      adaptive_status: adaptiveStatus,
      run_spec: workflowVersion.run_spec,
      metadata: { source: "demo_data" },
      created_at: workflowRun.created_at,
      updated_at: workflowRun.updated_at,
    });
  }
  if (method === "GET" && segments[0] === "runs" && segments[2] === "steps") {
    return json(
      tasks.map((task) => ({
        id: task.id,
        run_id: segments[1],
        job_id: "job-demo-001",
        spec_step_id: task.id,
        name: task.name,
        description: task.description,
        instruction: task.instruction,
        status: task.status,
        capability_id: task.tool_requests[0],
        input_bindings: {},
        metadata: {},
        created_at: workflowRun.created_at,
        updated_at: workflowRun.updated_at,
      }))
    );
  }
  if (method === "GET" && segments[0] === "runs" && segments[2] === "debugger") {
    return json({
      run: workflowRun,
      job: jobs[0],
      plan,
      planning_mode: "adaptive",
      current_revision_number: 2,
      adaptive_status: adaptiveStatus,
      revision_history: jobDetails.revision_history,
      last_replan_reason: jobDetails.last_replan_reason,
      generated_at: NOW,
      steps: tasks.map((task) => ({
        step: task,
        latest_result: taskResults[task.id as keyof typeof taskResults] || {},
        execution_requests: [],
        checkpoints: [],
        attempts: [],
        timeline: [],
        error: {},
      })),
      execution_requests: [],
      attempts: [],
      invocations: [],
      events: [],
      checkpoints: [],
    });
  }

  if (method === "GET" && pathname === "feedback/summary") {
    return json(feedbackSummary());
  }
  if (method === "GET" && segments[segments.length - 1] === "feedback") {
    return json({ items: [], summary: { total: 0, positive: 0, negative: 0, neutral: 0, partial: 0 } });
  }
  if (method === "POST" && pathname === "feedback") {
    return json({
      id: "feedback-demo",
      target_type: "job_outcome",
      target_id: "job-demo-001",
      actor_key: "demo",
      sentiment: "positive",
      score: 1,
      reason_codes: [],
      created_at: NOW,
      updated_at: NOW,
    });
  }

  return null;
}
