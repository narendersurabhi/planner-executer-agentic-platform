"use client";

import { memo, useEffect, useRef, useState } from "react";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TEMPLATE_STORAGE_KEY = "ape.templates.v1";
const TEMPLATE_ORDER_KEY = "ape.templates.order.v1";
const TEMPLATE_DEFAULTS_KEY = "ape.template.defaults.v1";
const MEMORY_LIMIT_STORAGE_KEY = "ape.memory.limit.v1";
const SIDEBAR_MIN_WIDTH = 260;
const SIDEBAR_MAX_WIDTH = 420;

type Job = {
  id: string;
  goal: string;
  status: string;
  created_at: string;
  priority: number;
  metadata?: Record<string, unknown>;
};

type Plan = {
  id: string;
  job_id: string;
  planner_version: string;
  tasks_summary: string;
  dag_edges: string[][];
};

type Task = {
  id: string;
  name: string;
  status: string;
  deps: string[];
  description: string;
  instruction?: string;
  expected_output_schema_ref?: string;
  tool_requests?: string[];
};

type ToolCall = {
  tool_name: string;
  input: Record<string, unknown>;
  status: string;
  output_or_error: Record<string, unknown>;
  started_at?: string;
  finished_at?: string | null;
  idempotency_key?: string;
  trace_id?: string;
};

type TaskResult = {
  task_id: string;
  status: string;
  outputs?: Record<string, unknown>;
  tool_calls?: ToolCall[];
  error?: string | null;
};

type EventEnvelope = {
  type: string;
  payload: Record<string, unknown>;
  job_id?: string;
  task_id?: string;
  correlation_id?: string;
  occurred_at?: string;
  version?: string;
};

type MemoryEntry = {
  id: string;
  name: string;
  scope: string;
  payload: Record<string, unknown>;
  key?: string | null;
  job_id?: string | null;
  user_id?: string | null;
  project_id?: string | null;
  metadata?: Record<string, unknown>;
  version?: string | null;
  created_at?: string;
  updated_at?: string;
  expires_at?: string | null;
};

type TemplateVariable = {
  id?: string;
  key: string;
  label: string;
  scope: "default" | "per_run";
  placeholder?: string;
  required?: boolean;
};

type Template = {
  id: string;
  name: string;
  description?: string;
  goal: string;
  contextJson: string;
  priority: number;
  builtIn?: boolean;
  variables?: TemplateVariable[];
};

const BUILT_IN_TEMPLATES: Template[] = [
  {
    id: "tpl-resume-tailor-text",
    name: "Resume Tailor (Structured Inputs)",
    description: "Use structured inputs to tailor a resume and return plain text sections.",
    goal:
      "Use llm_tailor_resume_text to tailor my resume for the {{target_role_name}} role and return text sections.",
    contextJson:
      '{\n  "job_description": "{{job_description}}",\n  "candidate_resume": "{{candidate_resume}}",\n  "target_role_name": "{{target_role_name}}",\n  "seniority_level": "{{seniority_level}}"\n}',
    priority: 2,
    builtIn: true,
    variables: [
      {
        key: "job_description",
        label: "Job Description",
        scope: "per_run",
        required: true,
        placeholder: "Paste job description"
      },
      {
        key: "candidate_resume",
        label: "Candidate Resume",
        scope: "default",
        required: true,
        placeholder: "Paste your current resume"
      },
      {
        key: "target_role_name",
        label: "Target Role Name",
        scope: "per_run",
        required: true,
        placeholder: "e.g., Senior Backend Engineer"
      },
      {
        key: "seniority_level",
        label: "Seniority Level",
        scope: "per_run",
        required: true,
        placeholder: "e.g., Senior"
      }
    ]
  },
  {
    id: "tpl-resume-tailor-docspec",
    name: "Resume Tailor -> ResumeDocSpec",
    description: "Tailor resume text, then generate a ResumeDocSpec from that text.",
    goal:
      "Use llm_tailor_resume_text to tailor my resume for the {{target_role_name}} role. Iteratively improve it with llm_iterative_improve_tailored_resume_text until alignment >= {{min_alignment_score}} or {{max_iterations}} iterations. Then use llm_generate_resume_doc_spec_from_text on the improved tailored text to produce resume_doc_spec. Validate with resume_doc_spec_validate. Convert with resume_doc_spec_to_document_spec. Finally render a DOCX with docx_generate_from_spec. You MUST derive a filename (e.g., role + date) and set tool_inputs.path to {{output_dir}}/<derived_filename>.docx.",
    contextJson:
      '{\n  "job_description": "{{job_description}}",\n  "candidate_resume": "{{candidate_resume}}",\n  "target_role_name": "{{target_role_name}}",\n  "seniority_level": "{{seniority_level}}",\n  "output_dir": "{{output_dir}}",\n  "min_alignment_score": "{{min_alignment_score}}",\n  "max_iterations": "{{max_iterations}}"\n}',
    priority: 2,
    builtIn: true,
    variables: [
      {
        key: "job_description",
        label: "Job Description",
        scope: "per_run",
        required: true,
        placeholder: "Paste job description"
      },
      {
        key: "candidate_resume",
        label: "Candidate Resume",
        scope: "default",
        required: true,
        placeholder: "Paste your current resume"
      },
      {
        key: "target_role_name",
        label: "Target Role Name",
        scope: "per_run",
        required: true,
        placeholder: "e.g., Senior Backend Engineer"
      },
      {
        key: "seniority_level",
        label: "Seniority Level",
        scope: "default",
        required: false,
        placeholder: "e.g., Senior"
      },
      {
        key: "output_dir",
        label: "Output Folder",
        scope: "default",
        required: false,
        placeholder: "resumes"
      },
      {
        key: "min_alignment_score",
        label: "Min Alignment Score",
        scope: "default",
        required: false,
        placeholder: "85"
      },
      {
        key: "max_iterations",
        label: "Max Iterations",
        scope: "default",
        required: false,
        placeholder: "2"
      }
    ]
  },
  {
    id: "tpl-product-launch",
    name: "Product Launch Plan",
    goal: "Create a phased launch plan with risks, dependencies, and measurable milestones.",
    contextJson:
      '{\n  "product": "New analytics dashboard",\n  "target_date": "2026-05-01",\n  "audience": ["enterprise admins", "data analysts"],\n  "constraints": {\n    "must_include": ["beta program", "security review", "docs"],\n    "no_external_agencies": true\n  }\n}',
    priority: 1,
    builtIn: true
  },
  {
    id: "tpl-research-brief",
    name: "Research Brief",
    goal: "Summarize recent research, identify gaps, and propose next steps.",
    contextJson:
      '{\n  "topic": "Retrieval augmented generation evaluation",\n  "scope": "2022-2026",\n  "output": {\n    "format": "bullet summary",\n    "include_citations": true\n  }\n}',
    priority: 1,
    builtIn: true
  }
];

const replaceTokens = (value: string, values: Record<string, string>) => {
  let result = value;
  for (const [key, replacement] of Object.entries(values)) {
    const safeKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`{{\\s*${safeKey}\\s*}}`, "g");
    result = result.replace(regex, replacement);
  }
  return result;
};

const replaceTokensForJson = (value: string, values: Record<string, string>) => {
  let result = value;
  for (const [key, replacement] of Object.entries(values)) {
    const safeKey = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`{{\\s*${safeKey}\\s*}}`, "g");
    const escaped = JSON.stringify(replacement).slice(1, -1);
    result = result.replace(regex, escaped);
  }
  return result;
};

const TemplatePreview = memo(function TemplatePreview({
  goal,
  context,
  isValid
}: {
  goal: string;
  context: string;
  isValid: boolean;
}) {
  return (
    <div className="mt-4 rounded-lg border border-white/10 bg-white/10 p-3">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-200">
        Preview
      </div>
      <div className="mt-3 space-y-2">
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-slate-200">Goal</div>
          <div className="mt-1 rounded-md bg-white/90 px-3 py-2 text-xs text-slate-900">
            {goal || "—"}
          </div>
        </div>
        <div>
          <div className="text-[11px] uppercase tracking-[0.2em] text-slate-200">
            Context JSON
          </div>
          {!isValid ? (
            <div className="mt-2 text-xs text-rose-200">
              Preview JSON is invalid. Check your placeholders and commas.
            </div>
          ) : null}
          <pre className="mt-1 max-h-40 overflow-auto rounded-md bg-white/90 px-3 py-2 text-[11px] text-slate-900">
            {context || "—"}
          </pre>
        </div>
      </div>
    </div>
  );
});

type DagNode = {
  id: string;
  name: string;
  status: string;
  x: number;
  y: number;
};

type DagEdge = {
  from: DagNode;
  to: DagNode;
};

type DagLayout = {
  nodes: DagNode[];
  edges: DagEdge[];
  width: number;
  height: number;
};

const statusColors: Record<string, { fill: string; stroke: string }> = {
  pending: { fill: "#e2e8f0", stroke: "#94a3b8" },
  ready: { fill: "#dbeafe", stroke: "#60a5fa" },
  running: { fill: "#fef3c7", stroke: "#f59e0b" },
  blocked: { fill: "#fee2e2", stroke: "#ef4444" },
  completed: { fill: "#dcfce7", stroke: "#22c55e" },
  accepted: { fill: "#bbf7d0", stroke: "#16a34a" },
  failed: { fill: "#fecaca", stroke: "#ef4444" },
  canceled: { fill: "#f1f5f9", stroke: "#94a3b8" }
};

const truncate = (value: string, length: number) =>
  value.length > length ? `${value.slice(0, length - 3)}...` : value;

const buildDagLayout = (tasks: Task[]): DagLayout => {
  const nodeWidth = 180;
  const nodeHeight = 56;
  const columnGap = 70;
  const rowGap = 20;

  const byId = new Map(tasks.map((task) => [task.id, task]));
  const byName = new Map(tasks.map((task) => [task.name, task]));

  const edges: Array<{ from: string; to: string }> = [];
  for (const task of tasks) {
    for (const dep of task.deps || []) {
      const depTask = byId.get(dep) || byName.get(dep);
      if (depTask) {
        edges.push({ from: depTask.id, to: task.id });
      }
    }
  }

  const depthCache = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (taskId: string): number => {
    if (depthCache.has(taskId)) {
      return depthCache.get(taskId) as number;
    }
    if (visiting.has(taskId)) {
      return 0;
    }
    visiting.add(taskId);
    const task = byId.get(taskId);
    if (!task) {
      visiting.delete(taskId);
      return 0;
    }
    const deps = task.deps || [];
    let maxDepth = 0;
    for (const dep of deps) {
      const depTask = byId.get(dep) || byName.get(dep);
      if (!depTask) {
        continue;
      }
      maxDepth = Math.max(maxDepth, depthOf(depTask.id) + 1);
    }
    visiting.delete(taskId);
    depthCache.set(taskId, maxDepth);
    return maxDepth;
  };

  const columns = new Map<number, Task[]>();
  for (const task of tasks) {
    const depth = depthOf(task.id);
    if (!columns.has(depth)) {
      columns.set(depth, []);
    }
    columns.get(depth)?.push(task);
  }

  for (const tasksAtDepth of columns.values()) {
    tasksAtDepth.sort((a, b) => a.name.localeCompare(b.name));
  }

  const nodes: DagNode[] = [];
  const columnIndices = Array.from(columns.keys()).sort((a, b) => a - b);
  for (const depth of columnIndices) {
    const tasksAtDepth = columns.get(depth) || [];
    tasksAtDepth.forEach((task, index) => {
      nodes.push({
        id: task.id,
        name: task.name,
        status: task.status,
        x: depth * (nodeWidth + columnGap),
        y: index * (nodeHeight + rowGap)
      });
    });
  }

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const dagEdges: DagEdge[] = edges
    .map((edge) => {
      const from = nodeById.get(edge.from);
      const to = nodeById.get(edge.to);
      if (!from || !to) {
        return null;
      }
      return { from, to };
    })
    .filter((edge): edge is DagEdge => edge !== null);

  const columnsCount = columnIndices.length || 1;
  const maxRows = Math.max(
    1,
    ...Array.from(columns.values()).map((tasksAtDepth) => tasksAtDepth.length)
  );

  return {
    nodes,
    edges: dagEdges,
    width: columnsCount * nodeWidth + (columnsCount - 1) * columnGap + 20,
    height: maxRows * nodeHeight + (maxRows - 1) * rowGap + 20
  };
};

export default function Home() {
  const [goal, setGoal] = useState("");
  const [contextJson, setContextJson] = useState("{}");
  const [priority, setPriority] = useState(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);
  const [selectedTasks, setSelectedTasks] = useState<Task[]>([]);
  const [taskResults, setTaskResults] = useState<Record<string, TaskResult>>({});
  const selectedJobIdRef = useRef<string | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateName, setTemplateName] = useState("");
  const [templateError, setTemplateError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [templateDefaults, setTemplateDefaults] = useState<Record<string, string>>({});
  const [activeTemplate, setActiveTemplate] = useState<Template | null>(null);
  const [templateInputs, setTemplateInputs] = useState<Record<string, string>>({});
  const [templateInputError, setTemplateInputError] = useState<string | null>(null);
  const [templateMissingKeys, setTemplateMissingKeys] = useState<Set<string>>(new Set());
  const [defaultsTemplateId, setDefaultsTemplateId] = useState<string>("");
  const [showTemplateModal, setShowTemplateModal] = useState(false);
  const [customVariables, setCustomVariables] = useState<TemplateVariable[]>([]);
  const [previewGoal, setPreviewGoal] = useState("");
  const [previewContext, setPreviewContext] = useState("");
  const [previewContextIsValid, setPreviewContextIsValid] = useState(true);
  const [showRawPlaceholders, setShowRawPlaceholders] = useState(true);
  const [isReorderMode, setIsReorderMode] = useState(false);
  const [draggingTemplateId, setDraggingTemplateId] = useState<string | null>(null);
  const [dragOverTemplateId, setDragOverTemplateId] = useState<string | null>(null);
  const [showTaskInputs, setShowTaskInputs] = useState(false);
  const [showRecentEvents, setShowRecentEvents] = useState(false);
  const [showMemory, setShowMemory] = useState(false);
  const [expandedTaskInputs, setExpandedTaskInputs] = useState<Set<string>>(new Set());
  const [expandedRecentEvents, setExpandedRecentEvents] = useState<Set<number>>(new Set());
  const [expandedMemoryGroups, setExpandedMemoryGroups] = useState<Set<string>>(new Set());
  const [expandedMemoryEntries, setExpandedMemoryEntries] = useState<
    Record<string, Set<number>>
  >({});
  const [memoryEntries, setMemoryEntries] = useState<Record<string, MemoryEntry[]>>({});
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [memoryLimitDefault, setMemoryLimitDefault] = useState(10);
  const [memoryLimits, setMemoryLimits] = useState<Record<string, number>>({
    job_context: 10,
    task_outputs: 10
  });
  const [memoryFilters, setMemoryFilters] = useState({ key: "", tool: "" });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const [isResizing, setIsResizing] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);
  const [hasSetInitialSidebar, setHasSetInitialSidebar] = useState(false);

  const selectedJob = selectedJobId
    ? jobs.find((job) => job.id === selectedJobId) || null
    : null;

  useEffect(() => {
    selectedJobIdRef.current = selectedJobId;
  }, [selectedJobId]);

  const loadJobs = async () => {
    const response = await fetch(`${apiUrl}/jobs`);
    const data = await response.json();
    setJobs(data);
  };

  useEffect(() => {
    loadJobs();
    const source = new EventSource(`${apiUrl}/events/stream`);
    source.onmessage = (event) => {
      try {
        const envelope = JSON.parse(event.data) as EventEnvelope;
        setEvents((prev) => [envelope, ...prev].slice(0, 50));
        const activeJobId = selectedJobIdRef.current;
        if (!activeJobId || !envelope?.type) {
          return;
        }
        if (envelope.type === "task.heartbeat") {
          return;
        }
        const payloadJobId =
          (typeof envelope.job_id === "string" && envelope.job_id) ||
          (typeof envelope.payload?.job_id === "string" && envelope.payload.job_id) ||
          (typeof envelope.payload?.id === "string" && envelope.payload.id) ||
          null;
        if (payloadJobId && payloadJobId === activeJobId) {
          loadJobDetails(activeJobId);
        }
      } catch {
        return;
      }
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    setExpandedRecentEvents(new Set());
  }, [events]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const raw = window.localStorage.getItem(TEMPLATE_STORAGE_KEY);
    const orderRaw = window.localStorage.getItem(TEMPLATE_ORDER_KEY);
    let storedOrder: string[] = [];
    if (orderRaw) {
      try {
        const parsedOrder = JSON.parse(orderRaw);
        if (Array.isArray(parsedOrder)) {
          storedOrder = parsedOrder.filter((entry) => typeof entry === "string");
        }
      } catch {
        storedOrder = [];
      }
    }
    if (!raw) {
      if (storedOrder.length > 0) {
        const builtInMap = new Map(
          BUILT_IN_TEMPLATES.map((template) => [template.id, template])
        );
        const orderedBuiltIns: Template[] = [];
        for (const id of storedOrder) {
          const builtIn = builtInMap.get(id);
          if (builtIn) {
            orderedBuiltIns.push(builtIn);
            builtInMap.delete(id);
          }
        }
        const remainingBuiltIns = Array.from(builtInMap.values());
        setTemplates([...orderedBuiltIns, ...remainingBuiltIns]);
      } else {
        setTemplates(BUILT_IN_TEMPLATES);
      }
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        const custom = parsed
          .filter((entry) => entry && entry.id && entry.name)
          .map((entry) => {
            return {
              ...entry,
              priority: typeof entry.priority === "number" ? entry.priority : 0
            };
          });
        const builtInMap = new Map(BUILT_IN_TEMPLATES.map((template) => [template.id, template]));
        const orderedBuiltIns: Template[] = [];
        for (const id of storedOrder) {
          const builtIn = builtInMap.get(id);
          if (builtIn) {
            orderedBuiltIns.push(builtIn);
            builtInMap.delete(id);
          }
        }
        const remainingBuiltIns = Array.from(builtInMap.values());
        const ordered = [...orderedBuiltIns, ...remainingBuiltIns, ...custom];
        setTemplates(ordered);
        return;
      }
    } catch {
      // ignore malformed storage
    }
    setTemplates(BUILT_IN_TEMPLATES);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const updateDesktop = () => {
      const desktop = window.innerWidth >= 1024;
      setIsDesktop(desktop);
      if (!hasSetInitialSidebar) {
        setSidebarOpen(false);
        setHasSetInitialSidebar(true);
        return;
      }
      if (!desktop && sidebarOpen) {
        setSidebarOpen(false);
      }
    };
    updateDesktop();
    window.addEventListener("resize", updateDesktop);
    return () => window.removeEventListener("resize", updateDesktop);
  }, [hasSetInitialSidebar, sidebarOpen]);

  useEffect(() => {
    if (!isResizing) {
      return;
    }
    const handleMouseMove = (event: MouseEvent) => {
      if (!sidebarOpen || !isDesktop) {
        return;
      }
      const nextWidth = Math.min(
        SIDEBAR_MAX_WIDTH,
        Math.max(SIDEBAR_MIN_WIDTH, event.clientX)
      );
      setSidebarWidth(nextWidth);
    };
    const handleMouseUp = () => {
      setIsResizing(false);
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, isDesktop, sidebarOpen]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const raw = window.localStorage.getItem(TEMPLATE_DEFAULTS_KEY);
    if (!raw) {
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        setTemplateDefaults(parsed);
      }
    } catch {
      return;
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const raw = window.localStorage.getItem(MEMORY_LIMIT_STORAGE_KEY);
    if (!raw) {
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      return;
    }
    const clamped = Math.max(10, Math.min(200, parsed));
    setMemoryLimitDefault(clamped);
    setMemoryLimits({ job_context: clamped, task_outputs: clamped });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(MEMORY_LIMIT_STORAGE_KEY, String(memoryLimitDefault));
  }, [memoryLimitDefault]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (templates.length === 0) {
      return;
    }
    const custom = templates.filter((template) => !template.builtIn);
    window.localStorage.setItem(TEMPLATE_STORAGE_KEY, JSON.stringify(custom));
    const order = templates.map((template) => template.id);
    window.localStorage.setItem(TEMPLATE_ORDER_KEY, JSON.stringify(order));
  }, [templates]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(TEMPLATE_DEFAULTS_KEY, JSON.stringify(templateDefaults));
  }, [templateDefaults]);

  const submitJob = async () => {
    setSubmitError(null);
    if (!goal.trim()) {
      setSubmitError("Goal is required.");
      return;
    }
    let parsedContext: Record<string, unknown> = {};
    try {
      parsedContext = JSON.parse(contextJson || "{}");
    } catch (error) {
      setSubmitError("Context JSON must be valid.");
      return;
    }
    try {
      const response = await fetch(`${apiUrl}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal,
          context_json: parsedContext,
          priority
        })
      });
      if (!response.ok) {
        const text = await response.text();
        setSubmitError(
          text ? `Failed to submit job (${response.status}): ${text}` : `Failed to submit job (${response.status}).`
        );
        return;
      }
      setGoal("");
      setContextJson("{}");
      setPriority(0);
      loadJobs();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Network error while submitting job.");
    }
  };

  const applyTemplate = (template: Template, values: Record<string, string>) => {
    setGoal(replaceTokens(template.goal, values));
    setContextJson(replaceTokensForJson(template.contextJson, values));
    setPriority(template.priority);
  };

  const saveTemplate = () => {
    setTemplateError(null);
    const name = templateName.trim();
    if (!name) {
      setTemplateError("Give this template a name.");
      return;
    }
    try {
      JSON.parse(contextJson || "{}");
    } catch {
      setTemplateError("Context JSON must be valid.");
      return;
    }
    if (!previewContextIsValid) {
      setTemplateError("Preview JSON is invalid. Fix placeholders before saving.");
      return;
    }
    for (const variable of customVariables) {
      if (!variable.key.trim() || !variable.label.trim()) {
        setTemplateError("Each variable needs a key and label.");
        return;
      }
    }
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `tpl-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const customTemplates = templates.filter((template) => !template.builtIn);
    setTemplates([
      ...BUILT_IN_TEMPLATES,
      {
        id,
        name,
        goal,
        contextJson,
        priority,
        variables: customVariables.length ? customVariables : undefined
      },
      ...customTemplates
    ]);
    setTemplateName("");
    setCustomVariables([]);
  };

  const deleteTemplate = (templateId: string) => {
    const customTemplates = templates.filter(
      (template) => !template.builtIn && template.id !== templateId
    );
    setTemplates([...BUILT_IN_TEMPLATES, ...customTemplates]);
  };

  const moveTemplate = (templateId: string, direction: "up" | "down") => {
    setTemplates((prev) => {
      const index = prev.findIndex((template) => template.id === templateId);
      if (index === -1) {
        return prev;
      }
      const targetIndex = direction === "up" ? index - 1 : index + 1;
      if (targetIndex < 0 || targetIndex >= prev.length) {
        return prev;
      }
      const next = [...prev];
      const [item] = next.splice(index, 1);
      next.splice(targetIndex, 0, item);
      return next;
    });
  };

  const reorderTemplates = (fromId: string, toId: string) => {
    if (fromId === toId) {
      return;
    }
    setTemplates((prev) => {
      const fromIndex = prev.findIndex((template) => template.id === fromId);
      const toIndex = prev.findIndex((template) => template.id === toId);
      if (fromIndex === -1 || toIndex === -1) {
        return prev;
      }
      const next = [...prev];
      const [item] = next.splice(fromIndex, 1);
      next.splice(toIndex, 0, item);
      return next;
    });
  };

const openTemplateModal = (template: Template) => {
  if (!template.variables || template.variables.length === 0) {
    applyTemplate(template, {});
    return;
  }
  const nextInputs: Record<string, string> = {};
    for (const variable of template.variables) {
      if (variable.scope === "default") {
        nextInputs[variable.key] = templateDefaults[variable.key] || "";
      } else {
        nextInputs[variable.key] = "";
      }
    }
  setActiveTemplate(template);
  setTemplateInputs(nextInputs);
  setTemplateInputError(null);
  setTemplateMissingKeys(new Set());
  setShowTemplateModal(true);
};

  const closeTemplateModal = () => {
    setShowTemplateModal(false);
    setActiveTemplate(null);
    setTemplateInputs({});
    setTemplateInputError(null);
    setTemplateMissingKeys(new Set());
  };

  const saveDefaultsFromModal = () => {
    if (!activeTemplate?.variables) {
      return;
    }
    const updates: Record<string, string> = {};
    for (const variable of activeTemplate.variables) {
      if (variable.scope === "default") {
        updates[variable.key] = templateInputs[variable.key] || "";
      }
    }
    setTemplateDefaults((prev) => ({ ...prev, ...updates }));
  };

  const applyTemplateFromModal = () => {
    if (!activeTemplate || !activeTemplate.variables) {
      return;
    }
    const missingKeys = new Set<string>();
    for (const variable of activeTemplate.variables) {
      if (variable.required && !templateInputs[variable.key]) {
        missingKeys.add(variable.key);
      }
    }
    if (missingKeys.size > 0) {
      setTemplateMissingKeys(missingKeys);
      setTemplateInputError("Fill the highlighted required fields.");
      return;
    }
    setTemplateInputError(null);
    setTemplateMissingKeys(new Set());
    const nextGoal = replaceTokens(activeTemplate.goal, templateInputs);
    const nextContext = replaceTokensForJson(activeTemplate.contextJson, templateInputs);
    try {
      JSON.parse(nextContext || "{}");
    } catch {
      setTemplateInputError("Rendered context JSON is invalid.");
      return;
    }
    setGoal(nextGoal);
    setContextJson(nextContext);
    setPriority(activeTemplate.priority);
    closeTemplateModal();
  };

  const defaultsTemplates = templates.filter((template) =>
    template.variables?.some((variable) => variable.scope === "default")
  );

  const selectedDefaultsTemplate =
    defaultsTemplates.find((template) => template.id === defaultsTemplateId) || null;

  useEffect(() => {
    if (defaultsTemplateId || defaultsTemplates.length === 0) {
      return;
    }
    setDefaultsTemplateId(defaultsTemplates[0].id);
  }, [defaultsTemplateId, defaultsTemplates]);

  const updateDefaultValue = (key: string, value: string) => {
    setTemplateDefaults((prev) => ({ ...prev, [key]: value }));
  };

  useEffect(() => {
    const previewValues = customVariables.reduce<Record<string, string>>((acc, variable) => {
      if (!variable.key) {
        return acc;
      }
      if (showRawPlaceholders) {
        acc[variable.key] = `{{${variable.key}}}`;
        return acc;
      }
      if (variable.scope === "default") {
        acc[variable.key] = templateDefaults[variable.key] || variable.placeholder || "";
        return acc;
      }
      acc[variable.key] = variable.placeholder || `<${variable.key}>`;
      return acc;
    }, {});
    const handle = window.setTimeout(() => {
      const nextGoal = replaceTokens(goal, previewValues);
      const nextContext = replaceTokensForJson(contextJson, previewValues);
      let isValid = true;
      try {
        JSON.parse(nextContext || "{}");
      } catch {
        isValid = false;
      }
      setPreviewGoal(nextGoal);
      setPreviewContext(nextContext);
      setPreviewContextIsValid(isValid);
    }, 200);
    return () => window.clearTimeout(handle);
  }, [goal, contextJson, customVariables, templateDefaults, showRawPlaceholders]);

  const addCustomVariable = () => {
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `var-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setCustomVariables((prev) => [
      ...prev,
      { id, key: "", label: "", scope: "per_run", required: false, placeholder: "" }
    ]);
  };

  const normalizeVariables = (variables: TemplateVariable[]) =>
    variables.map((variable) => {
      if (variable.id) {
        return variable;
      }
      const id =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `var-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      return { ...variable, id };
    });

  const editTemplate = (template: Template) => {
    setGoal(template.goal);
    setContextJson(template.contextJson);
    setPriority(template.priority);
    setTemplateName(template.builtIn ? `${template.name} (copy)` : template.name);
    setCustomVariables(template.variables ? normalizeVariables(template.variables) : []);
    setTemplateError(null);
  };

  const updateCustomVariable = (index: number, updates: Partial<TemplateVariable>) => {
    setCustomVariables((prev) =>
      prev.map((variable, idx) => (idx === index ? { ...variable, ...updates } : variable))
    );
  };

  const removeCustomVariable = (index: number) => {
    setCustomVariables((prev) => prev.filter((_, idx) => idx !== index));
  };

  const fetchJson = async (url: string) => {
    try {
      const response = await fetch(url);
      const status = response.status;
      if (!response.ok) {
        return { ok: false, status, data: null, error: null as string | null };
      }
      const text = await response.text();
      if (!text) {
        return { ok: true, status, data: null, error: null as string | null };
      }
      try {
        return { ok: true, status, data: JSON.parse(text), error: null as string | null };
      } catch (parseError) {
        return {
          ok: false,
          status,
          data: null,
          error: parseError instanceof Error ? parseError.message : "Invalid JSON"
        };
      }
    } catch (error) {
      return {
        ok: false,
        status: null as number | null,
        data: null,
        error: error instanceof Error ? error.message : "Network error"
      };
    }
  };


  const loadJobDetails = async (jobId: string) => {
    setSelectedJobId(jobId);
    setDetailsLoading(true);
    setDetailsError(null);

    const detailsResult = await fetchJson(`${apiUrl}/jobs/${jobId}/details`);
    if (detailsResult.ok && detailsResult.data && typeof detailsResult.data === "object") {
      const payload = detailsResult.data as {
        plan?: Plan | null;
        tasks?: Task[];
        task_results?: Record<string, TaskResult>;
      };
      setSelectedPlan(payload.plan ?? null);
      setSelectedTasks(Array.isArray(payload.tasks) ? payload.tasks : []);
      setTaskResults(payload.task_results && typeof payload.task_results === "object" ? payload.task_results : {});
    } else {
      setSelectedPlan(null);
      setSelectedTasks([]);
      setTaskResults({});
      if (detailsResult.status) {
        setDetailsError(`Failed to load job details (${detailsResult.status}).`);
      } else if (detailsResult.error) {
        setDetailsError(`Failed to load job details (${detailsResult.error}).`);
      } else {
        setDetailsError("Failed to load job details.");
      }
    }

    await loadMemoryEntries(jobId);

    setDetailsLoading(false);
  };

  const loadMemoryEntries = async (
    jobId: string,
    limits: Record<string, number> = memoryLimits
  ) => {
    setMemoryLoading(true);
    setMemoryError(null);
    const memoryNames = ["job_context", "task_outputs"];
    const memoryResults = await Promise.all(
      memoryNames.map((name) =>
        fetchJson(
          `${apiUrl}/memory/read?name=${encodeURIComponent(name)}&job_id=${encodeURIComponent(
            jobId
          )}&limit=${encodeURIComponent(String(limits[name] ?? 50))}`
        )
      )
    );
    const nextMemoryEntries: Record<string, MemoryEntry[]> = {};
    let memoryFailure = false;
    memoryResults.forEach((result, index) => {
      const name = memoryNames[index];
      if (result.ok && Array.isArray(result.data)) {
        nextMemoryEntries[name] = result.data as MemoryEntry[];
      } else {
        nextMemoryEntries[name] = [];
        memoryFailure = true;
      }
    });
    setMemoryEntries(nextMemoryEntries);
    if (memoryFailure) {
      if (memoryResults.some((result) => result.status)) {
        const status = memoryResults.find((result) => result.status)?.status;
        setMemoryError(`Failed to load memory entries (${status}).`);
      } else if (memoryResults.some((result) => result.error)) {
        const error = memoryResults.find((result) => result.error)?.error;
        setMemoryError(`Failed to load memory entries (${error}).`);
      } else {
        setMemoryError("Failed to load memory entries.");
      }
    }
    setMemoryLoading(false);
  };

  const filterMemoryEntries = (entries: MemoryEntry[]) => {
    const keyFilter = memoryFilters.key.trim().toLowerCase();
    const toolFilter = memoryFilters.tool.trim().toLowerCase();
    if (!keyFilter && !toolFilter) {
      return entries;
    }
    return entries.filter((entry) => {
      const keyValue = (entry.key || "").toLowerCase();
      const sourceTool =
        typeof entry.payload?.source_tool === "string"
          ? entry.payload.source_tool.toLowerCase()
          : "";
      if (keyFilter && !keyValue.includes(keyFilter)) {
        return false;
      }
      if (toolFilter && !sourceTool.includes(toolFilter)) {
        return false;
      }
      return true;
    });
  };

  const closeDetails = () => {
    setSelectedJobId(null);
    setSelectedPlan(null);
    setSelectedTasks([]);
    setTaskResults({});
    setDetailsError(null);
    setMemoryEntries({});
    setMemoryError(null);
    setMemoryLoading(false);
    setExpandedMemoryGroups(new Set());
    setExpandedMemoryEntries({});
    setMemoryFilters({ key: "", tool: "" });
    setMemoryLimits({ job_context: memoryLimitDefault, task_outputs: memoryLimitDefault });
  };

  const stopJob = async (jobId: string) => {
    const response = await fetch(`${apiUrl}/jobs/${jobId}/cancel`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  const resumeJob = async (jobId: string) => {
    const response = await fetch(`${apiUrl}/jobs/${jobId}/resume`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  const retryJob = async (jobId: string) => {
    const response = await fetch(`${apiUrl}/jobs/${jobId}/retry`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  const retryFailedTasks = async (jobId: string) => {
    const response = await fetch(`${apiUrl}/jobs/${jobId}/retry_failed`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  const replanJob = async (jobId: string) => {
    const response = await fetch(`${apiUrl}/jobs/${jobId}/replan`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  const clearJob = async (jobId: string) => {
    const confirmed = window.confirm("Clear this job and all tasks?");
    if (!confirmed) {
      return;
    }
    const response = await fetch(`${apiUrl}/jobs/${jobId}/clear`, { method: "POST" });
    if (response.ok) {
      loadJobs();
    }
  };

  return (
    <main className={`relative${isResizing ? " select-none" : ""}`}>
      <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-cyan-200/40 blur-3xl animate-float-soft" />
      <div className="pointer-events-none absolute top-48 -left-16 h-80 w-80 rounded-full bg-amber-200/50 blur-3xl animate-float-soft" />
      <button
        className="fixed left-4 top-4 z-40 rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-md lg:hidden"
        onClick={() => setSidebarOpen(true)}
      >
        Templates
      </button>
      {!sidebarOpen ? (
        <button
          className="fixed left-4 top-4 z-40 hidden rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-md lg:inline-flex"
          onClick={() => setSidebarOpen(true)}
        >
          Show templates
        </button>
      ) : null}
      <aside
        className={`fixed inset-y-0 left-0 z-50 transform bg-white/95 shadow-xl transition-transform duration-300 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
        style={{ width: sidebarWidth }}
      >
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <div className="font-display text-lg text-slate-900">Template Vault</div>
          <div className="flex items-center gap-2">
            <button
              className="rounded-full border border-slate-200 px-2 py-1 text-xs text-slate-600 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            >
              Close
            </button>
            <button
              className="hidden rounded-full border border-slate-200 px-2 py-1 text-xs text-slate-600 lg:inline-flex"
              onClick={() => setSidebarOpen(false)}
            >
              Collapse
            </button>
          </div>
        </div>
        <div
          className="absolute right-0 top-0 hidden h-full w-2 cursor-col-resize bg-transparent transition hover:bg-slate-200/50 lg:block"
          onMouseDown={(event) => {
            event.preventDefault();
            setIsResizing(true);
          }}
        />
        <div className="h-full overflow-y-auto px-5 pb-8 pt-4">
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                Templates
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="text-xs text-slate-500 underline decoration-slate-300 underline-offset-4"
                  onClick={() => setIsReorderMode((prev) => !prev)}
                >
                  {isReorderMode ? "Done" : "Reorder"}
                </button>
                <button
                  className="text-xs text-slate-500 underline decoration-slate-300 underline-offset-4"
                  onClick={() => {
                    setTemplates(BUILT_IN_TEMPLATES);
                    if (typeof window !== "undefined") {
                      window.localStorage.removeItem(TEMPLATE_ORDER_KEY);
                    }
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
            <div className="space-y-3">
              {templates.map((template, index) => (
                <div
                  key={template.id}
                  className={`rounded-xl border border-slate-200/70 bg-slate-50/80 p-3 transition hover:border-slate-300 hover:bg-white ${
                    isReorderMode ? "cursor-move" : ""
                  } ${draggingTemplateId === template.id ? "opacity-60" : ""}`}
                  draggable={isReorderMode}
                  onDragStart={() => setDraggingTemplateId(template.id)}
                  onDragEnd={() => {
                    setDraggingTemplateId(null);
                    setDragOverTemplateId(null);
                  }}
                  onDragOver={(event) => {
                    if (!isReorderMode) {
                      return;
                    }
                    event.preventDefault();
                    if (dragOverTemplateId !== template.id) {
                      setDragOverTemplateId(template.id);
                    }
                  }}
                  onDrop={(event) => {
                    if (!isReorderMode) {
                      return;
                    }
                    event.preventDefault();
                    if (draggingTemplateId) {
                      reorderTemplates(draggingTemplateId, template.id);
                    }
                    setDraggingTemplateId(null);
                    setDragOverTemplateId(null);
                  }}
                >
                  {dragOverTemplateId === template.id && draggingTemplateId !== template.id ? (
                    <div className="mb-2 h-1 rounded-full bg-cyan-200/80" />
                  ) : null}
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">{template.name}</div>
                      {template.description ? (
                        <div className="mt-1 text-xs text-slate-600">
                          {template.description}
                        </div>
                      ) : null}
                      <div className="mt-1 text-xs text-slate-500">
                        {truncate(template.goal, 90)}
                      </div>
                      {template.builtIn ? (
                        <span className="mt-2 inline-flex rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                          Built-in
                        </span>
                      ) : null}
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      {isReorderMode ? (
                        <div className="flex flex-col gap-2">
                          <button
                            className="rounded-lg bg-slate-900 px-3 py-1 text-xs font-semibold text-white disabled:opacity-40"
                            onClick={() => moveTemplate(template.id, "up")}
                            disabled={index === 0}
                          >
                            Up
                          </button>
                          <button
                            className="rounded-lg bg-slate-900 px-3 py-1 text-xs font-semibold text-white disabled:opacity-40"
                            onClick={() => moveTemplate(template.id, "down")}
                            disabled={index === templates.length - 1}
                          >
                            Down
                          </button>
                        </div>
                      ) : (
                        <>
                          <button
                            className="rounded-lg bg-slate-900 px-3 py-1 text-xs font-semibold text-white"
                            onClick={() => {
                              openTemplateModal(template);
                              setSidebarOpen(false);
                            }}
                          >
                            Use
                          </button>
                          <button
                            className="rounded-lg border border-slate-300 px-3 py-1 text-xs font-semibold text-slate-700"
                            onClick={() => {
                              editTemplate(template);
                              setSidebarOpen(false);
                            }}
                          >
                            Edit
                          </button>
                        </>
                      )}
                      {!template.builtIn ? (
                        <button
                          className="text-[11px] text-rose-500 underline decoration-rose-200/60 underline-offset-4"
                          onClick={() => deleteTemplate(template.id)}
                        >
                          Delete
                        </button>
                      ) : null}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="mt-6 rounded-2xl border border-slate-200/70 bg-white p-4 shadow-sm">
            <div className="text-sm font-semibold text-slate-800">Save current prompt</div>
            <div className="mt-3 flex flex-col gap-2">
              <input
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-200"
                placeholder="Template name"
                value={templateName}
                onChange={(event) => setTemplateName(event.target.value)}
              />
              {templateError ? <div className="text-xs text-rose-500">{templateError}</div> : null}
              <button
                className="rounded-lg bg-slate-900 px-3 py-2 text-xs font-semibold text-white"
                onClick={saveTemplate}
              >
                Save Template
              </button>
            </div>
            <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                  Variables
                </div>
                <button
                  className="rounded-full border border-slate-300 px-3 py-1 text-[11px] text-slate-600"
                  onClick={addCustomVariable}
                >
                  Add variable
                </button>
              </div>
              <label className="mt-3 inline-flex items-center gap-2 text-[11px] text-slate-500">
                <input
                  type="checkbox"
                  checked={showRawPlaceholders}
                  onChange={(event) => setShowRawPlaceholders(event.target.checked)}
                />
                Show raw placeholders in preview
              </label>
              {customVariables.length === 0 ? (
                <p className="mt-2 text-xs text-slate-500">
                  Optional. Use keys like <span className="font-semibold">company</span> and
                  reference them as <span className="font-semibold">{"{{company}}"}</span>.
                </p>
              ) : (
                <div className="mt-3 space-y-3">
                  {customVariables.map((variable, index) => (
                    <div key={variable.id || `var-${index}`} className="rounded-lg bg-white p-3">
                      <div className="grid gap-2 md:grid-cols-[1.1fr,1fr]">
                        <input
                          className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-900"
                          placeholder="Key (e.g. company)"
                          value={variable.key}
                          onChange={(event) =>
                            updateCustomVariable(index, { key: event.target.value })
                          }
                        />
                        <input
                          className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-900"
                          placeholder="Label (e.g. Company)"
                          value={variable.label}
                          onChange={(event) =>
                            updateCustomVariable(index, { label: event.target.value })
                          }
                        />
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-[1.2fr,0.8fr,0.6fr]">
                        <input
                          className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-900"
                          placeholder="Placeholder"
                          value={variable.placeholder || ""}
                          onChange={(event) =>
                            updateCustomVariable(index, { placeholder: event.target.value })
                          }
                        />
                        <select
                          className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-900"
                          value={variable.scope}
                          onChange={(event) =>
                            updateCustomVariable(index, {
                              scope: event.target.value as TemplateVariable["scope"]
                            })
                          }
                        >
                          <option value="per_run">Per run</option>
                          <option value="default">Saved default</option>
                        </select>
                        <label className="flex items-center gap-2 text-xs text-slate-600">
                          <input
                            type="checkbox"
                            checked={variable.required || false}
                            onChange={(event) =>
                              updateCustomVariable(index, { required: event.target.checked })
                            }
                          />
                          Required
                        </label>
                      </div>
                      <div className="mt-2 flex justify-end">
                        <button
                          className="text-[11px] text-rose-500 underline decoration-rose-200/60 underline-offset-4"
                          onClick={() => removeCustomVariable(index)}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <TemplatePreview
              goal={previewGoal}
              context={previewContext}
              isValid={previewContextIsValid}
            />
          </section>

          <section className="mt-6 rounded-2xl border border-slate-200/70 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold text-slate-800">Saved defaults</div>
              <select
                className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-900"
                value={defaultsTemplateId}
                onChange={(event) => setDefaultsTemplateId(event.target.value)}
              >
                <option value="">Pick template</option>
                {defaultsTemplates.map((template) => (
                  <option key={template.id} value={template.id}>
                    {template.name}
                  </option>
                ))}
              </select>
            </div>
            {selectedDefaultsTemplate ? (
              <div className="mt-3 space-y-2">
                {selectedDefaultsTemplate.variables
                  ?.filter((variable) => variable.scope === "default")
                  .map((variable) => (
                    <div key={variable.key}>
                      <label className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                        {variable.label}
                      </label>
                      <textarea
                        className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-900"
                        rows={variable.key === "resume_text" ? 4 : 2}
                        placeholder={variable.placeholder}
                        value={templateDefaults[variable.key] || ""}
                        onChange={(event) =>
                          updateDefaultValue(variable.key, event.target.value)
                        }
                      />
                    </div>
                  ))}
              </div>
            ) : (
              <p className="mt-3 text-xs text-slate-500">
                Choose a template to edit your saved defaults.
              </p>
            )}
          </section>
        </div>
      </aside>


      {showTemplateModal && activeTemplate ? (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-900/60 px-4 py-10 backdrop-blur-sm">
          <div className="w-full max-w-2xl max-h-[85vh] overflow-hidden rounded-2xl bg-white shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-6 py-4">
              <div>
                <h3 className="font-display text-xl">{activeTemplate.name}</h3>
                {activeTemplate.description ? (
                  <p className="mt-1 text-sm text-slate-600">
                    {activeTemplate.description}
                  </p>
                ) : null}
                <p className="mt-1 text-sm text-slate-500">
                  Fill values for this run. Defaults are saved automatically.
                </p>
              </div>
              <button
                className="text-sm text-slate-500"
                onClick={closeTemplateModal}
              >
                Close
              </button>
            </div>
            <div className="max-h-[60vh] overflow-y-auto px-6 py-5">
              <div className="grid gap-4">
                {activeTemplate.variables?.map((variable) => (
                  <div key={variable.key}>
                    <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                      {variable.label} {variable.scope === "per_run" ? "· per run" : "· default"}
                    </label>
                    <textarea
                      className={`mt-2 w-full rounded-xl border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 ${
                        templateMissingKeys.has(variable.key)
                          ? "border-rose-400 focus:border-rose-400 focus:ring-rose-200"
                          : "border-slate-200 focus:border-slate-400 focus:ring-slate-200"
                      }`}
                      rows={variable.key === "resume_text" ? 6 : 3}
                      placeholder={variable.placeholder}
                      value={templateInputs[variable.key] || ""}
                      onChange={(event) =>
                        setTemplateInputs((prev) => ({
                          ...prev,
                          [variable.key]: event.target.value
                        }))
                      }
                    />
                  </div>
                ))}
              </div>
              {templateInputError ? (
                <div className="mt-3 text-sm text-rose-600">{templateInputError}</div>
              ) : null}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 px-6 py-4">
              <button
                className="rounded-xl border border-slate-200 px-4 py-2 text-sm text-slate-600"
                onClick={saveDefaultsFromModal}
              >
                Save defaults
              </button>
              <button
                className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white"
                onClick={applyTemplateFromModal}
              >
                Apply template
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div
        className="space-y-10 transition-[margin] duration-300"
        style={{ marginLeft: isDesktop && sidebarOpen ? sidebarWidth : 0 }}
      >

        <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 p-8 text-white shadow-2xl animate-fade-up">
          <div className="pointer-events-none absolute -right-24 -top-20 h-64 w-64 rounded-full bg-emerald-400/20 blur-3xl animate-float-soft" />
          <div className="pointer-events-none absolute -bottom-16 left-10 h-52 w-52 rounded-full bg-sky-300/30 blur-3xl animate-float-soft" />
          <div className="relative">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h1 className="font-display text-3xl tracking-tight md:text-4xl">
                  Agentic Planner Executor
                </h1>
                <p className="mt-2 max-w-2xl text-sm text-slate-200">
                  Craft a goal, drop in context, and let the system orchestrate the plan. Save your
                  favorite prompt setups as templates for instant reuse.
                </p>
              </div>
              <div className="rounded-full border border-white/30 bg-white/10 px-4 py-2 text-xs uppercase tracking-[0.25em] text-slate-200">
                Compose
              </div>
            </div>
            <div className="mt-6">
              <div className="rounded-2xl bg-white/95 p-6 text-slate-900 shadow-lg ring-1 ring-white/30">
                <div className="flex items-center justify-between">
                  <h2 className="font-display text-xl">Compose Job</h2>
                  <span className="text-xs text-slate-500">Live orchestration</span>
                </div>
                <div className="mt-3 rounded-xl border border-slate-100 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  Tip: Use templates for repeatable workflows. Defaults are remembered for you.
                </div>
                {submitError ? (
                  <div className="mt-3 text-sm text-rose-600">{submitError}</div>
                ) : null}
                <div className="mt-4 grid gap-4">
                  <div>
                    <label className="text-sm font-medium text-slate-700">Goal</label>
                    <input
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-200"
                      value={goal}
                      onChange={(event) => setGoal(event.target.value)}
                      placeholder="Generate an implementation checklist"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700">Context JSON</label>
                    <textarea
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-200"
                      rows={6}
                      value={contextJson}
                      onChange={(event) => setContextJson(event.target.value)}
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium text-slate-700">Priority</label>
                    <input
                      type="number"
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-200"
                      value={priority}
                      onChange={(event) => setPriority(Number(event.target.value))}
                    />
                  </div>
                  <div className="grid gap-2 text-xs text-slate-500 md:grid-cols-2">
                    <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                      Saved templates live in your browser only.
                    </div>
                    <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                      Priority helps the planner sequence jobs.
                    </div>
                  </div>
                  <button
                    className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-md transition hover:bg-slate-800"
                    onClick={submitJob}
                  >
                    Submit Job
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
        <section className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm animate-fade-up-delayed">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-xl">Jobs</h2>
            <p className="mt-1 text-xs text-slate-500">
              Track submitted goals and manage their lifecycle.
            </p>
          </div>
          <div className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
            {jobs.length} total
          </div>
        </div>
        {jobs.length === 0 ? (
          <div className="mt-4 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
            No jobs yet. Submit a goal above to start planning.
          </div>
        ) : (
          <ul className="mt-4 grid gap-3">
            {jobs.map((job, index) => (
              <li
                key={job.id}
                className="rounded-xl border border-slate-100 bg-white p-4 shadow-sm transition hover:border-slate-200 hover:shadow-md animate-fade-up"
                style={{ animationDelay: `${index * 0.06}s` }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-slate-900 break-words">
                      {job.goal}
                    </div>
                    <div className="mt-1 text-xs text-slate-500 break-words">{job.id}</div>
                  </div>
                  <span className="shrink-0 self-start rounded-full bg-slate-100 px-2 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500">
                    {job.status}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
                    onClick={() => loadJobDetails(job.id)}
                  >
                    Details
                  </button>
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
                    onClick={() => stopJob(job.id)}
                  >
                    Stop
                  </button>
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
                    onClick={() => resumeJob(job.id)}
                  >
                    Resume
                  </button>
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
                    onClick={() => retryJob(job.id)}
                  >
                    Retry
                  </button>
                  <button
                    className="rounded-full border border-amber-200 px-3 py-1 text-amber-700 transition hover:border-amber-300 hover:text-amber-800"
                    onClick={() => retryFailedTasks(job.id)}
                  >
                    Retry failed
                  </button>
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
                    onClick={() => replanJob(job.id)}
                  >
                    Replan
                  </button>
                  <button
                    className="rounded-full border border-rose-200 px-3 py-1 text-rose-600 transition hover:border-rose-300 hover:text-rose-700"
                    onClick={() => clearJob(job.id)}
                  >
                    Clear
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

        <section className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm animate-fade-up-delayed-more">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-xl">Job Details</h2>
            <p className="mt-1 text-xs text-slate-500">
              Explore plan summaries, task dependencies, and the live DAG.
            </p>
          </div>
          {selectedJobId ? (
            <button
              className="rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-600 transition hover:border-slate-300 hover:text-slate-900"
              onClick={closeDetails}
            >
              Close
            </button>
          ) : null}
        </div>
        {!selectedJobId ? (
          <div className="mt-4 rounded-xl border border-dashed border-slate-200 bg-slate-50 p-6 text-sm text-slate-500">
            Select a job to view its plan, tasks, and DAG.
          </div>
        ) : detailsLoading ? (
          <p className="mt-3 text-sm text-slate-600">Loading job details...</p>
        ) : detailsError ? (
          <p className="mt-3 text-sm text-rose-600">{detailsError}</p>
        ) : (
            <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-sm text-slate-700">
              <div className="font-medium">Job ID</div>
              <div className="break-all text-xs text-slate-500">{selectedJobId}</div>
              <div className="mt-3 font-medium">Plan</div>
              {selectedPlan ? (
                <div className="text-xs text-slate-600">
                  {selectedPlan.tasks_summary || "Plan available."}
                </div>
              ) : selectedJob?.status === "failed" &&
                typeof selectedJob.metadata?.plan_error === "string" ? (
                <div className="text-xs text-rose-600">
                  Plan failed: {selectedJob.metadata.plan_error}
                </div>
              ) : (
                <div className="text-xs text-slate-600">Plan not created yet.</div>
              )}
            </div>

            {selectedTasks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
                No tasks yet.
              </div>
            ) : (
              <div className="overflow-auto rounded-xl border border-slate-100 bg-white p-3 shadow-sm">
                {(() => {
                  const dag = buildDagLayout(selectedTasks);
                  return (
                    <svg
                      width={dag.width}
                      height={dag.height}
                      viewBox={`0 0 ${dag.width} ${dag.height}`}
                    >
                      <defs>
                        <marker
                          id="arrow"
                          markerWidth="10"
                          markerHeight="10"
                          refX="8"
                          refY="3"
                          orient="auto"
                        >
                          <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
                        </marker>
                      </defs>
                      {dag.edges.map((edge, index) => {
                        const fromX = edge.from.x + 180;
                        const fromY = edge.from.y + 28;
                        const toX = edge.to.x;
                        const toY = edge.to.y + 28;
                        const midX = (fromX + toX) / 2;
                        return (
                          <path
                            key={`edge-${index}`}
                            d={`M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`}
                            stroke="#94a3b8"
                            strokeWidth="1.5"
                            fill="none"
                            markerEnd="url(#arrow)"
                          />
                        );
                      })}
                      {dag.nodes.map((node) => {
                        const colors = statusColors[node.status] || statusColors.pending;
                        return (
                          <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                            <rect
                              width="180"
                              height="56"
                              rx="8"
                              fill={colors.fill}
                              stroke={colors.stroke}
                              strokeWidth="1.5"
                            />
                            <text x="12" y="22" fontSize="12" fill="#0f172a">
                              {truncate(node.name, 20)}
                            </text>
                            <text x="12" y="40" fontSize="11" fill="#475569">
                              {node.status}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  );
                })()}
              </div>
            )}
            {selectedTasks.length > 0 ? (
              <div className="rounded-xl border border-slate-100 bg-white p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-slate-800">Task Inputs</div>
                    <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Instructions + Tools
                    </div>
                  </div>
                  <button
                    className="rounded-full border border-slate-200 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500"
                    onClick={() => setShowTaskInputs((prev) => !prev)}
                  >
                    {showTaskInputs ? "Hide" : "Show"}
                  </button>
                </div>
                {showTaskInputs ? (
                  <div className="mt-4 space-y-3">
                    {selectedTasks.map((task) => {
                      const isExpanded = expandedTaskInputs.has(task.id);
                      return (
                        <div
                          key={task.id}
                          className="rounded-xl border border-slate-100 bg-slate-50 p-3"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-slate-900">{task.name}</div>
                              <div className="mt-1 text-xs text-slate-500">{task.id}</div>
                            </div>
                            <div className="flex flex-col items-end gap-2">
                              <span className="rounded-full bg-white px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                                {task.status}
                              </span>
                              <button
                                className="rounded-full border border-slate-200 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500"
                                onClick={() =>
                                  setExpandedTaskInputs((prev) => {
                                    const next = new Set(prev);
                                    if (next.has(task.id)) {
                                      next.delete(task.id);
                                    } else {
                                      next.add(task.id);
                                    }
                                    return next;
                                  })
                                }
                              >
                                {isExpanded ? "Hide" : "Show"}
                              </button>
                            </div>
                          </div>
                          {isExpanded ? (
                            <div className="mt-3 grid gap-3 text-xs text-slate-600">
                              <div>
                                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                  Instruction
                                </div>
                                <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-600">
                                  {task.instruction || "No instruction available."}
                                </pre>
                              </div>
                              <div className="grid gap-3 md:grid-cols-2">
                                <div>
                                  <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                    Expected Schema
                                  </div>
                                  <div className="mt-1 text-xs text-slate-600">
                                    {task.expected_output_schema_ref || "—"}
                                  </div>
                                </div>
                                <div>
                                  <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                    Tool Requests
                                  </div>
                                  <div className="mt-1 text-xs text-slate-600">
                                    {task.tool_requests && task.tool_requests.length > 0
                                      ? task.tool_requests.join(", ")
                                      : "—"}
                                  </div>
                                </div>
                              </div>
                              <div>
                                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                  Tool Calls
                                </div>
                                {taskResults[task.id]?.tool_calls &&
                                taskResults[task.id].tool_calls!.length > 0 ? (
                                  <div className="mt-2 space-y-3">
                                    {taskResults[task.id].tool_calls!.map((call, index) => (
                                      <div
                                        key={`${task.id}-call-${index}`}
                                        className="rounded-lg border border-slate-200 bg-white p-3"
                                      >
                                        <div className="flex items-center justify-between">
                                          <div className="text-xs font-semibold text-slate-700">
                                            {call.tool_name}
                                          </div>
                                          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                                            {call.status}
                                          </span>
                                        </div>
                                        <div className="mt-2 grid gap-3 md:grid-cols-2">
                                          <div>
                                            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                              Input
                                            </div>
                                            <pre className="mt-1 whitespace-pre-wrap rounded-md border border-slate-100 bg-slate-50 px-2 py-1 text-[11px] text-slate-600">
                                              {JSON.stringify(call.input || {}, null, 2)}
                                            </pre>
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                              Output
                                            </div>
                                            <pre className="mt-1 whitespace-pre-wrap rounded-md border border-slate-100 bg-slate-50 px-2 py-1 text-[11px] text-slate-600">
                                              {JSON.stringify(call.output_or_error || {}, null, 2)}
                                            </pre>
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="mt-2 text-xs text-slate-500">
                                    No tool calls recorded yet.
                                  </div>
                                )}
                              </div>
                              <div>
                                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                  Outputs
                                </div>
                                <pre className="mt-2 whitespace-pre-wrap rounded-lg border border-slate-200 bg-white px-3 py-2 text-[11px] text-slate-600">
                                  {JSON.stringify(taskResults[task.id]?.outputs || {}, null, 2)}
                                </pre>
                              </div>
                            </div>
                          ) : (
                            <div className="mt-3 text-xs text-slate-500">Collapsed.</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            ) : null}

            {selectedJobId ? (
              <div className="rounded-xl border border-slate-100 bg-white p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-slate-800">Memory</div>
                    <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
                      Job Entries
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {memoryFilters.key.trim() || memoryFilters.tool.trim() ? (
                      <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-amber-700">
                        Filters active
                      </span>
                    ) : null}
                    <button
                      className="rounded-full border border-slate-200 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500"
                      onClick={() => setShowMemory((prev) => !prev)}
                    >
                      {showMemory ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>
                {showMemory ? (
                  <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <input
                      className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 md:w-56"
                      placeholder="Filter by key"
                      value={memoryFilters.key}
                      onChange={(event) =>
                        setMemoryFilters((prev) => ({ ...prev, key: event.target.value }))
                      }
                    />
                    <input
                      className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 md:w-56"
                      placeholder="Filter by source tool"
                      value={memoryFilters.tool}
                      onChange={(event) =>
                        setMemoryFilters((prev) => ({ ...prev, tool: event.target.value }))
                      }
                    />
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] uppercase tracking-[0.2em] text-slate-400">
                        Limit
                      </span>
                      <select
                        className="rounded-lg border border-slate-200 bg-white px-2 py-2 text-xs text-slate-700"
                        value={memoryLimitDefault}
                        onChange={(event) => {
                          const nextLimit = Number(event.target.value);
                          if (!Number.isFinite(nextLimit)) {
                            return;
                          }
                          setMemoryLimitDefault(nextLimit);
                          const nextLimits = {
                            job_context: nextLimit,
                            task_outputs: nextLimit
                          };
                          setMemoryLimits(nextLimits);
                          if (selectedJobId) {
                            loadMemoryEntries(selectedJobId, nextLimits);
                          }
                        }}
                      >
                        {[10, 25, 50, 100, 200].map((limit) => (
                          <option key={limit} value={limit}>
                            {limit}
                          </option>
                        ))}
                      </select>
                    </div>
                    <button
                      className="rounded-full border border-slate-200 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500"
                      onClick={() => setMemoryFilters({ key: "", tool: "" })}
                    >
                      Clear
                    </button>
                    <button
                      className="rounded-full border border-slate-200 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500"
                      onClick={() => selectedJobId && loadMemoryEntries(selectedJobId)}
                    >
                      Refresh
                    </button>
                  </div>
                ) : null}
                {memoryLoading ? (
                  <div className="mt-3 text-xs text-slate-500">Loading memory entries...</div>
                ) : memoryError ? (
                  <div className="mt-3 text-xs text-rose-600">{memoryError}</div>
                ) : showMemory ? (
                  <div className="mt-4 space-y-3">
                    {["job_context", "task_outputs"].map((name) => {
                      const entries = memoryEntries[name] || [];
                      const filteredEntries = filterMemoryEntries(entries);
                      const limit = memoryLimits[name] ?? 50;
                      const canLoadMore = limit < 200;
                      const groupExpanded = expandedMemoryGroups.has(name);
                      return (
                        <div
                          key={name}
                          className="rounded-xl border border-slate-100 bg-slate-50 p-3"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <div className="text-sm font-semibold text-slate-900">{name}</div>
                              <div className="mt-1 text-xs text-slate-500">
                                {filteredEntries.length} of {entries.length} entries
                              </div>
                            </div>
                            <div className="flex flex-col items-end gap-2">
                              <span className="rounded-full bg-white px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500">
                                limit {limit}
                              </span>
                              <div className="flex items-center gap-2">
                                <button
                                  className="rounded-full border border-slate-200 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500"
                                  onClick={() =>
                                    setExpandedMemoryGroups((prev) => {
                                      const next = new Set(prev);
                                      if (next.has(name)) {
                                        next.delete(name);
                                      } else {
                                        next.add(name);
                                      }
                                      return next;
                                    })
                                  }
                                >
                                  {groupExpanded ? "Hide" : "Show"}
                                </button>
                                <button
                                  className="rounded-full border border-slate-200 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500 disabled:opacity-50"
                                  disabled={!canLoadMore}
                                  onClick={() => {
                                    if (!selectedJobId || !canLoadMore) {
                                      return;
                                    }
                                    const nextLimits = {
                                      ...memoryLimits,
                                      [name]: Math.min(limit + 10, 200)
                                    };
                                    setMemoryLimits(nextLimits);
                                    loadMemoryEntries(selectedJobId, nextLimits);
                                  }}
                                >
                                  Show more
                                </button>
                              </div>
                            </div>
                          </div>
                          {groupExpanded ? (
                            filteredEntries.length > 0 ? (
                              <div className="mt-3 space-y-3">
                                {filteredEntries.map((entry, index) => {
                                  const entryExpanded =
                                    expandedMemoryEntries[name]?.has(index) ?? false;
                                  return (
                                    <div
                                      key={`${name}-${entry.id}-${index}`}
                                      className="rounded-lg border border-slate-200 bg-white p-3"
                                    >
                                      <div className="flex items-center justify-between gap-3">
                                        <div className="text-xs font-semibold text-slate-700">
                                          {entry.key || "Untitled entry"}
                                        </div>
                                        <button
                                          className="rounded-full border border-slate-200 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500"
                                          onClick={() =>
                                            setExpandedMemoryEntries((prev) => {
                                              const next = { ...prev };
                                              const current = new Set(next[name] ?? []);
                                              if (current.has(index)) {
                                                current.delete(index);
                                              } else {
                                                current.add(index);
                                              }
                                              next[name] = current;
                                              return next;
                                            })
                                          }
                                        >
                                          {entryExpanded ? "Hide" : "Show"}
                                        </button>
                                      </div>
                                      {entryExpanded ? (
                                        <div className="mt-2 grid gap-3 text-xs text-slate-600 md:grid-cols-2">
                                          <div>
                                            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                              Metadata
                                            </div>
                                            <pre className="mt-1 whitespace-pre-wrap rounded-md border border-slate-100 bg-slate-50 px-2 py-1 text-[11px] text-slate-600">
                                              {JSON.stringify(entry.metadata || {}, null, 2)}
                                            </pre>
                                          </div>
                                          <div>
                                            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                              Payload
                                            </div>
                                            <pre className="mt-1 whitespace-pre-wrap rounded-md border border-slate-100 bg-slate-50 px-2 py-1 text-[11px] text-slate-600">
                                              {JSON.stringify(entry.payload || {}, null, 2)}
                                            </pre>
                                          </div>
                                          <div className="md:col-span-2">
                                            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                              Updated
                                            </div>
                                            <div className="mt-1 text-xs text-slate-600">
                                              {entry.updated_at || entry.created_at || "—"}
                                            </div>
                                          </div>
                                        </div>
                                      ) : (
                                        <div className="mt-2 text-[11px] text-slate-500">
                                          Collapsed.
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            ) : (
                              <div className="mt-3 text-xs text-slate-500">
                                No memory entries match the filters.
                              </div>
                            )
                          ) : (
                            <div className="mt-3 text-xs text-slate-500">Collapsed.</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="mt-3 text-xs text-slate-500">Hidden by default.</div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </section>

      <section className="animate-fade-up-delayed-more rounded-2xl border border-slate-100 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-display text-xl">Recent Events</h2>
            <p className="mt-1 text-xs text-slate-500">Live event stream snapshots.</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
              {events.length} shown
            </div>
            <button
              className="rounded-full border border-slate-200 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-500"
              onClick={() => setShowRecentEvents((prev) => !prev)}
            >
              {showRecentEvents ? "Hide" : "Show"}
            </button>
          </div>
        </div>
        {showRecentEvents ? (
          <ul className="mt-4 space-y-2 text-xs">
            {events.map((event, index) => {
              const isExpanded = expandedRecentEvents.has(index);
              return (
                <li
                  key={index}
                  className="rounded-xl border border-slate-100 bg-slate-50 px-3 py-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="font-medium text-slate-700">{event.type}</div>
                    <button
                      className="rounded-full border border-slate-200 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-slate-500"
                      onClick={() =>
                        setExpandedRecentEvents((prev) => {
                          const next = new Set(prev);
                          if (next.has(index)) {
                            next.delete(index);
                          } else {
                            next.add(index);
                          }
                          return next;
                        })
                      }
                    >
                      {isExpanded ? "Hide" : "Show"}
                    </button>
                  </div>
                  {isExpanded ? (
                    <pre className="mt-2 whitespace-pre-wrap text-slate-500">
                      {JSON.stringify(event.payload, null, 2)}
                    </pre>
                  ) : (
                    <div className="mt-2 text-[11px] text-slate-500">Collapsed.</div>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="mt-4 text-xs text-slate-500">Hidden by default.</div>
        )}
      </section>
      </div>
    </main>
  );
}
