"use client";

import { useEffect, useState } from "react";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Job = {
  id: string;
  goal: string;
  status: string;
  created_at: string;
  priority: number;
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
};

type EventEnvelope = {
  type: string;
  payload: Record<string, unknown>;
};

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
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);

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
        const payload = JSON.parse(event.data);
        setEvents((prev) => [payload, ...prev].slice(0, 50));
      } catch {
        return;
      }
    };
    return () => source.close();
  }, []);

  const submitJob = async () => {
    const response = await fetch(`${apiUrl}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goal,
        context_json: JSON.parse(contextJson || "{}"),
        priority
      })
    });
    if (response.ok) {
      setGoal("");
      setContextJson("{}");
      setPriority(0);
      loadJobs();
    }
  };

  const loadJobDetails = async (jobId: string) => {
    setSelectedJobId(jobId);
    setDetailsLoading(true);
    setDetailsError(null);
    try {
      const [planResponse, tasksResponse] = await Promise.all([
        fetch(`${apiUrl}/jobs/${jobId}/plan`),
        fetch(`${apiUrl}/jobs/${jobId}/tasks`)
      ]);
      if (planResponse.ok) {
        const planData = await planResponse.json();
        setSelectedPlan(planData);
      } else {
        setSelectedPlan(null);
      }
      if (tasksResponse.ok) {
        const taskData = await tasksResponse.json();
        setSelectedTasks(taskData);
      } else {
        setSelectedTasks([]);
      }
    } catch (error) {
      setDetailsError("Failed to load job details.");
    } finally {
      setDetailsLoading(false);
    }
  };

  const closeDetails = () => {
    setSelectedJobId(null);
    setSelectedPlan(null);
    setSelectedTasks([]);
    setDetailsError(null);
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
    <main className="space-y-8">
      <section className="rounded-lg bg-white p-6 shadow">
        <h1 className="text-2xl font-semibold">Agentic Planner Executor</h1>
        <p className="mt-2 text-sm text-slate-600">
          Submit a job goal and watch events stream in real time.
        </p>
        <div className="mt-4 grid gap-4">
          <div>
            <label className="text-sm font-medium">Goal</label>
            <input
              className="mt-1 w-full rounded border px-3 py-2"
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="Generate an implementation checklist"
            />
          </div>
          <div>
            <label className="text-sm font-medium">Context JSON</label>
            <textarea
              className="mt-1 w-full rounded border px-3 py-2"
              rows={4}
              value={contextJson}
              onChange={(event) => setContextJson(event.target.value)}
            />
          </div>
          <div>
            <label className="text-sm font-medium">Priority</label>
            <input
              type="number"
              className="mt-1 w-full rounded border px-3 py-2"
              value={priority}
              onChange={(event) => setPriority(Number(event.target.value))}
            />
          </div>
          <button
            className="rounded bg-blue-600 px-4 py-2 text-white"
            onClick={submitJob}
          >
            Submit Job
          </button>
        </div>
      </section>

      <section className="rounded-lg bg-white p-6 shadow">
        <h2 className="text-xl font-semibold">Jobs</h2>
        <ul className="mt-4 space-y-2">
          {jobs.map((job) => (
            <li key={job.id} className="rounded border px-3 py-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">{job.goal}</span>
                <span className="text-xs text-slate-500">{job.status}</span>
              </div>
              <div className="text-xs text-slate-500">{job.id}</div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <button
                  className="rounded border px-2 py-1"
                  onClick={() => loadJobDetails(job.id)}
                >
                  Details
                </button>
                <button
                  className="rounded border px-2 py-1"
                  onClick={() => stopJob(job.id)}
                >
                  Stop
                </button>
                <button
                  className="rounded border px-2 py-1"
                  onClick={() => resumeJob(job.id)}
                >
                  Resume
                </button>
                <button
                  className="rounded border px-2 py-1"
                  onClick={() => retryJob(job.id)}
                >
                  Retry
                </button>
                <button
                  className="rounded border px-2 py-1 text-red-600"
                  onClick={() => clearJob(job.id)}
                >
                  Clear
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-lg bg-white p-6 shadow">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Job Details</h2>
          {selectedJobId ? (
            <button className="text-sm text-slate-500" onClick={closeDetails}>
              Close
            </button>
          ) : null}
        </div>
        {!selectedJobId ? (
          <p className="mt-3 text-sm text-slate-600">
            Select a job to view its plan, tasks, and DAG.
          </p>
        ) : detailsLoading ? (
          <p className="mt-3 text-sm text-slate-600">Loading job details...</p>
        ) : detailsError ? (
          <p className="mt-3 text-sm text-red-600">{detailsError}</p>
        ) : (
          <div className="mt-4 space-y-4">
            <div className="rounded border bg-slate-50 p-3 text-sm text-slate-700">
              <div className="font-medium">Job ID</div>
              <div className="break-all text-xs text-slate-500">{selectedJobId}</div>
              <div className="mt-2 font-medium">Plan</div>
              {selectedPlan ? (
                <div className="text-xs text-slate-600">
                  {selectedPlan.tasks_summary || "Plan available."}
                </div>
              ) : (
                <div className="text-xs text-slate-600">Plan not created yet.</div>
              )}
            </div>

            {selectedTasks.length === 0 ? (
              <div className="text-sm text-slate-600">No tasks yet.</div>
            ) : (
              <div className="overflow-auto rounded border bg-white p-3">
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
          </div>
        )}
      </section>

      <section className="rounded-lg bg-white p-6 shadow">
        <h2 className="text-xl font-semibold">Recent Events</h2>
        <ul className="mt-4 space-y-2 text-xs">
          {events.map((event, index) => (
            <li key={index} className="rounded border px-3 py-2">
              <div className="font-medium">{event.type}</div>
              <pre className="whitespace-pre-wrap text-slate-500">
                {JSON.stringify(event.payload, null, 2)}
              </pre>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
