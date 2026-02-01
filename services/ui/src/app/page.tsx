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

type EventEnvelope = {
  type: string;
  payload: Record<string, unknown>;
};

export default function Home() {
  const [goal, setGoal] = useState("");
  const [contextJson, setContextJson] = useState("{}");
  const [priority, setPriority] = useState(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [events, setEvents] = useState<EventEnvelope[]>([]);

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
            </li>
          ))}
        </ul>
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
