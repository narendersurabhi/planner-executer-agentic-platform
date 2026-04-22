"use client";

import ScreenHeader from "./ScreenHeader";
import WorkflowNodeCard, {
  workflowNodeCardToneForVisual,
  workflowNodeCardToneStyles,
} from "./workflow/WorkflowNodeCard";
import { resolveWorkflowNodeVisual } from "./workflow/WorkflowNodeIcon";

type GalleryNode = {
  id: string;
  title: string;
  subtitle: string;
  caption: string;
  capabilityId: string;
  nodeKind?: "capability" | "control";
  controlKind?: "if" | "if_else" | "switch" | "parallel" | null;
  width: number;
  height: number;
  x: number;
  y: number;
  badge: {
    label: string;
    background: string;
    color: string;
  };
  ports?: Array<{
    key: string;
    label: string;
    tone: "default" | "success" | "danger";
    top: number;
  }>;
};

type GalleryEdge = {
  id: string;
  path: string;
  color?: string;
  label?: string;
  labelX?: number;
  labelY?: number;
};

const featuredNodes: GalleryNode[] = [
  {
    id: "generate",
    title: "Generate",
    subtitle: "LLM Request",
    caption: "Completed",
    capabilityId: "llm.generate",
    width: 276,
    height: 124,
    x: 28,
    y: 42,
    badge: {
      label: "✓",
      background: "rgba(59, 130, 246, 0.86)",
      color: "#eff6ff",
    },
    ports: [{ key: "output", label: "Output", tone: "default", top: 82 }],
  },
  {
    id: "if-else",
    title: "If/Else",
    subtitle: "Logic Gate",
    caption: "",
    capabilityId: "workflow.control",
    nodeKind: "control",
    controlKind: "if_else",
    width: 276,
    height: 124,
    x: 340,
    y: 42,
    badge: {
      label: "↺",
      background: "rgba(202, 138, 4, 0.88)",
      color: "#fff7ed",
    },
    ports: [
      { key: "true", label: "True", tone: "success", top: 62 },
      { key: "false", label: "False", tone: "danger", top: 94 },
    ],
  },
  {
    id: "validation",
    title: "Validation",
    subtitle: "Schema Check",
    caption: "Failed: Missing Fields",
    capabilityId: "validation.schema",
    width: 276,
    height: 124,
    x: 652,
    y: 42,
    badge: {
      label: "!",
      background: "rgba(220, 38, 38, 0.9)",
      color: "#fff1f2",
    },
    ports: [{ key: "result", label: "Result", tone: "danger", top: 82 }],
  },
  {
    id: "memory",
    title: "Memory",
    subtitle: "State Storage",
    caption: "Pending: Persisting",
    capabilityId: "memory.write",
    width: 276,
    height: 124,
    x: 964,
    y: 42,
    badge: {
      label: "◌",
      background: "rgba(148, 163, 184, 0.3)",
      color: "#f8fafc",
    },
    ports: [{ key: "state", label: "Updated State", tone: "default", top: 82 }],
  },
];

const flowNodes: GalleryNode[] = [
  {
    id: "flow-generate",
    title: "Generate",
    subtitle: "LLM Request",
    caption: "",
    capabilityId: "llm.generate",
    width: 220,
    height: 96,
    x: 260,
    y: 320,
    badge: {
      label: "✓",
      background: "rgba(59, 130, 246, 0.86)",
      color: "#eff6ff",
    },
    ports: [{ key: "output", label: "Output", tone: "default", top: 48 }],
  },
  {
    id: "flow-validate",
    title: "Validation",
    subtitle: "Schema Check",
    caption: "",
    capabilityId: "validation.schema",
    width: 220,
    height: 96,
    x: 540,
    y: 320,
    badge: {
      label: "!",
      background: "rgba(220, 38, 38, 0.9)",
      color: "#fff1f2",
    },
    ports: [
      { key: "true", label: "True", tone: "success", top: 48 },
      { key: "false", label: "False", tone: "danger", top: 66 },
    ],
  },
  {
    id: "flow-memory",
    title: "Memory",
    subtitle: "State Storage",
    caption: "",
    capabilityId: "memory.write",
    width: 220,
    height: 96,
    x: 820,
    y: 260,
    badge: {
      label: "◌",
      background: "rgba(148, 163, 184, 0.3)",
      color: "#f8fafc",
    },
  },
  {
    id: "flow-control",
    title: "Control",
    subtitle: "If/Else",
    caption: "",
    capabilityId: "workflow.control",
    nodeKind: "control",
    controlKind: "if_else",
    width: 220,
    height: 96,
    x: 820,
    y: 390,
    badge: {
      label: "↺",
      background: "rgba(202, 138, 4, 0.88)",
      color: "#fff7ed",
    },
  },
];

const flowEdges: GalleryEdge[] = [
  { id: "generate-validate", path: "M 480 368 C 520 368, 520 368, 540 368" },
  {
    id: "validate-memory",
    path: "M 760 368 C 805 368, 792 308, 820 308",
    color: "#7dd3fc",
    label: "Valid",
    labelX: 782,
    labelY: 326,
  },
  {
    id: "validate-control",
    path: "M 760 384 C 805 384, 792 438, 820 438",
    color: "#f49cae",
    label: "Invalid",
    labelX: 786,
    labelY: 452,
  },
];

const allNodes = [...featuredNodes, ...flowNodes];

const galleryEdgeLabelTheme = (label?: string) => {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized.includes("invalid") || normalized.includes("false")) {
    return {
      background: "rgba(95, 31, 52, 0.94)",
      border: "rgba(251, 113, 133, 0.4)",
      text: "#ffe4e6",
    };
  }
  return {
    background: "rgba(31, 74, 60, 0.94)",
    border: "rgba(74, 222, 128, 0.38)",
    text: "#dcfce7",
  };
};

export default function ComponentsGalleryPage() {
  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.12),transparent_28%),linear-gradient(180deg,#f8fafc_0%,#eff4fb_100%)] px-6 py-6 text-slate-950">
      <ScreenHeader
        eyebrow="Agentic Workflow Studio"
        title="Node Component System Gallery"
        description="Detailed views of shared node variants for the AI workflow builder. This page is the visual calibration surface for the Studio graph."
        activeScreen="components"
        compact
      />

      <section className="mt-6 overflow-hidden rounded-[30px] border border-slate-900/10 bg-[#07111d] shadow-[0_28px_80px_rgba(15,23,42,0.22)]">
        <div className="border-b border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(10,16,28,0.94))] px-6 py-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-sky-200/78">
            Node Showcase
          </div>
          <h2 className="mt-2 text-4xl font-semibold tracking-[-0.04em] text-white md:text-5xl">
            Shared Workflow Node Variants
          </h2>
          <p className="mt-3 max-w-3xl text-base text-slate-300/78">
            Single-source visuals for LLM, control, validation, and memory nodes. The Studio
            canvas and future workflow gallery should now render from this same card system.
          </p>
        </div>

        <div className="relative overflow-hidden px-8 py-8 [background-image:linear-gradient(rgba(59,130,246,0.10)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.10)_1px,transparent_1px),radial-gradient(circle_at_14%_20%,rgba(186,230,253,0.20),transparent_16%),radial-gradient(circle_at_82%_18%,rgba(187,247,208,0.18),transparent_10%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(3,10,24,0.98))] [background-size:24px_24px,24px_24px,100%_100%,100%_100%,100%_100%]">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_12%_18%,rgba(191,219,254,0.18),transparent_10%),radial-gradient(circle_at_86%_24%,rgba(167,243,208,0.14),transparent_8%),radial-gradient(circle_at_78%_72%,rgba(56,189,248,0.08),transparent_16%)]" />

          <div className="relative mx-auto min-h-[700px] max-w-[1280px]">
            {allNodes.map((node) => {
              const visual = resolveWorkflowNodeVisual({
                capabilityId: node.capabilityId,
                controlKind: node.controlKind,
                nodeKind: node.nodeKind,
                taskName: node.title,
              });
              const tone = workflowNodeCardToneForVisual(visual);

              return (
                <div
                  key={node.id}
                  className="absolute"
                  style={{ left: node.x, top: node.y }}
                >
                  <WorkflowNodeCard
                    title={node.title}
                    subtitle={node.subtitle}
                    caption={node.caption || undefined}
                    visual={visual}
                    tone={tone}
                    width={node.width}
                    height={node.height}
                    borderColor={workflowNodeCardToneStyles[tone].border}
                    shadow={workflowNodeCardToneStyles[tone].shadow}
                    badge={node.badge}
                    ports={node.ports}
                  />
                </div>
              );
            })}

            <svg
              className="pointer-events-none absolute inset-0 h-full w-full"
              viewBox="0 0 1280 700"
              fill="none"
            >
              <defs>
                <marker
                  id="gallery-arrow"
                  markerWidth="10"
                  markerHeight="10"
                  refX="8"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L0,6 L9,3 z" fill="#74c0f5" />
                </marker>
              </defs>

              {flowEdges.map((edge) => (
                <g key={edge.id}>
                  <path
                    d={edge.path}
                    stroke="rgba(15,23,42,0.42)"
                    strokeWidth="4.8"
                    strokeLinecap="round"
                  />
                  <path
                    d={edge.path}
                    stroke={edge.color || "#74c0f5"}
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    markerEnd="url(#gallery-arrow)"
                  />
                  {edge.label && edge.labelX && edge.labelY ? (
                    <g>
                      <rect
                        x={edge.labelX - Math.max(28, edge.label.length * 4 + 12)}
                        y={edge.labelY - 14}
                        width={Math.max(56, edge.label.length * 8 + 16)}
                        height="22"
                        rx="10"
                        fill={galleryEdgeLabelTheme(edge.label).background}
                        stroke={galleryEdgeLabelTheme(edge.label).border}
                      />
                      <text
                        x={edge.labelX}
                        y={edge.labelY}
                        textAnchor="middle"
                        fontSize="10"
                        fill={galleryEdgeLabelTheme(edge.label).text}
                      >
                        {edge.label}
                      </text>
                    </g>
                  ) : null}
                </g>
              ))}
            </svg>

            <div className="absolute left-[220px] top-[250px] h-[290px] w-[780px] rounded-[26px] border border-white/10 bg-white/[0.02] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]" />
          </div>
        </div>
      </section>
    </main>
  );
}
