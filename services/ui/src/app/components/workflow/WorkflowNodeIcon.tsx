"use client";

type WorkflowNodeKind = "capability" | "control";
type WorkflowGlyph =
  | "branch"
  | "code"
  | "memory"
  | "node"
  | "parallel"
  | "render"
  | "search"
  | "spark"
  | "switch"
  | "transform"
  | "validate";

type WorkflowNodeVisualInput = {
  capabilityId?: string | null;
  controlKind?: string | null;
  nodeKind?: WorkflowNodeKind | null;
  taskName?: string | null;
  toolRequests?: string[] | null;
};

export type WorkflowNodeVisual = {
  badgeClassName: string;
  fill: string;
  glyph: WorkflowGlyph;
  iconColor: string;
  label: string;
  stroke: string;
  tone:
    | "code"
    | "control"
    | "default"
    | "io"
    | "llm"
    | "memory"
    | "render"
    | "transform"
    | "validate";
};

const VISUAL_TONES = {
  control: {
    badgeClassName: "border-amber-200 bg-amber-50 text-amber-700",
    fill: "#fef3c7",
    iconColor: "#b45309",
    stroke: "#fcd34d",
  },
  code: {
    badgeClassName: "border-slate-300 bg-slate-100 text-slate-700",
    fill: "#e2e8f0",
    iconColor: "#334155",
    stroke: "#94a3b8",
  },
  default: {
    badgeClassName: "border-slate-200 bg-white text-slate-700",
    fill: "#f8fafc",
    iconColor: "#475569",
    stroke: "#cbd5e1",
  },
  io: {
    badgeClassName: "border-cyan-200 bg-cyan-50 text-cyan-700",
    fill: "#cffafe",
    iconColor: "#0e7490",
    stroke: "#67e8f9",
  },
  llm: {
    badgeClassName: "border-sky-200 bg-sky-50 text-sky-700",
    fill: "#e0f2fe",
    iconColor: "#0369a1",
    stroke: "#7dd3fc",
  },
  memory: {
    badgeClassName: "border-teal-200 bg-teal-50 text-teal-700",
    fill: "#ccfbf1",
    iconColor: "#0f766e",
    stroke: "#5eead4",
  },
  render: {
    badgeClassName: "border-rose-200 bg-rose-50 text-rose-700",
    fill: "#ffe4e6",
    iconColor: "#be123c",
    stroke: "#fda4af",
  },
  transform: {
    badgeClassName: "border-orange-200 bg-orange-50 text-orange-700",
    fill: "#ffedd5",
    iconColor: "#c2410c",
    stroke: "#fdba74",
  },
  validate: {
    badgeClassName: "border-emerald-200 bg-emerald-50 text-emerald-700",
    fill: "#dcfce7",
    iconColor: "#15803d",
    stroke: "#86efac",
  },
} as const;

const PLATE_TONES: Record<
  WorkflowNodeVisual["tone"],
  { background: string; border: string; glyph: string; shadow: string }
> = {
  code: {
    background: "linear-gradient(180deg, #56677f 0%, #47586f 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#f8fafc",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.12)",
  },
  control: {
    background: "linear-gradient(180deg, #d39c28 0%, #bc7f14 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#fff7ed",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
  default: {
    background: "linear-gradient(180deg, #627a99 0%, #516882 100%)",
    border: "rgba(255,255,255,0.16)",
    glyph: "#f8fafc",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.12)",
  },
  io: {
    background: "linear-gradient(180deg, #4aa9e5 0%, #3389cc 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#eff6ff",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
  llm: {
    background: "linear-gradient(180deg, #55b2ee 0%, #3e9bd9 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#eff6ff",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
  memory: {
    background: "linear-gradient(180deg, #5d7190 0%, #4a607d 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#eff6ff",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.12)",
  },
  render: {
    background: "linear-gradient(180deg, #d9a33a 0%, #ba7e1b 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#fff7ed",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
  transform: {
    background: "linear-gradient(180deg, #33a773 0%, #24865c 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#ecfdf5",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
  validate: {
    background: "linear-gradient(180deg, #d97a8b 0%, #c55d72 100%)",
    border: "rgba(255,255,255,0.18)",
    glyph: "#fff1f2",
    shadow: "inset 0 1px 0 rgba(255,255,255,0.16)",
  },
};

const glyphForControlKind = (controlKind: string | null | undefined): WorkflowGlyph => {
  if (controlKind === "parallel") {
    return "parallel";
  }
  if (controlKind === "switch") {
    return "switch";
  }
  return "branch";
};

export const resolveWorkflowNodeVisual = (
  input: WorkflowNodeVisualInput
): WorkflowNodeVisual => {
  const normalizedCapability = String(input.capabilityId || "").trim().toLowerCase();
  const normalizedTask = String(input.taskName || "").trim().toLowerCase();
  const normalizedRequests = (input.toolRequests || [])
    .map((value) => String(value || "").trim().toLowerCase())
    .filter(Boolean);
  const searchBlob = [normalizedCapability, normalizedTask, ...normalizedRequests].join(" ");

  if (input.nodeKind === "control") {
    return {
      ...VISUAL_TONES.control,
      glyph: glyphForControlKind(input.controlKind),
      label: input.controlKind ? `Control ${input.controlKind}` : "Control node",
      tone: "control",
    };
  }

  const writesCode =
    searchBlob.includes("codegen") ||
    searchBlob.includes("publish_pr") ||
    searchBlob.includes("create_pull_request") ||
    searchBlob.includes("create_branch") ||
    searchBlob.includes("push_files");
  if (writesCode) {
    return { ...VISUAL_TONES.code, glyph: "code", label: "Code node", tone: "code" };
  }

  if (searchBlob.includes("memory")) {
    return { ...VISUAL_TONES.memory, glyph: "memory", label: "Memory node", tone: "memory" };
  }

  if (
    searchBlob.includes("validate") ||
    searchBlob.includes("critic") ||
    searchBlob.includes("check")
  ) {
    return {
      ...VISUAL_TONES.validate,
      glyph: "validate",
      label: "Validation node",
      tone: "validate",
    };
  }

  if (
    searchBlob.includes("render") ||
    searchBlob.includes("docx") ||
    searchBlob.includes("pdf")
  ) {
    return { ...VISUAL_TONES.render, glyph: "render", label: "Render node", tone: "render" };
  }

  if (
    searchBlob.includes("transform") ||
    searchBlob.includes("repair") ||
    searchBlob.includes("improve")
  ) {
    return {
      ...VISUAL_TONES.transform,
      glyph: "transform",
      label: "Transform node",
      tone: "transform",
    };
  }

  if (
    searchBlob.includes("github") ||
    searchBlob.includes("filesystem") ||
    searchBlob.includes("search") ||
    searchBlob.includes(".list") ||
    searchBlob.includes("read")
  ) {
    return { ...VISUAL_TONES.io, glyph: "search", label: "I/O node", tone: "io" };
  }

  if (
    searchBlob.includes("llm") ||
    searchBlob.includes("generate") ||
    searchBlob.includes("prompt")
  ) {
    return { ...VISUAL_TONES.llm, glyph: "spark", label: "LLM node", tone: "llm" };
  }

  return { ...VISUAL_TONES.default, glyph: "node", label: "Workflow node", tone: "default" };
};

const GlyphPaths = ({
  glyph,
  color,
}: {
  glyph: WorkflowGlyph;
  color: string;
}) => {
  const strokeProps = {
    fill: "none",
    stroke: color,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.8,
  };

  switch (glyph) {
    case "branch":
      return (
        <>
          <path {...strokeProps} d="M8 6v12" />
          <path {...strokeProps} d="M8 10h8" />
          <circle cx="8" cy="6" r="1.6" fill={color} />
          <circle cx="8" cy="18" r="1.6" fill={color} />
          <circle cx="16" cy="10" r="1.6" fill={color} />
        </>
      );
    case "code":
      return (
        <>
          <path {...strokeProps} d="M9 8 6 12l3 4" />
          <path {...strokeProps} d="M15 8l3 4-3 4" />
          <path {...strokeProps} d="m13 5-2 14" />
        </>
      );
    case "memory":
      return (
        <>
          <ellipse cx="12" cy="6.5" rx="5.5" ry="2.5" {...strokeProps} />
          <path {...strokeProps} d="M6.5 6.5v7c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5v-7" />
          <path {...strokeProps} d="M6.5 10c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5" />
          <path {...strokeProps} d="M6.5 13.5c0 1.4 2.5 2.5 5.5 2.5s5.5-1.1 5.5-2.5" />
        </>
      );
    case "parallel":
      return (
        <>
          <path {...strokeProps} d="M7 6v12" />
          <path {...strokeProps} d="M17 6v12" />
          <path {...strokeProps} d="M7 8h10" />
          <path {...strokeProps} d="M7 16h10" />
          <circle cx="7" cy="12" r="1.6" fill={color} />
          <circle cx="17" cy="12" r="1.6" fill={color} />
        </>
      );
    case "render":
      return (
        <>
          <path {...strokeProps} d="M8 5h6l3 3v11H8z" />
          <path {...strokeProps} d="M14 5v4h4" />
          <path {...strokeProps} d="M12 11v5" />
          <path {...strokeProps} d="m9.5 13.5 2.5 2.5 2.5-2.5" />
        </>
      );
    case "search":
      return (
        <>
          <circle cx="10.5" cy="10.5" r="4.5" {...strokeProps} />
          <path {...strokeProps} d="m14 14 4 4" />
        </>
      );
    case "spark":
      return (
        <>
          <path
            d="M12 4.5 13.8 9l4.5 1.8-4.5 1.7-1.8 4.5-1.8-4.5-4.5-1.7L10.2 9 12 4.5Z"
            fill={color}
            opacity="0.88"
          />
          <circle cx="18.2" cy="6.2" r="1.2" fill={color} />
          <circle cx="6.3" cy="17.2" r="1" fill={color} opacity="0.9" />
        </>
      );
    case "switch":
      return (
        <>
          <path {...strokeProps} d="M7 6v12" />
          <path {...strokeProps} d="M7 8h6l-2-2" />
          <path {...strokeProps} d="M7 16h8l-2 2" />
          <circle cx="7" cy="6" r="1.6" fill={color} />
          <circle cx="7" cy="18" r="1.6" fill={color} />
        </>
      );
    case "transform":
      return (
        <>
          <path {...strokeProps} d="M6 8h8" />
          <path {...strokeProps} d="m11 5 3 3-3 3" />
          <path {...strokeProps} d="M18 16h-8" />
          <path {...strokeProps} d="m13 13-3 3 3 3" />
        </>
      );
    case "validate":
      return (
        <>
          <circle cx="12" cy="12" r="6" {...strokeProps} />
          <path {...strokeProps} d="m9.2 12.2 1.9 1.9 3.8-4.3" />
        </>
      );
    case "node":
    default:
      return (
        <>
          <circle cx="12" cy="8" r="1.7" fill={color} />
          <circle cx="8" cy="16" r="1.7" fill={color} />
          <circle cx="16" cy="16" r="1.7" fill={color} />
          <path {...strokeProps} d="M12 9.7v2.1" />
          <path {...strokeProps} d="M10.1 14.8 8.8 16" />
          <path {...strokeProps} d="M13.9 14.8 15.2 16" />
        </>
      );
  }
};

export function WorkflowNodeIcon({
  className = "",
  size = 46,
  visual,
}: {
  className?: string;
  size?: number;
  visual: WorkflowNodeVisual;
}) {
  return (
    <span
      aria-hidden="true"
      className={`inline-flex items-center justify-center rounded-2xl border shadow-sm ${visual.badgeClassName} ${className}`.trim()}
      style={{ width: size, height: size }}
      title={visual.label}
    >
      <svg viewBox="0 0 24 24" width={Math.round(size * 0.52)} height={Math.round(size * 0.52)}>
        <GlyphPaths glyph={visual.glyph} color={visual.iconColor} />
      </svg>
    </span>
  );
}

export function WorkflowNodePlateIcon({
  className = "",
  size = 46,
  visual,
}: {
  className?: string;
  size?: number;
  visual: WorkflowNodeVisual;
}) {
  const tone = PLATE_TONES[visual.tone] || PLATE_TONES.default;
  return (
    <span
      aria-hidden="true"
      className={`inline-flex items-center justify-center rounded-[14px] border ${className}`.trim()}
      style={{
        width: size,
        height: size,
        background: tone.background,
        borderColor: tone.border,
        boxShadow: tone.shadow,
      }}
      title={visual.label}
    >
      <svg viewBox="0 0 24 24" width={Math.round(size * 0.56)} height={Math.round(size * 0.56)}>
        <GlyphPaths glyph={visual.glyph} color={tone.glyph} />
      </svg>
    </span>
  );
}

export function WorkflowNodeSvgIcon({
  size = 32,
  visual,
  x,
  y,
}: {
  size?: number;
  visual: WorkflowNodeVisual;
  x: number;
  y: number;
}) {
  const half = size / 2;
  const glyphScale = size / 24;
  return (
    <g transform={`translate(${x - half}, ${y - half})`}>
      <rect
        width={size}
        height={size}
        rx={Math.max(8, Math.round(size * 0.3))}
        fill={visual.fill}
        stroke={visual.stroke}
        strokeWidth="1.4"
      />
      <g transform={`translate(${(size - 24 * glyphScale) / 2}, ${(size - 24 * glyphScale) / 2}) scale(${glyphScale})`}>
        <GlyphPaths glyph={visual.glyph} color={visual.iconColor} />
      </g>
    </g>
  );
}
