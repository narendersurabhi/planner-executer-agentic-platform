"use client";

import type React from "react";

import {
  WorkflowNodePlateIcon,
  type WorkflowNodeVisual,
} from "./WorkflowNodeIcon";

export type WorkflowNodeCardTone = "slate" | "sky" | "emerald" | "amber" | "rose" | "steel";
export type WorkflowNodeCardPortTone = "default" | "success" | "danger";

export type WorkflowNodeCardPort = {
  key: string;
  label: string;
  tone: WorkflowNodeCardPortTone;
  top: number;
  active?: boolean;
  title?: string;
  onMouseDown?: (event: React.MouseEvent<HTMLButtonElement>) => void;
};

type WorkflowNodeCardBadge = {
  background: string;
  color: string;
  label: string;
  title?: string;
};

type WorkflowNodeCardProps = {
  title: string;
  subtitle: string;
  caption?: string;
  visual: WorkflowNodeVisual;
  tone: WorkflowNodeCardTone;
  width: number;
  height: number;
  borderColor: string;
  shadow: string;
  badge?: WorkflowNodeCardBadge;
  ports?: WorkflowNodeCardPort[];
  inputActive?: boolean;
  className?: string;
  style?: React.CSSProperties;
};

type WorkflowNodeCardToneStyle = {
  background: string;
  border: string;
  caption: string;
  shell: string;
  shadow: string;
  subtitle: string;
  title: string;
};

export const workflowNodeCardToneStyles: Record<
  WorkflowNodeCardTone,
  WorkflowNodeCardToneStyle
> = {
  slate: {
    background: "linear-gradient(180deg, #3c495a 0%, #2f3947 100%)",
    border: "#63778e",
    caption: "rgba(214, 228, 241, 0.78)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02))",
    shadow: "0 14px 30px rgba(15, 23, 42, 0.28)",
    subtitle: "rgba(226, 232, 240, 0.72)",
    title: "#f8fafc",
  },
  sky: {
    background: "linear-gradient(180deg, #304c67 0%, #253e57 100%)",
    border: "#58a3d6",
    caption: "rgba(205, 235, 255, 0.82)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.015))",
    shadow: "0 16px 32px rgba(37, 99, 235, 0.16)",
    subtitle: "rgba(198, 227, 248, 0.76)",
    title: "#f8fafc",
  },
  emerald: {
    background: "linear-gradient(180deg, #254a48 0%, #1c3837 100%)",
    border: "#40a484",
    caption: "rgba(190, 242, 218, 0.82)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.015))",
    shadow: "0 16px 32px rgba(16, 185, 129, 0.14)",
    subtitle: "rgba(190, 242, 218, 0.76)",
    title: "#f8fafc",
  },
  amber: {
    background: "linear-gradient(180deg, #4f4638 0%, #3f372b 100%)",
    border: "#cba257",
    caption: "rgba(253, 230, 138, 0.84)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.015))",
    shadow: "0 16px 32px rgba(245, 158, 11, 0.14)",
    subtitle: "rgba(253, 230, 138, 0.76)",
    title: "#f8fafc",
  },
  rose: {
    background: "linear-gradient(180deg, #523541 0%, #422936 100%)",
    border: "#d1778f",
    caption: "rgba(254, 205, 211, 0.84)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.015))",
    shadow: "0 16px 32px rgba(244, 63, 94, 0.16)",
    subtitle: "rgba(254, 205, 211, 0.78)",
    title: "#fff7f7",
  },
  steel: {
    background: "linear-gradient(180deg, #3e4b63 0%, #2f394c 100%)",
    border: "#8297bb",
    caption: "rgba(219, 234, 254, 0.82)",
    shell:
      "linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.015))",
    shadow: "0 16px 32px rgba(71, 85, 105, 0.18)",
    subtitle: "rgba(219, 234, 254, 0.74)",
    title: "#f8fafc",
  },
};

export const workflowNodeCardPortToneStyles: Record<
  WorkflowNodeCardPortTone,
  { background: string; border: string; text: string }
> = {
  default: {
    background: "#d7e9fb",
    border: "#6fb7ea",
    text: "#215e90",
  },
  success: {
    background: "#93ecac",
    border: "#2fa85b",
    text: "#14532d",
  },
  danger: {
    background: "#f7a6b1",
    border: "#d65c71",
    text: "#881337",
  },
};

const hexToRgba = (hex: string, alpha: number) => {
  const normalized = hex.replace("#", "");
  const expanded =
    normalized.length === 3
      ? normalized
          .split("")
          .map((char) => `${char}${char}`)
          .join("")
      : normalized;
  const value = Number.parseInt(expanded, 16);
  if (!Number.isFinite(value)) {
    return `rgba(148, 163, 184, ${alpha})`;
  }
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export const workflowNodeCardToneForVisual = (
  visual: WorkflowNodeVisual
): WorkflowNodeCardTone => {
  if (visual.tone === "llm" || visual.tone === "io") {
    return "sky";
  }
  if (visual.tone === "transform") {
    return "emerald";
  }
  if (visual.tone === "validate") {
    return "rose";
  }
  if (visual.tone === "memory") {
    return "steel";
  }
  if (visual.tone === "control" || visual.tone === "render") {
    return "amber";
  }
  return "slate";
};

export default function WorkflowNodeCard({
  title,
  subtitle,
  caption,
  visual,
  tone,
  width,
  height,
  borderColor,
  shadow,
  badge,
  ports = [],
  inputActive = false,
  className = "",
  style,
}: WorkflowNodeCardProps) {
  const toneStyle = workflowNodeCardToneStyles[tone];

  return (
    <div
      className={`relative overflow-visible rounded-[18px] border ${className}`.trim()}
      style={{
        width,
        height,
        borderColor,
        background: toneStyle.background,
        boxShadow: shadow,
        ...style,
      }}
    >
      <div
        className="pointer-events-none absolute inset-[1px] rounded-[17px]"
        style={{ background: toneStyle.shell }}
      />

      <div
        className="absolute -left-[6px] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border"
        style={{
          borderColor: inputActive ? "#22c55e" : "rgba(255,255,255,0.54)",
          background: inputActive ? "#86efac" : "rgba(241, 245, 249, 0.88)",
          boxShadow: inputActive
            ? "0 0 0 4px rgba(34, 197, 94, 0.14)"
            : "0 0 0 4px rgba(15, 23, 42, 0.08)",
        }}
      />

      {badge ? (
        <div
          className="absolute right-3 top-3 flex h-6 min-w-6 items-center justify-center rounded-full px-1.5 text-[11px] font-bold"
          style={{
            background: badge.background,
            color: badge.color,
          }}
          title={badge.title}
        >
          {badge.label}
        </div>
      ) : null}

      <div className="relative flex h-full flex-col px-4 py-3">
        <div className="flex items-start gap-3 pr-16">
          <WorkflowNodePlateIcon visual={visual} size={46} />
          <div className="min-w-0 flex-1">
            <div
              className="truncate text-[13px] font-semibold tracking-[-0.01em]"
              style={{ color: toneStyle.title }}
            >
              {title}
            </div>
            <div
              className="mt-0.5 truncate text-[11px] leading-5"
              style={{ color: toneStyle.subtitle }}
            >
              {subtitle}
            </div>
          </div>
        </div>

        {caption ? (
          <div
            className="mt-auto pr-16 text-[11px] font-medium"
            style={{ color: toneStyle.caption }}
          >
            {caption}
          </div>
        ) : null}
      </div>

      {ports.map((port) => {
        const portTone = workflowNodeCardPortToneStyles[port.tone];

        return (
          <div
            key={port.key}
            className="absolute right-3 flex items-center gap-2"
            style={{ top: port.top, transform: "translateY(-50%)" }}
          >
            <span className="text-[11px] font-medium" style={{ color: portTone.text }}>
              {port.label}
            </span>
            {port.onMouseDown ? (
              <button
                type="button"
                className="flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-semibold transition"
                style={{
                  borderColor: port.active ? "#111827" : portTone.border,
                  background: port.active ? portTone.border : portTone.background,
                  color: port.active ? "#ffffff" : portTone.text,
                  boxShadow: port.active
                    ? `0 0 0 4px ${hexToRgba(portTone.border, 0.24)}`
                    : "none",
                }}
                title={port.title}
                onMouseDown={port.onMouseDown}
              >
                +
              </button>
            ) : (
              <span
                className="flex h-5 w-5 items-center justify-center rounded-full border text-[10px] font-semibold"
                style={{
                  borderColor: portTone.border,
                  background: portTone.background,
                  color: portTone.text,
                }}
              >
                +
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
