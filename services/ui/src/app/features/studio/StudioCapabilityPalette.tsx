"use client";

import { useMemo, useState } from "react";

import {
  WorkflowNodePlateIcon,
  resolveWorkflowNodeVisual,
} from "../../components/workflow/WorkflowNodeIcon";
import type { CapabilityItem, StudioControlKind } from "./types";
import { getCapabilityRequiredInputs, taskNameFromCapability } from "./utils";

type StudioCapabilityPaletteProps = {
  capabilities: CapabilityItem[];
  groups: string[];
  loading: boolean;
  error: string | null;
  query: string;
  selectedGroup: string;
  onQueryChange: (value: string) => void;
  onGroupChange: (value: string) => void;
  onAddCapability: (capabilityId: string) => void;
  onAddControl: (kind: StudioControlKind) => void;
};

type PaletteControlItem = {
  description: string;
  kind: StudioControlKind;
  title: string;
};

type PaletteSection =
  | {
      id: string;
      kind: "control";
      title: string;
      items: PaletteControlItem[];
    }
  | {
      id: string;
      kind: "capability";
      title: string;
      items: CapabilityItem[];
    };

const inputClassName =
  "w-full rounded-xl border border-white/10 bg-slate-950/20 px-3 py-2 text-sm text-white outline-none transition placeholder:text-slate-300/45 focus:border-sky-300/35 focus:bg-slate-950/28";

const panelClassName =
  "rounded-[16px] border border-white/10 bg-[linear-gradient(180deg,rgba(26,36,48,0.78),rgba(35,46,59,0.86))] p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]";

const formatPaletteLabel = (value: string) => {
  const acronyms = new Set(["ai", "api", "csv", "html", "json", "llm", "pdf", "sql", "ui", "url", "xml"]);
  return value
    .trim()
    .split(/[\s/_-]+/)
    .filter(Boolean)
    .map((segment) => {
      const lower = segment.toLowerCase();
      if (acronyms.has(lower)) {
        return lower.toUpperCase();
      }
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
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

export default function StudioCapabilityPalette({
  capabilities,
  groups,
  loading,
  error,
  query,
  selectedGroup,
  onQueryChange,
  onGroupChange,
  onAddCapability,
  onAddControl,
}: StudioCapabilityPaletteProps) {
  const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());

  const normalizedQuery = query.trim().toLowerCase();
  const hasFilters = normalizedQuery.length > 0 || selectedGroup !== "all";
  const controlNodes: PaletteControlItem[] = [
    { kind: "if", title: "If", description: "Single-condition branch gate." },
    { kind: "if_else", title: "If / Else", description: "True/false branch split." },
    { kind: "switch", title: "Switch", description: "Route by named cases." },
    { kind: "parallel", title: "Parallel", description: "Fan-out or fan-in branches." },
  ];

  const visibleControlNodes = useMemo(
    () =>
      normalizedQuery
        ? controlNodes.filter((item) =>
            [item.title, item.description, item.kind].join(" ").toLowerCase().includes(normalizedQuery)
          )
        : controlNodes,
    [normalizedQuery]
  );

  const capabilitySections = useMemo(() => {
    const grouped = new Map<string, CapabilityItem[]>();
    capabilities.forEach((item) => {
      const groupName = item.group?.trim() || "Ungrouped";
      const items = grouped.get(groupName) || [];
      items.push(item);
      grouped.set(groupName, items);
    });
    return Array.from(grouped.entries())
      .map(([groupName, items]) => ({
        id: `group-${groupName.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
        title: formatPaletteLabel(groupName),
        items: [...items].sort((left, right) => left.id.localeCompare(right.id)),
      }))
      .sort((left, right) => left.title.localeCompare(right.title));
  }, [capabilities]);

  const sections: PaletteSection[] = useMemo(() => {
    const next: PaletteSection[] = [];
    if (visibleControlNodes.length > 0) {
      next.push({
        id: "control-flow",
        kind: "control",
        title: "Control Flow",
        items: visibleControlNodes,
      });
    }
    capabilitySections.forEach((section) => {
      next.push({
        id: section.id,
        kind: "capability",
        title: section.title,
        items: section.items,
      });
    });
    return next;
  }, [capabilitySections, visibleControlNodes]);

  const visibleNodeCount = visibleControlNodes.length + capabilities.length;

  const toggleSection = (sectionId: string) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  };

  return (
    <aside className="flex h-full min-h-0 flex-col px-3 py-3 text-slate-100">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[12px] font-semibold uppercase tracking-[0.16em] text-slate-300/76">
            Palette Scope
          </div>
          <div className="mt-1 text-[15px] font-semibold text-white">
            {selectedGroup === "all" ? "All sections" : formatPaletteLabel(selectedGroup)}
          </div>
        </div>
        <div className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-200">
          {visibleNodeCount} nodes
        </div>
      </div>

      <div className="mt-3 grid gap-2.5">
        <input
          className={inputClassName}
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search nodes"
        />
        <div className="flex items-center gap-2">
          <select
            className={`${inputClassName} min-w-0 flex-1 pr-8`}
            value={selectedGroup}
            onChange={(event) => onGroupChange(event.target.value)}
          >
            <option value="all">All groups</option>
            {groups.map((group) => (
              <option key={`studio-group-${group}`} value={group}>
                {formatPaletteLabel(group)}
              </option>
            ))}
          </select>
          {hasFilters ? (
            <button
              type="button"
              className="shrink-0 rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100 transition hover:border-white/18 hover:bg-white/[0.08]"
              onClick={() => {
                onQueryChange("");
                onGroupChange("all");
              }}
            >
              Clear
            </button>
          ) : null}
        </div>
      </div>

      {loading ? (
        <div className="mt-3 rounded-[14px] border border-sky-300/15 bg-sky-400/10 px-3 py-2 text-sm text-sky-100">
          Loading capability catalog...
        </div>
      ) : null}
      {error ? (
        <div className="mt-3 rounded-[14px] border border-rose-300/20 bg-rose-400/10 px-3 py-2 text-sm text-rose-100">
          {error}
        </div>
      ) : null}

      <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
        <div className="space-y-3 pb-1">
          {sections.map((section) => {
            const isCollapsed = collapsedSections.has(section.id);
            const isControlSection = section.kind === "control";
            return (
              <section
                key={section.id}
                className={`${panelClassName} ${
                  isControlSection
                    ? "border-amber-300/18 bg-[linear-gradient(180deg,rgba(120,84,24,0.22),rgba(39,50,63,0.84))]"
                    : ""
                }`}
              >
                <button
                  type="button"
                  className="flex w-full items-center justify-between gap-3 text-left"
                  onClick={() => toggleSection(section.id)}
                >
                  <div>
                    <div
                      className={`text-[11px] font-semibold uppercase tracking-[0.18em] ${
                        isControlSection ? "text-amber-100/88" : "text-slate-300/78"
                      }`}
                    >
                      {section.title}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-200">
                      {section.items.length}
                    </span>
                    <span className="flex h-6 w-6 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-[12px] text-slate-200">
                      {isCollapsed ? "+" : "−"}
                    </span>
                  </div>
                </button>

                {isCollapsed ? null : (
                  <div className="mt-3 space-y-2.5">
                    {section.kind === "control"
                      ? section.items.map((item) => {
                          const visual = resolveWorkflowNodeVisual({
                            nodeKind: "control",
                            controlKind: item.kind,
                            taskName: item.title,
                          });
                          return (
                            <button
                              key={`studio-control-${item.kind}`}
                              type="button"
                              className="group flex w-full items-center gap-3 rounded-[14px] border px-3 py-2.5 text-left transition hover:border-white/22 hover:bg-white/[0.06]"
                              style={{
                                borderColor: hexToRgba(visual.stroke, 0.28),
                                background: `linear-gradient(180deg, ${hexToRgba(
                                  visual.fill,
                                  0.18
                                )} 0%, rgba(26,35,46,0.58) 100%)`,
                              }}
                              onClick={() => onAddControl(item.kind)}
                            >
                              <WorkflowNodePlateIcon visual={visual} size={36} />
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-[13px] font-semibold text-white">
                                  {item.title}
                                </div>
                                <div className="mt-0.5 truncate text-[11px] text-slate-300/76">
                                  {item.description}
                                </div>
                              </div>
                              <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/12 bg-white/[0.04] text-lg leading-none text-slate-100 transition group-hover:border-sky-300/30 group-hover:bg-sky-400/12">
                                +
                              </span>
                            </button>
                          );
                        })
                      : section.items.map((item) => {
                          const requiredInputs = getCapabilityRequiredInputs(item);
                          const visual = resolveWorkflowNodeVisual({
                            capabilityId: item.id,
                            taskName: item.id,
                          });
                          return (
                            <button
                              key={`studio-capability-${item.id}`}
                              type="button"
                              className="group flex w-full items-center gap-3 rounded-[14px] border px-3 py-2.5 text-left transition hover:border-white/22 hover:bg-white/[0.06] disabled:cursor-not-allowed"
                              style={{
                                borderColor: hexToRgba(visual.stroke, item.enabled ? 0.26 : 0.14),
                                background: item.enabled
                                  ? `linear-gradient(180deg, ${hexToRgba(
                                      visual.fill,
                                      0.16
                                    )} 0%, rgba(26,35,46,0.56) 100%)`
                                  : "linear-gradient(180deg, rgba(71,85,105,0.16) 0%, rgba(26,35,46,0.44) 100%)",
                                opacity: item.enabled ? 1 : 0.56,
                              }}
                              onClick={() => onAddCapability(item.id)}
                              disabled={!item.enabled}
                            >
                              <WorkflowNodePlateIcon visual={visual} size={36} />
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-[13px] font-semibold text-white">
                                  {taskNameFromCapability(item.id)}
                                </div>
                                <div className="mt-0.5 truncate text-[11px] text-slate-300/76">
                                  {item.id}
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                {requiredInputs.length > 0 ? (
                                  <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-200">
                                    {requiredInputs.length} req
                                  </span>
                                ) : null}
                                <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/12 bg-white/[0.04] text-lg leading-none text-slate-100 transition group-hover:border-sky-300/30 group-hover:bg-sky-400/12">
                                  +
                                </span>
                              </div>
                            </button>
                          );
                        })}
                  </div>
                )}
              </section>
            );
          })}

          {capabilities.length === 0 && !loading ? (
            <div className="rounded-[16px] border border-dashed border-white/12 bg-white/[0.03] px-4 py-5 text-sm text-slate-300/72">
              No capability nodes match the current filters.
            </div>
          ) : null}
        </div>
      </div>
    </aside>
  );
}
