"use client";

import type { CapabilityItem } from "./types";
import { getCapabilityRequiredInputs } from "./utils";

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
}: StudioCapabilityPaletteProps) {
  return (
    <aside className="rounded-[28px] border border-slate-200 bg-white/90 p-4 shadow-sm backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">
            Capability Palette
          </div>
          <h2 className="mt-1 font-display text-2xl text-slate-900">Build From Catalog</h2>
        </div>
        <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] text-slate-600">
          {capabilities.length} visible
        </div>
      </div>

      <div className="mt-4 grid gap-3">
        <label className="block">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Search
          </div>
          <input
            className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="document, github, validate..."
          />
        </label>
        <label className="block">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
            Group
          </div>
          <select
            className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
            value={selectedGroup}
            onChange={(event) => onGroupChange(event.target.value)}
          >
            <option value="all">All groups</option>
            {groups.map((group) => (
              <option key={`studio-group-${group}`} value={group}>
                {group}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading ? <div className="mt-4 text-sm text-slate-500">Loading capability catalog...</div> : null}
      {error ? <div className="mt-4 rounded-xl bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</div> : null}

      <div className="mt-4 space-y-3">
        {capabilities.length === 0 && !loading ? (
          <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
            No capabilities match this filter.
          </div>
        ) : null}
        {capabilities.map((item) => {
          const requiredInputs = getCapabilityRequiredInputs(item);
          return (
            <div
              key={`studio-capability-${item.id}`}
              className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-slate-900">{item.id}</div>
                  <div className="mt-1 text-xs text-slate-500">
                    {[item.group || "ungrouped", item.subgroup || null].filter(Boolean).join(" / ")}
                  </div>
                </div>
                <button
                  className="shrink-0 rounded-full border border-slate-300 bg-white px-3 py-1 text-[11px] font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900"
                  onClick={() => onAddCapability(item.id)}
                  disabled={!item.enabled}
                >
                  Add
                </button>
              </div>
              {item.description ? (
                <div className="mt-2 line-clamp-3 text-sm text-slate-600">{item.description}</div>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <span className="rounded-full bg-slate-900 px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] text-white">
                  {item.risk_tier || "unknown"}
                </span>
                <span className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[10px] uppercase tracking-[0.16em] text-slate-600">
                  required {requiredInputs.length}
                </span>
                {item.tags.slice(0, 3).map((tag) => (
                  <span
                    key={`studio-capability-tag-${item.id}-${tag}`}
                    className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[10px] text-slate-600"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
