"use client";

import type { ReactNode } from "react";

type ScreenHeaderProps = {
  eyebrow: string;
  title: string;
  description: string;
  activeScreen?: string;
  actions?: ReactNode;
  children?: ReactNode;
  compact?: boolean;
  theme?: "default" | "studio";
};

export const screenHeaderSecondaryActionClassName =
  "rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50";

export const screenHeaderPrimaryActionClassName =
  "rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50";

export default function ScreenHeader({
  eyebrow,
  title,
  description,
  activeScreen: _activeScreen,
  actions,
  children,
  compact = false,
  theme = "default",
}: ScreenHeaderProps) {
  const isStudioTheme = theme === "studio";
  return (
    <section
      className={`relative overflow-hidden text-white animate-fade-up ${
        isStudioTheme
          ? "border border-white/10 bg-[linear-gradient(180deg,rgba(67,83,101,0.98),rgba(60,74,90,0.98))] shadow-[0_18px_36px_rgba(15,23,42,0.2),inset_0_-1px_0_rgba(255,255,255,0.08)]"
          : "bg-gradient-to-br from-stone-950 via-slate-900 to-sky-950 shadow-2xl"
      } ${compact ? "rounded-[30px] px-6 py-5" : "rounded-[36px] px-8 py-8"}`}
    >
      <div
        className={`pointer-events-none absolute rounded-full blur-3xl ${
          isStudioTheme ? "bg-white/10" : "bg-amber-300/20"
        } ${compact ? "-left-10 top-4 h-28 w-28" : "-left-14 top-8 h-44 w-44"}`}
      />
      <div
        className={`pointer-events-none absolute rounded-full blur-3xl ${
          isStudioTheme ? "bg-sky-300/12" : "bg-sky-400/25"
        } ${compact ? "-right-10 bottom-0 h-36 w-36" : "-right-12 bottom-0 h-56 w-56"}`}
      />
      <div className={`relative ${compact ? "space-y-4" : "space-y-6"}`}>
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-3xl">
            <div
              className={`text-[11px] font-semibold uppercase tracking-[0.28em] ${
                isStudioTheme ? "text-sky-100/78" : "text-sky-200"
              }`}
            >
              {eyebrow}
            </div>
            <h1
              className={`mt-2 font-display tracking-tight ${
                compact ? "text-3xl md:text-4xl" : "text-4xl md:text-5xl"
              }`}
            >
              {title}
            </h1>
            <p
              className={`${isStudioTheme ? "text-slate-200/78" : "text-slate-200"} ${
                compact ? "mt-2 text-sm leading-6" : "mt-3 text-sm leading-6 md:text-base"
              }`}
            >
              {description}
            </p>
          </div>
          <div className="flex max-w-full flex-col items-start gap-3 md:items-end">
            {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
          </div>
        </div>
        {children ? <div>{children}</div> : null}
      </div>
    </section>
  );
}
