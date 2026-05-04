"use client";

import type { ReactNode } from "react";
import { useAppTheme } from "../lib/theme";

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
  const { mounted, theme: appTheme } = useAppTheme();
  const isLightTheme = mounted && appTheme === "light";
  return (
    <section
      className={`screen-header relative overflow-hidden animate-fade-up ${
        isStudioTheme
          ? isLightTheme
            ? "screen-header-studio border border-slate-200 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(241,245,249,0.98))] text-slate-900 shadow-[0_18px_36px_rgba(148,163,184,0.12),inset_0_-1px_0_rgba(255,255,255,0.65)]"
            : "screen-header-studio border border-white/10 bg-[linear-gradient(180deg,rgba(67,83,101,0.98),rgba(60,74,90,0.98))] text-white shadow-[0_18px_36px_rgba(15,23,42,0.2),inset_0_-1px_0_rgba(255,255,255,0.08)]"
          : isLightTheme
            ? "screen-header-default border border-slate-200 bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(238,245,252,0.98))] text-slate-900 shadow-[0_18px_40px_rgba(148,163,184,0.16)]"
            : "screen-header-default bg-gradient-to-br from-stone-950 via-slate-900 to-sky-950 text-white shadow-2xl"
      } ${compact ? "rounded-[30px] px-6 py-5" : "rounded-[36px] px-8 py-8"}`}
    >
      <div
        className={`pointer-events-none absolute rounded-full blur-3xl ${
          isStudioTheme
            ? isLightTheme
              ? "bg-sky-200/30"
              : "bg-white/10"
            : isLightTheme
              ? "bg-amber-200/28"
              : "bg-amber-300/20"
        } ${compact ? "-left-10 top-4 h-28 w-28" : "-left-14 top-8 h-44 w-44"}`}
      />
      <div
        className={`pointer-events-none absolute rounded-full blur-3xl ${
          isStudioTheme
            ? isLightTheme
              ? "bg-sky-300/18"
              : "bg-sky-300/12"
            : isLightTheme
              ? "bg-sky-300/18"
              : "bg-sky-400/25"
        } ${compact ? "-right-10 bottom-0 h-36 w-36" : "-right-12 bottom-0 h-56 w-56"}`}
      />
      <div className={`relative ${compact ? "space-y-4" : "space-y-6"}`}>
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-3xl">
            <div
              className={`text-[11px] font-semibold uppercase tracking-[0.28em] ${
                isStudioTheme
                  ? isLightTheme
                    ? "text-sky-700"
                    : "text-sky-100/78"
                  : isLightTheme
                    ? "text-sky-700"
                    : "text-sky-200"
              }`}
            >
              {eyebrow}
            </div>
            <h1
              className={`mt-2 ${
                isStudioTheme
                  ? compact
                    ? "text-[30px] font-semibold tracking-[-0.03em] md:text-[34px]"
                    : "text-[34px] font-semibold tracking-[-0.03em] md:text-[42px]"
                  : compact
                    ? "font-display text-3xl tracking-tight md:text-4xl"
                    : "font-display text-4xl tracking-tight md:text-5xl"
              }`}
            >
              {title}
            </h1>
            <p
              className={`${
                isStudioTheme
                  ? isLightTheme
                    ? "text-slate-600"
                    : "text-slate-200/78"
                  : isLightTheme
                    ? "text-slate-600"
                    : "text-slate-200"
              } ${
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
