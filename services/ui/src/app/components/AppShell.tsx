"use client";

import type { ReactNode } from "react";
import Link from "next/link";

import StudioWorkbenchIcon from "../features/studio/StudioWorkbenchIcon";
import { PRIMARY_APP_NAV_ITEMS, type AppScreenId } from "../lib/app-navigation";
import { useAppTheme } from "../lib/theme";

export type AppBreadcrumb = {
  label: string;
  href?: string;
};

export type AppShellProps = {
  activeScreen: AppScreenId;
  title: string;
  breadcrumbs?: AppBreadcrumb[];
  actions?: ReactNode;
  children: ReactNode;
  contentClassName?: string;
};

function ThemeModeIcon({
  theme,
  className = "",
}: {
  theme: "dark" | "light";
  className?: string;
}) {
  if (theme === "light") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-hidden="true" fill="none">
        <path
          d="M12 4.5V2.5M12 21.5v-2M6.7 6.7 5.3 5.3M18.7 18.7l-1.4-1.4M4.5 12h-2M21.5 12h-2M6.7 17.3l-1.4 1.4M18.7 5.3l-1.4 1.4"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.7"
        />
        <circle
          cx="12"
          cy="12"
          r="4.25"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.7"
        />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true" fill="none">
      <path
        d="M14.5 3.5a7.7 7.7 0 1 0 6 12.4 8.8 8.8 0 0 1-6.8-12.4c.2-.4 0-.7-.4-.7-.3 0-.5.1-.8.7Z"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export default function AppShell({
  activeScreen,
  title,
  breadcrumbs = [],
  actions,
  children,
  contentClassName = "px-4 py-4",
}: AppShellProps) {
  const { mounted, theme, toggleTheme } = useAppTheme();
  const isLightTheme = mounted ? theme === "light" : false;

  return (
    <div
      className={`app-shell -mx-6 -my-8 min-h-screen ${
        isLightTheme
          ? "app-shell-light bg-[#eaf1f7] text-slate-900"
          : "app-shell-dark bg-[#4f6274] text-slate-50"
      }`}
      data-app-theme={mounted ? theme : "dark"}
    >
      <div
        className={`min-h-screen ${
          isLightTheme
            ? "bg-[linear-gradient(180deg,#e1e9f2_0px,#e1e9f2_78px,#f5f8fc_78px,#f5f8fc_100%)]"
            : "bg-[linear-gradient(180deg,#435365_0px,#435365_78px,#55697c_78px,#55697c_100%)]"
        }`}
      >
        <header
          className={`px-6 py-3 ${
            isLightTheme
              ? "border-b border-slate-200 bg-[linear-gradient(180deg,rgba(255,255,255,0.94),rgba(241,245,249,0.98))] shadow-[inset_0_-1px_0_rgba(148,163,184,0.16)]"
              : "border-b border-white/10 bg-[linear-gradient(180deg,rgba(67,83,101,0.98),rgba(60,74,90,0.98))] shadow-[inset_0_-1px_0_rgba(255,255,255,0.08)]"
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="min-w-0">
              <div
                className={`truncate text-[22px] font-semibold tracking-[-0.03em] ${
                  isLightTheme ? "text-slate-900" : "text-white"
                }`}
              >
                {title}
              </div>
              {breadcrumbs.length > 0 ? (
                <div
                  className={`mt-1 flex flex-wrap items-center gap-2 text-sm ${
                    isLightTheme ? "text-slate-500" : "text-slate-200/78"
                  }`}
                >
                  {breadcrumbs.map((breadcrumb, index) => (
                    <span key={`${breadcrumb.label}-${index}`} className="contents">
                      {breadcrumb.href ? (
                        <Link
                          href={breadcrumb.href}
                          className={`transition ${
                            isLightTheme ? "hover:text-slate-900" : "hover:text-white"
                          }`}
                        >
                          {breadcrumb.label}
                        </Link>
                      ) : (
                        <span className={isLightTheme ? "text-slate-700" : "text-white/95"}>
                          {breadcrumb.label}
                        </span>
                      )}
                      {index < breadcrumbs.length - 1 ? (
                        <span className={isLightTheme ? "text-slate-400" : "text-white/35"}>
                          ›
                        </span>
                      ) : null}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={toggleTheme}
                className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] transition ${
                  isLightTheme
                    ? "border border-slate-200 bg-white/85 text-slate-700 hover:border-slate-300 hover:bg-white"
                    : "border border-white/12 bg-white/[0.04] text-slate-100 hover:border-sky-300/35 hover:bg-white/[0.08]"
                }`}
                aria-label={isLightTheme ? "Switch to dark mode" : "Switch to light mode"}
              >
                <ThemeModeIcon theme={isLightTheme ? "light" : "dark"} className="h-4 w-4" />
                {isLightTheme ? "Dark Mode" : "Light Mode"}
              </button>
              {actions}
            </div>
          </div>
        </header>

        <div className="grid min-h-[calc(100vh-78px)] grid-cols-[52px_minmax(0,1fr)]">
          <aside
            className={`px-1.5 py-3 ${
              isLightTheme
                ? "border-r border-slate-200 bg-[linear-gradient(180deg,rgba(248,250,252,0.96),rgba(241,245,249,0.98))]"
                : "border-r border-white/10 bg-[linear-gradient(180deg,rgba(49,61,74,0.96),rgba(44,56,69,0.98))]"
            }`}
          >
            <div className="flex h-full flex-col items-center">
              <div className="space-y-3">
                {PRIMARY_APP_NAV_ITEMS.map((item) => (
                  <Link
                    key={item.id}
                    href={item.href}
                    title={item.label}
                    aria-label={item.label}
                    className={`flex h-11 w-11 items-center justify-center rounded-xl border transition ${
                      item.id === activeScreen
                        ? isLightTheme
                          ? "border-sky-300/55 bg-sky-100 text-sky-700 shadow-[0_8px_18px_rgba(14,165,233,0.12)]"
                          : "border-sky-300/35 bg-sky-400/18 text-sky-50 shadow-[0_8px_18px_rgba(14,165,233,0.16)]"
                        : isLightTheme
                          ? "border-slate-200 bg-white/82 text-slate-600 hover:border-slate-300 hover:bg-white"
                          : "border-white/10 bg-slate-950/18 text-slate-200 hover:border-white/18 hover:bg-slate-950/26"
                    }`}
                  >
                    <StudioWorkbenchIcon kind={item.icon} className="h-5 w-5" />
                  </Link>
                ))}
              </div>
            </div>
          </aside>

          <main className={`min-w-0 overflow-auto ${contentClassName}`}>{children}</main>
        </div>
      </div>
    </div>
  );
}
