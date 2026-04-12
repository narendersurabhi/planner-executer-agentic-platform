"use client";

import type { ReactNode } from "react";
import Link from "next/link";

import StudioWorkbenchIcon from "../features/studio/StudioWorkbenchIcon";
import { PRIMARY_APP_NAV_ITEMS, type AppScreenId } from "../lib/app-navigation";

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

export default function AppShell({
  activeScreen,
  title,
  breadcrumbs = [],
  actions,
  children,
  contentClassName = "px-4 py-4",
}: AppShellProps) {
  return (
    <div className="-mx-6 -my-8 min-h-screen bg-[#56697c] text-white">
      <div className="min-h-screen bg-[linear-gradient(180deg,#435365_0px,#435365_78px,#55697c_78px,#55697c_100%)]">
        <header className="border-b border-white/10 bg-[linear-gradient(180deg,rgba(67,83,101,0.98),rgba(60,74,90,0.98))] px-6 py-3 shadow-[inset_0_-1px_0_rgba(255,255,255,0.08)]">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="min-w-0">
              <div className="truncate text-[22px] font-semibold tracking-[-0.03em] text-white">
                {title}
              </div>
              {breadcrumbs.length > 0 ? (
                <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-200/78">
                  {breadcrumbs.map((breadcrumb, index) => (
                    <span key={`${breadcrumb.label}-${index}`} className="contents">
                      {breadcrumb.href ? (
                        <Link href={breadcrumb.href} className="transition hover:text-white">
                          {breadcrumb.label}
                        </Link>
                      ) : (
                        <span className="text-white/95">{breadcrumb.label}</span>
                      )}
                      {index < breadcrumbs.length - 1 ? (
                        <span className="text-white/35">›</span>
                      ) : null}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>

            {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
          </div>
        </header>

        <div className="grid min-h-[calc(100vh-78px)] grid-cols-[52px_minmax(0,1fr)]">
          <aside className="border-r border-white/10 bg-[linear-gradient(180deg,rgba(49,61,74,0.96),rgba(44,56,69,0.98))] px-1.5 py-3">
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
                        ? "border-sky-300/35 bg-sky-400/18 text-sky-50 shadow-[0_8px_18px_rgba(14,165,233,0.16)]"
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
