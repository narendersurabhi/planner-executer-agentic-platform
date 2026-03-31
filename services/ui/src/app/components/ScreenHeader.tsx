"use client";

import Link from "next/link";
import type { ReactNode } from "react";

export type ScreenHeaderScreen = "home" | "compose" | "chat" | "studio" | "memory" | "rag";

type ScreenHeaderProps = {
  eyebrow: string;
  title: string;
  description: string;
  activeScreen: ScreenHeaderScreen;
  actions?: ReactNode;
  children?: ReactNode;
};

const NAV_ITEMS: Array<{ id: ScreenHeaderScreen; label: string; href: string }> = [
  { id: "home", label: "Home", href: "/" },
  { id: "compose", label: "Compose", href: "/compose" },
  { id: "chat", label: "Chat", href: "/chat" },
  { id: "studio", label: "Studio", href: "/studio" },
  { id: "memory", label: "Memory", href: "/memory" },
  { id: "rag", label: "RAG", href: "/rag" }
];

export const screenHeaderSecondaryActionClassName =
  "rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50";

export const screenHeaderPrimaryActionClassName =
  "rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50";

export default function ScreenHeader({
  eyebrow,
  title,
  description,
  activeScreen,
  actions,
  children
}: ScreenHeaderProps) {
  return (
    <section className="relative overflow-hidden rounded-[36px] bg-gradient-to-br from-stone-950 via-slate-900 to-sky-950 px-8 py-8 text-white shadow-2xl animate-fade-up">
      <div className="pointer-events-none absolute -left-14 top-8 h-44 w-44 rounded-full bg-amber-300/20 blur-3xl" />
      <div className="pointer-events-none absolute -right-12 bottom-0 h-56 w-56 rounded-full bg-sky-400/25 blur-3xl" />
      <div className="relative space-y-6">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-3xl">
            <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-sky-200">
              {eyebrow}
            </div>
            <h1 className="mt-2 font-display text-4xl tracking-tight md:text-5xl">{title}</h1>
            <p className="mt-3 text-sm leading-6 text-slate-200 md:text-base">{description}</p>
          </div>
          <div className="flex max-w-full flex-col items-start gap-3 md:items-end">
            <div className="flex flex-wrap items-center gap-3">
              {NAV_ITEMS.map((item) =>
                item.id === activeScreen ? (
                  <div
                    key={item.id}
                    className="rounded-full border border-white/20 bg-white px-4 py-2 text-sm font-semibold text-slate-900"
                  >
                    {item.label}
                  </div>
                ) : (
                  <Link
                    key={item.id}
                    href={item.href}
                    className="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15"
                  >
                    {item.label}
                  </Link>
                )
              )}
            </div>
            {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
          </div>
        </div>
        {children ? <div>{children}</div> : null}
      </div>
    </section>
  );
}
