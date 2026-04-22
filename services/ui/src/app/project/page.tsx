import Link from "next/link";

import AppShell from "../components/AppShell";

const projectCards = [
  {
    href: "/workflows",
    eyebrow: "Saved Workflows",
    title: "Saved Workflows",
    description:
      "Manage reusable workflows, versions, triggers, and published automations.",
    cta: "Open Workflows",
  },
  {
    href: "/studio",
    eyebrow: "Workflow Builder",
    title: "Workflow Builder",
    description:
      "Design the steps, decisions, tools, and AI actions your process needs.",
    cta: "Open Builder",
  },
];

export default function ProjectPage() {
  return (
    <AppShell
      activeScreen="project"
      title="Project Workspace"
      breadcrumbs={[{ label: "Project" }]}
      actions={
        <>
          <Link
            href="/workflows"
            className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
          >
            Saved Workflows
          </Link>
          <Link
            href="/studio"
            className="rounded-xl border border-slate-200/18 bg-slate-950/25 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-white transition hover:border-white/30 hover:bg-slate-950/35"
          >
            Open Builder
          </Link>
        </>
      }
    >
      <section className="relative">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-100/72">
              Project
            </div>
            <h2 className="mt-1 text-[30px] font-semibold tracking-[-0.03em] text-white">
              Workflow Management
            </h2>
            <p className="mt-1 max-w-3xl text-[13px] leading-5 text-slate-200/74">
              Move between saved automations and the builder for designing business workflows.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]">
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              compose
            </span>
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              chat
            </span>
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              workflows
            </span>
            <span className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-slate-100">
              studio
            </span>
          </div>
        </div>

        <div className="mt-5 grid gap-4 xl:grid-cols-2">
          {projectCards.map((card) => (
            <Link
              key={card.href}
              href={card.href}
              className="group rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(63,78,95,0.62),rgba(37,49,62,0.82))] p-5 shadow-[0_24px_60px_rgba(15,23,42,0.18),inset_0_1px_0_rgba(255,255,255,0.05)] transition hover:border-sky-300/28 hover:bg-[linear-gradient(180deg,rgba(71,88,106,0.66),rgba(42,56,70,0.86))]"
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-100/72">
                    {card.eyebrow}
                  </div>
                  <h3 className="mt-2 text-[26px] font-semibold tracking-[-0.03em] text-white">
                    {card.title}
                  </h3>
                </div>
              </div>
              <p className="mt-3 max-w-xl text-sm leading-6 text-slate-300/82">
                {card.description}
              </p>
              <div className="mt-5 inline-flex items-center rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition group-hover:border-sky-300/30 group-hover:bg-white/[0.08]">
                {card.cta}
              </div>
            </Link>
          ))}
        </div>
      </section>
    </AppShell>
  );
}
