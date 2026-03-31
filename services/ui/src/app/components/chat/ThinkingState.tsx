export function ThinkingState() {
  return (
    <div className="max-w-[92%] rounded-[24px] border border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_rgba(15,23,42,0.92)_55%,_rgba(14,116,144,0.2)_100%)] px-4 py-3 text-sm text-slate-100 shadow-[0_18px_45px_rgba(15,23,42,0.35)]">
      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.18em]">
        <div className="flex items-center gap-2 text-slate-300">
          <span>assistant</span>
          <span className="h-1.5 w-1.5 rounded-full bg-sky-300 animate-thinking-glow" />
        </div>
        <span className="rounded-full border border-sky-300/20 bg-sky-300/10 px-2 py-0.5 text-[9px] font-semibold tracking-[0.12em] text-sky-100">
          Thinking
        </span>
      </div>

      <div className="mt-3 rounded-2xl border border-white/10 bg-slate-950/30 px-3 py-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-cyan-300 thinking-dot thinking-dot-delay-0" />
            <span className="h-2.5 w-2.5 rounded-full bg-sky-300 thinking-dot thinking-dot-delay-1" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-200 thinking-dot thinking-dot-delay-2" />
          </div>
          <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/10">
            <span className="absolute inset-y-0 left-[-35%] w-[38%] rounded-full bg-gradient-to-r from-transparent via-sky-300/80 to-transparent animate-thinking-beam" />
          </div>
        </div>

        <div className="mt-4 space-y-2">
          <div className="h-2.5 w-[78%] rounded-full bg-white/10 thinking-line thinking-line-delay-0" />
          <div className="h-2.5 w-[62%] rounded-full bg-white/10 thinking-line thinking-line-delay-1" />
          <div className="h-2.5 w-[70%] rounded-full bg-white/10 thinking-line thinking-line-delay-2" />
        </div>
      </div>
    </div>
  );
}
