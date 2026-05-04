"use client";

import { Suspense } from "react";

import { LegacyAwareHomeContent } from "./WorkspaceSurfaceContent";

export default function Home() {
  return (
    <Suspense fallback={<main className="min-h-screen bg-slate-50" />}>
      <LegacyAwareHomeContent />
    </Suspense>
  );
}
