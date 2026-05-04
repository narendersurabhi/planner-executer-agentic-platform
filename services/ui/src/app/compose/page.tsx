import { Suspense } from "react";

import { WorkspaceSurfaceContent } from "../WorkspaceSurfaceContent";

export default function ComposePage() {
  return (
    <Suspense fallback={<main className="min-h-screen bg-slate-50" />}>
      <WorkspaceSurfaceContent screen="compose" />
    </Suspense>
  );
}
