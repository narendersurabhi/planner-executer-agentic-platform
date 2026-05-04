import { Suspense } from "react";

import WorkflowStudio from "../features/studio/WorkflowStudio";

export default function StudioPage() {
  return (
    <Suspense fallback={null}>
      <WorkflowStudio />
    </Suspense>
  );
}
