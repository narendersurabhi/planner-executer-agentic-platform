## Chat Clarification Normalizer Phase 2: Selected Capability Contract

### Goal
Replace the phase-1 document-family heuristic with normalization driven by the selected capability contract.

### Design
1. Use the normalized intent envelope to collect ordered candidate capability IDs.
2. Resolve capability specs from the registry in that order.
3. For each capability, read:
   - required inputs from the input schema
   - `planner_hints.chat_collectible_fields`
4. Select the first capability whose collectible fields are still missing in chat context.
5. Run one JSON-only LLM normalization pass for only that capability and only those missing collectible fields.
6. Merge high-confidence canonical values into chat context before job creation.
7. If required collectible fields remain unresolved, keep the chat in clarification mode instead of submitting a failing job.

### Metadata
Capability entries can opt in via planner hints:
- `chat_collectible_fields`
- `chat_field_examples`
- `chat_field_descriptions`

Phase 2 adds this metadata first for:
- `document.spec.generate`
- `document.spec.generate_from_markdown`
- `document.spec.generate_iterative`
- `document.runbook.generate_iterative`
- `github.issue.search`
- `github.repo.list`

### Guardrails
- Only fields listed in `chat_collectible_fields` are normalized.
- Explicit user values always win.
- Unknown fields are ignored.
- Schema validation remains strict.
- Raw API behavior does not change.

### Acceptance Criteria
- Chat normalization is selected by capability contract, not family heuristics.
- Document flows still work.
- At least one non-document capability flow can normalize a collectible field before submit.
