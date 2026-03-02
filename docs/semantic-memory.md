# Semantic Memory Strategy

## What It Stores
`semantic_memory` stores distilled facts as durable knowledge entries for a user scope:
- `namespace`: domain bucket (`runtime`, `architecture`, `user_prefs`, etc.)
- `subject`: entity/topic the fact is about
- `fact`: normalized fact sentence
- `keywords`, `aliases`: retrieval anchors
- `confidence`: 0.0-1.0
- `source`, `source_ref`, optional `reasoning`

## Write Strategy (Distillation)
Write only stable, reusable facts (not transient task state):
1. Extract candidate facts from task outputs or user input.
2. Normalize and deduplicate (same namespace+subject+fact).
3. Persist via `memory.semantic.write`.
4. Keep confidence conservative unless the source is verified.

### Automatic Distillation (Worker)
Worker can auto-distill semantic facts from selected tool outputs after successful execution.
Controls:
- `SEMANTIC_MEMORY_AUTO_WRITE_ENABLED`
- `SEMANTIC_MEMORY_AUTO_WRITE_TOOLS` (glob-like tool/capability patterns)
- `SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACTS`
- `SEMANTIC_MEMORY_AUTO_WRITE_MAX_FACT_CHARS`
- `SEMANTIC_MEMORY_AUTO_WRITE_NAMESPACE`

## Retrieve Strategy (Grounding)
Retrieve facts before planning/tool execution:
1. Query with intent terms via `memory.semantic.search`.
2. Filter by `namespace` and/or `subject` when possible.
3. Inject top matches into context for downstream reasoning.
4. Apply `min_score` threshold to avoid noisy matches.

## API Endpoints
- `POST /memory/semantic/write`
- `POST /memory/semantic/search`

## Capabilities
- `memory.semantic.write`
- `memory.semantic.search`

## UI
In **Job Details -> Memory**, use the **Semantic Memory** block to:
- store distilled facts
- search and inspect ranked semantic matches

## Capability I/O
- Write: `memory.semantic.write`
- Read/Search: `memory.semantic.search`

Typical chain pattern:
1. `memory.semantic.search` with goal/domain query
2. downstream generation/validation/render capabilities
3. `memory.semantic.write` for durable learned facts

## Example Capability Chain
1. `memory.semantic.search` with query from current goal.
2. `document.spec.generate` or `llm.text.generate` grounded by retrieved facts.
3. `memory.semantic.write` to persist new stable facts learned during the run.
