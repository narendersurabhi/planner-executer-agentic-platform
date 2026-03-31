# Chat User Profile and Memory Promotion Implementation Plan

## Goal

Make chat profile-aware without turning raw chat history into long-term memory. Use structured user profile storage for exact preferences and vector-backed semantic memory for fuzzy recall.

## Principles

- `user_id` must come from server auth context, not client payloads.
- `user_profile` is the canonical structured store.
- `semantic_memory` and `interaction_summaries_compact` are retrieval layers.
- Memory promotion is selective, not automatic for every message.
- Sensitive content is redacted or excluded before indexing.
- Memory quality is evaluated separately from model quality.

## Phase 1

- Add typed memory promotion and redaction contracts.
- Add a redaction layer for semantic memory writes.
- Add a promotion helper for explicit stable preferences and semantic facts.
- Route existing intent workflow outcome memory writes through the new helper.
- Keep `user_profile` exact-only for now.
- Do not enable chat-to-profile auto-writes until auth-derived `user_id` is wired into chat.

## Phase 2

- Add a profile service for exact `user_profile` load and update.
- Hydrate profile into chat context at turn start using server-derived `user_id`.
- Promote explicit reusable chat preferences into `user_profile`.
- Upgrade semantic memory retrieval to hybrid lexical plus vector search.

## Phase 3

- Add compact summary and task-pattern indexing.
- Add memory evaluation harness and gold fixtures.
- Add stale-memory, privacy-leak, and retrieval-quality metrics.

## Storage Boundaries

- `user_profile`: exact, structured, user-scoped, not primary vector source.
- `semantic_memory`: exact record plus safe retrieval text.
- `interaction_summaries_compact`: compact session memory, optional vector indexing.
- raw chat messages: remain in chat/session storage.

## Initial Implementation Scope

- save the plan
- implement redaction and promotion primitives
- harden semantic memory writes
- route job outcome semantic persistence through the new layer
- add focused tests
