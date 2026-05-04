## Chat Clarification Normalizer Implementation Plan

### Goal
Add a single structured clarification-normalization pass before chat submits a job, starting with document-generation flows.

### Why
Some chat clarification flows reach planning with missing canonical capability inputs such as:
- `instruction`
- `topic`
- `audience`
- `tone`

This creates preventable plan failures for document-generation capabilities.

### Scope
Phase 1 is intentionally narrow:
- run only for chat turns that are about to `submit_job`
- run only when the chat session is already in pending clarification
- support document-generation capabilities only

### Phase 1 Design
1. Detect document-generation submit flows from the normalized intent envelope and candidate capabilities.
2. Collect the canonical missing document fields:
   - `instruction`
   - `topic`
   - `audience`
   - `tone`
3. Run one small JSON-only LLM normalization pass over:
   - draft goal
   - appended clarification text
   - current merged context
   - candidate capabilities
   - missing canonical fields
4. Merge high-confidence normalized values back into chat context before `create_job(...)`.
5. If required document fields are still missing after normalization, ask one more clarification instead of submitting a job that will fail later.

### Guardrails
- Only canonical fields may be returned.
- Explicit user-provided values win over normalized values.
- Unknown keys are ignored.
- Low-confidence fields are not merged.
- Raw API behavior is unchanged.

### Phase 1 Files
- `services/api/app/chat_service.py`
- `services/api/app/main.py`
- `services/api/app/chat_clarification_normalizer.py`
- `services/api/tests/test_chat_api.py`

### Config
- `CHAT_CLARIFICATION_NORMALIZER_ENABLED`
- `CHAT_CLARIFICATION_NORMALIZER_MODEL`
- `CHAT_CLARIFICATION_NORMALIZER_CONFIDENCE_THRESHOLD`

### Phase 1 Acceptance Criteria
- Pending-clarification document requests normalize document fields before job creation.
- Missing `tone`/`topic`/`audience`/`instruction` no longer fall through silently for supported document flows.
- If normalization cannot safely fill the remaining required document fields, chat asks for clarification instead of submitting a failing job.
- Existing non-document and no-clarification chat flows remain unchanged.

### Follow-On Phases
- Expand to GitHub/filesystem scalar fields.
- Add capability-driven `chat_collectible_fields` metadata to the registry.
- Move from phase-1 document heuristics to generalized capability-driven field selection.
