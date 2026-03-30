## Chat Clarification Mapper Implementation Plan

### Goal
Add a dedicated LLM-backed clarification-answer mapper that converts pending clarification answers into canonical slot updates without rewriting the original user goal.

### Why
Current chat clarification handling is vulnerable to:
- short answers like `yes`, `sure`, `scalability`, and `any` being treated as new goals
- clarification loops losing the original request context
- pending required fields staying unresolved because answers are not mapped to canonical capability fields

### Design Principle
This is not a free-form goal rewriting step.

The mapper should:
- read the original goal
- read the current pending clarification state
- map the latest answer onto the currently pending fields
- return typed slot updates only

The mapper should not:
- replace the original goal
- invent new fields
- submit jobs directly

### Phase 1
Add typed contracts only.

Files:
- `libs/core/chat_contracts.py`
- `libs/core/tests/test_chat_contracts.py`

Contracts:
- `ClarificationResolvedField`
- `ClarificationPendingState`
- `ClarificationMappingRequest`
- `ClarificationMappingResult`

Required fields:
- original goal
- latest answer
- pending questions
- pending fields
- known slot values
- candidate capabilities
- resolved fields
- remaining fields
- confidence by field
- user-intent-changed flag
- auto-path-allowed flag

### Phase 2
Add a dedicated mapper runtime module.

File:
- `services/api/app/chat_clarification_mapper.py`

Responsibilities:
- accept the current pending clarification state plus latest answer
- call a fast LLM once
- return canonical slot updates only

### Phase 3
Persist structured clarification state.

Files:
- `services/api/app/chat_service.py`
- `services/api/app/context_service.py`

Store:
- original goal
- clarification answers
- known slot values
- pending fields
- answered fields
- question history

### Phase 4
Run the mapper only when `pending_clarification` exists.

Files:
- `services/api/app/chat_service.py`
- `services/api/app/main.py`

Behavior:
- map latest answer to pending fields
- merge high-confidence slot updates
- recompute remaining required fields
- only submit when required fields are satisfied

### Phase 5
Support chat-mode auto-path consent.

Files:
- `services/api/app/chat_clarification_mapper.py`
- `services/api/app/main.py`

Examples:
- `any`
- `auto`
- `you choose`
- `any filename`

Should map to:
- `auto_path_allowed = true`

### Guardrails
- only currently pending fields may be filled
- only capability-approved chat-collectible fields may be filled
- low-confidence results keep clarification open
- mapper exceptions fail closed
- original goal remains the source of truth

### Acceptance Criteria
- clarification answers do not overwrite the original goal
- short answers map to canonical slot values
- `any` can authorize auto filename derivation in chat mode
- pending clarification stays on the execution path
- jobs are submitted only after required fields are resolved
