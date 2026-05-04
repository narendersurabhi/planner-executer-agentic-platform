# Agent Registry Phased Implementation Plan

## Summary

Add an Agent Registry so Agent Sandbox users can create, save, edit, delete, and reuse named agent definitions instead of rebuilding an agent from raw run JSON. Scope v1 to Agent Sandbox / Workbench only: saved agent definitions are reusable launch profiles with policy constraints, while execution continues through the existing Workbench agent-run path.

The design follows common production agent-platform patterns:

- OpenAI-style agent definition: name, instructions, model settings, tools, guardrails, and handoff-ready metadata.
- Azure AI Foundry-style agent resource: persistent profile with instructions, tools, operational metadata, and lifecycle controls.
- Amazon Bedrock-style agent configuration: instructions, action/tool groups, memory/session policy, guardrails, and alias/snapshot behavior.

References:

- OpenAI Agents SDK agents: https://openai.github.io/openai-agents-python/agents/
- OpenAI Agents SDK guardrails: https://openai.github.io/openai-agents-python/guardrails/
- Azure AI Foundry Agent Service: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview
- Amazon Bedrock create agents: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html
- Amazon Bedrock action groups: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-action-create.html
- Amazon Bedrock memory: https://docs.aws.amazon.com/bedrock/latest/userguide/agents-configure-memory.html

## Phase 0 - Contracts And Migration

Goal: introduce the persistent agent-definition contract without changing Agent Sandbox behavior.

Implementation:

- Add an `agent_definitions` SQLAlchemy record in the API model layer and an additive Alembic migration.
- Add shared request/response models for create, update, list, and launch-time snapshots.
- Keep all fields optional only where the UI can produce a safe default; require `name`, `agent_capability_id`, and `instructions`.
- Do not wire the registry into Workbench execution in this phase.

Minimum `agent_definitions` fields:

- `id`: UUID primary key.
- `name`: required display name.
- `description`: optional human-readable purpose.
- `agent_capability_id`: required primary agent capability, defaulting to `codegen.autonomous` in the UI.
- `instructions`: required system/task guidance used to seed the Agent Sandbox instruction field.
- `default_goal`: optional default run goal.
- `default_workspace_path`: optional default workspace path.
- `default_constraints`: string array for reusable constraints.
- `default_max_steps`: integer default for the sandbox run.
- `model_config`: JSON object for provider/model/temperature/max-token style settings.
- `allowed_capability_ids`: string array restricting extra Workbench steps/tools that can be attached to this agent.
- `memory_policy`: JSON object for disabled/session/profile-style memory settings.
- `guardrail_policy`: JSON object for input/output safety, schema, and capability constraints.
- `workspace_policy`: JSON object for allowed workspace roots and write permissions.
- `metadata`: JSON object for labels, owner notes, and UI grouping.
- `enabled`: boolean for soft-disable.
- `user_id`: optional owner until the platform has stronger tenancy boundaries.
- `created_at` / `updated_at`: timestamps.

Acceptance gate:

- Migration upgrade creates only `agent_definitions`.
- Migration downgrade removes only `agent_definitions`.
- Existing Workbench, workflow, and run tests continue to pass without using the new table.

## Phase 1 - Registry CRUD API

Goal: make agent definitions manageable through backend APIs, still without changing run launch behavior.

Implementation:

- Add `GET /agents/definitions?user_id=&include_disabled=` to list saved definitions.
- Add `POST /agents/definitions` to create a definition.
- Add `GET /agents/definitions/{agent_id}` to fetch one definition.
- Add `PUT /agents/definitions/{agent_id}` to update mutable fields.
- Add `DELETE /agents/definitions/{agent_id}` as a soft delete by setting `enabled=false`.
- Return a consistent API shape that the Studio UI can consume directly.

Validation rules:

- Reject missing `name`, `agent_capability_id`, or `instructions`.
- Reject `default_max_steps <= 0`.
- Reject unknown `agent_capability_id`.
- Reject non-agentic primary capabilities.
- Reject unknown `allowed_capability_ids` values when the capability catalog can validate them.
- Preserve disabled records for historical run snapshots and auditability.

Acceptance gate:

- API tests cover create, list, get, update, and soft-delete.
- API tests cover required-field, invalid-capability, and invalid-step-allowlist failures.
- Existing Workbench run creation remains unchanged.

## Phase 2 - Workbench Run Integration

Goal: allow Agent Sandbox runs to launch from a saved profile while preserving reproducible run history.

Implementation:

- Extend `POST /workbench/agent-runs` with optional `agent_definition_id`.
- When `agent_definition_id` is provided, load the enabled definition and hydrate missing run request fields from its defaults.
- Validate the resolved run after defaults are applied, including preflight checks.
- Require the selected primary capability to match the definition's `agent_capability_id`.
- If `allowed_capability_ids` is non-empty, reject additional Workbench steps whose capability ids are outside the allowlist.
- Store an immutable `agent_definition_snapshot` in run metadata.

Execution semantics:

- A saved definition does not create a new runtime worker.
- The primary agent capability remains the orchestrating agent for the run.
- Extra steps remain capabilities/tools attached to the run.
- Preflight validates the resolved request, not the partial profile input.
- Editing a definition after launch must not change historical run metadata.

Acceptance gate:

- Launching with `agent_definition_id` applies profile defaults.
- Launching with a disabled or missing definition fails cleanly.
- Disallowed extra capabilities are rejected before execution.
- Run debugger or metadata inspection shows the captured `agent_definition_snapshot`.
- Editing the definition after a run does not mutate the previous run snapshot.

## Phase 3 - Agent Sandbox UI

Goal: let users create and reuse agents from UI controls instead of editing raw JSON.

Implementation:

- Update Agent Sandbox in `StudioWorkbenchSurface.tsx` with an Agent Profile panel above the current Agent card.
- Add an Agent Profile selector that loads enabled definitions.
- Add `New`, `Save`, `Save as`, and `Delete` actions.
- Selecting a profile hydrates Agent capability, task/goal, workspace path, max steps, constraints, and instructions.
- Saving a profile serializes the current Agent card values into the Agent Definition API shape.
- Launching a selected saved profile sends `agent_definition_id`.
- Launching an unsaved draft keeps the current direct Agent Sandbox run behavior.
- Keep Raw RunSpec available and synchronized as the advanced/debug path.

UI behavior:

- `New` clears profile selection and starts a draft from current Agent Sandbox defaults.
- `Save` updates the selected profile.
- `Save as` creates a new profile from current values and selects it.
- `Delete` asks for confirmation, soft-disables the selected profile, and returns to an unsaved draft.
- API failures surface inline without clearing the user's draft.

Acceptance gate:

- TypeScript compile passes.
- Agent Profile selector loads saved definitions.
- `New`, `Save`, `Save as`, and `Delete` update UI state correctly.
- Selecting a saved profile hydrates the Agent card.
- Launching a saved profile includes `agent_definition_id`.
- Raw RunSpec remains visible and reflects the resolved run request.

## Phase 4 - End-To-End Verification And Staging Rollout

Goal: prove the complete flow works on staging and does not regress current Workbench usage.

Implementation:

- Add focused API tests for the new CRUD and run-integration paths.
- Add focused UI checks for profile load/save/select/launch behavior.
- Run migration upgrade in staging before deploying API code that depends on the new table.
- Deploy API first, then UI.
- Keep existing direct Agent Sandbox launch path available as fallback.

Acceptance gate:

- Create a profile in staging from Agent Sandbox.
- Select the profile and launch a run without touching raw JSON.
- Confirm the run reaches the existing Workbench execution path.
- Confirm run metadata contains the profile snapshot.
- Confirm an unsaved draft run still works.
- Confirm a deleted profile no longer appears by default.

## Phase 5 - Hardening And Follow-Ups

Goal: prepare the registry for broader platform use after the Workbench-only v1 is stable.

Follow-ups:

- Add immutable versions or aliases for published agent definitions.
- Add stronger tenancy and sharing semantics once user/org boundaries are finalized.
- Add planner/chat retrieval of registered agents when the platform is ready to route outside Workbench.
- Add richer guardrail enforcement beyond stored policy metadata.
- Add runtime memory integration when a dedicated memory service is available.
- Add audit views for definition changes and runs launched from each profile.

## Test Plan

Required before merge:

- API CRUD tests for create, list, get, update, and soft-delete.
- API validation tests for missing required fields and invalid capabilities.
- API run-integration tests for default hydration, allowlist enforcement, disabled-definition failure, and immutable snapshots.
- UI TypeScript compile.
- UI tests or manual Playwright checks for profile selector, save, save-as, delete, hydrate, and launch.
- Migration upgrade/downgrade check.
- Existing Workbench and workflow tests remain green.

## Assumptions And Defaults

- V1 is Workbench-only; chat routing, planner-wide agent retrieval, and direct invocation outside Agent Sandbox are out of scope.
- V1 stores mutable agent definitions plus immutable run snapshots; full version publishing and aliases are deferred to Phase 5.
- Definitions are user-owned when `user_id` is available, but the implementation should not block on a complete tenancy model.
- The default profile starts with `codegen.autonomous`, `default_max_steps=6`, and the current Agent Sandbox workspace default.
- Memory policy is stored now for parity with production agent platforms, but runtime memory behavior can remain disabled until the platform has a dedicated memory service integration.
