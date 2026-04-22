# Feedback, Evaluation, and Fine-Tuning Playbook

## Goal
Turn live user feedback into a disciplined optimization loop for chat quality, intent understanding, planning quality, and end-to-end outcomes.

This document explains:

- how the feedback system fits together
- how to use the exported feedback dataset for model and prompt optimization
- when fine-tuning is justified
- how to wire fine-tuned models into this codebase

Related implementation docs:

- [`docs/user-feedback-implementation-plan.md`](./user-feedback-implementation-plan.md)
- [`docs/user-feedback-analytics-implementation-plan.md`](./user-feedback-analytics-implementation-plan.md)

## How It All Comes Together

The current feedback loop has five layers.

### 1. Collect

Users leave explicit feedback in the UI against four target types:

- `chat_message`
- `intent_assessment`
- `plan`
- `job_outcome`

Those controls are rendered in:

- [`services/ui/src/app/components/feedback/FeedbackControl.tsx`](../services/ui/src/app/components/feedback/FeedbackControl.tsx)
- [`services/ui/src/app/page.tsx`](../services/ui/src/app/page.tsx)

The UI submits feedback to:

- `POST /feedback`

implemented in:

- [`services/api/app/main.py`](../services/api/app/main.py)
- [`services/api/app/feedback_service.py`](../services/api/app/feedback_service.py)

Each feedback row stores:

- the rated target
- sentiment and reason codes
- an optional comment
- a snapshot of the rated artifact at submission time
- linked identifiers such as `session_id`, `job_id`, `plan_id`, and `message_id`

That snapshot is important. It lets you analyze the exact message, intent view, plan, or outcome the user reacted to, even if the object changes later.

### 2. Enrich

When feedback is written, the API derives normalized dimensions and stores them in feedback metadata. Current dimensions include:

- `workflow_source`
- `llm_provider`
- `llm_model`
- `planner_version`
- `job_status_at_feedback`
- `assistant_action_type`
- `has_comment`
- `reason_count`

This enrichment happens in:

- [`services/api/app/feedback_service.py`](../services/api/app/feedback_service.py)

This gives you analysis-friendly dimensions without mutating the original sentiment or comment.

### 3. Analyze

The API exposes summary analytics through:

- `GET /feedback/summary`

It aggregates:

- sentiment counts
- target-type counts
- top negative reasons
- workflow source breakdowns
- model breakdowns
- planner-version breakdowns
- job-status breakdowns
- explicit-feedback rates such as chat helpfulness and plan approval
- operational correlates such as replans, retries, failed tasks, plan failures, and clarification turns

This path is implemented in:

- [`services/api/app/main.py`](../services/api/app/main.py)
- [`services/api/app/feedback_service.py`](../services/api/app/feedback_service.py)

The UI reads that summary into the operator-facing insights panel:

- [`services/ui/src/app/components/feedback/FeedbackInsightsPanel.tsx`](../services/ui/src/app/components/feedback/FeedbackInsightsPanel.tsx)

### 4. Export

The API exposes reusable evaluation examples through:

- `GET /feedback/examples`

By default, it exports only `negative` and `partial` feedback, which is usually the highest-value slice for debugging and training.

It supports:

- `target_type`
- repeated `sentiment`
- `reason_code`
- `workflow_source`
- `llm_model`
- `planner_version`
- `since`
- `until`
- `limit`
- `format=json|jsonl`

JSONL export is shaped by:

- [`libs/core/feedback_eval.py`](../libs/core/feedback_eval.py)

That gives you a stable bridge from live product feedback into offline evaluation and training workflows.

### 5. Observe

The API also emits Prometheus counters for feedback activity:

- `feedback_submitted_total{target_type,sentiment}`
- `feedback_reason_total{target_type,reason_code}`
- `feedback_summary_requests_total`
- `feedback_examples_export_total`

These metrics let you track whether people are actually rating the system and which failure modes are becoming more common.

## The Optimization Loop

The right loop in this repo is:

1. Collect explicit feedback from real usage.
2. Slice it by `target_type`, `reason_code`, `workflow_source`, `llm_model`, and `planner_version`.
3. Export the bad or partial examples.
4. Use those examples to optimize the narrow subsystem that produced the bad outcome.
5. Validate offline before shipping.
6. Roll out the change.
7. Watch live feedback and correlated runtime signals to confirm improvement.

Do not treat all bad feedback as a single model problem.

In this system, the failure could come from:

- chat response generation
- chat routing
- intent assessment
- intent decomposition
- planner generation
- worker-side LLM tools
- capability selection or validation

The right fix depends on which target type is failing.

## How To Use the Eval Dataset To Optimize the Model

Start by treating the exported dataset as an evaluation and triage dataset, not as direct fine-tuning data.

### Step 1: Export a Narrow Slice

Examples:

```bash
curl "http://localhost:18000/feedback/examples?target_type=chat_message&sentiment=negative&format=jsonl" \
  > eval/chat_feedback_negative.jsonl

curl "http://localhost:18000/feedback/examples?target_type=plan&sentiment=negative&reason_code=wrong_capability&format=jsonl" \
  > eval/plan_wrong_capability.jsonl

curl "http://localhost:18000/feedback/summary?target_type=intent_assessment"
```

Keep slices narrow. A dataset mixed across chat style, planner failures, and end-to-end outcome dissatisfaction is too noisy to optimize well.

### Step 2: Group by Failure Mode

Use `target_type` and `reason_code` first.

Recommended interpretation:

- `chat_message`
  - optimize answer quality, brevity, specificity, tone, or tool-handoff behavior
- `intent_assessment`
  - optimize request understanding, clarification behavior, and slot capture
- `plan`
  - optimize decomposition quality, task ordering, and capability selection
- `job_outcome`
  - optimize the end-to-end path only after the intermediate layers are healthy

### Step 3: Optimize the Cheapest Lever First

Use this order:

1. Prompt changes
2. Heuristic/rule changes
3. Capability or planner contract changes
4. Model selection changes
5. Fine-tuning

This repo already supports separate model knobs for several subsystems, so you often do not need fine-tuning first.

### Step 4: Run Offline Before Shipping

Use the exported rows as a regression set for:

- prompt revisions
- model swaps
- routing threshold changes
- planner prompt changes
- capability-search tuning

A good pattern is:

1. freeze a JSONL slice from `/feedback/examples`
2. add reviewed expectations for that slice
3. compare baseline vs candidate prompt/model
4. ship only if the candidate improves the target slice without breaking adjacent slices

### Step 5: Confirm Online

After rollout, watch:

- `GET /feedback/summary`
- the `Feedback Insights` panel
- Prometheus feedback counters
- job failure rates and replan/retry correlates

The offline set tells you whether the change is plausible. Live feedback tells you whether it actually helped real users.

## Recommended Optimization Order by Subsystem

### Chat Response Quality

Use negative `chat_message` feedback to improve:

- system prompt wording
- response mode behavior
- `CHAT_RESPONSE_MODEL`

Most `too_generic`, `unclear`, and `too_verbose` failures should be attacked here first.

### Intent Understanding

Use `intent_assessment` feedback to improve:

- clarification prompts
- slot extraction and normalization
- `INTENT_ASSESS_MODEL`
- `INTENT_DECOMPOSE_MODEL`

This is often a better fine-tuning target than general chat, because the outputs are narrower and easier to evaluate.

### Planning

Use `plan` feedback to improve:

- planner prompt structure
- capability hints
- task-intent reconciliation
- planner contract enforcement
- the planner service LLM model when `PLANNER_MODE=llm`

### End-to-End Outcome

Use `job_outcome` feedback as the final acceptance signal.

If outcome feedback is poor while chat, intent, and plan feedback are healthy, the problem is usually in execution, capability behavior, payload resolution, or render/runtime paths rather than chat quality.

## Fine-Tuning

Fine-tuning should be the last optimization step for a narrow, stable failure mode.

### Entry Criteria

Do not fine-tune just because you have negative feedback.

Fine-tuning becomes reasonable when all of the following are true:

1. The failure mode is narrow and recurring.
   - Example: `intent_assessment + wrong_scope`
   - Bad example: all negative feedback across every target type

2. Prompt and heuristic changes have mostly plateaued.
   - You have already tried prompt tightening, thresholds, or planner/contract fixes.

3. You have enough reviewed examples.
   - Rough guideline:
   - `200-500` reviewed examples for narrow classification-style behavior
   - `500-2000+` reviewed examples for generative behavior

4. The examples are clean enough.
   - low ambiguity
   - low contradiction
   - stable reason codes
   - no sensitive content you should not train on

5. You have a holdout set.
   - Keep a non-training validation or test slice from the same failure family.

6. You have explicit success metrics.
   - Example: improve intent agreement on `wrong_scope` cases without degrading unrelated chat helpfulness.

### What the Current Feedback Export Is Good For

The current export is excellent for:

- regression testing
- failure clustering
- prompt comparison
- candidate model comparison
- building reviewed training candidates

It is not, by itself, perfect supervised fine-tuning data.

Why:

- negative feedback tells you what was bad
- the snapshot tells you what the system produced
- but it does not always give you the corrected ideal answer or ideal structured output

So before fine-tuning, you usually need a review step that adds:

- the correct assistant response
- the correct intent label
- the correct plan shape
- or the preferred outcome

### How To Fine-Tune

Recommended workflow:

1. Pick one subsystem.
   - Do not fine-tune one model for everything at once.

2. Export the narrow slice from `/feedback/examples`.

3. Review and rewrite the examples into training pairs.
   - For chat: user input plus preferred answer
   - For intent: user input plus preferred structured label
   - For planning: normalized input plus preferred structured plan or plan skeleton

4. Remove or redact sensitive content.

5. Split into:
   - train
   - validation
   - final holdout

6. Fine-tune with your provider outside this repo.
   - This repository does not currently orchestrate provider fine-tuning jobs for you.

7. Register the resulting model ID.

8. Plug the model ID into the narrow service knob, not the whole system.

9. Re-run offline evals and then canary in production.

### Exit Criteria

A fine-tuned model is ready to promote only when:

1. It clearly improves the target slice.
   - Example: materially better chat helpfulness on the targeted failure family.

2. It does not regress neighboring slices.
   - Example: better intent classification without worse clarification behavior elsewhere.

3. Latency and cost remain acceptable.

4. Structured-output validity remains acceptable.
   - This is especially important for intent and planning paths.

5. Online feedback confirms the improvement.
   - Watch at least:
   - summary rates
   - negative reason-code frequency
   - retries, replans, and failed outcomes

6. Rollback is easy.
   - Promotion should be an env-model swap, not a code rewrite.

## How To Integrate Fine-Tuned Models in This Workflow

This codebase already routes different responsibilities to different model environment variables.

### API Service Model Knobs

Defined in:

- [`services/api/app/main.py`](../services/api/app/main.py)
- [`.env.example`](../.env.example)

Current API knobs:

- `OPENAI_MODEL`
- `CHAT_ROUTER_MODEL`
- `CHAT_RESPONSE_MODEL`
- `CHAT_PENDING_CORRECTION_MODEL`
- `INTENT_ASSESS_MODEL`
- `INTENT_DECOMPOSE_MODEL`

Recommended mapping:

- fine-tuned chat answer model -> `CHAT_RESPONSE_MODEL`
- fine-tuned router/classifier model -> `CHAT_ROUTER_MODEL`
- fine-tuned intent assessment model -> `INTENT_ASSESS_MODEL`
- fine-tuned intent decomposition model -> `INTENT_DECOMPOSE_MODEL`
- fallback shared API model -> `OPENAI_MODEL`

### Planner Service Model Knobs

The planner service currently uses:

- `OPENAI_MODEL`

from:

- [`services/planner/app/bootstrap.py`](../services/planner/app/bootstrap.py)
- [`services/planner/app/runtime_service.py`](../services/planner/app/runtime_service.py)

That means if you want a planner-specific fine-tuned model today, you set the planner deployment's `OPENAI_MODEL` to that model ID.

### Worker Service Model Knobs

The worker service currently uses:

- `OPENAI_MODEL`

from:

- [`services/worker/app/main.py`](../services/worker/app/main.py)

That is the integration point for fine-tuned document-spec or worker-side LLM generation models.

### Provider Requirements

The OpenAI provider in this repo sends model requests through the Responses API:

- [`libs/core/llm_provider.py`](../libs/core/llm_provider.py)

So a fine-tuned model integrates cleanly when:

- the provider is reachable through `OPENAI_BASE_URL`
- the model ID is valid for that provider
- the model supports the request shape used by `OpenAIProvider`

### Deployment Pattern

1. Choose the narrow subsystem model knob.
2. Set the model ID in the service environment.
3. Redeploy only the affected service if possible.
4. Keep all other model knobs unchanged.
5. Monitor `/feedback/summary` and metrics after rollout.

Example:

```bash
CHAT_RESPONSE_MODEL=ft:gpt-4.1-mini:team:chat-helpfulness-v3
INTENT_ASSESS_MODEL=ft:gpt-5-nano:team:intent-scope-v2
INTENT_DECOMPOSE_MODEL=ft:gpt-4.1-mini:team:intent-decompose-v1
```

If only the chat answer quality is changing, do not also swap the intent or planner model in the same rollout.

## What To Track After Integration

For each rollout, compare before vs after on:

- chat helpfulness rate
- intent agreement rate
- plan approval rate
- outcome satisfaction
- top negative reason codes
- retry/replan/plan-failure correlates
- latency and error rates

Record at least:

- model ID
- model family
- prompt version
- planner version
- rollout date
- target slice

## What Not To Do

- Do not train on every feedback row indiscriminately.
- Do not mix chat, intent, plan, and outcome failures into one training set.
- Do not fine-tune before you have a reviewed holdout set.
- Do not ship a fine-tuned model system-wide if the problem is confined to one subsystem.
- Do not replace explicit user feedback with a synthetic blended quality score.

## Recommended Immediate Practice

Use this sequence:

1. Collect more feedback.
2. Export narrow negative slices from `/feedback/examples`.
3. Use those slices for offline prompt and model comparisons.
4. Only promote to fine-tuning after the failure mode is stable and the data is reviewed.
5. Integrate the fine-tuned model through the narrowest available env var.

That is the lowest-risk way to improve quality in this architecture.
