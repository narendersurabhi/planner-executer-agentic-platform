# AgenticOps Playbook

## Purpose

This document is a practical blueprint for setting up AgenticOps: the operating model, telemetry, evaluation gates, release process, and incident handling needed to run agentic systems reliably. It is written as a general playbook first and then mapped onto this repository so it can be customized later.

## What AgenticOps Is

AgenticOps is the production discipline around AI agents and agentic workflows. It sits at the intersection of:

- runtime observability
- evaluation and regression gating
- policy and safety controls
- release engineering
- human review and incident response

For a planner/executor system, AgenticOps is the difference between "the model answered" and "the system can be promoted, audited, rolled back, and improved safely."

## What Public Guidance From Leading Teams Converges On

Public guidance from Anthropic, OpenAI, AWS, Microsoft, and Google Cloud is directionally consistent on a few points.

### 1. Start with the simplest agent shape that works

Top teams do not begin with large multi-agent graphs by default. They start with:

- prompt + tool calling
- a single orchestrator with explicit steps
- deterministic workflows around the model

Only after they can measure failure modes do they add more dynamic routing, delegation, memory, or multi-agent behavior.

Operational implication: keep the first production surface small enough that you can trace every step, replay failures, and define crisp pass/fail criteria.

### 2. Evaluate both outcomes and trajectories

Strong teams do not stop at "final answer looked good." They inspect:

- final task success
- tool selection correctness
- policy compliance
- intermediate reasoning behavior that is externally observable through traces, tool calls, and state transitions
- latency, retries, and recovery behavior

Operational implication: build regression suites that validate both business outcomes and execution behavior.

### 3. Treat observability as part of the product, not a backend afterthought

Agentic systems need deeper telemetry than conventional CRUD services. Production teams instrument:

- request, trace, and run identifiers
- model/provider metadata
- tool invocations
- retrieval inputs and outputs
- policy decisions
- user feedback
- token, latency, and cost envelopes

Operational implication: every run should be reconstructible after the fact.

### 4. Add guardrails around tools, not just prompts

Public guidance consistently emphasizes policy layers, human approvals for high-impact actions, and explicit tool governance. The main risk surface in agentic systems is not only text generation; it is tool use, side effects, and unbounded autonomy.

Operational implication: maintain allowlists, approval thresholds, environment boundaries, and rollback paths.

### 5. Promotion to production should be gated by replay and live staging checks

Leading teams use layered release gates:

- offline evals
- staging validation
- canary rollout
- production monitoring with rollback triggers

Operational implication: promotion should happen only for an already-tested artifact, not a newly built one.

### 6. Feedback loops are core infrastructure

Top companies continuously convert incidents, low-confidence outputs, and human corrections into:

- labeled failure datasets
- regression tests
- routing or threshold updates
- prompt/policy changes
- retraining or calibration inputs

Operational implication: every production miss should have a path into the regression suite.

## A Reference AgenticOps Stack

An effective AgenticOps platform usually contains the following layers.

### 1. Runtime Control Plane

This is the production execution surface.

It should manage:

- request intake
- routing
- orchestration and workflow execution
- tool access
- policy checks
- persistence of run state

Key requirement: every run must have a stable identifier that ties together application logs, traces, tool activity, evaluation artifacts, and user feedback.

### 2. Observability Plane

This is how operators understand behavior in production.

Minimum signals:

- run throughput
- success/failure rates
- latency percentiles by route, workflow, and provider
- token and cost usage
- tool success/error rates
- policy denials
- retry counts
- queue depth and stuck work

Minimum data model:

- `request_id`
- `session_id`
- `run_id`
- `workflow_id`
- `task_id`
- `provider`
- `model`
- `tool_name`
- `policy_decision`
- `latency_ms`
- `token_counts`
- `cost_estimate`

### 3. Evaluation Plane

This is the release gate.

You need three kinds of evaluation:

1. Offline gold-set evaluation  
   Curated prompts and expected behaviors.

2. Replay evaluation  
   Production or staging conversations replayed against a candidate change.

3. Live staging evaluation  
   A deployed environment hit through public or internal APIs with the real release artifact.

This is where most teams separate toy demos from production systems.

### 4. Policy and Safety Plane

This layer decides what is allowed.

It usually covers:

- tool allowlists and deny rules
- protected actions requiring approval
- content policy enforcement
- data handling restrictions
- prompt injection defenses
- secret and credential boundaries
- workspace or filesystem limits

### 5. Release Plane

This connects engineering changes to production risk controls.

It should support:

- versioned artifacts
- versioned prompts/configs/policies
- staging deployment
- canary deployment
- promotion of the exact same image or bundle
- rollback by artifact version

### 6. Feedback Plane

This closes the loop between operations and improvement.

It should collect:

- thumbs up/down or structured operator feedback
- failure taxonomies
- route mismatches
- policy false positives and false negatives
- escalation and manual override reasons

The output of this plane should become evaluation data, routing updates, and operational fixes.

## Operating Model

The cleanest setup is to make ownership explicit.

### Platform or AgentOps Team

Owns:

- shared orchestration framework
- observability standards
- evaluation harnesses
- release gates
- dashboards and alerts
- incident tooling
- model/provider governance

### Product or Workflow Teams

Own:

- domain-specific prompts
- tool definitions
- workflow logic
- acceptance criteria
- golden sets for their use cases

### Security, Risk, or Policy Owner

Owns:

- protected action policies
- data access rules
- human approval requirements
- audit expectations
- escalation paths

### On-Call or Operations Owner

Owns:

- alert response
- rollback execution
- incident classification
- hotfix promotion decisions

## The Production Workflow Used By Strong Teams

The public pattern is usually some version of this sequence.

### Step 1. Define task classes and routing boundaries

Do not send all prompts into one generic agent path. Define classes such as:

- FAQ or direct answer
- retrieval-augmented answer
- planner/executor workflow
- human escalation
- unsupported request

Each class needs:

- entry criteria
- success metrics
- failure modes
- fallback behavior

### Step 2. Instrument every run before expanding autonomy

Before adding more routing logic or more tools, make sure you can answer:

- what path was chosen
- why it was chosen
- which provider/model handled the task
- which tools were invoked
- where latency was spent
- why a policy denied or allowed the action
- whether the user later corrected the output

### Step 3. Build a release gate that combines quality and operational health

A serious gate includes:

- gold-set pass rate
- critical safety checks
- tool-use correctness
- latency budget
- error budget
- cost budget

Do not promote based on average quality alone. A route that is slightly better but much slower or less reliable can still be a production regression.

### Step 4. Run the candidate in staging with the real artifact

The staging environment should be close enough to production that you can validate:

- routing behavior
- tool access
- policy rules
- environment-specific configuration
- dashboards and alerts

The key rule is simple: promote the same image tag that passed staging.

### Step 5. Use canaries with rollback triggers

Typical canary signals:

- latency regression
- error-rate regression
- spike in policy denials
- route mismatch increase
- tool failure rate increase
- user dissatisfaction

Rollback should be automatic or one command away.

### Step 6. Convert misses into regression assets

Every important failure should become one or more of:

- a new golden case
- a new live regression case
- a routing threshold adjustment
- a policy rule change
- a runbook entry

This is where the system compounds in quality over time.

## The Metrics That Matter

A production AgenticOps setup tracks four metric groups.

### Quality

- task success rate
- grounded answer rate
- route accuracy
- tool selection accuracy
- human override rate

### Reliability

- request error rate
- workflow completion rate
- retry rate
- stuck-run rate
- queue delay

### Latency

- time to first token
- time to first tool call
- end-to-end completion time
- per-tool latency
- retrieval latency

### Cost

- cost per run
- tokens per route
- expensive-route share
- failed-run cost

Each metric should be sliceable by:

- environment
- provider
- model
- workflow
- tenant or customer tier
- route class

## Required Dashboards

At minimum, maintain these dashboards.

### Executive Health

- total requests
- success rate
- median and p95 latency
- cost per day
- open incidents

### Routing

- traffic share by route
- route confidence distribution
- fallback frequency
- route confusion pairs

### Tooling

- tool call volume
- tool error rate
- top failing tools
- approval-required tool activity

### Policy

- policy denies by rule
- false-positive review outcomes
- blocked high-risk actions

### Release

- staging pass/fail history
- canary health
- production rollback history

## Guardrails That Matter Most

If you only implement a few hard controls early, make them these:

1. Tool allowlists by route and environment
2. Human approval for side-effectful or irreversible actions
3. Prompt-injection and unsafe-instruction checks around external content
4. Per-route timeouts, retry limits, and budget caps
5. Safe fallback when confidence or policy conditions are not met
6. Versioned prompts, configs, and policies tied to the deployed artifact

## Release Process

This is the most defensible promotion flow for an agentic platform.

### Pre-merge

- run unit and integration tests
- run offline golden-set evaluation
- review policy or routing changes explicitly

### Staging

- deploy candidate artifact
- run live staging regression suite
- run route-specific or workflow-specific smoke tests
- inspect traces for a sample of complex runs

### Promotion

- promote the exact same artifact version
- enable a limited canary if risk is nontrivial
- watch operational dashboards during the canary window

### Post-promotion

- compare production metrics against baseline
- label incidents quickly
- add regressions before the next release

## Incident Response Model

Treat agent failures like product incidents, not just model oddities.

Every incident should record:

- affected route or workflow
- impact level
- user-visible behavior
- triggering prompt or input pattern
- tool or policy involvement
- provider/model version
- corrective action
- regression artifact added

Good incident classes:

- bad answer
- missed workflow trigger
- wrong route
- unsafe tool attempt
- policy miss
- timeout or cost blow-up
- non-deterministic drift

## A Practical 30/60/90 Day Rollout Plan

### Days 0-30: Establish control and visibility

Goals:

- instrument traces, metrics, and structured logs
- define top 3 to 5 route classes
- create a first golden dataset
- define staging deployment and rollback

Deliverables:

- run identifiers across the stack
- baseline dashboards
- initial live regression suite
- on-call runbook

### Days 31-60: Add release discipline

Goals:

- gate staging with regression runs
- add route accuracy and tool-use evaluation
- enforce policy boundaries for risky actions
- create incident taxonomy

Deliverables:

- staging promotion checklist
- policy dashboard
- failure-to-regression loop
- canary procedure

### Days 61-90: Add adaptive optimization

Goals:

- calibrate routing thresholds from real traffic
- segment metrics by tenant and workflow
- optimize cost and latency by route
- automate rollback triggers

Deliverables:

- replay harness from production feedback
- route confidence calibration
- automated release scorecard
- artifact-based rollback

## How To Adapt This Repo

This repository already has useful building blocks for AgenticOps.

### What Already Exists

- observability guidance in [`docs/observability.md`](/Users/narendersurabhi/planner-executer-agentic-platform/docs/observability.md)
- staging deployment guidance in [`docs/staging-deployment.md`](/Users/narendersurabhi/planner-executer-agentic-platform/docs/staging-deployment.md)
- request routing architecture notes in [`docs/request-routing-architecture.md`](/Users/narendersurabhi/planner-executer-agentic-platform/docs/request-routing-architecture.md)
- staging routing gate workflow in [`.github/workflows/staging-routing-gate.yml`](/Users/narendersurabhi/planner-executer-agentic-platform/.github/workflows/staging-routing-gate.yml)
- live regression and calibrator targets in [`Makefile`](/Users/narendersurabhi/planner-executer-agentic-platform/Makefile)

### Recommended Next Repo-Specific Moves

1. Define a formal route taxonomy
   - direct answer
   - retrieval answer
   - planner/executor workflow
   - escalate or reject

2. Make route choice observable
   - log route label, confidence, fallback reason, and policy decision on every request

3. Promote the existing staging regression suite into the default release gate
   - run live regression before promotion
   - run the routing calibrator replay on recent feedback

4. Add an incident-to-regression rule
   - no important production miss closes without a new regression asset

5. Add canary scorecards
   - compare success, latency, and route mix between current and candidate releases

6. Version policy and prompt bundles with releases
   - avoid deploying code separately from route definitions and policy rules

## A Minimal Promotion Checklist

Use this before shipping any agentic change to production.

1. Candidate artifact built and versioned
2. Config, prompts, and policies versioned with it
3. Offline gold-set evaluation passes
4. Staging deployment healthy
5. Live regression suite passes in staging
6. Operational dashboards show no regressions
7. Canary plan and rollback path confirmed
8. Promotion uses the same artifact that passed staging

## Sources

This playbook is based on public guidance from:

- OpenAI, "A practical guide to building agents"  
  https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/

- Anthropic, "Building effective agents"  
  https://www.anthropic.com/engineering/building-effective-agents

- AWS, "Evaluating AI agents: Real-world lessons from building agentic systems at Amazon"  
  https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/

- Microsoft Learn, "Agent evaluators"  
  https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/agent-evaluators

- Google Cloud, "AI agent observability overview"  
  https://cloud.google.com/stackdriver/docs/instrumentation/ai-agent-overview

- Google Cloud, "Model observability"  
  https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-observability

## Final Recommendation

If you want a compact operating principle, use this:

Keep the agent simple enough to understand, instrument every step, gate releases with replay and staging regressions, and convert every important miss into a permanent regression artifact.
