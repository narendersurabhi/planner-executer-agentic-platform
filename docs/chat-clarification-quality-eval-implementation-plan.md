# Chat Clarification Quality Eval Implementation Plan

## Goal

Turn the real-time clarification loop into a measurable, gateable behavior instead of a set of untracked heuristics.

## Problem

The clarification loop now:

- maps answers in real time
- advances the active question
- handles request restarts
- emits feedback dimensions

But there is still no dedicated offline eval gate for clarification quality. That means regressions in:

- active-field resolution
- queue advancement
- restart-vs-continue decisions
- wrong-field assignment

can slip through until they show up in live chat traffic.

## Design

### Phase 1: Clarification Gold Set

Create a repo-native YAML fixture for common clarification patterns:

- direct slot answers
- queue advancement after slot resolution
- redirect-style request restarts
- redirect wording that should still remain a slot answer
- opportunistic non-active-field captures that should not advance the queue

### Phase 2: Offline Evaluator

Add a core evaluator and CLI that score:

- overall clarification outcome accuracy
- restart decision accuracy
- resolved-active-field accuracy
- queue-advance accuracy
- wrong-field assignment rate

### Phase 3: Release Gating

Wire the evaluator into:

- `make eval-chat-clarification`
- `make eval-chat-clarification-gate`
- CI

## Acceptance Criteria

- clarification quality has a dedicated gold fixture under `eval/`
- an offline script writes a JSON report under `artifacts/evals/`
- CI fails if clarification restart or queue-advance quality regresses past thresholds

## Implementation Status

- Phase 1: implemented
- Phase 2: implemented
- Phase 3: implemented
