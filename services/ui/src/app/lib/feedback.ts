"use client";

export type FeedbackTargetType = "chat_message" | "intent_assessment" | "plan" | "job_outcome";
export type FeedbackSentiment = "positive" | "negative" | "neutral" | "partial";

export type FeedbackReasonOption = {
  code: string;
  label: string;
};

export type FeedbackEntry = {
  id: string;
  target_type: FeedbackTargetType;
  target_id: string;
  session_id?: string | null;
  job_id?: string | null;
  plan_id?: string | null;
  message_id?: string | null;
  user_id?: string | null;
  actor_key?: string | null;
  sentiment: FeedbackSentiment;
  score?: number | null;
  reason_codes: string[];
  comment?: string | null;
  snapshot?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type FeedbackListResponse = {
  items: FeedbackEntry[];
  summary?: {
    total: number;
    positive: number;
    negative: number;
    neutral: number;
    partial: number;
  };
};

const FEEDBACK_ACTOR_STORAGE_KEY = "ape.feedback.actor_id.v1";

export const CHAT_FEEDBACK_REASONS: FeedbackReasonOption[] = [
  { code: "incorrect", label: "Incorrect" },
  { code: "unclear", label: "Unclear" },
  { code: "missed_request", label: "Missed request" },
  { code: "too_generic", label: "Too generic" },
  { code: "too_verbose", label: "Too verbose" }
];

export const INTENT_FEEDBACK_REASONS: FeedbackReasonOption[] = [
  { code: "wrong_goal", label: "Wrong goal" },
  { code: "wrong_scope", label: "Wrong scope" },
  { code: "missed_constraint", label: "Missed constraint" },
  { code: "asked_unnecessary_clarification", label: "Unnecessary clarification" }
];

export const PLAN_FEEDBACK_REASONS: FeedbackReasonOption[] = [
  { code: "missing_step", label: "Missing step" },
  { code: "wrong_order", label: "Wrong order" },
  { code: "wrong_capability", label: "Wrong capability" },
  { code: "too_complex", label: "Too complex" },
  { code: "unsafe", label: "Unsafe" }
];

export const OUTCOME_FEEDBACK_REASONS: FeedbackReasonOption[] = [
  { code: "did_not_finish", label: "Did not finish" },
  { code: "wrong_result", label: "Wrong result" },
  { code: "poor_quality", label: "Poor quality" },
  { code: "took_too_long", label: "Took too long" }
];

export const feedbackTargetKey = (targetType: FeedbackTargetType, targetId: string) =>
  `${targetType}:${targetId}`;

const makeFallbackActorId = () =>
  `feedback-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export const getFeedbackActorId = (preferred?: string) => {
  const normalizedPreferred = String(preferred || "").trim();
  if (normalizedPreferred) {
    return normalizedPreferred;
  }
  if (typeof window === "undefined") {
    return "feedback-anonymous";
  }
  const stored = window.localStorage.getItem(FEEDBACK_ACTOR_STORAGE_KEY);
  if (stored && stored.trim()) {
    return stored.trim();
  }
  const next = makeFallbackActorId();
  window.localStorage.setItem(FEEDBACK_ACTOR_STORAGE_KEY, next);
  return next;
};
