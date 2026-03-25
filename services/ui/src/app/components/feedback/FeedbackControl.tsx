"use client";

import { useEffect, useState } from "react";

import type { FeedbackEntry, FeedbackReasonOption, FeedbackSentiment } from "../../lib/feedback";

type SentimentOption = {
  value: FeedbackSentiment;
  label: string;
};

type FeedbackControlProps = {
  title: string;
  reasonOptions: FeedbackReasonOption[];
  sentimentOptions: SentimentOption[];
  autoSubmitSentiments?: FeedbackSentiment[];
  existing?: FeedbackEntry | null;
  submitting?: boolean;
  onSubmit: (payload: {
    sentiment: FeedbackSentiment;
    reason_codes: string[];
    comment?: string;
    score?: number;
  }) => Promise<void> | void;
};

const defaultAutoSubmit: FeedbackSentiment[] = ["positive"];

export default function FeedbackControl({
  title,
  reasonOptions,
  sentimentOptions,
  autoSubmitSentiments = defaultAutoSubmit,
  existing,
  submitting = false,
  onSubmit
}: FeedbackControlProps) {
  const [draftSentiment, setDraftSentiment] = useState<FeedbackSentiment | null>(null);
  const [selectedReasons, setSelectedReasons] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!existing) {
      return;
    }
    setDraftSentiment(existing.sentiment);
    setSelectedReasons(Array.isArray(existing.reason_codes) ? existing.reason_codes : []);
    setComment(typeof existing.comment === "string" ? existing.comment : "");
    setIsExpanded(false);
  }, [existing]);

  const toggleReason = (code: string) => {
    setSelectedReasons((previous) =>
      previous.includes(code) ? previous.filter((entry) => entry !== code) : [...previous, code]
    );
  };

  const handleSentimentSelect = async (sentiment: FeedbackSentiment) => {
    setDraftSentiment(sentiment);
    if (autoSubmitSentiments.includes(sentiment)) {
      await Promise.resolve(
        onSubmit({
          sentiment,
          reason_codes: [],
          comment: ""
        })
      );
      setSelectedReasons([]);
      setComment("");
      setIsExpanded(false);
      return;
    }
    setIsExpanded(true);
  };

  const submitExpandedFeedback = async () => {
    if (!draftSentiment) {
      return;
    }
    await Promise.resolve(
      onSubmit({
        sentiment: draftSentiment,
        reason_codes: selectedReasons,
        comment: comment.trim() || undefined,
        score: draftSentiment === "partial" ? 3 : undefined
      })
    );
    setIsExpanded(false);
  };

  return (
    <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-slate-700">{title}</span>
        {existing ? (
          <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-700">
            Saved
          </span>
        ) : null}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {sentimentOptions.map((option) => {
          const isActive = draftSentiment === option.value;
          return (
            <button
              key={`${title}-${option.value}`}
              type="button"
              className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${
                isActive
                  ? "border-slate-800 bg-slate-900 text-white"
                  : "border-slate-300 bg-white text-slate-700 hover:border-slate-400"
              }`}
              disabled={submitting}
              onClick={() => void handleSentimentSelect(option.value)}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      {isExpanded ? (
        <div className="mt-3 space-y-2">
          <div className="flex flex-wrap gap-2">
            {reasonOptions.map((reason) => {
              const isActive = selectedReasons.includes(reason.code);
              return (
                <button
                  key={`${title}-${reason.code}`}
                  type="button"
                  className={`rounded-full border px-2 py-1 text-[11px] transition ${
                    isActive
                      ? "border-amber-300 bg-amber-100 text-amber-900"
                      : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
                  }`}
                  onClick={() => toggleReason(reason.code)}
                >
                  {reason.label}
                </button>
              );
            })}
          </div>
          <textarea
            className="min-h-[4.5rem] w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 focus:border-slate-400 focus:outline-none"
            placeholder="Optional comment"
            value={comment}
            onChange={(event) => setComment(event.target.value)}
          />
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              className="rounded-full border border-slate-300 px-3 py-1 text-[11px] text-slate-600"
              onClick={() => setIsExpanded(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="rounded-full bg-slate-900 px-3 py-1 text-[11px] font-semibold text-white disabled:opacity-50"
              disabled={submitting || !draftSentiment}
              onClick={() => void submitExpandedFeedback()}
            >
              {submitting ? "Saving..." : "Save feedback"}
            </button>
          </div>
        </div>
      ) : existing?.comment ? (
        <div className="mt-2 text-[11px] text-slate-500">{existing.comment}</div>
      ) : null}
    </div>
  );
}
