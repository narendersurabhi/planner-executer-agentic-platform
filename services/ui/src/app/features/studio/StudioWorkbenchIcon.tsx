"use client";

export type StudioWorkbenchIconKind =
  | "home"
  | "menu"
  | "palette"
  | "chat"
  | "graph"
  | "library"
  | "inspect"
  | "run";

export default function StudioWorkbenchIcon({
  kind,
  className = "",
}: {
  kind: StudioWorkbenchIconKind;
  className?: string;
}) {
  const sharedProps = {
    fill: "none",
    stroke: "currentColor",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.8,
  };

  switch (kind) {
    case "home":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <path d="m4.5 11.5 7.5-6 7.5 6" {...sharedProps} />
          <path d="M7 10.5v8h10v-8" {...sharedProps} />
        </svg>
      );
    case "menu":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <path d="M5 7h14M5 12h9M5 17h14" {...sharedProps} />
        </svg>
      );
    case "palette":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <rect x="4" y="4" width="16" height="16" rx="3.5" {...sharedProps} />
          <path d="M7 8h10M7 12h10M7 16h7" {...sharedProps} />
        </svg>
      );
    case "chat":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <path d="M6 6.5h12a2.5 2.5 0 0 1 2.5 2.5v5a2.5 2.5 0 0 1-2.5 2.5H11l-4.5 3v-3H6A2.5 2.5 0 0 1 3.5 14V9A2.5 2.5 0 0 1 6 6.5Z" {...sharedProps} />
          <path d="M8 10.5h8M8 13.5h5" {...sharedProps} />
        </svg>
      );
    case "graph":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <circle cx="6.5" cy="17.5" r="2.5" {...sharedProps} />
          <circle cx="12" cy="6.5" r="2.5" {...sharedProps} />
          <circle cx="18" cy="14" r="2.5" {...sharedProps} />
          <path d="M8.4 15.8 10.2 8.4M14.2 8.2l2.2 3.5M8.7 16.4l6.8-1.6" {...sharedProps} />
        </svg>
      );
    case "library":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <rect x="4" y="5" width="6" height="14" rx="1.8" {...sharedProps} />
          <rect x="14" y="5" width="6" height="14" rx="1.8" {...sharedProps} />
          <path
            d="M8 8.5h.01M8 12h.01M8 15.5h.01M18 8.5h.01M18 12h.01M18 15.5h.01"
            {...sharedProps}
          />
        </svg>
      );
    case "inspect":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <rect x="4" y="5" width="16" height="14" rx="3" {...sharedProps} />
          <path d="M10 5v14M13.5 9h3M13.5 12h3M13.5 15h2" {...sharedProps} />
        </svg>
      );
    case "run":
      return (
        <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
          <path d="m9 7 8 5-8 5z" {...sharedProps} />
          <path d="M4.5 19.5h15" {...sharedProps} />
        </svg>
      );
  }
}
