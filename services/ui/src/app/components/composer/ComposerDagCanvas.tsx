"use client";

import { useEffect, useRef, useState } from "react";
import type React from "react";

import {
  resolveWorkflowNodeVisual,
} from "../workflow/WorkflowNodeIcon";
import WorkflowNodeCard, {
  type WorkflowNodeCardPortTone,
  type WorkflowNodeCardTone,
  workflowNodeCardToneForVisual,
  workflowNodeCardToneStyles,
} from "../workflow/WorkflowNodeCard";

type CanvasPoint = {
  x: number;
  y: number;
};

type ComposerDraftEdge = {
  fromNodeId: string;
  toNodeId: string;
  branchLabel?: string;
};

type ComposerDraftNode = {
  id: string;
  taskName: string;
  capabilityId: string;
  outputPath: string;
  nodeKind?: "capability" | "control";
  controlKind?: "if" | "if_else" | "switch" | "parallel" | null;
};

type DagCanvasEdge = {
  fromNodeId: string;
  toNodeId: string;
  edgeKey: string;
  fromTaskName: string;
  toTaskName: string;
  path: string;
  midX: number;
  midY: number;
  labelX: number;
  labelY: number;
  branchLabel?: string;
};

type DagCanvasNode = {
  node: ComposerDraftNode;
  position: CanvasPoint;
};

type DagConnectorDragState = {
  sourceNodeId: string;
  x: number;
  y: number;
  branchLabel?: string;
  sourcePortY?: number;
};

type DagCanvasPanState = {
  startClientX: number;
  startClientY: number;
  startScrollLeft: number;
  startScrollTop: number;
  clearSelectionOnClick: boolean;
};

type ComposerDagCanvasProps = {
  visualChainNodes: ComposerDraftNode[];
  dagEdgeDraftSourceNodeId: string | null;
  setDagEdgeDraftSourceNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  setDagConnectorDrag: React.Dispatch<React.SetStateAction<DagConnectorDragState | null>>;
  setDagConnectorHoverTargetNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  autoLayoutDagCanvas: () => void;
  dagCanvasViewportRef: React.RefObject<HTMLDivElement | null>;
  dagCanvasRef: React.RefObject<HTMLDivElement | null>;
  dagCanvasSurface: { width: number; height: number };
  dagCanvasEdges: DagCanvasEdge[];
  hoveredDagEdgeKey: string | null;
  setHoveredDagEdgeKey: React.Dispatch<React.SetStateAction<string | null>>;
  removeDagEdge: (fromNodeId: string, toNodeId: string) => void;
  dagConnectorPreview: { path: string } | null;
  dagCanvasNodes: DagCanvasNode[];
  composerDraftEdges: ComposerDraftEdge[];
  dagNodeAdjacency: {
    incoming: Record<string, number>;
    outgoing: Record<string, number>;
  };
  visualChainNodeStatusById: Map<
    string,
    {
      missingCount: number;
      requiredCount: number;
    }
  >;
  selectedDagNodeId: string | null;
  setSelectedDagNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  dagConnectorDrag: DagConnectorDragState | null;
  dagCanvasDraggingNodeId: string | null;
  dagConnectorHoverTargetNodeId: string | null;
  addDagEdge: (fromNodeId: string, toNodeId: string, branchLabel?: string) => void;
  beginDagNodeDrag: (event: React.MouseEvent<HTMLDivElement>, nodeId: string) => void;
  isInteractiveCanvasTarget: (target: EventTarget | null) => boolean;
  beginDagConnectorDrag: (
    event: React.MouseEvent<HTMLButtonElement>,
    nodeId: string,
    options?: { branchLabel?: string; sourcePortY?: number }
  ) => void;
  centerDagNodeInView: (nodeId: string) => void;
  nodeWidth: number;
  nodeHeight: number;
  dagCanvasZoom?: number;
  showToolbar?: boolean;
  showBlueprintPreview?: boolean;
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  zoomInDisabled?: boolean;
  zoomOutDisabled?: boolean;
  onToggleFocusGraph?: () => void;
  focusGraphActive?: boolean;
  onRunWorkflow?: () => void;
  runWorkflowPending?: boolean;
  runWorkflowDisabled?: boolean;
};

const toolbarButtonClassName =
  "inline-flex h-8 items-center rounded-lg border border-black/15 bg-[rgba(54,68,84,0.94)] px-2.5 text-[10px] font-semibold tracking-[0.04em] text-slate-50 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] transition hover:border-white/18 hover:bg-[rgba(61,77,95,0.98)] disabled:cursor-not-allowed disabled:opacity-40";
const DAG_CANVAS_PAN_THRESHOLD = 6;

type BlueprintPreviewNode = {
  id: string;
  x: number;
  y: number;
  title: string;
  subtitle: string;
  caption?: string;
  tone: WorkflowNodeCardTone;
  capabilityId: string;
  nodeKind?: "capability" | "control";
  controlKind?: "if" | "if_else" | "switch" | "parallel" | null;
};

type BlueprintPreviewEdge = {
  id: string;
  path: string;
  color: string;
  label?: string;
  labelX?: number;
  labelY?: number;
};

type ComposerNodePort = {
  key: string;
  label: string;
  branchLabel?: string;
  tone: WorkflowNodeCardPortTone;
  y: number;
};

const subtitleForNode = (
  node: ComposerDraftNode,
  visual: ReturnType<typeof resolveWorkflowNodeVisual>
) => {
  if (node.nodeKind === "control") {
    return "Logic Gate";
  }
  if (visual.tone === "llm") {
    return "LLM Request";
  }
  if (visual.tone === "validate") {
    return "Schema Check";
  }
  if (visual.tone === "memory") {
    return "State Storage";
  }
  if (visual.tone === "render") {
    return "Document Process";
  }
  if (visual.tone === "transform") {
    return "Data Processing";
  }
  if (visual.tone === "io") {
    return "Integration";
  }
  if (visual.tone === "code") {
    return "Code Action";
  }
  return "Workflow Step";
};

const statusForNode = (
  node: ComposerDraftNode,
  missingCount: number,
  requiredCount: number
) => {
  if (node.nodeKind === "control") {
    return {
      badgeBackground: "rgba(217, 119, 6, 0.92)",
      badgeColor: "#fff7ed",
      badgeLabel: "•",
      label: node.controlKind === "parallel" ? "Branch Group" : "Conditional Logic",
    };
  }
  if (missingCount > 0) {
    return {
      badgeBackground: "rgba(220, 38, 38, 0.92)",
      badgeColor: "#fff1f2",
      badgeLabel: "!",
      label: `${missingCount} missing field${missingCount === 1 ? "" : "s"}`,
    };
  }
  if (requiredCount > 0) {
    return {
      badgeBackground: "rgba(37, 99, 235, 0.88)",
      badgeColor: "#eff6ff",
      badgeLabel: "✓",
      label: "Ready",
    };
  }
  return {
    badgeBackground: "rgba(71, 85, 105, 0.82)",
    badgeColor: "#e2e8f0",
    badgeLabel: "•",
    label: "Configured",
  };
};

const outputPortsForNode = (
  node: ComposerDraftNode,
  visual: ReturnType<typeof resolveWorkflowNodeVisual>,
  nodeHeight: number
): ComposerNodePort[] => {
  if (node.nodeKind === "control" && node.controlKind === "if_else") {
    return [
      {
        key: "true",
        label: "True",
        branchLabel: "true",
        tone: "success",
        y: 40,
      },
      {
        key: "false",
        label: "False",
        branchLabel: "false",
        tone: "danger",
        y: 64,
      },
    ];
  }
  if (node.nodeKind === "control") {
    return [
      {
        key: "branch",
        label: "Branch",
        tone: "default",
        y: nodeHeight / 2,
      },
    ];
  }
  if (visual.tone === "validate") {
    return [{ key: "result", label: "Result", tone: "danger", y: nodeHeight / 2 }];
  }
  if (visual.tone === "memory") {
    return [{ key: "state", label: "State", tone: "default", y: nodeHeight / 2 }];
  }
  return [{ key: "output", label: "Output", tone: "default", y: nodeHeight / 2 }];
};

const dagBranchTheme = (branchLabel?: string) => {
  const normalized = String(branchLabel || "").trim().toLowerCase();
  if (normalized.includes("false") || normalized.includes("else")) {
    return {
      background: "rgba(95, 31, 52, 0.94)",
      border: "rgba(251, 113, 133, 0.4)",
      text: "#ffe4e6",
    };
  }
  if (normalized) {
    return {
      background: "rgba(31, 74, 60, 0.94)",
      border: "rgba(74, 222, 128, 0.38)",
      text: "#dcfce7",
    };
  }
  return {
    background: "rgba(50, 63, 79, 0.94)",
    border: "rgba(214, 228, 241, 0.22)",
    text: "#f8fafc",
  };
};

const emptyBlueprintNodes: BlueprintPreviewNode[] = [
  {
    id: "preview-control",
    x: 170,
    y: 270,
    title: "Conditional Check",
    subtitle: "Logic Gate",
    caption: "Conditional Logic",
    tone: "amber",
    capabilityId: "workflow.control",
    nodeKind: "control",
    controlKind: "if_else",
  },
  {
    id: "preview-summarize",
    x: 520,
    y: 110,
    title: "Summarize Text",
    subtitle: "LLM Request",
    caption: "Ready",
    tone: "sky",
    capabilityId: "llm.text.generate",
  },
  {
    id: "preview-reason",
    x: 520,
    y: 245,
    title: "GPT-4 Reasoning",
    subtitle: "LLM Request",
    caption: "Ready",
    tone: "sky",
    capabilityId: "llm.reason",
  },
  {
    id: "preview-process-top",
    x: 890,
    y: 250,
    title: "Processing PDF",
    subtitle: "Document Process",
    caption: "Configured",
    tone: "amber",
    capabilityId: "document.process",
  },
  {
    id: "preview-extract",
    x: 520,
    y: 430,
    title: "Extract Data",
    subtitle: "Data Processing",
    caption: "Ready",
    tone: "emerald",
    capabilityId: "document.process",
  },
  {
    id: "preview-process-bottom",
    x: 520,
    y: 610,
    title: "Processing PDF",
    subtitle: "Document Process",
    caption: "Configured",
    tone: "amber",
    capabilityId: "document.process",
  },
  {
    id: "preview-validate",
    x: 980,
    y: 450,
    title: "Data Validation",
    subtitle: "Schema Check",
    caption: "Failed: Missing Fields",
    tone: "rose",
    capabilityId: "validation.schema",
  },
  {
    id: "preview-notify",
    x: 1330,
    y: 455,
    title: "Notify Admin",
    subtitle: "Workflow Step",
    caption: "Configured",
    tone: "slate",
    capabilityId: "notification.send",
  },
];

const emptyBlueprintEdges: BlueprintPreviewEdge[] = [
  {
    id: "control-summary",
    path: "M 390 315 C 460 315, 450 150, 520 150",
    color: "rgba(116, 137, 158, 0.94)",
    label: "If true",
    labelX: 430,
    labelY: 236,
  },
  {
    id: "control-reason",
    path: "M 390 325 C 450 325, 455 285, 520 285",
    color: "rgba(116, 137, 158, 0.94)",
  },
  {
    id: "control-extract",
    path: "M 390 338 C 455 338, 450 470, 520 470",
    color: "rgba(116, 137, 158, 0.94)",
    label: "Else",
    labelX: 428,
    labelY: 394,
  },
  {
    id: "control-process-bottom",
    path: "M 390 348 C 450 348, 455 650, 520 650",
    color: "rgba(116, 137, 158, 0.94)",
  },
  {
    id: "reason-process-top",
    path: "M 740 285 C 810 285, 820 285, 890 285",
    color: "rgba(124, 146, 168, 0.92)",
  },
  {
    id: "process-top-extract",
    path: "M 1110 290 C 1170 290, 1175 420, 740 470",
    color: "rgba(124, 146, 168, 0.72)",
  },
  {
    id: "extract-validate",
    path: "M 740 470 C 860 470, 860 485, 980 485",
    color: "rgba(116, 137, 158, 0.94)",
  },
  {
    id: "process-bottom-validate",
    path: "M 740 650 C 860 650, 860 500, 980 500",
    color: "rgba(116, 137, 158, 0.94)",
  },
  {
    id: "validate-notify",
    path: "M 1200 490 C 1260 490, 1270 490, 1330 490",
    color: "rgba(124, 146, 168, 0.9)",
  },
];

export default function ComposerDagCanvas({
  visualChainNodes,
  dagEdgeDraftSourceNodeId,
  setDagEdgeDraftSourceNodeId,
  setDagConnectorDrag,
  setDagConnectorHoverTargetNodeId,
  autoLayoutDagCanvas,
  dagCanvasViewportRef,
  dagCanvasRef,
  dagCanvasSurface,
  dagCanvasEdges,
  hoveredDagEdgeKey,
  setHoveredDagEdgeKey,
  removeDagEdge,
  dagConnectorPreview,
  dagCanvasNodes,
  composerDraftEdges,
  dagNodeAdjacency,
  visualChainNodeStatusById,
  selectedDagNodeId,
  setSelectedDagNodeId,
  dagConnectorDrag,
  dagCanvasDraggingNodeId,
  dagConnectorHoverTargetNodeId,
  addDagEdge,
  beginDagNodeDrag,
  isInteractiveCanvasTarget,
  beginDagConnectorDrag,
  centerDagNodeInView,
  nodeWidth,
  nodeHeight,
  dagCanvasZoom = 1,
  showToolbar = false,
  showBlueprintPreview = false,
  onZoomIn,
  onZoomOut,
  zoomInDisabled = false,
  zoomOutDisabled = false,
  onToggleFocusGraph,
  focusGraphActive = false,
  onRunWorkflow,
  runWorkflowPending = false,
  runWorkflowDisabled = false,
}: ComposerDagCanvasProps) {
  const showEmptyBlueprint = showBlueprintPreview && visualChainNodes.length === 0;
  const [dagCanvasPanState, setDagCanvasPanState] = useState<DagCanvasPanState | null>(null);
  const [dagCanvasPanModifierPressed, setDagCanvasPanModifierPressed] = useState(false);
  const dagCanvasPanMovedRef = useRef(false);

  const canStartDagCanvasPan = (target: EventTarget | null) => {
    if (isInteractiveCanvasTarget(target)) {
      return false;
    }
    if (!(target instanceof Element)) {
      return true;
    }
    if (target.closest("[data-composer-node='true']")) {
      return false;
    }
    if (target.closest("[data-composer-edge='true']")) {
      return false;
    }
    return true;
  };

  useEffect(() => {
    const isEditableTarget = (target: EventTarget | null) => {
      if (!(target instanceof Element)) {
        return false;
      }
      return Boolean(
        target.closest(
          "input,textarea,select,button,a,label,[contenteditable='true']"
        )
      );
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !isEditableTarget(event.target)) {
        setDagCanvasPanModifierPressed(false);
        setDagCanvasPanState(null);
        setSelectedDagNodeId(null);
        return;
      }
      if (event.code !== "Space" || isEditableTarget(event.target)) {
        return;
      }
      event.preventDefault();
      setDagCanvasPanModifierPressed(true);
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code !== "Space") {
        return;
      }
      setDagCanvasPanModifierPressed(false);
    };

    const resetPanModifier = () => {
      setDagCanvasPanModifierPressed(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", resetPanModifier);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", resetPanModifier);
    };
  }, []);

  const beginDagCanvasPan = (event: React.MouseEvent<HTMLDivElement>) => {
    const viewport = dagCanvasViewportRef.current;
    if (!viewport || dagCanvasDraggingNodeId || dagConnectorDrag) {
      return;
    }
    if (event.button !== 0 && event.button !== 1) {
      return;
    }
    if (!canStartDagCanvasPan(event.target)) {
      return;
    }
    event.preventDefault();
    dagCanvasPanMovedRef.current = false;
    setDagCanvasPanState({
      startClientX: event.clientX,
      startClientY: event.clientY,
      startScrollLeft: viewport.scrollLeft,
      startScrollTop: viewport.scrollTop,
      clearSelectionOnClick: event.button === 0 && !dagCanvasPanModifierPressed,
    });
  };

  useEffect(() => {
    if (!dagCanvasPanState) {
      return;
    }
    const viewport = dagCanvasViewportRef.current;
    if (!viewport) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const deltaX = event.clientX - dagCanvasPanState.startClientX;
      const deltaY = event.clientY - dagCanvasPanState.startClientY;
      if (
        !dagCanvasPanMovedRef.current &&
        Math.abs(deltaX) < DAG_CANVAS_PAN_THRESHOLD &&
        Math.abs(deltaY) < DAG_CANVAS_PAN_THRESHOLD
      ) {
        return;
      }
      dagCanvasPanMovedRef.current = true;
      viewport.scrollLeft = dagCanvasPanState.startScrollLeft - deltaX;
      viewport.scrollTop = dagCanvasPanState.startScrollTop - deltaY;
    };

    const finishPan = () => {
      if (!dagCanvasPanMovedRef.current && dagCanvasPanState.clearSelectionOnClick) {
        setSelectedDagNodeId(null);
      }
      dagCanvasPanMovedRef.current = false;
      setDagCanvasPanState(null);
    };

    const previousBodyCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "grabbing";
    document.body.style.userSelect = "none";

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", finishPan);
    window.addEventListener("blur", finishPan);
    return () => {
      document.body.style.cursor = previousBodyCursor;
      document.body.style.userSelect = previousUserSelect;
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", finishPan);
      window.removeEventListener("blur", finishPan);
    };
  }, [
    dagCanvasPanState,
    dagCanvasDraggingNodeId,
    dagCanvasViewportRef,
    dagConnectorDrag,
    setSelectedDagNodeId,
  ]);

  return (
    <div className="relative h-full">
      {showToolbar ? (
        <div className="pointer-events-none absolute right-4 top-4 z-20 flex justify-end">
          <div className="pointer-events-auto flex items-center gap-1.5">
            <button
              className={toolbarButtonClassName}
              onClick={onZoomIn}
              disabled={zoomInDisabled}
              type="button"
            >
              + Zoom
            </button>
            <button
              className={toolbarButtonClassName}
              onClick={onZoomOut}
              disabled={zoomOutDisabled}
              type="button"
            >
              - Zoom
            </button>
            <div className="flex h-8 items-center rounded-lg border border-white/10 bg-black/10 px-2.5 text-[10px] font-semibold tracking-[0.06em] text-slate-200">
              {Math.round(dagCanvasZoom * 100)}%
            </div>
            <button
              className={toolbarButtonClassName}
              onClick={autoLayoutDagCanvas}
              disabled={visualChainNodes.length === 0}
              type="button"
            >
              Layout
            </button>
            <button
              className={`${toolbarButtonClassName} ${
                focusGraphActive
                  ? "border-sky-300/35 bg-[rgba(46,91,128,0.96)] text-sky-50"
                  : ""
              }`.trim()}
              onClick={onToggleFocusGraph}
              type="button"
            >
              {focusGraphActive ? "Exit Focus" : "Focus Graph"}
            </button>
            <button
              className={`${toolbarButtonClassName} border-white/16 bg-[rgba(38,48,61,0.98)]`}
              onClick={() => {
                onRunWorkflow?.();
              }}
              disabled={runWorkflowDisabled}
              type="button"
            >
              {runWorkflowPending ? "Starting..." : "Run"}
            </button>
          </div>
        </div>
      ) : null}

      <div
        ref={dagCanvasViewportRef}
        data-composer-canvas-viewport="true"
        className={`relative h-full overflow-auto bg-[#506478] [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px),radial-gradient(circle_at_16%_18%,rgba(255,255,255,0.08),transparent_18%),radial-gradient(circle_at_82%_24%,rgba(125,211,252,0.08),transparent_14%),linear-gradient(180deg,rgba(24,36,49,0.2),rgba(9,16,27,0.34))] [background-size:24px_24px,24px_24px,100%_100%,100%_100%,100%_100%] ${
          dagCanvasPanState ? "cursor-grabbing" : "cursor-grab"
        }`}
        onAuxClick={(event) => {
          if (event.button === 1) {
            event.preventDefault();
          }
        }}
        onMouseDown={beginDagCanvasPan}
      >
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_16%,rgba(191,219,254,0.08),transparent_12%),radial-gradient(circle_at_84%_22%,rgba(103,232,249,0.06),transparent_11%),radial-gradient(circle_at_50%_100%,rgba(15,23,42,0.14),transparent_40%)]" />
        <div
          className="relative min-h-full min-w-full"
          style={{
            width: dagCanvasSurface.width * dagCanvasZoom,
            height: dagCanvasSurface.height * dagCanvasZoom,
          }}
        >
          <div
            ref={dagCanvasRef}
            className="relative"
            style={{
              width: dagCanvasSurface.width,
              height: dagCanvasSurface.height,
              transform: `scale(${dagCanvasZoom})`,
              transformOrigin: "top left",
            }}
          >
          {showEmptyBlueprint ? (
            <svg
              className="pointer-events-none absolute left-0 top-0"
              width={dagCanvasSurface.width}
              height={dagCanvasSurface.height}
              viewBox={`0 0 ${dagCanvasSurface.width} ${dagCanvasSurface.height}`}
            >
              {emptyBlueprintEdges.map((edge) => (
                <g key={`empty-blueprint-edge-${edge.id}`}>
                  <path d={edge.path} stroke={edge.color} strokeWidth="2.4" fill="none" />
                  {edge.label && edge.labelX && edge.labelY ? (
                    <g>
                      <rect
                        x={edge.labelX - 32}
                        y={edge.labelY - 16}
                        width="64"
                        height="24"
                        rx="9"
                        fill="rgba(55, 69, 84, 0.94)"
                        stroke="rgba(255,255,255,0.12)"
                      />
                      <text
                        x={edge.labelX}
                        y={edge.labelY}
                        textAnchor="middle"
                        fontSize="11"
                        fill="#e2e8f0"
                      >
                        {edge.label}
                      </text>
                    </g>
                  ) : null}
                </g>
              ))}
            </svg>
          ) : null}

          <svg
            className="absolute left-0 top-0"
            width={dagCanvasSurface.width}
            height={dagCanvasSurface.height}
            viewBox={`0 0 ${dagCanvasSurface.width} ${dagCanvasSurface.height}`}
          >
            <defs>
              <marker
                id="composer-arrow"
                markerWidth="10"
                markerHeight="10"
                refX="8"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L0,6 L9,3 z" fill="#7dd3fc" />
              </marker>
            </defs>
            {dagCanvasEdges.map((edge) => {
              const isHovered = hoveredDagEdgeKey === edge.edgeKey;
              const labelTheme = dagBranchTheme(edge.branchLabel);
              const labelWidth = Math.max(52, edge.branchLabel ? edge.branchLabel.length * 8 + 18 : 0);
              return (
                <g
                  key={`composer-edge-${edge.edgeKey}`}
                  data-composer-edge="true"
                  onMouseEnter={() => setHoveredDagEdgeKey(edge.edgeKey)}
                  onMouseLeave={() =>
                    setHoveredDagEdgeKey((prev) => (prev === edge.edgeKey ? null : prev))
                  }
                >
                  <path
                    d={edge.path}
                    stroke={isHovered ? "rgba(15,23,42,0.5)" : "rgba(15,23,42,0.32)"}
                    strokeWidth={isHovered ? "6.2" : "5"}
                    fill="none"
                    strokeLinecap="round"
                  />
                  <path
                    d={edge.path}
                    stroke={isHovered ? "#e7f3ff" : "rgba(182, 205, 227, 0.88)"}
                    strokeWidth={isHovered ? "2.8" : "2.1"}
                    fill="none"
                    strokeLinecap="round"
                    markerEnd="url(#composer-arrow)"
                  />
                  <path
                    d={edge.path}
                    stroke="transparent"
                    strokeWidth="12"
                    fill="none"
                    className="cursor-pointer"
                    onClick={() => removeDagEdge(edge.fromNodeId, edge.toNodeId)}
                  />
                  {edge.branchLabel ? (
                    <g>
                      <rect
                        x={edge.labelX - labelWidth / 2}
                        y={edge.labelY - 15}
                        rx="10"
                        ry="10"
                        width={labelWidth}
                        height="22"
                        fill={labelTheme.background}
                        stroke={labelTheme.border}
                      />
                      <text
                        x={edge.labelX}
                        y={edge.labelY - 1}
                        textAnchor="middle"
                        fontSize="10"
                        fill={labelTheme.text}
                      >
                        {edge.branchLabel}
                      </text>
                    </g>
                  ) : null}
                  {isHovered ? (
                    <g
                      className="cursor-pointer"
                      onClick={() => removeDagEdge(edge.fromNodeId, edge.toNodeId)}
                    >
                      <circle
                        cx={edge.midX}
                        cy={edge.midY}
                        r="11"
                        fill="rgba(8, 15, 29, 0.96)"
                        stroke="rgba(251, 113, 133, 0.65)"
                      />
                      <text
                        x={edge.midX}
                        y={edge.midY + 4}
                        textAnchor="middle"
                        fontSize="12"
                        fill="#fecdd3"
                      >
                        ×
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}
            {dagConnectorPreview ? (
              <g data-composer-edge="true">
                <path
                  d={dagConnectorPreview.path}
                  stroke="rgba(15,23,42,0.35)"
                  strokeWidth="5"
                  fill="none"
                  strokeLinecap="round"
                />
                <path
                  d={dagConnectorPreview.path}
                  stroke="#8ed3ff"
                  strokeWidth="2.4"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray="8 5"
                  markerEnd="url(#composer-arrow)"
                />
              </g>
            ) : null}
          </svg>

          {showEmptyBlueprint
            ? emptyBlueprintNodes.map((node) => {
                const visual = resolveWorkflowNodeVisual({
                  capabilityId: node.capabilityId,
                  controlKind: node.controlKind,
                  nodeKind: node.nodeKind,
                  taskName: node.title,
                });
                const previewNode: ComposerDraftNode = {
                  id: node.id,
                  taskName: node.title,
                  capabilityId: node.capabilityId,
                  outputPath: "result",
                  nodeKind: node.nodeKind,
                  controlKind: node.controlKind,
                };
                const ports = outputPortsForNode(previewNode, visual, 96);
                return (
                  <div
                    key={`empty-blueprint-node-${node.id}`}
                    className="pointer-events-none absolute"
                    style={{
                      left: node.x,
                      top: node.y,
                    }}
                  >
                    <WorkflowNodeCard
                      title={node.title}
                      subtitle={node.subtitle}
                      caption={node.caption}
                      visual={visual}
                      tone={node.tone}
                      width={248}
                      height={96}
                      borderColor={workflowNodeCardToneStyles[node.tone].border}
                      shadow={workflowNodeCardToneStyles[node.tone].shadow}
                      badge={{
                        background:
                          node.tone === "rose"
                            ? "rgba(220, 38, 38, 0.92)"
                            : "rgba(37, 99, 235, 0.84)",
                        color: node.tone === "rose" ? "#fff1f2" : "#eff6ff",
                        label: node.tone === "rose" ? "!" : "✓",
                      }}
                      ports={ports.map((port) => ({
                        key: `${node.id}-${port.key}`,
                        label: port.label,
                        tone: port.tone,
                        top: port.y,
                      }))}
                    />
                  </div>
                );
              })
            : null}

          {dagCanvasNodes.map(({ node, position }) => {
            const nodeStatus = visualChainNodeStatusById.get(node.id);
            const missingCount = nodeStatus?.missingCount || 0;
            const requiredCount = nodeStatus?.requiredCount || 0;
            const isSelected = selectedDagNodeId === node.id;
            const visual = resolveWorkflowNodeVisual({
              capabilityId: node.capabilityId,
              controlKind: node.controlKind,
              nodeKind: node.nodeKind,
              taskName: node.taskName,
            });
            const tone = workflowNodeCardToneStyles[workflowNodeCardToneForVisual(visual)];
            const status = statusForNode(node, missingCount, requiredCount);
            const ports = outputPortsForNode(node, visual, nodeHeight);
            const isConnectorHoverTarget =
              dagConnectorDrag &&
              dagConnectorDrag.sourceNodeId !== node.id &&
              dagConnectorHoverTargetNodeId === node.id;
            const borderColor = isConnectorHoverTarget
              ? "#22c55e"
              : dagEdgeDraftSourceNodeId === node.id
                ? "#d97706"
                : isSelected
                  ? "#2563eb"
                  : tone.border;
            const cardShadow = isSelected
              ? `${tone.shadow}, 0 0 0 2px rgba(37, 99, 235, 0.22)`
              : dagEdgeDraftSourceNodeId === node.id
                ? `${tone.shadow}, 0 0 0 2px rgba(217, 119, 6, 0.18)`
                : tone.shadow;
            return (
              <div
                key={`composer-node-${node.id}`}
                data-composer-node="true"
                className="absolute"
                style={{
                  left: position.x,
                  top: position.y,
                  cursor: dagCanvasDraggingNodeId === node.id ? "grabbing" : "grab",
                }}
                onClick={() => setSelectedDagNodeId(node.id)}
                onMouseEnter={() => {
                  if (dagConnectorDrag && dagConnectorDrag.sourceNodeId !== node.id) {
                    setDagConnectorHoverTargetNodeId(node.id);
                  }
                }}
                onMouseLeave={() => {
                  setDagConnectorHoverTargetNodeId((prev) => (prev === node.id ? null : prev));
                }}
                onMouseUp={(event) => {
                  if (dagConnectorDrag && dagConnectorDrag.sourceNodeId !== node.id) {
                    event.preventDefault();
                    event.stopPropagation();
                    addDagEdge(
                      dagConnectorDrag.sourceNodeId,
                      node.id,
                      dagConnectorDrag.branchLabel
                    );
                    setDagConnectorDrag(null);
                    setDagConnectorHoverTargetNodeId(null);
                    setDagEdgeDraftSourceNodeId(null);
                  }
                }}
                onMouseDown={(event) => {
                  if (isInteractiveCanvasTarget(event.target)) {
                    return;
                  }
                  event.preventDefault();
                  beginDagNodeDrag(event, node.id);
                }}
              >
                <WorkflowNodeCard
                  title={node.taskName}
                  subtitle={subtitleForNode(node, visual)}
                  caption={status.label}
                  visual={visual}
                  tone={workflowNodeCardToneForVisual(visual)}
                  width={nodeWidth}
                  height={nodeHeight}
                  borderColor={borderColor}
                  shadow={cardShadow}
                  inputActive={Boolean(isConnectorHoverTarget)}
                  badge={{
                    background: status.badgeBackground,
                    color: status.badgeColor,
                    label: status.badgeLabel,
                    title: status.label,
                  }}
                  ports={ports.map((port) => ({
                    key: `${node.id}-${port.key}`,
                    label: port.label,
                    tone: port.tone,
                    top: port.y,
                    active:
                      dagConnectorDrag?.sourceNodeId === node.id &&
                      (dagConnectorDrag.branchLabel || "") === (port.branchLabel || ""),
                    title: `Drag ${port.label} connector`,
                    onMouseDown: (event) => {
                      event.stopPropagation();
                      beginDagConnectorDrag(event, node.id, {
                        branchLabel: port.branchLabel,
                        sourcePortY: port.y,
                      });
                    },
                  }))}
                />
              </div>
            );
          })}
        </div>
        </div>
      </div>
    </div>
  );
}
