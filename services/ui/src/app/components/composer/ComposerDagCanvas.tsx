"use client";

import type React from "react";

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
  branchLabel?: string;
};

type DagCanvasNode = {
  node: ComposerDraftNode;
  position: CanvasPoint;
};

type ComposerDagCanvasProps = {
  visualChainNodes: ComposerDraftNode[];
  dagEdgeDraftSourceNodeId: string | null;
  setDagEdgeDraftSourceNodeId: React.Dispatch<React.SetStateAction<string | null>>;
  setDagConnectorDrag: React.Dispatch<
    React.SetStateAction<{ sourceNodeId: string; x: number; y: number } | null>
  >;
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
  dagConnectorDrag: { sourceNodeId: string; x: number; y: number } | null;
  dagCanvasDraggingNodeId: string | null;
  dagConnectorHoverTargetNodeId: string | null;
  addDagEdge: (fromNodeId: string, toNodeId: string) => void;
  beginDagNodeDrag: (event: React.MouseEvent<HTMLDivElement>, nodeId: string) => void;
  isInteractiveCanvasTarget: (target: EventTarget | null) => boolean;
  beginDagConnectorDrag: (event: React.MouseEvent<HTMLButtonElement>, nodeId: string) => void;
  centerDagNodeInView: (nodeId: string) => void;
  nodeWidth: number;
  nodeHeight: number;
};

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
}: ComposerDagCanvasProps) {
  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
          DAG Canvas
        </div>
        <div className="flex items-center gap-2 text-[11px]">
          <button
            className="rounded-md border border-slate-300 px-2 py-1 text-slate-700"
            onClick={autoLayoutDagCanvas}
            disabled={visualChainNodes.length === 0}
          >
            Auto Layout
          </button>
          {dagEdgeDraftSourceNodeId ? (
            <button
              className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1 text-amber-700"
              onClick={() => {
                setDagEdgeDraftSourceNodeId(null);
                setDagConnectorDrag(null);
                setDagConnectorHoverTargetNodeId(null);
              }}
            >
              Edge source:{" "}
              {visualChainNodes.find((node) => node.id === dagEdgeDraftSourceNodeId)?.taskName ||
                dagEdgeDraftSourceNodeId}{" "}
              (cancel)
            </button>
          ) : (
            <span className="text-slate-500">
              Drag from a node&apos;s right connector to another node, or use &quot;Start Edge&quot;.
            </span>
          )}
        </div>
      </div>
      <div
        ref={dagCanvasViewportRef}
        className="mt-2 overflow-auto rounded-md border border-slate-200 bg-white"
      >
        <div
          ref={dagCanvasRef}
          className="relative bg-[radial-gradient(circle_at_1px_1px,#e2e8f0_1px,transparent_0)] [background-size:16px_16px]"
          style={{ width: dagCanvasSurface.width, height: dagCanvasSurface.height }}
        >
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
                <path d="M0,0 L0,6 L9,3 z" fill="#94a3b8" />
              </marker>
            </defs>
            {dagCanvasEdges.map((edge) => {
              const isHovered = hoveredDagEdgeKey === edge.edgeKey;
              return (
                <g
                  key={`composer-edge-${edge.edgeKey}`}
                  onMouseEnter={() => setHoveredDagEdgeKey(edge.edgeKey)}
                  onMouseLeave={() =>
                    setHoveredDagEdgeKey((prev) => (prev === edge.edgeKey ? null : prev))
                  }
                >
                  <path
                    d={edge.path}
                    stroke={isHovered ? "#475569" : "#94a3b8"}
                    strokeWidth={isHovered ? "2.5" : "1.5"}
                    fill="none"
                    markerEnd="url(#composer-arrow)"
                  />
                  <path
                    d={edge.path}
                    stroke="transparent"
                    strokeWidth="10"
                    fill="none"
                    className="cursor-pointer"
                    onClick={() => removeDagEdge(edge.fromNodeId, edge.toNodeId)}
                  />
                  {edge.branchLabel ? (
                    <g>
                      <rect
                        x={edge.midX - 24}
                        y={edge.midY - 18}
                        rx="8"
                        ry="8"
                        width="48"
                        height="18"
                        fill="white"
                        stroke="#cbd5e1"
                      />
                      <text
                        x={edge.midX}
                        y={edge.midY - 6}
                        textAnchor="middle"
                        fontSize="10"
                        fill="#475569"
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
                      <circle cx={edge.midX} cy={edge.midY} r="10" fill="white" stroke="#cbd5e1" />
                      <text
                        x={edge.midX}
                        y={edge.midY + 3}
                        textAnchor="middle"
                        fontSize="12"
                        fill="#475569"
                      >
                        ×
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}
            {dagConnectorPreview ? (
              <path
                d={dagConnectorPreview.path}
                stroke="#0ea5e9"
                strokeWidth="2"
                fill="none"
                strokeDasharray="6 4"
                markerEnd="url(#composer-arrow)"
              />
            ) : null}
          </svg>
          {dagCanvasNodes.map(({ node, position }) => {
            const edgeFromSource =
              dagEdgeDraftSourceNodeId &&
              dagEdgeDraftSourceNodeId !== node.id &&
              composerDraftEdges.some(
                (edge) =>
                  edge.fromNodeId === dagEdgeDraftSourceNodeId && edge.toNodeId === node.id
              );
            const incomingCount = dagNodeAdjacency.incoming[node.id] || 0;
            const outgoingCount = dagNodeAdjacency.outgoing[node.id] || 0;
            const nodeStatus = visualChainNodeStatusById.get(node.id);
            const missingCount = nodeStatus?.missingCount || 0;
            const requiredCount = nodeStatus?.requiredCount || 0;
            const isSelected = selectedDagNodeId === node.id;
            const isControlNode = node.nodeKind === "control";
            const isConnectorHoverTarget =
              dagConnectorDrag &&
              dagConnectorDrag.sourceNodeId !== node.id &&
              dagConnectorHoverTargetNodeId === node.id;
            return (
              <div
                key={`composer-node-${node.id}`}
                className={`absolute rounded-xl border bg-white shadow-sm ${
                  isConnectorHoverTarget
                    ? "border-emerald-400 ring-2 ring-emerald-100"
                    : isSelected
                      ? "border-sky-400 ring-2 ring-sky-100"
                      : isControlNode
                        ? "border-amber-300"
                        : "border-slate-300"
                }`}
                style={{
                  left: position.x,
                  top: position.y,
                  width: nodeWidth,
                  minHeight: nodeHeight,
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
                    addDagEdge(dagConnectorDrag.sourceNodeId, node.id);
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
                <div
                  className={`absolute -left-2 top-1/2 h-3 w-3 -translate-y-1/2 rounded-full border ${
                    isConnectorHoverTarget
                      ? "border-emerald-500 bg-emerald-400"
                      : "border-slate-300 bg-white"
                  }`}
                />
                <button
                  type="button"
                  className={`absolute -right-2 top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border text-[10px] ${
                    dagConnectorDrag?.sourceNodeId === node.id
                      ? "border-sky-500 bg-sky-500 text-white"
                      : "border-slate-300 bg-white text-slate-600 hover:border-sky-400 hover:text-sky-600"
                  }`}
                  title="Drag to connect"
                  onMouseDown={(event) => {
                    event.stopPropagation();
                    beginDagConnectorDrag(event, node.id);
                  }}
                >
                  +
                </button>
                <div className={`border-b px-2 py-1 ${isControlNode ? "border-amber-200 bg-amber-50" : "border-slate-200 bg-slate-50"}`}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-[11px] font-semibold text-slate-800">{node.taskName}</div>
                    <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] text-slate-500">
                      in {incomingCount} • out {outgoingCount}
                    </span>
                  </div>
                  {isControlNode ? (
                    <div className="mt-1 inline-flex rounded-full bg-amber-100 px-1.5 py-0.5 text-[9px] text-amber-700">
                      control {node.controlKind || "node"}
                    </div>
                  ) : requiredCount > 0 ? (
                    <div
                      className={`mt-1 inline-flex rounded-full px-1.5 py-0.5 text-[9px] ${
                        missingCount > 0 ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      {missingCount > 0
                        ? `${missingCount} missing`
                        : `${requiredCount}/${requiredCount} ready`}
                    </div>
                  ) : (
                    <div className="mt-1 inline-flex rounded-full bg-slate-100 px-1.5 py-0.5 text-[9px] text-slate-600">
                      no required inputs
                    </div>
                  )}
                  <div className="text-[10px] text-slate-500">{node.capabilityId}</div>
                </div>
                <div className="space-y-1 px-2 py-1">
                  <div className="text-[10px] text-slate-500">output: {node.outputPath}</div>
                  <div className="text-[10px] text-slate-500">Drag card to move node</div>
                  <div className="flex flex-wrap gap-1">
                    <button
                      className="rounded border border-slate-300 px-1.5 py-0.5 text-[10px] text-slate-700"
                      onClick={() => {
                        setSelectedDagNodeId(node.id);
                        centerDagNodeInView(node.id);
                      }}
                    >
                      Center
                    </button>
                    <button
                      className={`rounded border px-1.5 py-0.5 text-[10px] ${
                        dagEdgeDraftSourceNodeId === node.id
                          ? "border-amber-300 bg-amber-50 text-amber-700"
                          : "border-slate-300 text-slate-700"
                      }`}
                      onClick={() => {
                        setSelectedDagNodeId(node.id);
                        setDagEdgeDraftSourceNodeId(node.id);
                      }}
                    >
                      {dagEdgeDraftSourceNodeId === node.id ? "Edge Source" : "Start Edge"}
                    </button>
                    {dagEdgeDraftSourceNodeId && dagEdgeDraftSourceNodeId !== node.id ? (
                      <button
                        className={`rounded border px-1.5 py-0.5 text-[10px] ${
                          edgeFromSource
                            ? "border-rose-300 bg-rose-50 text-rose-700"
                            : "border-emerald-300 bg-emerald-50 text-emerald-700"
                        }`}
                        onClick={() => {
                          if (edgeFromSource) {
                            removeDagEdge(dagEdgeDraftSourceNodeId, node.id);
                          } else {
                            addDagEdge(dagEdgeDraftSourceNodeId, node.id);
                          }
                          setSelectedDagNodeId(node.id);
                          setDagEdgeDraftSourceNodeId(null);
                        }}
                      >
                        {edgeFromSource ? "Disconnect" : "Connect"}
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {dagCanvasEdges.length > 0 ? (
        <div className="mt-2 flex flex-wrap items-center gap-1 text-[10px]">
          {dagCanvasEdges.map((edge) => (
            <button
              key={`composer-edge-chip-${edge.edgeKey}`}
              className={`rounded-md border px-1.5 py-0.5 ${
                hoveredDagEdgeKey === edge.edgeKey
                  ? "border-rose-300 bg-rose-50 text-rose-700"
                  : "border-slate-200 bg-white text-slate-600"
              }`}
              onMouseEnter={() => setHoveredDagEdgeKey(edge.edgeKey)}
              onMouseLeave={() => setHoveredDagEdgeKey((prev) => (prev === edge.edgeKey ? null : prev))}
              onClick={() => removeDagEdge(edge.fromNodeId, edge.toNodeId)}
            >
              {edge.fromTaskName} → {edge.toTaskName} ×
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
