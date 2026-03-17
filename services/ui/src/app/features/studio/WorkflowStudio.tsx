"use client";

import Link from "next/link";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import ComposerDagCanvas from "../../components/composer/ComposerDagCanvas";
import ComposerValidationPanel from "../../components/composer/ComposerValidationPanel";
import StudioCapabilityPalette from "./StudioCapabilityPalette";
import StudioCompilePanel from "./StudioCompilePanel";
import StudioNodeInspector from "./StudioNodeInspector";
import type {
  CanvasPoint,
  CapabilityCatalog,
  ChainPreflightResult,
  ComposerCompileResponse,
  ComposerDraft,
  ComposerDraftEdge,
  ComposerDraftNode,
  ComposerInputBinding,
  ComposerIssueFocus,
} from "./types";
import {
  CHAINABLE_REQUIRED_FIELDS,
  DAG_CANVAS_MIN_HEIGHT,
  DAG_CANVAS_MIN_WIDTH,
  DAG_CANVAS_NODE_HEIGHT,
  DAG_CANVAS_NODE_WIDTH,
  DAG_CANVAS_PADDING,
  DAG_CANVAS_SNAP,
  capabilityInputSchemaProperties,
  collectContextPathSuggestions,
  collectComposerValidationIssues,
  defaultDagNodePosition,
  detectDagCycle,
  formatTimestamp,
  getCapabilityRequiredInputs,
  inferCapabilityOutputPath,
  inferOutputExtensionForCapability,
  isContextInputPresent,
  isInteractiveCanvasTarget,
  isPathOutputReference,
  normalizeComposerEdges,
  outputPathSuggestionsForNode,
  readContextObject,
  schemaPropertyTypeLabel,
  taskNameFromCapability,
  uniqueTaskName,
} from "./utils";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "/api";

const initialStudioDraft = (): ComposerDraft => ({
  summary: "Workflow Studio draft",
  nodes: [],
  edges: [],
});

const createStudioOutput = () => ({
  id: `output-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  name: "",
  path: "",
  description: "",
});

const createStudioVariable = () => ({
  id: `variable-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
  key: "",
  value: "",
  description: "",
});

const initialContextJson = () =>
  JSON.stringify(
    {
      today: new Date().toISOString().slice(0, 10),
      output_dir: "documents",
    },
    null,
    2
  );

export default function WorkflowStudio() {
  const [goal, setGoal] = useState("");
  const [contextJson, setContextJson] = useState(initialContextJson);
  const [composerDraft, setComposerDraft] = useState<ComposerDraft>(initialStudioDraft);
  const [capabilityCatalog, setCapabilityCatalog] = useState<CapabilityCatalog | null>(null);
  const [capabilityLoading, setCapabilityLoading] = useState(true);
  const [capabilityError, setCapabilityError] = useState<string | null>(null);
  const [paletteQuery, setPaletteQuery] = useState("");
  const [paletteGroup, setPaletteGroup] = useState("all");
  const [selectedDagNodeId, setSelectedDagNodeId] = useState<string | null>(null);
  const [studioNotice, setStudioNotice] = useState<string | null>(null);
  const [chainPreflightLoading, setChainPreflightLoading] = useState(false);
  const [composerCompileLoading, setComposerCompileLoading] = useState(false);
  const [chainPreflightResult, setChainPreflightResult] = useState<ChainPreflightResult | null>(null);
  const [composerCompileResult, setComposerCompileResult] = useState<ComposerCompileResponse | null>(null);
  const [activeComposerIssueFocus, setActiveComposerIssueFocus] = useState<ComposerIssueFocus | null>(
    null
  );
  const [composerNodePositions, setComposerNodePositions] = useState<Record<string, CanvasPoint>>({});
  const [hoveredDagEdgeKey, setHoveredDagEdgeKey] = useState<string | null>(null);
  const [dagEdgeDraftSourceNodeId, setDagEdgeDraftSourceNodeId] = useState<string | null>(null);
  const [dagConnectorDrag, setDagConnectorDrag] = useState<{
    sourceNodeId: string;
    x: number;
    y: number;
  } | null>(null);
  const [dagCanvasDraggingNodeId, setDagCanvasDraggingNodeId] = useState<string | null>(null);
  const [dagConnectorHoverTargetNodeId, setDagConnectorHoverTargetNodeId] = useState<string | null>(
    null
  );

  const dagCanvasDragOffsetRef = useRef<CanvasPoint>({ x: 0, y: 0 });
  const dagCanvasViewportRef = useRef<HTMLDivElement | null>(null);
  const dagCanvasRef = useRef<HTMLDivElement | null>(null);
  const inspectorBindingRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const deferredPaletteQuery = useDeferredValue(paletteQuery);
  const visualChainNodes = composerDraft.nodes;
  const composerDraftEdges = composerDraft.edges;
  const contextState = useMemo(() => readContextObject(contextJson), [contextJson]);
  const contextPathSuggestions = useMemo(
    () => collectContextPathSuggestions(contextState.context),
    [contextState.context]
  );

  useEffect(() => {
    let cancelled = false;
    const loadCapabilities = async () => {
      setCapabilityLoading(true);
      setCapabilityError(null);
      try {
        const response = await fetch(`${apiUrl}/capabilities?with_schemas=true`);
        if (!response.ok) {
          throw new Error(`Capability catalog request failed (${response.status}).`);
        }
        const data = (await response.json()) as CapabilityCatalog;
        if (!cancelled) {
          setCapabilityCatalog(data);
        }
      } catch (error) {
        if (!cancelled) {
          setCapabilityCatalog(null);
          setCapabilityError(
            error instanceof Error
              ? error.message
              : `Network error while loading capabilities from ${apiUrl}/capabilities?with_schemas=true.`
          );
        }
      } finally {
        if (!cancelled) {
          setCapabilityLoading(false);
        }
      }
    };
    void loadCapabilities();
    return () => {
      cancelled = true;
    };
  }, []);

  const availableCapabilities = useMemo(() => {
    const items = capabilityCatalog?.items || [];
    return [...items].sort((left, right) => left.id.localeCompare(right.id));
  }, [capabilityCatalog]);

  const capabilityById = useMemo(
    () => new Map(availableCapabilities.map((item) => [item.id, item])),
    [availableCapabilities]
  );

  const paletteGroups = useMemo(
    () =>
      Array.from(
        new Set(
          availableCapabilities
            .map((item) => (item.group || "").trim())
            .filter(Boolean)
        )
      ).sort((left, right) => left.localeCompare(right)),
    [availableCapabilities]
  );

  const paletteCapabilities = useMemo(() => {
    const query = deferredPaletteQuery.trim().toLowerCase();
    return availableCapabilities.filter((item) => {
      if (paletteGroup !== "all" && (item.group || "").trim() !== paletteGroup) {
        return false;
      }
      if (!query) {
        return true;
      }
      const haystack = [
        item.id,
        item.description,
        item.group,
        item.subgroup,
        ...(item.tags || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [availableCapabilities, deferredPaletteQuery, paletteGroup]);

  useEffect(() => {
    setComposerNodePositions((prev) => {
      const next = { ...prev };
      let changed = false;
      visualChainNodes.forEach((node, index) => {
        if (!next[node.id]) {
          next[node.id] = defaultDagNodePosition(index);
          changed = true;
        }
      });
      Object.keys(next).forEach((nodeId) => {
        if (!visualChainNodes.some((node) => node.id === nodeId)) {
          delete next[nodeId];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [visualChainNodes]);

  useEffect(() => {
    if (!selectedDagNodeId) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === selectedDagNodeId)) {
      setSelectedDagNodeId(null);
    }
  }, [selectedDagNodeId, visualChainNodes]);

  useEffect(() => {
    if (!dagEdgeDraftSourceNodeId) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === dagEdgeDraftSourceNodeId)) {
      setDagEdgeDraftSourceNodeId(null);
    }
  }, [dagEdgeDraftSourceNodeId, visualChainNodes]);

  useEffect(() => {
    if (!dagConnectorDrag) {
      return;
    }
    if (!visualChainNodes.some((node) => node.id === dagConnectorDrag.sourceNodeId)) {
      setDagConnectorDrag(null);
      setDagConnectorHoverTargetNodeId(null);
    }
  }, [dagConnectorDrag, visualChainNodes]);

  useEffect(() => {
    setChainPreflightResult(null);
    setComposerCompileResult(null);
    setActiveComposerIssueFocus(null);
  }, [goal, contextJson, visualChainNodes, composerDraftEdges]);

  useEffect(() => {
    if (!dagCanvasDraggingNodeId && !dagConnectorDrag) {
      return;
    }
    const handleMove = (event: MouseEvent) => {
      const canvas = dagCanvasRef.current;
      if (!canvas) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (dagCanvasDraggingNodeId) {
        const rawX = event.clientX - rect.left - dagCanvasDragOffsetRef.current.x;
        const rawY = event.clientY - rect.top - dagCanvasDragOffsetRef.current.y;
        const x = Math.max(0, Math.round(rawX / DAG_CANVAS_SNAP) * DAG_CANVAS_SNAP);
        const y = Math.max(0, Math.round(rawY / DAG_CANVAS_SNAP) * DAG_CANVAS_SNAP);
        setComposerNodePositions((prev) => ({
          ...prev,
          [dagCanvasDraggingNodeId]: { x, y },
        }));
      }
      if (dagConnectorDrag) {
        setDagConnectorDrag((prev) =>
          prev
            ? {
                ...prev,
                x: Math.max(0, event.clientX - rect.left),
                y: Math.max(0, event.clientY - rect.top),
              }
            : prev
        );
      }
    };

    const handleUp = () => {
      if (dagCanvasDraggingNodeId) {
        setDagCanvasDraggingNodeId(null);
      }
      if (dagConnectorDrag) {
        setDagConnectorDrag(null);
        setDagConnectorHoverTargetNodeId(null);
        setDagEdgeDraftSourceNodeId(null);
      }
    };

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
  }, [dagCanvasDraggingNodeId, dagConnectorDrag]);

  const setVisualChainNodes = (
    next: ComposerDraftNode[] | ((prev: ComposerDraftNode[]) => ComposerDraftNode[])
  ) => {
    setComposerDraft((prev) => {
      const nextNodes =
        typeof next === "function"
          ? (next as (nodes: ComposerDraftNode[]) => ComposerDraftNode[])(prev.nodes)
          : next;
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, prev.edges),
      };
    });
  };

  const centerDagNodeInView = (nodeId: string) => {
    const viewport = dagCanvasViewportRef.current;
    if (!viewport) {
      return;
    }
    const fallbackIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    const position =
      composerNodePositions[nodeId] ||
      (fallbackIndex >= 0 ? defaultDagNodePosition(fallbackIndex) : null);
    if (!position) {
      return;
    }
    const left = Math.max(0, position.x - Math.max(0, viewport.clientWidth - DAG_CANVAS_NODE_WIDTH) / 2);
    const top = Math.max(0, position.y - Math.max(0, viewport.clientHeight - DAG_CANVAS_NODE_HEIGHT) / 2);
    viewport.scrollTo({ left, top, behavior: "smooth" });
  };

  const setVisualBindingFromSource = (
    nodeId: string,
    field: string,
    sourceNodeId: string,
    preferredPath?: string
  ) => {
    setVisualChainNodes((prev) => {
      const sourceNode = prev.find((node) => node.id === sourceNodeId);
      if (!sourceNode) {
        return prev;
      }
      return prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const existing = node.inputBindings[field];
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              kind: "step_output",
              sourceNodeId: sourceNode.id,
              sourcePath:
                preferredPath ||
                (existing?.kind === "step_output" ? existing.sourcePath : "") ||
                sourceNode.outputPath ||
                "result",
            },
          },
        };
      });
    });
  };

  const setVisualBindingFromPrevious = (nodeId: string, field: string) => {
    const targetIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    if (targetIndex <= 0) {
      return;
    }
    const previousNode = visualChainNodes[targetIndex - 1];
    if (!previousNode) {
      return;
    }
    setVisualBindingFromSource(nodeId, field, previousNode.id, previousNode.outputPath || "result");
    setStudioNotice(`Wired ${field} from previous step.`);
  };

  const clearVisualBinding = (nodeId: string, field: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const nextBindings = { ...node.inputBindings };
        delete nextBindings[field];
        return { ...node, inputBindings: nextBindings };
      })
    );
  };

  const setVisualBindingLiteral = (nodeId: string, field: string, value: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "literal", value },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingContext = (nodeId: string, field: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: { kind: "context", path: field },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingMemory = (
    nodeId: string,
    field: string,
    scope: "job" | "global" = "job"
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              inputBindings: {
                ...node.inputBindings,
                [field]: {
                  kind: "memory",
                  scope,
                  name: "task_outputs",
                },
              },
            }
          : node
      )
    );
  };

  const setVisualBindingMode = (
    nodeId: string,
    field: string,
    mode: "context" | "from" | "literal" | "memory"
  ) => {
    if (mode === "context") {
      setVisualBindingContext(nodeId, field);
      return;
    }
    if (mode === "literal") {
      setVisualBindingLiteral(nodeId, field, "");
      return;
    }
    if (mode === "memory") {
      setVisualBindingMemory(nodeId, field, "job");
      return;
    }
    const sourceNodes = visualChainNodes.filter((node) => node.id !== nodeId);
    if (sourceNodes.length === 0) {
      setStudioNotice(`No source step is available for ${field}.`);
      return;
    }
    const sourceNode = sourceNodes[sourceNodes.length - 1];
    setVisualBindingFromSource(nodeId, field, sourceNode.id, sourceNode.outputPath || "result");
  };

  const updateVisualBindingSourceNode = (nodeId: string, field: string, sourceNodeId: string) => {
    const sourceNode = visualChainNodes.find((node) => node.id === sourceNodeId);
    if (!sourceNode) {
      return;
    }
    setVisualBindingFromSource(nodeId, field, sourceNodeId, sourceNode.outputPath || "result");
  };

  const updateVisualBindingPath = (nodeId: string, field: string, sourcePath: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "step_output") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              sourcePath,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingLiteral = (nodeId: string, field: string, value: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "literal") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              value,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingContextPath = (nodeId: string, field: string, path: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "context") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              path,
            },
          },
        };
      })
    );
  };

  const updateVisualBindingMemory = (
    nodeId: string,
    field: string,
    patch: { scope?: "job" | "global"; name?: string; key?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const currentBinding = node.inputBindings[field];
        if (!currentBinding || currentBinding.kind !== "memory") {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [field]: {
              ...currentBinding,
              ...patch,
            },
          },
        };
      })
    );
  };

  const addCustomInputField = (nodeId: string, field: string) => {
    const trimmed = field.trim();
    if (!trimmed) {
      return;
    }
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId || node.inputBindings[trimmed]) {
          return node;
        }
        return {
          ...node,
          inputBindings: {
            ...node.inputBindings,
            [trimmed]: {
              kind: "context",
              path: trimmed,
            },
          },
        };
      })
    );
  };

  const removeCustomInputField = (nodeId: string, field: string) => {
    clearVisualBinding(nodeId, field);
  };

  const addNodeOutput = (nodeId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, outputs: [...node.outputs, createStudioOutput()] }
          : node
      )
    );
  };

  const updateNodeOutput = (
    nodeId: string,
    outputId: string,
    patch: { name?: string; path?: string; description?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        return {
          ...node,
          outputs: node.outputs.map((output) =>
            output.id === outputId ? { ...output, ...patch } : output
          ),
        };
      })
    );
  };

  const removeNodeOutput = (nodeId: string, outputId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, outputs: node.outputs.filter((output) => output.id !== outputId) }
          : node
      )
    );
  };

  const addNodeVariable = (nodeId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, variables: [...node.variables, createStudioVariable()] }
          : node
      )
    );
  };

  const updateNodeVariable = (
    nodeId: string,
    variableId: string,
    patch: { key?: string; value?: string; description?: string }
  ) => {
    setVisualChainNodes((prev) =>
      prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        return {
          ...node,
          variables: node.variables.map((variable) =>
            variable.id === variableId ? { ...variable, ...patch } : variable
          ),
        };
      })
    );
  };

  const removeNodeVariable = (nodeId: string, variableId: string) => {
    setVisualChainNodes((prev) =>
      prev.map((node) =>
        node.id === nodeId
          ? { ...node, variables: node.variables.filter((variable) => variable.id !== variableId) }
          : node
      )
    );
  };

  const addCapabilityNodeToStudio = (capabilityId: string) => {
    const capability = capabilityById.get(capabilityId);
    const context = contextState.context;
    const nodeId = `studio-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    setComposerDraft((prev) => {
      const anchorNode =
        prev.nodes.find((node) => node.id === selectedDagNodeId) || prev.nodes[prev.nodes.length - 1] || null;
      const inputBindings: Record<string, ComposerInputBinding> = {};
      const requiredInputs = getCapabilityRequiredInputs(capability);
      requiredInputs.forEach((field) => {
        if (!CHAINABLE_REQUIRED_FIELDS.has(field)) {
          return;
        }
        if (isContextInputPresent(context[field])) {
          return;
        }
        if (!anchorNode) {
          return;
        }
        inputBindings[field] = {
          kind: "step_output",
          sourceNodeId: anchorNode.id,
          sourcePath: anchorNode.outputPath || "result",
        };
      });

      const nextNodes = [
        ...prev.nodes,
        {
          id: nodeId,
          taskName: uniqueTaskName(taskNameFromCapability(capabilityId), prev.nodes),
          capabilityId,
          outputPath: inferCapabilityOutputPath(capabilityId),
          inputBindings,
          outputs: [],
          variables: [],
        },
      ];
      const nextEdges = anchorNode ? [...prev.edges, { fromNodeId: anchorNode.id, toNodeId: nodeId }] : prev.edges;
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, nextEdges),
      };
    });

    setComposerNodePositions((prev) => {
      const anchorPosition = selectedDagNodeId ? prev[selectedDagNodeId] : null;
      return {
        ...prev,
        [nodeId]: anchorPosition
          ? {
              x: anchorPosition.x + DAG_CANVAS_NODE_WIDTH + 64,
              y: anchorPosition.y,
            }
          : defaultDagNodePosition(Object.keys(prev).length),
      };
    });
    setSelectedDagNodeId(nodeId);
    setStudioNotice(`Added ${capabilityId} to the workflow.`);
  };

  const updateVisualChainNode = (
    nodeId: string,
    patch: Partial<Pick<ComposerDraftNode, "taskName" | "capabilityId" | "outputPath">>
  ) => {
    setVisualChainNodes((prev) => {
      const current = prev.find((node) => node.id === nodeId);
      if (!current) {
        return prev;
      }
      return prev.map((node) => {
        if (node.id !== nodeId) {
          return node;
        }
        const nextTaskName =
          patch.taskName !== undefined ? uniqueTaskName(patch.taskName, prev, nodeId) : node.taskName;
        const nextCapabilityId = patch.capabilityId ?? node.capabilityId;
        let nextOutputPath = patch.outputPath ?? node.outputPath;
        if (patch.capabilityId && patch.capabilityId !== node.capabilityId && !patch.outputPath) {
          nextOutputPath = inferCapabilityOutputPath(patch.capabilityId);
        }
        return {
          ...node,
          taskName: nextTaskName,
          capabilityId: nextCapabilityId,
          outputPath: nextOutputPath,
        };
      });
    });
  };

  const removeVisualChainNode = (nodeId: string) => {
    setComposerDraft((prev) => {
      const nextNodes = prev.nodes
        .filter((node) => node.id !== nodeId)
        .map((node) => {
          const nextBindings: Record<string, ComposerInputBinding> = {};
          Object.entries(node.inputBindings).forEach(([field, binding]) => {
            if (binding.kind === "step_output" && binding.sourceNodeId === nodeId) {
              return;
            }
            nextBindings[field] = binding;
          });
          return { ...node, inputBindings: nextBindings };
        });
      const nextEdges = prev.edges.filter(
        (edge) => edge.fromNodeId !== nodeId && edge.toNodeId !== nodeId
      );
      return {
        ...prev,
        nodes: nextNodes,
        edges: normalizeComposerEdges(nextNodes, nextEdges),
      };
    });
    setComposerNodePositions((prev) => {
      const next = { ...prev };
      delete next[nodeId];
      return next;
    });
    setSelectedDagNodeId((prev) => (prev === nodeId ? null : prev));
    setStudioNotice("Removed step from workflow.");
  };

  const autoBindTargetFromSource = (sourceNodeId: string, targetNodeId: string) => {
    const sourceNode = visualChainNodes.find((node) => node.id === sourceNodeId);
    const targetNode = visualChainNodes.find((node) => node.id === targetNodeId);
    if (!sourceNode || !targetNode) {
      return false;
    }
    const targetStatus = visualChainNodeStatusById.get(targetNodeId);
    if (!targetStatus || targetStatus.requiredStatus.length === 0) {
      return false;
    }
    const missingFields = targetStatus.requiredStatus
      .filter((status) => status.status === "missing")
      .map((status) => status.field);
    const candidateField = missingFields[0] || targetStatus.requiredStatus[0]?.field || "";
    if (!candidateField) {
      return false;
    }
    setVisualBindingFromSource(targetNodeId, candidateField, sourceNodeId, sourceNode.outputPath || "result");
    return true;
  };

  const addDagEdge = (fromNodeId: string, toNodeId: string) => {
    if (!fromNodeId || !toNodeId || fromNodeId === toNodeId) {
      return;
    }
    setComposerDraft((prev) => ({
      ...prev,
      edges: normalizeComposerEdges(prev.nodes, [...prev.edges, { fromNodeId, toNodeId }]),
    }));
    const mapped = autoBindTargetFromSource(fromNodeId, toNodeId);
    setSelectedDagNodeId(toNodeId);
    centerDagNodeInView(toNodeId);
    setStudioNotice(mapped ? "Connected edge and wired a missing input." : "Connected DAG edge.");
  };

  const removeDagEdge = (fromNodeId: string, toNodeId: string) => {
    setHoveredDagEdgeKey(null);
    setComposerDraft((prev) => ({
      ...prev,
      edges: prev.edges.filter(
        (edge) => !(edge.fromNodeId === fromNodeId && edge.toNodeId === toNodeId)
      ),
    }));
  };

  const beginDagNodeDrag = (event: React.MouseEvent<HTMLDivElement>, nodeId: string) => {
    const canvas = dagCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const current = composerNodePositions[nodeId] || defaultDagNodePosition(0);
    dagCanvasDragOffsetRef.current = {
      x: event.clientX - rect.left - current.x,
      y: event.clientY - rect.top - current.y,
    };
    setDagCanvasDraggingNodeId(nodeId);
  };

  const beginDagConnectorDrag = (
    event: React.MouseEvent<HTMLButtonElement>,
    sourceNodeId: string
  ) => {
    event.preventDefault();
    const canvas = dagCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    setDagConnectorDrag({
      sourceNodeId,
      x: Math.max(0, event.clientX - rect.left),
      y: Math.max(0, event.clientY - rect.top),
    });
    setDagConnectorHoverTargetNodeId(null);
    setDagEdgeDraftSourceNodeId(sourceNodeId);
  };

  const autoLayoutDagCanvas = () => {
    setComposerNodePositions(() => {
      const next: Record<string, CanvasPoint> = {};
      visualChainNodes.forEach((node, index) => {
        next[node.id] = defaultDagNodePosition(index);
      });
      return next;
    });
    setStudioNotice("Auto-layout applied to canvas.");
  };

  const buildDeriveOutputBindings = (
    context: Record<string, unknown>,
    targetCapabilityId: string,
    documentTypeHint: string
  ): Record<string, ComposerInputBinding> => {
    const extension = inferOutputExtensionForCapability(targetCapabilityId);
    const topicValue =
      typeof context.topic === "string" && context.topic.trim().length > 0
        ? context.topic.trim()
        : "generated_document";
    const outputDirValue =
      typeof context.output_dir === "string" && context.output_dir.trim().length > 0
        ? context.output_dir.trim()
        : "documents";
    const todayValue =
      typeof context.today === "string" && context.today.trim().length > 0
        ? context.today.trim()
        : new Date().toISOString().slice(0, 10);

    return {
      topic: { kind: "literal", value: topicValue },
      output_dir: { kind: "literal", value: outputDirValue },
      document_type: { kind: "literal", value: documentTypeHint || "document" },
      output_extension: { kind: "literal", value: extension },
      today: { kind: "literal", value: todayValue },
    };
  };

  const insertDeriveOutputPathStepForNode = (nodeId: string) => {
    const context = contextState.context;
    setComposerDraft((prev) => {
      const targetIndex = prev.nodes.findIndex((node) => node.id === nodeId);
      if (targetIndex < 0) {
        return prev;
      }
      const targetNode = prev.nodes[targetIndex];
      if (targetNode.capabilityId === "document.output.derive") {
        return prev;
      }

      const deriveNodeId = `studio-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const deriveNode: ComposerDraftNode = {
        id: deriveNodeId,
        taskName: uniqueTaskName("DeriveOutputPath", prev.nodes),
        capabilityId: "document.output.derive",
        outputPath: "path",
        inputBindings: buildDeriveOutputBindings(
          context,
          targetNode.capabilityId,
          targetNode.capabilityId.includes("runbook") ? "runbook" : "document"
        ),
        outputs: [],
        variables: [],
      };

      const updatedTargetNode: ComposerDraftNode = {
        ...targetNode,
        inputBindings: {
          ...targetNode.inputBindings,
          path: {
            kind: "step_output",
            sourceNodeId: deriveNodeId,
            sourcePath: "path",
          },
        },
      };

      const nextNodes = [
        ...prev.nodes.slice(0, targetIndex),
        deriveNode,
        updatedTargetNode,
        ...prev.nodes.slice(targetIndex + 1),
      ];
      const nextEdges = normalizeComposerEdges(nextNodes, [
        ...prev.edges,
        { fromNodeId: deriveNodeId, toNodeId: targetNode.id },
      ]);

      return {
        ...prev,
        nodes: nextNodes,
        edges: nextEdges,
      };
    });
    setSelectedDagNodeId(nodeId);
  };

  const visualChainNodesWithStatus = useMemo(() => {
    const context = contextState.context;
    return visualChainNodes.map((node, index) => {
      const capability = capabilityById.get(node.capabilityId);
      const schemaProperties = capabilityInputSchemaProperties(capability);
      const required = getCapabilityRequiredInputs(capability);
      const requiredStatus = required.map((field) => {
        const property = schemaProperties[field];
        const schemaType = schemaPropertyTypeLabel(property);
        const schemaDescription =
          property && typeof property.description === "string" ? property.description : "";
        const binding = node.inputBindings[field];
        if (binding?.kind === "step_output") {
          const source = visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId);
          return {
            field,
            status: "from_chain" as const,
            detail: source ? `${source.taskName}.${binding.sourcePath}` : binding.sourcePath,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "literal" && binding.value.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: "literal value",
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "memory" && binding.name.trim()) {
          return {
            field,
            status: "provided" as const,
            detail: `memory:${binding.scope}/${binding.name}${binding.key ? `/${binding.key}` : ""}`,
            schemaType,
            schemaDescription,
          };
        }
        if (binding?.kind === "context" && binding.path.trim()) {
          return {
            field,
            status: "from_context" as const,
            detail: binding.path,
            schemaType,
            schemaDescription,
          };
        }
        if (isContextInputPresent(context[field])) {
          return {
            field,
            status: "from_context" as const,
            detail: "context_json",
            schemaType,
            schemaDescription,
          };
        }
        return {
          field,
          status: "missing" as const,
          detail: "missing",
          schemaType,
          schemaDescription,
        };
      });
      return {
        node,
        index,
        requiredStatus,
      };
    });
  }, [capabilityById, contextState.context, visualChainNodes]);

  const visualChainNodeStatusById = useMemo(() => {
    const next = new Map<
      string,
      {
        missingCount: number;
        resolvedCount: number;
        requiredCount: number;
        requiredStatus: {
          field: string;
          status: "missing" | "from_chain" | "from_context" | "provided";
          detail: string;
          schemaType: string;
          schemaDescription: string;
        }[];
      }
    >();
    visualChainNodesWithStatus.forEach(({ node, requiredStatus }) => {
      const missingCount = requiredStatus.filter((status) => status.status === "missing").length;
      next.set(node.id, {
        missingCount,
        resolvedCount: requiredStatus.length - missingCount,
        requiredCount: requiredStatus.length,
        requiredStatus,
      });
    });
    return next;
  }, [visualChainNodesWithStatus]);

  const selectedDagNode = useMemo(
    () => visualChainNodes.find((node) => node.id === selectedDagNodeId) || null,
    [selectedDagNodeId, visualChainNodes]
  );

  const selectedDagNodeStatus = useMemo(() => {
    if (!selectedDagNodeId) {
      return null;
    }
    return visualChainNodeStatusById.get(selectedDagNodeId) || null;
  }, [selectedDagNodeId, visualChainNodeStatusById]);

  const selectedDagNodeInspectorFields = useMemo(() => {
    if (!selectedDagNode) {
      return [];
    }

    const requiredStatusByField = new Map(
      (selectedDagNodeStatus?.requiredStatus || []).map((status) => [status.field, status])
    );
    const capability = capabilityById.get(selectedDagNode.capabilityId);
    const schemaProperties = capabilityInputSchemaProperties(capability);
    const customFields = Object.keys(selectedDagNode.inputBindings).filter(
      (field) => !requiredStatusByField.has(field)
    );

    const customFieldStatus = customFields.map((field) => {
      const property = schemaProperties[field];
      const schemaType = schemaPropertyTypeLabel(property);
      const schemaDescription =
        property && typeof property.description === "string"
          ? property.description
          : "Custom Studio binding";
      const binding = selectedDagNode.inputBindings[field];
      if (binding?.kind === "step_output") {
        const source = visualChainNodes.find((candidate) => candidate.id === binding.sourceNodeId);
        return {
          field,
          required: false,
          status: "from_chain" as const,
          detail: source ? `${source.taskName}.${binding.sourcePath}` : binding.sourcePath,
          schemaType,
          schemaDescription,
        };
      }
      if (binding?.kind === "literal") {
        return {
          field,
          required: false,
          status: binding.value.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.value.trim() ? "literal value" : "literal value missing",
          schemaType,
          schemaDescription,
        };
      }
      if (binding?.kind === "memory") {
        return {
          field,
          required: false,
          status: binding.name.trim() ? ("provided" as const) : ("missing" as const),
          detail: binding.name.trim()
            ? `memory:${binding.scope}/${binding.name}${binding.key ? `/${binding.key}` : ""}`
            : "memory name missing",
          schemaType,
          schemaDescription,
        };
      }
      if (binding?.kind === "context") {
        return {
          field,
          required: false,
          status: binding.path.trim() ? ("from_context" as const) : ("missing" as const),
          detail: binding.path.trim() || "context path missing",
          schemaType,
          schemaDescription,
        };
      }
      return {
        field,
        required: false,
        status: "missing" as const,
        detail: "missing",
        schemaType,
        schemaDescription,
      };
    });

    return [
      ...(selectedDagNodeStatus?.requiredStatus || []).map((status) => ({
        ...status,
        required: true,
      })),
      ...customFieldStatus,
    ];
  }, [capabilityById, selectedDagNode, selectedDagNodeStatus, visualChainNodes]);

  const canInsertDeriveForSelectedNode = useMemo(() => {
    if (!selectedDagNode) {
      return false;
    }
    if (!capabilityById.has("document.output.derive")) {
      return false;
    }
    if (selectedDagNode.capabilityId === "document.output.derive") {
      return false;
    }
    const requiredInputs = getCapabilityRequiredInputs(capabilityById.get(selectedDagNode.capabilityId));
    if (!requiredInputs.includes("path")) {
      return false;
    }
    const pathBinding = selectedDagNode.inputBindings.path;
    if (pathBinding?.kind === "step_output") {
      const sourceNode = visualChainNodes.find((node) => node.id === pathBinding.sourceNodeId);
      if (sourceNode?.capabilityId === "document.output.derive") {
        return false;
      }
    }
    return true;
  }, [capabilityById, selectedDagNode, visualChainNodes]);

  const quickFixNodeBindings = (nodeId: string) => {
    const targetNode = visualChainNodes.find((node) => node.id === nodeId);
    if (!targetNode) {
      return;
    }
    const context = contextState.context;
    const requiredFields = getCapabilityRequiredInputs(capabilityById.get(targetNode.capabilityId));
    if (requiredFields.length === 0) {
      setStudioNotice("Selected node has no required inputs.");
      return;
    }
    const sourceNodeIds = composerDraftEdges
      .filter((edge) => edge.toNodeId === nodeId)
      .map((edge) => edge.fromNodeId);
    const sourceNodes = sourceNodeIds
      .map((id) => visualChainNodes.find((node) => node.id === id))
      .filter((node): node is ComposerDraftNode => Boolean(node));
    let updatedCount = 0;
    requiredFields.forEach((field) => {
      const existingBinding = targetNode.inputBindings[field];
      if (existingBinding) {
        return;
      }
      if (isContextInputPresent(context[field])) {
        return;
      }
      const sourceNode = sourceNodes[sourceNodes.length - 1];
      if (!sourceNode) {
        return;
      }
      setVisualBindingFromSource(nodeId, field, sourceNode.id, sourceNode.outputPath || "result");
      updatedCount += 1;
    });
    setStudioNotice(
      updatedCount > 0
        ? `Quick-fixed ${updatedCount} missing input(s) for ${targetNode.taskName}.`
        : "No missing inputs could be auto-fixed for selected node."
    );
  };

  const autoWireNodeBindings = (nodeId: string) => {
    const targetIndex = visualChainNodes.findIndex((node) => node.id === nodeId);
    if (targetIndex <= 0) {
      setStudioNotice("Auto-wire requires a previous step.");
      return;
    }
    const targetNode = visualChainNodes[targetIndex];
    const previousNode = visualChainNodes[targetIndex - 1];
    if (!targetNode || !previousNode) {
      setStudioNotice("Unable to auto-wire selected node.");
      return;
    }
    const context = contextState.context;
    const requiredInputs = getCapabilityRequiredInputs(capabilityById.get(targetNode.capabilityId));
    let updatedCount = 0;
    requiredInputs.forEach((field) => {
      if (!CHAINABLE_REQUIRED_FIELDS.has(field)) {
        return;
      }
      if (targetNode.inputBindings[field]) {
        return;
      }
      if (isContextInputPresent(context[field])) {
        return;
      }
      setVisualBindingFromSource(nodeId, field, previousNode.id, previousNode.outputPath || "result");
      updatedCount += 1;
    });
    setStudioNotice(
      updatedCount > 0
        ? `Auto-wired ${updatedCount} input(s) from previous step.`
        : "No missing inputs were eligible for auto-wire."
    );
  };

  const focusComposerValidationIssue = (issue: {
    nodeId?: string;
    field?: string;
  }) => {
    const nodeId = issue.nodeId;
    if (!nodeId) {
      return;
    }
    const nodeStatus = visualChainNodeStatusById.get(nodeId);
    const resolvedField =
      issue.field && nodeStatus?.requiredStatus.some((entry) => entry.field === issue.field)
        ? issue.field
        : undefined;
    setSelectedDagNodeId(nodeId);
    centerDagNodeInView(nodeId);
    setActiveComposerIssueFocus({ nodeId, field: resolvedField });
    if (resolvedField) {
      const refKey = `${nodeId}::${resolvedField}`;
      window.setTimeout(() => {
        inspectorBindingRefs.current[refKey]?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }, 180);
    }
  };

  const visualChainSummary = useMemo(() => {
    const summary = {
      steps: visualChainNodesWithStatus.length,
      dagEdges: composerDraftEdges.length,
      requiredInputs: 0,
      missingInputs: 0,
      contextInputs: 0,
      chainedInputs: 0,
      literalInputs: 0,
    };
    visualChainNodesWithStatus.forEach(({ requiredStatus }) => {
      summary.requiredInputs += requiredStatus.length;
      requiredStatus.forEach((item) => {
        if (item.status === "missing") {
          summary.missingInputs += 1;
        } else if (item.status === "from_context") {
          summary.contextInputs += 1;
        } else if (item.status === "from_chain") {
          summary.chainedInputs += 1;
        } else {
          summary.literalInputs += 1;
        }
      });
    });
    return summary;
  }, [composerDraftEdges.length, visualChainNodesWithStatus]);

  const dagCanvasNodes = useMemo(
    () =>
      visualChainNodes.map((node, index) => ({
        node,
        position: composerNodePositions[node.id] || defaultDagNodePosition(index),
      })),
    [composerNodePositions, visualChainNodes]
  );

  const dagCanvasNodeById = useMemo(
    () => new Map(dagCanvasNodes.map((entry) => [entry.node.id, entry])),
    [dagCanvasNodes]
  );

  const dagCanvasEdges = useMemo(
    () =>
      composerDraftEdges
        .map((edge) => {
          const fromEntry = dagCanvasNodeById.get(edge.fromNodeId);
          const toEntry = dagCanvasNodeById.get(edge.toNodeId);
          if (!fromEntry || !toEntry) {
            return null;
          }
          const edgeKey = `${edge.fromNodeId}->${edge.toNodeId}`;
          const startX = fromEntry.position.x + DAG_CANVAS_NODE_WIDTH;
          const startY = fromEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
          const endX = toEntry.position.x;
          const endY = toEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
          const controlX = (startX + endX) / 2;
          const midX = controlX;
          const midY = (startY + endY) / 2;
          return {
            ...edge,
            edgeKey,
            fromTaskName: fromEntry.node.taskName,
            toTaskName: toEntry.node.taskName,
            path: `M ${startX} ${startY} C ${controlX} ${startY}, ${controlX} ${endY}, ${endX} ${endY}`,
            midX,
            midY,
          };
        })
        .filter(
          (
            item
          ): item is {
            fromNodeId: string;
            toNodeId: string;
            edgeKey: string;
            fromTaskName: string;
            toTaskName: string;
            path: string;
            midX: number;
            midY: number;
          } => item !== null
        ),
    [composerDraftEdges, dagCanvasNodeById]
  );

  const dagConnectorPreview = useMemo(() => {
    if (!dagConnectorDrag) {
      return null;
    }
    const sourceEntry = dagCanvasNodeById.get(dagConnectorDrag.sourceNodeId);
    if (!sourceEntry) {
      return null;
    }
    const startX = sourceEntry.position.x + DAG_CANVAS_NODE_WIDTH;
    const startY = sourceEntry.position.y + DAG_CANVAS_NODE_HEIGHT / 2;
    const endX = dagConnectorDrag.x;
    const endY = dagConnectorDrag.y;
    const controlX = (startX + endX) / 2;
    return {
      path: `M ${startX} ${startY} C ${controlX} ${startY}, ${controlX} ${endY}, ${endX} ${endY}`,
    };
  }, [dagCanvasNodeById, dagConnectorDrag]);

  const dagCanvasSurface = useMemo(() => {
    let maxX = DAG_CANVAS_MIN_WIDTH;
    let maxY = DAG_CANVAS_MIN_HEIGHT;
    dagCanvasNodes.forEach((entry) => {
      maxX = Math.max(maxX, entry.position.x + DAG_CANVAS_NODE_WIDTH + DAG_CANVAS_PADDING);
      maxY = Math.max(maxY, entry.position.y + DAG_CANVAS_NODE_HEIGHT + DAG_CANVAS_PADDING);
    });
    return { width: maxX, height: maxY };
  }, [dagCanvasNodes]);

  const dagNodeAdjacency = useMemo(() => {
    const incoming: Record<string, number> = {};
    const outgoing: Record<string, number> = {};
    visualChainNodes.forEach((node) => {
      incoming[node.id] = 0;
      outgoing[node.id] = 0;
    });
    composerDraftEdges.forEach((edge) => {
      if (incoming[edge.toNodeId] !== undefined) {
        incoming[edge.toNodeId] += 1;
      }
      if (outgoing[edge.fromNodeId] !== undefined) {
        outgoing[edge.fromNodeId] += 1;
      }
    });
    return { incoming, outgoing };
  }, [composerDraftEdges, visualChainNodes]);

  const compileRequestPayload = useMemo(() => {
    const parsedContext = contextState.invalid ? { __invalid_context_json: true } : contextState.context;
    return {
      draft: {
        summary: composerDraft.summary || "Workflow Studio draft",
        nodes: visualChainNodes.map((node) => ({
          id: node.id,
          taskName: node.taskName,
          capabilityId: node.capabilityId,
          bindings: node.inputBindings,
        })),
        edges: composerDraftEdges,
      },
      job_context: parsedContext,
      goal: goal.trim() || undefined,
    };
  }, [composerDraft.summary, composerDraftEdges, contextState.context, contextState.invalid, goal, visualChainNodes]);

  const draftPayloadPreview = useMemo(() => {
    const parsedContext = contextState.invalid ? { __invalid_context_json: true } : contextState.context;
    return {
      draft: {
        summary: composerDraft.summary || "Workflow Studio draft",
        nodes: visualChainNodes.map((node) => ({
          id: node.id,
          taskName: node.taskName,
          capabilityId: node.capabilityId,
          bindings: node.inputBindings,
          outputPath: node.outputPath,
          outputs: node.outputs,
          variables: node.variables,
        })),
        edges: composerDraftEdges,
      },
      job_context: parsedContext,
      goal: goal.trim() || undefined,
    };
  }, [composerDraft.summary, composerDraftEdges, contextState.context, contextState.invalid, goal, visualChainNodes]);

  const composerIssues = useMemo(
    () => collectComposerValidationIssues(chainPreflightResult, composerCompileResult, visualChainNodes),
    [chainPreflightResult, composerCompileResult, visualChainNodes]
  );

  const runChainPreflight = async () => {
    const localErrors: string[] = [];
    if (visualChainNodes.length === 0) {
      localErrors.push("No chain steps configured.");
    }

    const seenTaskNames = new Set<string>();
    visualChainNodes.forEach((node, index) => {
      const taskName = node.taskName.trim();
      if (!taskName) {
        localErrors.push(`Step ${index + 1}: task name is required.`);
      } else if (seenTaskNames.has(taskName)) {
        localErrors.push(`Duplicate task name: ${taskName}`);
      } else {
        seenTaskNames.add(taskName);
      }
      if (!capabilityById.has(node.capabilityId)) {
        localErrors.push(
          `Step ${index + 1} (${taskName || "unnamed"}): capability ${node.capabilityId} not found in catalog.`
        );
      }
      if (!node.outputPath.trim()) {
        localErrors.push(`Step ${index + 1} (${taskName || "unnamed"}): output path is required.`);
      }
      const seenOutputNames = new Set<string>();
      const seenOutputPaths = new Set<string>();
      node.outputs.forEach((output, outputIndex) => {
        const name = output.name.trim();
        const path = output.path.trim();
        if (!name) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): extra output ${outputIndex + 1} needs a name.`
          );
        } else if (seenOutputNames.has(name)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate extra output name '${name}'.`
          );
        } else {
          seenOutputNames.add(name);
        }
        if (!path) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): extra output ${outputIndex + 1} needs a path.`
          );
        } else if (seenOutputPaths.has(path)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate extra output path '${path}'.`
          );
        } else {
          seenOutputPaths.add(path);
        }
      });
      const seenVariableKeys = new Set<string>();
      node.variables.forEach((variable, variableIndex) => {
        const key = variable.key.trim();
        if (!key) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): variable ${variableIndex + 1} needs a name.`
          );
        } else if (seenVariableKeys.has(key)) {
          localErrors.push(
            `Step ${index + 1} (${taskName || "unnamed"}): duplicate variable name '${key}'.`
          );
        } else {
          seenVariableKeys.add(key);
        }
      });
    });

    visualChainNodesWithStatus.forEach(({ node, requiredStatus }) => {
      requiredStatus
        .filter((entry) => entry.status === "missing")
        .forEach((entry) => {
          localErrors.push(
            `Step ${node.taskName}: required input '${entry.field}' is missing (not in chain or context).`
          );
        });
      Object.entries(node.inputBindings).forEach(([field, binding]) => {
        if (binding.kind !== "step_output") {
          if (binding.kind === "context" && !binding.path.trim()) {
            localErrors.push(`Step ${node.taskName}: context path for '${field}' is empty.`);
          }
          if (binding.kind === "memory" && !binding.name.trim()) {
            localErrors.push(`Step ${node.taskName}: memory name for '${field}' is required.`);
          }
          return;
        }
        if (!binding.sourcePath.trim()) {
          localErrors.push(`Step ${node.taskName}: binding for '${field}' has empty source path.`);
        }
        if (field === "path" && binding.sourcePath.trim() && !isPathOutputReference(binding.sourcePath)) {
          localErrors.push(
            `Step ${node.taskName}: binding for 'path' should reference a path output (for example 'path').`
          );
        }
        const sourceIndex = visualChainNodes.findIndex((candidate) => candidate.id === binding.sourceNodeId);
        if (sourceIndex < 0) {
          localErrors.push(
            `Step ${node.taskName}: binding for '${field}' references a removed source step.`
          );
        }
      });
    });

    const nodeIds = visualChainNodes.map((node) => node.id);
    const nodeIdSet = new Set(nodeIds);
    const explicitEdges = normalizeComposerEdges(visualChainNodes, composerDraftEdges);
    explicitEdges.forEach((edge) => {
      if (!nodeIdSet.has(edge.fromNodeId) || !nodeIdSet.has(edge.toNodeId)) {
        localErrors.push(`DAG edge '${edge.fromNodeId} -> ${edge.toNodeId}' references missing node(s).`);
      }
      if (edge.fromNodeId === edge.toNodeId) {
        localErrors.push(`DAG edge '${edge.fromNodeId} -> ${edge.toNodeId}' is a self-cycle.`);
      }
    });

    const implicitEdges: ComposerDraftEdge[] = [];
    visualChainNodes.forEach((node) => {
      Object.values(node.inputBindings).forEach((binding) => {
        if (binding.kind === "step_output") {
          implicitEdges.push({ fromNodeId: binding.sourceNodeId, toNodeId: node.id });
        }
      });
    });

    const combinedEdges = normalizeComposerEdges(visualChainNodes, [...explicitEdges, ...implicitEdges]);
    if (detectDagCycle(nodeIds, combinedEdges)) {
      localErrors.push("DAG contains a cycle (including step-output dependencies).");
    }
    if (contextState.invalid) {
      localErrors.push("Context JSON is invalid JSON.");
    }

    let serverErrors: Record<string, string> = {};
    let serverDiagnostics: {
      severity?: "error" | "warning";
      code: string;
      field?: string;
      message: string;
      slot_fields?: string[];
    }[] = [];
    let compiledPlan: Record<string, unknown> | null = null;

    if (localErrors.length === 0) {
      setComposerCompileLoading(true);
      setChainPreflightLoading(true);
      try {
        const compileResponse = await fetch(`${apiUrl}/composer/compile`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(compileRequestPayload),
        });
        const compileBody = (await compileResponse.json()) as ComposerCompileResponse | { detail?: unknown };
        if (!compileResponse.ok) {
          localErrors.push(
            typeof (compileBody as { detail?: unknown }).detail === "string"
              ? (compileBody as { detail: string }).detail
              : `Compile request failed (${compileResponse.status}).`
          );
        } else {
          const typedCompile = compileBody as ComposerCompileResponse;
          setComposerCompileResult(typedCompile);
          if (!typedCompile.valid) {
            typedCompile.diagnostics.errors.forEach((diag) => {
              serverErrors[diag.code] = diag.message;
            });
            serverErrors = { ...serverErrors, ...(typedCompile.preflight_errors || {}) };
          } else {
            compiledPlan = typedCompile.plan;
          }
        }

        if (compiledPlan) {
          const response = await fetch(`${apiUrl}/plans/preflight`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              plan: compiledPlan,
              job_context: contextState.context,
              goal: goal.trim() || undefined,
            }),
          });
          const body = (await response.json()) as
            | {
                valid: boolean;
                errors: Record<string, string>;
                diagnostics?: {
                  severity?: "error" | "warning";
                  code: string;
                  field?: string;
                  message: string;
                  slot_fields?: string[];
                }[];
              }
            | { detail?: unknown };
          if (!response.ok) {
            localErrors.push(
              typeof (body as { detail?: unknown }).detail === "string"
                ? (body as { detail: string }).detail
                : `Preflight request failed (${response.status}).`
            );
          } else if (body && typeof body === "object" && "errors" in body && body.errors) {
            serverErrors = { ...serverErrors, ...body.errors };
            if (Array.isArray(body.diagnostics)) {
              serverDiagnostics = body.diagnostics;
            }
          }
        } else if (Object.keys(serverErrors).length === 0 && localErrors.length === 0) {
          localErrors.push("Compile succeeded but returned no executable plan.");
        }
      } catch (error) {
        localErrors.push(error instanceof Error ? error.message : "Compile/preflight request failed.");
      } finally {
        setComposerCompileLoading(false);
        setChainPreflightLoading(false);
      }
    } else {
      setComposerCompileResult(null);
    }

    setChainPreflightResult({
      valid: localErrors.length === 0 && Object.keys(serverErrors).length === 0,
      localErrors,
      serverErrors,
      serverDiagnostics,
      checkedAt: new Date().toISOString(),
    });

    setStudioNotice(
      localErrors.length === 0 && Object.keys(serverErrors).length === 0
        ? "Compile + preflight passed."
        : "Compile or preflight found issues."
    );
  };

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-[36px] bg-gradient-to-br from-stone-950 via-slate-900 to-sky-950 px-8 py-8 text-white shadow-2xl">
        <div className="pointer-events-none absolute -left-14 top-8 h-44 w-44 rounded-full bg-amber-300/20 blur-3xl" />
        <div className="pointer-events-none absolute -right-12 bottom-0 h-56 w-56 rounded-full bg-sky-400/25 blur-3xl" />
        <div className="relative flex flex-wrap items-start justify-between gap-6">
          <div className="max-w-3xl">
            <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-sky-200">
              Workflow Studio
            </div>
            <h1 className="mt-2 font-display text-4xl tracking-tight md:text-5xl">
              Design DAGs before you run them.
            </h1>
            <p className="mt-3 text-sm leading-6 text-slate-200 md:text-base">
              This surface is the first dedicated studio route: capability palette on the left,
              graph canvas in the middle, step inspector and compile preview on the right. It compiles
              through the existing composer and plan preflight endpoints instead of inventing a second runtime.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/"
              className="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15"
            >
              Back to Compose
            </Link>
            <button
              className="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/15"
              onClick={() => {
                setGoal("");
                setContextJson(initialContextJson());
                setComposerDraft(initialStudioDraft());
                setComposerNodePositions({});
                setSelectedDagNodeId(null);
                setChainPreflightResult(null);
                setComposerCompileResult(null);
                setStudioNotice("Started a fresh studio draft.");
              }}
            >
              New Draft
            </button>
            <button
              className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-900 transition hover:bg-slate-100"
              onClick={runChainPreflight}
              disabled={composerCompileLoading || chainPreflightLoading}
            >
              {composerCompileLoading || chainPreflightLoading ? "Compiling..." : "Compile Preview"}
            </button>
          </div>
        </div>
      </section>

      {studioNotice ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800">
          {studioNotice}
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)_360px]">
        <StudioCapabilityPalette
          capabilities={paletteCapabilities}
          groups={paletteGroups}
          loading={capabilityLoading}
          error={capabilityError}
          query={paletteQuery}
          selectedGroup={paletteGroup}
          onQueryChange={setPaletteQuery}
          onGroupChange={setPaletteGroup}
          onAddCapability={addCapabilityNodeToStudio}
        />

        <div className="space-y-6">
          <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
              <label className="block">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Goal
                </div>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                  value={goal}
                  onChange={(event) => setGoal(event.target.value)}
                  placeholder="Generate a document pipeline with validation and render output"
                />
              </label>
              <label className="block">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Draft Summary
                </div>
                <input
                  className="mt-1 w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                  value={composerDraft.summary}
                  onChange={(event) =>
                    setComposerDraft((prev) => ({ ...prev, summary: event.target.value }))
                  }
                  placeholder="Workflow Studio draft"
                />
              </label>
            </div>
            <label className="mt-4 block">
              <div className="flex items-center justify-between gap-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Context JSON
                </div>
                <div className="text-xs text-slate-500">
                  {contextState.invalid ? "Invalid JSON" : "Object ready"}
                </div>
              </div>
              <textarea
                className="mt-1 min-h-[180px] w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 font-mono text-xs text-slate-900 outline-none transition focus:border-slate-400 focus:bg-white"
                value={contextJson}
                onChange={(event) => setContextJson(event.target.value)}
              />
            </label>
            <div className="mt-4 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                Steps <span className="font-semibold">{visualChainSummary.steps}</span>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                DAG edges <span className="font-semibold">{visualChainSummary.dagEdges}</span>
              </div>
              <div
                className={`rounded-2xl border px-3 py-2 ${
                  visualChainSummary.missingInputs > 0
                    ? "border-rose-200 bg-rose-50 text-rose-700"
                    : "border-emerald-200 bg-emerald-50 text-emerald-700"
                }`}
              >
                Missing <span className="font-semibold">{visualChainSummary.missingInputs}</span>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-slate-700">
                Context hits <span className="font-semibold">{visualChainSummary.contextInputs}</span>
              </div>
            </div>
            <ComposerValidationPanel
              preflightResult={chainPreflightResult}
              compileLoading={composerCompileLoading || chainPreflightLoading}
              issues={composerIssues}
              needsValidation={visualChainNodes.length > 0}
              onIssueClick={focusComposerValidationIssue}
              activeIssue={activeComposerIssueFocus}
              formatTimestamp={formatTimestamp}
            />
          </section>

          <section className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  Canvas
                </div>
                <h2 className="mt-1 font-display text-2xl text-slate-900">Workflow Graph</h2>
              </div>
              <button
                className="rounded-full border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-900 hover:text-slate-900 disabled:opacity-50"
                onClick={autoLayoutDagCanvas}
                disabled={visualChainNodes.length === 0}
              >
                Auto Layout
              </button>
            </div>

            <ComposerDagCanvas
              visualChainNodes={visualChainNodes}
              dagEdgeDraftSourceNodeId={dagEdgeDraftSourceNodeId}
              setDagEdgeDraftSourceNodeId={setDagEdgeDraftSourceNodeId}
              setDagConnectorDrag={setDagConnectorDrag}
              setDagConnectorHoverTargetNodeId={setDagConnectorHoverTargetNodeId}
              autoLayoutDagCanvas={autoLayoutDagCanvas}
              dagCanvasViewportRef={dagCanvasViewportRef}
              dagCanvasRef={dagCanvasRef}
              dagCanvasSurface={dagCanvasSurface}
              dagCanvasEdges={dagCanvasEdges}
              hoveredDagEdgeKey={hoveredDagEdgeKey}
              setHoveredDagEdgeKey={setHoveredDagEdgeKey}
              removeDagEdge={removeDagEdge}
              dagConnectorPreview={dagConnectorPreview}
              dagCanvasNodes={dagCanvasNodes}
              composerDraftEdges={composerDraftEdges}
              dagNodeAdjacency={dagNodeAdjacency}
              visualChainNodeStatusById={visualChainNodeStatusById as Map<
                string,
                { missingCount: number; requiredCount: number }
              >}
              selectedDagNodeId={selectedDagNodeId}
              setSelectedDagNodeId={setSelectedDagNodeId}
              dagConnectorDrag={dagConnectorDrag}
              dagCanvasDraggingNodeId={dagCanvasDraggingNodeId}
              dagConnectorHoverTargetNodeId={dagConnectorHoverTargetNodeId}
              addDagEdge={addDagEdge}
              beginDagNodeDrag={beginDagNodeDrag}
              isInteractiveCanvasTarget={isInteractiveCanvasTarget}
              beginDagConnectorDrag={beginDagConnectorDrag}
              centerDagNodeInView={centerDagNodeInView}
              nodeWidth={DAG_CANVAS_NODE_WIDTH}
              nodeHeight={DAG_CANVAS_NODE_HEIGHT}
            />

            <div className="mt-4 flex flex-wrap gap-2">
              {visualChainNodes.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
                  Add capabilities from the palette to start shaping a DAG.
                </div>
              ) : (
                visualChainNodes.map((node, index) => {
                  const isSelected = selectedDagNodeId === node.id;
                  return (
                    <div
                      key={`studio-node-chip-${node.id}`}
                      className={`flex items-center gap-2 rounded-full border px-3 py-2 text-sm ${
                        isSelected
                          ? "border-sky-300 bg-sky-50 text-sky-900"
                          : "border-slate-200 bg-slate-50 text-slate-700"
                      }`}
                    >
                      <button onClick={() => setSelectedDagNodeId(node.id)}>
                        {index + 1}. {node.taskName}
                      </button>
                      <button
                        className="rounded-full border border-current px-1.5 py-0 text-[11px]"
                        onClick={() => removeVisualChainNode(node.id)}
                        title="Remove step"
                      >
                        ×
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </div>

        <div className="space-y-6">
          <StudioNodeInspector
            selectedDagNode={selectedDagNode}
            selectedDagNodeStatus={selectedDagNodeStatus}
            inputFields={selectedDagNodeInspectorFields}
            activeComposerIssueFocus={activeComposerIssueFocus}
            inspectorBindingRefs={inspectorBindingRefs}
            visualChainNodes={visualChainNodes}
            outputPathSuggestionsForNode={outputPathSuggestionsForNode}
            contextPathSuggestions={contextPathSuggestions}
            autoWireNodeBindings={autoWireNodeBindings}
            quickFixNodeBindings={quickFixNodeBindings}
            setSelectedDagNodeId={setSelectedDagNodeId}
            capabilityIdOptionsId="studio-capability-id-options"
            onDeleteNode={removeVisualChainNode}
            updateNodeBasics={updateVisualChainNode}
            setVisualBindingMode={setVisualBindingMode}
            clearVisualBinding={clearVisualBinding}
            removeCustomInputField={removeCustomInputField}
            addCustomInputField={addCustomInputField}
            updateVisualBindingSourceNode={updateVisualBindingSourceNode}
            updateVisualBindingPath={updateVisualBindingPath}
            updateVisualBindingLiteral={updateVisualBindingLiteral}
            updateVisualBindingContextPath={updateVisualBindingContextPath}
            updateVisualBindingMemory={updateVisualBindingMemory}
            setVisualBindingFromPrevious={setVisualBindingFromPrevious}
            canInsertDeriveOutputPath={canInsertDeriveForSelectedNode}
            onInsertDeriveOutputPath={insertDeriveOutputPathStepForNode}
            addNodeOutput={addNodeOutput}
            updateNodeOutput={updateNodeOutput}
            removeNodeOutput={removeNodeOutput}
            addNodeVariable={addNodeVariable}
            updateNodeVariable={updateNodeVariable}
            removeNodeVariable={removeNodeVariable}
          />

          <StudioCompilePanel
            compileLoading={composerCompileLoading || chainPreflightLoading}
            compileResult={composerCompileResult}
            preflightResult={chainPreflightResult}
            issues={composerIssues}
            draftPayloadPreview={draftPayloadPreview}
            onCompile={runChainPreflight}
          />
        </div>
      </div>

      <datalist id="studio-capability-id-options">
        {availableCapabilities.map((item) => (
          <option key={`studio-capability-id-option-${item.id}`} value={item.id} />
        ))}
      </datalist>
    </div>
  );
}
