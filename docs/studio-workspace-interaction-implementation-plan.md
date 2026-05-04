# Workflow Studio Workspace Interaction Implementation Plan

This document defines a phased implementation plan for improving the `Workflow Studio` workspace so the graph remains primary, floating utilities stop obstructing core authoring, and graph navigation becomes direct and predictable.

It is written against the current UI implementation in:

- `services/ui/src/app/features/studio/WorkflowStudio.tsx`
- `services/ui/src/app/components/composer/ComposerDagCanvas.tsx`
- `services/ui/src/app/features/studio/StudioCapabilityPalette.tsx`
- `services/ui/src/app/features/studio/StudioCompilePanel.tsx`
- `services/ui/src/app/features/studio/StudioWorkflowInterfacePanel.tsx`
- `services/ui/src/app/features/studio/StudioNodeInspector.tsx`

## 1. Problem Statement

The current Studio behaves like a floating-window workspace, but the graph is still the core artifact the user edits. That creates a conflict:

- floating panels obscure nodes and edges
- panels compete with the graph for visual priority
- panning the graph requires scrollbars rather than direct manipulation
- the default layout is still too freeform for common editing flows
- minimized panels disappear instead of collapsing into a predictable recovery area

The goal is not to remove the floating model entirely. The goal is to make the graph the primary surface and make panels behave like utilities around it.

## 2. Target UX Model

The target pattern is a hybrid workspace:

- the graph is the primary canvas
- utilities are docked by default
- advanced users may still float or resize panels
- only one panel is strongly contextual: `Node Inspector`
- the graph supports direct pan with pointer gestures and keyboard modifiers
- minimized panels collapse into a visible shelf instead of vanishing
- the user can quickly switch between `editing`, `focused graph`, and `inspection` modes

## 3. Design Principles

1. Graph first
The graph should own the center of the workspace. Panels should default to the edges.

2. Contextual over persistent
Only show panels continuously if they are needed continuously. `Node Inspector` should appear only when a node is selected.

3. Deterministic recovery
Every panel state must be recoverable through visible controls: restore, reset layout, reopen from shelf.

4. Direct manipulation
Users should be able to drag the canvas to pan it. Scrollbars should be secondary, not primary.

5. Progressive complexity
Default behavior should be structured and predictable. Floating behavior should remain available for power users.

## 4. Current State Summary

### What already exists

- floating panel layout state in `WorkflowStudio.tsx`
- drag, resize, minimize, restore
- persisted panel layout in `localStorage`
- drag-end snap to common positions
- graph viewport with `overflow-auto`
- node centering and zoom-preserving viewport scroll logic
- `Node Inspector` only renders when a node is selected

### What is missing

- true drag-to-pan interaction
- docked default panel behavior with explicit left/right/bottom zones
- minimized panel shelf
- graph focus mode
- overlap-aware panel management
- keyboard shortcuts for workspace control
- clear separation between docked and floating panel modes

## 5. Scope

### In scope

- graph panning interactions
- docked panel model
- minimized shelf
- focus graph mode
- panel recovery and reset affordances
- keyboard shortcuts
- persistence model updates
- visual and accessibility polish for the interaction model

### Out of scope

- changing workflow semantics
- redesigning node cards or edge routing
- replacing the current DAG layout algorithm
- multiplayer or collaborative layout state
- mobile-first Studio redesign

## 6. Target Architecture

Introduce an explicit workspace layout model instead of treating all panels as equivalent free-floating boxes.

### Proposed layout concepts

- `panelMode`: `docked | floating | minimized`
- `dockZone`: `left | right | bottom | overlay | none`
- `panelGroup`: used for future trays or tab groups
- `workspaceMode`: `default | focus_graph | inspect_node`
- `panState`: current canvas panning interaction state

### Proposed state shape

Add or evolve the floating panel layout state in `WorkflowStudio.tsx` toward:

```ts
type StudioPanelMode = "docked" | "floating" | "minimized";
type StudioDockZone = "left" | "right" | "bottom" | "overlay" | "none";
type StudioWorkspaceMode = "default" | "focus_graph" | "inspect_node";

type StudioPanelLayout = {
  x: number;
  y: number;
  width: number;
  height: number;
  zIndex: number;
  minimized: boolean;
  mode: StudioPanelMode;
  dockZone: StudioDockZone;
  order: number;
};
```

This can be introduced incrementally by extending the existing `FloatingStudioPanelLayout` type rather than replacing it in one pass.

## 7. Recommended Phase Order

Implement these phases in order. Earlier phases improve usability immediately and reduce rework in later phases.

## Phase 0. Baseline and Cleanup

### Goal

Stabilize the current panel model before introducing a docked layout.

### Changes

- audit current panel defaults in `createInitialFloatingStudioPanelLayouts`
- document desired default roles for each panel:
  - `palette`: left utility rail
  - `compile`: right utility panel
  - `setup`: right utility panel
  - `interface`: bottom utility tray
  - `library`: secondary overlay or bottom tray
  - `inspector`: contextual right panel
- standardize panel IDs, restore behavior, and persistence schema versioning
- add a clear `workspace layout version` constant so persisted layouts can be invalidated safely when structure changes

### File targets

- `WorkflowStudio.tsx`

### Acceptance criteria

- layout persistence can be invalidated on schema change
- current panel defaults are explicitly documented in code
- panel restore logic remains deterministic

## Phase 1. Direct Graph Panning

### Goal

Allow the user to move around the graph without relying on scrollbars.

### UX

- drag empty canvas to pan
- support `Space + drag` panning
- support middle-mouse drag panning if feasible
- show grab/grabbing cursor on empty canvas
- do not interfere with node drag, edge creation, or panel drag

### Implementation approach

Add panning logic to `ComposerDagCanvas.tsx` using the existing `dagCanvasViewportRef`.

### Suggested behavior rules

- if pointer down starts on empty canvas:
  - with `Space` held, start pan
  - without `Space`, either start pan directly or preserve click-to-deselect on click-only release
- if pointer down starts on:
  - a node
  - a connector
  - a toolbar button
  - any panel
  then do not start pan
- pan should update viewport scroll:
  - `viewport.scrollLeft -= deltaX`
  - `viewport.scrollTop -= deltaY`

### Technical tasks

- add local pan state to `ComposerDagCanvas.tsx`
- distinguish click from drag threshold so deselect still works
- track `Space` key press in component or parent
- block text selection while panning
- add cursor changes:
  - `grab` on empty canvas
  - `grabbing` while panning

### File targets

- `ComposerDagCanvas.tsx`
- optionally `WorkflowStudio.tsx` if keyboard state is shared there

### Acceptance criteria

- user can pan the graph by dragging empty canvas
- node drag still works
- edge creation still works
- click empty canvas still clears selection
- zoom behavior still preserves viewport focus

### Test ideas

- Playwright drag empty canvas and assert scroll offsets changed
- ensure dragging a node still changes node position rather than viewport scroll

## Phase 2. Docked-by-Default Panel Layout

### Goal

Stop using free-floating windows as the default mental model.

### UX

Panels should open in stable zones:

- left: `Node Palette`
- right stack: `Compile Preview`, `Workflow Setup`
- bottom tray: `Workflow Interface`, optionally `Workflow Library`
- contextual right: `Node Inspector`

Floating should still exist, but opt-in.

### Implementation approach

Add `mode` and `dockZone` metadata to the panel layout state. A docked panel should be positioned by layout calculations, not by ad hoc x/y drag.

### Technical tasks

- extend panel layout type with `mode` and `dockZone`
- create zone-aware layout functions:
  - `buildDockedPanelLayouts(...)`
  - `clampFloatingPanelLayout(...)`
  - `resolveWorkspacePanelRects(...)`
- render docked panels inside dedicated overlay zones rather than as generic floating windows
- preserve the existing floating renderer for panels explicitly switched to floating mode

### Suggested layout model

- left dock:
  - fixed width range
  - full usable stage height minus top spacing
- right dock:
  - stacked cards with resize between sections later if needed
- bottom dock:
  - tabbed or segmented tray
  - resizable height

### File targets

- `WorkflowStudio.tsx`

### Acceptance criteria

- default panel layout no longer obscures the center graph area
- docked panels remain stable after reload
- floating mode still works for explicitly floated panels

## Phase 3. Minimized Shelf

### Goal

Minimized panels must remain visible and recoverable.

### UX

- minimized panels collapse into a shelf
- recommended shelf position: bottom edge of the stage
- each minimized panel shows:
  - panel name
  - optional badge or status
  - restore action
- clicking a minimized chip restores the panel

### Implementation approach

Instead of hiding minimized panels in place, route them into a dedicated minimized shelf renderer.

### Technical tasks

- introduce a minimized shelf component inside `WorkflowStudio.tsx`
- split panel rendering into:
  - docked panels
  - floating panels
  - minimized shelf items
- preserve minimized state in persistence
- show `Node Inspector` minimized state only when relevant

### File targets

- `WorkflowStudio.tsx`

### Acceptance criteria

- minimizing a panel removes it from graph obstruction
- minimized panels are still visible and restorable
- persisted layout restores minimized shelf state correctly

## Phase 4. Focus Graph Mode

### Goal

Give users a one-action way to clear the workspace and concentrate on the graph.

### UX

- toolbar control: `Focus Graph`
- keyboard shortcut: `F` or `Shift+F`
- hides or minimizes non-essential panels
- preserves `Node Inspector` if a node is selected, depending on design choice
- second toggle restores the previous layout

### Implementation approach

Add `workspaceMode` and preserve the prior panel layout before entering focus mode.

### Technical tasks

- add `workspaceMode` state
- snapshot current visible panel state before focus mode
- minimize or hide non-essential panels when focus mode enters
- restore snapshot when focus mode exits
- visually indicate focus mode in workspace chrome

### File targets

- `WorkflowStudio.tsx`
- possibly `ComposerDagCanvas.tsx` for toolbar affordance

### Acceptance criteria

- focus mode can be entered and exited without losing layout
- graph becomes visually primary
- user can still inspect nodes without full workspace clutter

## Phase 5. Dock/Floating Toggle and Panel Menus

### Goal

Make the panel model explicit and user-controlled.

### UX

Each panel should have a small menu or control set:

- dock left
- dock right
- dock bottom
- float
- minimize
- restore defaults

### Implementation approach

Extend panel header controls. The current traffic-light controls are not enough once panel behavior becomes more structured.

### Technical tasks

- add panel menu trigger in floating/docked headers
- expose allowed transitions per panel
- restrict unsupported dock targets where necessary
  - example: `Node Palette` should not dock right by default unless explicitly allowed
- wire menu actions to layout state updates

### File targets

- `WorkflowStudio.tsx`

### Acceptance criteria

- users can explicitly convert a panel between docked and floating modes
- layout changes persist
- unsupported transitions are either hidden or handled cleanly

## Phase 6. Bottom Tray for Interface and Library

### Goal

Move lower-priority tools out of the graph field and into a structured tray.

### UX

Replace multiple lower overlays with a bottom tray:

- tabs: `Workflow Interface`, `Workflow Library`
- optional future tabs: `Validation`, `Run Output`
- resizable tray height
- collapsed, peek, and expanded states

### Implementation approach

Render these panels inside a shared bottom dock container rather than as independent windows.

### Technical tasks

- create bottom tray state:
  - active tab
  - height
  - collapsed state
- move current panel content into tray tabs
- preserve content scroll independently inside each tab

### File targets

- `WorkflowStudio.tsx`
- maybe extract a new component such as `StudioBottomTray.tsx`

### Acceptance criteria

- interface and library no longer obscure the center graph area by default
- tray can be resized and collapsed
- panel content remains accessible and persistent

## Phase 7. Inspector Experience Refinement

### Goal

Make `Node Inspector` clearly contextual rather than just another panel.

### UX

- open only when node selected
- dock on right by default
- auto-focus relevant section when node has validation issues
- optional compact mode when graph focus is active

### Implementation approach

Keep the current conditional rendering but upgrade behavior.

### Technical tasks

- pin inspector to right contextual zone
- auto-restore when a node is selected
- animate entry/exit subtly
- optionally prevent it from becoming a random floating obstruction by default

### File targets

- `WorkflowStudio.tsx`
- `StudioNodeInspector.tsx`

### Acceptance criteria

- inspector feels contextual
- inspector does not obstruct the graph unnecessarily
- selecting and deselecting nodes produces predictable behavior

## Phase 8. Accessibility and Keyboard Model

### Goal

Make the workspace usable without relying entirely on pointer precision.

### UX / keyboard behavior

- `Space + drag`: pan graph
- `Esc`: clear temporary interaction
  - end edge draft
  - exit pan
  - optionally deselect node
- `F`: toggle focus graph
- `0`: reset zoom to 100%
- `Shift+1/2/3`: open key dock regions or common panels

### Technical tasks

- add keyboard event handling
- ensure buttons and panel controls have proper labels
- add visible focus states
- support reduced motion if animations are introduced

### Acceptance criteria

- keyboard controls work without conflicting with form inputs
- panel controls remain screen-reader discoverable
- focus order is coherent

## Phase 9. Persistence Hardening

### Goal

Make layout persistence stable over future iterations.

### Changes

- persist:
  - panel mode
  - dock zone
  - size
  - order
  - minimized state
  - active tray tab
  - focus mode if desired
- version the persistence schema
- invalidate gracefully on incompatible changes

### Suggested storage shape

```ts
type PersistedStudioWorkspaceLayoutV2 = {
  version: 2;
  workspaceMode: "default" | "focus_graph" | "inspect_node";
  bottomTray: {
    activeTab: string;
    height: number;
    collapsed: boolean;
  };
  panels: Record<string, {
    mode: "docked" | "floating" | "minimized";
    dockZone: "left" | "right" | "bottom" | "overlay" | "none";
    x: number;
    y: number;
    width: number;
    height: number;
    zIndex: number;
    order: number;
    minimized: boolean;
  }>;
};
```

### Acceptance criteria

- invalid or old storage does not break Studio
- layout restores predictably after refresh
- future phase rollout can bump schema safely

## 8. Recommended Delivery Milestones

Use these bundles if implementation needs to be split into practical releases.

### Milestone A

- Phase 0
- Phase 1

Outcome:
Users can pan the graph directly, and the existing floating model becomes less frustrating immediately.

### Milestone B

- Phase 2
- Phase 3

Outcome:
Default layout becomes docked and minimized panels stop blocking work.

### Milestone C

- Phase 4
- Phase 5

Outcome:
Users gain workspace modes and explicit control over docked vs floating behavior.

### Milestone D

- Phase 6
- Phase 7
- Phase 8
- Phase 9

Outcome:
The Studio becomes a coherent hybrid workspace rather than a collection of floating windows.

## 9. Testing Strategy

### Unit and component-level

- layout calculation helpers
- dock zone rect generation
- snap target logic
- persistence read/write and schema invalidation
- pan state transitions

### Playwright coverage

- drag empty canvas pans graph
- drag node still moves node, not viewport
- drag panel header still moves panel
- docked panels render in expected regions
- minimize sends panel to shelf
- restore from shelf works
- focus graph mode hides non-essential panels
- persisted workspace layout survives reload

### Manual QA checklist

- zoom in, pan, then drag node
- select node and confirm inspector opens
- deselect node and confirm inspector hides
- minimize compile/setup/interface panels and confirm graph becomes clearer
- reload and confirm layout restored
- reset layout and confirm defaults restored

## 10. Risks and Mitigations

### Risk: interaction conflicts

`drag-to-pan`, `drag-node`, and `drag-edge` can conflict.

Mitigation:

- use clear gesture precedence
- require empty-canvas start for pan
- require drag threshold before pan engages

### Risk: persistence churn

Frequent schema changes can break stored layouts.

Mitigation:

- version persisted layout
- centralize read/write helpers

### Risk: too much mode complexity

Docked, floating, minimized, focus mode, and tray state can become hard to reason about.

Mitigation:

- implement phases in order
- keep each panel’s allowed states explicit
- avoid introducing all modes at once

## 11. Suggested First Implementation Slice

If only one phase should be started first, begin with:

1. `Phase 1: Direct Graph Panning`
2. `Phase 3: Minimized Shelf`
3. `Phase 2: Docked-by-Default Layout`

Reason:

- panning solves the most immediate usability issue
- minimized shelf reduces graph obstruction quickly
- docked layout is the structural fix that should follow once navigation is improved

## 12. Definition of Done

This workspace interaction project should be considered complete when:

- the graph is the clear primary surface
- the default layout does not block central graph editing
- users can pan directly by dragging the canvas
- contextual inspection works without persistent clutter
- minimized panels are visible and recoverable
- panel layout persists safely across reloads
- keyboard and accessibility behavior are coherent

