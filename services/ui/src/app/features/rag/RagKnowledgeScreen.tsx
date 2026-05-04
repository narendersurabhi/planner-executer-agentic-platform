"use client";

import { useEffect, useMemo, useState } from "react";

import AppShell from "../../components/AppShell";
import ScreenHeader from "../../components/ScreenHeader";

const apiUrl = process.env.NEXT_PUBLIC_API_URL || "/api";
const MEMORY_USER_ID_KEY = "ape.memory.user_id.v1";
const DEFAULT_COLLECTION = "rag_default";
const DEFAULT_NAMESPACE = "docs";
const DEFAULT_USER_ID = "default-user";

type IndexMode = "markdown" | "text" | "workspace_file" | "workspace_directory";

type RagDocumentSummary = {
  document_id: string;
  source_uri: string;
  namespace?: string | null;
  tenant_id?: string | null;
  user_id?: string | null;
  workspace_id?: string | null;
  chunk_count: number;
  chunking_strategy?: string | null;
  content_type?: string | null;
  filename?: string | null;
  path?: string | null;
  repo?: string | null;
  indexed_at?: string | null;
  metadata: Record<string, unknown>;
};

type RagDocumentListResponse = {
  collection_name: string;
  truncated: boolean;
  scanned_point_count: number;
  documents: RagDocumentSummary[];
};

type RagDocumentChunk = {
  chunk_id: string;
  document_id: string;
  source_uri: string;
  text: string;
  chunk_index?: number | null;
  metadata: Record<string, unknown>;
};

type RagDocumentChunksResponse = {
  collection_name: string;
  document: RagDocumentSummary;
  chunks: RagDocumentChunk[];
};

type RagDeleteResponse = {
  collection_name: string;
  document_id: string;
  deleted_chunk_count: number;
};

type RagReplaceResponse = {
  deleted: RagDeleteResponse;
  indexed: Record<string, unknown>;
};

const INDEX_MODES: Array<{ id: IndexMode; label: string; description: string }> = [
  {
    id: "markdown",
    label: "Paste Markdown",
    description: "Chunk markdown by headings and index it as one logical document.",
  },
  {
    id: "text",
    label: "Paste Text",
    description: "Store raw text directly when you do not need markdown-aware sectioning.",
  },
  {
    id: "workspace_file",
    label: "Workspace File",
    description: "Index one file that already exists under the shared workspace.",
  },
  {
    id: "workspace_directory",
    label: "Workspace Directory",
    description: "Walk a workspace directory and index all allowed files in one run.",
  },
];

const fieldGroupClassName = "space-y-2 text-sm text-slate-200";
const fieldLabelClassName =
  "text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100";
const fieldInputClassName =
  "w-full rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-slate-50 caret-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:bg-slate-950/28 focus:outline-none";

const prettyJson = (value: unknown) => JSON.stringify(value ?? {}, null, 2);

const formatTimestamp = (value?: string | null) => {
  if (!value) {
    return "—";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
};

const asErrorMessage = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback;

function RagModeButton({
  active,
  label,
  description,
  onClick,
}: {
  active: boolean;
  label: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-2xl border px-4 py-3 text-left transition ${
        active
          ? "border-sky-300/35 bg-sky-400/18 text-sky-50 shadow-[0_8px_18px_rgba(14,165,233,0.16)]"
          : "border-white/10 bg-slate-950/18 text-slate-200 hover:border-white/18 hover:bg-slate-950/26"
      }`}
    >
      <div className="text-base font-semibold tracking-[-0.02em]">{label}</div>
      <div className={`mt-1 text-sm leading-6 ${active ? "text-sky-100/78" : "text-slate-300/68"}`}>
        {description}
      </div>
    </button>
  );
}

export default function RagKnowledgeScreen() {
  const [collectionName, setCollectionName] = useState(DEFAULT_COLLECTION);
  const [namespace, setNamespace] = useState(DEFAULT_NAMESPACE);
  const [userId, setUserId] = useState(DEFAULT_USER_ID);
  const [workspaceId, setWorkspaceId] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const [documents, setDocuments] = useState<RagDocumentSummary[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);

  const [chunkResponse, setChunkResponse] = useState<RagDocumentChunksResponse | null>(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [chunksError, setChunksError] = useState<string | null>(null);

  const [indexMode, setIndexMode] = useState<IndexMode>("markdown");
  const [documentIdInput, setDocumentIdInput] = useState("");
  const [sourceUriInput, setSourceUriInput] = useState("");
  const [markdownText, setMarkdownText] = useState("");
  const [plainText, setPlainText] = useState("");
  const [workspacePath, setWorkspacePath] = useState("docs/rag-playbook.md");
  const [directoryPath, setDirectoryPath] = useState("docs");
  const [recursiveDirectory, setRecursiveDirectory] = useState(true);
  const [metadataText, setMetadataText] = useState(prettyJson({}));
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [replacing, setReplacing] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const stored = window.localStorage.getItem(MEMORY_USER_ID_KEY);
    if (stored && stored.trim()) {
      setUserId(stored.trim());
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(MEMORY_USER_ID_KEY, userId);
  }, [userId]);

  const selectedDocument = useMemo(() => {
    if (chunkResponse?.document && chunkResponse.document.document_id === selectedDocumentId) {
      return chunkResponse.document;
    }
    return documents.find((document) => document.document_id === selectedDocumentId) ?? null;
  }, [chunkResponse, documents, selectedDocumentId]);

  const scopeSummary = useMemo(
    () =>
      [
        collectionName.trim() ? `Collection ${collectionName.trim()}` : null,
        namespace.trim() ? `Namespace ${namespace.trim()}` : null,
        userId.trim() ? `User ${userId.trim()}` : null,
        workspaceId.trim() ? `Workspace ${workspaceId.trim()}` : null,
        tenantId.trim() ? `Tenant ${tenantId.trim()}` : null,
      ].filter((value): value is string => Boolean(value)),
    [collectionName, namespace, tenantId, userId, workspaceId]
  );

  const buildScopeParams = () => {
    const params = new URLSearchParams();
    if (collectionName.trim()) {
      params.set("collection_name", collectionName.trim());
    }
    if (namespace.trim()) {
      params.set("namespace", namespace.trim());
    }
    if (userId.trim()) {
      params.set("user_id", userId.trim());
    }
    if (workspaceId.trim()) {
      params.set("workspace_id", workspaceId.trim());
    }
    if (tenantId.trim()) {
      params.set("tenant_id", tenantId.trim());
    }
    if (searchQuery.trim()) {
      params.set("query", searchQuery.trim());
    }
    return params;
  };

  const refreshDocuments = async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const params = buildScopeParams();
      const response = await fetch(`${apiUrl}/rag/documents?${params.toString()}`);
      const body = (await response.json()) as RagDocumentListResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: string }).detail === "string"
            ? (body as { detail: string }).detail
            : `Failed to load knowledge documents (${response.status})`
        );
      }
      const nextDocuments = (body as RagDocumentListResponse).documents;
      setDocuments(nextDocuments);
      if (
        selectedDocumentId &&
        !nextDocuments.some((document) => document.document_id === selectedDocumentId)
      ) {
        setSelectedDocumentId(null);
        setChunkResponse(null);
      }
    } catch (error) {
      setDocumentsError(asErrorMessage(error, "Failed to load knowledge documents."));
    } finally {
      setDocumentsLoading(false);
    }
  };

  const loadDocumentChunks = async (documentId: string) => {
    setChunksLoading(true);
    setChunksError(null);
    try {
      const params = buildScopeParams();
      params.set("document_id", documentId);
      const response = await fetch(`${apiUrl}/rag/documents/chunks?${params.toString()}`);
      const body = (await response.json()) as RagDocumentChunksResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: string }).detail === "string"
            ? (body as { detail: string }).detail
            : `Failed to load document chunks (${response.status})`
        );
      }
      setChunkResponse(body as RagDocumentChunksResponse);
    } catch (error) {
      setChunksError(asErrorMessage(error, "Failed to load document chunks."));
    } finally {
      setChunksLoading(false);
    }
  };

  useEffect(() => {
    void refreshDocuments();
  }, [collectionName, namespace, tenantId, userId, workspaceId]);

  const parseMetadata = () => {
    const parsed = JSON.parse(metadataText || "{}") as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Metadata must be a JSON object.");
    }
    return parsed as Record<string, unknown>;
  };

  const buildIndexPayload = (documentIdOverride?: string) => {
    const metadata = parseMetadata();
    const payload: Record<string, unknown> = {
      mode: indexMode,
      collection_name: collectionName.trim() || null,
      namespace: namespace.trim() || null,
      tenant_id: tenantId.trim() || null,
      user_id: userId.trim() || null,
      workspace_id: workspaceId.trim() || null,
      metadata,
    };
    const effectiveDocumentId = documentIdOverride || documentIdInput.trim();
    const effectiveSourceUri = sourceUriInput.trim() || documentIdOverride || documentIdInput.trim();
    if (effectiveDocumentId) {
      payload.document_id = effectiveDocumentId;
    }
    if (effectiveSourceUri) {
      payload.source_uri = effectiveSourceUri;
    }
    if (indexMode === "markdown") {
      const value = markdownText.trim();
      if (!value) {
        throw new Error("Markdown content is required.");
      }
      payload.markdown_text = value;
    } else if (indexMode === "text") {
      const value = plainText.trim();
      if (!value) {
        throw new Error("Text content is required.");
      }
      payload.text = value;
    } else if (indexMode === "workspace_file") {
      const value = workspacePath.trim();
      if (!value) {
        throw new Error("Workspace path is required.");
      }
      payload.path = value;
    } else {
      const value = directoryPath.trim();
      if (!value) {
        throw new Error("Directory path is required.");
      }
      payload.directory_path = value;
      payload.recursive = recursiveDirectory;
    }
    return payload;
  };

  const submitIndex = async () => {
    setFormError(null);
    setNotice(null);
    let payload: Record<string, unknown>;
    try {
      payload = buildIndexPayload();
    } catch (error) {
      setFormError(asErrorMessage(error, "Invalid knowledge indexing payload."));
      return;
    }
    setIndexing(true);
    try {
      const response = await fetch(`${apiUrl}/rag/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = (await response.json()) as Record<string, unknown> | { detail?: string };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: string }).detail === "string"
            ? (body as { detail: string }).detail
            : `Failed to index content (${response.status})`
        );
      }
      const result = body as Record<string, unknown>;
      setNotice("Indexed content into the knowledge base successfully.");
      const resultDocumentId =
        typeof result.document_id === "string" && result.document_id
          ? result.document_id
          : typeof payload.document_id === "string" && payload.document_id
            ? payload.document_id
            : null;
      await refreshDocuments();
      if (resultDocumentId) {
        setSelectedDocumentId(resultDocumentId);
        await loadDocumentChunks(resultDocumentId);
      }
    } catch (error) {
      setFormError(asErrorMessage(error, "Failed to index content."));
    } finally {
      setIndexing(false);
    }
  };

  const replaceSelectedDocument = async () => {
    if (!selectedDocumentId) {
      setFormError("Select a document to replace.");
      return;
    }
    if (indexMode !== "markdown" && indexMode !== "text") {
      setFormError("Replace currently supports pasted markdown or pasted text modes only.");
      return;
    }
    setFormError(null);
    setNotice(null);
    let payload: Record<string, unknown>;
    try {
      payload = buildIndexPayload(selectedDocumentId);
    } catch (error) {
      setFormError(asErrorMessage(error, "Invalid replacement payload."));
      return;
    }
    setReplacing(true);
    try {
      const response = await fetch(
        `${apiUrl}/rag/documents?document_id=${encodeURIComponent(selectedDocumentId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );
      const body = (await response.json()) as RagReplaceResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: string }).detail === "string"
            ? (body as { detail: string }).detail
            : `Failed to replace document (${response.status})`
        );
      }
      const result = body as RagReplaceResponse;
      setNotice("Replaced indexed document content.");
      await refreshDocuments();
      setSelectedDocumentId(result.deleted.document_id);
      await loadDocumentChunks(result.deleted.document_id);
    } catch (error) {
      setFormError(asErrorMessage(error, "Failed to replace document."));
    } finally {
      setReplacing(false);
    }
  };

  const deleteSelectedDocument = async () => {
    if (!selectedDocumentId) {
      return;
    }
    const confirmed = window.confirm(`Delete indexed document ${selectedDocumentId}?`);
    if (!confirmed) {
      return;
    }
    setDeleting(true);
    setFormError(null);
    setNotice(null);
    try {
      const params = buildScopeParams();
      params.set("document_id", selectedDocumentId);
      const response = await fetch(`${apiUrl}/rag/documents?${params.toString()}`, {
        method: "DELETE",
      });
      const body = (await response.json()) as RagDeleteResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(
          typeof (body as { detail?: string }).detail === "string"
            ? (body as { detail: string }).detail
            : `Failed to delete document (${response.status})`
        );
      }
      const result = body as RagDeleteResponse;
      setNotice(
        `Deleted ${result.document_id} (${result.deleted_chunk_count} chunks removed).`
      );
      setSelectedDocumentId(null);
      setChunkResponse(null);
      await refreshDocuments();
    } catch (error) {
      setFormError(asErrorMessage(error, "Failed to delete document."));
    } finally {
      setDeleting(false);
    }
  };

  const clearForm = () => {
    setDocumentIdInput("");
    setSourceUriInput("");
    setMarkdownText("");
    setPlainText("");
    setWorkspacePath("docs/rag-playbook.md");
    setDirectoryPath("docs");
    setRecursiveDirectory(true);
    setMetadataText(prettyJson({}));
    setFormError(null);
    setNotice(null);
  };

  return (
    <AppShell
      activeScreen="rag"
      title="Knowledge Base"
      breadcrumbs={[
        { label: "Project", href: "/project" },
        { label: "Knowledge Base" },
      ]}
      actions={
        <>
          <button
            type="button"
            className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void refreshDocuments()}
            disabled={documentsLoading}
          >
            Refresh
          </button>
          <button
            type="button"
            className="rounded-xl border border-slate-200/18 bg-slate-950/25 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-white transition hover:border-white/30 hover:bg-slate-950/35 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={() => void submitIndex()}
            disabled={indexing}
          >
            {indexing ? "Indexing..." : "Index Now"}
          </button>
        </>
      }
    >
        <ScreenHeader
          eyebrow="Knowledge Base"
          title="Knowledge Base"
          description="Connect documents and workspace content so AI workflows can retrieve the right context."
          activeScreen="rag"
          theme="studio"
          compact
        >
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Collection</span>
              <input
                value={collectionName}
                onChange={(event) => setCollectionName(event.target.value)}
                className={fieldInputClassName}
                placeholder="rag_default"
              />
            </label>
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Namespace</span>
              <input
                value={namespace}
                onChange={(event) => setNamespace(event.target.value)}
                className={fieldInputClassName}
                placeholder="docs"
              />
            </label>
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Context User ID</span>
              <input
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                className={fieldInputClassName}
                placeholder="default-user"
              />
            </label>
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Workspace ID</span>
              <input
                value={workspaceId}
                onChange={(event) => setWorkspaceId(event.target.value)}
                className={fieldInputClassName}
                placeholder="optional"
              />
            </label>
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Tenant ID</span>
              <input
                value={tenantId}
                onChange={(event) => setTenantId(event.target.value)}
                className={fieldInputClassName}
                placeholder="optional"
              />
            </label>
            <label className={fieldGroupClassName}>
              <span className={fieldLabelClassName}>Search</span>
              <div className="flex gap-2">
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  className={fieldInputClassName}
                  placeholder="document, source, metadata..."
                />
                <button
                  type="button"
                  className="rounded-xl border border-white/12 bg-white/[0.04] px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:border-sky-300/35 hover:bg-white/[0.08]"
                  onClick={() => void refreshDocuments()}
                >
                  Go
                </button>
              </div>
            </label>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {scopeSummary.map((item) => (
              <div
                key={item}
                className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-100"
              >
                {item}
              </div>
            ))}
          </div>
        </ScreenHeader>

        <div className="mt-5 grid gap-4 xl:grid-cols-[1.2fr,1fr,1fr]">
          <section className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(63,78,95,0.62),rgba(37,49,62,0.82))] p-5 shadow-[0_24px_60px_rgba(15,23,42,0.18),inset_0_1px_0_rgba(255,255,255,0.05)]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-100/68">
                  Index New
                </div>
                <h2 className="mt-2 text-[26px] font-semibold tracking-[-0.03em] text-white">Manual Indexing</h2>
                <p className="mt-2 text-sm leading-6 text-slate-300/78">
                  Choose a source mode, attach scope, then index new content or replace the currently
                  selected document.
                </p>
              </div>
              <button
                type="button"
                className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-white/16 hover:bg-white/[0.08]"
                onClick={clearForm}
              >
                Clear Form
              </button>
            </div>

            <div className="mt-6 grid gap-3">
              {INDEX_MODES.map((mode) => (
                <RagModeButton
                  key={mode.id}
                  active={indexMode === mode.id}
                  label={mode.label}
                  description={mode.description}
                  onClick={() => setIndexMode(mode.id)}
                />
              ))}
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <label className={fieldGroupClassName}>
                <span className={fieldLabelClassName}>Document ID</span>
                <input
                  value={documentIdInput}
                  onChange={(event) => setDocumentIdInput(event.target.value)}
                  className="w-full rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                  placeholder="docs/user-guide.md"
                />
              </label>
              <label className={fieldGroupClassName}>
                <span className={fieldLabelClassName}>Source URI</span>
                <input
                  value={sourceUriInput}
                  onChange={(event) => setSourceUriInput(event.target.value)}
                  className="w-full rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                  placeholder="docs/user-guide.md"
                />
              </label>
            </div>

            {indexMode === "markdown" ? (
              <label className={`mt-4 block ${fieldGroupClassName}`}>
                <span className={fieldLabelClassName}>Markdown Content</span>
                <textarea
                  value={markdownText}
                  onChange={(event) => setMarkdownText(event.target.value)}
                  className="h-64 w-full rounded-3xl border border-white/10 bg-slate-950/18 px-4 py-4 font-mono text-sm text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                  placeholder="# User Guide&#10;&#10;Paste markdown content here."
                />
              </label>
            ) : null}

            {indexMode === "text" ? (
              <label className={`mt-4 block ${fieldGroupClassName}`}>
                <span className={fieldLabelClassName}>Plain Text</span>
                <textarea
                  value={plainText}
                  onChange={(event) => setPlainText(event.target.value)}
                  className="h-64 w-full rounded-3xl border border-white/10 bg-slate-950/18 px-4 py-4 font-mono text-sm text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                  placeholder="Paste plain text content here."
                />
              </label>
            ) : null}

            {indexMode === "workspace_file" ? (
              <label className={`mt-4 block ${fieldGroupClassName}`}>
                <span className={fieldLabelClassName}>Workspace File Path</span>
                <input
                  value={workspacePath}
                  onChange={(event) => setWorkspacePath(event.target.value)}
                  className="w-full rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                  placeholder="docs/rag-playbook.md"
                />
              </label>
            ) : null}

            {indexMode === "workspace_directory" ? (
              <div className="mt-4 grid gap-4 md:grid-cols-[1fr,auto]">
                <label className={fieldGroupClassName}>
                  <span className={fieldLabelClassName}>Workspace Directory</span>
                  <input
                    value={directoryPath}
                    onChange={(event) => setDirectoryPath(event.target.value)}
                    className="w-full rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
                    placeholder="docs"
                  />
                </label>
                <label className="flex items-center gap-3 rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-100">
                  <input
                    type="checkbox"
                    checked={recursiveDirectory}
                    onChange={(event) => setRecursiveDirectory(event.target.checked)}
                    className="h-4 w-4 rounded border-white/20"
                  />
                  Recursive
                </label>
              </div>
            ) : null}

            <label className={`mt-4 block ${fieldGroupClassName}`}>
              <span className={fieldLabelClassName}>Metadata JSON</span>
              <textarea
                value={metadataText}
                onChange={(event) => setMetadataText(event.target.value)}
                className="h-44 w-full rounded-3xl border border-white/10 bg-slate-950/18 px-4 py-4 font-mono text-sm text-white placeholder:text-slate-400/70 focus:border-sky-300/40 focus:outline-none"
              />
            </label>

            {formError ? (
              <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {formError}
              </div>
            ) : null}
            {notice ? (
              <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                {notice}
              </div>
            ) : null}

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                type="button"
                className="rounded-full border border-slate-200/18 bg-slate-950/25 px-5 py-3 text-sm font-semibold text-white transition hover:border-white/30 hover:bg-slate-950/35 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => void submitIndex()}
                disabled={indexing}
              >
                {indexing ? "Indexing..." : "Index Now"}
              </button>
              <button
                type="button"
                className="rounded-full border border-white/10 bg-white/[0.05] px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-white/16 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => void replaceSelectedDocument()}
                disabled={replacing || !selectedDocumentId}
              >
                {replacing ? "Replacing..." : "Replace Selected"}
              </button>
            </div>
          </section>

          <section className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(63,78,95,0.62),rgba(37,49,62,0.82))] p-5 shadow-[0_24px_60px_rgba(15,23,42,0.18),inset_0_1px_0_rgba(255,255,255,0.05)]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-100/68">
                  Documents
                </div>
                <h2 className="mt-2 text-[26px] font-semibold tracking-[-0.03em] text-white">Indexed Inventory</h2>
                <p className="mt-2 text-sm leading-6 text-slate-300/78">
                  Review indexed documents in the active scope, then open one to inspect its stored
                  chunks.
                </p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-950/18 px-4 py-3 text-right">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-300/68">
                  Loaded
                </div>
                <div className="mt-1 text-[26px] font-semibold tracking-[-0.03em] text-white">
                  {documents.length}
                </div>
              </div>
            </div>

            {documentsError ? (
              <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {documentsError}
              </div>
            ) : null}

            <div className="mt-6 space-y-3">
              {documentsLoading ? (
                <div className="rounded-3xl border border-dashed border-white/12 bg-slate-950/18 px-4 py-12 text-center text-sm text-slate-300/74">
                  Loading indexed documents...
                </div>
              ) : documents.length === 0 ? (
                <div className="rounded-3xl border border-dashed border-white/12 bg-slate-950/18 px-4 py-12 text-center text-sm text-slate-300/74">
                  No indexed documents match the current scope.
                </div>
              ) : (
                documents.map((document) => {
                  const selected = document.document_id === selectedDocumentId;
                  return (
                    <button
                      key={document.document_id}
                      type="button"
                      onClick={() => {
                        setSelectedDocumentId(document.document_id);
                        void loadDocumentChunks(document.document_id);
                      }}
                      className={`w-full rounded-3xl border p-4 text-left transition ${
                        selected
                          ? "border-sky-300/35 bg-sky-400/18 text-sky-50 shadow-[0_8px_18px_rgba(14,165,233,0.16)]"
                          : "border-white/10 bg-slate-950/18 text-slate-100 hover:border-white/16 hover:bg-slate-950/26"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="truncate text-lg font-semibold tracking-[-0.02em]">
                            {document.filename || document.document_id}
                          </div>
                          <div
                            className={`mt-1 truncate text-xs uppercase tracking-[0.14em] ${
                              selected ? "text-sky-100/78" : "text-slate-300/68"
                            }`}
                          >
                            {document.source_uri}
                          </div>
                        </div>
                        <div
                          className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
                            selected
                              ? "bg-white/15 text-white"
                              : "bg-white/[0.05] text-slate-200"
                          }`}
                        >
                          {document.chunk_count} chunks
                        </div>
                      </div>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {document.namespace ? (
                          <div
                            className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
                              selected
                                ? "bg-white/10 text-slate-100"
                                : "bg-white/[0.05] text-slate-200"
                            }`}
                          >
                            {document.namespace}
                          </div>
                        ) : null}
                        {document.chunking_strategy ? (
                          <div
                            className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
                              selected
                                ? "bg-white/10 text-slate-100"
                                : "bg-white/[0.05] text-slate-200"
                            }`}
                          >
                            {document.chunking_strategy}
                          </div>
                        ) : null}
                        {document.content_type ? (
                          <div
                            className={`rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] ${
                              selected
                                ? "bg-white/10 text-slate-100"
                                : "bg-white/[0.05] text-slate-200"
                            }`}
                          >
                            {document.content_type}
                          </div>
                        ) : null}
                      </div>
                      <div
                        className={`mt-4 text-[11px] font-medium uppercase tracking-[0.14em] ${
                          selected ? "text-sky-100/78" : "text-slate-300/68"
                        }`}
                      >
                        Indexed {formatTimestamp(document.indexed_at)}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </section>

          <section className="rounded-[30px] border border-white/10 bg-[linear-gradient(180deg,rgba(63,78,95,0.62),rgba(37,49,62,0.82))] p-5 shadow-[0_24px_60px_rgba(15,23,42,0.18),inset_0_1px_0_rgba(255,255,255,0.05)]">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-100/68">
                  Inspector
                </div>
                <h2 className="mt-2 text-[26px] font-semibold tracking-[-0.03em] text-white">Document Details</h2>
                <p className="mt-2 text-sm leading-6 text-slate-300/78">
                  Inspect document metadata and chunk payloads before you rerank or generate against
                  them.
                </p>
              </div>
              <button
                type="button"
                className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-white/16 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                onClick={() => selectedDocumentId && void loadDocumentChunks(selectedDocumentId)}
                disabled={!selectedDocumentId || chunksLoading}
              >
                Reload
              </button>
            </div>

            {chunksError ? (
              <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {chunksError}
              </div>
            ) : null}

            {!selectedDocument ? (
              <div className="mt-6 rounded-3xl border border-dashed border-white/12 bg-slate-950/18 px-4 py-12 text-center text-sm text-slate-300/74">
                Select a document to inspect its stored chunks and lifecycle actions.
              </div>
            ) : (
              <>
                <div className="mt-6 rounded-3xl border border-white/10 bg-slate-950/18 p-4">
                  <div className="text-[26px] font-semibold tracking-[-0.03em] text-white">
                    {selectedDocument.filename || selectedDocument.document_id}
                  </div>
                  <div className="mt-2 break-all text-xs uppercase tracking-[0.14em] text-slate-300/74">
                    {selectedDocument.source_uri}
                  </div>
                  <dl className="mt-4 grid gap-3 text-sm md:grid-cols-2">
                    <div>
                      <dt className={fieldLabelClassName}>Document ID</dt>
                      <dd className="mt-1 break-all text-slate-300/74">
                        {selectedDocument.document_id}
                      </dd>
                    </div>
                    <div>
                      <dt className={fieldLabelClassName}>Indexed</dt>
                      <dd className="mt-1 text-slate-300/74">
                        {formatTimestamp(selectedDocument.indexed_at)}
                      </dd>
                    </div>
                    <div>
                      <dt className={fieldLabelClassName}>Namespace</dt>
                      <dd className="mt-1 text-slate-300/74">{selectedDocument.namespace || "—"}</dd>
                    </div>
                    <div>
                      <dt className={fieldLabelClassName}>Chunk Count</dt>
                      <dd className="mt-1 text-slate-300/74">{selectedDocument.chunk_count}</dd>
                    </div>
                  </dl>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <button
                      type="button"
                      className="rounded-full border border-rose-300/20 bg-rose-300/10 px-4 py-2 text-sm font-semibold text-rose-100 transition hover:border-rose-300/30 hover:bg-rose-300/14 disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={() => void deleteSelectedDocument()}
                      disabled={deleting}
                    >
                      {deleting ? "Deleting..." : "Delete Document"}
                    </button>
                    <button
                      type="button"
                      className="rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-white/16 hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
                      onClick={() => void replaceSelectedDocument()}
                      disabled={replacing || (indexMode !== "markdown" && indexMode !== "text")}
                    >
                      {replacing ? "Replacing..." : "Replace With Form"}
                    </button>
                  </div>
                </div>

                <div className="mt-6">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[22px] font-semibold tracking-[-0.03em] text-white">Stored Chunks</div>
                    <div className="text-xs uppercase tracking-[0.24em] text-slate-300/68">
                      {chunkResponse?.chunks.length ?? 0} loaded
                    </div>
                  </div>
                  <div className="mt-3 space-y-3">
                    {chunksLoading ? (
                      <div className="rounded-3xl border border-dashed border-white/12 bg-slate-950/18 px-4 py-10 text-center text-sm text-slate-300/74">
                        Loading chunks...
                      </div>
                    ) : (
                      chunkResponse?.chunks.map((chunk) => (
                        <article
                          key={chunk.chunk_id}
                          className="rounded-3xl border border-white/10 bg-slate-950/18 p-4"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-base font-semibold tracking-[-0.02em] text-white">
                              Chunk {chunk.chunk_index ?? "—"}
                            </div>
                            <div className="truncate text-[11px] uppercase tracking-[0.14em] text-slate-300/68">
                              {chunk.chunk_id}
                            </div>
                          </div>
                          <div className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-200">
                            {chunk.text}
                          </div>
                          {Object.keys(chunk.metadata || {}).length > 0 ? (
                            <pre className="mt-3 overflow-x-auto rounded-2xl bg-slate-950 px-4 py-3 text-xs text-slate-100">
                              {prettyJson(chunk.metadata)}
                            </pre>
                          ) : null}
                        </article>
                      ))
                    )}
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
    </AppShell>
  );
}
