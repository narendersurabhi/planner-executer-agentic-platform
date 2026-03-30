# RAG Playbook

This guide covers the current Qdrant-backed RAG path in this repo.

The RAG stack is intentionally split into separate capabilities:

- `rag.collection.ensure`
- `rag.index.upsert_texts`
- `rag.index.workspace_file`
- `rag.index.markdown`
- `rag.index.workspace_directory`
- `rag.retrieve`
- `rag.retrieve.rerank`

Use them together when you want an explicit retrieval pipeline instead of a hidden "retrieve and answer" agent.

## What Gets Deployed

For local Kubernetes, the repo now includes:

- a `qdrant` Deployment and Service
- persistent storage for Qdrant data
- a `rag-retriever-mcp` service that:
  - creates/ensures the default collection
  - embeds text with OpenAI embeddings
  - upserts chunks into Qdrant
  - retrieves ranked matches from Qdrant

Default local settings:

- `QDRANT_URL=http://qdrant:6333`
- `QDRANT_COLLECTION=rag_default`
- `RAG_EMBEDDING_MODEL=text-embedding-3-small`
- `QDRANT_VECTOR_SIZE=1536`
- `QDRANT_DISTANCE=Cosine`

## Capability Contracts

### `rag.collection.ensure`

Use this when you want to explicitly create or validate the target collection before indexing.

Typical input:

```json
{
  "collection_name": "rag_default",
  "vector_size": 1536,
  "distance": "Cosine",
  "create_payload_indexes": true
}
```

Typical output:

```json
{
  "collection_name": "rag_default",
  "status": "created",
  "vector_size": 1536,
  "distance": "Cosine",
  "payload_keyword_fields": [
    "document_id",
    "namespace",
    "tenant_id",
    "user_id",
    "workspace_id",
    "source_uri"
  ]
}
```

### `rag.index.upsert_texts`

Use this when you already have chunks of text to index.

Typical input:

```json
{
  "namespace": "docs",
  "tenant_id": "tenant-a",
  "workspace_id": "workspace-1",
  "entries": [
    {
      "document_id": "readme",
      "text": "Workflow Studio saves drafts and publishes versions.",
      "source_uri": "README.md#workflow-studio",
      "metadata": {
        "path": "README.md",
        "section": "Workflow Studio"
      }
    }
  ]
}
```

Typical output:

```json
{
  "collection_name": "rag_default",
  "upserted_count": 1,
  "chunk_ids": ["8ad0..."]
}
```

### `rag.retrieve`

Use this after indexing to pull back grounded chunks.

Typical input:

```json
{
  "query": "How does Workflow Studio versioning work?",
  "namespace": "docs",
  "tenant_id": "tenant-a",
  "workspace_id": "workspace-1",
  "top_k": 5
}
```

### `rag.retrieve.rerank`

Use this after retrieval when you want a deterministic second-pass ordering before
passing context into an answer step.

Typical input:

```json
{
  "query": "How does Workflow Studio versioning work?",
  "top_k": 3,
  "matches": [
    {
      "chunk_id": "chunk-1",
      "document_id": "docs/user-guide.md",
      "text": "Workflow Studio supports saved drafts and published versions.",
      "score": 0.71,
      "metadata": {
        "heading_path": ["Studio", "Versions"]
      },
      "source_uri": "docs/user-guide.md#studio"
    }
  ]
}
```

This capability:

- reranks `rag.retrieve` results without making another vector-database call
- combines original retrieval score with lexical overlap against chunk text and metadata
- is useful before grounded answer generation or citation selection

### `rag.index.workspace_file`

Use this when the source content already exists in the shared workspace.

Typical input:

```json
{
  "path": "docs/user-guide.md",
  "namespace": "docs",
  "tenant_id": "tenant-a",
  "workspace_id": "workspace-1",
  "metadata": {
    "repo": "agentic-workflow-studio"
  }
}
```

This capability:

- reads a text-like file under `WORKSPACE_DIR`
- chunks it into overlapping text segments
- embeds the chunks
- upserts them into the target Qdrant collection

### `rag.index.markdown`

Use this when the markdown content is already available as text and you want
heading-aware chunk metadata instead of raw character windows only.

Typical input:

```json
{
  "markdown_text": "# Overview\nWorkflow Studio supports saved drafts.\n\n## RAG\nUse markdown-aware indexing.",
  "source_uri": "docs/overview.md",
  "namespace": "docs",
  "user_id": "narendersurabhi",
  "metadata": {
    "repo": "agentic-workflow-studio"
  }
}
```

This capability:

- splits markdown into sections by headings
- preserves section metadata such as heading path and heading level
- chunks large sections into bounded overlapping slices
- embeds and upserts those section-aware chunks into Qdrant

### `rag.index.workspace_directory`

Use this when you want to index a whole directory under `WORKSPACE_DIR` instead
of a single file.

Typical input:

```json
{
  "directory_path": "docs",
  "namespace": "docs",
  "user_id": "narendersurabhi",
  "recursive": true,
  "max_files": 50,
  "metadata": {
    "repo": "agentic-workflow-studio"
  }
}
```

This capability:

- walks a bounded workspace directory
- skips hidden and obvious junk paths such as `.git` and `node_modules`
- uses markdown-aware chunking for `.md`, `.markdown`, and `.mdx`
- uses plain text chunking for other allowed text-like files
- returns per-file indexing results plus skipped-file reasons

## Recommended Workflow

### Minimal indexing flow

1. `rag.collection.ensure`
2. `rag.index.upsert_texts`, `rag.index.workspace_file`, `rag.index.markdown`, or `rag.index.workspace_directory`
3. `rag.retrieve`
4. optional `rag.retrieve.rerank`

### Grounded answering flow

1. `rag.collection.ensure`
2. `rag.index.upsert_texts`, `rag.index.workspace_file`, `rag.index.markdown`, or `rag.index.workspace_directory`
3. `rag.retrieve`
4. optional `rag.retrieve.rerank`
5. `llm.text.generate` or another answer-generation capability

In the generation step, pass the retrieved chunks as explicit context. Do not hide retrieval inside the answer step if you want predictable grounding and debuggability.

## Workflow Studio Pattern

In `Workflow Studio`, the clean graph is:

1. `rag.collection.ensure`
2. `rag.index.workspace_directory`
3. `rag.retrieve`
4. optional `rag.retrieve.rerank`
5. optional answer/render step

Recommended bindings:

- `rag.index.upsert_texts.collection_name` from a workflow input or leave blank to use the default collection
- `rag.index.upsert_texts.entries` from:
  - literal JSON for small tests
  - memory/context for dynamic content
  - previous step output if you add a chunking/extraction step later
- `rag.index.markdown.markdown_text` from a literal, memory entry, or previous extraction step
- `rag.index.workspace_directory.directory_path` from a workflow input such as `docs` or `workspace/repo-docs`
- `rag.retrieve.collection_name` should match the same collection
- `rag.retrieve.namespace`, `tenant_id`, `workspace_id`, or `user_id` should match the scope used at index time
- `rag.retrieve.rerank.matches` from the `rag.retrieve.matches` output when you want a second-pass ordering step

## Scope and Filtering

This RAG path is designed to be scope-aware.

Recommended filter fields:

- `namespace`
- `tenant_id`
- `user_id`
- `workspace_id`
- `document_id`
- `source_uri`

Important rule:

- if you index with scope fields, retrieve with the same scope fields

Otherwise recall quality and security boundaries both get worse.

## Operational Notes

- The retriever service will try to ensure the default collection at startup when `QDRANT_COLLECTION` is set.
- `rag.index.upsert_texts` also auto-ensures the collection by default.
- Chunk ids are deterministic when not supplied, so repeated upserts of the same text payload reuse the same point id.
- `rag.retrieve` is read-only and can be exposed to direct chat.
- `rag.retrieve.rerank` is also read-only, but it is usually more useful inside workflows than as a direct chat surface.
- `rag.collection.ensure` and `rag.index.upsert_texts` are write capabilities and should be used through Compose or Workflow Studio, not direct chat.

## Local Verification

Port-forward Qdrant:

```bash
kubectl port-forward -n awe svc/qdrant 16333:6333
```

Check health:

```bash
curl http://localhost:16333/readyz
```

## Next Extensions

Good follow-up capabilities:

- `rag.index.delete`
- `rag.index.sync_workspace_directory`
- `rag.retrieve.multi_query`
- `rag.answer.generate`
- `rag.answer.verify_grounding`
