# RAG Playbook

This guide covers the current Qdrant-backed RAG path in this repo.

The RAG stack is intentionally split into separate capabilities:

- `rag.collection.ensure`
- `rag.index.upsert_texts`
- `rag.index.workspace_file`
- `rag.retrieve`

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

## Recommended Workflow

### Minimal indexing flow

1. `rag.collection.ensure`
2. `rag.index.upsert_texts` or `rag.index.workspace_file`
3. `rag.retrieve`

### Grounded answering flow

1. `rag.collection.ensure`
2. `rag.index.upsert_texts` or `rag.index.workspace_file`
3. `rag.retrieve`
4. `llm.text.generate` or another answer-generation capability

In the generation step, pass the retrieved chunks as explicit context. Do not hide retrieval inside the answer step if you want predictable grounding and debuggability.

## Workflow Studio Pattern

In `Workflow Studio`, the clean graph is:

1. `rag.collection.ensure`
2. `rag.index.workspace_file`
3. `rag.retrieve`
4. optional answer/render step

Recommended bindings:

- `rag.index.upsert_texts.collection_name` from a workflow input or leave blank to use the default collection
- `rag.index.upsert_texts.entries` from:
  - literal JSON for small tests
  - memory/context for dynamic content
  - previous step output if you add a chunking/extraction step later
- `rag.retrieve.collection_name` should match the same collection
- `rag.retrieve.namespace`, `tenant_id`, `workspace_id`, or `user_id` should match the scope used at index time

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
- `rag.index.workspace_file`
- `rag.index.markdown_document`
- `rag.retrieve.rerank`
- `rag.answer.generate`
