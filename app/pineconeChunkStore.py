"""Pinecone-backed ``ChunkStore``.

A plugin, not a dependency of ``app.ragIngestionPipeline`` -- that module
only knows the ``ChunkStore`` protocol. This is one implementation of it,
opted into by passing ``PineconeChunkStore()`` as ``RagIngestionPipeline``'s
``store``; nothing here is imported unless a caller wants Pinecone.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from functools import lru_cache

from pinecone import Pinecone

from app.ragIngestionPipeline import Chunk, IngestionError

ENV_PINECONE_API_KEY = "PINECONE_API_KEY"
PINECONE_INDEX_NAME = os.environ.get("RAG_PINECONE_INDEX", "rag-chunks")
PINECONE_CLOUD = os.environ.get("RAG_PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("RAG_PINECONE_REGION", "us-east-1")
# Pinecone-hosted embedding model -- the index embeds chunk text itself, so
# this pipeline never has to run its own embedding step.
PINECONE_EMBED_MODEL = os.environ.get("RAG_PINECONE_EMBED_MODEL", "llama-text-embed-v2")


def namespaceFor(ragDbId: str) -> str:
    """One Pinecone namespace per ``ragDbId``, so ``get`` can list its chunks
    without a metadata filter. Hashed to sidestep any character Pinecone
    namespaces don't like in a raw id."""
    return hashlib.sha256(ragDbId.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def pineconeClient() -> Pinecone:
    apiKey = os.environ.get(ENV_PINECONE_API_KEY)
    if not apiKey:
        raise IngestionError(f"{ENV_PINECONE_API_KEY} is not set.")
    return Pinecone(api_key=apiKey)


class PineconeChunkStore:
    """Chunks stored as records in a Pinecone index with integrated
    (server-side) embedding -- ``save``/``get`` are the only two things
    ``RagIngestionPipeline`` needs, so this is a drop-in swap for
    ``InMemoryChunkStore`` wherever a ``ChunkStore`` is expected.
    """

    def __init__(self, indexName: str = PINECONE_INDEX_NAME) -> None:
        # Nothing here touches Pinecone or reads the API key: this store is
        # constructed at import time (see app.ragProcessor), long before any
        # job needs it, and a process that never ingests should not require a
        # key to start.
        self.indexName = indexName
        self.index = None

    async def ensureIndex(self) -> None:
        if self.index is not None:
            return
        client = pineconeClient()
        existing = await asyncio.to_thread(client.list_indexes)
        if self.indexName not in [i["name"] for i in existing]:
            await asyncio.to_thread(
                client.create_index_for_model,
                name=self.indexName,
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
                embed={"model": PINECONE_EMBED_MODEL, "field_map": {"text": "chunkText"}},
            )
        self.index = client.Index(self.indexName)

    async def save(self, ragDbId: str, chunks: list[Chunk]) -> None:
        await self.ensureIndex()
        records = [
            {
                "_id": f"{ragDbId}::{chunk.index}",
                "chunkText": chunk.text,
                "ragDbId": ragDbId,
                "chunkIndex": chunk.index,
                "tokenCount": chunk.tokenCount,
            }
            for chunk in chunks
        ]
        await asyncio.to_thread(self.index.upsert_records, namespaceFor(ragDbId), records)

    async def get(self, ragDbId: str) -> list[Chunk]:
        await self.ensureIndex()
        namespace = namespaceFor(ragDbId)

        ids: list[str] = []
        for batch in await asyncio.to_thread(lambda: list(self.index.list(namespace=namespace))):
            ids.extend(batch)
        if not ids:
            return []

        fetched = await asyncio.to_thread(self.index.fetch, ids=ids, namespace=namespace)
        chunks = [
            Chunk(
                text=record.metadata.get("chunkText", ""),
                index=record.metadata.get("chunkIndex", 0),
                tokenCount=record.metadata.get("tokenCount", 0),
            )
            for record in fetched.vectors.values()
        ]
        chunks.sort(key=lambda c: c.index)
        return chunks
