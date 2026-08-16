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
from functools import lru_cache, partial

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException

from app.ragIngestionPipeline import Chunk, IngestionError, SearchResult

ENV_PINECONE_API_KEY = "PINECONE_API_KEY"
PINECONE_INDEX_NAME = os.environ.get("RAG_PINECONE_INDEX", "rag-chunks")
PINECONE_CLOUD = os.environ.get("RAG_PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("RAG_PINECONE_REGION", "us-east-1")
# Pinecone-hosted embedding model -- the index embeds chunk text itself, so
# this pipeline never has to run its own embedding step.
PINECONE_EMBED_MODEL = os.environ.get("RAG_PINECONE_EMBED_MODEL", "llama-text-embed-v2")

# Records per upsert call. Pinecone's own ceiling (the embedding model reports
# it as max_batch_size); exceeding it fails the entire request rather than
# truncating the batch.
UPSERT_BATCH_SIZE = 96


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

        namespace = namespaceFor(ragDbId)
        # Pinecone rejects the whole request over its batch limit, and any
        # document worth chunking produces more records than that -- a 10k
        # token article is a few hundred. Sent in order, one batch at a time,
        # so a rate limit is not provoked by fanning them out.
        for start in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[start : start + UPSERT_BATCH_SIZE]
            # Keyword arguments: upsert_records takes them keyword-only, and
            # passing them positionally raises rather than being silently
            # misordered.
            await asyncio.to_thread(
                partial(self.index.upsert_records, records=batch, namespace=namespace)
            )

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

    async def search(self, ragDbId: str, query: str, topK: int = 5) -> list[SearchResult]:
        """Nearest chunks to ``query`` within this database's namespace.

        The query text is embedded by Pinecone with the same model the
        records were, which is the point of an integrated-embedding index:
        nothing here has to know the model, or keep in step with it.

        A namespace that was never written to is not an error -- it means
        nothing has been ingested under this ragDbId yet, so there is nothing
        to match.
        """
        await self.ensureIndex()
        try:
            response = await asyncio.to_thread(
                self.index.search,
                namespace=namespaceFor(ragDbId),
                top_k=topK,
                inputs={"text": query},
                fields=["chunkText", "chunkIndex"],
            )
        except NotFoundException:
            return []

        return [
            SearchResult(
                text=hit.fields.get("chunkText", ""),
                index=int(hit.fields.get("chunkIndex", 0)),
                score=float(hit.score),
            )
            for hit in response.result.hits
        ]

    async def delete(self, ragDbId: str) -> None:
        """Drop every chunk stored for ``ragDbId``.

        Deleting the whole namespace rather than the ids in it: the namespace
        holds nothing else, and a namespace that was never written to is not
        an error worth raising -- Pinecone reports that as a 404, which here
        just means there was nothing to delete.
        """
        await self.ensureIndex()
        try:
            await asyncio.to_thread(
                self.index.delete, delete_all=True, namespace=namespaceFor(ragDbId)
            )
        except NotFoundException:
            pass
