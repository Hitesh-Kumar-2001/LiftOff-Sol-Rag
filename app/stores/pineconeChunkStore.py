"""Pinecone-backed ``ChunkStore``.

A plugin, not a dependency of ``app.ingestion.ragIngestionPipeline`` -- that module
only knows the ``ChunkStore`` protocol. This is one implementation of it,
opted into by passing ``PineconeChunkStore()`` as ``RagIngestionPipeline``'s
``store``; nothing here is imported unless a caller wants Pinecone.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from functools import lru_cache, partial

from pinecone import Pinecone
from pinecone.exceptions import NotFoundException, PineconeApiException

from app.ingestion.ragIngestionPipeline import Chunk, IngestionError, SearchResult

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

# Ids per delete call, when clearing records a re-ingest left behind.
DELETE_BATCH_SIZE = 1000

logger = logging.getLogger(__name__)


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
        # constructed at import time (see app.ingestion.ragProcessor), long before any
        # job needs it, and a process that never ingests should not require a
        # key to start.
        self.indexName = indexName
        self.index = None
        self.indexLock = asyncio.Lock()

    async def ensureIndex(self) -> None:
        """Create the index if it is missing, once however many jobs ask.

        Jobs run concurrently, so without the lock two documents submitted
        together both look for the index, both find it missing, and both try
        to create it -- and the loser's job fails. The check is repeated
        inside the lock because the first caller through will have created it
        by the time the second acquires.

        A conflict is still possible from another *process* racing this one,
        which no lock here can prevent; that means the index now exists,
        which is all this method wanted.
        """
        if self.index is not None:
            return

        async with self.indexLock:
            if self.index is not None:
                return

            client = pineconeClient()
            existing = await asyncio.to_thread(client.list_indexes)
            if self.indexName not in [i["name"] for i in existing]:
                try:
                    await asyncio.to_thread(
                        client.create_index_for_model,
                        name=self.indexName,
                        cloud=PINECONE_CLOUD,
                        region=PINECONE_REGION,
                        embed={
                            "model": PINECONE_EMBED_MODEL,
                            "field_map": {"text": "chunkText"},
                        },
                    )
                except PineconeApiException as exc:
                    if getattr(exc, "status", None) != 409:
                        raise
                    logger.info("Index '%s' was created concurrently.", self.indexName)
            self.index = client.Index(self.indexName)

    async def listIds(self, ragDbId: str) -> list[str]:
        """Every record id stored for ``ragDbId``.

        ``index.list`` pages, and each page is a ``ListResponse`` whose
        ``vectors`` are ``ListItem`` objects -- the ids are on ``.id``, not
        the items themselves, which the rest of the SDK expects as plain
        strings.
        """
        await self.ensureIndex()
        namespace = namespaceFor(ragDbId)

        def readPages() -> list[str]:
            return [
                item.id
                for page in self.index.list(namespace=namespace)
                for item in page.vectors
            ]

        try:
            return await asyncio.to_thread(readPages)
        except NotFoundException:
            return []

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
        # What is already stored, so a re-ingest can clear whatever the new
        # document does not overwrite. An upsert only replaces ids it names:
        # re-ingesting a shorter document leaves every record past its last
        # chunk in place, and search goes on returning that old text as if it
        # belonged to the new document.
        stale = set(await self.listIds(ragDbId))

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

        # After the new records are in, not before: clearing first would
        # leave the database empty for the length of the upsert, and lose
        # the old document entirely if the upsert then failed.
        stale -= {record["_id"] for record in records}
        if stale:
            logger.info("Removing %d record(s) left by a previous ingest.", len(stale))
            await self.deleteIds(sorted(stale), namespace)

    async def deleteIds(self, ids: list[str], namespace: str) -> None:
        for start in range(0, len(ids), DELETE_BATCH_SIZE):
            await asyncio.to_thread(
                partial(
                    self.index.delete,
                    ids=ids[start : start + DELETE_BATCH_SIZE],
                    namespace=namespace,
                )
            )

    async def get(self, ragDbId: str) -> list[Chunk]:
        namespace = namespaceFor(ragDbId)

        ids = await self.listIds(ragDbId)
        if not ids:
            return []

        fetched = await asyncio.to_thread(
            partial(self.index.fetch, ids=ids, namespace=namespace)
        )
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
