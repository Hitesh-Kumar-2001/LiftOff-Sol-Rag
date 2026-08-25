"""Which ``ChunkStore`` this process ingests into.

The same shape as ``app.stores.projectStore.buildProjectStore``: one place picks the
implementation from the environment, so nothing else has to know that more
than one exists.

Test mode is opt-in and never the default. A process that has not set
``RAG_TEST_MODE`` ingests into Pinecone -- there is no configuration
mistake that silently downgrades a real deployment to writing JSON into a
directory nothing can query, because the local store refuses to be built at
all without the flag.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.ingestion.ragIngestionPipeline import ChunkStore
from app.stores.localChunkStore import LocalChunkStore, testModeEnabled
from app.stores.pineconeChunkStore import PineconeChunkStore

logger = logging.getLogger(__name__)


def buildChunkStore() -> ChunkStore:
    """The chunk store for this process: local on disk under test mode,
    Pinecone otherwise."""
    if testModeEnabled():
        logger.warning("RAG_TEST_MODE is on -- ingesting to the local chunk store, not Pinecone.")
        return LocalChunkStore()
    return PineconeChunkStore()


@lru_cache(maxsize=1)
def getChunkStore() -> ChunkStore:
    """FastAPI dependency: the process-wide chunk store.

    One instance for the life of the process, same pattern as JOB_MANAGER --
    ingestion writes to it and search reads from it, and those have to be the
    same store or a search would look somewhere nothing was ever written.
    Built lazily so importing this module needs no credentials.
    """
    return buildChunkStore()
