"""Which ``ChunkStore`` this process ingests into.

The same shape as ``app.credentials.buildCredentialSource``: one place picks
the implementation from the environment, so nothing else has to know that
more than one exists.

Test mode is opt-in and never the default. A process that has not set
``RAG_TEST_MODE`` ingests into Pinecone -- there is no configuration
mistake that silently downgrades a real deployment to writing JSON into a
directory nothing can query, because the local store refuses to be built at
all without the flag.
"""

from __future__ import annotations

import logging

from app.localChunkStore import LocalChunkStore, testModeEnabled
from app.pineconeChunkStore import PineconeChunkStore
from app.ragIngestionPipeline import ChunkStore

logger = logging.getLogger(__name__)


def buildChunkStore() -> ChunkStore:
    """The chunk store for this process: local on disk under test mode,
    Pinecone otherwise."""
    if testModeEnabled():
        logger.warning("RAG_TEST_MODE is on -- ingesting to the local chunk store, not Pinecone.")
        return LocalChunkStore()
    return PineconeChunkStore()
