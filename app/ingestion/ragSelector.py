"""Picks a chunking strategy from a document's total token count.

Single-choice per size band today; ``availableImplementations`` is where a
real registry -- multiple RAG backends to weigh against each other -- will
grow into once there's more than one option per band.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ingestion.ragIngestionPipeline import ChunkingStrategy

if TYPE_CHECKING:
    # Only needed for the type hint below -- importing it for real would make
    # app.ingestion.documents and app.ingestion.ragSelector import each other.
    from app.ingestion.documents import DocumentMetadata

logger = logging.getLogger(__name__)

# Token-count bands (inclusive lower bound):
#   < RAW_MAX_TOKENS                         -> RAW    (too small to bother chunking)
#   [RAW_MAX_TOKENS, NON_AI_MAX_TOKENS)       -> NON_AI (simple vector segmentation)
#   [NON_AI_MAX_TOKENS, AI_LOG_THRESHOLD)     -> AI
#   >= AI_LOG_THRESHOLD                       -> AI, logged (unusually large)
RAW_MAX_TOKENS = 2000
NON_AI_MAX_TOKENS = 10000
AI_LOG_THRESHOLD_TOKENS = 100000


class RagSelector:
    """Suggests a chunking strategy for a document. Single-choice for now;
    will be modularized into a registry of selectable implementations later.
    """

    def suggest(self, documentMetadata: "DocumentMetadata") -> ChunkingStrategy:
        """Return the chunking strategy to ingest ``documentMetadata`` with."""
        return self.score(documentMetadata)

    def score(self, documentMetadata: "DocumentMetadata") -> ChunkingStrategy:
        tokenCount = documentMetadata.tokenCount

        if tokenCount < RAW_MAX_TOKENS:
            return ChunkingStrategy.RAW

        if tokenCount < NON_AI_MAX_TOKENS:
            return ChunkingStrategy.NON_AI

        if tokenCount >= AI_LOG_THRESHOLD_TOKENS:
            logger.warning(
                "Document '%s' has %d tokens (>= %d) -- unusually large; "
                "using AI chunking anyway.",
                documentMetadata.sourceUrl,
                tokenCount,
                AI_LOG_THRESHOLD_TOKENS,
            )

        return ChunkingStrategy.AI

    def availableImplementations(self) -> list[ChunkingStrategy]:
        """Placeholder for the future registry of selectable implementations."""
        return list(ChunkingStrategy)
