"""The end-to-end job processor: analyze -> select -> ingest -> store.

This is the composition root for ingestion. ``app.documents`` knows how to
read a document, ``app.ragSelector`` knows which chunking strategy suits it,
``app.ragIngestionPipeline`` knows how to chunk and store -- none of them
import each other, and this module is the only place that knows all three.
Swapping any one of them out is a change here and nowhere else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.chunkStoreFactory import buildChunkStore
from app.documents import analyze, download, extractText
from app.ragIngestionPipeline import RagIngestionPipeline
from app.ragSelector import RagSelector

if TYPE_CHECKING:
    from app.jobs import Job

logger = logging.getLogger(__name__)


class RagIngestionProcessor:
    """Runs a job's document all the way into its RAG database.

    Any exception here -- a bad URL, an unsupported type, a Pinecone failure
    -- is left to propagate; ``app.jobs.runJob`` is what catches it and marks
    the job failed.
    """

    def __init__(
        self,
        selector: RagSelector | None = None,
        pipeline: RagIngestionPipeline | None = None,
    ) -> None:
        self.selector = selector or RagSelector()
        self.pipeline = pipeline or RagIngestionPipeline(buildChunkStore())

    async def process(self, job: "Job") -> None:
        data, filename, contentType = await download(job.documentLink)
        metadata = analyze(job.documentLink, filename, data, contentType)
        job.metadata = metadata

        # The strategy is chosen from the metadata we just extracted, so a
        # 500-token note and a 500k-token manual are not chunked the same way.
        strategy = self.selector.suggest(metadata)

        # The document's bytes are already in hand, so the text comes from
        # them rather than from a second download inside the pipeline.
        text = extractText(filename, data, contentType)

        # job.jobId *is* the ragDbId being populated (see app.jobs.Job), so
        # the chunks land under the id the query side will ask for. Pinecone
        # does not hand out ids of its own -- the namespace and record ids are
        # ours to pick, which is why re-ingesting a ragDbId overwrites it.
        result = await self.pipeline.runText(
            text, job.jobId, sourceUrl=job.documentLink, strategy=strategy
        )

        job.detail = (
            f"Analyzed {metadata.fileCount} file(s) "
            f"({metadata.sourceKind}, {metadata.totalSizeBytes} bytes, "
            f"{metadata.tokenCount} token(s)) -- ingested "
            f"{len(result.chunks)} chunk(s) into '{job.jobId}' "
            f"using {strategy.name} chunking."
        )
