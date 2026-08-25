"""The end-to-end job processor: analyze -> select -> ingest -> store.

This is the composition root for ingestion. ``app.ingestion.documents`` knows how to
read a document, ``app.ingestion.ragSelector`` knows which chunking strategy suits it,
``app.ingestion.ragIngestionPipeline`` knows how to chunk and store -- none of them
import each other, and this module is the only place that knows all three.
Swapping any one of them out is a change here and nowhere else.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.ingestion.documents import analyze, download, extractText
from app.ingestion.ragIngestionPipeline import RagIngestionPipeline
from app.ingestion.ragSelector import RagSelector
from app.stores.chunkStoreFactory import getChunkStore

if TYPE_CHECKING:
    from app.jobs.job import Job

logger = logging.getLogger(__name__)


class RagIngestionProcessor:
    """Runs a job's document all the way into its RAG database.

    Any exception here -- a bad URL, an unsupported type, a Pinecone failure
    -- is left to propagate; ``app.jobs.job.runJob`` is what catches it and marks
    the job failed.
    """

    def __init__(
        self,
        selector: RagSelector | None = None,
        pipeline: RagIngestionPipeline | None = None,
    ) -> None:
        self.selector = selector or RagSelector()
        # The store search reads from, not a second instance of it -- writing
        # to one store and searching another would find nothing.
        self.pipeline = pipeline or RagIngestionPipeline(getChunkStore())

    async def process(self, job: "Job") -> None:
        data, filename, contentType = await download(job.documentLink)

        # In a thread, not inline. `analyze` and `extractText` below are plain
        # synchronous functions doing real CPU work -- pdfplumber parsing every
        # page, tiktoken encoding the whole document -- and calling them
        # directly here runs them on whichever event loop owns this coroutine.
        # In the worker that is merely untidy; under the in-memory manager it
        # is the API's loop, and a large PDF freezes every other request for
        # however many seconds it takes.
        metadata = await asyncio.to_thread(
            analyze, job.documentLink, filename, data, contentType
        )
        job.metadata = metadata

        # The strategy is chosen from the metadata we just extracted, so a
        # 500-token note and a 500k-token manual are not chunked the same way.
        # Recorded on the job because a RAW document has no vector database to
        # query, and /document/status answers with the link instead.
        strategy = self.selector.suggest(metadata)
        job.strategy = strategy

        # The document's bytes are already in hand, so the text comes from
        # them rather than from a second download inside the pipeline. In a
        # thread for the same reason as `analyze` above.
        text = await asyncio.to_thread(extractText, filename, data, contentType)

        # job.jobId *is* the ragDbId being populated (see app.jobs.job.Job), so
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
