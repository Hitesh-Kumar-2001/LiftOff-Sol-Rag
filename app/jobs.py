"""A job's data, and the mechanics of running one.

A ``Job`` is just a record -- id, status, timestamps. ``runJob`` is the one
function that knows how to execute it: hand it to a processor, translate the
outcome into a status. Deciding when a job runs, keeping it reachable by id,
and returning it to the API is ``app.jobManager.JobManager``'s job, not this
module's -- this module only knows how to run one, given one.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.documents import DocumentMetadata, DocumentProcessor

logger = logging.getLogger(__name__)


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    # Deliberately the ragDbId being ingested into, not a generated id -- a
    # job exists to populate one RAG database, so there is nothing else
    # meaningful to key it by. See JobManager.create.
    jobId: str
    serverId: str
    documentLink: str
    status: JobStatus = JobStatus.QUEUED
    detail: str | None = None
    # Set by DocumentAnalyzerProcessor once the download is analyzed. None
    # until then, and for processors (like the stub) that don't produce it.
    metadata: "DocumentMetadata | None" = None
    createdAt: datetime = field(default_factory=_now)
    updatedAt: datetime = field(default_factory=_now)


async def runJob(job: Job, processor: "DocumentProcessor") -> None:
    """Execute ``job`` against ``processor``, updating its status as it goes.

    A processor failure is caught and recorded on the job rather than raised,
    so a caller scheduling this as a background task never sees an unhandled
    exception surface from that task.
    """
    job.status = JobStatus.PROCESSING
    job.updatedAt = _now()
    try:
        await processor.process(job)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Job %s failed.", job.jobId)
        job.status = JobStatus.FAILED
        job.detail = str(exc)
    else:
        job.status = JobStatus.DONE
    job.updatedAt = _now()
