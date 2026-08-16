"""Scheduling and tracking document-processing jobs.

This is what the API layer talks to: create a job, look one up by id, shut
down cleanly. Running a single job -- the queued/processing/done/failed
mechanics -- is ``app.jobs.runJob``; this module decides *when* that runs,
keeps the result reachable, and is what routes.py depends on.
"""

import asyncio
import logging

from app.documents import DocumentProcessor
from app.jobs import Job, JobStatus, runJob
from app.ragProcessor import RagIngestionProcessor

logger = logging.getLogger(__name__)


class JobConflictError(Exception):
    """A different document is already being ingested into this ragDbId."""


class JobManager:
    """In-memory job table, plus the background tasks running them.

    Jobs do not survive a restart -- there is no persistence layer. Fine for a
    single node; revisit if a queued job needs to survive a deploy.

    A job's id is its ``ragDbId`` (see ``Job.jobId``), so every submission for
    one database lands on one job. What a second submission means depends on
    whether it is the same work:

    * the same document again -- a retry, a duplicate delivery, an impatient
      caller -- returns the existing job untouched. Ingesting it twice would
      cost the same money to produce the same records.
    * a *different* document, while the first is still running, is refused.
      Both jobs write to one database under ids derived from chunk position,
      so whichever finishes last silently overwrites part of the other and
      leaves the database holding half of each. There is no safe way to run
      them at once, and cancelling the first has no safe halfway point to
      stop at either.
    * a different document once nothing is running replaces what is there,
      which is the ordinary way to re-ingest a database.

    A failed job is never reused -- resubmitting the same document after a
    failure retries it.
    """

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def __len__(self) -> int:
        return len(self._jobs)

    def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        """Record a queued job under ``ragDbId`` and run it in the background.

        Raises ``JobConflictError`` if a different document is mid-ingest into
        the same database.
        """
        existing = self._jobs.get(ragDbId)
        if existing is not None and existing.status is not JobStatus.FAILED:
            if (existing.documentLink, existing.serverId) == (documentLink, serverId):
                logger.info("Reusing job '%s'; same document already submitted.", ragDbId)
                return existing
            if existing.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                raise JobConflictError(
                    f"'{existing.documentLink}' is already being ingested into "
                    f"'{ragDbId}'. Wait for it to finish before submitting another."
                )

        job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
        self._jobs[job.jobId] = job

        task = asyncio.create_task(runJob(job, self._processor))
        # Hold a reference so the task can't be garbage-collected mid-flight,
        # and drop it on completion so the set doesn't grow without bound.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return job

    def get(self, jobId: str) -> Job | None:
        return self._jobs.get(jobId)

    async def shutdown(self) -> None:
        """Wait for every in-flight job to finish. Called when the app shuts down.

        Deliberately not a cancel: a job mid-download or mid-analysis has no
        safe halfway point to leave it at, so shutdown waits rather than
        killing it mid-write. This can't hang indefinitely -- a download is
        capped at ``DOWNLOAD_TIMEOUT_SECONDS`` and analysis afterward runs
        against bytes already in memory, so every job finishes (or fails) on
        its own within roughly that bound.
        """
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# One manager for the life of the process, same pattern as SERVER_REGISTRY.
JOB_MANAGER = JobManager(RagIngestionProcessor())


def getJobManager() -> JobManager:
    """FastAPI dependency: the process-wide job manager."""
    return JOB_MANAGER
