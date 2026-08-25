"""Jobs claimed in Redis, run by a worker process somewhere else.

The half of the old ``CeleryJobManager`` that was actually doing something.
Accepting a submission -- resolve the project, claim the ragDbId atomically,
answer 409 or 503 -- is identical whether the work then runs here or on a
worker. Only the last step changes, and here that step is one ``LPUSH``.

Two things follow from the split, and both are deliberate:

* the claim still happens *here*, before the enqueue, not in the worker. If the
  worker claimed, the API could not answer 409 -- it would have accepted the
  submission and returned 202 before anything checked, and the caller would
  learn about the conflict never.
* the job table must be shared, which is why this requires Redis rather than
  the dict ``JobManager`` keeps. See the factory in ``app.jobs.jobManager``.
"""

from __future__ import annotations

import asyncio
import logging

from redis import Redis

from app.ingestion.documents import DocumentProcessor
from app.jobs.job import Job, JobDispatchError, JobStatus, JobStore, runJob
from app.jobs.jobQueue import enqueue

logger = logging.getLogger(__name__)


class QueuedJobManager:
    """``JobManager``'s interface, with the table in Redis and the work on a
    worker."""

    def __init__(self, store: JobStore, redis: Redis) -> None:
        self._store = store
        self._redis = redis

    async def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        """Claim ``ragDbId`` and queue it, or return the job already on it.

        Raises ``JobConflictError`` if a different document is mid-ingest into
        the same database. See ``app.jobs.job.resolveSubmission`` for the rules --
        shared with every other manager rather than restated here.

        Async, and the claim goes to a thread, because redis-py is synchronous.
        A round trip is usually a millisecond, but running it on the event loop
        would stall every other request in the process for its duration, and
        against an unreachable Redis it blocks for the connect timeout.
        """
        job, isOurs = await asyncio.to_thread(
            self._store.claim, serverId=serverId, documentLink=documentLink, ragDbId=ragDbId
        )
        if not isOurs:
            logger.info("Reusing job '%s'; same document already submitted.", ragDbId)
            return job

        try:
            await asyncio.to_thread(enqueue, self._redis, job.jobId)
        except Exception as exc:
            # The claim is already written. Leaving it there would hold the
            # ragDbId at QUEUED with nothing running and nothing coming to run
            # it, so every later submission of a different document would be
            # refused with a 409 -- until the record's TTL expired, which is
            # days. Recording the failure hands the id back now.
            logger.exception("Could not queue job '%s'; releasing the claim.", ragDbId)
            job.status = JobStatus.FAILED
            job.detail = f"Could not be queued for processing: {exc}"
            await self._record(job)
            raise JobDispatchError(
                f"'{ragDbId}' could not be queued for processing. Nothing was started; "
                f"resubmit when the queue is reachable."
            ) from exc

        logger.info("Queued job '%s' for a worker.", job.jobId)
        return job

    async def _record(self, job: Job) -> None:
        """Write the job's current state to the table, best effort.

        Deliberately catches everything rather than one exception class. The
        work has already happened -- or is about to -- so a bookkeeping failure
        of any kind should leave the job stale and loudly logged, not raise out
        of a path where nothing is waiting to handle it.
        """
        try:
            await asyncio.to_thread(self._store.save, job)
        except Exception:
            logger.exception("Could not record the status of job '%s'.", job.jobId)

    async def get(self, jobId: str) -> Job | None:
        return await asyncio.to_thread(self._store.get, jobId)

    def __len__(self) -> int:
        return self._store.count()

    async def shutdown(self) -> None:
        """Returns immediately -- no job runs in this process.

        Not an oversight, and the main operational win of dispatching: the API
        can be redeployed or scaled down without waiting on ingestion, because
        there is nothing in flight here to wait for. In-flight jobs belong to
        the worker, which is stopped on its own.
        """


async def runJobHere(job: Job, processor: DocumentProcessor, store: JobStore) -> None:
    """Run one job and write its status to the table at both ends.

    Lives here rather than in the worker so the worker stays a loop and this
    stays testable without one.
    """

    async def record(current: Job) -> None:
        try:
            await asyncio.to_thread(store.save, current)
        except Exception:
            logger.exception("Could not record the status of job '%s'.", current.jobId)

    await runJob(job, processor, onStart=record)
    await record(job)
