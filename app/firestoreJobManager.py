"""Jobs stored in Firestore, run as background tasks in this process.

The state problem is solved by ``FirestoreJobStore``; this adds the half that
actually runs the work. Ingestion runs here, as an asyncio task, which is a
deliberate limit: it needs a host that keeps running after the response is
sent (Cloud Run with an idle instance, Render, Railway, a VM), and it does
not work on request-scoped serverless, where the process is frozen once the
response is returned.

``CeleryJobManager`` is the same class with the work moved off this process
entirely -- see ``app.celeryJobManager``. Firestore fixes *state*; Celery is
what fixes *where the work runs*.
"""

from __future__ import annotations

import asyncio
import logging

from app.documents import DocumentProcessor
from app.firestoreJobStore import FirestoreJobStore
from app.jobs import Job, JobDispatchError, JobStatus, JobStore, runJob

logger = logging.getLogger(__name__)


class FirestoreJobManager:
    """``JobManager``'s interface, with the job table in Firestore."""

    def __init__(self, processor: DocumentProcessor, store: JobStore | None = None) -> None:
        self._processor = processor
        self._store = store if store is not None else FirestoreJobStore()
        self._tasks: set[asyncio.Task[None]] = set()

    async def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        """Claim ``ragDbId`` and start ingesting, or return the job already on it.

        Raises ``JobConflictError`` if a different document is mid-ingest into
        the same database. See ``app.jobs.resolveSubmission`` for the rules --
        they are shared with every other manager rather than restated here.

        Async, and the claim goes to a thread, because the Firestore client is
        synchronous: a transaction is a round trip to GCP, and running it
        directly on the event loop would stall every other request in the
        process for its duration. The in-memory manager has nothing to block
        on and is async only so both satisfy one interface.
        """
        job, isOurs = await asyncio.to_thread(
            self._store.claim, serverId=serverId, documentLink=documentLink, ragDbId=ragDbId
        )
        if not isOurs:
            logger.info("Reusing job '%s'; same document already submitted.", ragDbId)
            return job

        try:
            await self._start(job)
        except Exception as exc:
            # The claim is already written. Leaving it there would hold the
            # ragDbId at QUEUED with nothing running and nothing coming to run
            # it, so every later submission of a different document would be
            # refused with a 409 -- permanently, since only a finished or
            # failed job frees the id. Recording the failure hands it back.
            logger.exception("Could not start job '%s'; releasing the claim.", ragDbId)
            job.status = JobStatus.FAILED
            job.detail = f"Could not be queued for processing: {exc}"
            await self._record(job)
            raise JobDispatchError(
                f"'{ragDbId}' could not be queued for processing. Nothing was started; "
                f"resubmit when the queue is reachable."
            ) from exc

        return job

    async def _start(self, job: Job) -> None:
        """Begin the work. The one thing a Celery deployment does differently."""
        task = asyncio.create_task(self._run(job))
        # Hold a reference so the task can't be garbage-collected mid-flight,
        # and drop it on completion so the set doesn't grow without bound.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job: Job) -> None:
        """Run the job, writing its status to the table at both ends."""
        await runJob(job, self._processor, onStart=self._record)
        await self._record(job)

    async def _record(self, job: Job) -> None:
        """Write the job's current state to the table, best effort.

        Deliberately catches everything rather than only ``GoogleAPIError``.
        The work itself has already happened -- or is about to, in the case of
        the PROCESSING write -- so a bookkeeping failure of any kind should
        leave the job stale and loudly logged, not raise out of a background
        task where nothing is waiting to handle it. Narrowing this to one
        exception class only means the next unanticipated one escapes.
        """
        try:
            await asyncio.to_thread(self._store.save, job)
        except Exception:
            logger.exception("Could not record the status of job '%s'.", job.jobId)

    async def get(self, jobId: str) -> Job | None:
        return await asyncio.to_thread(self._store.get, jobId)

    def __len__(self) -> int:
        """A round trip to Firestore. Used by tests and the live checks, not by
        a route -- ``/document/status`` reads one job, never counts them."""
        return self._store.count()

    async def shutdown(self) -> None:
        """Wait for in-flight jobs, exactly as the in-memory manager does.

        Firestore holds the state, but the work is still running here, and a
        job killed mid-write leaves a half-populated database behind.
        """
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
