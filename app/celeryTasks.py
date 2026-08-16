"""The worker end: what a Celery worker actually does with a queued job.

The task takes a ragDbId and nothing else. The job's fields -- link, caller,
status -- are read from the job table rather than carried in the message, so
the worker always acts on the authoritative record. A message carrying a copy
of the job would be a second source of truth that goes stale the moment
anything writes to the table, and would have to be kept in step with the
dataclass forever.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.celeryApp import celeryApp
from app.jobs import Job, JobStatus, JobStore, runJob

logger = logging.getLogger(__name__)

TASK_NAME = "rag.ingestDocument"


@lru_cache(maxsize=1)
def _workerLoop() -> asyncio.AbstractEventLoop:
    """One event loop for the life of this worker process.

    Deliberately not ``asyncio.run`` per task. Loop-affine objects outlive a
    single task here: the chunk store is cached per process (see
    ``getChunkStore``) and holds an ``asyncio.Lock``, which binds to the first
    loop that acquires it and raises "bound to a different event loop" on any
    later one. A fresh loop per task would work until the first task that
    failed before the store finished initialising, then break every task after
    it -- a failure that only appears under load, in a worker, after an
    unrelated error. One loop avoids the whole class of it.

    Safe because a worker process runs one task at a time under the prefork
    and solo pools. It is not safe under gevent/eventlet, which are not the
    right pools for CPU- and IO-heavy ingestion anyway.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


@lru_cache(maxsize=1)
def _store() -> JobStore:
    from app.firestoreJobStore import FirestoreJobStore

    return FirestoreJobStore()


@lru_cache(maxsize=1)
def _processor():
    from app.ragProcessor import RagIngestionProcessor

    # Built lazily, on first use inside each worker process: with the prefork
    # pool the parent forks after import, and a Pinecone client built before
    # the fork would have its connection pool shared between children.
    return RagIngestionProcessor()


def runIngestion(ragDbId: str, store: JobStore, processor, loop=None) -> Job | None:
    """Run the queued job for ``ragDbId`` and write the outcome back.

    Separate from the task so it can be tested without a broker, and so the
    Celery decorator stays a thin wrapper over ordinary code.

    Returns None if the job is gone -- the database was deleted, or the table
    was cleared, between dispatch and pickup. That is not an error worth
    retrying: there is nothing left to describe what to ingest.
    """
    job = store.get(ragDbId)
    if job is None:
        logger.warning("Job '%s' is no longer in the job table; nothing to run.", ragDbId)
        return None

    if job.status is JobStatus.DONE:
        # Delivery is at-least-once, and acks are late: a worker killed between
        # finishing and acking leaves its message on the queue to be handed out
        # again. A claim always precedes a dispatch, so a job already DONE at
        # pickup was finished by someone else -- re-running it would pay for
        # the same Gemini calls and upserts twice, and put a second writer into
        # a namespace another ingestion may already be re-populating.
        logger.info("Job '%s' is already done; skipping a redelivered message.", ragDbId)
        return job

    async def record(current: Job) -> None:
        """Publish the status transition. Blocking the loop is fine here --
        this one belongs to the worker and has nothing else on it."""
        save(current)

    def save(current: Job) -> None:
        try:
            store.save(current)
        except Exception:
            # The work has already happened, or is about to. Losing a
            # bookkeeping write leaves the job stale and loudly logged; failing
            # the task would re-run the entire ingestion to fix it.
            logger.exception("Could not record the status of job '%s'.", ragDbId)

    loop = loop or _workerLoop()
    # runJob records a processor failure on the job rather than raising, so a
    # permanently bad document (404 link, unsupported type) is written back as
    # FAILED instead of throwing out of the task. That matters: a raise here
    # would let Celery treat a document that can never succeed as a transient
    # fault, and the caller would never see why. Resubmitting a FAILED job is
    # already how a retry is requested -- see app.jobs.resolveSubmission.
    #
    # Celery's SoftTimeLimitExceeded is an ordinary Exception, so an overrunning
    # job lands there too and is recorded as failed with the reason, rather than
    # vanishing when the hard limit kills the worker.
    loop.run_until_complete(runJob(job, processor, onStart=record))
    save(job)

    return job


@celeryApp.task(name=TASK_NAME)
def ingestDocument(ragDbId: str) -> str | None:
    """Ingest the document for ``ragDbId``. Returns its final status."""
    job = runIngestion(ragDbId, _store(), _processor())
    return job.status.value if job is not None else None
