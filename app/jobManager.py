"""Scheduling and tracking document-processing jobs.

This is what the API layer talks to: create a job, look one up by id, shut
down cleanly. Running a single job -- the queued/processing/done/failed
mechanics -- is ``app.jobs.runJob``; this module decides *when* that runs,
keeps the result reachable, and is what routes.py depends on.
"""

import asyncio
import logging
import os

from app.documents import DocumentProcessor
from app.jobs import (
    Job,
    JobConflictError,
    JobDispatchError,
    Submission,
    conflictError,
    resolveSubmission,
    runJob,
)
from app.ragProcessor import RagIngestionProcessor

# Re-exported: routes.py and the tests import these from here, and they are
# defined in app.jobs so that every job manager raises the same classes.
__all__ = ["JobConflictError", "JobDispatchError", "JobManager", "getJobManager"]

logger = logging.getLogger(__name__)

# Firestore is used when a GCP project is named for it -- not merely when
# credentials happen to be present, since a machine can have application
# default credentials for unrelated reasons.
_USE_FIREBASE = bool(os.environ.get("GCP_PROJECT_ID"))

# Likewise: a broker URL is what says ingestion should be dispatched rather
# than run here.
_USE_CELERY = bool(os.environ.get("CELERY_BROKER_URL"))


class JobManager:
    """In-memory job table, plus the background tasks running them.

    Jobs do not survive a restart -- there is no persistence layer. Fine for a
    single node; revisit if a queued job needs to survive a deploy.

    A job's id is its ``ragDbId`` (see ``Job.jobId``), so every submission for
    one database lands on one job. What a second submission for that id means
    -- reuse, conflict, or a fresh job -- is ``app.jobs.resolveSubmission``,
    shared with the Firestore and Celery managers so the three cannot answer
    it differently.
    """

    def __init__(self, processor: DocumentProcessor) -> None:
        self._processor = processor
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def __len__(self) -> int:
        return len(self._jobs)

    async def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        """Record a queued job under ``ragDbId`` and run it in the background.

        Raises ``JobConflictError`` if a different document is mid-ingest into
        the same database.

        Nothing here blocks -- it is a dict write. Async only so that this and
        the managers backed by Firestore and Celery, which do block on network
        calls and have to get off the event loop to do it, present one
        interface to routes.py.
        """
        existing = self._jobs.get(ragDbId)
        outcome = resolveSubmission(existing, serverId=serverId, documentLink=documentLink)
        if outcome is Submission.REUSE:
            logger.info("Reusing job '%s'; same document already submitted.", ragDbId)
            return existing
        if outcome is Submission.CONFLICT:
            raise conflictError(existing, ragDbId)

        job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
        self._jobs[job.jobId] = job

        task = asyncio.create_task(runJob(job, self._processor))
        # Hold a reference so the task can't be garbage-collected mid-flight,
        # and drop it on completion so the set doesn't grow without bound.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return job

    async def get(self, jobId: str) -> Job | None:
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


def buildJobManager():
    """Pick the job manager from the environment.

    Three combinations, in order of how much they can survive:

    * broker + project -- Firestore holds the jobs, Celery workers run them.
    * project only -- Firestore holds the jobs, this process runs them.
    * neither -- a dict in this process holds them and runs them.

    Deliberately not a silent fallback. A misconfigured deployment that
    quietly reverted to the in-memory manager would look healthy while losing
    every job to the next restart and dropping the conflict check between
    instances -- the exact failures the other two exist to prevent. Better to
    refuse to start.
    """
    if _USE_CELERY:
        if not _USE_FIREBASE:
            # A broker moves the work to another process, so the job table has
            # to be reachable from both. With the in-memory table the API
            # would write a job the worker cannot see, the worker's status
            # writes would land nowhere the API reads, and /document/status
            # would answer 404 for every job actually running -- while the
            # whole thing looked configured and healthy.
            raise RuntimeError(
                "CELERY_BROKER_URL is set without GCP_PROJECT_ID. Dispatching to a "
                "worker needs a job table both processes can read; set GCP_PROJECT_ID "
                "(and GOOGLE_APPLICATION_CREDENTIALS outside GCP) to use Firestore, "
                "or unset CELERY_BROKER_URL to run ingestion in this process."
            )
        from app.celeryApp import HARD_TIME_LIMIT_SECONDS
        from app.celeryJobManager import CeleryJobManager
        from app.firestoreJobStore import FirestoreJobStore

        # Only here is reclaiming a stuck job safe. Celery's hard time limit
        # kills a task outright, so nothing can still be running past it, and a
        # job left QUEUED or PROCESSING well beyond it has definitively lost
        # its worker -- otherwise its ragDbId is blocked by a 409 forever, with
        # no restart to clear it now that the table outlives the process. The
        # margin is generous on purpose: reclaiming too early would start a
        # second ingestion alongside a live one. See resolveSubmission.
        staleAfter = float(
            os.environ.get("RAG_STALE_JOB_SECONDS", 2 * HARD_TIME_LIMIT_SECONDS)
        )
        logger.info("Using the Celery job manager; ingestion runs on workers.")
        return CeleryJobManager(
            RagIngestionProcessor(), FirestoreJobStore(staleAfterSeconds=staleAfter)
        )

    if _USE_FIREBASE:
        from app.firestoreJobManager import FirestoreJobManager

        logger.info("Using the Firestore job manager; ingestion runs in this process.")
        return FirestoreJobManager(RagIngestionProcessor())

    logger.info("Using the in-memory job manager; jobs will not survive a restart.")
    return JobManager(RagIngestionProcessor())


# One manager for the life of the process.
JOB_MANAGER = buildJobManager()


def getJobManager():
    """FastAPI dependency: the process-wide job manager."""
    return JOB_MANAGER
