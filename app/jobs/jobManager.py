"""Scheduling and tracking document-processing jobs.

This is what the API layer talks to: create a job, look one up by id, shut
down cleanly. Running a single job -- the queued/processing/done/failed
mechanics -- is ``app.jobs.job.runJob``; this module decides *when* that runs,
keeps the result reachable, and is what routes.py depends on.
"""

import asyncio
import logging
import os

from app.infra.redisClient import redisClient
from app.ingestion.documents import DocumentProcessor
from app.ingestion.ragProcessor import RagIngestionProcessor
from app.jobs.job import (
    Job,
    JobConflictError,
    JobDispatchError,
    Submission,
    conflictError,
    resolveSubmission,
    runJob,
)

# Re-exported: routes.py and the tests import these from here, and they are
# defined in app.jobs.job so that every job manager raises the same classes.
__all__ = ["JobConflictError", "JobDispatchError", "JobManager", "getJobManager"]

logger = logging.getLogger(__name__)

# Firestore is used when a GCP project is named for it -- not merely when
# credentials happen to be present, since a machine can have application
# default credentials for unrelated reasons. It holds the project mapping, not
# the job table; see app.stores.projectStore.
_USE_FIREBASE = bool(os.environ.get("GCP_PROJECT_ID"))


class JobManager:
    """In-memory job table, plus the background tasks running them.

    The development and test path: no Redis, no worker, ingestion on the API's
    own event loop, and nothing survives a restart. Fine for one process where
    losing a queued job costs nothing; wrong for a deployment, which is what
    the Redis path below exists for.

    A job's id is its ``ragDbId`` (see ``Job.jobId``), so every submission for
    one database lands on one job. What a second submission for that id means
    -- reuse, conflict, or a fresh job -- is ``app.jobs.job.resolveSubmission``,
    shared with the Redis store so the two cannot answer it differently.
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
        the Redis-backed manager, which does block on network calls and has to
        get off the event loop to do it, present one interface to routes.py.
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

        It is also the reason this path is not for deployments: a redeploy
        waits out whatever ingestion is running. The worker exists so the API
        has nothing to wait for.
        """
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def buildJobManager():
    """Pick the job manager from the environment.

    Two paths now, where there used to be three:

    * ``REDIS_URL`` set -- the job table is in Redis and a worker process runs
      the ingestion (``python -m app.jobs.worker``).
    * unset -- a dict in this process holds the jobs and runs them.

    Celery used to sit at the top of this list. It bought horizontal scaling,
    routing, a result backend and a scheduler, none of which a single node
    used, and cost a broker abstraction plus time limits that silently do not
    work on Windows. What was actually wanted -- get the work off the API
    process, and do not lose it if that process dies -- is what the Redis queue
    does in ``app.jobs.jobQueue``.

    Deliberately not a silent fallback. A misconfigured deployment that quietly
    reverted to the in-memory manager would look healthy while losing every job
    to the next restart and dropping the conflict check between instances --
    the exact failures the Redis path exists to prevent.
    """
    redis = redisClient()

    if redis is not None:
        if not _USE_FIREBASE:
            # Redis makes the *job table* shared, but the project mapping is a
            # separate store, and in-memory it lives only in the API process.
            # Jobs would then survive a restart while the mapping that resolves
            # a projectId to them did not, so /document/status would 404 jobs
            # that are still running, and a resubmitted project would mint a
            # second ragDbId and orphan the first one's vectors.
            raise RuntimeError(
                "REDIS_URL is set without GCP_PROJECT_ID. Durable jobs need a durable "
                "projectId -> ragDbId mapping to resolve them; set GCP_PROJECT_ID (and "
                "GOOGLE_APPLICATION_CREDENTIALS outside GCP) to keep the mapping in "
                "Firestore, or unset REDIS_URL to run everything in this process."
            )

        from app.jobs.queuedJobManager import QueuedJobManager
        from app.jobs.redisJobStore import RedisJobStore

        # Off by default, unlike under Celery. Celery's task_time_limit killed
        # an overrunning task outright, which is what made presuming a stuck
        # job dead safe; nothing here can kill CPU-bound work, so reclaiming is
        # a judgement call about the longest a job could legitimately take.
        # Setting it too low starts a second ingestion alongside a live one --
        # the interleaving the conflict check exists to prevent, arriving
        # through the check itself. See resolveSubmission.
        stale = os.environ.get("RAG_STALE_JOB_SECONDS")
        staleAfter = float(stale) if stale else None
        if staleAfter is None:
            logger.info(
                "RAG_STALE_JOB_SECONDS is unset: a job whose worker dies holds its "
                "project until the record's TTL expires."
            )

        logger.info("Using the Redis job manager; ingestion runs on a worker.")
        return QueuedJobManager(RedisJobStore(redis, staleAfterSeconds=staleAfter), redis)

    logger.info("Using the in-memory job manager; jobs will not survive a restart.")
    return JobManager(RagIngestionProcessor())


# One manager for the life of the process.
JOB_MANAGER = buildJobManager()


def getJobManager():
    """FastAPI dependency: the process-wide job manager."""
    return JOB_MANAGER
