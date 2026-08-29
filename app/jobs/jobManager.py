"""Scheduling and tracking document-processing jobs.

This is what the API layer talks to: create a job, look one up by id, shut
down cleanly. Running a single job -- the queued/processing/done/failed
mechanics -- is ``app.jobs.job.runJob``; this module decides *when* that runs,
keeps the result reachable, and is what routes.py depends on.
"""

import logging
import os
from functools import lru_cache
from typing import Protocol

from app.infra.redisClient import redisClient
from app.jobs.job import Job, JobConflictError, JobDispatchError

# Re-exported: routes.py and the tests import these from here, and they are
# defined in app.jobs.job so that every job manager raises the same classes.
__all__ = ["JobConflictError", "JobDispatchError", "JobManager", "getJobManager"]

logger = logging.getLogger(__name__)

# Firestore is used when a GCP project is named for it -- not merely when
# credentials happen to be present, since a machine can have application
# default credentials for unrelated reasons. It holds the project mapping, not
# the job table; see app.stores.projectStore.
def _useFirestore() -> bool:
    """Read at build time, not import time -- the manager is built lazily now,
    and a module-level constant would freeze whatever the environment happened
    to say when the first consumer imported this."""
    return bool(os.environ.get("GCP_PROJECT_ID"))


class JobManager(Protocol):
    """What ``routes.py`` needs from a job manager, whichever one is built.

    A Protocol rather than a class now that there is one implementation
    (``app.jobs.queuedJobManager.QueuedJobManager``): the routes are typed
    against the interface, not against Redis, so a second backend is a class
    that satisfies this and a branch in ``buildJobManager`` -- the same seam
    ``ChunkStore`` and ``ProjectStore`` use.

    A job's id is its ``ragDbId`` (see ``Job.jobId``), so every submission for
    one database lands on one job. What a second submission for that id means
    -- reuse, conflict, or a fresh job -- is ``app.jobs.job.resolveSubmission``,
    which every backend calls rather than reimplementing.
    """

    async def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        """Claim ``ragDbId`` and schedule the ingestion.

        Raises ``JobConflictError`` if a different document is mid-ingest into
        the same database, and ``JobDispatchError`` if the work could not be
        handed off -- in which case the claim is released, or that ragDbId
        would answer 409 forever.
        """
        ...

    async def get(self, jobId: str) -> Job | None:
        """The job for ``jobId``, or None if there is no record of one."""
        ...

    async def shutdown(self) -> None:
        """Release anything held open. Must not wait on the worker."""
        ...


def buildJobManager():
    """Build the job manager. Redis, or nothing.

    One path, where there used to be three. ``REDIS_URL`` holds the job table
    and the queue, and ``python -m app.jobs.worker`` is what ingests.

    The in-process manager is gone. It kept the job table in a dict and ran
    ingestion on the API's own event loop, which made it two problems rather
    than one: jobs died with the process, and a large document blocked every
    other request -- including the health check -- for as long as chunking took,
    so a platform watching the service killed a task that was working. Running
    the work somewhere else is the fix for both, and there is no version of
    "somewhere else" that lives in this process.

    Celery used to sit at the top of this list. It bought horizontal scaling,
    routing, a result backend and a scheduler, none of which a single node
    used, and cost a broker abstraction plus time limits that silently do not
    work on Windows. What was actually wanted -- get the work off the API
    process, and do not lose it if that process dies -- is what the Redis queue
    does in ``app.jobs.jobQueue``.

    Raising rather than falling back is the whole point. A misconfigured
    deployment that quietly reverted to a dict looked healthy while losing every
    job to the next restart and dropping the conflict check between instances --
    the exact failures the Redis path exists to prevent.
    """
    redis = redisClient()

    if redis is not None:
        if not _useFirestore():
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
                "Firestore. There is no longer an in-process fallback to drop to."
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

    raise RuntimeError(
        "REDIS_URL is not set, so there is nowhere to queue ingestion. It used to "
        "fall back to a dict in this process; that ran chunking on the API's event "
        "loop, which blocked every other request -- the health check included -- for "
        "the length of the document. Set REDIS_URL and run `python -m app.jobs.worker` "
        "alongside the API (docker-compose.yml does both)."
    )


@lru_cache(maxsize=1)
def getJobManager():
    """FastAPI dependency: the process-wide job manager.

    Built on first use, not at import. It used to be a module-level
    ``JOB_MANAGER = buildJobManager()``, which made *importing this module*
    require a working configuration -- so every consumer, including anything
    that only wanted the exception types, had to set the environment first and
    in the right order. Lazy and cached matches ``getProjectStore``,
    ``getConversationStore`` and ``getChunkStore``, and ``app.main.checkConfiguration``
    calls it during startup so a missing REDIS_URL still fails the deploy
    rather than the first upload.
    """
    return buildJobManager()
