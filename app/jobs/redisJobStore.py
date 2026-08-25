"""The job table, in Redis. Storage only -- nothing here runs a job.

A job is working state: queued, processing, then done or failed, and of no
interest a week later. Redis suits that exactly, and it is already here for
the queue (see ``app.jobs.jobQueue``), so the table and the queue the worker reads
are one piece of infrastructure rather than two.

What is deliberately *not* here is the ``projectId`` -> ``ragDbId`` mapping.
That one is permanent and its loss orphans vectors in Pinecone with no way to
find them again, so it lives in Firestore (``app.stores.projectStore``). Losing this
table costs the status of jobs currently in flight and nothing else.

Records expire (see ``JOB_TTL_SECONDS``), which is also how job eviction
finally happens -- the Firestore table this replaces accumulated finished jobs
forever.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from redis import Redis
from redis.exceptions import WatchError

from app.jobs.job import (
    Job,
    JobDispatchError,
    JobStatus,
    Submission,
    conflictError,
    resolveSubmission,
)

logger = logging.getLogger(__name__)

KEY_PREFIX = os.environ.get("RAG_REDIS_JOB_PREFIX", "ragJob:")

# How long a job record survives its last write. Refreshed on every save, so a
# job that is still running cannot expire underneath itself -- the clock only
# starts once nothing is touching it any more. Long enough that a caller
# polling /document/status still finds a finished job; short enough that the
# table does not grow forever, which is the eviction the previous Firestore
# table never had. A project whose job has expired reads as "never submitted",
# and resubmitting starts a fresh job, which is the right answer by then.
JOB_TTL_SECONDS = int(os.environ.get("RAG_JOB_TTL_SECONDS", 7 * 24 * 60 * 60))

# How many times to retry a claim whose key was written by someone else
# mid-transaction. Contention is per-ragDbId, so this only matters when two
# submissions for one project land together; a handful of attempts is far more
# than that ever needs.
CLAIM_ATTEMPTS = 5


def toJson(job: Job) -> str:
    """The stored shape.

    ``metadata`` is deliberately absent. It is a large nested dataclass that
    only the processor holding the job in memory ever reads, and nothing that
    reads the table back -- /document/status, the worker -- looks at it.
    """
    return json.dumps(
        {
            "jobId": job.jobId,
            "serverId": job.serverId,
            "documentLink": job.documentLink,
            "status": job.status.value,
            "detail": job.detail,
            "strategy": job.strategy.value if job.strategy is not None else None,
            "createdAt": job.createdAt.isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
    )


def fromJson(raw: str | bytes) -> Job:
    # Imported here rather than at module scope: ragIngestionPipeline pulls in
    # the chunking stack, and reading a job should not require it.
    from app.ingestion.ragIngestionPipeline import ChunkingStrategy

    data = json.loads(raw)
    job = Job(
        jobId=data["jobId"],
        serverId=data["serverId"],
        documentLink=data["documentLink"],
    )
    job.status = JobStatus(data["status"])
    job.detail = data.get("detail")
    strategy = data.get("strategy")
    job.strategy = ChunkingStrategy(strategy) if strategy else None
    for field in ("createdAt", "updatedAt"):
        stamp = data.get(field)
        if stamp:
            setattr(job, field, datetime.fromisoformat(stamp))
    return job


class RedisJobStore:
    """``JobStore`` backed by one Redis key per ragDbId."""

    def __init__(self, redis: Redis, staleAfterSeconds: float | None = None) -> None:
        self._redis = redis
        # None disables reclaiming stuck jobs. See resolveSubmission for why
        # the threshold is dangerous to set without an upper bound on runtime.
        self._staleAfterSeconds = staleAfterSeconds

    def keyFor(self, ragDbId: str) -> str:
        return f"{KEY_PREFIX}{ragDbId}"

    def claim(self, *, serverId: str, documentLink: str, ragDbId: str) -> tuple[Job, bool]:
        """Claim ``ragDbId``, or return the job already holding it.

        A WATCH/MULTI transaction rather than a read followed by a write. Two
        submissions arriving at the same moment would otherwise both read an
        empty slot, both decide there was no conflict, and both ingest into one
        namespace. WATCH makes the write conditional on nothing else having
        touched the key since the read; if something did, EXEC fails and this
        starts over and sees the other's job.

        Deliberately *not* a Lua script, even though Lua would be atomic
        without the retry: the rules live in ``resolveSubmission`` and are
        shared with every other store, and a second copy of them in Lua is a
        second place for them to drift. See that function.
        """
        key = self.keyFor(ragDbId)

        with self._redis.pipeline() as pipe:
            for _ in range(CLAIM_ATTEMPTS):
                try:
                    # After watch() the pipeline runs commands immediately, so
                    # this get() returns a value rather than queueing.
                    pipe.watch(key)
                    raw = pipe.get(key)
                    existing = fromJson(raw) if raw else None

                    outcome = resolveSubmission(
                        existing,
                        serverId=serverId,
                        documentLink=documentLink,
                        staleAfterSeconds=self._staleAfterSeconds,
                    )
                    if outcome is Submission.REUSE:
                        assert existing is not None  # REUSE needs one.
                        pipe.unwatch()
                        return existing, False
                    if outcome is Submission.CONFLICT:
                        assert existing is not None
                        pipe.unwatch()
                        raise conflictError(existing, ragDbId)

                    job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
                    pipe.multi()
                    pipe.set(key, toJson(job), ex=JOB_TTL_SECONDS)
                    pipe.execute()
                    return job, True
                except WatchError:
                    logger.info("Claim on '%s' raced another submission; retrying.", ragDbId)
                    continue

        # Only reachable if the same ragDbId is being claimed continuously by
        # other processes, which for a per-project key means something is very
        # wrong. The caller advice is the same as any other dispatch failure:
        # nothing was started, resubmit this exact request.
        raise JobDispatchError(
            f"Could not claim '{ragDbId}' after {CLAIM_ATTEMPTS} attempts; "
            f"it is being submitted concurrently. Nothing was started."
        )

    def get(self, jobId: str) -> Job | None:
        raw = self._redis.get(self.keyFor(jobId))
        return fromJson(raw) if raw else None

    def save(self, job: Job) -> None:
        # The TTL is reset here, not merely preserved: a plain SET would drop
        # it and leave the record forever, and KEEPTTL would let a long job's
        # record expire while it was still running.
        self._redis.set(self.keyFor(job.jobId), toJson(job), ex=JOB_TTL_SECONDS)

    def count(self) -> int:
        """Used by tests and the live checks, not by a route.

        ``scan_iter`` rather than ``keys``: this is a shared Redis and KEYS
        blocks it for the length of the scan.
        """
        return sum(1 for _ in self._redis.scan_iter(match=f"{KEY_PREFIX}*"))
