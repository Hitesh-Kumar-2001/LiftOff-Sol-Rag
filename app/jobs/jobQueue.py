"""The queue between the API and the worker. A Redis list, and little else.

This is what replaced Celery. Celery bought horizontal scaling, a result
backend, retries, routing and a scheduler; on one node none of that was used,
and it cost a broker abstraction, a second serializer, time-limit settings that
do not work on Windows, and a two-process debugging story. What was actually
wanted is here: hand a ragDbId to another process, and do not lose it if that
process dies.

**The message carries only the ragDbId.** The worker reads the rest from the
job table, which is the record that stays current. A message carrying a copy of
the job would be a second source of truth, stale the moment anything writes.

**Crash recovery is the reliable-queue pattern.** A plain ``BLPOP`` removes the
id the instant the worker takes it, so a worker killed mid-ingestion loses the
job with nothing to say it ever existed. ``BLMOVE`` instead moves the id to a
processing list in the same atomic step, and the worker removes it from there
only once the job is finished. Anything left in that list is work a dead worker
was holding, and ``requeueAbandoned`` puts it back.

That recovery assumes **one worker**. Two workers sharing a processing list
cannot tell their own in-flight job from a live one belonging to the other, and
requeueing on startup would hand a running document to a second worker -- the
interleaving the conflict check exists to prevent, arriving underneath it. One
node, one worker; if that stops being true, give each worker its own processing
list keyed by a worker id.
"""

from __future__ import annotations

import logging
import os

from redis import Redis

logger = logging.getLogger(__name__)

QUEUE_KEY = os.environ.get("RAG_REDIS_QUEUE", "ragQueue")
PROCESSING_KEY = f"{QUEUE_KEY}:processing"

# How long a worker blocks waiting for work before looping. Not a timeout on
# anything -- it just gives the loop a chance to notice a shutdown signal
# rather than sitting in a blocking call forever.
POP_TIMEOUT_SECONDS = int(os.environ.get("RAG_QUEUE_POP_TIMEOUT", 5))


def enqueue(redis: Redis, ragDbId: str) -> None:
    """Hand a claimed ragDbId to the worker."""
    redis.lpush(QUEUE_KEY, ragDbId)


def takeNext(redis: Redis, timeout: int = POP_TIMEOUT_SECONDS) -> str | None:
    """Block for the next ragDbId, moving it to the processing list.

    Returns None when the wait elapses with nothing queued, which is the
    worker's cue to check whether it has been asked to stop.
    """
    value = redis.blmove(QUEUE_KEY, PROCESSING_KEY, timeout, "RIGHT", "LEFT")
    return value.decode() if isinstance(value, bytes) else value


def markDone(redis: Redis, ragDbId: str) -> None:
    """Drop a finished id from the processing list.

    Count 1, not 0: if the same project were somehow queued twice, removing
    every copy would discard work still waiting to run.
    """
    redis.lrem(PROCESSING_KEY, 1, ragDbId)


def requeueAbandoned(redis: Redis) -> int:
    """Put back whatever a previous worker died holding. Returns how many.

    Called at worker startup. Safe to run when the list is empty, which is the
    ordinary case.
    """
    moved = 0
    while redis.rpoplpush(PROCESSING_KEY, QUEUE_KEY) is not None:
        moved += 1
    if moved:
        logger.warning("Requeued %d job(s) abandoned by a previous worker.", moved)
    return moved


def depth(redis: Redis) -> int:
    """How many ids are waiting. For the live checks and tests."""
    return redis.llen(QUEUE_KEY)
