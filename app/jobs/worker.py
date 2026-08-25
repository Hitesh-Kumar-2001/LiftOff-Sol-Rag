"""The worker: take a ragDbId off the queue, ingest it, write the status back.

Run one with::

    python -m app.jobs.worker

It needs the same environment as the API -- ``REDIS_URL`` for the queue and the
job table, ``GCP_PROJECT_ID`` for the project mapping, plus the Pinecone and
Gemini keys, since this is the process that actually does the work.

One worker per deployment. The crash recovery in ``app.jobs.jobQueue`` assumes it;
see the note there before starting a second.

Stops cleanly on SIGINT/SIGTERM: the signal sets a flag, the loop notices when
its current job finishes, and the process exits. A job is never abandoned
part-way for the sake of a fast shutdown -- there is no safe halfway point in
the middle of writing a namespace. A supervisor that insists on a fast exit
will SIGKILL eventually, and the id is still on the processing list for the
next start to pick up.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from types import FrameType

from app.infra.redisClient import redisClient
from app.jobs.job import JobStatus, JobStore
from app.jobs.jobQueue import markDone, requeueAbandoned, takeNext
from app.jobs.queuedJobManager import runJobHere

logger = logging.getLogger(__name__)

_stopping = False


def requestStop(signum: int, frame: FrameType | None) -> None:
    global _stopping
    logger.info("Signal %s received; finishing the current job then stopping.", signum)
    _stopping = True


async def runOne(ragDbId: str, store: JobStore, processor) -> None:
    """Ingest the queued job for ``ragDbId``.

    Does nothing if the job is gone -- its record expired, or the table was
    cleared, between the enqueue and the pickup. Not an error worth retrying:
    there is nothing left to say what to ingest.
    """
    job = await asyncio.to_thread(store.get, ragDbId)
    if job is None:
        logger.warning("Job '%s' is no longer in the job table; nothing to run.", ragDbId)
        return

    if job.status is JobStatus.DONE:
        # A worker killed after finishing but before removing the id from the
        # processing list leaves it to be requeued on the next start. A job
        # already DONE at pickup was finished by that earlier run, and
        # re-running it would pay for the same Gemini calls and upserts twice.
        logger.info("Job '%s' is already done; skipping a requeued id.", ragDbId)
        return

    # runJob records a processor failure on the job rather than raising, so a
    # permanently bad document (dead link, unsupported type) is written back as
    # FAILED with the reason instead of throwing out of the loop and killing
    # the worker. Resubmitting a FAILED job is how a retry is requested.
    await runJobHere(job, processor, store)
    logger.info("Job '%s' finished as %s.", ragDbId, job.status.value)


async def workForever() -> None:
    from app.ingestion.ragProcessor import RagIngestionProcessor
    from app.jobs.redisJobStore import RedisJobStore

    redis = redisClient()
    if redis is None:
        sys.exit("REDIS_URL is not set; the worker has no queue to read.")

    store = RedisJobStore(redis)
    processor = RagIngestionProcessor()

    await asyncio.to_thread(requeueAbandoned, redis)
    logger.info("Worker ready.")

    while not _stopping:
        # In a thread: this blocks for POP_TIMEOUT_SECONDS, and doing that on
        # the event loop would freeze everything else the loop is running --
        # including, on shutdown, the chance to notice the flag.
        ragDbId = await asyncio.to_thread(takeNext, redis)
        if ragDbId is None:
            continue

        try:
            await runOne(ragDbId, store, processor)
        except Exception:
            # runJobHere already records processor failures on the job, so
            # reaching here means something outside the job itself broke -- a
            # Redis blip, a bug. Log it and carry on rather than taking the
            # worker down and stopping every other project's ingestion.
            logger.exception("Unhandled failure while running job '%s'.", ragDbId)
        finally:
            # Off the processing list either way. A job that failed is recorded
            # as FAILED and resubmitting is how a retry is asked for; leaving
            # the id here would have the next worker start re-run it silently.
            await asyncio.to_thread(markDone, redis, ragDbId)

    logger.info("Worker stopped.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    signal.signal(signal.SIGINT, requestStop)
    signal.signal(signal.SIGTERM, requestStop)
    asyncio.run(workForever())


if __name__ == "__main__":
    main()
