"""Jobs claimed in Firestore, run by a Celery worker somewhere else.

The only difference from ``FirestoreJobManager`` is what ``_start`` does, and
that is the whole point: accepting a submission -- authenticate, resolve
against whatever holds the ragDbId, claim it atomically -- is identical
whether the work then runs here or on a worker. Only the last step changes.

Why this exists: ingestion is minutes of work started by a request that
returns in milliseconds. In-process, that work owns a slice of the API's
memory and CPU, dies with any redeploy, and cannot run at all where the
process is frozen after the response. Dispatched, the API does nothing but
write a row and enqueue an id, and the workers scale on their own.

Two things follow from the split, and both are deliberate:

* the claim still happens *here*, before dispatch, not in the worker. If the
  worker claimed, the API could not answer 409 -- it would have accepted the
  submission and returned 202 before anything checked, and the caller would
  learn about the conflict never.
* the job table must be shared. An in-memory table cannot be, so this
  requires Firestore; see the factory in ``app.jobManager``.
"""

from __future__ import annotations

import asyncio
import logging

from app.celeryTasks import ingestDocument
from app.firestoreJobManager import FirestoreJobManager
from app.jobs import Job

logger = logging.getLogger(__name__)


class CeleryJobManager(FirestoreJobManager):
    """``FirestoreJobManager`` with ingestion handed to a Celery worker."""

    async def _start(self, job: Job) -> None:
        # Only the id: the worker reads the rest from the job table, which is
        # the record that stays current. See app.celeryTasks.
        #
        # In a thread because kombu's publish is synchronous. Usually a
        # millisecond or two, but against an unreachable broker it retries with
        # backoff -- on the event loop that would hang every other request in
        # the process, not just this one.
        await asyncio.to_thread(ingestDocument.delay, job.jobId)
        logger.info("Queued job '%s' for a Celery worker.", job.jobId)

    async def shutdown(self) -> None:
        """Returns immediately -- no job is running in this process.

        Not an oversight, and the main operational win of dispatching: the API
        can be redeployed or scaled down without waiting on ingestion, because
        there is nothing in flight here to wait for. In-flight jobs belong to
        workers, which are stopped on their own (``celery control shutdown``,
        or SIGTERM, both of which let a running task finish).
        """
