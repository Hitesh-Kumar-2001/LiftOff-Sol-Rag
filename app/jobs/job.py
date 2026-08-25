"""A job's data, and the mechanics of running one.

A ``Job`` is just a record -- id, status, timestamps. ``runJob`` is the one
function that knows how to execute it: hand it to a processor, translate the
outcome into a status. Deciding when a job runs, keeping it reachable by id,
and returning it to the API is ``app.jobs.jobManager.JobManager``'s job, not this
module's -- this module only knows how to run one, given one.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.ingestion.documents import DocumentMetadata, DocumentProcessor
    from app.ingestion.ragIngestionPipeline import ChunkingStrategy

logger = logging.getLogger(__name__)


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobConflictError(Exception):
    """A different document is already being ingested into this ragDbId.

    Lives here rather than beside either job manager so that both raise the
    same class. routes.py catches exactly one, and a second definition
    elsewhere would slip past that handler and surface as a 500 instead of
    the 409 it is.
    """


class JobDispatchError(Exception):
    """The job was claimed but could not be handed to anything that runs it.

    Distinct from a conflict because it says something different to the
    caller: nothing is wrong with the request, the infrastructure behind it is
    down, and resubmitting the identical request later is the right response.
    That is a 503, not a 409 and not a 500.
    """


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    # Deliberately the ragDbId being ingested into, not a generated id -- a
    # job exists to populate one RAG database, so there is nothing else
    # meaningful to key it by. See JobManager.create.
    jobId: str
    serverId: str
    documentLink: str
    status: JobStatus = JobStatus.QUEUED
    detail: str | None = None
    # Set by DocumentAnalyzerProcessor once the download is analyzed. None
    # until then, and for processors (like the stub) that don't produce it.
    metadata: "DocumentMetadata | None" = None
    # How the document was chunked, once a processor has chosen. None until
    # then. RAW means the document was small enough to keep whole, so there
    # is no vector database to query and callers are pointed back at
    # documentLink instead -- see the /document/status route.
    strategy: "ChunkingStrategy | None" = None
    createdAt: datetime = field(default_factory=_now)
    updatedAt: datetime = field(default_factory=_now)


class Submission(enum.StrEnum):
    """What a submission means, given whatever job already holds that ragDbId."""

    NEW = "new"
    REUSE = "reuse"
    CONFLICT = "conflict"


def resolveSubmission(
    existing: Job | None,
    *,
    serverId: str,
    documentLink: str,
    staleAfterSeconds: float | None = None,
) -> Submission:
    """Decide what to do with a submission for a ragDbId ``existing`` holds.

    One function rather than a copy inside each job store. The in-memory
    manager and the Redis store both call it, and they have to answer
    identically or the same request gets accepted on one deployment and refused
    on another. A second copy of the rules -- including a clever one in Lua
    inside the Redis claim -- is a second place for them to drift, and the drift
    would only show up in production.

    The rules, and why:

    * the same document from the same caller is REUSE -- a retry, a duplicate
      delivery, an impatient caller. Ingesting it twice costs the same money
      to produce the same records.
    * a *different* document while one is still running is a CONFLICT. Both
      write to one database under ids derived from chunk position, so
      whichever finishes last silently overwrites part of the other and
      leaves the database holding half of each. Cancelling the first has no
      safe halfway point to stop at either.
    * anything else is NEW: a different document once nothing is running is
      the ordinary way to re-ingest, and a FAILED job is never reused, so
      resubmitting after a failure retries rather than returning the failure.

    The caller match is on ``serverId`` as well as the link -- two tenants
    submitting the same public URL are not the same submission.

    ``staleAfterSeconds`` is the escape hatch for a job that will never finish.
    Once the table outlives the process, a job left QUEUED or PROCESSING by a
    worker that died -- or by a message the broker lost -- blocks its ragDbId
    with a 409 forever, where an in-memory table would have cleared it on the
    next restart. Past that age the job is presumed dead and the id can be
    claimed again.

    **The threshold must exceed the longest a job could still legitimately be
    running.** Set it too low and this hands a live ragDbId to a second
    ingestion, which is the exact interleaving the CONFLICT case exists to
    prevent -- arriving through the check itself. It is therefore None (off) by
    default, and only worth setting where something guarantees an upper bound
    on runtime. Nothing here can hard-kill CPU-bound work -- a worker can be
    SIGKILLed by a supervisor, but the application cannot stop a job mid-parse
    -- so the threshold is a judgement about the longest a document could
    legitimately take. See ``buildJobManager``.
    """
    if existing is None or existing.status is JobStatus.FAILED:
        return Submission.NEW

    running = existing.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
    if running and staleAfterSeconds is not None:
        age = (_now() - existing.updatedAt).total_seconds()
        if age > staleAfterSeconds:
            # Checked before REUSE as well: returning a dead job to a caller
            # polling for it would leave them waiting on something that is
            # never going to finish.
            logger.warning(
                "Job '%s' has been %s for %.0fs (limit %.0fs); presuming its worker "
                "died and reclaiming the id.",
                existing.jobId, existing.status.value, age, staleAfterSeconds,
            )
            return Submission.NEW

    if (existing.documentLink, existing.serverId) == (documentLink, serverId):
        return Submission.REUSE
    if running:
        return Submission.CONFLICT
    return Submission.NEW


def conflictError(existing: Job, ragDbId: str) -> JobConflictError:
    """The 409 a CONFLICT turns into, worded the same wherever it is raised."""
    return JobConflictError(
        f"'{existing.documentLink}' is already being ingested into "
        f"'{ragDbId}'. Wait for it to finish before submitting another."
    )


class JobStore(Protocol):
    """The job table, with no opinion about who runs the jobs.

    Split out because the work no longer necessarily runs where the job was
    accepted: the worker picks a job up in another process entirely, and needs
    the same table the API wrote to. Storage that knows nothing about the
    execution model is what the API and the worker can share.
    """

    def claim(self, *, serverId: str, documentLink: str, ragDbId: str) -> tuple[Job, bool]:
        """Resolve a submission and, if it is NEW, record the job.

        Returns the job and whether this caller is the one that must start
        the work -- False when an existing job was reused, so a duplicate
        submission cannot start a second ingestion. Raises
        ``JobConflictError`` on a CONFLICT.

        Must be atomic. A read followed by a write lets two instances both
        see an empty slot, both decide there is no conflict, and both ingest
        into one namespace -- the exact interleaving the conflict check
        exists to stop.
        """
        ...

    def get(self, jobId: str) -> Job | None: ...

    def save(self, job: Job) -> None:
        """Write a job's current state back, overwriting what is stored."""
        ...

    def count(self) -> int: ...


async def runJob(job: Job, processor: "DocumentProcessor", onStart=None) -> None:
    """Execute ``job`` against ``processor``, updating its status as it goes.

    A processor failure is caught and recorded on the job rather than raised,
    so a caller scheduling this as a background task never sees an unhandled
    exception surface from that task.

    ``onStart`` is awaited once the job is PROCESSING but before any work
    begins. It exists for the managers whose table is a *copy* of the job
    rather than the job itself: the in-memory manager hands out the same
    object this mutates, so its status is live, but a Firestore document only
    changes when something writes to it. Without this hook that document sits
    at QUEUED for the whole ingestion and jumps straight to DONE, so
    /document/status reports 'queued' for a job that is minutes into running
    -- and answers differently depending on which manager is deployed.
    """
    job.status = JobStatus.PROCESSING
    job.updatedAt = _now()
    if onStart is not None:
        await onStart(job)
    try:
        await processor.process(job)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Job %s failed.", job.jobId)
        job.status = JobStatus.FAILED
        job.detail = str(exc)
    else:
        job.status = JobStatus.DONE
    job.updatedAt = _now()
