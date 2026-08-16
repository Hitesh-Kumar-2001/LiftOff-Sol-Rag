"""Dispatching to a worker, and what the worker does on the other end.

No broker and no Firestore: the queue is a list that records what was sent,
and the job table is a dict implementing the same ``JobStore`` protocol
Firestore does. What is worth testing here is not that Celery can deliver a
message -- it can -- but the parts specific to this system: that the claim
happens before dispatch, that a duplicate or a conflict queues nothing, and
that a worker writes an outcome back instead of raising out of the task.
"""

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.celeryJobManager import CeleryJobManager
from app.celeryTasks import runIngestion
from app.documents import StubDocumentProcessor
from app.jobs import (
    Job,
    JobConflictError,
    JobDispatchError,
    JobStatus,
    Submission,
    conflictError,
    resolveSubmission,
)

LINK = "https://example.com/doc.pdf"
OTHER_LINK = "https://example.com/other.pdf"
RAG_DB = "handbook"


def copyJob(job: Job) -> Job:
    """A detached copy, the way a Firestore round trip produces one."""
    clone = Job(jobId=job.jobId, serverId=job.serverId, documentLink=job.documentLink)
    clone.status, clone.detail, clone.strategy = job.status, job.detail, job.strategy
    clone.createdAt, clone.updatedAt = job.createdAt, job.updatedAt
    return clone


class FakeJobStore:
    """A ``JobStore`` in a dict, sharing the real submission rules.

    Stores and returns *copies*, which is the whole point. A real table holds
    documents, not the live ``Job`` the runner is mutating, so a status only
    appears there when something writes it. Handing back the live object would
    make every status assertion here pass regardless of whether the code under
    test ever wrote anything -- which is exactly how "the table says queued for
    the entire run" went unnoticed.

    Atomicity is free here because a test is single-threaded, and that is the
    one thing this double cannot check about the Firestore store. That is what
    scripts/liveFirestoreCheck.py is for.
    """

    def __init__(self, staleAfterSeconds: float | None = None) -> None:
        self.jobs: dict[str, Job] = {}
        self.saves: list[tuple[str, str]] = []
        self.staleAfterSeconds = staleAfterSeconds

    def claim(self, *, serverId: str, documentLink: str, ragDbId: str) -> tuple[Job, bool]:
        existing = self.jobs.get(ragDbId)
        outcome = resolveSubmission(
            existing,
            serverId=serverId,
            documentLink=documentLink,
            staleAfterSeconds=self.staleAfterSeconds,
        )
        if outcome is Submission.REUSE:
            return existing, False
        if outcome is Submission.CONFLICT:
            raise conflictError(existing, ragDbId)

        job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
        self.jobs[ragDbId] = copyJob(job)
        return job, True

    def get(self, jobId: str) -> Job | None:
        job = self.jobs.get(jobId)
        return copyJob(job) if job is not None else None

    def save(self, job: Job) -> None:
        self.jobs[job.jobId] = copyJob(job)
        self.saves.append((job.jobId, job.status.value))

    def count(self) -> int:
        return len(self.jobs)


class FakeQueue:
    """Stands in for ``ingestDocument``, recording what was dispatched."""

    def __init__(self) -> None:
        self.dispatched: list[str] = []
        self.threads: list[int] = []

    def delay(self, ragDbId: str) -> None:
        self.dispatched.append(ragDbId)
        self.threads.append(threading.get_ident())


class FailingProcessor:
    async def process(self, job: Job) -> None:
        raise ValueError("boom")


class UnreachableProcessor:
    async def process(self, job: Job) -> None:
        raise AssertionError("the processor ran in the API process")


@pytest.fixture
def queue(monkeypatch) -> FakeQueue:
    """Intercept dispatch. CeleryJobManager imports the task by name, so the
    patch has to land on its module, not on app.celeryTasks."""
    fake = FakeQueue()
    monkeypatch.setattr("app.celeryJobManager.ingestDocument", fake)
    return fake


@pytest.fixture
def store() -> FakeJobStore:
    return FakeJobStore()


@pytest.fixture
def loop():
    """A loop per test, so nothing leaks into the worker's cached one."""
    made = asyncio.new_event_loop()
    yield made
    made.close()


def submit(manager, *, documentLink: str, serverId: str = "svc", ragDbId: str = RAG_DB) -> Job:
    """One submission. ``create`` is async because the Firestore claim and the
    broker publish both block, and neither may run on the event loop."""
    return asyncio.run(
        manager.create(serverId=serverId, documentLink=documentLink, ragDbId=ragDbId)
    )


def fetch(manager, jobId: str) -> Job | None:
    return asyncio.run(manager.get(jobId))


def testASubmissionIsClaimedThenDispatched(queue, store) -> None:
    manager = CeleryJobManager(UnreachableProcessor(), store)

    job = submit(manager, documentLink=LINK)

    assert job.jobId == RAG_DB
    assert store.get(RAG_DB) is not None, "the id was dispatched without being claimed"
    # Only the id travels: the worker reads the rest from the table.
    assert queue.dispatched == [RAG_DB]


def testTheApiProcessDoesNotRunTheJob(queue, store) -> None:
    """UnreachableProcessor raises if it is ever called -- the point of
    dispatching is that ingestion does not happen here."""
    manager = CeleryJobManager(UnreachableProcessor(), store)

    submit(manager, documentLink=LINK)

    assert store.get(RAG_DB).status is JobStatus.QUEUED


def testAResubmittedDocumentQueuesNothingFurther(queue, store) -> None:
    manager = CeleryJobManager(UnreachableProcessor(), store)
    first = submit(manager, documentLink=LINK)

    second = submit(manager, documentLink=LINK)

    assert second.jobId == first.jobId
    assert queue.dispatched == [RAG_DB], "a duplicate submission queued a second ingestion"


def testAConflictIsRaisedBeforeAnythingIsQueued(queue, store) -> None:
    """The claim has to happen in the API, not the worker. If the worker
    claimed, this request would have been answered 202 and the caller would
    never learn the document was refused."""
    manager = CeleryJobManager(UnreachableProcessor(), store)
    submit(manager, documentLink=LINK)

    with pytest.raises(JobConflictError):
        submit(manager, documentLink=OTHER_LINK)

    assert queue.dispatched == [RAG_DB], "a refused submission was queued anyway"


def testShutdownDoesNotWaitOnWorkItDoesNotOwn(queue, store) -> None:
    """Nothing runs in this process, so a redeploy need not wait for ingestion
    -- in-flight jobs belong to workers and are stopped with them."""
    manager = CeleryJobManager(UnreachableProcessor(), store)
    submit(manager, documentLink=LINK)

    asyncio.run(asyncio.wait_for(manager.shutdown(), timeout=1.0))


def testStatusIsReadFromTheSharedTable(queue, store) -> None:
    """The failure this replaces: a second API instance had its own empty dict,
    so /document/status answered 404 for a job that was running."""
    manager = CeleryJobManager(UnreachableProcessor(), store)
    submit(manager, documentLink=LINK)

    elsewhere = CeleryJobManager(UnreachableProcessor(), store)

    assert fetch(elsewhere, RAG_DB).documentLink == LINK
    assert len(elsewhere) == 1


def testNeitherTheClaimNorTheDispatchRunsOnTheEventLoop(queue, store) -> None:
    """Both are synchronous network calls -- a Firestore round trip and a
    broker publish. Run directly on the loop they stall every other request in
    the process, and an unreachable broker (kombu retries with backoff) hangs
    the whole API rather than failing this one request.

    Checked by thread identity because that is the observable difference; a
    timing test would only fail once the machine was slow enough.
    """
    claimed: list[int] = []
    realClaim = store.claim

    def recordingClaim(**kwargs):
        claimed.append(threading.get_ident())
        return realClaim(**kwargs)

    store.claim = recordingClaim
    manager = CeleryJobManager(UnreachableProcessor(), store)

    submit(manager, documentLink=LINK)

    here = threading.get_ident()
    assert claimed and claimed[0] != here, "the Firestore claim ran on the event loop"
    assert queue.threads and queue.threads[0] != here, "the dispatch ran on the event loop"


class DeadBroker:
    """A broker that refuses the publish, the way an unreachable Redis does
    after kombu exhausts its retry policy."""

    def delay(self, ragDbId: str) -> None:
        raise OSError("Error 10061 connecting to localhost:6379")


def testAFailedDispatchDoesNotStrandTheRagDbId(monkeypatch, store) -> None:
    """The claim is written before the dispatch, so a broker outage would
    otherwise leave the id QUEUED with nothing running and nothing coming to
    run it -- and since only a finished or failed job frees the id, every later
    submission of a different document would be refused forever."""
    monkeypatch.setattr("app.celeryJobManager.ingestDocument", DeadBroker())
    manager = CeleryJobManager(UnreachableProcessor(), store)

    with pytest.raises(JobDispatchError):
        submit(manager, documentLink=LINK)

    assert store.get(RAG_DB).status is JobStatus.FAILED
    assert "could not be queued" in store.get(RAG_DB).detail.lower()


def testADifferentDocumentIsAcceptedAfterAFailedDispatch(monkeypatch, store) -> None:
    monkeypatch.setattr("app.celeryJobManager.ingestDocument", DeadBroker())
    manager = CeleryJobManager(UnreachableProcessor(), store)
    with pytest.raises(JobDispatchError):
        submit(manager, documentLink=LINK)

    working = FakeQueue()
    monkeypatch.setattr("app.celeryJobManager.ingestDocument", working)

    submit(manager, documentLink=OTHER_LINK)

    assert working.dispatched == [RAG_DB]


def testAJobStuckPastTheTimeLimitIsReclaimed() -> None:
    """A worker killed hard, or a message the broker lost, leaves a job
    PROCESSING with nothing behind it. The table outlives the process now, so
    no restart clears it and the ragDbId would 409 forever."""
    store = FakeJobStore(staleAfterSeconds=60)
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.PROCESSING
    job.updatedAt = datetime.now(timezone.utc) - timedelta(seconds=90)
    store.save(job)

    replacement, isOurs = store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)

    assert isOurs
    assert replacement.documentLink == OTHER_LINK


def testAJobStillWithinTheTimeLimitIsNotReclaimed() -> None:
    """The dangerous direction. Reclaiming early hands a live ragDbId to a
    second ingestion -- the interleaving the conflict check exists to stop,
    arriving through the check itself."""
    store = FakeJobStore(staleAfterSeconds=60)
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.PROCESSING
    job.updatedAt = datetime.now(timezone.utc) - timedelta(seconds=30)
    store.save(job)

    with pytest.raises(JobConflictError):
        store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)


def testReclaimingIsOffUnlessAThresholdIsGiven() -> None:
    """Default None: without an upper bound on how long a job may legitimately
    run, no age proves it is dead."""
    store = FakeJobStore()
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.PROCESSING
    job.updatedAt = datetime.now(timezone.utc) - timedelta(days=7)
    store.save(job)

    with pytest.raises(JobConflictError):
        store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)


def testAStaleJobIsRerunRatherThanReturnedToItsOwnCaller() -> None:
    """Checked before the reuse rule: handing a dead job back to the caller
    polling for it leaves them waiting on something that will never finish."""
    store = FakeJobStore(staleAfterSeconds=60)
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.PROCESSING
    job.updatedAt = datetime.now(timezone.utc) - timedelta(seconds=90)
    store.save(job)

    again, isOurs = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    assert isOurs, "the same document was handed back a job that had died"
    assert again.status is JobStatus.QUEUED


# --- the worker end -------------------------------------------------------


def testTheWorkerRunsTheJobAndWritesTheOutcomeBack(store, loop) -> None:
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    job = runIngestion(RAG_DB, store, StubDocumentProcessor(), loop=loop)

    assert job.status is JobStatus.DONE
    assert store.get(RAG_DB).status is JobStatus.DONE
    # Both ends, in order. The first write is what /document/status reads while
    # the job is running -- without it the table shows 'queued' for the whole
    # ingestion and then jumps to 'done'.
    assert store.saves == [(RAG_DB, "processing"), (RAG_DB, "done")]


def testTheTableShowsProcessingWhileTheJobRuns(store) -> None:
    """The discrepancy this closes: the in-memory manager hands out the live
    Job, so its status is always current, while a table holds a copy that only
    changes when written. One API answering differently depending on which
    manager is deployed is the bug."""
    seen: list[str] = []

    class WatchingProcessor:
        async def process(self, job: Job) -> None:
            seen.append(store.get(RAG_DB).status.value)

    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    runIngestion(RAG_DB, store, WatchingProcessor())

    assert seen == ["processing"], "the shared table did not reflect a running job"


def testARedeliveredMessageDoesNotRerunAFinishedJob(store, loop) -> None:
    """Delivery is at-least-once and acks are late, so a worker killed between
    finishing and acking leaves its message to be handed out again. Re-running
    pays for the same Gemini calls and upserts twice, and puts a second writer
    into a namespace a newer ingestion may already be re-populating."""
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    runIngestion(RAG_DB, store, StubDocumentProcessor(), loop=loop)
    store.saves.clear()

    redelivered = runIngestion(RAG_DB, store, UnreachableProcessor(), loop=loop)

    assert redelivered.status is JobStatus.DONE
    assert store.saves == [], "a completed job was run and written a second time"


def testAFailedDocumentIsRecordedRatherThanRaised(store, loop) -> None:
    """A raise here would let Celery treat a document that can never succeed
    as a transient fault and retry it, and the caller would never see why."""
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    job = runIngestion(RAG_DB, store, FailingProcessor(), loop=loop)

    assert job.status is JobStatus.FAILED
    assert job.detail == "boom"
    assert store.get(RAG_DB).status is JobStatus.FAILED


def testAJobMissingFromTheTableIsNotAnError(store, loop) -> None:
    """The database was deleted between dispatch and pickup. There is nothing
    left saying what to ingest, so retrying could not help."""
    assert runIngestion("never-claimed", store, StubDocumentProcessor(), loop=loop) is None


def testLosingTheStatusWriteDoesNotFailTheTask(store, loop, caplog) -> None:
    """The ingest already finished. Failing the task would re-run all of it to
    fix a bookkeeping write."""
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    def explode(job: Job) -> None:
        raise RuntimeError("firestore is unreachable")

    store.save = explode

    job = runIngestion(RAG_DB, store, StubDocumentProcessor(), loop=loop)

    assert job.status is JobStatus.DONE
    assert "Could not record the status" in caplog.text


def testTheWorkerReusesOneLoopAcrossJobs(store) -> None:
    """Loop-affine state outlives a single task -- the chunk store is cached
    per process and holds an asyncio.Lock, which raises if a later task runs
    on a different loop. Two jobs in a row is the cheapest way to catch a
    regression to asyncio.run-per-task."""
    from app.celeryTasks import _workerLoop

    store.claim(serverId="svc", documentLink=LINK, ragDbId="first")
    store.claim(serverId="svc", documentLink=LINK, ragDbId="second")

    runIngestion("first", store, StubDocumentProcessor())
    runIngestion("second", store, StubDocumentProcessor())

    assert store.get("first").status is JobStatus.DONE
    assert store.get("second").status is JobStatus.DONE
    assert not _workerLoop().is_closed()
