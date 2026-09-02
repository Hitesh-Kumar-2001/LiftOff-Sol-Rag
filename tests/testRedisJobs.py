"""The Redis job table, the queue, and the worker that drains it.

Replaces the Celery tests. What is being pinned down is the same as it always
was: that a claim is atomic, that the submission rules answer identically to
every other store, that a failed dispatch hands the ragDbId back, and that a
job crossing to another process comes back.

``fakeredis`` runs in-process, so none of this needs a server.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.jobs import jobQueue, worker
from app.jobs.job import Job, JobConflictError, JobDispatchError, JobStatus
from app.jobs.queuedJobManager import QueuedJobManager
from app.jobs.redisJobStore import JOB_TTL_SECONDS, RedisJobStore

LINK = "https://example.com/handbook.pdf"
OTHER_LINK = "https://example.com/other.pdf"
RAG_DB = "handbook-9f86d0818884"


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    # decode_responses mirrors app.infra.redisClient, so the tests see the `str`
    # everything above the client expects rather than bytes.
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(redis: fakeredis.FakeRedis) -> RedisJobStore:
    return RedisJobStore(redis)


class RecordingProcessor:
    """Marks the job done and remembers it ran."""

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def process(self, job: Job) -> None:
        self.ran.append(job.jobId)
        job.detail = "done by the test processor"


class FailingProcessor:
    async def process(self, job: Job) -> None:
        raise RuntimeError("that document is unreadable")


# --- the store -------------------------------------------------------------


def testAFirstClaimIsOurs(store: RedisJobStore) -> None:
    job, isOurs = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    assert isOurs
    assert job.jobId == RAG_DB
    assert job.status is JobStatus.QUEUED


def testTheSameDocumentIsReusedRatherThanRunTwice(store: RedisJobStore) -> None:
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    job, isOurs = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    assert not isOurs, "a duplicate submission must not start a second ingestion"
    assert job.documentLink == LINK


def testADifferentDocumentMidIngestConflicts(store: RedisJobStore) -> None:
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    with pytest.raises(JobConflictError):
        store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)


def testADifferentDocumentAfterOneFinishedIsANewJob(store: RedisJobStore) -> None:
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.DONE
    store.save(job)

    replacement, isOurs = store.claim(
        serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB
    )

    assert isOurs
    assert replacement.documentLink == OTHER_LINK


def testAJobRoundTripsThroughRedis(store: RedisJobStore) -> None:
    from app.ingestion.ragIngestionPipeline import ChunkingStrategy

    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.DONE
    job.detail = "412 chunks"
    job.strategy = ChunkingStrategy.AI
    store.save(job)

    read = store.get(RAG_DB)

    assert read.status is JobStatus.DONE
    assert read.detail == "412 chunks"
    assert read.strategy is ChunkingStrategy.AI
    assert read.serverId == "svc"


def testAnUnknownJobIsNone(store: RedisJobStore) -> None:
    assert store.get("never-submitted") is None


def testAClaimSetsAnExpiry(store: RedisJobStore, redis: fakeredis.FakeRedis) -> None:
    """Records expire, which is how finished jobs finally get evicted -- the
    Firestore table this replaced kept them forever."""
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    assert 0 < redis.ttl(store.keyFor(RAG_DB)) <= JOB_TTL_SECONDS


def testSavingRefreshesTheExpiry(store: RedisJobStore, redis: fakeredis.FakeRedis) -> None:
    """A long-running job must not have its record expire underneath it."""
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    redis.expire(store.keyFor(RAG_DB), 10)

    store.save(job)

    assert redis.ttl(store.keyFor(RAG_DB)) > 10


def backdate(redis: fakeredis.FakeRedis, store: RedisJobStore, ragDbId: str, age) -> None:
    """Age a job's record by writing it directly.

    ``save`` stamps ``updatedAt`` with the current time on purpose -- it means
    "when this was last written" -- so a job that has been sitting untouched
    for hours cannot be simulated through it.
    """
    record = json.loads(redis.get(store.keyFor(ragDbId)))
    record["updatedAt"] = (datetime.now(timezone.utc) - age).isoformat()
    redis.set(store.keyFor(ragDbId), json.dumps(record))


def testAStuckJobIsReclaimedOnceItIsPastTheThreshold(
    redis: fakeredis.FakeRedis,
) -> None:
    """Without this a worker that died holds its project at 409 until the
    record's TTL expires, which is days."""
    store = RedisJobStore(redis, staleAfterSeconds=60.0)
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    backdate(redis, store, RAG_DB, timedelta(hours=2))

    replacement, isOurs = store.claim(
        serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB
    )

    assert isOurs
    assert replacement.documentLink == OTHER_LINK


def testALiveJobIsNotReclaimedBeforeTheThreshold(redis: fakeredis.FakeRedis) -> None:
    """The dangerous direction. Reclaiming early starts a second ingestion
    beside a live one -- the interleaving the conflict check exists to stop,
    arriving through the check itself."""
    store = RedisJobStore(redis, staleAfterSeconds=3600.0)
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    with pytest.raises(JobConflictError):
        store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)


def testCountSeesOnlyJobKeys(store: RedisJobStore, redis: fakeredis.FakeRedis) -> None:
    store.claim(serverId="svc", documentLink=LINK, ragDbId="first")
    store.claim(serverId="svc", documentLink=LINK, ragDbId="second")
    redis.set("somethingElse", "not a job")

    assert store.count() == 2


# --- the queue -------------------------------------------------------------


def testAnEnqueuedIdComesBackOut(redis: fakeredis.FakeRedis) -> None:
    jobQueue.enqueue(redis, RAG_DB)

    assert jobQueue.depth(redis) == 1
    assert jobQueue.takeNext(redis, timeout=1) == RAG_DB


def testAnEmptyQueueReturnsNothing(redis: fakeredis.FakeRedis) -> None:
    """The worker's cue to check whether it has been asked to stop."""
    assert jobQueue.takeNext(redis, timeout=1) is None


def testTakingAnIdLeavesItOnTheProcessingList(redis: fakeredis.FakeRedis) -> None:
    """A plain BLPOP would drop it here, and a worker killed mid-job would take
    the only record that it was ever queued with it."""
    jobQueue.enqueue(redis, RAG_DB)

    jobQueue.takeNext(redis, timeout=1)

    assert redis.lrange(jobQueue.PROCESSING_KEY, 0, -1) == [RAG_DB]


def testFinishingClearsTheProcessingList(redis: fakeredis.FakeRedis) -> None:
    jobQueue.enqueue(redis, RAG_DB)
    jobQueue.takeNext(redis, timeout=1)

    jobQueue.markDone(redis, RAG_DB)

    assert redis.lrange(jobQueue.PROCESSING_KEY, 0, -1) == []


def testWorkAbandonedByADeadWorkerIsRequeued(redis: fakeredis.FakeRedis) -> None:
    jobQueue.enqueue(redis, RAG_DB)
    jobQueue.takeNext(redis, timeout=1)  # Worker takes it, then "dies".

    assert jobQueue.requeueAbandoned(redis) == 1
    assert jobQueue.takeNext(redis, timeout=1) == RAG_DB


def testRequeueingIsSafeWhenNothingWasAbandoned(redis: fakeredis.FakeRedis) -> None:
    assert jobQueue.requeueAbandoned(redis) == 0


def testASocketTimeoutOnTheBlockingPopReadsAsAnEmptyQueue() -> None:
    """The bug this exists to prevent killed every idle worker in seconds.

    Two clocks run during a BLMOVE: the server holding the reply for the pop
    timeout, and the client's own socket read timeout. They were both five
    seconds by default, the socket usually won, and the exception came out of
    a call the worker makes *before* its try/except -- so the process died on
    its first idle poll and ingestion only worked when a job happened to already
    be queued at startup. Nothing said so: /document/status went on answering
    'queued' forever for a job with nobody left to pick it up.
    """

    class TimingOutRedis:
        def blmove(self, *args, **kwargs):
            raise RedisTimeoutError("Timeout reading from socket")

    assert jobQueue.takeNext(TimingOutRedis(), timeout=1) is None


def testThePopTimeoutStaysUnderTheSocketTimeout() -> None:
    """The defaults must not race in the first place. The catch above keeps a
    misconfigured worker alive; this keeps the shipped one from reconnecting on
    every idle cycle to get there."""
    from app.infra.redisClient import SOCKET_TIMEOUT_SECONDS

    assert jobQueue.POP_TIMEOUT_SECONDS < SOCKET_TIMEOUT_SECONDS


# --- the manager -----------------------------------------------------------


def testCreateClaimsAndEnqueues(store: RedisJobStore, redis: fakeredis.FakeRedis) -> None:
    manager = QueuedJobManager(store, redis)

    job = asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    assert job.status is JobStatus.QUEUED
    assert jobQueue.depth(redis) == 1


def testAReusedSubmissionIsNotQueuedTwice(
    store: RedisJobStore, redis: fakeredis.FakeRedis
) -> None:
    """Otherwise a caller polling impatiently pays for the same ingestion
    twice over."""
    manager = QueuedJobManager(store, redis)
    asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    assert jobQueue.depth(redis) == 1


def testAConflictQueuesNothing(store: RedisJobStore, redis: fakeredis.FakeRedis) -> None:
    manager = QueuedJobManager(store, redis)
    asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    with pytest.raises(JobConflictError):
        asyncio.run(
            manager.create(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)
        )

    assert jobQueue.depth(redis) == 1


def testAFailedEnqueueReleasesTheClaim(
    store: RedisJobStore, redis: fakeredis.FakeRedis, monkeypatch
) -> None:
    """The claim is written before the enqueue, so a failure between the two
    would otherwise hold the project at QUEUED with nothing coming to run it --
    a 409 on every later submission until the record's TTL expired."""

    def unreachable(*args, **kwargs):
        raise ConnectionError("redis is down")

    monkeypatch.setattr("app.jobs.queuedJobManager.enqueue", unreachable)
    manager = QueuedJobManager(store, redis)

    with pytest.raises(JobDispatchError):
        asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    assert store.get(RAG_DB).status is JobStatus.FAILED
    # FAILED is never reused, so the next submission is a fresh job.
    _, isOurs = store.claim(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)
    assert isOurs


def testShutdownDoesNotWaitOnTheWorker(
    store: RedisJobStore, redis: fakeredis.FakeRedis
) -> None:
    """The operational win of dispatching: the API redeploys without waiting
    for ingestion, because nothing is running in it."""
    manager = QueuedJobManager(store, redis)
    asyncio.run(manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB))

    asyncio.run(asyncio.wait_for(manager.shutdown(), timeout=1))


# --- the worker ------------------------------------------------------------


def testTheWorkerRunsAQueuedJobAndRecordsIt(store: RedisJobStore) -> None:
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    processor = RecordingProcessor()

    asyncio.run(worker.runOne(RAG_DB, store, processor))

    assert processor.ran == [RAG_DB]
    assert store.get(RAG_DB).status is JobStatus.DONE


def testTheWorkerRecordsAFailureRatherThanRaising(store: RedisJobStore) -> None:
    """A dead link is a permanent failure, not a transient one. Raising here
    would take the worker down and stop every other project's ingestion."""
    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    asyncio.run(worker.runOne(RAG_DB, store, FailingProcessor()))

    job = store.get(RAG_DB)
    assert job.status is JobStatus.FAILED
    assert "unreadable" in job.detail


def testTheWorkerSkipsAJobThatIsAlreadyDone(store: RedisJobStore) -> None:
    """A worker killed after finishing but before clearing the processing list
    leaves the id to be requeued. Re-running it would pay for the same Gemini
    calls and Pinecone upserts twice."""
    job, _ = store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
    job.status = JobStatus.DONE
    store.save(job)
    processor = RecordingProcessor()

    asyncio.run(worker.runOne(RAG_DB, store, processor))

    assert processor.ran == []


def testTheWorkerToleratesAJobThatHasGone(store: RedisJobStore) -> None:
    """Its record expired between the enqueue and the pickup. There is nothing
    left to say what to ingest, and nothing worth retrying."""
    processor = RecordingProcessor()

    asyncio.run(worker.runOne("never-submitted", store, processor))

    assert processor.ran == []


def testTheWorkerPublishesProcessingBeforeItFinishes(store: RedisJobStore) -> None:
    """/document/status must not report 'queued' for a job minutes into
    running, which is what happens if the status is only written at the end."""
    seen: list[str] = []

    class Watching:
        async def process(self, job: Job) -> None:
            seen.append(store.get(job.jobId).status.value)

    store.claim(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)

    asyncio.run(worker.runOne(RAG_DB, store, Watching()))

    assert seen == [JobStatus.PROCESSING.value]


def testAQueueReadFailureDoesNotKillTheWorker(monkeypatch, redis) -> None:
    """The guard that was missing, and the reason a dead worker is so expensive.

    The worker's ``except`` covers *running* a job. Reading the queue sat
    outside any handler, so anything the pop raised ended the process -- which
    is how an idle worker died of a five-second socket timeout. ``takeNext``
    now answers None for that specific case; this covers the rest of the family
    (a Redis restart, a failover, a dropped connection) by driving the real
    loop with a pop that raises.

    Silence is what makes it costly: the API goes on accepting documents and
    ``/document/status`` goes on answering ``queued``, forever, for work with
    nobody left to pick it up.
    """
    calls: list[int] = []

    def failThenStop(_redis, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("redis went away")
        # Survived the first failure and came back for more, which is the
        # whole claim. Stop the loop so the test terminates.
        monkeypatch.setattr(worker, "_stopping", True)
        return None

    monkeypatch.setattr(worker, "takeNext", failThenStop)
    monkeypatch.setattr(worker, "requeueAbandoned", lambda _redis: 0)
    monkeypatch.setattr(worker, "redisClient", lambda: redis)
    monkeypatch.setattr(worker, "QUEUE_RETRY_SECONDS", 0)
    monkeypatch.setattr(worker, "_stopping", False)

    class _Processor:
        pass

    monkeypatch.setattr(
        "app.ingestion.ragProcessor.RagIngestionProcessor", lambda *a, **k: _Processor()
    )

    asyncio.run(asyncio.wait_for(worker.workForever(), timeout=5))

    assert len(calls) == 2, "the worker did not come back after a failed queue read"
