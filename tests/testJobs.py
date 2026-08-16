import asyncio

import pytest

from app.documents import StubDocumentProcessor
from app.jobManager import JobConflictError, JobManager
from app.jobs import Job, JobStatus


class BlockingProcessor:
    """Never finishes on its own -- lets a test control exactly when it does."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def process(self, job: Job) -> None:
        self.started.set()
        await self.release.wait()


class FailingProcessor:
    async def process(self, job: Job) -> None:
        raise ValueError("boom")


async def _waitUntil(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


def testANewJobIsQueuedAndReturnedImmediately() -> None:
    async def scenario() -> None:
        manager = JobManager(StubDocumentProcessor())

        job = await manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-1"
        )

        assert job.serverId == "svc"
        assert job.documentLink == "https://example.com/doc.pdf"
        assert job.jobId
        assert await manager.get(job.jobId) is job

    asyncio.run(scenario())


def testJobIdsDoNotCollide() -> None:
    async def scenario() -> None:
        manager = JobManager(StubDocumentProcessor())
        return {
            (
                await manager.create(
                    serverId="svc", documentLink="https://example.com/a", ragDbId=f"job-{i}"
                )
            ).jobId
            for i in range(50)
        }

    ids = asyncio.run(scenario())

    assert len(ids) == 50


def testAnUnknownJobIdIsNotFound() -> None:
    async def scenario() -> None:
        manager = JobManager(StubDocumentProcessor())

        assert await manager.get("does-not-exist") is None

    asyncio.run(scenario())


def testTheStubProcessorMarksAJobDoneInTheBackground() -> None:
    async def scenario() -> Job:
        manager = JobManager(StubDocumentProcessor())
        job = await manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-done"
        )
        await _waitUntil(lambda: job.status == JobStatus.DONE)
        return job

    job = asyncio.run(scenario())

    assert job.detail == "Processing is not implemented yet."


def testAJobIsProcessingBeforeItCompletes() -> None:
    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        job = await manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-processing"
        )

        await processor.started.wait()
        assert job.status == JobStatus.PROCESSING

        processor.release.set()
        await _waitUntil(lambda: job.status == JobStatus.DONE)

    asyncio.run(scenario())


def testAFailingProcessorMarksTheJobFailed() -> None:
    async def scenario() -> Job:
        manager = JobManager(FailingProcessor())
        job = await manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-failed"
        )
        await _waitUntil(lambda: job.status == JobStatus.FAILED)
        return job

    job = asyncio.run(scenario())

    assert job.detail == "boom"


def testShutdownWaitsForJobsStillInFlight() -> None:
    """Shutdown is deliberately not a cancel: a job mid-download or mid-write
    has no safe halfway point, so it is allowed to finish."""

    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        job = await manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-shutdown"
        )

        await processor.started.wait()

        shutdown = asyncio.create_task(manager.shutdown())
        await asyncio.sleep(0)  # Let it run far enough to finish early if it would.
        assert not shutdown.done(), "shutdown returned while a job was still running"

        processor.release.set()
        await asyncio.wait_for(shutdown, timeout=1.0)
        assert job.status == JobStatus.DONE

    asyncio.run(scenario())


LINK = "https://example.com/doc.pdf"
OTHER_LINK = "https://example.com/other.pdf"


def testResubmittingTheSameDocumentReusesItsJob() -> None:
    """A retry or a duplicate delivery should not pay to ingest the same
    document twice into the same database."""

    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)

        first = await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await processor.started.wait()
        second = await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")

        assert second is first
        assert len(manager._tasks) == 1, "a second ingestion was started"

        processor.release.set()
        await _waitUntil(lambda: first.status == JobStatus.DONE)

    asyncio.run(scenario())


def testADifferentDocumentIsRefusedWhileOneIsStillIngesting() -> None:
    """Both would write to one database under ids derived from chunk
    position, leaving it holding part of each document."""

    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await processor.started.wait()

        with pytest.raises(JobConflictError):
            await manager.create(serverId="svc", documentLink=OTHER_LINK, ragDbId="handbook")

        processor.release.set()

    asyncio.run(scenario())


def testADifferentDocumentReplacesAFinishedOne() -> None:
    """Re-ingesting a database once nothing is running is ordinary."""

    async def scenario() -> None:
        manager = JobManager(StubDocumentProcessor())
        first = await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await _waitUntil(lambda: first.status == JobStatus.DONE)

        second = await manager.create(serverId="svc", documentLink=OTHER_LINK, ragDbId="handbook")

        assert second is not first
        assert await manager.get("handbook") is second

    asyncio.run(scenario())


def testAFailedJobIsRetriedRatherThanReused() -> None:
    async def scenario() -> None:
        manager = JobManager(FailingProcessor())
        first = await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await _waitUntil(lambda: first.status == JobStatus.FAILED)

        second = await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")

        assert second is not first

    asyncio.run(scenario())


def testAnotherServerSubmittingTheSameLinkIsNotTreatedAsADuplicate() -> None:
    """Same link, different caller -- the match has to be on both."""

    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await processor.started.wait()

        with pytest.raises(JobConflictError):
            await manager.create(serverId="other", documentLink=LINK, ragDbId="handbook")

        processor.release.set()

    asyncio.run(scenario())


def testDifferentDatabasesNeverConflict() -> None:
    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        await manager.create(serverId="svc", documentLink=LINK, ragDbId="handbook")
        await processor.started.wait()

        await manager.create(serverId="svc", documentLink=OTHER_LINK, ragDbId="policies")

        assert len(manager) == 2
        processor.release.set()

    asyncio.run(scenario())


def testShutdownWithNoJobsReturnsImmediately() -> None:
    asyncio.run(JobManager(StubDocumentProcessor()).shutdown())
