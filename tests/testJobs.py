import asyncio

import pytest

from app.documents import StubDocumentProcessor
from app.jobManager import JobManager
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

        job = manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-1"
        )

        assert job.serverId == "svc"
        assert job.documentLink == "https://example.com/doc.pdf"
        assert job.jobId
        assert manager.get(job.jobId) is job

    asyncio.run(scenario())


def testJobIdsDoNotCollide() -> None:
    async def scenario() -> None:
        manager = JobManager(StubDocumentProcessor())
        return {
            manager.create(
                serverId="svc", documentLink="https://example.com/a", ragDbId=f"job-{i}"
            ).jobId
            for i in range(50)
        }

    ids = asyncio.run(scenario())

    assert len(ids) == 50


def testAnUnknownJobIdIsNotFound() -> None:
    manager = JobManager(StubDocumentProcessor())

    assert manager.get("does-not-exist") is None


def testTheStubProcessorMarksAJobDoneInTheBackground() -> None:
    async def scenario() -> Job:
        manager = JobManager(StubDocumentProcessor())
        job = manager.create(
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
        job = manager.create(
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
        job = manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-failed"
        )
        await _waitUntil(lambda: job.status == JobStatus.FAILED)
        return job

    job = asyncio.run(scenario())

    assert job.detail == "boom"


def testShutdownCancelsJobsStillInFlight() -> None:
    async def scenario() -> None:
        processor = BlockingProcessor()
        manager = JobManager(processor)
        manager.create(
            serverId="svc", documentLink="https://example.com/doc.pdf", ragDbId="job-shutdown"
        )

        await processor.started.wait()
        await manager.shutdown()  # Should not hang waiting for `release`.

    asyncio.run(scenario())  # Timing out here is the failure mode.


def testShutdownWithNoJobsReturnsImmediately() -> None:
    asyncio.run(JobManager(StubDocumentProcessor()).shutdown())
