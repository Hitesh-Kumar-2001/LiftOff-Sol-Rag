import asyncio

import pytest

from app.jobs import Job, JobStatus, JobStore, StubDocumentProcessor


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


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


def test_a_new_job_is_queued_and_returned_immediately() -> None:
    async def scenario() -> None:
        store = JobStore(StubDocumentProcessor())

        job = store.create(server_id="svc", document_link="https://example.com/doc.pdf")

        assert job.server_id == "svc"
        assert job.document_link == "https://example.com/doc.pdf"
        assert job.job_id
        assert store.get(job.job_id) is job

    asyncio.run(scenario())


def test_job_ids_do_not_collide() -> None:
    async def scenario() -> None:
        store = JobStore(StubDocumentProcessor())
        return {store.create("svc", "https://example.com/a").job_id for _ in range(50)}

    ids = asyncio.run(scenario())

    assert len(ids) == 50


def test_an_unknown_job_id_is_not_found() -> None:
    store = JobStore(StubDocumentProcessor())

    assert store.get("does-not-exist") is None


def test_the_stub_processor_marks_a_job_done_in_the_background() -> None:
    async def scenario() -> Job:
        store = JobStore(StubDocumentProcessor())
        job = store.create("svc", "https://example.com/doc.pdf")
        await _wait_until(lambda: job.status == JobStatus.DONE)
        return job

    job = asyncio.run(scenario())

    assert job.detail == "Processing is not implemented yet."


def test_a_job_is_processing_before_it_completes() -> None:
    async def scenario() -> None:
        processor = BlockingProcessor()
        store = JobStore(processor)
        job = store.create("svc", "https://example.com/doc.pdf")

        await processor.started.wait()
        assert job.status == JobStatus.PROCESSING

        processor.release.set()
        await _wait_until(lambda: job.status == JobStatus.DONE)

    asyncio.run(scenario())


def test_a_failing_processor_marks_the_job_failed() -> None:
    async def scenario() -> Job:
        store = JobStore(FailingProcessor())
        job = store.create("svc", "https://example.com/doc.pdf")
        await _wait_until(lambda: job.status == JobStatus.FAILED)
        return job

    job = asyncio.run(scenario())

    assert job.detail == "boom"


def test_shutdown_cancels_jobs_still_in_flight() -> None:
    async def scenario() -> None:
        processor = BlockingProcessor()
        store = JobStore(processor)
        store.create("svc", "https://example.com/doc.pdf")

        await processor.started.wait()
        await store.shutdown()  # Should not hang waiting for `release`.

    asyncio.run(scenario())  # Timing out here is the failure mode.


def test_shutdown_with_no_jobs_returns_immediately() -> None:
    asyncio.run(JobStore(StubDocumentProcessor()).shutdown())
