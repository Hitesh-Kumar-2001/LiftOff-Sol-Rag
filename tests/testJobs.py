"""Running one job, and the reuse/conflict rules every backend shares.

These used to go through the in-memory job manager. That manager is gone -- it
kept the table in a dict and ran ingestion on the API's event loop -- and
testing the rules through *a* backend was slightly beside the point anyway:
``resolveSubmission`` is deliberately the single copy of them (invariant 8), so
the honest place to pin them is directly, not through whichever store happens
to be calling.

The Redis backend that now does the scheduling is covered by
``tests/testRedisJobs.py``, including that its claim reaches the same verdicts
as the function below.
"""

import asyncio

import pytest

from app.ingestion.documents import StubDocumentProcessor
from app.jobs.job import Job, JobStatus, Submission, resolveSubmission, runJob

LINK = "https://example.com/handbook.pdf"
OTHER_LINK = "https://example.com/policies.pdf"
SERVER = "billing-service"


def job(**overrides) -> Job:
    fields = {"jobId": "handbook", "serverId": SERVER, "documentLink": LINK} | overrides
    return Job(**fields)


# --- running one job -------------------------------------------------------


class FailingProcessor:
    async def process(self, job: Job) -> None:
        raise ValueError("boom")


class RecordingProcessor:
    """Captures the status the job carried while it was being processed."""

    def __init__(self) -> None:
        self.statusDuringProcessing: JobStatus | None = None

    async def process(self, job: Job) -> None:
        self.statusDuringProcessing = job.status


def testASuccessfulJobEndsDone() -> None:
    subject = job()

    asyncio.run(runJob(subject, StubDocumentProcessor()))

    assert subject.status is JobStatus.DONE


def testAJobIsProcessingWhileItRuns() -> None:
    """/document/status has to be able to say 'started' rather than only
    'queued' or 'finished'."""
    processor = RecordingProcessor()

    asyncio.run(runJob(job(), processor))

    assert processor.statusDuringProcessing is JobStatus.PROCESSING


def testAFailingProcessorIsRecordedRatherThanRaised() -> None:
    """A dead link is a permanent failure, not a transient one to retry, and a
    background task that raises is one nobody is listening to."""
    subject = job()

    asyncio.run(runJob(subject, FailingProcessor()))

    assert subject.status is JobStatus.FAILED
    assert "boom" in subject.detail


def testOnStartRunsBeforeTheProcessor() -> None:
    """It publishes PROCESSING to a durable table, so it has to land before the
    work rather than after it -- otherwise the record reads 'queued' for a job
    that is minutes into running."""
    order: list[str] = []

    async def onStart(subject: Job) -> None:
        order.append("onStart")

    class Ordered:
        async def process(self, subject: Job) -> None:
            order.append("process")

    asyncio.run(runJob(job(), Ordered(), onStart=onStart))

    assert order == ["onStart", "process"]


def testAFailingOnStartLeavesTheJobFailedNotWedged() -> None:
    """``onStart`` writes to a durable table, so it is a network call that can
    fail.

    It used to sit outside ``runJob``'s try block, so that failure escaped onto
    a background task nobody awaits: the job stayed PROCESSING for the life of
    the process, /document/status answered 'processing' forever, and since the
    claim was never released every resubmission of that project answered 409.
    A failed job at least says why and can be resubmitted.
    """

    async def brokenOnStart(subject: Job) -> None:
        raise RuntimeError("job table unreachable")

    subject = job()

    # The assertion is as much that this returns at all as what it leaves behind.
    asyncio.run(runJob(subject, StubDocumentProcessor(), onStart=brokenOnStart))

    assert subject.status is JobStatus.FAILED
    assert "job table unreachable" in subject.detail


# --- the rules every backend shares ---------------------------------------


def testAnUnclaimedDatabaseIsANewJob() -> None:
    assert resolveSubmission(None, serverId=SERVER, documentLink=LINK) is Submission.NEW


def testTheSameDocumentFromTheSameCallerIsReused() -> None:
    """A retry, a duplicate delivery, or an impatient caller. Ingesting it
    twice costs the same money to produce the same records."""
    existing = job(status=JobStatus.PROCESSING)

    assert resolveSubmission(existing, serverId=SERVER, documentLink=LINK) is Submission.REUSE


def testADifferentDocumentMidIngestConflicts() -> None:
    """Both write to one database under ids derived from chunk position, so
    whichever finishes last silently overwrites part of the other."""
    existing = job(status=JobStatus.PROCESSING)

    assert (
        resolveSubmission(existing, serverId=SERVER, documentLink=OTHER_LINK)
        is Submission.CONFLICT
    )


def testADifferentDocumentAfterOneFinishedIsANewJob() -> None:
    """Re-ingesting is the ordinary way to replace a project's contents."""
    existing = job(status=JobStatus.DONE)

    assert (
        resolveSubmission(existing, serverId=SERVER, documentLink=OTHER_LINK)
        is Submission.NEW
    )


def testAFailedJobIsRetriedRatherThanReused() -> None:
    """Resubmitting is how a retry is requested, so a FAILED record must not
    hand the old failure back."""
    existing = job(status=JobStatus.FAILED, detail="dead link")

    assert resolveSubmission(existing, serverId=SERVER, documentLink=LINK) is Submission.NEW


def testAnotherCallerSubmittingTheSameLinkIsNotADuplicate() -> None:
    """Two callers naming one project is a collision they need to be told
    about, not a retry -- even though nothing verifies serverId."""
    existing = job(status=JobStatus.PROCESSING)

    assert (
        resolveSubmission(existing, serverId="other-service", documentLink=LINK)
        is Submission.CONFLICT
    )


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.PROCESSING])
def testAnUnfinishedJobHoldsItsDatabase(status: JobStatus) -> None:
    existing = job(status=status)

    assert (
        resolveSubmission(existing, serverId=SERVER, documentLink=OTHER_LINK)
        is Submission.CONFLICT
    )
