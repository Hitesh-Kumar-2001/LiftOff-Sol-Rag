"""The submission rules, tested once, where every job manager shares them.

These used to be restated inside each manager. They are now one function, so
this is the only place the rules are pinned down -- and a manager that stops
calling it fails the tests in testCeleryJobs.py instead of silently answering
differently from the others.
"""

import pytest

from app.jobs import Job, JobStatus, Submission, resolveSubmission

LINK = "https://example.com/doc.pdf"
OTHER_LINK = "https://example.com/other.pdf"


def makeJob(status: JobStatus, link: str = LINK, serverId: str = "svc") -> Job:
    job = Job(jobId="handbook", serverId=serverId, documentLink=link)
    job.status = status
    return job


def testAnUnclaimedIdIsNew() -> None:
    assert resolveSubmission(None, serverId="svc", documentLink=LINK) is Submission.NEW


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.DONE])
def testTheSameDocumentFromTheSameCallerIsReused(status: JobStatus) -> None:
    """A retry, a duplicate delivery, an impatient caller -- all the same
    request, and ingesting it twice costs the same money for the same records."""
    outcome = resolveSubmission(makeJob(status), serverId="svc", documentLink=LINK)

    assert outcome is Submission.REUSE


@pytest.mark.parametrize("status", [JobStatus.QUEUED, JobStatus.PROCESSING])
def testADifferentDocumentMidIngestConflicts(status: JobStatus) -> None:
    """Both would write to one database under ids derived from chunk position,
    leaving it holding half of each."""
    outcome = resolveSubmission(makeJob(status), serverId="svc", documentLink=OTHER_LINK)

    assert outcome is Submission.CONFLICT


def testADifferentDocumentAfterOneFinishedIsNew() -> None:
    """Nothing is running, so there is nothing to interleave with. This is the
    ordinary way to re-ingest a database."""
    outcome = resolveSubmission(makeJob(JobStatus.DONE), serverId="svc", documentLink=OTHER_LINK)

    assert outcome is Submission.NEW


@pytest.mark.parametrize("link", [LINK, OTHER_LINK])
def testAFailedJobIsAlwaysRetriedRatherThanReused(link: str) -> None:
    outcome = resolveSubmission(makeJob(JobStatus.FAILED), serverId="svc", documentLink=link)

    assert outcome is Submission.NEW


def testAnotherCallerSubmittingTheSameLinkIsNotADuplicate() -> None:
    """Same URL, different tenant -- not the same submission, and the running
    one still owns the database."""
    existing = makeJob(JobStatus.PROCESSING, serverId="svc")

    outcome = resolveSubmission(existing, serverId="other", documentLink=LINK)

    assert outcome is Submission.CONFLICT
