"""A job manager for route tests: real rules, no Redis, work in this process.

This is the old in-process ``JobManager``, moved here. It was removed from
``app.jobs`` because it was wrong as a *production* backend -- the table died
with the process and ingestion ran on the API's event loop, blocking every
other request including the health check. None of that matters to a test that
finishes in milliseconds and wants to assert what ``/document`` returns.

**The rules are not reimplemented here.** ``resolveSubmission`` is the single
copy of reuse/conflict/new (invariant 8) and this calls it, exactly as
``RedisJobStore.claim`` does, so these tests cannot drift from the deployed
behaviour. That the Redis backend wires itself to the same function correctly
is ``tests/testRedisJobs.py``'s job; what is being tested through this double is
the route -- its status codes, and what it hands the manager.

Not named ``test*.py``, so pytest imports it rather than collecting it.
"""

import asyncio

from app.jobs.job import Job, Submission, conflictError, resolveSubmission, runJob


class LocalJobManager:
    """Satisfies ``app.jobs.jobManager.JobManager`` against a dict."""

    def __init__(self, processor) -> None:
        self._processor = processor
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def __len__(self) -> int:
        return len(self._jobs)

    async def create(self, *, serverId: str, documentLink: str, ragDbId: str) -> Job:
        existing = self._jobs.get(ragDbId)
        outcome = resolveSubmission(existing, serverId=serverId, documentLink=documentLink)
        if outcome is Submission.REUSE:
            return existing
        if outcome is Submission.CONFLICT:
            raise conflictError(existing, ragDbId)

        job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
        self._jobs[job.jobId] = job

        task = asyncio.create_task(runJob(job, self._processor))
        # Held so the task cannot be garbage-collected mid-flight, and dropped
        # on completion so the set does not grow without bound.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def get(self, jobId: str) -> Job | None:
        return self._jobs.get(jobId)

    async def shutdown(self) -> None:
        """Wait for in-flight work, and cancel what will never finish.

        Tests use processors that block forever on purpose. Waiting on those
        would hang the suite, so anything still running when the client closes
        is cancelled -- the opposite of the deployed shutdown, and right here
        for the same reason the rest of this class is.
        """
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
