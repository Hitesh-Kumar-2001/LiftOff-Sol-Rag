"""Check FirestoreJobManager against real Firestore.

    GOOGLE_APPLICATION_CREDENTIALS=keys/<key>.json \
    GCP_PROJECT_ID=<project> python scripts/liveFirestoreCheck.py

Covers what a stand-in cannot: that jobs round-trip through Firestore, that a
second instance sees the first's jobs, and that the conflict check holds when
two submissions race -- which is the whole reason for moving the table out of
process memory.

Cleans up the documents it creates.
"""

from __future__ import annotations

import asyncio
import os
import sys

from app.jobs import Job, JobConflictError, JobStatus

RAG_DB = "live-firestore-check"
LINK = "https://example.com/handbook.pdf"
OTHER_LINK = "https://example.com/other.pdf"

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"  {'ok ' if ok else 'FAIL'}  {label}: {actual!r}" + ("" if ok else f" != {expected!r}"))
    if not ok:
        failures.append(label)


class SlowProcessor:
    """Holds a job in PROCESSING until released, so a race can be staged."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def process(self, job: Job) -> None:
        # Nothing is downloaded or ingested: this check is about the job table,
        # and a real document would only add Pinecone and Gemini to what could
        # fail. runJob ignores the return value and reads the job's status.
        self.started.set()
        await self.release.wait()


async def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("Set GCP_PROJECT_ID (and GOOGLE_APPLICATION_CREDENTIALS).")

    from app.firestoreJobManager import FirestoreJobManager
    from app.firestoreJobStore import COLLECTION, firestoreClient

    processor = SlowProcessor()
    manager = FirestoreJobManager(processor)
    collection = firestoreClient().collection(COLLECTION)
    collection.document(RAG_DB).delete()

    try:
        print(f"collection: {COLLECTION}\n")

        print("a submitted job is written to Firestore")
        job = await manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
        await processor.started.wait()
        check("job id", job.jobId, RAG_DB)
        stored = collection.document(RAG_DB).get().to_dict()
        check("stored in Firestore", stored is not None, True)
        check("status recorded", stored["status"], "processing")
        check("link recorded", stored["documentLink"], LINK)

        print("\na second instance, with its own empty memory, sees the job")
        # The failure this replaces: a fresh JobManager knows nothing, so
        # /document/status answered "no such job" for a running one.
        second = FirestoreJobManager(processor)
        seen = await second.get(RAG_DB)
        check("found by another instance", seen is not None, True)
        check("same document", seen.documentLink if seen else None, LINK)

        print("\nthe same document again is deduplicated, not re-run")
        again = await manager.create(serverId="svc", documentLink=LINK, ragDbId=RAG_DB)
        check("same job returned", again.jobId, RAG_DB)

        print("\na different document mid-ingest is refused across instances")
        try:
            await second.create(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)
            check("conflict raised", False, True)
        except JobConflictError:
            check("conflict raised", True, True)

        print("\nthe final status is written back when the job finishes")
        processor.release.set()
        await manager.shutdown()
        stored = collection.document(RAG_DB).get().to_dict()
        check("final status", stored["status"], JobStatus.DONE.value)

        print("\nand a later instance reads that finished job")
        check("status seen by another instance", (await second.get(RAG_DB)).status, JobStatus.DONE)
    finally:
        collection.document(RAG_DB).delete()
        print(f"\ncleaned up '{RAG_DB}'")

    print("\n" + ("FAILURES: " + ", ".join(failures) if failures else "all checks passed"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
