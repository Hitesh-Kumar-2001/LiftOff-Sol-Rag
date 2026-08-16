"""The job table, in Firestore. Storage only -- nothing here runs a job.

``JobManager`` holds jobs in a dict on one process, which is fine for a single
node and wrong everywhere else: the jobs vanish on restart, and a second
instance cannot see them at all -- so ``/document/status`` answers "no such
job" for jobs that are running, and the conflict check misses concurrent
submissions because each instance reads its own empty table.

This is the same table in a Firestore collection, so every instance reads one
table and a restart loses nothing. It is deliberately separate from the
managers because there is now more than one consumer: the API accepting a
submission, and -- when ingestion is dispatched to Celery -- a worker in
another process that has to read the job the API wrote.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

from app.jobs import Job, JobStatus, Submission, conflictError, resolveSubmission

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_JOBS_COLLECTION", "ragJobs")

# Where the service account key lives. Unset in a managed GCP environment,
# where application default credentials are supplied by the platform.
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Which database in the project. A project's first database is usually the
# special one literally called "(default)", and the client assumes it -- but a
# project can hold several, and one *named* "default" is a different database
# from "(default)". The failure is confusing enough to be worth naming: the
# client reports 'The database (default) does not exist for project X' while
# the console plainly shows a database sitting there.
DATABASE_ID = os.environ.get("FIRESTORE_DATABASE_ID") or None


_initLock = threading.Lock()


def firestoreClient() -> firestore.Client:
    """The shared Firestore client, initialised once per process.

    Locked because the check and the initialisation are separate steps, and
    ``initialize_app`` raises if the default app already exists. Claims run in
    a thread pool (see ``FirestoreJobManager.create``), so two of them arriving
    together is an ordinary occurrence rather than a theoretical one.
    """
    if not firebase_admin._apps:
        with _initLock:
            if not firebase_admin._apps:
                # An explicit key file when one is named, otherwise the
                # credentials the platform provides -- which is how this runs
                # on Cloud Run without a key file on disk at all.
                cred = credentials.Certificate(CREDENTIALS_PATH) if CREDENTIALS_PATH else None
                firebase_admin.initialize_app(cred)
    return firestore.client(database_id=DATABASE_ID)


def toDocument(job: Job) -> dict:
    return {
        "jobId": job.jobId,
        "serverId": job.serverId,
        "documentLink": job.documentLink,
        "status": job.status.value,
        "detail": job.detail,
        "strategy": job.strategy.value if job.strategy is not None else None,
        "createdAt": job.createdAt.isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def fromDocument(data: dict) -> Job:
    # Imported here rather than at module scope: ragIngestionPipeline pulls in
    # the chunking stack, and reading a job should not require it.
    from app.ragIngestionPipeline import ChunkingStrategy

    job = Job(
        jobId=data["jobId"],
        serverId=data["serverId"],
        documentLink=data["documentLink"],
    )
    job.status = JobStatus(data["status"])
    job.detail = data.get("detail")
    strategy = data.get("strategy")
    job.strategy = ChunkingStrategy(strategy) if strategy else None
    for field in ("createdAt", "updatedAt"):
        stamp = data.get(field)
        if stamp:
            setattr(job, field, datetime.fromisoformat(stamp))
    return job


class FirestoreJobStore:
    """``JobStore`` backed by a Firestore collection keyed by ragDbId."""

    def __init__(self, collection=None, staleAfterSeconds: float | None = None) -> None:
        self._db = firestoreClient()
        self._collection = collection or self._db.collection(COLLECTION)
        # None disables reclaiming stuck jobs. See resolveSubmission for why
        # the threshold is dangerous to set without an upper bound on runtime.
        self._staleAfterSeconds = staleAfterSeconds

    def claim(self, *, serverId: str, documentLink: str, ragDbId: str) -> tuple[Job, bool]:
        """Claim ``ragDbId``, or return the job already holding it.

        The claim is a Firestore transaction rather than a read followed by a
        write. Two instances receiving submissions at the same moment would
        otherwise both read an empty slot, both decide there was no conflict,
        and both ingest into one namespace. Inside a transaction the second
        one sees the first's write and refuses.
        """
        reference = self._collection.document(ragDbId)

        @firestore.transactional
        def claimInTransaction(transaction) -> tuple[Job, bool]:
            snapshot = reference.get(transaction=transaction)
            existing = fromDocument(snapshot.to_dict()) if snapshot.exists else None

            outcome = resolveSubmission(
                existing,
                serverId=serverId,
                documentLink=documentLink,
                staleAfterSeconds=self._staleAfterSeconds,
            )
            if outcome is Submission.REUSE:
                assert existing is not None  # REUSE is only reachable with one.
                return existing, False
            if outcome is Submission.CONFLICT:
                assert existing is not None
                raise conflictError(existing, ragDbId)

            job = Job(jobId=ragDbId, serverId=serverId, documentLink=documentLink)
            transaction.set(reference, toDocument(job))
            return job, True

        return claimInTransaction(self._db.transaction())

    def get(self, jobId: str) -> Job | None:
        snapshot = self._collection.document(jobId).get()
        return fromDocument(snapshot.to_dict()) if snapshot.exists else None

    def save(self, job: Job) -> None:
        self._collection.document(job.jobId).set(toDocument(job))

    def count(self) -> int:
        return sum(1 for _ in self._collection.list_documents())
