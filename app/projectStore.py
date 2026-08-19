"""Where a caller's ``projectId`` becomes the ``ragDbId`` everything else uses.

A caller names a *project* and nothing else. The ``ragDbId`` behind it is
internal: it is the job's id, the Pinecone namespace, and the thing the
conflict check claims -- and none of that appears on the wire. Keeping the two
apart is the whole point of this module. A project is a caller's permanent name
for "my documents"; a ragDbId is where those documents happen to live right
now, and that is a decision this codebase should stay free to change.
Rebuilding a project into a fresh namespace, versioning it, or one day fanning
it across several databases are all changes to this mapping alone, invisible to
whoever is calling.

They are one-to-one today. Nothing outside this module may assume that stays
true, and nothing anywhere may recompute a ragDbId from a projectId -- which is
why the id below is random rather than derived.

The mapping is **write-once**. Resolving a project has to return the same
ragDbId every time, forever: hand back a different one and the chunks already
ingested under the old id are still sitting in Pinecone, still costing money,
and unreachable by any request. Nothing here updates or deletes a mapping, and
any future code that does has to move the chunks along with it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_PROJECTS_COLLECTION", "ragProjects")

# Everything a Firestore document id and a Pinecone namespace both tolerate.
# Pinecone namespaces are hashed anyway (see pineconeChunkStore.namespaceFor),
# so this is about the readable prefix, not about correctness.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def newRagDbId(projectId: str) -> str:
    """Mint a fresh ragDbId for ``projectId``.

    The readable prefix is for whoever ends up reading Firestore or a job table
    by eye -- the same trade ``app.localChunkStore.fileNameFor`` makes. The
    random suffix is what makes this an *id* rather than a transformation:
    a derived id is one nothing can ever change, and the day a project needs a
    second database, or a rebuild into a clean namespace, that assumption is
    exactly what breaks. Callers resolve; they never compute.
    """
    return f"{_UNSAFE.sub('_', projectId)[:64]}-{uuid.uuid4().hex[:12]}"


class ProjectStore(Protocol):
    """The projectId -> ragDbId mapping, however it is stored.

    Two methods because the two callers want different things. ``/document``
    may create -- a project's first submission is what brings its database into
    existence. Everything else resolves read-only: a project nobody has ever
    ingested into has no database, and minting one on a search would leave an
    empty namespace behind for every typo'd projectId that ever arrived.
    """

    async def resolve(self, projectId: str) -> str | None:
        """The ragDbId for ``projectId``, or None if it has never had one."""
        ...

    async def resolveOrCreate(self, projectId: str) -> str:
        """The ragDbId for ``projectId``, minting one if this is its first.

        Must be atomic. Two first submissions for one project arriving together
        must not mint two ragDbIds -- the loser's ingestion would populate a
        namespace nothing will ever resolve to again.
        """
        ...


class InMemoryProjectStore:
    """The mapping in a dict. Single process, and gone on restart."""

    def __init__(self) -> None:
        self._byProjectId: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, projectId: str) -> str | None:
        return self._byProjectId.get(projectId)

    async def resolveOrCreate(self, projectId: str) -> str:
        existing = self._byProjectId.get(projectId)
        if existing is not None:
            return existing

        async with self._lock:
            # Re-checked inside the lock: two submissions for a new project can
            # both miss above, and the first through will have written it by
            # the time the second acquires. Without this they mint two ids and
            # one document lands in a namespace nothing resolves to.
            if projectId not in self._byProjectId:
                self._byProjectId[projectId] = newRagDbId(projectId)
            return self._byProjectId[projectId]

    def __len__(self) -> int:
        return len(self._byProjectId)


class FirestoreProjectStore:
    """The mapping in a Firestore collection keyed by ``projectId``.

    Shares the client and the credential handling of
    ``app.firestoreJobStore`` -- one project, one Firestore setup, one place
    that knows how to reach it.
    """

    def __init__(self, collection=None) -> None:
        # Imported here rather than at module scope so that importing this
        # module -- which routes.py does unconditionally -- neither requires
        # Firestore credentials nor opens a client. Same reasoning as the lazy
        # imports in app.jobManager.
        from app.firestoreJobStore import firestoreClient

        self._db = firestoreClient()
        self._collection = collection or self._db.collection(COLLECTION)

    async def resolve(self, projectId: str) -> str | None:
        return await asyncio.to_thread(self._resolve, projectId)

    def _resolve(self, projectId: str) -> str | None:
        snapshot = self._collection.document(projectId).get()
        return snapshot.to_dict()["ragDbId"] if snapshot.exists else None

    async def resolveOrCreate(self, projectId: str) -> str:
        return await asyncio.to_thread(self._resolveOrCreate, projectId)

    def _resolveOrCreate(self, projectId: str) -> str:
        """A transaction, not a read followed by a write.

        Two instances taking a project's first submission at the same moment
        would otherwise both read an empty slot and both mint an id, and one of
        the two ingestions would populate a namespace no later request can
        resolve to. Inside a transaction the second one sees the first's write
        and returns it. Same reasoning, and same shape, as
        ``FirestoreJobStore.claim``.
        """
        from firebase_admin import firestore

        reference = self._collection.document(projectId)

        @firestore.transactional
        def createInTransaction(transaction) -> str:
            snapshot = reference.get(transaction=transaction)
            if snapshot.exists:
                return snapshot.to_dict()["ragDbId"]

            ragDbId = newRagDbId(projectId)
            transaction.set(
                reference,
                {
                    "projectId": projectId,
                    "ragDbId": ragDbId,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                },
            )
            return ragDbId

        return createInTransaction(self._db.transaction())


def buildProjectStore() -> ProjectStore:
    """Pick the mapping store from the environment.

    Firestore when a GCP project is named, matching how the job table is
    chosen in ``app.jobManager`` -- the two are the same durability decision
    and should not be able to disagree.
    """
    if os.environ.get("GCP_PROJECT_ID"):
        return FirestoreProjectStore()

    # Worth a warning rather than a quiet default. This mapping is the only
    # record of where a project's chunks live: lose it on a restart and the
    # next submission for that projectId mints a *new* ragDbId, leaving the
    # previously ingested vectors in Pinecone, billable, and unreachable. That
    # is fine for tests and local runs (where RAG_TEST_MODE usually means there
    # are no real vectors at all) and wrong for anything else.
    logger.warning(
        "No GCP_PROJECT_ID: the projectId -> ragDbId mapping is in memory and will "
        "not survive a restart. Previously ingested chunks become unreachable if it "
        "is lost."
    )
    return InMemoryProjectStore()


@lru_cache(maxsize=1)
def getProjectStore() -> ProjectStore:
    """FastAPI dependency: the process-wide project store.

    One instance for the life of the process, same as ``getChunkStore``. With
    the in-memory implementation this is what makes a mapping outlive the
    request that created it at all.
    """
    return buildProjectStore()
