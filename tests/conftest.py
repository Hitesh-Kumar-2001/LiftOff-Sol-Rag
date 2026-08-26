"""The suite runs against real Firestore. This is what keeps that safe.

There are no in-process stores any more: ``buildProjectStore`` and
``buildChatStore`` return Firestore or raise. That is a deliberate trade. A dict
with a lock agrees with itself by construction, so the two things most worth
testing -- the transaction in ``resolveOrCreate`` and the one in ``appendTurn``,
both of which exist solely to survive concurrency -- were never actually
exercised by a stand-in. Now they are.

The cost is that every test writes to a real database, which brings two hazards
this module exists to remove:

**Collisions.** Fixed ids like ``"handbook"`` would be shared by every run on
every machine, so two developers, or one developer and CI, would delete each
other's documents mid-test. Every id here is prefixed with a per-run token, so
concurrent runs cannot see each other's data.

**Leakage.** Firestore keeps whatever a crashed test left behind, forever.
``scratch`` deletes every document it handed out an id for, including the
subcollections Firestore will not remove with their parent, and it does so in
fixture teardown so a failing assertion still cleans up.

Quota, for reference: the free tier allows 20k writes and 50k reads a day. A
full run is on the order of a few hundred operations, so the ceiling is dozens
of runs a day rather than a handful -- but it is a real ceiling, and a runaway
loop in a test would find it.
"""

import os
import uuid

import pytest

import app  # noqa: F401  -- runs load_dotenv() before anything reads config.

# Distinguishes this run's documents from every other run's, everywhere.
RUN_ID = uuid.uuid4().hex[:10]


def pytest_configure(config) -> None:
    """Fail the whole session early, with an explanation, rather than letting
    every Firestore-touching test fail one at a time with a RuntimeError."""
    if not os.environ.get("GCP_PROJECT_ID"):
        raise pytest.UsageError(
            "GCP_PROJECT_ID is not set. The suite runs against real Firestore -- "
            "there are no in-process stores. Set GCP_PROJECT_ID and "
            "GOOGLE_APPLICATION_CREDENTIALS in .env (see docs/chatSchema.md)."
        )


class FirestoreScratch:
    """Unique project ids, and the cleanup that follows them.

    Ids are handed out rather than chosen by the test so that nothing can
    accidentally name a real project. Everything handed out is deleted on
    teardown -- there is no opt-out, because the failure mode of forgetting is
    silent accumulation in a database nobody is watching.
    """

    def __init__(self) -> None:
        self.projectIds: list[str] = []
        self._byName: dict[str, str] = {}

    def projectId(self, name: str = "project") -> str:
        """A fresh id every call."""
        projectId = f"test-{RUN_ID}-{name}-{len(self.projectIds)}"
        self.projectIds.append(projectId)
        return projectId

    def named(self, name: str) -> str:
        """The *same* id every call within one test, for a given name.

        What the route-level tests need: several requests in one test all
        naming "handbook" have to reach one project, while "handbook" in the
        next test must be a project that has never existed. Minting per call
        would break the first; a fixed literal would break the second.
        """
        if name not in self._byName:
            self._byName[name] = self.projectId(name)
        return self._byName[name]

    def cleanup(self) -> None:
        from app.infra.firestoreClient import firestoreClient
        from app.stores.chatStore import COLLECTION as CHATS
        from app.stores.projectStore import COLLECTION as PROJECTS

        db = firestoreClient()
        for projectId in self.projectIds:
            # Best effort, per document: one failure must not strand the rest,
            # and a test that never created a given document is the normal case
            # rather than an error.
            try:
                db.collection(PROJECTS).document(projectId).delete()
            except Exception:
                pass

            try:
                chats = db.collection(CHATS).document(projectId).collection("chats")
                for chat in chats.stream():
                    # Firestore does not delete subcollections with their
                    # parent; a chat deleted without these leaves orphaned
                    # messages that nothing will ever read or remove.
                    for name in ("messages", "context"):
                        for document in chat.reference.collection(name).stream():
                            document.reference.delete()
                    chat.reference.delete()
                db.collection(CHATS).document(projectId).delete()
            except Exception:
                pass


@pytest.fixture(scope="session", autouse=True)
def fakeRedisEverywhere():
    """Give the whole suite a Redis that needs no server.

    The in-process job manager is gone, so ``buildJobManager`` requires
    REDIS_URL and the API refuses to start without it -- which is right for a
    deployment and would otherwise mean every test needed a running Redis.
    ``fakeredis`` is a full implementation in the process, including the
    WATCH/MULTI the claim depends on, so the *real* ``QueuedJobManager`` and
    ``RedisJobStore`` are what these tests exercise. Only the server is fake.

    Patched at the source and at every module that bound the name at import
    time: ``redisClient`` is imported directly by ``app.jobs.jobManager``, so
    replacing it only in ``app.infra`` would leave that copy pointing at the
    original.
    """
    import fakeredis

    from app.infra import redisClient as redisModule
    from app.jobs import jobManager as jobManagerModule

    shared = fakeredis.FakeRedis(decode_responses=True)
    patch = pytest.MonkeyPatch()
    patch.setenv("REDIS_URL", "redis://fake-for-tests")
    patch.setattr(redisModule, "redisClient", lambda: shared)
    patch.setattr(jobManagerModule, "redisClient", lambda: shared)

    # Anything built before the patch landed would be holding the real client.
    jobManagerModule.getJobManager.cache_clear()
    try:
        yield shared
    finally:
        patch.undo()
        jobManagerModule.getJobManager.cache_clear()


@pytest.fixture
def scratch():
    """Per-test scratch ids, cleaned up however the test ends."""
    workspace = FirestoreScratch()
    try:
        yield workspace
    finally:
        workspace.cleanup()


@pytest.fixture
def projects(scratch):
    """A real ``FirestoreProjectStore``."""
    from app.stores.projectStore import FirestoreProjectStore

    return FirestoreProjectStore()


@pytest.fixture
def chats(scratch):
    """A real ``FirestoreChatStore``, with no Redis in front of it.

    Deliberately uncached: a cache hit would answer reads without Firestore,
    and then a test asserting what was stored would be asserting what was
    remembered. ``tests/testChatStore.py`` exercises the cache explicitly, with
    a fake Redis, where that is the point.
    """
    from app.stores.chatStore import FirestoreChatStore

    return FirestoreChatStore(redis=None)
