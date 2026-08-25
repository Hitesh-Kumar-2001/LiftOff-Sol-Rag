"""System prompts: the two-hop Firestore lookup, and the Redis cache in front.

What matters is that the hot path is one Redis GET, that a cache or store
failure degrades to a working prompt rather than a 500, and that nothing is
held in process memory -- a second instance must see an invalidation.
"""

import asyncio

import fakeredis
import pytest

from app.agent.promptStore import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_TTL_SECONDS,
    PromptStore,
)

PROJECT = "acme"
PROMPT_ID = "support-v3"
PROMPT_TEXT = "You are Acme's support assistant. Be concise."


class FakeDocument:
    def __init__(self, data: dict | None) -> None:
        self._data = data
        self.exists = data is not None

    def to_dict(self) -> dict | None:
        return self._data


class FakeReference:
    def __init__(self, collection: "FakeCollection", documentId: str) -> None:
        self._collection = collection
        self._id = documentId

    def get(self) -> FakeDocument:
        self._collection.reads.append(self._id)
        return FakeDocument(self._collection.documents.get(self._id))

    def set(self, data: dict) -> None:
        self._collection.documents[self._id] = data


class FakeCollection:
    def __init__(self, documents: dict | None = None) -> None:
        self.documents = documents or {}
        self.reads: list[str] = []

    def document(self, documentId: str) -> FakeReference:
        return FakeReference(self, documentId)


class FakeFirestore:
    """Just enough of the client to exercise the two hops, and to count them."""

    def __init__(self, assignments: dict, prompts: dict) -> None:
        self.collections = {
            "projectPrompts": FakeCollection(assignments),
            "systemPrompts": FakeCollection(prompts),
        }

    def collection(self, name: str) -> FakeCollection:
        return self.collections[name]

    @property
    def readCount(self) -> int:
        return sum(len(c.reads) for c in self.collections.values())


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def firestore() -> FakeFirestore:
    return FakeFirestore(
        assignments={PROJECT: {"promptId": PROMPT_ID}},
        prompts={PROMPT_ID: {"prompt": PROMPT_TEXT}},
    )


@pytest.fixture
def store(redis, firestore) -> PromptStore:
    return PromptStore(redis=redis, firestore=firestore)


def testAProjectResolvesToItsAssignedPrompt(store: PromptStore) -> None:
    assert asyncio.run(store.systemPromptFor(PROJECT)) == PROMPT_TEXT


def testTheResolvedPromptIsCachedInRedis(store: PromptStore, redis) -> None:
    asyncio.run(store.systemPromptFor(PROJECT))

    assert redis.get(store.cacheKey(PROJECT)) == PROMPT_TEXT


def testTheCachedPromptExpires(store: PromptStore, redis) -> None:
    """Bounded staleness is what stands in for invalidating an edited prompt."""
    asyncio.run(store.systemPromptFor(PROJECT))

    assert 0 < redis.ttl(store.cacheKey(PROJECT)) <= PROMPT_TTL_SECONDS


def testASecondQuestionCostsNoFirestoreReads(store: PromptStore, firestore) -> None:
    """The hot path is one Redis GET. Caching the *resolved* text rather than
    the two hops is what buys that."""
    asyncio.run(store.systemPromptFor(PROJECT))
    readsAfterFirst = firestore.readCount

    asyncio.run(store.systemPromptFor(PROJECT))

    assert firestore.readCount == readsAfterFirst


def testNothingIsCachedInProcessMemory(redis, firestore) -> None:
    """A second instance -- a second API process -- must see what the first
    cached, and must see an invalidation. A dict on the object would not."""
    first = PromptStore(redis=redis, firestore=firestore)
    asyncio.run(first.systemPromptFor(PROJECT))

    second = PromptStore(redis=redis, firestore=firestore)
    readsBefore = firestore.readCount
    assert asyncio.run(second.systemPromptFor(PROJECT)) == PROMPT_TEXT
    assert firestore.readCount == readsBefore, "the second instance re-read Firestore"

    asyncio.run(second.invalidate(PROJECT))
    assert redis.get(first.cacheKey(PROJECT)) is None


def testAProjectWithNoPromptGetsTheDefault(store: PromptStore) -> None:
    """Otherwise every new project's first question would fail."""
    assert asyncio.run(store.systemPromptFor("never-configured")) == DEFAULT_SYSTEM_PROMPT


def testADanglingAssignmentGetsTheDefault(redis) -> None:
    """The project points at a prompt that has been deleted. That should
    degrade to a working assistant, not a 500."""
    firestore = FakeFirestore(assignments={PROJECT: {"promptId": "gone"}}, prompts={})
    store = PromptStore(redis=redis, firestore=firestore)

    assert asyncio.run(store.systemPromptFor(PROJECT)) == DEFAULT_SYSTEM_PROMPT


class BrokenFirestore:
    def collection(self, name):
        raise ConnectionError("firestore is down")


def testAFirestoreFailureStillAnswers(redis) -> None:
    """A prompt lookup failing must not fail the question."""
    store = PromptStore(redis=redis, firestore=BrokenFirestore())

    assert asyncio.run(store.systemPromptFor(PROJECT)) == DEFAULT_SYSTEM_PROMPT


def testAFirestoreFailureIsNotCached(redis) -> None:
    """The default stands in for an unreachable store, but must not be written
    to the cache: a two-second outage would otherwise hold this project on the
    wrong prompt for a full TTL, long after Firestore came back."""
    broken = PromptStore(redis=redis, firestore=BrokenFirestore())
    assert asyncio.run(broken.systemPromptFor(PROJECT)) == DEFAULT_SYSTEM_PROMPT

    assert redis.get(broken.cacheKey(PROJECT)) is None


def testTheRightPromptIsServedAsSoonAsFirestoreReturns(redis, firestore) -> None:
    """The point of not caching the failure. The next question after an outage
    gets the project's real prompt, not the default it was answered with."""
    asyncio.run(PromptStore(redis=redis, firestore=BrokenFirestore()).systemPromptFor(PROJECT))

    recovered = PromptStore(redis=redis, firestore=firestore)

    assert asyncio.run(recovered.systemPromptFor(PROJECT)) == PROMPT_TEXT


def testAProjectWithNoPromptIsCached(store: PromptStore, redis) -> None:
    """Unlike a failure. "Nothing is assigned" is a real answer, and re-reading
    Firestore on every question to be told so again costs two reads a question
    for every project that never configures a prompt."""
    asyncio.run(store.systemPromptFor("never-configured"))

    assert redis.get(store.cacheKey("never-configured")) == DEFAULT_SYSTEM_PROMPT


def testARedisFailureFallsBackToFirestore(firestore) -> None:
    """A cache miss is the safe failure: it costs two reads, not the answer."""

    class BrokenRedis:
        def get(self, *args, **kwargs):
            raise ConnectionError("redis is down")

        def set(self, *args, **kwargs):
            raise ConnectionError("redis is down")

    store = PromptStore(redis=BrokenRedis(), firestore=firestore)

    assert asyncio.run(store.systemPromptFor(PROJECT)) == PROMPT_TEXT


def testWithNoFirestoreEveryProjectGetsTheDefault(redis) -> None:
    """Local runs without GCP_PROJECT_ID still answer."""
    store = PromptStore(redis=redis, firestore=None)

    assert asyncio.run(store.systemPromptFor(PROJECT)) == DEFAULT_SYSTEM_PROMPT


def testAssigningAPromptInvalidatesImmediately(store: PromptStore, redis, firestore) -> None:
    """Unlike editing a prompt's text, switching which prompt a project uses is
    a deliberate act -- waiting an hour to see it would read as a failed call."""
    asyncio.run(store.systemPromptFor(PROJECT))
    assert redis.get(store.cacheKey(PROJECT)) is not None

    asyncio.run(store.savePrompt("terse-v1", "Answer in one sentence."))
    asyncio.run(store.assignPrompt(PROJECT, "terse-v1"))

    assert redis.get(store.cacheKey(PROJECT)) is None
    assert asyncio.run(store.systemPromptFor(PROJECT)) == "Answer in one sentence."
