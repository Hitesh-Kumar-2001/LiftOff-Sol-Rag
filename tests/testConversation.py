"""The two conversation endpoints: create one, then post questions to it.

    POST /api/v1/conversation          -> conversationId
    POST /api/v1/conversation/message  -> answer

``/query`` is the one-call form of the same thing and is covered in
tests/testQuery.py; all three run the same ``_runTurn``, so what is tested here
is what the pair adds -- an id that exists before any question, and an endpoint
that requires one.

The last section covers the read path the two-call form depends on: **Redis
first, Firestore only on a miss.** That is the reason a follow-up is cheap, and
it is invisible in the response, so it is asserted by counting what each store
was actually asked for.

Real Firestore, like the rest of the suite. The agent is stubbed wherever a
question is asked, because none of this is about what the model said.
"""

import asyncio
from collections.abc import Iterator

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AgentAnswer
from app.agent.reviewer import ReviewOutcome
from app.api import routes
from app.main import app
from app.promptConfig import defaultSystemPrompt
from app.stores.conversationStore import FirestoreConversationStore, getConversationStore
from app.stores.projectStore import FirestoreProjectStore, getProjectStore

PROJECT_ID = "unset"


@pytest.fixture(autouse=True)
def projectId(scratch) -> Iterator[str]:
    global PROJECT_ID
    PROJECT_ID = scratch.projectId("conversation")
    yield PROJECT_ID
    PROJECT_ID = "unset"


@pytest.fixture
def projectStore(projectId) -> Iterator[FirestoreProjectStore]:
    store = FirestoreProjectStore()
    app.dependency_overrides[getProjectStore] = lambda: store
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def conversationStore(projectStore) -> FirestoreConversationStore:
    """The real store with no Redis in front of it: a cache hit would answer a
    read without Firestore, and a test asserting what was stored would then be
    asserting what was remembered. The cache gets its own fixture below."""
    store = FirestoreConversationStore(redis=None)
    app.dependency_overrides[getConversationStore] = lambda: store
    return store


@pytest.fixture
def client(conversationStore) -> Iterator[TestClient]:
    with TestClient(app) as testClient:
        yield testClient


@pytest.fixture
def stubbedAgent(monkeypatch) -> list[dict]:
    """Replace the agent with a recorder, and hand back what it was asked."""
    recorded: list[dict] = []

    async def fakeAnswer(**kwargs):
        recorded.append(kwargs)
        return AgentAnswer(
            answer="Thirty days from purchase.",
            reviewOutcome=ReviewOutcome(score=0.9, suggestion="", retried=False),
        )

    monkeypatch.setattr(routes, "answerQuestion", fakeAnswer)
    return recorded


def body(**overrides) -> dict:
    return {"serverId": "billing-service", "projectId": PROJECT_ID} | overrides


def startConversation(client: TestClient, **overrides) -> str:
    return client.post("/api/v1/conversation", json=body(**overrides)).json()[
        "conversationId"
    ]


def ask(client: TestClient, conversationId: str, question: str):
    return client.post(
        "/api/v1/conversation/message",
        json=body(conversationId=conversationId, question=question),
    )


# --- POST /api/v1/conversation --------------------------------------------


def testCreatingAConversationReturnsAnIdThatResolves(client: TestClient, conversationStore) -> None:
    """The whole point of the endpoint: an id, before any question exists."""
    response = client.post("/api/v1/conversation", json=body())

    assert response.status_code == 201
    payload = response.json()
    assert payload["projectId"] == PROJECT_ID
    assert payload["conversationId"]
    assert (
        asyncio.run(conversationStore.loadWindow(PROJECT_ID, payload["conversationId"])) is not None
    )


def testANewConversationIsEmpty(client: TestClient, conversationStore) -> None:
    window = asyncio.run(conversationStore.loadWindow(PROJECT_ID, startConversation(client)))

    assert window.messages == []
    assert window.context == []
    assert window.turnCount == 0


def testTheProjectsPromptIsSnapshottedOntoIt(client: TestClient, conversationStore) -> None:
    """It answers under the instructions it was opened with, even if the
    project's prompt is edited later -- so the prompt is decided here, and
    returned so the caller can see which one it got."""
    payload = client.post("/api/v1/conversation", json=body()).json()

    assert payload["systemPrompt"] == defaultSystemPrompt()
    window = asyncio.run(conversationStore.loadWindow(PROJECT_ID, payload["conversationId"]))
    assert window.systemPrompt == defaultSystemPrompt()


def testEachCallStartsADifferentConversation(client: TestClient) -> None:
    assert startConversation(client) != startConversation(client)


def testStartingOneDoesNotCreateARagDatabase(client: TestClient, projectStore) -> None:
    """Only /document may mint a mapping. Otherwise every mistyped projectId
    that ever opened a conversation would leave an empty database behind."""
    startConversation(client)

    assert asyncio.run(projectStore.resolve(PROJECT_ID)) is None


def testAGivenTitleIsStored(client: TestClient, conversationStore) -> None:
    conversationId = startConversation(client, title="Refunds")

    stored = conversationStore._conversationRef(PROJECT_ID, conversationId).get().to_dict()

    assert stored["title"] == "Refunds"


def testAnOmittedTitleIsFilledByTheFirstQuestion(
    client: TestClient, conversationStore, stubbedAgent
) -> None:
    """A conversation list wants a line per conversation. One opened from a "new
    conversation" button has no name yet, so the first thing asked in it becomes one."""
    conversationId = startConversation(client)

    ask(client, conversationId, "What is the refund window?")

    stored = conversationStore._conversationRef(PROJECT_ID, conversationId).get().to_dict()
    assert stored["title"] == "What is the refund window?"


def testAnUnreachableStoreIsA503OnCreate(
    client: TestClient, conversationStore, monkeypatch
) -> None:
    """Unlike the answering routes, which degrade: this request *is* the
    creation, so there is nothing to hand back but the thing that did not
    happen. A 201 carrying an id nothing stored would be a conversation the
    caller can address and every later request will 404."""

    async def unreachable(**kwargs):
        raise RuntimeError("firestore is down")

    monkeypatch.setattr(conversationStore, "createConversation", unreachable)

    assert client.post("/api/v1/conversation", json=body()).status_code == 503


def testAnUnknownFieldIsRejected(client: TestClient) -> None:
    """extra="forbid", like every other request model here: a caller sending
    `conversationID` and getting a 201 for a field that was ignored is worse than a 422."""
    assert client.post("/api/v1/conversation", json=body(ragDbId="nope")).status_code == 422


# --- POST /api/v1/conversation/message ------------------------------------


def testAQuestionIsAnsweredInTheConversation(
    client: TestClient, conversationStore, stubbedAgent
) -> None:
    """The two halves have to meet: an id from /conversation is an id
    /conversation/message accepts, and the turn lands on that conversation."""
    conversationId = startConversation(client)

    response = ask(client, conversationId, "What is the refund window?")

    assert response.status_code == 200
    assert response.json()["answer"] == "Thirty days from purchase."
    assert response.json()["conversationId"] == conversationId
    window = asyncio.run(conversationStore.loadWindow(PROJECT_ID, conversationId))
    assert [message.content for message in window.messages] == [
        "What is the refund window?",
        "Thirty days from purchase.",
    ]


def testAFollowUpSeesTheEarlierTurns(client: TestClient, stubbedAgent) -> None:
    """The point of the whole mechanism: turn two can see turn one."""
    conversationId = startConversation(client)
    ask(client, conversationId, "What is the refund window?")

    ask(client, conversationId, "And for gift cards?")

    assert [m.content for m in stubbedAgent[1]["conversationWindow"].messages] == [
        "What is the refund window?",
        "Thirty days from purchase.",
    ]


def testAConversationIdIsRequired(client: TestClient) -> None:
    """The one difference from /query. A caller with nothing to continue should
    be sent to /conversation or /query, not quietly given a new conversation
    under an endpoint whose name says it is addressing an existing one."""
    response = client.post(
        "/api/v1/conversation/message", json=body(question="What is the refund window?")
    )

    assert response.status_code == 422


def testAnUnknownConversationIsA404(client: TestClient, stubbedAgent) -> None:
    """Not a new conversation. A typo silently starting a fresh one looks, to
    the caller, exactly like a model that has forgotten everything."""
    assert ask(client, "does-not-exist", "Anything?").status_code == 404


def testAConversationIsScopedToItsProject(
    client: TestClient, scratch, stubbedAgent
) -> None:
    conversationId = startConversation(client)

    response = client.post(
        "/api/v1/conversation/message",
        json=body(
            projectId=scratch.projectId("other"),
            conversationId=conversationId,
            question="Anything?",
        ),
    )

    assert response.status_code == 404


def testThePromptIsNotReResolvedPerTurn(client: TestClient, stubbedAgent) -> None:
    """A prompt edited mid-conversation must not rewrite the instructions the
    earlier answers were given under, so the snapshot is what the agent gets."""
    conversationId = startConversation(client)

    ask(client, conversationId, "What is the refund window?")

    assert stubbedAgent[0]["conversationWindow"].systemPrompt == defaultSystemPrompt()


# --- the read path: Redis first, Firestore on a miss ----------------------


@pytest.fixture
def cachedStore(projectStore) -> tuple[FirestoreConversationStore, fakeredis.FakeRedis]:
    """A store with a real cache in front, and the Redis it is using.

    fakeredis rather than a server: what is being asserted is the *order* the
    two stores are consulted in, which does not need a network.
    """
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = FirestoreConversationStore(redis=redis)
    app.dependency_overrides[getConversationStore] = lambda: store
    return store, redis


@pytest.fixture
def cachedClient(cachedStore) -> Iterator[TestClient]:
    with TestClient(app) as testClient:
        yield testClient


def testTheWindowIsCachedInRedis(cachedClient: TestClient, cachedStore, stubbedAgent) -> None:
    """Written on the way past, so the next question does not re-read Firestore."""
    store, redis = cachedStore
    conversationId = startConversation(cachedClient)

    ask(cachedClient, conversationId, "What is the refund window?")

    assert redis.get(store.cacheKey(PROJECT_ID, conversationId)) is not None


def testASecondQuestionIsAnsweredFromRedis(
    cachedClient: TestClient, cachedStore, stubbedAgent, monkeypatch
) -> None:
    """The claim this whole layer rests on. Firestore is not touched to load the
    window on a cache hit -- one GET instead of a document read plus two range
    queries over the messages and context subcollections."""
    store, redis = cachedStore
    conversationId = startConversation(cachedClient)
    ask(cachedClient, conversationId, "What is the refund window?")

    firestoreReads = []
    original = store._loadFromStore
    monkeypatch.setattr(
        store,
        "_loadFromStore",
        lambda *a, **k: (firestoreReads.append(a), original(*a, **k))[1],
    )

    ask(cachedClient, conversationId, "And for gift cards?")

    assert firestoreReads == []
    # And it was a real window, not an empty one: the follow-up still saw turn one.
    assert len(stubbedAgent[1]["conversationWindow"].messages) == 2


def testAMissFallsBackToFirestore(
    cachedClient: TestClient, cachedStore, stubbedAgent
) -> None:
    """Redis holds nothing Firestore does not, so an evicted or expired entry
    costs reads and never a conversation."""
    store, redis = cachedStore
    conversationId = startConversation(cachedClient)
    ask(cachedClient, conversationId, "What is the refund window?")

    redis.flushall()

    response = ask(cachedClient, conversationId, "And for gift cards?")

    assert response.status_code == 200
    assert len(stubbedAgent[1]["conversationWindow"].messages) == 2
    # Re-cached on the way through, so the miss is paid once and not per turn.
    assert redis.get(store.cacheKey(PROJECT_ID, conversationId)) is not None


def testABrokenCacheStillAnswers(
    cachedClient: TestClient, cachedStore, stubbedAgent, monkeypatch
) -> None:
    """A cache failure costs the reads it was there to save, and nothing else."""
    store, _ = cachedStore
    conversationId = startConversation(cachedClient)
    ask(cachedClient, conversationId, "What is the refund window?")

    def broken(*args, **kwargs):
        raise RuntimeError("redis is down")

    monkeypatch.setattr(store._redis, "get", broken)
    monkeypatch.setattr(store._redis, "set", broken)

    response = ask(cachedClient, conversationId, "And for gift cards?")

    assert response.status_code == 200
    assert len(stubbedAgent[1]["conversationWindow"].messages) == 2
