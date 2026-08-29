"""The /query route. The agent itself is stubbed -- what this covers is the
route's contract: validation, what it hands the agent, and what it does when
the agent cannot run at all. The loop inside is tests/testAgent.py."""

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AgentAnswer
from app.agent.llmManager import LlmConfigError
from app.agent.reviewer import ReviewOutcome
from app.api import routes
from app.main import app
from app.stores.conversationStore import (
    ConversationStoreError,
    FirestoreConversationStore,
    getConversationStore,
)
from app.stores.projectStore import FirestoreProjectStore, getProjectStore

# The project every request in this module names. Rebound per test by the
# autouse fixture below to a scratch id, so these tests can write to real
# Firestore without two runs -- or a run and a real project -- colliding on a
# fixed name like "handbook". A module global rather than a fixture argument
# only because ``body()`` is called from every test in the file.
PROJECT_ID = "unset"


@pytest.fixture(autouse=True)
def projectId(scratch) -> Iterator[str]:
    global PROJECT_ID
    PROJECT_ID = scratch.projectId("query")
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
    """The real conversation store, overriding the lru_cached process-wide one.

    No Redis in front: a cache hit would answer a read without Firestore, and
    a test asserting what was stored would then be asserting what was
    remembered.
    """
    store = FirestoreConversationStore(redis=None)
    app.dependency_overrides[getConversationStore] = lambda: store
    return store


@pytest.fixture
def plannedSearches() -> list[dict]:
    """What the stubbed agent should pretend its search tool retrieved.

    Appended to the route's ``searchLog`` exactly as the real tool would, which
    is what lets these tests check that retrievals are stored on the conversation
    without standing up a vector store.
    """
    return []


@pytest.fixture
def calls(monkeypatch, conversationStore, plannedSearches) -> list[dict]:
    """Replace the agent with a recorder, and hand back what it was asked."""
    recorded: list[dict] = []

    async def fakeAnswer(**kwargs):
        recorded.append(kwargs)
        if kwargs.get("searchLog") is not None:
            kwargs["searchLog"].extend(plannedSearches)
        return AgentAnswer(
            answer="Thirty days from purchase.",
            reviewOutcome=ReviewOutcome(score=0.9, suggestion="", retried=False),
        )

    monkeypatch.setattr(routes, "answerQuestion", fakeAnswer)
    return recorded


@pytest.fixture
def client(calls) -> Iterator[TestClient]:
    with TestClient(app) as testClient:
        yield testClient


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "question": "What is the refund window?",
        "projectId": PROJECT_ID,
    }
    return payload | overrides


def testAQuestionGetsTheAgentsAnswer(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["projectId"] == PROJECT_ID
    assert payload["answer"] == "Thirty days from purchase."


def testAQuestionWithNoConversationIdStartsOneAndReturnsIt(
    client: TestClient, conversationStore
) -> None:
    """The only way a caller learns the id of a conversation it did not name."""
    response = client.post("/api/v1/query", json=body())

    conversationId = response.json()["conversationId"]
    assert conversationId
    assert asyncio.run(conversationStore.loadWindow(PROJECT_ID, conversationId)) is not None


def testTheFirstTurnOfANewConversationHasNoHistory(client: TestClient, calls) -> None:
    client.post("/api/v1/query", json=body())

    window = calls[0]["conversationWindow"]
    assert window.messages == []
    assert window.context == []


def testAFollowUpIsGivenTheEarlierTurns(client: TestClient, calls) -> None:
    """The point of the whole mechanism: turn two can see turn one."""
    conversationId = client.post("/api/v1/query", json=body()).json()["conversationId"]

    client.post(
        "/api/v1/query",
        json=body(question="And for gift cards?", conversationId=conversationId),
    )

    window = calls[1]["conversationWindow"]
    assert [message.content for message in window.messages] == [
        "What is the refund window?",
        "Thirty days from purchase.",
    ]


def testRetrievedPassagesAreStoredAndReplayed(
    client: TestClient, calls, plannedSearches
) -> None:
    """A follow-up starts holding what the first turn retrieved, so it does not
    have to pay for the same vector search again."""
    plannedSearches.append({"query": "refund window", "passages": ["Refunds: 30 days."]})

    conversationId = client.post("/api/v1/query", json=body()).json()["conversationId"]
    plannedSearches.clear()
    client.post(
        "/api/v1/query", json=body(question="Gift cards?", conversationId=conversationId)
    )

    context = calls[1]["conversationWindow"].context
    assert [entry.query for entry in context] == ["refund window"]
    assert context[0].passages == ["Refunds: 30 days."]


def testASearchThatFoundNothingIsStillRecorded(
    client: TestClient, calls, plannedSearches
) -> None:
    """"The documents do not cover this" is a finding, and an expensive one to
    rediscover on every follow-up."""
    plannedSearches.append({"query": "parental leave", "passages": []})

    conversationId = client.post("/api/v1/query", json=body()).json()["conversationId"]
    plannedSearches.clear()
    client.post(
        "/api/v1/query", json=body(question="Anything?", conversationId=conversationId)
    )

    assert [entry.query for entry in calls[1]["conversationWindow"].context] == ["parental leave"]


def testTheConversationKeepsThePromptItStartedWith(client: TestClient, calls) -> None:
    """A prompt edited mid-conversation must not rewrite the instructions the
    earlier answers were given under, so it is snapshotted onto the conversation."""
    from app.promptConfig import defaultSystemPrompt

    conversationId = client.post("/api/v1/query", json=body()).json()["conversationId"]
    client.post(
        "/api/v1/query", json=body(question="Again?", conversationId=conversationId)
    )

    assert calls[1]["conversationWindow"].systemPrompt == defaultSystemPrompt()


def testAnUnknownConversationIdIsA404(client: TestClient) -> None:
    """Not a new conversation. A typo silently starting a fresh conversation looks, to
    the caller, exactly like a model that has forgotten everything."""
    response = client.post("/api/v1/query", json=body(conversationId="does-not-exist"))

    assert response.status_code == 404


def testAConversationIsScopedToItsProject(client: TestClient, scratch) -> None:
    conversationId = client.post("/api/v1/query", json=body()).json()["conversationId"]

    response = client.post(
        "/api/v1/query",
        json=body(projectId=scratch.projectId("other"), conversationId=conversationId),
    )

    assert response.status_code == 404


def testAnUnreachableConversationStoreStillAnswers(
    client: TestClient, monkeypatch, conversationStore
) -> None:
    """The model call is the expensive, irreversible step. A conversation that
    could not be loaded is a reason for a worse answer, not for no answer."""

    async def unreachable(*args, **kwargs):
        raise ConversationStoreError("Firestore is down")

    monkeypatch.setattr(conversationStore, "loadWindow", unreachable)
    conversationId = "some-conversation"

    response = client.post("/api/v1/query", json=body(conversationId=conversationId))

    assert response.status_code == 200
    # The caller's own id comes back: their conversation still exists, this one turn
    # simply did not reach it.
    assert response.json()["conversationId"] == conversationId


def testAFailedWriteDoesNotLoseTheAnswer(
    client: TestClient, monkeypatch, conversationStore
) -> None:
    """The answer is already paid for. Losing it because history could not be
    written is a strictly worse outcome than losing the history."""

    async def broken(**kwargs):
        raise ConversationStoreError("write failed")

    monkeypatch.setattr(conversationStore, "appendTurn", broken)

    response = client.post("/api/v1/query", json=body())

    assert response.status_code == 200
    assert response.json()["answer"] == "Thirty days from purchase."


def testTheAgentIsGivenTheProjectsDatabase(client: TestClient, calls, projectStore) -> None:
    """Resolved here so the agent's search tool can be bound to it."""
    ragDbId = asyncio.run(projectStore.resolveOrCreate(PROJECT_ID))

    client.post("/api/v1/query", json=body())

    assert calls[0]["ragDbId"] == ragDbId
    assert calls[0]["projectId"] == PROJECT_ID
    assert calls[0]["question"] == "What is the refund window?"


def testAnUningestedProjectStillAnswers(client: TestClient, calls, scratch) -> None:
    """Asking a question does not create a database. The agent runs with no
    search tool rather than the request 404ing."""
    response = client.post(
        "/api/v1/query", json=body(projectId=scratch.projectId("never-ingested"))
    )

    assert response.status_code == 200
    assert calls[0]["ragDbId"] is None


def testQueryingDoesNotCreateADatabase(client: TestClient, projectStore, scratch) -> None:
    mistyped = scratch.projectId("mistyped")

    client.post("/api/v1/query", json=body(projectId=mistyped))

    assert asyncio.run(projectStore.resolve(mistyped)) is None


def testAnUnconfiguredAgentIsA503(monkeypatch, projectStore) -> None:
    """The request was fine; this deployment cannot reach a model. A 503 says
    'fix the configuration and retry', which a 500 does not."""

    async def unconfigured(**kwargs):
        raise LlmConfigError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(routes, "answerQuestion", unconfigured)

    with TestClient(app) as client:
        response = client.post("/api/v1/query", json=body())

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def testAProviderFailureIsA502(monkeypatch, projectStore) -> None:
    """Not a 500. The request was fine and the failure is behind the API, and
    the detail names the exception type rather than quoting the provider --
    a provider message can echo the system prompt back, and that prompt is
    another project's configuration."""

    async def broken(**kwargs):
        raise RuntimeError("upstream 529: prompt was 'You are Acme's assistant'")

    monkeypatch.setattr(routes, "answerQuestion", broken)

    with TestClient(app) as client:
        response = client.post("/api/v1/query", json=body())

    assert response.status_code == 502
    assert "RuntimeError" in response.json()["detail"]
    assert "Acme" not in response.json()["detail"]


def testAnAgentThatNeverFinishesIsA504(monkeypatch, projectStore) -> None:
    """Bounded on purpose: nothing inside the model client caps a whole answer,
    so a stalled provider would otherwise hold the connection indefinitely."""
    import asyncio

    async def hangs(**kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(routes, "answerQuestion", hangs)
    monkeypatch.setattr(routes, "ANSWER_TIMEOUT_SECONDS", 0.05)

    with TestClient(app) as client:
        response = client.post("/api/v1/query", json=body())

    assert response.status_code == 504


@pytest.mark.parametrize("field", ["serverId", "question", "projectId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "question", "projectId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/query", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(role="admin"))

    assert response.status_code == 422
