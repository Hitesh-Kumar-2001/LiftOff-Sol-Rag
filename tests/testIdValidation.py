"""Ids that reach Firestore as document ids, and what happens to a bad one.

A `projectId` and a `conversationId` are both used verbatim as Firestore
document ids. Firestore refuses some strings -- over 1500 bytes, containing a
slash, exactly "." or "..", wrapped in double underscores -- and it refuses them
with an `InvalidArgument` raised several layers below the request. Left
unchecked, that surfaced two different ways and both were wrong:

* an unchecked `projectId` escaped the route as a **500**;
* an unchecked `conversationId` was worse. `loadWindow` wraps any Firestore
  failure as "store unreachable", the route then degrades to answering without
  history -- which is correct behaviour for a real outage -- and the caller got
  a cheerful **200** under an id that can never work. That is exactly the
  failure a 404 on an unknown id exists to prevent, arriving through the guard
  meant to prevent it. It also spent a model call per attempt.

Both now go through `checkDocumentId` before anything touches Firestore.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.agent.agent import AgentAnswer
from app.agent.reviewer import ReviewOutcome
from app.api import conversationRoutes
from app.api.schemas import ID_MAX_CHARS, checkDocumentId
from app.main import app

# Every string Firestore will not accept as a document id.
REFUSED = [
    pytest.param("", id="empty"),
    pytest.param("." * 1, id="dot"),
    pytest.param("..", id="dotdot"),
    pytest.param("a/b", id="slash"),
    pytest.param("__x__", id="doubleUnderscore"),
    pytest.param("x" * (ID_MAX_CHARS + 1), id="tooLong"),
]

ACCEPTED = ["handbook", "db-3", "a" * ID_MAX_CHARS, "_leading", "trailing_", "a.b", "a b"]


# --- the rule itself -------------------------------------------------------


@pytest.mark.parametrize("value", REFUSED)
def testFirestoreHostileIdsAreRefused(value: str) -> None:
    with pytest.raises(ValueError):
        checkDocumentId(value)


@pytest.mark.parametrize("value", ACCEPTED)
def testOrdinaryIdsAreAccepted(value: str) -> None:
    """Deliberately permissive apart from what Firestore itself rejects. A
    stricter pattern would be easy to write and would break callers whose ids
    are perfectly usable."""
    assert checkDocumentId(value) == value


def testADotIsRefusedButADottedNameIsNot() -> None:
    """Only the two reserved names, not everything containing a dot."""
    with pytest.raises(ValueError):
        checkDocumentId(".")
    assert checkDocumentId("v1.2") == "v1.2"


# --- through the routes ----------------------------------------------------


@pytest.fixture
def calls(monkeypatch) -> list[dict]:
    """Records every agent invocation, so a test can assert none happened."""
    recorded: list[dict] = []

    async def fakeAnswer(**kwargs):
        recorded.append(kwargs)
        return AgentAnswer(
            answer="Thirty days from purchase.",
            reviewOutcome=ReviewOutcome(score=0.9, suggestion="", retried=False),
        )

    monkeypatch.setattr(conversationRoutes, "answerQuestion", fakeAnswer)
    return recorded


@pytest.fixture
def client(calls) -> Iterator[TestClient]:
    with TestClient(app) as testClient:
        yield testClient


def body(**overrides) -> dict:
    return {"serverId": "billing-service", "question": "What is the refund window?"} | overrides


@pytest.mark.parametrize("projectId", ["x" * (ID_MAX_CHARS + 1), "__x__"])
def testABadProjectIdInThePathIsA422(client: TestClient, calls, projectId: str) -> None:
    """It used to be a 500: the constraint lived on the pydantic body field, and
    moving the project into the URL left it behind."""
    response = client.post(f"/api/v1/conversations/{projectId}/web", json=body())

    assert response.status_code == 422
    assert calls == []


def testABadProjectIdOnCreateIsA422(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/conversations/{'x' * (ID_MAX_CHARS + 1)}", json={"serverId": "s"}
    )

    assert response.status_code == 422


def testABadProjectIdOnAWebhookIsA422(client: TestClient) -> None:
    """Before the store is consulted, so it cannot be mistaken for a 503."""
    response = client.post(
        f"/api/v1/conversations/{'x' * (ID_MAX_CHARS + 1)}/whatsapp", content=b"{}"
    )

    assert response.status_code == 422


@pytest.mark.parametrize("conversationId", ["..", "a/b", "__x__"])
def testABadConversationIdIsA422AndCostsNothing(
    client: TestClient, calls, conversationId: str
) -> None:
    """The expensive half: it used to answer 200 from an empty conversation,
    paying for a model call, under an id that could never be read back."""
    response = client.post(
        "/api/v1/conversations/handbook/web", json=body(conversationId=conversationId)
    )

    assert response.status_code == 422
    assert calls == []


def testTheErrorNamesTheFieldWithoutEchoingTheBody(client: TestClient) -> None:
    """Pydantic attaches the whole request body to an error as ``input``. The
    web gateway validates by hand, so it has to apply the same allowlist the
    app-wide 422 handler does."""
    response = client.post(
        "/api/v1/conversations/handbook/web", json=body(conversationId="..", secret="hunter2")
    )

    detail = response.json()["detail"]
    assert response.status_code == 422
    assert any("conversationId" in str(error.get("loc", "")) for error in detail)
    assert "hunter2" not in response.text
