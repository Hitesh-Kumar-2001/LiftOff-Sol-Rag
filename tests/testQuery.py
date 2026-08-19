from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Nothing to override: /query has no dependencies left. It authenticates
    # nothing, and retrieval is not wired up yet.
    yield TestClient(app)


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "question": "What is the refund window?",
        "projectId": "handbook",
    }
    return payload | overrides


def testAQuestionGetsAResponse(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["projectId"] == "handbook"
    assert "What is the refund window?" in payload["answer"]


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