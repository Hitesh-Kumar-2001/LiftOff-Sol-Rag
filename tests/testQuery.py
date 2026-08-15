import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hashSecret
from app.main import app
from app.security import ServerRegistry, getServerRegistry

SECRET = "s3cr3t-api-key"


@pytest.fixture
def client() -> Iterator[TestClient]:
    registry = ServerRegistry(
        InMemoryCredentialSource(
            [ServerCredential(serverId="billing-service", secretHash=hashSecret(SECRET))]
        )
    )
    asyncio.run(registry.loadAll())

    app.dependency_overrides[getServerRegistry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "serverSecret": SECRET,
        "question": "What is the refund window?",
        "ragDbId": "handbook",
    }
    return payload | overrides


def testAVerifiedServerGetsAResponse(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ragDbId"] == "handbook"
    assert "What is the refund window?" in payload["answer"]


def testWrongSecretIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def testUnknownServerIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverId="nobody"))

    assert response.status_code == 401


def testUnknownServerAndWrongSecretAreIndistinguishable(client: TestClient) -> None:
    unknown = client.post("/api/v1/query", json=body(serverId="nobody"))
    wrongSecret = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert unknown.json() == wrongSecret.json()


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "question", "ragDbId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "question", "ragDbId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/query", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(role="admin"))

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "question", "ragDbId"])
def testAValidationErrorNeverEchoesTheSecret(client: TestClient, field: str) -> None:
    payload = body(serverSecret="SUPER-SECRET-KEY")
    del payload[field]

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 422
    assert "SUPER-SECRET-KEY" not in response.text


def testTheSecretIsNeverEchoedBack(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text
