import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hashSecret
from app.documents import StubDocumentProcessor
from app.jobManager import JobManager, getJobManager
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

    # One manager per test, not one per request -- a fresh instance per call
    # would make jobs vanish between the create and any later lookup.
    jobManager = JobManager(StubDocumentProcessor())

    app.dependency_overrides[getServerRegistry] = lambda: registry
    app.dependency_overrides[getJobManager] = lambda: jobManager
    with TestClient(app) as testClient:
        yield testClient
    app.dependency_overrides.clear()


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "serverSecret": SECRET,
        "documentLink": "https://example.com/handbook.pdf",
    }
    return payload | overrides


def testAVerifiedServerGetsAJobId(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body())

    assert response.status_code == 202
    payload = response.json()
    assert payload["jobId"]
    assert payload["status"] == "queued"


def testEachSubmissionGetsADistinctJobId(client: TestClient) -> None:
    first = client.post("/api/v1/document", json=body()).json()
    second = client.post("/api/v1/document", json=body()).json()

    assert first["jobId"] != second["jobId"]


def testWrongSecretIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def testUnknownServerIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverId="nobody"))

    assert response.status_code == 401


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(role="admin"))

    assert response.status_code == 422


def testTheSecretIsNeverEchoedBack(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text


def testARejectedSubmissionCreatesNoJob(client: TestClient) -> None:
    manager = app.dependency_overrides[getJobManager]()  # Same instance every call.
    before = len(manager)

    client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert len(manager) == before
