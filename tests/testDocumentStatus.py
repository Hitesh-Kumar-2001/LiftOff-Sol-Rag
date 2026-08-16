import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hashSecret
from app.documents import StubDocumentProcessor
from app.jobManager import JobManager, getJobManager
from app.jobs import JobStatus
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
        "ragDbId": "handbook",
    }
    return payload | overrides


def submit(client: TestClient, ragDbId: str = "handbook") -> None:
    client.post(
        "/api/v1/document",
        json={
            "serverId": "billing-service",
            "serverSecret": SECRET,
            "documentLink": "https://example.com/handbook.pdf",
            "ragDbId": ragDbId,
        },
    )


def waitForDone(client: TestClient, ragDbId: str = "handbook", attempts: int = 50) -> dict:
    """The stub processor finishes almost immediately, but on a background
    task -- poll rather than assume it has landed."""
    for _ in range(attempts):
        payload = client.post("/api/v1/document/status", json=body(ragDbId=ragDbId)).json()
        if payload["status"] == JobStatus.DONE.value:
            return payload
    raise AssertionError(f"job never reached {JobStatus.DONE.value}")


def testAQueuedJobReportsItsStatus(client: TestClient) -> None:
    submit(client)

    response = client.post("/api/v1/document/status", json=body())

    assert response.status_code == 200
    assert response.json()["status"] in {
        JobStatus.QUEUED.value,
        JobStatus.PROCESSING.value,
        JobStatus.DONE.value,
    }


def testAFinishedJobReportsStatusAndRagDbIdOnly(client: TestClient) -> None:
    """The whole contract: two fields, nothing else."""
    submit(client)

    payload = waitForDone(client)

    assert payload == {"status": "done", "ragDbId": "handbook"}


def testTheStatusIsReportedForTheRagDbIdAsked(client: TestClient) -> None:
    submit(client, ragDbId="policies")

    payload = client.post("/api/v1/document/status", json=body(ragDbId="policies")).json()

    assert payload["ragDbId"] == "policies"


def testAnUnknownRagDbIdIsNotFound(client: TestClient) -> None:
    response = client.post("/api/v1/document/status", json=body(ragDbId="never-submitted"))

    assert response.status_code == 404


def testWrongSecretIsRejected(client: TestClient) -> None:
    submit(client)

    response = client.post("/api/v1/document/status", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def testUnknownServerIsRejected(client: TestClient) -> None:
    submit(client)

    response = client.post("/api/v1/document/status", json=body(serverId="nobody"))

    assert response.status_code == 401


def testAnUnknownRagDbIdIsIndistinguishableWithoutCredentials(client: TestClient) -> None:
    """Authentication runs first, so a bad caller cannot probe which ragDbIds
    exist by comparing 401s against 404s."""
    submit(client)

    known = client.post("/api/v1/document/status", json=body(serverSecret="wrong"))
    unknown = client.post(
        "/api/v1/document/status", json=body(ragDbId="never-submitted", serverSecret="wrong")
    )

    assert known.status_code == unknown.status_code == 401
    assert known.json() == unknown.json()


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "ragDbId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document/status", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "ragDbId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document/status", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document/status", json=body(role="admin"))

    assert response.status_code == 422


def testTheSecretIsNeverEchoedBack(client: TestClient) -> None:
    submit(client)

    response = client.post("/api/v1/document/status", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text


def testAValidationErrorNeverEchoesTheSecret(client: TestClient) -> None:
    payload = body(serverSecret="SUPER-SECRET-KEY")
    del payload["ragDbId"]

    response = client.post("/api/v1/document/status", json=payload)

    assert response.status_code == 422
    assert "SUPER-SECRET-KEY" not in response.text
