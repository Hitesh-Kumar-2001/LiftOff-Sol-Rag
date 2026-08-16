import asyncio
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hashSecret
from app.documents import StubDocumentProcessor
from app.jobManager import JobManager, getJobManager
from app.jobs import Job
from app.main import app
from app.security import ServerRegistry, getServerRegistry

SECRET = "s3cr3t-api-key"


class BlockingProcessor:
    """Never finishes, so a job stays mid-ingest for as long as a test needs."""

    async def process(self, job: Job) -> None:
        await asyncio.Event().wait()


@contextmanager
def clientUsing(processor) -> Generator[TestClient]:
    registry = ServerRegistry(
        InMemoryCredentialSource(
            [ServerCredential(serverId="billing-service", secretHash=hashSecret(SECRET))]
        )
    )
    asyncio.run(registry.loadAll())

    # One manager per test, not one per request -- a fresh instance per call
    # would make jobs vanish between the create and any later lookup.
    jobManager = JobManager(processor)

    app.dependency_overrides[getServerRegistry] = lambda: registry
    app.dependency_overrides[getJobManager] = lambda: jobManager
    try:
        with TestClient(app) as testClient:
            yield testClient
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with clientUsing(StubDocumentProcessor()) as testClient:
        yield testClient


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "serverSecret": SECRET,
        "documentLink": "https://example.com/handbook.pdf",
        "ragDbId": "handbook",
    }
    return payload | overrides


def testAVerifiedServerGetsAJobId(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body())

    assert response.status_code == 202
    payload = response.json()
    assert payload["jobId"]
    assert payload["status"] == "queued"


def testEachRagDbIdGetsItsOwnJobId(client: TestClient) -> None:
    first = client.post("/api/v1/document", json=body(ragDbId="handbook")).json()
    second = client.post("/api/v1/document", json=body(ragDbId="policies")).json()

    assert first["jobId"] != second["jobId"]


def testResubmittingARagDbIdReusesItsJobId(client: TestClient) -> None:
    """A job's id *is* its ragDbId, so a second document for the same database
    lands on the same job rather than creating a competing one."""
    first = client.post("/api/v1/document", json=body(ragDbId="handbook")).json()
    second = client.post("/api/v1/document", json=body(ragDbId="handbook")).json()

    assert first["jobId"] == second["jobId"] == "handbook"


def testResubmittingTheSameDocumentIsAccepted(client: TestClient) -> None:
    """The stub finishes immediately, so this is the settled case: the same
    document again is a duplicate, not a conflict."""
    first = client.post("/api/v1/document", json=body(ragDbId="handbook"))
    second = client.post("/api/v1/document", json=body(ragDbId="handbook"))

    assert first.status_code == second.status_code == 202


def testADifferentDocumentMidIngestIsAConflict() -> None:
    """409 rather than 202: nothing was queued, because running it would
    leave the database holding part of each document."""
    with clientUsing(BlockingProcessor()) as client:
        client.post("/api/v1/document", json=body(ragDbId="handbook"))

        response = client.post(
            "/api/v1/document",
            json=body(ragDbId="handbook", documentLink="https://example.com/other.pdf"),
        )

        assert response.status_code == 409
        assert "already being ingested" in response.json()["detail"]


def testAConflictQueuesNothing() -> None:
    with clientUsing(BlockingProcessor()) as client:
        client.post("/api/v1/document", json=body(ragDbId="handbook"))
        manager = app.dependency_overrides[getJobManager]()
        before = len(manager)

        client.post(
            "/api/v1/document",
            json=body(ragDbId="handbook", documentLink="https://example.com/other.pdf"),
        )

        assert len(manager) == before


def testWrongSecretIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def testUnknownServerIsRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverId="nobody"))

    assert response.status_code == 401


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink", "ragDbId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink", "ragDbId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(role="admin"))

    assert response.status_code == 422


def testTheSecretIsNeverEchoedBack(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text


@pytest.mark.parametrize("field", ["serverId", "documentLink", "ragDbId"])
def testAValidationErrorNeverEchoesTheSecret(client: TestClient, field: str) -> None:
    """A missing field is reported by pydantic with the whole body as its
    ``input``, which would put the plaintext secret in the 422."""
    payload = body(serverSecret="SUPER-SECRET-KEY")
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422
    assert "SUPER-SECRET-KEY" not in response.text


def testAValidationErrorStillSaysWhatWasWrong(client: TestClient) -> None:
    payload = body()
    del payload["ragDbId"]

    response = client.post("/api/v1/document", json=payload)

    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "ragDbId"]
    assert error["type"] == "missing"
    assert error["msg"]


def testARejectedSubmissionCreatesNoJob(client: TestClient) -> None:
    manager = app.dependency_overrides[getJobManager]()  # Same instance every call.
    before = len(manager)

    client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert len(manager) == before
