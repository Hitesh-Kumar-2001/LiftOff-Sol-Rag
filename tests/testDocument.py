import asyncio
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.ingestion.documents import StubDocumentProcessor
from app.jobs.job import Job
from app.jobs.jobManager import JobManager, getJobManager
from app.main import app
from app.stores.projectStore import InMemoryProjectStore, getProjectStore


class BlockingProcessor:
    """Never finishes, so a job stays mid-ingest for as long as a test needs."""

    async def process(self, job: Job) -> None:
        await asyncio.Event().wait()


@contextmanager
def clientUsing(processor) -> Generator[TestClient]:
    # One manager per test, not one per request -- a fresh instance per call
    # would make jobs vanish between the create and any later lookup.
    jobManager = JobManager(processor)
    # Likewise one mapping per test, and a fresh one each time: the real store
    # is a process-wide singleton, so without this a projectId minted by one
    # test would still resolve in the next -- and "never submitted" would stop
    # meaning that.
    projectStore = InMemoryProjectStore()

    app.dependency_overrides[getJobManager] = lambda: jobManager
    app.dependency_overrides[getProjectStore] = lambda: projectStore
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
        "documentLink": "https://example.com/handbook.pdf",
        "projectId": "handbook",
    }
    return payload | overrides


def resolve(projectId: str) -> str | None:
    """The ragDbId the API minted for ``projectId``, read straight from the
    store the test installed."""
    return asyncio.run(app.dependency_overrides[getProjectStore]().resolve(projectId))


def testASubmissionGetsItsProjectIdBack(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body())

    assert response.status_code == 202
    assert response.json() == {"projectId": "handbook", "status": "queued"}


def testTheInternalRagDbIdIsNeverReturned(client: TestClient) -> None:
    """The point of the indirection: a caller ends up holding a projectId and
    nothing that names where its chunks actually live."""
    response = client.post("/api/v1/document", json=body())

    ragDbId = resolve("handbook")
    assert ragDbId is not None, "the submission should have minted a database"
    assert ragDbId not in response.text
    assert "ragDbId" not in response.text


def testEachProjectGetsItsOwnDatabase(client: TestClient) -> None:
    client.post("/api/v1/document", json=body(projectId="handbook"))
    client.post("/api/v1/document", json=body(projectId="policies"))

    assert resolve("handbook") != resolve("policies")


def testResubmittingAProjectReusesItsDatabase(client: TestClient) -> None:
    """A project's ragDbId is minted once and never again. Minting a second
    would strand the first document's chunks in a namespace nothing resolves
    to -- still in Pinecone, still billable, unreachable by any request."""
    client.post("/api/v1/document", json=body(projectId="handbook"))
    first = resolve("handbook")

    client.post("/api/v1/document", json=body(projectId="handbook"))

    assert resolve("handbook") == first


def testResubmittingTheSameDocumentIsAccepted(client: TestClient) -> None:
    """The stub finishes immediately, so this is the settled case: the same
    document again is a duplicate, not a conflict."""
    first = client.post("/api/v1/document", json=body(projectId="handbook"))
    second = client.post("/api/v1/document", json=body(projectId="handbook"))

    assert first.status_code == second.status_code == 202


def testADifferentDocumentMidIngestIsAConflict() -> None:
    """409 rather than 202: nothing was queued, because running it would
    leave the database holding part of each document."""
    with clientUsing(BlockingProcessor()) as client:
        client.post("/api/v1/document", json=body(projectId="handbook"))

        response = client.post(
            "/api/v1/document",
            json=body(projectId="handbook", documentLink="https://example.com/other.pdf"),
        )

        assert response.status_code == 409
        assert "already being ingested" in response.json()["detail"]


def testAConflictQueuesNothing() -> None:
    with clientUsing(BlockingProcessor()) as client:
        client.post("/api/v1/document", json=body(projectId="handbook"))
        manager = app.dependency_overrides[getJobManager]()
        before = len(manager)

        client.post(
            "/api/v1/document",
            json=body(projectId="handbook", documentLink="https://example.com/other.pdf"),
        )

        assert len(manager) == before


@pytest.mark.parametrize("field", ["serverId", "documentLink", "projectId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "documentLink", "projectId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(role="admin"))

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "documentLink", "projectId"])
def testAValidationErrorDoesNotEchoTheRequestBody(client: TestClient, field: str) -> None:
    """Pydantic reports a missing field with the whole body as its ``input``,
    which would reflect whatever was sent into every log and proxy that records
    the response. app.main.validationErrorHandler allowlists what survives."""
    payload = body(documentLink="https://example.com/DISTINCTIVE-VALUE.pdf")
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422
    assert "DISTINCTIVE-VALUE" not in response.text


def testAValidationErrorStillSaysWhatWasWrong(client: TestClient) -> None:
    payload = body()
    del payload["projectId"]

    response = client.post("/api/v1/document", json=payload)

    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "projectId"]
    assert error["type"] == "missing"
    assert error["msg"]