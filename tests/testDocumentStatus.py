import asyncio
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.documents import StubDocumentProcessor
from app.jobManager import JobManager, getJobManager
from app.jobs import Job, JobStatus
from app.main import app
from app.projectStore import InMemoryProjectStore, getProjectStore
from app.ragIngestionPipeline import ChunkingStrategy


class SelectingProcessor:
    """Records a chunking strategy the way RagIngestionProcessor does, without
    downloading or analyzing anything."""

    def __init__(self, strategy: ChunkingStrategy) -> None:
        self.strategy = strategy

    async def process(self, job: Job) -> None:
        job.strategy = self.strategy


class BlockingProcessor:
    """Never finishes, so a job stays before the point where a strategy is
    chosen."""

    async def process(self, job: Job) -> None:
        await asyncio.Event().wait()


@contextmanager
def clientUsing(processor) -> Generator[TestClient]:
    # One manager for the whole test, not one per request -- a fresh instance
    # per call would make jobs vanish between the create and the lookup.
    jobManager = JobManager(processor)
    # Likewise one mapping per test. The real store is a process-wide
    # singleton, so a shared one would leave projects minted by earlier tests
    # still resolving here.
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


@pytest.fixture
def rawClient() -> Iterator[TestClient]:
    with clientUsing(SelectingProcessor(ChunkingStrategy.RAW)) as testClient:
        yield testClient


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "projectId": "handbook",
    }
    return payload | overrides


def submit(client: TestClient, projectId: str = "handbook") -> None:
    client.post(
        "/api/v1/document",
        json={
            "serverId": "billing-service",
            "documentLink": "https://example.com/handbook.pdf",
            "projectId": projectId,
        },
    )


def waitForDone(client: TestClient, projectId: str = "handbook", attempts: int = 50) -> dict:
    """The stub processor finishes almost immediately, but on a background
    task -- poll rather than assume it has landed."""
    for _ in range(attempts):
        payload = client.post("/api/v1/document/status", json=body(projectId=projectId)).json()
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


def testAFinishedJobReportsStatusAndProjectIdOnly(client: TestClient) -> None:
    """The whole contract: two fields, nothing else."""
    submit(client)

    payload = waitForDone(client)

    assert payload == {"status": "done", "projectId": "handbook"}


def testTheInternalRagDbIdIsNeverReturned(client: TestClient) -> None:
    """A caller polls with a projectId and is answered with one. Where the
    chunks actually live stays ours to change."""
    submit(client)
    ragDbId = asyncio.run(app.dependency_overrides[getProjectStore]().resolve("handbook"))

    response = client.post("/api/v1/document/status", json=body())

    assert ragDbId is not None, "the submission should have minted a database"
    assert ragDbId not in response.text
    assert "ragDbId" not in response.text


def testARawDocumentAnswersWithItsLinkInsteadOfAProjectId(rawClient: TestClient) -> None:
    """A document kept whole was never written to a vector database, so there
    is nothing to query at all."""
    submit(rawClient)

    payload = waitForDone(rawClient)

    assert payload == {"status": "done", "documentLink": "https://example.com/handbook.pdf"}


@pytest.mark.parametrize(
    "strategy", [ChunkingStrategy.NON_AI, ChunkingStrategy.AI], ids=["nonAi", "ai"]
)
def testAChunkedDocumentAnswersWithItsProjectId(strategy: ChunkingStrategy) -> None:
    with clientUsing(SelectingProcessor(strategy)) as client:
        submit(client)

        payload = waitForDone(client)

        assert payload == {"status": "done", "projectId": "handbook"}


def testTheTwoAnswersAreNeverSentTogether(rawClient: TestClient) -> None:
    """The absent field is omitted rather than sent as null, so a caller can
    tell which arrived by presence alone."""
    submit(rawClient)

    payload = waitForDone(rawClient)

    assert ("projectId" in payload) != ("documentLink" in payload)


def testAJobStillQueuedAnswersWithItsProjectId() -> None:
    """The strategy is not known until the document has been analyzed, so a
    job that has not got there yet is answered the ordinary way."""
    with clientUsing(BlockingProcessor()) as client:
        submit(client)

        payload = client.post("/api/v1/document/status", json=body()).json()

        assert payload["projectId"] == "handbook"
        assert "documentLink" not in payload


def testTheStatusIsReportedForTheProjectIdAsked(client: TestClient) -> None:
    submit(client, projectId="policies")

    payload = client.post("/api/v1/document/status", json=body(projectId="policies")).json()

    assert payload["projectId"] == "policies"


def testAnUnknownProjectIdIsNotFound(client: TestClient) -> None:
    response = client.post("/api/v1/document/status", json=body(projectId="never-submitted"))

    assert response.status_code == 404


@pytest.mark.parametrize("field", ["serverId", "projectId"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document/status", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "projectId"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document/status", json=body(**{field: ""}))

    assert response.status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    response = client.post("/api/v1/document/status", json=body(role="admin"))

    assert response.status_code == 422