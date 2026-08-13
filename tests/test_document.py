import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hash_secret
from app.jobs import JobStore, StubDocumentProcessor, get_job_store
from app.main import app
from app.security import ServerRegistry, get_server_registry

SECRET = "s3cr3t-api-key"


@pytest.fixture
def client() -> Iterator[TestClient]:
    registry = ServerRegistry(
        InMemoryCredentialSource(
            [ServerCredential(server_id="billing-service", secret_hash=hash_secret(SECRET))]
        )
    )
    asyncio.run(registry.load_all())

    # One store per test, not one per request -- a fresh instance per call
    # would make jobs vanish between the create and any later lookup.
    job_store = JobStore(StubDocumentProcessor())

    app.dependency_overrides[get_server_registry] = lambda: registry
    app.dependency_overrides[get_job_store] = lambda: job_store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def body(**overrides: str) -> dict[str, str]:
    payload = {
        "serverId": "billing-service",
        "serverSecret": SECRET,
        "documentLink": "https://example.com/handbook.pdf",
    }
    return payload | overrides


def test_a_verified_server_gets_a_job_id(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body())

    assert response.status_code == 202
    payload = response.json()
    assert payload["jobId"]
    assert payload["status"] == "queued"


def test_each_submission_gets_a_distinct_job_id(client: TestClient) -> None:
    first = client.post("/api/v1/document", json=body()).json()
    second = client.post("/api/v1/document", json=body()).json()

    assert first["jobId"] != second["jobId"]


def test_wrong_secret_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def test_unknown_server_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverId="nobody"))

    assert response.status_code == 401


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink"])
def test_every_field_is_required(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/document", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "documentLink"])
def test_blank_fields_are_rejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/document", json=body(**{field: ""}))

    assert response.status_code == 422


def test_unexpected_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(role="admin"))

    assert response.status_code == 422


def test_the_secret_is_never_echoed_back(client: TestClient) -> None:
    response = client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text


def test_a_rejected_submission_creates_no_job(client: TestClient) -> None:
    store = app.dependency_overrides[get_job_store]()  # Same instance every call.
    before = len(store)

    client.post("/api/v1/document", json=body(serverSecret="wrong"))

    assert len(store) == before
