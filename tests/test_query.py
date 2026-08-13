import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.credentials import InMemoryCredentialSource, ServerCredential, hash_secret
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

    app.dependency_overrides[get_server_registry] = lambda: registry
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


def test_a_verified_server_gets_a_response(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ragDbId"] == "handbook"
    assert "What is the refund window?" in payload["answer"]


def test_wrong_secret_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert response.status_code == 401


def test_unknown_server_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverId="nobody"))

    assert response.status_code == 401


def test_unknown_server_and_wrong_secret_are_indistinguishable(client: TestClient) -> None:
    unknown = client.post("/api/v1/query", json=body(serverId="nobody"))
    wrong_secret = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert unknown.json() == wrong_secret.json()


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "question", "ragDbId"])
def test_every_field_is_required(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    response = client.post("/api/v1/query", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "question", "ragDbId"])
def test_blank_fields_are_rejected(client: TestClient, field: str) -> None:
    response = client.post("/api/v1/query", json=body(**{field: ""}))

    assert response.status_code == 422


def test_unexpected_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(role="admin"))

    assert response.status_code == 422


def test_the_secret_is_never_echoed_back(client: TestClient) -> None:
    response = client.post("/api/v1/query", json=body(serverSecret="wrong"))

    assert "wrong" not in response.text
