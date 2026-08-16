import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.chunkStoreFactory import getChunkStore
from app.credentials import InMemoryCredentialSource, ServerCredential, hashSecret
from app.localChunkStore import ENV_TEST_MODE, LocalChunkStore
from app.main import app
from app.ragIngestionPipeline import Chunk, lexicalSearch
from app.security import ServerRegistry, getServerRegistry

SECRET = "s3cr3t-api-key"

CHUNKS = [
    Chunk(text="Refunds are issued within 30 days of purchase.", index=0, tokenCount=9),
    Chunk(text="The warranty covers manufacturing defects for two years.", index=1, tokenCount=9),
    Chunk(text="Shipping fees are never refunded.", index=2, tokenCount=6),
]


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv(ENV_TEST_MODE, "1")
    registry = ServerRegistry(
        InMemoryCredentialSource(
            [ServerCredential(serverId="billing-service", secretHash=hashSecret(SECRET))]
        )
    )
    asyncio.run(registry.loadAll())

    store = LocalChunkStore(root=tmp_path)
    asyncio.run(store.save("handbook", CHUNKS))

    app.dependency_overrides[getServerRegistry] = lambda: registry
    app.dependency_overrides[getChunkStore] = lambda: store
    with TestClient(app) as testClient:
        yield testClient
    app.dependency_overrides.clear()


def body(**overrides) -> dict:
    payload = {
        "serverId": "billing-service",
        "serverSecret": SECRET,
        "ragDbId": "handbook",
        "query": "refund",
    }
    return payload | overrides


def testAQueryReturnsTheMatchingChunk(client: TestClient) -> None:
    response = client.post("/api/v1/search", json=body(query="refunds purchase"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["ragDbId"] == "handbook"
    assert payload["hits"], "expected at least one hit"
    assert "Refunds are issued" in payload["hits"][0]["text"]


def testHitsCarryTheirChunkIndexAndScore(client: TestClient) -> None:
    hit = client.post("/api/v1/search", json=body(query="warranty defects")).json()["hits"][0]

    assert hit["chunkIndex"] == 1
    assert hit["score"] > 0


def testHitsComeBackBestFirst(client: TestClient) -> None:
    hits = client.post("/api/v1/search", json=body(query="refunded fees shipping")).json()["hits"]

    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def testTopKLimitsTheNumberOfHits(client: TestClient) -> None:
    hits = client.post("/api/v1/search", json=body(query="are", topK=1)).json()["hits"]

    assert len(hits) == 1


def testAQueryMatchingNothingReturnsNoHits(client: TestClient) -> None:
    payload = client.post("/api/v1/search", json=body(query="zzzznotpresent")).json()

    assert payload["hits"] == []


def testAnUningestedDatabaseReturnsNoHits(client: TestClient) -> None:
    """Nothing was stored under this id, which is the same answer as nothing
    matching -- not a 404."""
    response = client.post("/api/v1/search", json=body(ragDbId="never-ingested"))

    assert response.status_code == 200
    assert response.json()["hits"] == []


def testWrongSecretIsRejected(client: TestClient) -> None:
    assert client.post("/api/v1/search", json=body(serverSecret="wrong")).status_code == 401


def testUnknownServerIsRejected(client: TestClient) -> None:
    assert client.post("/api/v1/search", json=body(serverId="nobody")).status_code == 401


def testSearchDoesNotRunForAnUnverifiedServer(client: TestClient) -> None:
    """Authentication comes first, so a bad caller cannot read chunks."""
    response = client.post("/api/v1/search", json=body(serverSecret="wrong"))

    assert "Refunds are issued" not in response.text


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "ragDbId", "query"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    assert client.post("/api/v1/search", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["serverId", "serverSecret", "ragDbId", "query"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    assert client.post("/api/v1/search", json=body(**{field: ""})).status_code == 422


@pytest.mark.parametrize("topK", [0, -1, 51])
def testAnOutOfRangeTopKIsRejected(client: TestClient, topK: int) -> None:
    assert client.post("/api/v1/search", json=body(topK=topK)).status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    assert client.post("/api/v1/search", json=body(role="admin")).status_code == 422


def testTheSecretIsNeverEchoedBack(client: TestClient) -> None:
    assert "wrong" not in client.post("/api/v1/search", json=body(serverSecret="wrong")).text


def testLongerChunksDoNotOutrankFocusedOnesOnLengthAlone() -> None:
    focused = Chunk(text="refund policy", index=0, tokenCount=2)
    padded = Chunk(text="refund " + "unrelated " * 200, index=1, tokenCount=200)

    results = lexicalSearch([focused, padded], "refund", topK=2)

    assert results[0].index == 0


def testAnEmptyQueryMatchesNothing() -> None:
    assert lexicalSearch(CHUNKS, "   ", topK=5) == []
