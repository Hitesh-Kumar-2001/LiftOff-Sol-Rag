import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.ingestion.ragIngestionPipeline import Chunk, lexicalSearch
from app.main import app
from app.stores.chunkStoreFactory import getChunkStore
from app.stores.localChunkStore import ENV_TEST_MODE, LocalChunkStore
from app.stores.projectStore import FirestoreProjectStore, getProjectStore

CHUNKS = [
    Chunk(text="Refunds are issued within 30 days of purchase.", index=0, tokenCount=9),
    Chunk(text="The warranty covers manufacturing defects for two years.", index=1, tokenCount=9),
    Chunk(text="Shipping fees are never refunded.", index=2, tokenCount=6),
]


# Project ids come from the scratch workspace rather than being literals: these
# tests write to real Firestore, so a fixed name like "handbook" would be shared
# by every run on every machine. `named` returns one id per name per test, so
# several requests in one test reach one project, while the next test's
# "handbook" is a project that has never existed.
_SCRATCH = None


@pytest.fixture(autouse=True)
def _scratchWorkspace(scratch):
    global _SCRATCH
    _SCRATCH = scratch
    yield
    _SCRATCH = None


def pid(name: str) -> str:
    return _SCRATCH.named(name)


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv(ENV_TEST_MODE, "1")
    projectStore = FirestoreProjectStore()
    # The chunks are stored under the ragDbId the route will resolve to, not
    # under the project id. Storing them by project id would pass only because
    # the two happened to be the same string -- exactly the assumption this
    # indirection exists to break.
    ragDbId = asyncio.run(projectStore.resolveOrCreate(pid("handbook")))

    store = LocalChunkStore(root=tmp_path)
    asyncio.run(store.save(ragDbId, CHUNKS))

    app.dependency_overrides[getChunkStore] = lambda: store
    app.dependency_overrides[getProjectStore] = lambda: projectStore
    with TestClient(app) as testClient:
        yield testClient
    app.dependency_overrides.clear()


def body(**overrides) -> dict:
    payload = {
        "serverId": "billing-service",
        "projectId": pid("handbook"),
        "query": "refund",
    }
    return payload | overrides


def testAQueryReturnsTheMatchingChunk(client: TestClient) -> None:
    response = client.post("/api/v1/search", json=body(query="refunds purchase"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["projectId"] == pid("handbook")
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
    response = client.post("/api/v1/search", json=body(projectId=pid("never-ingested")))

    assert response.status_code == 200
    assert response.json()["hits"] == []


def testSearchingAnUnknownProjectCreatesNoDatabase(client: TestClient) -> None:
    """Only /document may mint a database. If searching did too, every mistyped
    projectId would leave an empty one behind forever."""
    projects = app.dependency_overrides[getProjectStore]()

    client.post("/api/v1/search", json=body(projectId=pid("mistyped")))

    assert asyncio.run(projects.resolve(pid("mistyped"))) is None


def testTheInternalRagDbIdIsNeverReturned(client: TestClient) -> None:
    """The caller gets back the project it asked about, not where the chunks
    turned out to live."""
    ragDbId = asyncio.run(app.dependency_overrides[getProjectStore]().resolve(pid("handbook")))

    response = client.post("/api/v1/search", json=body(query="refunds purchase"))

    assert response.json()["projectId"] == pid("handbook")
    assert response.json()["hits"], "expected the search to have actually matched"
    assert ragDbId not in response.text


@pytest.mark.parametrize("field", ["serverId", "projectId", "query"])
def testEveryFieldIsRequired(client: TestClient, field: str) -> None:
    payload = body()
    del payload[field]

    assert client.post("/api/v1/search", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["serverId", "projectId", "query"])
def testBlankFieldsAreRejected(client: TestClient, field: str) -> None:
    assert client.post("/api/v1/search", json=body(**{field: ""})).status_code == 422


@pytest.mark.parametrize("topK", [0, -1, 51])
def testAnOutOfRangeTopKIsRejected(client: TestClient, topK: int) -> None:
    assert client.post("/api/v1/search", json=body(topK=topK)).status_code == 422


def testUnexpectedFieldsAreRejected(client: TestClient) -> None:
    assert client.post("/api/v1/search", json=body(role="admin")).status_code == 422


def testLongerChunksDoNotOutrankFocusedOnesOnLengthAlone() -> None:
    focused = Chunk(text="refund policy", index=0, tokenCount=2)
    padded = Chunk(text="refund " + "unrelated " * 200, index=1, tokenCount=200)

    results = lexicalSearch([focused, padded], "refund", topK=2)

    assert results[0].index == 0


def testAnEmptyQueryMatchesNothing() -> None:
    assert lexicalSearch(CHUNKS, "   ", topK=5) == []
