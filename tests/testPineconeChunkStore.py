"""Pinecone request/response shaping, without talking to Pinecone.

Every test here covers a bug that a live run found and the rest of the
suite could not: the SDK's argument and result shapes are only wrong at
the boundary, so they are checked against the real Hit type and a stand-in
index that records what it was asked to do.
"""

import asyncio
from types import SimpleNamespace

from pinecone.models.vectors.search import Hit

from app.pineconeChunkStore import UPSERT_BATCH_SIZE, PineconeChunkStore, namespaceFor
from app.ragIngestionPipeline import Chunk


class RecordingIndex:
    """Stands in for a Pinecone index and remembers each call.

    ``upsert_records`` is keyword-only here exactly as it is on the real
    client, so a positional call fails the same way it did in production.
    """

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.batches: list[tuple[str, list[dict]]] = []
        self.searches: list[dict] = []
        self.hits = hits or []

    def upsert_records(self, *, records: list[dict], namespace: str) -> None:
        self.batches.append((namespace, records))

    def search(self, **kwargs) -> SimpleNamespace:
        self.searches.append(kwargs)
        return SimpleNamespace(result=SimpleNamespace(hits=self.hits))


def storeWith(index: RecordingIndex) -> PineconeChunkStore:
    store = PineconeChunkStore()
    store.index = index  # Already "created", so ensureIndex does nothing.
    return store


def chunks(count: int) -> list[Chunk]:
    return [Chunk(text=f"chunk {i}", index=i, tokenCount=2) for i in range(count)]


def testAlargeDocumentIsUpsertedInBatches() -> None:
    """Pinecone rejects the whole request over its batch ceiling, and a real
    document produces far more records than that."""
    index = RecordingIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    assert len(index.batches) == 3
    assert all(len(records) <= UPSERT_BATCH_SIZE for _, records in index.batches)


def testEveryChunkSurvivesBatching() -> None:
    index = RecordingIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    sent = [record for _, records in index.batches for record in records]
    assert len(sent) == 250
    assert [r["chunkIndex"] for r in sent] == list(range(250))


def testASingleBatchIsStillOneCall() -> None:
    index = RecordingIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(5)))

    assert len(index.batches) == 1


def testEveryBatchGoesToTheDatabasesOwnNamespace() -> None:
    index = RecordingIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    assert {namespace for namespace, _ in index.batches} == {namespaceFor("handbook")}


def testSavingNothingCallsPineconeNotAtAll() -> None:
    index = RecordingIndex()

    asyncio.run(storeWith(index).save("handbook", []))

    assert index.batches == []


def testHitsAreReadFromTheFieldsTheSdkActuallyUses() -> None:
    """Against the real Hit type: _id/_score are exposed as id/score, and a
    chunk's own data lives under fields."""
    index = RecordingIndex(
        hits=[
            Hit(id_="handbook::7", score_=0.42, fields={"chunkText": "a passage", "chunkIndex": 7})
        ]
    )

    results = asyncio.run(storeWith(index).search("handbook", "a question", topK=3))

    assert len(results) == 1
    assert results[0].text == "a passage"
    assert results[0].index == 7
    assert results[0].score == 0.42


def testSearchAsksTheDatabasesNamespaceForItsOwnFields() -> None:
    index = RecordingIndex()

    asyncio.run(storeWith(index).search("handbook", "a question", topK=3))

    sent = index.searches[0]
    assert sent["namespace"] == namespaceFor("handbook")
    assert sent["top_k"] == 3
    assert sent["inputs"] == {"text": "a question"}
    assert "chunkText" in sent["fields"]


def testNoMatchesIsAnEmptyListNotAnError() -> None:
    assert asyncio.run(storeWith(RecordingIndex()).search("handbook", "q", topK=3)) == []
