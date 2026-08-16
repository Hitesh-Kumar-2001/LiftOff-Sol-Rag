"""Pinecone request/response shaping, without talking to Pinecone.

Every test here covers a bug that a live run found and the rest of the
suite could not: the SDK's argument and result shapes are only wrong at
the boundary, so they are checked against the real Hit type and a stand-in
index that records what it was asked to do.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pinecone.models.vectors import FetchResponse, ListItem, ListResponse, Vector
from pinecone.models.vectors.search import Hit

from app.pineconeChunkStore import UPSERT_BATCH_SIZE, PineconeChunkStore, namespaceFor
from app.ragIngestionPipeline import Chunk


class SimulatedIndex:
    """A stand-in that behaves the way the real index does, and remembers how
    it was called.

    Built from the SDK's own response types rather than dictionaries, because
    every bug this file covers was a disagreement about those types: an
    upsert replaces only the ids it names, ``list`` pages hold ListItem
    objects rather than id strings, and ``fetch`` returns Vectors keyed by id.
    Every method is keyword-only exactly as the real client's are, so a
    positional call fails here the same way it did in production.
    """

    PAGE_SIZE = 100

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.records: dict[str, dict] = {}
        self.namespaceOf: dict[str, str] = {}
        self.batches: list[tuple[str, list[dict]]] = []
        self.searches: list[dict] = []
        self.hits = hits or []
        self.failUpsertAfter: int | None = None

    def upsert_records(self, *, records: list[dict], namespace: str) -> None:
        if self.failUpsertAfter is not None and len(self.batches) >= self.failUpsertAfter:
            raise RuntimeError("upsert failed")
        self.batches.append((namespace, records))
        for record in records:
            self.records[record["_id"]] = record
            self.namespaceOf[record["_id"]] = namespace

    def search(self, **kwargs) -> SimpleNamespace:
        self.searches.append(kwargs)
        return SimpleNamespace(result=SimpleNamespace(hits=self.hits))

    def idsIn(self, namespace: str) -> list[str]:
        return sorted(i for i, ns in self.namespaceOf.items() if ns == namespace)

    def list(self, *, namespace: str = "", **kwargs):
        ids = self.idsIn(namespace)
        for start in range(0, len(ids), self.PAGE_SIZE):
            yield ListResponse(
                vectors=[ListItem(id=i) for i in ids[start : start + self.PAGE_SIZE]],
                namespace=namespace,
            )

    def fetch(self, *, ids, namespace: str = "", **kwargs) -> FetchResponse:
        return FetchResponse(
            vectors={
                i: Vector(id=i, values=[0.0], metadata=self.records[i])
                for i in ids
                if i in self.records
            },
            namespace=namespace,
        )

    def delete(self, *, ids=None, delete_all: bool = False, namespace: str = "", **kwargs) -> None:
        targets = self.idsIn(namespace) if delete_all else (ids or [])
        for i in targets:
            self.records.pop(i, None)
            self.namespaceOf.pop(i, None)


def storeWith(index) -> PineconeChunkStore:
    store = PineconeChunkStore()
    store.index = index  # Already "created", so ensureIndex does nothing.
    return store


def chunks(count: int) -> list[Chunk]:
    return [Chunk(text=f"chunk {i}", index=i, tokenCount=2) for i in range(count)]


def testAlargeDocumentIsUpsertedInBatches() -> None:
    """Pinecone rejects the whole request over its batch ceiling, and a real
    document produces far more records than that."""
    index = SimulatedIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    assert len(index.batches) == 3
    assert all(len(records) <= UPSERT_BATCH_SIZE for _, records in index.batches)


def testEveryChunkSurvivesBatching() -> None:
    index = SimulatedIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    sent = [record for _, records in index.batches for record in records]
    assert len(sent) == 250
    assert [r["chunkIndex"] for r in sent] == list(range(250))


def testASingleBatchIsStillOneCall() -> None:
    index = SimulatedIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(5)))

    assert len(index.batches) == 1


def testEveryBatchGoesToTheDatabasesOwnNamespace() -> None:
    index = SimulatedIndex()

    asyncio.run(storeWith(index).save("handbook", chunks(250)))

    assert {namespace for namespace, _ in index.batches} == {namespaceFor("handbook")}


def testSavingNothingCallsPineconeNotAtAll() -> None:
    index = SimulatedIndex()

    asyncio.run(storeWith(index).save("handbook", []))

    assert index.batches == []


def testAShorterReingestClearsWhatItDoesNotOverwrite() -> None:
    """An upsert only replaces the ids it names, so without this every record
    past the new document's last chunk stays live and keeps being searched."""
    index = SimulatedIndex()
    store = storeWith(index)

    asyncio.run(store.save("handbook", chunks(200)))
    asyncio.run(store.save("handbook", chunks(5)))

    assert sorted(index.records) == sorted(f"handbook::{i}" for i in range(5))


def testALongerReingestKeepsEveryNewRecord() -> None:
    index = SimulatedIndex()
    store = storeWith(index)

    asyncio.run(store.save("handbook", chunks(5)))
    asyncio.run(store.save("handbook", chunks(200)))

    assert len(index.records) == 200


def testReingestReplacesTheTextOfAChunkThatSurvives() -> None:
    index = SimulatedIndex()
    store = storeWith(index)

    asyncio.run(store.save("handbook", [Chunk(text="old", index=0, tokenCount=1)]))
    asyncio.run(store.save("handbook", [Chunk(text="new", index=0, tokenCount=1)]))

    assert index.records["handbook::0"]["chunkText"] == "new"


def testStaleRecordsAreClearedOnlyAfterTheNewOnesLand() -> None:
    """Clearing first would empty the database for the length of the upsert,
    and lose the old document altogether if the upsert then failed."""
    index = SimulatedIndex()
    store = storeWith(index)
    asyncio.run(store.save("handbook", chunks(200)))

    index.failUpsertAfter = 1
    with pytest.raises(RuntimeError):
        asyncio.run(store.save("handbook", chunks(5)))

    assert len(index.records) == 200, "old document was lost by a failed re-ingest"


def testAnotherDatabasesRecordsAreNeverCleared() -> None:
    index = SimulatedIndex()
    store = storeWith(index)
    asyncio.run(store.save("handbook", chunks(50)))
    asyncio.run(store.save("policies", chunks(50)))

    asyncio.run(store.save("handbook", chunks(1)))

    assert sum(1 for k in index.records if k.startswith("policies::")) == 50


def testIdsAreReadOffListItemsRatherThanThePageItself() -> None:
    """``index.list`` pages hold ListItem objects; the rest of the SDK wants
    plain id strings."""
    index = SimulatedIndex()
    store = storeWith(index)
    asyncio.run(store.save("handbook", chunks(3)))

    ids = asyncio.run(store.listIds("handbook"))

    assert sorted(ids) == ["handbook::0", "handbook::1", "handbook::2"]
    assert all(isinstance(i, str) for i in ids)


def testGetReadsBackWhatWasSaved() -> None:
    index = SimulatedIndex()
    store = storeWith(index)
    asyncio.run(store.save("handbook", chunks(3)))

    stored = asyncio.run(store.get("handbook"))

    assert [c.index for c in stored] == [0, 1, 2]
    assert [c.text for c in stored] == ["chunk 0", "chunk 1", "chunk 2"]


def testGetOnAnEmptyDatabaseIsEmpty() -> None:
    assert asyncio.run(storeWith(SimulatedIndex()).get("never-written")) == []


def testHitsAreReadFromTheFieldsTheSdkActuallyUses() -> None:
    """Against the real Hit type: _id/_score are exposed as id/score, and a
    chunk's own data lives under fields."""
    index = SimulatedIndex(
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
    index = SimulatedIndex()

    asyncio.run(storeWith(index).search("handbook", "a question", topK=3))

    sent = index.searches[0]
    assert sent["namespace"] == namespaceFor("handbook")
    assert sent["top_k"] == 3
    assert sent["inputs"] == {"text": "a question"}
    assert "chunkText" in sent["fields"]


def testNoMatchesIsAnEmptyListNotAnError() -> None:
    assert asyncio.run(storeWith(SimulatedIndex()).search("handbook", "q", topK=3)) == []
