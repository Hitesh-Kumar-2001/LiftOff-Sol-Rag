"""Check PineconeChunkStore against the real Pinecone.

    python scripts/livePineconeCheck.py

Covers what a stand-in index cannot: that save/get/search/delete agree with
the service about ids, pages, fields and batch limits. Creates an index,
exercises it, and deletes it again.

No Gemini here -- chunks are made up, because what is under test is the
store, not the chunking.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

if os.environ.get("RAG_TEST_MODE"):
    sys.exit("RAG_TEST_MODE is set; unset it or this checks the local store.")

from app.ingestion.ragIngestionPipeline import Chunk  # noqa: E402
from app.stores.pineconeChunkStore import (  # noqa: E402
    PINECONE_INDEX_NAME,
    PineconeChunkStore,
    pineconeClient,
)

RAG_DB = "live-store-check"
OTHER_DB = "live-store-neighbour"

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"  {'ok ' if ok else 'FAIL'}  {label}: {actual!r}" + ("" if ok else f" != {expected!r}"))
    if not ok:
        failures.append(label)


def chunks(count: int, version: str) -> list[Chunk]:
    return [
        Chunk(text=f"{version} chunk {i} about planetary atmospheres", index=i, tokenCount=6)
        for i in range(count)
    ]


async def settle(store: PineconeChunkStore, ragDbId: str, expected: int, timeout: int = 120) -> None:
    """Pinecone indexes writes asynchronously; wait for the count to agree."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(await store.get(ragDbId)) == expected:
            return
        await asyncio.sleep(3)


async def main() -> None:
    store = PineconeChunkStore()

    print("creating index (this takes a moment)")
    await store.ensureIndex()

    try:
        print("\nsave 200 chunks, then read them back")
        await store.save(RAG_DB, chunks(200, "v1"))
        await settle(store, RAG_DB, 200)
        stored = await store.get(RAG_DB)
        check("chunks read back", len(stored), 200)
        check("order preserved", [c.index for c in stored[:3]], [0, 1, 2])
        check("text intact", stored[0].text.startswith("v1 chunk 0"), True)

        print("\nsave a neighbouring database, to prove it is untouched later")
        await store.save(OTHER_DB, chunks(20, "other"))
        await settle(store, OTHER_DB, 20)

        print("\nre-ingest the same database with only 5 chunks")
        await store.save(RAG_DB, chunks(5, "v2"))
        await settle(store, RAG_DB, 5)
        stored = await store.get(RAG_DB)
        check("stale records cleared", len(stored), 5)
        check("no v1 text survives", [c for c in stored if c.text.startswith("v1")], [])
        check("neighbour untouched", len(await store.get(OTHER_DB)), 20)

        print("\nsearch returns only the new document")
        hits = await store.search(RAG_DB, "planetary atmospheres", topK=5)
        check("hits found", len(hits) > 0, True)
        check("no v1 in hits", [h for h in hits if h.text.startswith("v1")], [])
        if hits:
            print(f"      top hit: [{hits[0].score:.3f}] #{hits[0].index} {hits[0].text[:60]}")

        print("\ndelete")
        await store.delete(RAG_DB)
        await settle(store, RAG_DB, 0)
        check("database emptied", await store.get(RAG_DB), [])
        check("neighbour still there", len(await store.get(OTHER_DB)), 20)
        await store.delete(OTHER_DB)
    finally:
        print(f"\ndeleting index '{PINECONE_INDEX_NAME}'")
        pineconeClient().delete_index(PINECONE_INDEX_NAME)
        print("index deleted")

    print("\n" + ("FAILURES: " + ", ".join(failures) if failures else "all checks passed"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
