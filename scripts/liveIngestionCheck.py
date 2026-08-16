"""Drive the whole pipeline against the real Pinecone and Gemini.

    python scripts/liveIngestionCheck.py <url> [<url> ...]

Ingests each URL through the API, polls its status to completion, searches
what landed, and then deletes it again. Nothing here is mocked: real
downloads, real Gemini chunking, real vectors in a real index.

Not part of the test suite on purpose -- it costs money, needs network, and
writes to a live index. RAG_TEST_MODE must be off, or ingestion would go to
the local store and this would prove nothing.
"""

from __future__ import annotations

import os
import sys
import time

# Credentials for this run only. Set before app.security is imported, since
# the registry reads its source at import time.
SERVER_ID = "live-check"
SERVER_SECRET = "live-check-secret"
os.environ["RAG_SERVER_CREDENTIALS"] = (
    '{"' + SERVER_ID + '": {"secret": "' + SERVER_SECRET + '"}}'
)
os.environ.pop("RAG_CREDENTIALS_FILE", None)

if os.environ.get("RAG_TEST_MODE"):
    sys.exit("RAG_TEST_MODE is set; unset it or this checks the local store, not Pinecone.")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pineconeChunkStore import PINECONE_INDEX_NAME, pineconeClient  # noqa: E402

# Asked of each database once it is ingested, to show retrieval returning
# something related rather than merely returning something.
QUERIES = [
    "what is the atmosphere made of",
    "how long is a day on this planet",
    "what did spacecraft missions discover",
]

POLL_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 3


def credentials() -> dict[str, str]:
    return {"serverId": SERVER_ID, "serverSecret": SERVER_SECRET}


def ragDbIdFor(url: str) -> str:
    """A readable id per document, derived from its filename."""
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    return f"live-{name.rsplit('.', 1)[0][:40] or 'document'}"


def submit(client: TestClient, url: str, ragDbId: str) -> None:
    response = client.post(
        "/api/v1/document",
        json={**credentials(), "documentLink": url, "ragDbId": ragDbId},
    )
    response.raise_for_status()
    print(f"  submitted -> {response.json()}")


def waitForCompletion(client: TestClient, ragDbId: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    lastStatus = None
    while time.time() < deadline:
        response = client.post(
            "/api/v1/document/status", json={**credentials(), "ragDbId": ragDbId}
        )
        response.raise_for_status()
        payload = response.json()
        if payload["status"] != lastStatus:
            print(f"  status    -> {payload}")
            lastStatus = payload["status"]
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SystemExit(f"'{ragDbId}' never finished within {POLL_TIMEOUT_SECONDS}s")


def waitUntilSearchable(client: TestClient, ragDbId: str, attempts: int = 20) -> None:
    """Pinecone indexes an upsert asynchronously, so a search fired the
    instant a job reports done can legitimately find nothing yet."""
    for attempt in range(attempts):
        response = client.post(
            "/api/v1/search",
            json={**credentials(), "ragDbId": ragDbId, "query": "the", "topK": 1},
        )
        response.raise_for_status()
        if response.json()["hits"]:
            if attempt:
                print(f"  searchable after {attempt * POLL_INTERVAL_SECONDS}s")
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    print("  WARNING: nothing searchable yet; querying anyway")


def search(client: TestClient, ragDbId: str, query: str, topK: int = 3) -> None:
    response = client.post(
        "/api/v1/search",
        json={**credentials(), "ragDbId": ragDbId, "query": query, "topK": topK},
    )
    response.raise_for_status()
    hits = response.json()["hits"]
    print(f'  "{query}" -> {len(hits)} hit(s)')
    for hit in hits:
        snippet = " ".join(hit["text"].split())[:110]
        print(f"      [{hit['score']:.3f}] #{hit['chunkIndex']:<4} {snippet}")


def main(urls: list[str]) -> None:
    ragDbIds = [ragDbIdFor(url) for url in urls]
    store = None

    with TestClient(app) as client:
        try:
            for url, ragDbId in zip(urls, ragDbIds):
                print(f"\n=== {ragDbId} <- {url}")
                started = time.time()
                submit(client, url, ragDbId)
                result = waitForCompletion(client, ragDbId)
                print(f"  finished in {time.time() - started:.1f}s")

                if result["status"] == "failed":
                    print("  FAILED -- skipping search for this one")
                    continue
                waitUntilSearchable(client, ragDbId)
                for query in QUERIES:
                    search(client, ragDbId, query)
        finally:
            # Runs even on failure, so a broken run does not leave vectors
            # behind in a live index.
            print("\n=== cleanup")
            from app.pineconeChunkStore import PineconeChunkStore

            store = PineconeChunkStore()
            import asyncio

            for ragDbId in ragDbIds:
                asyncio.run(store.delete(ragDbId))
                remaining = asyncio.run(store.get(ragDbId))
                print(f"  deleted {ragDbId}: {len(remaining)} chunk(s) left")

    if "--keep-index" not in sys.argv:
        print(f"  deleting index '{PINECONE_INDEX_NAME}'")
        pineconeClient().delete_index(PINECONE_INDEX_NAME)
        print("  index deleted")


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not arguments:
        sys.exit(__doc__)
    main(arguments)
