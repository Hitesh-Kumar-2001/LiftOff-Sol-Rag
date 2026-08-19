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

# Who this run says it is. Nothing verifies it -- the API has no
# authentication -- so it exists to label this script's requests in the log.
SERVER_ID = "live-check"

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


def caller() -> dict[str, str]:
    return {"serverId": SERVER_ID}


def projectIdFor(url: str) -> str:
    """A readable project per document, derived from its filename."""
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    return f"live-{name.rsplit('.', 1)[0][:40] or 'document'}"


def submit(client: TestClient, url: str, projectId: str) -> None:
    response = client.post(
        "/api/v1/document",
        json={**caller(), "documentLink": url, "projectId": projectId},
    )
    response.raise_for_status()
    print(f"  submitted -> {response.json()}")


def waitForCompletion(client: TestClient, projectId: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    lastStatus = None
    while time.time() < deadline:
        response = client.post(
            "/api/v1/document/status", json={**caller(), "projectId": projectId}
        )
        response.raise_for_status()
        payload = response.json()
        if payload["status"] != lastStatus:
            print(f"  status    -> {payload}")
            lastStatus = payload["status"]
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise SystemExit(f"'{projectId}' never finished within {POLL_TIMEOUT_SECONDS}s")


def waitUntilSearchable(client: TestClient, projectId: str, attempts: int = 20) -> None:
    """Pinecone indexes an upsert asynchronously, so a search fired the
    instant a job reports done can legitimately find nothing yet."""
    for attempt in range(attempts):
        response = client.post(
            "/api/v1/search",
            json={**caller(), "projectId": projectId, "query": "the", "topK": 1},
        )
        response.raise_for_status()
        if response.json()["hits"]:
            if attempt:
                print(f"  searchable after {attempt * POLL_INTERVAL_SECONDS}s")
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    print("  WARNING: nothing searchable yet; querying anyway")


def search(client: TestClient, projectId: str, query: str, topK: int = 3) -> None:
    response = client.post(
        "/api/v1/search",
        json={**caller(), "projectId": projectId, "query": query, "topK": topK},
    )
    response.raise_for_status()
    hits = response.json()["hits"]
    print(f'  "{query}" -> {len(hits)} hit(s)')
    for hit in hits:
        snippet = " ".join(hit["text"].split())[:110]
        print(f"      [{hit['score']:.3f}] #{hit['chunkIndex']:<4} {snippet}")


def main(urls: list[str]) -> None:
    projectIds = [projectIdFor(url) for url in urls]
    store = None

    with TestClient(app) as client:
        try:
            for url, projectId in zip(urls, projectIds):
                print(f"\n=== {projectId} <- {url}")
                started = time.time()
                submit(client, url, projectId)
                result = waitForCompletion(client, projectId)
                print(f"  finished in {time.time() - started:.1f}s")

                if result["status"] == "failed":
                    print("  FAILED -- skipping search for this one")
                    continue
                waitUntilSearchable(client, projectId)
                for query in QUERIES:
                    search(client, projectId, query)
        finally:
            # Runs even on failure, so a broken run does not leave vectors
            # behind in a live index.
            print("\n=== cleanup")
            from app.pineconeChunkStore import PineconeChunkStore

            store = PineconeChunkStore()
            import asyncio

            from app.projectStore import getProjectStore

            # Pinecone namespaces are keyed by ragDbId, not projectId, so the
            # ids have to be resolved before anything can be deleted. This is
            # the same process-wide store the API just used, so the mapping it
            # minted is the one read back here.
            projects = getProjectStore()
            for projectId in projectIds:
                ragDbId = asyncio.run(projects.resolve(projectId))
                if ragDbId is None:
                    print(f"  {projectId}: no database was ever created, nothing to delete")
                    continue
                asyncio.run(store.delete(ragDbId))
                remaining = asyncio.run(store.get(ragDbId))
                print(f"  deleted {projectId} ({ragDbId}): {len(remaining)} chunk(s) left")

    if "--keep-index" not in sys.argv:
        print(f"  deleting index '{PINECONE_INDEX_NAME}'")
        pineconeClient().delete_index(PINECONE_INDEX_NAME)
        print("  index deleted")


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not arguments:
        sys.exit(__doc__)
    main(arguments)
