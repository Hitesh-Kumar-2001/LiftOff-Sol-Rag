"""On-disk ``ChunkStore`` for tests.

Writes chunks to JSON files instead of a vector database, so a test can
ingest a corpus and read back exactly what was stored without a network
call, an API key, or anything left behind in a real index.

Refuses to be constructed unless test mode is on. That guard is the point
of the class: a misconfigured deployment should fail loudly at startup
rather than quietly accept ingestion into a directory nothing can query.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from app.ragIngestionPipeline import Chunk, IngestionError

ENV_TEST_MODE = "RAG_TEST_MODE"
ENV_LOCAL_STORE_DIR = "RAG_LOCAL_STORE_DIR"

DEFAULT_LOCAL_STORE_DIR = ".localChunkStore"

_TRUTHY = {"1", "true", "yes", "on"}


def testModeEnabled() -> bool:
    """Whether this process is allowed to store chunks outside a real
    vector database. Off unless deliberately switched on."""
    return os.environ.get(ENV_TEST_MODE, "").strip().lower() in _TRUTHY


def fileNameFor(ragDbId: str) -> str:
    """A filesystem-safe name for ``ragDbId``.

    The readable part is kept so a stored corpus can be inspected by eye
    during a test; the digest keeps two ids that sanitize to the same string
    from colliding.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ragDbId)[:64]
    digest = hashlib.sha256(ragDbId.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}.json"


class LocalChunkStore:
    """Chunks as one JSON file per ``ragDbId``.

    A drop-in ``ChunkStore``, so a test drives exactly the pipeline
    production drives -- only the destination differs.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if not testModeEnabled():
            raise IngestionError(
                f"LocalChunkStore is for tests only; set {ENV_TEST_MODE}=1 to use it. "
                "Production ingestion goes to Pinecone."
            )
        self.root = Path(
            root or os.environ.get(ENV_LOCAL_STORE_DIR) or DEFAULT_LOCAL_STORE_DIR
        )

    def pathFor(self, ragDbId: str) -> Path:
        return self.root / fileNameFor(ragDbId)

    async def save(self, ragDbId: str, chunks: list[Chunk]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "ragDbId": ragDbId,
            "chunks": [
                {"index": c.index, "tokenCount": c.tokenCount, "text": c.text} for c in chunks
            ],
        }
        self.pathFor(ragDbId).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def get(self, ragDbId: str) -> list[Chunk]:
        path = self.pathFor(ragDbId)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(text=c["text"], index=c["index"], tokenCount=c["tokenCount"])
            for c in payload["chunks"]
        ]
        chunks.sort(key=lambda c: c.index)
        return chunks

    async def delete(self, ragDbId: str) -> None:
        self.pathFor(ragDbId).unlink(missing_ok=True)
