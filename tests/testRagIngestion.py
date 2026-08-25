"""End-to-end ingestion over the corpora in ``tests/data``.

Walks each database through the real chain -- analyze, select a strategy,
chunk, store, delete -- and reports what each component decided. Run with
``-s`` to see the report:

    pytest tests/testRagIngestion.py -s

Two things are deliberately not real here. Chunks go to ``LocalChunkStore``
rather than Pinecone, so no test leaves records in a live index; and the AI
chunker is a stand-in, because db-3 splits into ~6,800 sections and the AI
path calls the model once per section. Selection, chunking, storage, and
deletion are the real code either way.

Reading the .rar corpora needs an unrar/bsdtar/7z binary; without one the
whole module skips rather than failing.
"""

import asyncio
import os
from pathlib import Path

import pytest
import rarfile

from app.ingestion.documents import analyze, extractText
from app.ingestion.ragIngestionPipeline import ChunkingStrategy, RagIngestionPipeline
from app.ingestion.ragSelector import RagSelector
from app.stores.localChunkStore import ENV_TEST_MODE, LocalChunkStore

# Parsing db-3 twice takes about a minute on its own; skip the module with
# `pytest -m "not slow"` when that is not worth waiting for.
pytestmark = pytest.mark.slow

DATA_DIR = Path(__file__).parent / "data"

# (database, files it should contain)
CASES = [("db-1", 2), ("db-2", 10), ("db-3", 200)]

UNRAR_CANDIDATES = [
    os.environ.get("RAG_UNRAR_TOOL"),
    "unrar",
    "bsdtar",
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files\7-Zip\7z.exe",
]


def findUnrarTool() -> str | None:
    """The first candidate that can actually read a member, not just list one.

    Listing a rar needs no external tool, so a tool that is missing only
    shows up when bytes are read -- which is why this opens a real member.
    """
    sample = DATA_DIR / "db-1" / "db-1.rar"
    for candidate in UNRAR_CANDIDATES:
        if not candidate:
            continue
        original = rarfile.UNRAR_TOOL
        rarfile.UNRAR_TOOL = candidate
        try:
            with rarfile.RarFile(sample) as archive:
                archive.read(archive.infolist()[0])
            return candidate
        except Exception:
            rarfile.UNRAR_TOOL = original
    return None


@pytest.fixture(scope="module", autouse=True)
def unrarTool() -> str:
    if not (DATA_DIR / "db-1" / "db-1.rar").exists():
        pytest.skip("test corpora not present under tests/data")
    tool = findUnrarTool()
    if tool is None:
        pytest.skip("no unrar/bsdtar/7z binary available to read the .rar corpora")
    rarfile.UNRAR_TOOL = tool
    return tool


async def fakeAiChunker(sections: list[str]) -> list[str]:
    """Stands in for Gemini: one chunk per section.

    Distinguishable from ``chunkWithoutAi`` on purpose -- if the pipeline
    ever routed an AI job down the non-AI path, the chunk count would not
    match what this produced.
    """
    return [section for section in sections if section.strip()]


@pytest.fixture(scope="module")
def ingested(unrarTool: str) -> dict[str, dict]:
    """Run every corpus through the chain once and keep the results.

    Module-scoped because parsing db-3 twice (once for metadata, once for
    text) takes about a minute on its own -- no test should pay that twice.
    """
    os.environ[ENV_TEST_MODE] = "1"
    store = LocalChunkStore(root=DATA_DIR / ".localStore")
    pipeline = RagIngestionPipeline(store, aiChunker=fakeAiChunker)
    selector = RagSelector()

    results: dict[str, dict] = {}
    try:
        for name, _ in CASES:
            data = (DATA_DIR / name / f"{name}.rar").read_bytes()

            metadata = analyze(f"{name}.rar", f"{name}.rar", data, None)
            strategy = selector.suggest(metadata)
            text = extractText(f"{name}.rar", data)
            result = asyncio.run(
                pipeline.runText(text, name, sourceUrl=f"{name}.rar", strategy=strategy)
            )

            results[name] = {
                "metadata": metadata,
                "strategy": strategy,
                "result": result,
                "storedPath": store.pathFor(name),
            }

        yield results
    finally:
        # The vector data goes away whatever happened above, so a failed run
        # does not leave a populated store behind for the next one.
        for name, _ in CASES:
            asyncio.run(store.delete(name))
        os.environ.pop(ENV_TEST_MODE, None)


@pytest.mark.parametrize("name,expectedFiles", CASES)
def testTheCorpusIsAnalyzedAsExpected(ingested: dict, name: str, expectedFiles: int) -> None:
    metadata = ingested[name]["metadata"]

    assert metadata.sourceKind == "folder"
    assert metadata.fileCount == expectedFiles
    assert metadata.tokenCount > 0
    assert all(f.error is None for f in metadata.files), "a member failed to parse"


@pytest.mark.parametrize("name,expectedFiles", CASES)
def testTheSelectedStrategyMatchesTheTokenCount(
    ingested: dict, name: str, expectedFiles: int
) -> None:
    """The selector's own bands, checked against what it actually chose."""
    metadata = ingested[name]["metadata"]
    strategy = ingested[name]["strategy"]

    if metadata.tokenCount < 2000:
        assert strategy is ChunkingStrategy.RAW
    elif metadata.tokenCount < 10000:
        assert strategy is ChunkingStrategy.NON_AI
    else:
        assert strategy is ChunkingStrategy.AI


@pytest.mark.parametrize("name,expectedFiles", CASES)
def testChunksAreStoredAndReadBackIntact(ingested: dict, name: str, expectedFiles: int) -> None:
    result = ingested[name]["result"]
    store = LocalChunkStore(root=DATA_DIR / ".localStore")

    assert result.chunks, "ingestion produced no chunks"
    assert result.ragDbId == name

    stored = asyncio.run(store.get(name))
    assert len(stored) == len(result.chunks)
    assert [c.index for c in stored] == list(range(len(stored))), "indices not contiguous"
    assert stored[0].text == result.chunks[0].text


def testTheReport(ingested: dict) -> None:
    """Not an assertion so much as the summary the run exists to produce."""
    lines = [
        "",
        f"{'database':10} {'files':>6} {'tokens':>10} {'pages':>6} "
        f"{'strategy':>9} {'chunks':>8} {'avg tok/chunk':>14}",
        "-" * 70,
    ]
    for name, _ in CASES:
        metadata = ingested[name]["metadata"]
        chunks = ingested[name]["result"].chunks
        average = sum(c.tokenCount for c in chunks) // len(chunks) if chunks else 0
        lines.append(
            f"{name:10} {metadata.fileCount:>6} {metadata.tokenCount:>10,} "
            f"{metadata.pageCount:>6} {ingested[name]['strategy'].name:>9} "
            f"{len(chunks):>8,} {average:>14,}"
        )
    print("\n".join(lines))


def testDeletingLeavesNothingBehind(ingested: dict) -> None:
    """Deletion is what the fixture teardown relies on, so it is checked here
    against a database of its own rather than one the other tests still read."""
    store = LocalChunkStore(root=DATA_DIR / ".localStore")
    chunks = ingested["db-1"]["result"].chunks

    asyncio.run(store.save("db-temp", chunks))
    assert store.pathFor("db-temp").exists()

    asyncio.run(store.delete("db-temp"))

    assert not store.pathFor("db-temp").exists()
    assert asyncio.run(store.get("db-temp")) == []
