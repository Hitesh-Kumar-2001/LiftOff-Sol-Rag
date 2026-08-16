"""The chunk-store choice, and the guard that keeps the local one out of
production."""

import asyncio
import os
from collections.abc import Iterator

import pytest

from app.chunkStoreFactory import buildChunkStore
from app.localChunkStore import ENV_TEST_MODE, LocalChunkStore, fileNameFor
from app.pineconeChunkStore import PineconeChunkStore
from app.ragIngestionPipeline import (
    MAX_EMBED_TOKENS,
    Chunk,
    ChunkingStrategy,
    IngestionError,
    RagIngestionPipeline,
    chunkWithoutAi,
    countTokens,
    enforceEmbedLimit,
)


@pytest.fixture
def testMode() -> Iterator[None]:
    previous = os.environ.get(ENV_TEST_MODE)
    os.environ[ENV_TEST_MODE] = "1"
    yield
    if previous is None:
        os.environ.pop(ENV_TEST_MODE, None)
    else:
        os.environ[ENV_TEST_MODE] = previous


@pytest.fixture
def productionMode() -> Iterator[None]:
    previous = os.environ.get(ENV_TEST_MODE)
    os.environ.pop(ENV_TEST_MODE, None)
    yield
    if previous is not None:
        os.environ[ENV_TEST_MODE] = previous


def testTheLocalStoreRefusesToBeBuiltOutsideTestMode(productionMode: None) -> None:
    with pytest.raises(IngestionError, match=ENV_TEST_MODE):
        LocalChunkStore()


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def testOnlyAnAffirmativeFlagEnablesTheLocalStore(value: str) -> None:
    os.environ[ENV_TEST_MODE] = value
    try:
        with pytest.raises(IngestionError):
            LocalChunkStore()
    finally:
        os.environ.pop(ENV_TEST_MODE, None)


def testProductionGetsPinecone(productionMode: None) -> None:
    assert isinstance(buildChunkStore(), PineconeChunkStore)


def testTestModeGetsTheLocalStore(testMode: None) -> None:
    assert isinstance(buildChunkStore(), LocalChunkStore)


def testChunksSurviveARoundTrip(testMode: None, tmp_path) -> None:
    store = LocalChunkStore(root=tmp_path)
    chunks = [
        Chunk(text="first", index=0, tokenCount=1),
        Chunk(text="second", index=1, tokenCount=1),
    ]

    asyncio.run(store.save("handbook", chunks))
    stored = asyncio.run(store.get("handbook"))

    assert [c.text for c in stored] == ["first", "second"]
    assert [c.index for c in stored] == [0, 1]


def testAnUnknownDatabaseIsEmptyRatherThanAnError(testMode: None, tmp_path) -> None:
    store = LocalChunkStore(root=tmp_path)

    assert asyncio.run(store.get("never-written")) == []


def testDeletingIsSafeToRepeat(testMode: None, tmp_path) -> None:
    store = LocalChunkStore(root=tmp_path)
    asyncio.run(store.save("handbook", [Chunk(text="x", index=0, tokenCount=1)]))

    asyncio.run(store.delete("handbook"))
    asyncio.run(store.delete("handbook"))  # Already gone; must not raise.

    assert asyncio.run(store.get("handbook")) == []


def testSavingReplacesRatherThanAppends(testMode: None, tmp_path) -> None:
    """Re-ingesting a ragDbId overwrites it, the same contract Pinecone's
    namespace-per-database gives."""
    store = LocalChunkStore(root=tmp_path)

    asyncio.run(store.save("handbook", [Chunk(text=f"v1-{i}", index=i, tokenCount=1) for i in range(5)]))
    asyncio.run(store.save("handbook", [Chunk(text="v2", index=0, tokenCount=1)]))

    stored = asyncio.run(store.get("handbook"))
    assert [c.text for c in stored] == ["v2"]


def testAnOversizedChunkIsResplitRatherThanTruncated() -> None:
    """The embedder truncates silently past its limit, so nothing may reach a
    store over it -- whichever strategy produced it."""
    oversized = "word " * 6000
    assert countTokens(oversized) > MAX_EMBED_TOKENS

    chunks = enforceEmbedLimit([oversized])

    assert len(chunks) > 1
    assert all(countTokens(c) <= MAX_EMBED_TOKENS for c in chunks)


def testChunksAlreadyWithinTheLimitArePassedThrough() -> None:
    chunks = ["short one", "short two"]

    assert enforceEmbedLimit(chunks) == chunks


def testTheRawStrategyCannotProduceAnUnembeddableChunk(testMode: None, tmp_path) -> None:
    """RAW stores a document whole, so it is the strategy most able to exceed
    the limit if nothing re-splits it."""
    pipeline = RagIngestionPipeline(LocalChunkStore(root=tmp_path))

    result = asyncio.run(
        pipeline.runText("word " * 6000, "big", strategy=ChunkingStrategy.RAW)
    )

    assert len(result.chunks) > 1
    assert all(c.tokenCount <= MAX_EMBED_TOKENS for c in result.chunks)


@pytest.mark.parametrize("overlap", [400, 401, 1000])
def testChunkingRefusesAnOverlapItCouldNotAdvancePast(overlap: int) -> None:
    """With an overlap at least as large as the window, each chunk starts
    where the last one did -- the loop would never end and never raise."""
    with pytest.raises(IngestionError, match="smaller than the chunk"):
        chunkWithoutAi(["word " * 100], chunkTokens=400, overlapTokens=overlap)


def testChunkingProceedsWhenTheOverlapLeavesRoom() -> None:
    chunks = chunkWithoutAi(["word " * 100], chunkTokens=400, overlapTokens=399)

    assert chunks


def testIdsThatSanitizeAlikeDoNotCollide() -> None:
    assert fileNameFor("a/b") != fileNameFor("a:b")


def testTheFileNameKeepsTheIdReadable() -> None:
    assert fileNameFor("handbook").startswith("handbook-")
