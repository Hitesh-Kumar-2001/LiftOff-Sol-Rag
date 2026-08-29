"""Load -> split -> chunk -> store.

The loader here reads plain text over http(s) or from a local path; it does
not (yet) share app.ingestion.documents.download, which already handles PDFs/docx/csv/
archives -- a document downloaded there for metadata still gets re-fetched
here as raw text, so a PDF's bytes are not usefully chunked yet.

Chunks are stored under a caller-supplied ``ragDbId`` -- the id of the RAG
database being populated, not the source URL -- so re-ingesting a different
document into the same ``ragDbId`` overwrites what's there, same as a job
resubmission under an existing ``ragDbId`` (see ``app.jobs.jobManager``).

**Every CPU-bound step runs in a thread.** ``split``, ``chunkWithoutAi``,
``enforceEmbedLimit`` and ``buildChunks`` are ordinary synchronous functions,
and awaiting them directly held the event loop for the length of the whole
document -- minutes on a large corpus. Nothing else in the process ran during
that: not another request, and not the health check, so a platform watching the
service concluded it was dead and killed a task that was working perfectly.

The reason a thread is enough, rather than a process: tiktoken's encode and
decode are Rust and release the GIL, and they are almost all of the cost.
``split`` is a single ``re`` pass and ``re`` does *not* release the GIL, so that
one still blocks -- but it is one pass over the text rather than one per chunk,
which is a far smaller window and not worth the cost of shipping megabytes of
text to another process to avoid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx
import tiktoken
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Which model does the AI chunking is not decided here: it is the [chunker]
# role in config/models.toml, resolved through app.agent.llmManager like every
# other model this service calls. This module used to pin
# `gemini-3.5-flash-lite` and reach for Gemini's SDK directly -- the one AI call
# that could not be moved without a code change, and the one that stopped a
# large ingestion dead when that vendor's per-day quota ran out.

DOWNLOAD_TIMEOUT_SECONDS = 60.0
TOKEN_ENCODING_NAME = "cl100k_base"

DEFAULT_CHUNK_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 40

# The smallest input window among the embedders we target: Pinecone's
# llama-text-embed-v2 caps at 2048 tokens and, by default, truncates from the
# END rather than refusing -- an oversized chunk loses its tail with no error
# anywhere. Nothing upstream guarantees a chunk fits: RAW stores a document
# whole, and the AI chunker is *asked* for a size but can return anything.
# enforceEmbedLimit below is what actually makes it true.
MAX_EMBED_TOKENS = int(os.environ.get("RAG_MAX_EMBED_TOKENS", "2048"))

# How many sections may be in flight at the API at once. Enough to keep a
# large document from taking minutes of serial round trips, low enough not to
# look like a burst worth rate-limiting.
AI_CHUNK_CONCURRENCY = int(os.environ.get("RAG_AI_CHUNK_CONCURRENCY", "8"))

AI_CHUNK_PROMPT = (
    "Split the TEXT below into self-contained semantic chunks. Each chunk "
    "should cover one coherent idea and be no more than {maxTokens} tokens. "
    "Respond with ONLY a JSON array of strings (one chunk per string) -- no "
    "markdown fences, no other text.\n\nTEXT:\n{text}"
)


class IngestionError(Exception):
    """The document could not be loaded, chunked, or stored."""


class ChunkingStrategy(Enum):
    RAW = auto()  # store the whole document as one chunk, no splitting
    NON_AI = auto()  # fixed-size token windows with overlap
    AI = auto()  # the configured chunker model decides chunk boundaries


@dataclass
class Chunk:
    text: str
    index: int
    tokenCount: int


@dataclass
class IngestedDocument:
    ragDbId: str
    sourceUrl: str
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class SearchResult:
    """One chunk a search matched.

    ``score`` is comparable only within a single store: Pinecone returns a
    cosine similarity over embeddings, the offline stores return a lexical
    overlap. Ordering is meaningful either way; the absolute number is not.
    """

    text: str
    index: int
    score: float


# A query term worth matching on. Splitting on non-word characters rather
# than whitespace so "vector-database." and "vector database" agree.
_TERM_PATTERN = re.compile(r"\w+")


def lexicalSearch(chunks: list[Chunk], query: str, topK: int) -> list[SearchResult]:
    """Rank ``chunks`` against ``query`` by shared terms.

    What the offline stores use in place of embeddings. It is a keyword
    overlap and nothing more -- no synonyms, no semantics -- so it is good
    enough to prove a chunk was stored and is reachable, and not good enough
    to judge retrieval quality by. Longer chunks are divided down so a large
    chunk cannot outrank a focused one just by containing more words.
    """
    queryTerms = set(_TERM_PATTERN.findall(query.lower()))
    if not queryTerms:
        return []

    scored: list[SearchResult] = []
    for chunk in chunks:
        chunkTerms = _TERM_PATTERN.findall(chunk.text.lower())
        if not chunkTerms:
            continue
        overlap = queryTerms.intersection(chunkTerms)
        if not overlap:
            continue
        score = len(overlap) / math.sqrt(len(set(chunkTerms)))
        scored.append(SearchResult(text=chunk.text, index=chunk.index, score=score))

    scored.sort(key=lambda r: (-r.score, r.index))
    return scored[:topK]


@lru_cache(maxsize=1)
def tokenEncoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKEN_ENCODING_NAME)


def countTokens(text: str) -> int:
    return len(tokenEncoding().encode(text)) if text else 0


async def load(source: str) -> str:
    """Fetch ``source`` as plain text -- an http(s) URL or a local file path."""
    if urlparse(source).scheme in ("http", "https"):
        return await loadFromUrl(source)
    return loadFromFile(source)


async def loadFromUrl(url: str) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise IngestionError(f"Could not download '{url}': {exc}") from exc
    return response.text


def loadFromFile(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"Could not read '{path}': {exc}") from exc


def split(text: str) -> list[str]:
    """Split raw text into sections on blank lines (paragraph breaks)."""
    sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    if sections:
        return sections
    return [text.strip()] if text.strip() else []


def chunkWithoutAi(
    sections: list[str],
    *,
    chunkTokens: int = DEFAULT_CHUNK_TOKENS,
    overlapTokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Fixed-size token windows with overlap. No network calls, no cost."""
    # Each window starts ``chunkTokens - overlapTokens`` further on, so an
    # overlap at least as large as the window never advances: the loop below
    # would append the same tokens until memory ran out. A hung job is a
    # worse way to learn that than a failed one, and both numbers come from
    # configuration (RAG_MAX_EMBED_TOKENS among them), so this is reachable
    # without touching the code.
    if overlapTokens >= chunkTokens:
        raise IngestionError(
            f"Chunk overlap ({overlapTokens}) must be smaller than the chunk "
            f"size ({chunkTokens}); otherwise chunking cannot make progress."
        )

    encoding = tokenEncoding()
    chunks: list[str] = []
    for section in sections:
        tokens = encoding.encode(section)
        start = 0
        while start < len(tokens):
            end = min(start + chunkTokens, len(tokens))
            chunks.append(encoding.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start = end - overlapTokens
    return chunks


def buildChunks(texts: list[str]) -> list["Chunk"]:
    """Attach an index and a token count to every chunk.

    Split out of ``runText`` so the whole loop can be handed to a thread in one
    go: it is a tiktoken pass per chunk, which on a large corpus is tens of
    thousands of encodes and seconds of solid CPU.
    """
    return [
        Chunk(text=text, index=index, tokenCount=countTokens(text))
        for index, text in enumerate(texts)
    ]


def enforceEmbedLimit(chunks: list[str]) -> list[str]:
    """Re-split anything an embedder would silently truncate.

    A last line of defence rather than the main chunking step: whatever
    strategy produced these, a chunk over the limit reaches the store as a
    record whose tail is dropped on the way in, and nothing downstream can
    tell that happened.
    """
    limited: list[str] = []
    for chunk in chunks:
        if countTokens(chunk) <= MAX_EMBED_TOKENS:
            limited.append(chunk)
            continue
        logger.warning(
            "Chunk of %d tokens exceeds the %d-token embed limit; re-splitting.",
            countTokens(chunk),
            MAX_EMBED_TOKENS,
        )
        limited.extend(
            chunkWithoutAi(
                [chunk],
                chunkTokens=MAX_EMBED_TOKENS,
                overlapTokens=DEFAULT_CHUNK_OVERLAP_TOKENS,
            )
        )
    return limited


class ChunkList(BaseModel):
    """The shape the chunker is required to answer in.

    A wrapper around a bare list because a JSON schema needs an object at its
    root -- no provider will accept a top-level array as a structured-output
    schema, so the list is given a name and lives one level down.
    """

    chunks: list[str] = Field(
        description="The section split into self-contained chunks, in document order."
    )


@lru_cache(maxsize=1)
def chunkerRunnable():
    """The configured chunker model, bound to the ChunkList schema.

    Built once. ``chunkerModel`` is itself cached per (provider, model) because
    each client owns a connection pool, and ``with_structured_output`` is a thin
    wrapper -- but this is called once per *section*, of which a large corpus
    has thousands, so the wrapper is worth not rebuilding either.

    Structured output rather than "please answer in JSON" in the prompt: told
    only in words, a model asked to chunk real prose returns JSON containing
    invalid escapes -- a backslash from the source text passed through verbatim
    -- and the document is then chunked by the fallback path for reasons nobody
    can see. Asking the provider to constrain the output is what made that stop
    happening, and it survives the move between vendors because every provider
    this service supports implements it.
    """
    from app.agent.llmManager import chunkerModel

    return chunkerModel().with_structured_output(ChunkList)


def chunksFromAnswer(answer) -> list[str]:
    """The chunk list out of whatever structured output actually returned.

    Three shapes, because ``with_structured_output`` is not uniform across the
    providers this can be pointed at: the Pydantic object it promises, a plain
    dict from an integration that skips validation, and a raw string from one
    that fell back to text. The last is why ``parseAiChunks`` still exists.
    """
    if isinstance(answer, ChunkList):
        chunks = answer.chunks
    elif isinstance(answer, dict):
        chunks = answer.get("chunks") or []
    elif isinstance(answer, str):
        chunks = parseAiChunks(answer)
    else:
        chunks = getattr(answer, "chunks", None) or []

    return [str(chunk).strip() for chunk in chunks if str(chunk).strip()]


async def chunkWithAi(
    sections: list[str],
    *,
    chunker=None,
    maxTokens: int = DEFAULT_CHUNK_TOKENS,
) -> list[str]:
    """Ask the configured model to pick semantic chunk boundaries per section.

    Which model that is comes from ``[chunker]`` in ``config/models.toml`` --
    this used to call Gemini through its own SDK, with its own key and its own
    pinned model name, which made it the one AI call in the service that could
    not be moved without editing this file. It is now the same provider layer
    the agent, the reviewer and the summariser go through.

    One call per section, run several at a time. Sections are independent --
    each is chunked on its own -- so waiting for one before starting the next
    only adds latency: a 10k-token document is ~60 sections, which is a
    minute of round trips serially and a few seconds in parallel. The
    semaphore keeps that from becoming an unbounded burst of requests at the
    API on a large document.

    ``gather`` preserves order, so chunks come back in document order rather
    than completion order.

    ``chunker`` is injectable for tests, which is the only reason it is a
    parameter; nothing in the application passes it.
    """
    runnable = chunker or chunkerRunnable()
    semaphore = asyncio.Semaphore(AI_CHUNK_CONCURRENCY)

    async def chunkSection(section: str) -> list[str]:
        prompt = AI_CHUNK_PROMPT.format(maxTokens=maxTokens, text=section)
        async with semaphore:
            try:
                answer = await runnable.ainvoke([{"role": "user", "content": prompt}])
            except Exception as exc:
                # Deliberately fatal to the whole document rather than falling
                # back per section. A bad key, an exhausted quota or a retired
                # model fails every section identically, and quietly chunking
                # the entire corpus by the non-AI path would store a database
                # that looks successful and is worse than the one that was
                # asked for. The job goes FAILED and resubmitting is the retry
                # -- see app.jobs.job.runJob.
                raise IngestionError(f"AI chunking failed: {exc}") from exc

        try:
            chunks = chunksFromAnswer(answer)
        except IngestionError as exc:
            # One unusable answer should not lose a whole document. The
            # non-AI split is a worse chunk boundary, not a wrong one.
            logger.warning("Falling back to non-AI chunking for one section: %s", exc)
            return chunkWithoutAi([section], chunkTokens=maxTokens)

        if not chunks:
            logger.warning(
                "The chunker returned nothing for one section; falling back to "
                "non-AI chunking for it."
            )
            return chunkWithoutAi([section], chunkTokens=maxTokens)
        return chunks

    perSection = await asyncio.gather(*(chunkSection(s) for s in sections))
    return [chunk for section in perSection for chunk in section]


def parseAiChunks(raw: str | None) -> list[str]:
    """A JSON array of strings out of a plain-text answer.

    Only reached when structured output degraded to text. Kept because that is
    exactly the case where the answer is least likely to be clean -- a model
    answering in prose wraps its JSON in markdown fences.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            firstLine, text = text.split("\n", 1)
            if firstLine.strip().lower() not in ("", "json"):
                text = firstLine + "\n" + text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"The chunker returned unparsable chunks: {exc}") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise IngestionError("The chunker did not return a JSON array of strings.")

    return [item.strip() for item in parsed if item.strip()]


class ChunkStore(Protocol):
    """Where finished chunks end up, keyed by ``ragDbId``.

    Two implementations exist:
    ``app.stores.localChunkStore.LocalChunkStore``, which writes JSON files and
    refuses to construct without RAG_TEST_MODE, and
    ``app.stores.pineconeChunkStore.PineconeChunkStore`` for real use. Nothing in
    this module knows which one it has.
    """

    async def save(self, ragDbId: str, chunks: list[Chunk]) -> None: ...

    async def get(self, ragDbId: str) -> list[Chunk]: ...

    async def delete(self, ragDbId: str) -> None: ...

    async def search(self, ragDbId: str, query: str, topK: int) -> list[SearchResult]: ...


class RagIngestionPipeline:
    """load -> split -> chunk -> store."""

    def __init__(
        self,
        store: ChunkStore,
        aiChunker: Callable[[list[str]], Awaitable[list[str]]] | None = None,
    ) -> None:
        # Required, with no default. It used to fall back to a dict, which made
        # "I forgot to pass a store" indistinguishable from a successful
        # ingestion: chunks landed somewhere, the job went DONE, and the
        # vectors existed nowhere any later search could reach them.
        self.store = store
        # Injectable so a test can exercise the AI *path* -- selection,
        # chunking, storage -- without a per-section call to a paid API for
        # every section of a million-token corpus.
        self.aiChunker = aiChunker or chunkWithAi

    async def run(
        self,
        source: str,
        ragDbId: str,
        strategy: ChunkingStrategy = ChunkingStrategy.NON_AI,
    ) -> IngestedDocument:
        """Load ``source``, then chunk and store it."""
        text = await load(source)
        return await self.runText(text, ragDbId, sourceUrl=source, strategy=strategy)

    async def runText(
        self,
        text: str,
        ragDbId: str,
        *,
        sourceUrl: str = "",
        strategy: ChunkingStrategy = ChunkingStrategy.NON_AI,
    ) -> IngestedDocument:
        """Chunk and store text already in hand.

        The entry point for a caller that has extracted the text itself --
        ``load`` above only reads plain text, so anything that started as a
        PDF, docx, or archive has to come through here.
        """
        texts = await self.chunk(text, strategy)

        # Another full tokenisation pass, so it goes to a thread as well.
        chunks = await asyncio.to_thread(buildChunks, texts)
        await self.store.save(ragDbId, chunks)

        return IngestedDocument(ragDbId=ragDbId, sourceUrl=sourceUrl, chunks=chunks)

    async def chunk(self, text: str, strategy: ChunkingStrategy) -> list[str]:
        chunks = await self.chunkBy(text, strategy)
        # Off the loop: one tiktoken pass per chunk, and there can be tens of
        # thousands of them. See the note on threads in the module docstring.
        return await asyncio.to_thread(enforceEmbedLimit, chunks)

    async def chunkBy(self, text: str, strategy: ChunkingStrategy) -> list[str]:
        if strategy is ChunkingStrategy.RAW:
            # A strip on text already in memory. Not worth a thread hop.
            return [text.strip()] if text.strip() else []

        sections = await asyncio.to_thread(split, text)
        if strategy is ChunkingStrategy.AI:
            return await self.aiChunker(sections)
        return await asyncio.to_thread(chunkWithoutAi, sections)
