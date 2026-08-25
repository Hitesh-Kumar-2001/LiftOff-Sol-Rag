"""Load -> split -> chunk -> store.

The loader here reads plain text over http(s) or from a local path; it does
not (yet) share app.ingestion.documents.download, which already handles PDFs/docx/csv/
archives -- a document downloaded there for metadata still gets re-fetched
here as raw text, so a PDF's bytes are not usefully chunked yet.

Chunks are stored under a caller-supplied ``ragDbId`` -- the id of the RAG
database being populated, not the source URL -- so re-ingesting a different
document into the same ``ragDbId`` overwrites what's there, same as a job
resubmission under an existing ``ragDbId`` (see ``app.jobs.jobManager``).
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
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
# Flash-lite: the cheapest tier, and picking chunk boundaries does not need
# more. Pinned rather than `gemini-flash-latest` so a model retirement or a
# behaviour change is something we opt into -- gemini-2.0-flash was the
# default here until Google retired it out from under this code.
GEMINI_MODEL = os.environ.get("RAG_GEMINI_MODEL", "gemini-3.5-flash-lite")

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
    AI = auto()  # Gemini Flash decides chunk boundaries


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


@lru_cache(maxsize=1)
def geminiClient() -> genai.Client:
    apiKey = os.environ.get(ENV_GEMINI_API_KEY)
    if not apiKey:
        raise IngestionError(f"{ENV_GEMINI_API_KEY} is not set.")
    return genai.Client(api_key=apiKey)


async def chunkWithAi(
    sections: list[str],
    *,
    model: str = GEMINI_MODEL,
    maxTokens: int = DEFAULT_CHUNK_TOKENS,
) -> list[str]:
    """Ask Gemini Flash to pick semantic chunk boundaries within each section.

    One call per section, run several at a time. Sections are independent --
    each is chunked on its own -- so waiting for one before starting the next
    only adds latency: a 10k-token document is ~60 sections, which is a
    minute of round trips serially and a few seconds in parallel. The
    semaphore keeps that from becoming an unbounded burst of requests at the
    API on a large document.

    ``gather`` preserves order, so chunks come back in document order rather
    than completion order.
    """
    client = geminiClient()
    semaphore = asyncio.Semaphore(AI_CHUNK_CONCURRENCY)

    async def chunkSection(section: str) -> list[str]:
        prompt = AI_CHUNK_PROMPT.format(maxTokens=maxTokens, text=section)
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    # Ask the API to constrain the output rather than trusting
                    # the prompt to. Told only in words, the model answers
                    # real prose with JSON containing invalid escapes -- a
                    # backslash from the source text passed through verbatim.
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[str],
                    ),
                )
            except Exception as exc:
                raise IngestionError(f"Gemini chunking failed: {exc}") from exc

        try:
            return parseAiChunks(response.text)
        except IngestionError as exc:
            # One unusable answer should not lose a whole document. The
            # non-AI split is a worse chunk boundary, not a wrong one.
            logger.warning("Falling back to non-AI chunking for one section: %s", exc)
            return chunkWithoutAi([section], chunkTokens=maxTokens)

    perSection = await asyncio.gather(*(chunkSection(s) for s in sections))
    return [chunk for section in perSection for chunk in section]


def parseAiChunks(raw: str | None) -> list[str]:
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
        raise IngestionError(f"Gemini returned unparsable chunks: {exc}") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise IngestionError("Gemini did not return a JSON array of strings.")

    return [item.strip() for item in parsed if item.strip()]


class ChunkStore(Protocol):
    """Where finished chunks end up, keyed by ``ragDbId``.

    Three implementations exist: ``InMemoryChunkStore`` here,
    ``app.stores.localChunkStore.LocalChunkStore`` for tests, and
    ``app.stores.pineconeChunkStore.PineconeChunkStore`` for real use. Nothing in
    this module knows which one it has.
    """

    async def save(self, ragDbId: str, chunks: list[Chunk]) -> None: ...

    async def get(self, ragDbId: str) -> list[Chunk]: ...

    async def delete(self, ragDbId: str) -> None: ...

    async def search(self, ragDbId: str, query: str, topK: int) -> list[SearchResult]: ...


class InMemoryChunkStore:
    def __init__(self) -> None:
        self.byRagDbId: dict[str, list[Chunk]] = {}

    async def save(self, ragDbId: str, chunks: list[Chunk]) -> None:
        self.byRagDbId[ragDbId] = chunks

    async def get(self, ragDbId: str) -> list[Chunk]:
        return self.byRagDbId.get(ragDbId, [])

    async def delete(self, ragDbId: str) -> None:
        self.byRagDbId.pop(ragDbId, None)

    async def search(self, ragDbId: str, query: str, topK: int = 5) -> list[SearchResult]:
        return lexicalSearch(await self.get(ragDbId), query, topK)


class RagIngestionPipeline:
    """load -> split -> chunk -> store."""

    def __init__(
        self,
        store: ChunkStore | None = None,
        aiChunker: Callable[[list[str]], Awaitable[list[str]]] | None = None,
    ) -> None:
        self.store = store or InMemoryChunkStore()
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

        chunks = [
            Chunk(text=chunkText, index=i, tokenCount=countTokens(chunkText))
            for i, chunkText in enumerate(texts)
        ]
        await self.store.save(ragDbId, chunks)

        return IngestedDocument(ragDbId=ragDbId, sourceUrl=sourceUrl, chunks=chunks)

    async def chunk(self, text: str, strategy: ChunkingStrategy) -> list[str]:
        chunks = await self.chunkBy(text, strategy)
        return enforceEmbedLimit(chunks)

    async def chunkBy(self, text: str, strategy: ChunkingStrategy) -> list[str]:
        if strategy is ChunkingStrategy.RAW:
            return [text.strip()] if text.strip() else []

        sections = split(text)
        if strategy is ChunkingStrategy.AI:
            return await self.aiChunker(sections)
        return chunkWithoutAi(sections)
