"""What a job does to a document, and the processor contract that defines it.

``DocumentProcessor`` is what ``app.jobs.run_job`` executes; this module owns
that contract because the contract is about documents, not about how a job is
scheduled or tracked (that's ``app.job_manager``). ``DocumentAnalyzerProcessor``
is the real implementation: a job's ``document_link`` is downloaded, its format
is detected, and metadata is pulled out of it. A ``.zip`` or ``.rar`` is
unpacked and every supported member analyzed the same way, so its
``DocumentMetadata.source_kind`` is ``"folder"``; anything else is ``"single"``.

Unlike zip, rar has no format-decoding logic in Python's standard library --
``rarfile`` shells out to an ``unrar`` (or ``bsdtar``/``7z``) binary that must
already be on the machine. If it isn't, opening a ``.rar`` raises a clear
error rather than the library's own exception, which is otherwise easy to
mistake for a corrupt archive.

What "processing" means beyond metadata extraction -- chunking, embedding,
writing into a ``ragDbId`` -- is still undecided; this module only answers
"what did we just download".
"""

from __future__ import annotations

import csv
import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

import docx
import httpx
import pdfplumber
import rarfile

# Where to find the unrar/bsdtar/7z binary rarfile shells out to. Not on every
# machine by default the way zip support is -- override per-environment with
# RAG_UNRAR_TOOL if it isn't on PATH under its usual name.
rarfile.UNRAR_TOOL = os.environ.get("RAG_UNRAR_TOOL", rarfile.UNRAR_TOOL)

if TYPE_CHECKING:
    # Only needed for the type hint below -- importing it for real would make
    # app.jobs and app.documents import each other.
    from app.jobs import Job

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 60.0
# A generous ceiling for a personal project, not a production sizing decision.
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}
ARCHIVE_EXTENSIONS = {".zip", ".rar"}

# Used when the URL doesn't carry a recognizable extension (e.g. a bare
# `/download?id=123`), keyed on the response's Content-Type.
CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/vnd.rar": ".rar",
    "application/x-rar-compressed": ".rar",
}


class DownloadError(Exception):
    """The document could not be fetched."""


class UnsupportedDocumentError(Exception):
    """The file -- or a member of an archive -- is not one of the supported
    formats."""


class ArchiveToolMissingError(Exception):
    """The archive needs an external tool (e.g. unrar) that isn't available."""


@dataclass
class FileMetadata:
    """What we could tell about one file."""

    filename: str
    extension: str
    size_bytes: int
    table_count: int = 0
    image_count: int = 0
    page_count: int | None = None  # PDF
    word_count: int | None = None  # docx, txt, md
    line_count: int | None = None  # txt, md
    row_count: int | None = None  # csv
    column_count: int | None = None  # csv
    # Set instead of raising when one file (typically inside a zip) fails to
    # parse, so one bad member doesn't lose the metadata for the rest.
    error: str | None = None


@dataclass
class DocumentMetadata:
    """What we could tell about the thing that was downloaded.

    ``page_count``/``image_count``/``table_count`` are totals across every
    file -- including files nested arbitrarily deep inside archives-within-
    archives, not just the top level. Per-file detail is still in ``files``;
    these are the roll-up most callers actually want.
    """

    source_url: str
    source_kind: str  # "single" | "folder"
    content_type: str | None
    file_count: int
    total_size_bytes: int
    page_count: int
    image_count: int
    table_count: int
    files: list[FileMetadata] = field(default_factory=list)


def _extension_of(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def _filename_from_url(url: str) -> str:
    name = PurePosixPath(urlparse(url).path).name
    return name or "document"


async def download(url: str) -> tuple[bytes, str, str | None]:
    """Fetch ``url``. Returns the body, a filename to analyze it as, and the
    server's Content-Type (used as a fallback when the URL has no extension)."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        raise DownloadError(
                            f"Document exceeds the {MAX_DOWNLOAD_BYTES // (1024 * 1024)}MB limit."
                        )
                content_type = response.headers.get("content-type")
    except httpx.HTTPError as exc:
        raise DownloadError(f"Could not download '{url}': {exc}") from exc

    return bytes(body), _filename_from_url(url), content_type


def _resolve_extension(filename: str, content_type: str | None) -> str:
    extension = _extension_of(filename)
    if extension in SUPPORTED_EXTENSIONS or extension in ARCHIVE_EXTENSIONS:
        return extension

    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime in CONTENT_TYPE_EXTENSIONS:
            return CONTENT_TYPE_EXTENSIONS[mime]

    return extension  # Unrecognized either way; caller raises.


def analyze_bytes(filename: str, data: bytes, *, extension: str | None = None) -> FileMetadata:
    """Extract metadata from one file's contents.

    A parse failure is recorded on ``FileMetadata.error`` rather than raised,
    so analyzing a zip full of files doesn't abort on the first bad one.
    """
    extension = extension if extension is not None else _extension_of(filename)
    meta = FileMetadata(filename=filename, extension=extension, size_bytes=len(data))

    if extension not in SUPPORTED_EXTENSIONS:
        meta.error = f"Unsupported file type '{extension or filename}'."
        return meta

    try:
        if extension == ".pdf":
            _analyze_pdf(data, meta)
        elif extension == ".docx":
            _analyze_docx(data, meta)
        elif extension == ".csv":
            _analyze_csv(data, meta)
        else:  # .txt, .md
            _analyze_text(data, meta)
    except Exception as exc:
        logger.warning("Failed to analyze '%s': %s", filename, exc)
        meta.error = str(exc)

    return meta


def _analyze_pdf(data: bytes, meta: FileMetadata) -> None:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        meta.page_count = len(pdf.pages)
        meta.table_count = sum(len(page.find_tables()) for page in pdf.pages)
        meta.image_count = sum(len(page.images) for page in pdf.pages)


def _analyze_docx(data: bytes, meta: FileMetadata) -> None:
    document = docx.Document(io.BytesIO(data))
    meta.table_count = len(document.tables)
    meta.image_count = sum(1 for rel in document.part.rels.values() if "image" in rel.reltype)
    meta.word_count = sum(len(paragraph.text.split()) for paragraph in document.paragraphs)


def _analyze_csv(data: bytes, meta: FileMetadata) -> None:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
    meta.row_count = len(rows)
    meta.column_count = len(rows[0]) if rows else 0
    meta.table_count = 1 if rows else 0


def _analyze_text(data: bytes, meta: FileMetadata) -> None:
    text = data.decode("utf-8", errors="replace")
    meta.line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    meta.word_count = len(text.split())


# How many archive levels to unpack before giving up. Guards against a
# maliciously (or accidentally) deep chain of archives-within-archives eating
# unbounded memory/CPU -- five is far beyond anything a real submission needs.
MAX_ARCHIVE_DEPTH = 5


def _open_archive(extension: str, data: bytes):
    """Open a zip or rar so its members can be listed and read.

    ``zipfile.ZipFile`` and ``rarfile.RarFile`` expose the same
    ``infolist()`` / ``read(info)`` / ``is_dir()`` shape by design, so one
    analysis loop below handles either.
    """
    if extension == ".zip":
        return zipfile.ZipFile(io.BytesIO(data))
    return rarfile.RarFile(io.BytesIO(data))


def _analyze_archive_entries(
    extension: str, data: bytes, *, path_prefix: str, depth: int
) -> list[FileMetadata]:
    """Walk one archive, recursing into any member that is itself a zip/rar.

    Returns a flat list -- a nested member's ``filename`` carries the full
    path (``outer.zip/inner.rar/doc.pdf``), so the hierarchy stays visible
    without the result needing to be tree-shaped.
    """
    if depth > MAX_ARCHIVE_DEPTH:
        return [
            FileMetadata(
                filename=path_prefix.rstrip("/"),
                extension=extension,
                size_bytes=len(data),
                error=f"Archive nesting exceeds the {MAX_ARCHIVE_DEPTH}-level limit.",
            )
        ]

    files: list[FileMetadata] = []
    try:
        with _open_archive(extension, data) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename).name
                # Skip directory entries and junk archives love to carry
                # (__MACOSX/, .DS_Store, Thumbs.db) rather than reporting
                # them as unsupported files.
                if info.is_dir() or not name or name.startswith("."):
                    continue

                member_path = f"{path_prefix}{info.filename}"
                member_extension = _extension_of(name)
                member_bytes = archive.read(info)

                if member_extension in ARCHIVE_EXTENSIONS:
                    files.extend(
                        _analyze_archive_entries(
                            member_extension,
                            member_bytes,
                            path_prefix=f"{member_path}/",
                            depth=depth + 1,
                        )
                    )
                else:
                    files.append(analyze_bytes(member_path, member_bytes))
    except rarfile.RarExecError as exc:
        # Listing a rar's contents needs no external tool; reading a member's
        # bytes does. That means this can fail partway through the walk --
        # raised as one clear error instead of leaving `files` half-built.
        raise ArchiveToolMissingError(
            "Cannot read .rar archives: no unrar/bsdtar/7z tool found "
            f"(looked for '{rarfile.UNRAR_TOOL}'). Install one, or point "
            "RAG_UNRAR_TOOL at its path."
        ) from exc

    return files


def _analyze_archive(
    url: str, extension: str, data: bytes, content_type: str | None
) -> DocumentMetadata:
    files = _analyze_archive_entries(extension, data, path_prefix="", depth=0)

    return DocumentMetadata(
        source_url=url,
        source_kind="folder",
        content_type=content_type,
        file_count=len(files),
        total_size_bytes=sum(f.size_bytes for f in files),
        page_count=sum(f.page_count or 0 for f in files),
        image_count=sum(f.image_count for f in files),
        table_count=sum(f.table_count for f in files),
        files=files,
    )


def analyze(url: str, filename: str, data: bytes, content_type: str | None = None) -> DocumentMetadata:
    """Analyze a downloaded document. A zip or rar becomes a folder of files;
    anything else is a single file."""
    extension = _resolve_extension(filename, content_type)

    if extension in ARCHIVE_EXTENSIONS:
        return _analyze_archive(url, extension, data, content_type)

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported document type for '{filename}' "
            f"(content-type: {content_type or 'unknown'})."
        )

    file_meta = analyze_bytes(filename, data, extension=extension)
    return DocumentMetadata(
        source_url=url,
        source_kind="single",
        content_type=content_type,
        file_count=1,
        total_size_bytes=file_meta.size_bytes,
        page_count=file_meta.page_count or 0,
        image_count=file_meta.image_count,
        table_count=file_meta.table_count,
        files=[file_meta],
    )


class DocumentProcessor(Protocol):
    """The one thing a job needs done to it.

    Whatever an implementation does -- fetch the link, chunk it, embed it,
    write it to a ragDbId -- it should raise on failure and otherwise mutate
    nothing but ``job.detail`` / ``job.metadata``; ``app.jobs.run_job`` owns
    ``job.status``.
    """

    async def process(self, job: "Job") -> None: ...


class StubDocumentProcessor:
    """Placeholder: marks every job done immediately, does no real work."""

    async def process(self, job: "Job") -> None:
        job.detail = "Processing is not implemented yet."


class DocumentAnalyzerProcessor:
    """Downloads ``job.document_link``, analyzes it, and records the result.

    Any exception here -- a bad URL, an unsupported type, a network failure --
    is left to propagate; ``app.jobs.run_job`` is what catches it and marks
    the job failed.
    """

    async def process(self, job: "Job") -> None:
        data, filename, content_type = await download(job.document_link)
        metadata = analyze(job.document_link, filename, data, content_type)

        job.metadata = metadata
        job.detail = (
            f"Analyzed {metadata.file_count} file(s) "
            f"({metadata.source_kind}, {metadata.total_size_bytes} bytes) -- "
            f"{metadata.page_count} page(s), {metadata.image_count} image(s), "
            f"{metadata.table_count} table(s)."
        )
