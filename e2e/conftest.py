"""Fixtures for the live end-to-end suite: a running server, a corpus, a project.

**Nothing here is faked.** That is the whole point of this directory, and it is
why it is not under ``tests/``: ``tests/conftest.py`` patches the entire
session's Redis to ``fakeredis`` and hands out scratch ids, both of which would
be actively wrong here. The server under test is a *different process* with its
own real Redis, its own real Firestore and its own real model provider, and this
process must not be pretending about any of it.

The separation is enforced by ``testpaths = ["tests"]`` in ``pyproject.toml``:
``uv run pytest`` never reaches this directory, and ``uv run pytest e2e`` never
reaches the other one.

What has to be running before any of this works
-----------------------------------------------
Three processes, and they are yours to start -- these fixtures deliberately do
not launch them::

    uv run uvicorn app.main:app       the API
    python -m app.jobs.worker         the ingestion worker
    redis-server / docker compose     Redis

A fixture that started the server would be convenient exactly until something
broke, at which point it would hide which half broke. ``liveServer`` skips the
whole suite with an instruction instead.

Environment
-----------
``RAG_E2E_BASE_URL``     where the API is. Default ``http://127.0.0.1:8000``.
``RAG_E2E_DOCUMENT_URL`` a URL the *worker* can fetch the corpus from. Set this
                         when the worker is not on this machine -- see
                         ``corpusUrl``.
``RAG_E2E_PROJECT_ID``   reuse an already-ingested project and skip ingestion
                         entirely. The reason it exists is money: ingesting the
                         corpus is ~220 chunker calls, and iterating on the
                         conversation itself should not pay for that every run.
``RAG_E2E_SERVER_ID``    the caller label. Unverified by the service; a log line.
"""

from __future__ import annotations

import functools
import http.server
import os
import socketserver
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest

from harness import BASE_URL, SERVER_ID, apiUrl

DOCUMENTS = Path(__file__).parent / "documents"
CORPUS = "wanderlynTravel.txt"

# Ingestion of this corpus is AI chunking: ~220 blank-line sections, eight
# concurrent chunker calls. Minutes, not seconds, and slower when the provider
# is rate limiting. Ten minutes is not generous, it is realistic.
INGEST_TIMEOUT_SECONDS = float(os.environ.get("RAG_E2E_INGEST_TIMEOUT", 600))
INGEST_POLL_SECONDS = 5.0

# One turn is an agent run (possibly twice, if the review fails) plus a reviewer
# call, bounded server-side by RAG_ANSWER_TIMEOUT_SECONDS, which defaults to 120.
# The client timeout has to sit above that or it fails first and reports a
# timeout the server was about to answer.
TURN_TIMEOUT_SECONDS = float(os.environ.get("RAG_E2E_TURN_TIMEOUT", 180))


@pytest.fixture(scope="session")
def liveServer() -> str:
    """The base URL, once something has answered on it.

    Skips rather than fails. A missing server is a setup problem, not a defect
    in the code under test, and a red suite for "you did not start uvicorn"
    trains people to ignore red suites.
    """
    try:
        response = httpx.get(apiUrl("/health"), timeout=10)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(
            f"No API answering at {BASE_URL} ({exc}). Start it with "
            f"`uv run uvicorn app.main:app`, start the worker with "
            f"`python -m app.jobs.worker`, and make sure Redis is up. "
            f"Point somewhere else with RAG_E2E_BASE_URL."
        )
    return BASE_URL


@pytest.fixture(scope="session")
def httpClient(liveServer) -> httpx.Client:
    """One client for the session, with a timeout that outlasts a model call."""
    with httpx.Client(timeout=TURN_TIMEOUT_SECONDS) as client:
        yield client


@pytest.fixture(scope="session")
def corpusUrl() -> str:
    """A URL the corpus can be downloaded from, served from this process.

    The subtlety worth stating: the document is fetched by the **worker**, not
    by this test. So the URL has to be reachable from wherever the worker runs.
    A thread serving ``e2e/documents`` on an ephemeral port is right when the
    worker is a local process, and wrong the moment it is in a container, where
    ``127.0.0.1`` is the container itself.

    Hence ``RAG_E2E_DOCUMENT_URL``: point it at anything the worker can GET --
    a host-gateway address, a bucket, a gist -- and this server is not started
    at all. The ingest fixture repeats this in its failure message, because a
    download failure is otherwise a very confusing way to learn about it.
    """
    override = os.environ.get("RAG_E2E_DOCUMENT_URL")
    if override:
        yield override
        return

    handler = functools.partial(_QuietHandler, directory=str(DOCUMENTS))
    # Port 0 lets the OS pick, so two runs on one machine cannot collide, and
    # allow_reuse_address stops a TIME_WAIT socket from a previous run refusing
    # the bind.
    with socketserver.TCPServer(("0.0.0.0", 0), handler) as server:
        server.allow_reuse_address = True
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/{CORPUS}"
        finally:
            server.shutdown()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """The stdlib handler, minus the per-request line on stderr.

    Left in, it interleaves with pytest's own output for every retry the worker
    makes and makes a failing run harder to read than it needs to be.
    """

    def log_message(self, fmt, *args) -> None:  # noqa: A003 - stdlib signature
        pass


@pytest.fixture(scope="session")
def projectId() -> str:
    """A project id for this run.

    Fresh every run by default, so a half-ingested project from a failed run
    can never be silently reused -- which would produce a suite that passes
    against yesterday's index and tells you nothing about today's code.

    ``RAG_E2E_PROJECT_ID`` opts out of that deliberately, for the case where you
    are iterating on the conversation and do not want to pay for ingestion each
    time.
    """
    return os.environ.get("RAG_E2E_PROJECT_ID") or f"e2e-{uuid.uuid4().hex[:10]}-travel"


@pytest.fixture(scope="session")
def ingested(httpClient: httpx.Client, projectId: str, corpusUrl: str) -> str:
    """Ingest the corpus and block until the job is done. Yields the projectId.

    Session-scoped, because ingestion is the expensive part and every test in
    the suite asks questions of the same index.
    """
    if os.environ.get("RAG_E2E_PROJECT_ID"):
        # An explicitly named project is assumed already ingested. Confirm the
        # job actually finished rather than trusting it, because a project
        # pointed at a FAILED job would otherwise produce twenty answers that
        # all honestly say the documents do not cover it.
        status = _statusOf(httpClient, projectId)
        if status.get("status") != "done":
            pytest.fail(
                f"RAG_E2E_PROJECT_ID='{projectId}' names a project whose ingestion "
                f"is '{status.get('status')}', not 'done'. Unset it to ingest fresh."
            )
        return projectId

    response = httpClient.post(
        apiUrl("/api/v1/document"),
        json={"serverId": SERVER_ID, "documentLink": corpusUrl, "projectId": projectId},
    )
    assert response.status_code == 202, (
        f"Submitting the corpus failed: {response.status_code} {response.text}"
    )

    deadline = time.monotonic() + INGEST_TIMEOUT_SECONDS
    last = {}
    while time.monotonic() < deadline:
        last = _statusOf(httpClient, projectId)
        state = last.get("status")

        if state == "done":
            # RAW would mean the corpus fell under 2000 tokens and was stored
            # whole with no vector database behind it, and every question after
            # this would be answered from nothing. It is ~16.5k tokens today,
            # but a future edit could shrink it, and the failure would otherwise
            # look like a bad agent rather than a document that was never indexed.
            assert "documentLink" not in last, (
                "The corpus was small enough for the RAW strategy, so nothing was "
                "indexed and there is nothing to ask questions of. It needs to stay "
                "above 2000 tokens -- see app/ingestion/ragSelector.py."
            )
            return projectId

        if state == "failed":
            pytest.fail(
                f"Ingestion of '{projectId}' FAILED. Check the worker log. The two "
                f"usual causes are: the worker could not download {corpusUrl} (set "
                f"RAG_E2E_DOCUMENT_URL to something it can reach -- if the worker is "
                f"in Docker, 127.0.0.1 is the container, not this machine), or the "
                f"chunker's provider rejected the calls (a bad or exhausted key -- "
                f"one failed chunker call fails the whole job, by design)."
            )

        time.sleep(INGEST_POLL_SECONDS)

    pytest.fail(
        f"Ingestion of '{projectId}' was still '{last.get('status')}' after "
        f"{INGEST_TIMEOUT_SECONDS:.0f}s. AI chunking this corpus is ~220 provider "
        f"calls; raise RAG_E2E_INGEST_TIMEOUT if the provider is simply slow, and "
        f"check the worker is running at all if the status never left 'queued'."
    )


def _statusOf(client: httpx.Client, projectId: str) -> dict:
    response = client.post(
        apiUrl("/api/v1/document/status"),
        json={"serverId": SERVER_ID, "projectId": projectId},
    )
    if response.status_code == 404:
        return {"status": "unknown"}
    response.raise_for_status()
    return response.json()
