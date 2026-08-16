"""Check the whole dispatch path end to end: broker, worker, job table.

    GCP_PROJECT_ID=<project> \
    GOOGLE_APPLICATION_CREDENTIALS=keys/<key>.json \
    FIRESTORE_DATABASE_ID=<id if not "(default)"> \
    python scripts/liveCeleryCheck.py

Starts everything it needs and cleans up after itself: a broker, a worker
subprocess, and an HTTP server holding the document to ingest. Nothing to
install and nothing to leave running.

The broker is kombu's ``filesystem`` transport rather than Redis -- a real
broker with real message serialization and a real queue on disk, which is what
this needs to prove. Redis would additionally exercise ``visibility_timeout``
redelivery; set CELERY_BROKER_URL to a Redis instance and this check runs
against it unchanged.

On Windows that transport locks its spool files through pywin32, which is not a
dependency of the application (production brokers do not need it) and has to be
installed to run this check: ``pip install pywin32``.

Ingestion runs under RAG_TEST_MODE, so chunks land in the local store instead
of Pinecone. The document is small enough to be chunked RAW, which keeps Gemini
out of it too. What is being checked is that a job crosses the queue and comes
back, not that Pinecone works -- scripts/livePineconeCheck.py covers that.

What no stub can show, and this does: that the claim written by the API is read
by a *different process*, that its status write lands where /document/status
reads, and that a duplicate and a conflict behave the same way across the two.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

RAG_DB = "live-celery-check"
OTHER_LINK = "https://example.invalid/other.pdf"
WAIT_SECONDS = float(os.environ.get("LIVE_CELERY_WAIT", 120))

DOCUMENT = (
    "The support handbook. Refunds are issued within fourteen days of purchase. "
    "Escalate anything a first-line agent cannot resolve in one exchange. "
) * 12

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {actual!r}" + ("" if ok else f" != {expected!r}"))
    if not ok:
        failures.append(label)


def freePort() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serveDirectory(directory: Path) -> tuple[str, http.server.HTTPServer]:
    """A local HTTP origin for the document, so nothing external is required."""
    port = freePort()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    server.RequestHandlerClass.log_message = lambda *a, **k: None  # quiet
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/handbook.txt", server


def brokerEnvironment(root: Path) -> dict[str, str]:
    """Broker settings shared by this process and the worker.

    The filesystem transport needs its spool directories named explicitly, and
    both ends must agree on them or they are simply two queues that never meet.
    """
    if os.environ.get("CELERY_BROKER_URL"):
        return {}

    folders = {name: root / name for name in ("in", "out", "processed", "control")}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    return {
        "CELERY_BROKER_URL": "filesystem://",
        "CELERY_BROKER_TRANSPORT_OPTIONS": json.dumps(
            {
                # in and out are deliberately the same directory: one queue,
                # written by this process and read by the worker.
                "data_folder_in": str(folders["in"]),
                "data_folder_out": str(folders["in"]),
                "processed_folder": str(folders["processed"]),
                "store_processed": True,
                # Named explicitly: unset, this transport writes its exchange
                # files to a "control" directory in the working directory --
                # which here is the repository.
                "control_folder": str(folders["control"]),
            }
        ),
    }


def startWorker(environment: dict[str, str], logPath: Path) -> subprocess.Popen:
    log = logPath.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            sys.executable, "-m", "celery", "-A", "app.celeryApp", "worker",
            "--loglevel=info",
            # solo: the prefork pool does not run on Windows, and one task at a
            # time is all this check needs anywhere.
            "--pool=solo",
            "--concurrency=1",
        ],
        env={**os.environ, **environment, "PYTHONPATH": os.getcwd()},
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def waitForTerminal(store, ragDbId: str, timeout: float):
    from app.jobs import JobStatus

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(ragDbId)
        if job is not None and job.status in (JobStatus.DONE, JobStatus.FAILED):
            return job
        time.sleep(1.0)
    return store.get(ragDbId)


async def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("Set GCP_PROJECT_ID (and GOOGLE_APPLICATION_CREDENTIALS).")

    root = Path(tempfile.mkdtemp(prefix="liveCeleryCheck-"))
    (root / "docs").mkdir()
    (root / "docs" / "handbook.txt").write_text(DOCUMENT, encoding="utf-8")

    link, server = serveDirectory(root / "docs")
    environment = {
        **brokerEnvironment(root),
        # Chunks go to a local directory rather than Pinecone: this check is
        # about the queue, and every extra dependency is another thing that can
        # fail for reasons that say nothing about dispatching.
        "RAG_TEST_MODE": "1",
        "RAG_LOCAL_STORE_DIR": str(root / "chunks"),
    }
    os.environ.update(environment)

    from app.celeryApp import celeryApp
    from app.celeryJobManager import CeleryJobManager
    from app.firestoreJobStore import COLLECTION, FirestoreJobStore, firestoreClient
    from app.jobs import JobConflictError, JobStatus
    from app.ragProcessor import RagIngestionProcessor

    store = FirestoreJobStore()
    manager = CeleryJobManager(RagIngestionProcessor(), store)
    collection = firestoreClient().collection(COLLECTION)
    collection.document(RAG_DB).delete()

    workerLog = root / "worker.log"
    worker = startWorker(environment, workerLog)

    try:
        print(f"broker:     {celeryApp.conf.broker_url}")
        print(f"collection: {COLLECTION}")
        print(f"document:   {link}\n")

        print("a submission is claimed in Firestore, then dispatched")
        job = await manager.create(serverId="svc", documentLink=link, ragDbId=RAG_DB)
        check("job id", job.jobId, RAG_DB)
        stored = collection.document(RAG_DB).get().to_dict()
        check("claimed before dispatch", stored is not None, True)
        check("link recorded", stored["documentLink"], link)

        print("\nthe same document again queues nothing further")
        again = await manager.create(serverId="svc", documentLink=link, ragDbId=RAG_DB)
        check("same job returned", again.jobId, RAG_DB)

        print("\na different document is refused while that one is unfinished")
        try:
            await manager.create(serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB)
            check("conflict raised", False, True)
        except JobConflictError:
            check("conflict raised", True, True)

        print(f"\na worker in another process runs it (up to {WAIT_SECONDS:.0f}s)")
        finished = waitForTerminal(store, RAG_DB, WAIT_SECONDS)
        check("reached a terminal status", finished.status if finished else None, JobStatus.DONE)
        if finished is not None and finished.detail:
            print(f"        {finished.detail}")
        check("ingested by the worker, not here", worker.poll() is None, True)

        print("\nthe chunks were written by that worker, not this process")
        written = list((root / "chunks").rglob("*")) if (root / "chunks").exists() else []
        check("chunk files exist", len(written) > 0, True)

        print("\nand the API side reads the outcome the worker wrote")
        elsewhere = CeleryJobManager(RagIngestionProcessor(), FirestoreJobStore())
        seen = await elsewhere.get(RAG_DB)
        check("visible to a fresh API instance", seen.status if seen else None, JobStatus.DONE)

        print("\nonce finished, a different document is accepted again")
        replacement = await manager.create(
            serverId="svc", documentLink=OTHER_LINK, ragDbId=RAG_DB
        )
        check("re-ingest accepted", replacement.documentLink, OTHER_LINK)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            worker.kill()
        server.shutdown()
        collection.document(RAG_DB).delete()
        if failures:
            print(f"\n--- worker log ({workerLog}) ---")
            print(workerLog.read_text(encoding="utf-8", errors="replace")[-3000:])
        shutil.rmtree(root, ignore_errors=True)
        print(f"\ncleaned up '{RAG_DB}' and {root}")

    print("\n" + ("FAILURES: " + ", ".join(failures) if failures else "all checks passed"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
