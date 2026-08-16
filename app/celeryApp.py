"""The Celery application: broker, queue, and the settings that matter here.

Ingestion is minutes of work -- download, extract, chunk (sometimes through
Gemini), then a few hundred Pinecone upserts -- kicked off by a request that
has to return in milliseconds. Running it in the API process means the API
cannot be redeployed or scaled without killing jobs mid-write, and cannot run
on anything request-scoped at all. This moves it to a worker.

Configuration is entirely from the environment, so nothing here has to change
between a laptop and a deployment::

    CELERY_BROKER_URL=redis://localhost:6379/0
    GCP_PROJECT_ID=<project>            # the job table; required alongside
    GOOGLE_APPLICATION_CREDENTIALS=...  # only outside GCP

Run a worker with::

    celery -A app.celeryApp worker --loglevel=info        # Linux/macOS
    celery -A app.celeryApp worker --loglevel=info --pool=solo   # Windows
"""

from __future__ import annotations

import json
import os

from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")

# Longer than any single ingestion should take: download is capped at
# DOWNLOAD_TIMEOUT_SECONDS, and the work after it is AI chunking plus batched
# upserts. The soft limit raises inside the task so runJob can mark the job
# failed with a reason; the hard limit kills a worker that ignored it.
SOFT_TIME_LIMIT_SECONDS = int(os.environ.get("CELERY_SOFT_TIME_LIMIT", 25 * 60))
HARD_TIME_LIMIT_SECONDS = int(os.environ.get("CELERY_TIME_LIMIT", 30 * 60))

# How long Redis waits before deciding a delivered message was never handled
# and giving it to someone else. This MUST exceed the hard time limit. If it
# does not, a job still legitimately running gets handed to a second worker,
# and the two ingest different halves into one namespace -- precisely the
# interleaving the conflict check in app.jobs exists to prevent, arriving by
# a route that bypasses it entirely.
VISIBILITY_TIMEOUT_SECONDS = int(
    os.environ.get("CELERY_VISIBILITY_TIMEOUT", HARD_TIME_LIMIT_SECONDS * 2)
)

# Extra broker settings as JSON, merged over the defaults below. Brokers other
# than Redis need their own (an SQS region, the spool directories the
# filesystem transport writes to), and none of them should require editing this
# file. Anything named here wins, including visibility_timeout -- read the
# warning above before overriding that one.
TRANSPORT_OPTIONS = json.loads(os.environ.get("CELERY_BROKER_TRANSPORT_OPTIONS", "{}"))

celeryApp = Celery("rag", broker=BROKER_URL, include=["app.celeryTasks"])

celeryApp.conf.update(
    # JSON only. Celery's pickle serializer executes what it deserializes, so
    # anything that can reach the broker can run code in a worker. Task
    # arguments here are a single ragDbId string; JSON is ample.
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # No result backend: the job's outcome belongs in the job table, which the
    # API already reads for /document/status. A second copy in Redis would be
    # one more thing to keep in step, and the one that expires.
    task_ignore_result=True,
    # Redeliver a job whose worker died rather than losing it. Safe because
    # ingestion is idempotent: chunk ids are derived from position, and a
    # re-run overwrites them and sweeps whatever the interrupted run left.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # One job in flight per worker process. The default prefetches four, which
    # for minutes-long tasks means three sit idle in a busy worker's buffer
    # while other workers have nothing to do.
    worker_prefetch_multiplier=1,
    task_soft_time_limit=SOFT_TIME_LIMIT_SECONDS,
    task_time_limit=HARD_TIME_LIMIT_SECONDS,
    broker_transport_options={
        "visibility_timeout": VISIBILITY_TIMEOUT_SECONDS,
        **TRANSPORT_OPTIONS,
    },
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    enable_utc=True,
)
