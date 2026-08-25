"""Reaching Redis: one client per process, built from ``REDIS_URL``.

Redis holds the working state -- the job table (``app.jobs.redisJobStore``) and the
queue the worker reads (``app.jobs.jobQueue``). Whether it is configured at all is
what decides how this deployment runs, so ``redisClient`` returning ``None`` is
a meaningful answer rather than a failure: see ``app.jobs.jobManager.buildJobManager``.

``decode_responses`` is on, so everything above this reads ``str`` and no
module has to remember which redis-py calls hand back bytes.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from redis import Redis

logger = logging.getLogger(__name__)

ENV_REDIS_URL = "REDIS_URL"

# Fail rather than hang when Redis is unreachable. The API claims and enqueues
# inside a request that has to answer in milliseconds; waiting out a default
# socket timeout there turns one unreachable dependency into a pile of stuck
# requests. A failure surfaces as a 503 and tells the caller to resubmit.
SOCKET_TIMEOUT_SECONDS = float(os.environ.get("RAG_REDIS_TIMEOUT", 5))


@lru_cache(maxsize=1)
def redisClient() -> Redis | None:
    """The process-wide Redis client, or None if no URL is configured.

    Cached because a ``Redis`` instance owns a connection pool: building one
    per call would open a new pool per request and leak connections until the
    server refused them.
    """
    url = os.environ.get(ENV_REDIS_URL, "").strip()
    if not url:
        return None

    return Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
        health_check_interval=30,
    )
