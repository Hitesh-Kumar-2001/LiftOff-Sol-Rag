"""System prompts: written in Firestore, read from Redis.

Two Firestore collections, deliberately separate:

* ``systemPrompts/{promptId}`` -> ``{"prompt": "..."}`` -- the text, stored once
  however many projects use it.
* ``projectPrompts/{projectId}`` -> ``{"promptId": "..."}`` -- which prompt a
  project answers with.

Splitting them is what lets twenty projects share one prompt and be re-tuned by
editing a single document. Storing the text against each project instead would
mean twenty edits and nineteen chances to miss one.

**Redis holds the resolved text, not the two hops.** The cache key is the
*projectId*, and its value is the finished prompt -- so the hot path is a single
``GET`` rather than a lookup for the id followed by a lookup for the text. The
cost is that one prompt shared by twenty projects occupies twenty small cache
entries, and that editing it takes up to ``PROMPT_TTL_SECONDS`` to be seen
everywhere. Prompts are a few kilobytes and change rarely; a round trip is paid
on every question. ``invalidate`` is there for when the wait is not acceptable.

Nothing is cached in process memory. A second API instance would not see the
first's invalidation, and an operator editing a prompt would have to guess how
many processes were holding a stale copy and restart each of them.

Only a prompt that was actually *resolved* is cached. Every failure here still
answers with ``DEFAULT_SYSTEM_PROMPT`` -- a prompt lookup must not fail a
question -- but a default that stood in for an unreachable Firestore is never
written to the cache, or a momentary outage would hold a project on the wrong
prompt for a full TTL.
"""

from __future__ import annotations

import asyncio
import logging
import os

from redis import Redis

logger = logging.getLogger(__name__)

PROMPT_COLLECTION = os.environ.get("FIRESTORE_PROMPTS_COLLECTION", "systemPrompts")
PROJECT_PROMPT_COLLECTION = os.environ.get(
    "FIRESTORE_PROJECT_PROMPTS_COLLECTION", "projectPrompts"
)

CACHE_PREFIX = os.environ.get("RAG_PROMPT_CACHE_PREFIX", "ragPrompt:")

# How long a resolved prompt is served before Firestore is asked again. This is
# the window in which an edited prompt is still answered with the old text.
PROMPT_TTL_SECONDS = int(os.environ.get("RAG_PROMPT_TTL_SECONDS", 3600))

# Answers a project that has never been given a prompt of its own. Without it
# every new project would fail its first question, which is a poor introduction
# to the service.
DEFAULT_SYSTEM_PROMPT = os.environ.get("RAG_DEFAULT_SYSTEM_PROMPT") or (
    "You are a retrieval-grounded assistant for one project's documents.\n"
    "Answer from what the project's own documents say. Use the searchProject "
    "tool to find them before answering anything factual -- do not answer from "
    "memory when the answer should come from the documents.\n"
    "If the documents do not contain the answer, say so plainly rather than "
    "guessing, and say what they do cover. Keep answers direct and specific."
)


class PromptStore:
    """Resolved system prompts, cached in Redis and stored in Firestore."""

    def __init__(self, redis: Redis | None = None, firestore=None) -> None:
        self._redis = redis
        self._firestore = firestore

    # --- reading ----------------------------------------------------------

    def cacheKey(self, projectId: str) -> str:
        return f"{CACHE_PREFIX}{projectId}"

    async def systemPromptFor(self, projectId: str) -> str:
        """The prompt this project answers with.

        Falls back to ``DEFAULT_SYSTEM_PROMPT`` when the project has no prompt
        assigned, or when the id it is assigned no longer exists -- a dangling
        assignment should degrade to a working assistant, not a 500.
        """
        cached = await self._cacheGet(projectId)
        if cached is not None:
            return cached

        resolved = await asyncio.to_thread(self._resolveFromStore, projectId)
        if resolved is None:
            # Firestore could not be read. Answering with the default is right;
            # *caching* it is not -- a two-second outage would otherwise pin the
            # default onto this project for the next PROMPT_TTL_SECONDS, so a
            # blip nobody noticed would quietly cost an hour of wrong prompts.
            return DEFAULT_SYSTEM_PROMPT

        await self._cacheSet(projectId, resolved)
        return resolved

    def _resolveFromStore(self, projectId: str) -> str | None:
        """The two Firestore hops. Synchronous -- the client is.

        Returns None when the store could not be read at all. That is a
        different thing from "this project has no prompt", which resolves to
        the default and is worth caching; see the caller.
        """
        if self._firestore is None:
            return DEFAULT_SYSTEM_PROMPT

        try:
            assignment = (
                self._firestore.collection(PROJECT_PROMPT_COLLECTION)
                .document(projectId)
                .get()
            )
            if not assignment.exists:
                return DEFAULT_SYSTEM_PROMPT

            promptId = (assignment.to_dict() or {}).get("promptId")
            if not promptId:
                return DEFAULT_SYSTEM_PROMPT

            document = (
                self._firestore.collection(PROMPT_COLLECTION).document(promptId).get()
            )
            if not document.exists:
                logger.warning(
                    "Project '%s' points at prompt '%s', which does not exist.",
                    projectId,
                    promptId,
                )
                return DEFAULT_SYSTEM_PROMPT

            return (document.to_dict() or {}).get("prompt") or DEFAULT_SYSTEM_PROMPT
        except Exception:
            # A prompt lookup failing should not fail the question. The default
            # prompt still produces a grounded answer; a 500 produces nothing.
            logger.exception("Could not read the system prompt for '%s'.", projectId)
            return None

    async def _cacheGet(self, projectId: str) -> str | None:
        if self._redis is None:
            return None
        try:
            return await asyncio.to_thread(self._redis.get, self.cacheKey(projectId))
        except Exception:
            # A cache miss is the safe failure: it costs two Firestore reads.
            logger.warning("Prompt cache read failed for '%s'.", projectId, exc_info=True)
            return None

    async def _cacheSet(self, projectId: str, prompt: str) -> None:
        if self._redis is None:
            return
        try:
            await asyncio.to_thread(
                self._redis.set, self.cacheKey(projectId), prompt, ex=PROMPT_TTL_SECONDS
            )
        except Exception:
            logger.warning("Prompt cache write failed for '%s'.", projectId, exc_info=True)

    # --- writing ----------------------------------------------------------

    async def savePrompt(self, promptId: str, prompt: str) -> None:
        """Create or replace one prompt's text.

        Does not invalidate the projects using it -- there is no index from a
        prompt back to its projects, and building one to support an edit that
        happens rarely is not worth the write on every assignment. They pick it
        up within PROMPT_TTL_SECONDS.
        """
        if self._firestore is None:
            raise RuntimeError("No Firestore configured; set GCP_PROJECT_ID.")
        await asyncio.to_thread(
            self._firestore.collection(PROMPT_COLLECTION).document(promptId).set,
            {"promptId": promptId, "prompt": prompt},
        )

    async def assignPrompt(self, projectId: str, promptId: str) -> None:
        """Point a project at a prompt, and drop its cached text immediately.

        Unlike editing a prompt's text, this one *is* invalidated eagerly: it is
        a deliberate switch, and waiting an hour to see it would read as the
        call having silently failed.
        """
        if self._firestore is None:
            raise RuntimeError("No Firestore configured; set GCP_PROJECT_ID.")
        await asyncio.to_thread(
            self._firestore.collection(PROJECT_PROMPT_COLLECTION).document(projectId).set,
            {"projectId": projectId, "promptId": promptId},
        )
        await self.invalidate(projectId)

    async def invalidate(self, projectId: str) -> None:
        """Drop a project's cached prompt so the next question re-reads it."""
        if self._redis is None:
            return
        try:
            await asyncio.to_thread(self._redis.delete, self.cacheKey(projectId))
        except Exception:
            logger.warning(
                "Could not invalidate the prompt cache for '%s'; it will expire in "
                "at most %ds.",
                projectId,
                PROMPT_TTL_SECONDS,
                exc_info=True,
            )


def buildPromptStore() -> PromptStore:
    """The prompt store for this process.

    Both halves are optional and degrade rather than fail: no Firestore means
    every project gets the default prompt, no Redis means every question pays
    the Firestore reads. Neither is a reason to refuse to start.
    """
    from app.infra.redisClient import redisClient

    redis = redisClient()
    firestore = None
    if os.environ.get("GCP_PROJECT_ID"):
        from app.infra.firestoreClient import firestoreClient

        firestore = firestoreClient()
    else:
        logger.info("No GCP_PROJECT_ID: every project will use the default system prompt.")

    if redis is None:
        logger.warning(
            "No REDIS_URL: system prompts are read from Firestore on every question."
        )

    return PromptStore(redis=redis, firestore=firestore)


_STORE: PromptStore | None = None


def getPromptStore() -> PromptStore:
    """FastAPI dependency: the process-wide prompt store.

    One instance because it holds the Redis and Firestore clients, both of which
    own connection pools. The prompts themselves are not held here -- see the
    module docstring.
    """
    global _STORE
    if _STORE is None:
        _STORE = buildPromptStore()
    return _STORE
