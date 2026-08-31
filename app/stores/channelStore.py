"""Which messaging gateways a project is reachable through, and their settings.

Layout
------
::

    ragChannels/{projectId}                    one document per project
      { channels: { whatsapp: {...}, line: {...} }, updatedAt }
      threads/{channel}:{userId}               -> { conversationId }

**One document, with a map per channel** -- not a subcollection. A project has a
handful of gateways at most and each config is a few short strings, so the whole
set fits far inside the 1 MiB limit and a webhook needs exactly one read to
answer "is this project on WhatsApp, and what is its app secret?". A
subcollection would turn that into a read per channel for no benefit; contrast
``app.stores.conversationStore``, where the growth is unbounded and
subcollections are the only workable shape.

The ``threads`` subcollection is the other half, and it does need to be a
subcollection: one entry per *person* who has ever messaged this project, which
grows without limit. It maps a platform user to the conversation their history
lives in, so a WhatsApp number that asked something last week is answered with
that context rather than from nothing.

Credentials
-----------
**These are secrets sitting in a database with no field-level encryption.** An
access token here is enough to send messages as the customer's business, and a
Firestore reader can read every one of them. That is acceptable only because
this service is not exposed yet and no real credential has been stored. Before
it is: put them in Secret Manager and store a resource name here instead, and
lock this collection with a rules deny-all -- the same standing gap as the rest
of the service's security posture, but with a sharper edge, because these
credentials belong to somebody else.

Nothing here has a default and nothing is invented. A project with no document,
or a channel absent from its map, is simply not reachable that way, and its
webhook is refused.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Protocol

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_CHANNELS_COLLECTION", "ragChannels")

THREADS = "threads"


class ChannelStoreError(Exception):
    """The channel configuration could not be read."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChannelStore(Protocol):
    """Per-project gateway configuration, and who is talking through it."""

    async def configFor(self, projectId: str, channel: str) -> dict[str, Any] | None: ...

    async def allChannels(self, projectId: str) -> dict[str, dict[str, Any]]: ...

    async def saveConfig(self, projectId: str, channel: str, config: dict[str, Any]) -> None: ...

    async def conversationFor(self, projectId: str, threadKey: str) -> str | None: ...

    async def linkConversation(
        self, projectId: str, threadKey: str, conversationId: str
    ) -> None: ...


class FirestoreChannelStore:
    """The Firestore implementation. Firestore, or nothing."""

    def __init__(self, firestore=None) -> None:
        # Imported here so importing this module needs no credentials and opens
        # no client -- same reasoning as the other stores.
        if firestore is None:
            from app.infra.firestoreClient import firestoreClient

            firestore = firestoreClient()
        self._db = firestore

    def _projectRef(self, projectId: str):
        return self._db.collection(COLLECTION).document(projectId)

    # --- configuration ----------------------------------------------------

    async def configFor(self, projectId: str, channel: str) -> dict[str, Any] | None:
        """One gateway's settings, or None if this project is not on it."""
        channels = await self.allChannels(projectId)
        config = channels.get(channel)
        return dict(config) if isinstance(config, dict) else None

    async def allChannels(self, projectId: str) -> dict[str, dict[str, Any]]:
        try:
            return await asyncio.to_thread(self._readChannels, projectId)
        except Exception as exc:
            # Raised rather than answered as "no channels". The difference
            # matters at the webhook: no channels is a 404 the caller should
            # act on, an unreadable store is a 503 they should retry -- and a
            # platform that gets a 404 may disable the webhook entirely.
            raise ChannelStoreError(
                f"Could not read the channel configuration for '{projectId}'."
            ) from exc

    def _readChannels(self, projectId: str) -> dict[str, dict[str, Any]]:
        snapshot = self._projectRef(projectId).get()
        if not snapshot.exists:
            return {}
        channels = (snapshot.to_dict() or {}).get("channels") or {}
        return {
            name: dict(config)
            for name, config in channels.items()
            if isinstance(config, dict)
        }

    async def saveConfig(self, projectId: str, channel: str, config: dict[str, Any]) -> None:
        """Add or replace one gateway's settings, leaving the others alone.

        There is no endpoint behind this yet -- configuration is written by hand
        or by a script, like the system prompts in ``app.agent.promptStore``.
        It exists because a test needs to set one up, and because the shape of
        the write is part of the schema.
        """
        await asyncio.to_thread(self._saveConfig, projectId, channel, config)

    def _saveConfig(self, projectId: str, channel: str, config: dict[str, Any]) -> None:
        # merge=True with a nested key, so adding LINE cannot erase WhatsApp.
        self._projectRef(projectId).set(
            {"projectId": projectId, "channels": {channel: config}, "updatedAt": _now()},
            merge=True,
        )

    # --- who is talking ---------------------------------------------------

    async def conversationFor(self, projectId: str, threadKey: str) -> str | None:
        """The conversation this platform user's history lives in, if any."""
        try:
            return await asyncio.to_thread(self._readThread, projectId, threadKey)
        except Exception:
            # Not fatal, unlike the config read. Losing the link costs this
            # person their history for one message -- the answer still goes out,
            # from an empty conversation -- where failing would cost them the
            # answer entirely.
            logger.exception(
                "Could not read the conversation link for %s on '%s'; answering "
                "without history.",
                threadKey,
                projectId,
            )
            return None

    def _readThread(self, projectId: str, threadKey: str) -> str | None:
        snapshot = self._projectRef(projectId).collection(THREADS).document(threadKey).get()
        if not snapshot.exists:
            return None
        return (snapshot.to_dict() or {}).get("conversationId") or None

    async def linkConversation(
        self, projectId: str, threadKey: str, conversationId: str
    ) -> None:
        """Remember which conversation to continue for this person. Never raises."""
        try:
            await asyncio.to_thread(self._linkConversation, projectId, threadKey, conversationId)
        except Exception:
            logger.exception(
                "Could not link %s on '%s' to conversation '%s'; their next "
                "message will start a new one.",
                threadKey,
                projectId,
                conversationId,
            )

    def _linkConversation(self, projectId: str, threadKey: str, conversationId: str) -> None:
        self._projectRef(projectId).collection(THREADS).document(threadKey).set(
            {
                "threadKey": threadKey,
                "conversationId": conversationId,
                "updatedAt": _now(),
            }
        )


def buildChannelStore() -> ChannelStore:
    """The channel store. Firestore, or nothing.

    Same rule as the project and conversation stores: an in-process substitute
    would drop every gateway's credentials on restart, so a deployment would
    come back up looking healthy and refusing every webhook.
    """
    if not os.environ.get("GCP_PROJECT_ID"):
        raise RuntimeError(
            "GCP_PROJECT_ID is not set, so messaging channel configuration has "
            "nowhere to live. Set it (and GOOGLE_APPLICATION_CREDENTIALS outside "
            "GCP) -- there is no in-process substitute."
        )
    return FirestoreChannelStore()


@lru_cache(maxsize=1)
def getChannelStore() -> ChannelStore:
    """FastAPI dependency: the process-wide channel store.

    One instance, because it holds the Firestore client and that owns a
    connection pool.
    """
    return buildChannelStore()
