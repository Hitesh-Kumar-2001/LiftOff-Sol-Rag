"""What each project has spent, per conversation and per message.

Layout
------
::

    ragUsage/{projectId}                        the project's running total
      conversations/{conversationId}            the conversation's total, by role
        messages/{messageId}                    one answered question

Three levels because three questions get asked of this data and each wants a
different one: *what is this customer costing* (project), *is this one
conversation running away* (conversation), and *what did that particular answer
cost* (message). Rolling up from the leaves every time would mean reading every
message a project has ever produced to answer the first one.

**The rollups are atomic increments, not read-modify-write.** Two questions
answered in the same conversation at the same moment would otherwise both read
the old total and both write their own, and one would vanish. ``Increment`` is
applied by the server, so concurrent writers add up. It is also why there is no
transaction here: increments do not need one, and a transaction would serialise
every answer in a busy project for no benefit.

**A message document is written once and never incremented.** It is the record
of one answer, keyed by that answer's turn index, so a redelivered or retried
write overwrites it with the same thing rather than doubling it. The rollups
cannot offer that guarantee -- an increment applied twice counts twice -- which
is the honest trade for having them be atomic. See ``recordTurn``.

Nothing here may fail a request
-------------------------------
The model call has already been made and paid for by the time any of this runs.
An accounting write that does not land costs a number in a report; raising here
would cost the caller an answer they are owed. Every method swallows and logs,
the same rule the conversation store follows for its own writes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_USAGE_COLLECTION", "ragUsage")

CONVERSATIONS = "conversations"
MESSAGES = "messages"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UsageStore(Protocol):
    """Where the cost of an answer is written, and how it is read back."""

    async def recordTurn(
        self,
        *,
        projectId: str,
        conversationId: str,
        messageId: str,
        byRole: dict[str, dict],
        channel: str = "web",
        externalMessageId: str = "",
    ) -> None: ...

    async def conversationTotal(self, projectId: str, conversationId: str) -> dict | None: ...

    async def projectTotal(self, projectId: str) -> dict | None: ...


class FirestoreUsageStore:
    """The Firestore implementation."""

    def __init__(self, firestore=None) -> None:
        if firestore is None:
            from app.infra.firestoreClient import firestoreClient

            firestore = firestoreClient()
        self._db = firestore

    def _projectRef(self, projectId: str):
        return self._db.collection(COLLECTION).document(projectId)

    def _conversationRef(self, projectId: str, conversationId: str):
        return self._projectRef(projectId).collection(CONVERSATIONS).document(conversationId)

    # --- writing ----------------------------------------------------------

    async def recordTurn(
        self,
        *,
        projectId: str,
        conversationId: str,
        messageId: str,
        byRole: dict[str, dict],
        channel: str = "web",
        externalMessageId: str = "",
    ) -> None:
        """Store one answer's cost and roll it up. Never raises."""
        if not byRole:
            # Every model was stubbed, or nothing reported usage. Writing an
            # empty document would put a zero-token turn in the record and make
            # "how many answers did this cost nothing" a real question.
            return
        try:
            await asyncio.to_thread(
                self._recordTurn,
                projectId,
                conversationId,
                messageId,
                byRole,
                channel,
                externalMessageId,
            )
        except Exception:
            logger.warning(
                "Could not record usage for %s/%s/%s.",
                projectId,
                conversationId,
                messageId,
                exc_info=True,
            )

    def _recordTurn(
        self,
        projectId: str,
        conversationId: str,
        messageId: str,
        byRole: dict[str, dict],
        channel: str,
        externalMessageId: str,
    ) -> None:
        from google.cloud.firestore_v1 import Increment

        now = _now()
        totals = _sum(byRole)

        # The leaf: set, not merged, so re-recording the same turn is idempotent.
        self._conversationRef(projectId, conversationId).collection(MESSAGES).document(
            messageId
        ).set(
            {
                "messageId": messageId,
                "conversationId": conversationId,
                "projectId": projectId,
                # Which gateway the question came in on, and the platform's own
                # id for it where there is one -- so a WhatsApp bill can be
                # traced back to the message the customer actually sent.
                "channel": channel,
                "externalMessageId": externalMessageId,
                "roles": byRole,
                **totals,
                "createdAt": now,
            }
        )

        rollup = {name: Increment(value) for name, value in totals.items()}
        rollup["turns"] = Increment(1)

        # Per-role rollups carry the provider and model as plain values beside
        # the counters, so a report says "3.1M tokens on openai/gpt-5.6-luna"
        # rather than a number with no idea what produced it. They are
        # overwritten rather than incremented, which is correct: they describe
        # the most recent answer's configuration, and a change of model is
        # exactly what somebody reading a jump in cost needs to see.
        roleFields: dict[str, object] = {}
        for role, row in byRole.items():
            for name, value in row.items():
                if isinstance(value, (int, float)):
                    roleFields[f"roles.{role}.{name}"] = Increment(value)
                else:
                    roleFields[f"roles.{role}.{name}"] = value

        self._conversationRef(projectId, conversationId).set(
            {"projectId": projectId, "conversationId": conversationId, "createdAt": now},
            merge=True,
        )
        self._conversationRef(projectId, conversationId).update(
            {**rollup, **roleFields, "updatedAt": now, "channel": channel}
        )

        self._projectRef(projectId).set(
            {"projectId": projectId, "createdAt": now}, merge=True
        )
        self._projectRef(projectId).update({**rollup, **roleFields, "updatedAt": now})

    # --- reading ----------------------------------------------------------

    async def conversationTotal(self, projectId: str, conversationId: str) -> dict | None:
        return await asyncio.to_thread(
            self._read, self._conversationRef(projectId, conversationId)
        )

    async def projectTotal(self, projectId: str) -> dict | None:
        return await asyncio.to_thread(self._read, self._projectRef(projectId))

    def _read(self, reference) -> dict | None:
        snapshot = reference.get()
        return snapshot.to_dict() if snapshot.exists else None


def _sum(byRole: dict[str, dict]) -> dict[str, int]:
    """The counters that roll up. Only the numeric ones, and only these names."""
    names = (
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cachedInputTokens",
        "reasoningTokens",
    )
    return {name: sum(int(row.get(name) or 0) for row in byRole.values()) for name in names}


def buildUsageStore() -> UsageStore:
    """The usage store. Firestore, or nothing.

    Same rule as the other stores: an in-process substitute would lose the
    record of what every conversation cost on each restart, which for something
    whose entire purpose is a running total is worse than not having it.
    """
    if not os.environ.get("GCP_PROJECT_ID"):
        raise RuntimeError(
            "GCP_PROJECT_ID is not set, so token usage has nowhere to live. Set it "
            "(and GOOGLE_APPLICATION_CREDENTIALS outside GCP) -- there is no "
            "in-process substitute."
        )
    return FirestoreUsageStore()


@lru_cache(maxsize=1)
def getUsageStore() -> UsageStore:
    """FastAPI dependency: the process-wide usage store."""
    return buildUsageStore()
