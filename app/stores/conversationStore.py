"""One conversation: the prompt it answers with, what it has already retrieved,
and the turns it has taken.

A conversation holds three things, and they are three different kinds of data:

* **``systemPrompt``** -- snapshotted onto the conversation when it is created, not
  re-resolved per turn. A project's prompt is editable (see
  ``app.agent.promptStore``), and a conversation whose instructions change
  underneath it half way through stops being one conversation: the model is
  asked to keep faith with earlier answers it would no longer have given. New
  conversations pick up the new prompt; running ones finish under the one they started.
* **``context``** -- the passages this conversation has already retrieved, one entry per
  search the agent ran. This is the expensive half. Re-deriving it means paying
  for the same vector searches on every follow-up, so it is stored and replayed
  into the prompt rather than fetched again. See
  ``app.agent.summariser.renderContext``.
* **``messages``** -- the turns themselves, user and assistant alternating.

Both of the last two grow without bound, which is what the summariser is for:
old context and old turns are folded into ``contextSummary`` and the documents
behind them stop being read. See ``app.agent.summariser``.

Layout
------
::

    ragConversations/{projectId}              one project's conversations
      conversations/{conversationId}          prompt, summary, counters
        messages/{turnIndex zero-padded}      one turn each
        context/{entryIndex zero-padded}      one retrieval each

**Subcollections, not fields on the conversation document.** A Firestore document is
capped at 1 MiB and every write rewrites the whole thing, so a ``conversationHistory``
array would make each turn cost the length of the conversation so far and would
hard-stop a few hundred turns in. Retrieved passages are worse again -- six
chunks a search, several searches a turn. One document per item makes appending
O(1) and unbounded, at the price of one read per item, which is exactly the
price the summariser exists to cap.

**Document ids are the zero-padded index, and the index is also a field.** The
id makes a write idempotent: turn 7 is always ``messages/000007``, so a retried
append overwrites rather than duplicating. The field is what queries use --
``where(turnIndex >= n)`` needs no composite index and no ordering trick, and
reads only the tail the summary does not already cover.

**Redis in front, Firestore behind**, the same split as ``app.agent.promptStore``:
Firestore is the record, Redis holds the assembled window so a follow-up is one
``GET`` instead of one read per message. Only a window that was actually
resolved is cached -- a Firestore failure degrades this turn, it does not get
written down and inflicted on the next hour of them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

COLLECTION = os.environ.get("FIRESTORE_CONVERSATIONS_COLLECTION", "ragConversations")

CACHE_PREFIX = os.environ.get("RAG_CONVERSATION_CACHE_PREFIX", "ragConversation:")

# How long an assembled window is served before Firestore is asked again. A
# generous value is safe here in a way it is not for prompts: every append
# rewrites the entry, so this only ever covers a conversation nobody has spoken in, and
# a miss costs reads rather than correctness.
CACHE_TTL_SECONDS = int(os.environ.get("RAG_CONVERSATION_CACHE_TTL_SECONDS", 3600))

# How long a conversation survives its last turn. Written onto every document as
# ``expiresAt`` for a Firestore TTL policy to act on -- **the field alone
# deletes nothing**; a TTL policy has to be created on each of the three
# collection groups (conversations, messages, context) for it to mean anything. Storing
# it regardless means turning expiry on later is a console change rather than a
# backfill over every conversation ever held. 90 days.
CONVERSATION_TTL_SECONDS = int(os.environ.get("RAG_CONVERSATION_TTL_SECONDS", 90 * 24 * 3600))

# Caps, so one enormous turn cannot walk a document up towards the 1 MiB limit
# and start failing every subsequent write to that conversation. Generous enough that
# no real answer is touched.
MAX_MESSAGE_CHARS = int(os.environ.get("RAG_CONVERSATION_MAX_MESSAGE_CHARS", 20000))
MAX_PASSAGE_CHARS = int(os.environ.get("RAG_CONVERSATION_MAX_PASSAGE_CHARS", 4000))

# What a conversation is listed under, and how much of the last answer is kept beside
# it, so a conversation list is one read per conversation rather than one subcollection scan.
TITLE_CHARS = 80


class ConversationStoreError(Exception):
    """The conversation store could not be reached.

    Distinct from "no such conversation", which is a ``None`` return. The caller turns
    the first into a degraded answer and the second into a 404, and it has to
    be able to tell them apart -- answering a real conversation as though it were empty
    silently drops a conversation the caller can see they were having.
    """


def newConversationId() -> str:
    """A fresh conversation id. Random, and not derived from anything.

    Same reasoning as ``app.stores.projectStore.newRagDbId``: an id computed
    from its contents is an id that can never be changed. Nothing may recompute
    one of these from a projectId, a question, or a timestamp.
    """
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expiry() -> datetime:
    return _now() + timedelta(seconds=CONVERSATION_TTL_SECONDS)


def _isoOf(value) -> str:
    """A stored timestamp as a string, whichever way it came back.

    Firestore hands back ``DatetimeWithNanoseconds``; Redis hands back the
    string this function already produced. Normalising here keeps every
    consumer -- and every JSON round trip through the cache -- dealing with one
    type instead of two.

    Note the asymmetry with ``app.stores.projectStore``, which stores ISO
    strings in Firestore. These documents are *queried* and *expired* by their
    timestamps, and both of those need a real Firestore timestamp, so the
    native type is what goes in and the string is what comes out.
    """
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


@dataclass(frozen=True)
class ConversationMessage:
    """One turn. ``reviewScore`` and ``retried`` are the assistant's only."""

    turnIndex: int
    role: str  # "user" | "assistant"
    content: str
    createdAt: str = ""
    reviewScore: float | None = None
    retried: bool = False


@dataclass(frozen=True)
class ContextEntry:
    """One retrieval the agent performed, and what came back.

    The passages are kept verbatim rather than as chunk ids. The whole point is
    that a follow-up never pays for this search again, and an id would mean a
    Pinecone round trip to turn it back into text -- which is precisely the cost
    being avoided. ``turnIndex`` is the turn that caused the search, so the
    summariser can fold a retrieval away at the same moment as the exchange it
    belongs to.
    """

    entryIndex: int
    turnIndex: int
    query: str
    passages: list[str]
    kind: str = "search"
    createdAt: str = ""


@dataclass
class ConversationWindow:
    """Everything one turn needs, assembled: prompt, summary, and live tail.

    ``context`` and ``messages`` hold only what the summary does not already
    cover -- ``summarisedThroughTurn`` and ``summarisedThroughContext`` are the
    watermarks, and the documents below them are never read again. That is what
    stops a long conversation costing more every time somebody adds to it.

    ``turnCount`` and ``contextCount`` are the *next* indices, not the number of
    items present here; the window is a tail, the counters describe the whole
    conversation.
    """

    conversationId: str
    projectId: str
    systemPrompt: str
    ragDbId: str | None = None
    contextSummary: str = ""
    summarisedThroughTurn: int = 0
    summarisedThroughContext: int = 0
    turnCount: int = 0
    contextCount: int = 0
    context: list[ContextEntry] = field(default_factory=list)
    messages: list[ConversationMessage] = field(default_factory=list)

    def toJson(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def fromJson(cls, raw: str) -> "ConversationWindow":
        data = json.loads(raw)
        return cls(
            **{
                **data,
                "context": [ContextEntry(**entry) for entry in data.get("context", [])],
                "messages": [
                    ConversationMessage(**message)
                    for message in data.get("messages", [])
                ],
            }
        )

    def copy(self) -> "ConversationWindow":
        """A detached copy, so a caller editing a window cannot edit the store."""
        return ConversationWindow.fromJson(self.toJson())


def _messageDocument(message: ConversationMessage, now: datetime) -> dict:
    return {
        "turnIndex": message.turnIndex,
        "role": message.role,
        "content": message.content[:MAX_MESSAGE_CHARS],
        "createdAt": now,
        "expiresAt": _expiry(),
        "reviewScore": message.reviewScore,
        "retried": message.retried,
    }


def _contextDocument(entry: ContextEntry, now: datetime) -> dict:
    return {
        "entryIndex": entry.entryIndex,
        "turnIndex": entry.turnIndex,
        "kind": entry.kind,
        "query": entry.query[:MAX_PASSAGE_CHARS],
        "passages": [passage[:MAX_PASSAGE_CHARS] for passage in entry.passages],
        "createdAt": now,
        "expiresAt": _expiry(),
    }


class ConversationStore(Protocol):
    """A conversation's storage, however it is backed.

    Four operations, which is the whole surface: start one, read the window a
    turn needs, append a completed turn, and replace the folded-away part with
    a summary.
    """

    async def createConversation(
        self, *, projectId: str, systemPrompt: str, ragDbId: str | None, title: str = ""
    ) -> ConversationWindow:
        """Bring a new conversation into existence and return its empty window."""
        ...

    async def loadWindow(self, projectId: str, conversationId: str) -> ConversationWindow | None:
        """The window for an existing conversation, or None if there is no such conversation.

        Raises ``ConversationStoreError`` when the store could not be reached -- a
        different answer from "no such conversation", and it must stay different.
        """
        ...

    async def appendTurn(
        self,
        *,
        window: ConversationWindow,
        question: str,
        answer: str,
        context: list[ContextEntry],
        reviewScore: float | None = None,
        retried: bool = False,
    ) -> ConversationWindow:
        """Record one completed exchange, and return the window after it."""
        ...

    async def saveSummary(
        self,
        *,
        projectId: str,
        conversationId: str,
        summary: str,
        throughTurn: int,
        throughContext: int,
    ) -> None:
        """Replace everything below the watermarks with ``summary``."""
        ...


class FirestoreConversationStore:
    """Conversations in Firestore, with the assembled window cached in Redis.

    Redis is optional and its failures are non-events: a miss costs reads, and
    a write that never lands costs the next turn the same reads. Firestore is
    not optional -- if it cannot be reached the caller is told, because quietly
    answering as though the conversation were empty loses it.
    """

    def __init__(self, firestore=None, redis=None) -> None:
        # Imported here, not at module scope, so importing this module needs no
        # credentials and opens no client. Same reasoning as
        # ``app.stores.projectStore.FirestoreProjectStore``.
        if firestore is None:
            from app.infra.firestoreClient import firestoreClient

            firestore = firestoreClient()
        self._db = firestore
        self._redis = redis

    # --- references -------------------------------------------------------

    def _conversationRef(self, projectId: str, conversationId: str):
        return (
            self._db.collection(COLLECTION)
            .document(projectId)
            .collection("conversations")
            .document(conversationId)
        )

    def cacheKey(self, projectId: str, conversationId: str) -> str:
        return f"{CACHE_PREFIX}{projectId}:{conversationId}"

    # --- creating ---------------------------------------------------------

    async def createConversation(
        self, *, projectId: str, systemPrompt: str, ragDbId: str | None, title: str = ""
    ) -> ConversationWindow:
        return await asyncio.to_thread(
            self._createConversation, projectId, systemPrompt, ragDbId, title
        )

    def _createConversation(
        self, projectId: str, systemPrompt: str, ragDbId: str | None, title: str
    ) -> ConversationWindow:
        window = ConversationWindow(
            conversationId=newConversationId(),
            projectId=projectId,
            systemPrompt=systemPrompt,
            ragDbId=ragDbId,
        )
        now = _now()

        # The parent document exists only so a project's conversations can be found by
        # listing rather than by collection-group query. Firestore is perfectly
        # happy to hold a subcollection under a document that was never
        # written, and such a conversation is reachable but invisible in the console.
        self._db.collection(COLLECTION).document(projectId).set(
            {"projectId": projectId, "updatedAt": now}, merge=True
        )

        self._conversationRef(projectId, window.conversationId).set(
            {
                "conversationId": window.conversationId,
                "projectId": projectId,
                # Snapshotted deliberately -- see the module docstring.
                "systemPrompt": systemPrompt,
                # Audit only. Retrieval resolves the ragDbId from the projectId
                # at the route on every request, and nothing may resolve it
                # from here: a conversation would otherwise outlive a rebuilt database
                # and keep searching a namespace the project no longer uses.
                "ragDbId": ragDbId,
                "title": title[:TITLE_CHARS],
                "lastMessage": "",
                "contextSummary": "",
                "summarisedThroughTurn": 0,
                "summarisedThroughContext": 0,
                "summarisedAt": None,
                "turnCount": 0,
                "contextCount": 0,
                "createdAt": now,
                "updatedAt": now,
                "expiresAt": _expiry(),
            }
        )
        return window

    # --- reading ----------------------------------------------------------

    async def loadWindow(self, projectId: str, conversationId: str) -> ConversationWindow | None:
        cached = await self._cacheGet(projectId, conversationId)
        if cached is not None:
            return cached

        try:
            window = await asyncio.to_thread(self._loadFromStore, projectId, conversationId)
        except Exception as exc:
            # Raised rather than swallowed into an empty window: the caller has
            # to be able to tell "Firestore is down, answer this turn without
            # its history" from "there is no such conversation, 404".
            raise ConversationStoreError(
                f"Could not read conversation '{conversationId}'."
            ) from exc

        if window is not None:
            await self._cacheSet(window)
        return window

    def _loadFromStore(self, projectId: str, conversationId: str) -> ConversationWindow | None:
        """The Firestore half: the conversation document, then only the live tail.

        Two range queries rather than two collection scans.
        ``summarisedThroughTurn`` and ``summarisedThroughContext`` are what keep
        this from growing -- everything below them is already inside
        ``contextSummary`` and its documents are never read again, however long
        the conversation runs.
        """
        from google.cloud.firestore_v1.base_query import FieldFilter

        reference = self._conversationRef(projectId, conversationId)
        snapshot = reference.get()
        if not snapshot.exists:
            return None

        data = snapshot.to_dict() or {}
        throughTurn = int(data.get("summarisedThroughTurn") or 0)
        throughContext = int(data.get("summarisedThroughContext") or 0)

        messages = [
            ConversationMessage(
                turnIndex=int(raw.get("turnIndex", 0)),
                role=raw.get("role", "user"),
                content=raw.get("content", ""),
                createdAt=_isoOf(raw.get("createdAt")),
                reviewScore=raw.get("reviewScore"),
                retried=bool(raw.get("retried", False)),
            )
            for raw in (
                document.to_dict() or {}
                for document in reference.collection("messages")
                .where(filter=FieldFilter("turnIndex", ">=", throughTurn))
                .order_by("turnIndex")
                .stream()
            )
        ]

        context = [
            ContextEntry(
                entryIndex=int(raw.get("entryIndex", 0)),
                turnIndex=int(raw.get("turnIndex", 0)),
                query=raw.get("query", ""),
                passages=list(raw.get("passages") or []),
                kind=raw.get("kind", "search"),
                createdAt=_isoOf(raw.get("createdAt")),
            )
            for raw in (
                document.to_dict() or {}
                for document in reference.collection("context")
                .where(filter=FieldFilter("entryIndex", ">=", throughContext))
                .order_by("entryIndex")
                .stream()
            )
        ]

        return ConversationWindow(
            conversationId=conversationId,
            projectId=projectId,
            systemPrompt=data.get("systemPrompt", ""),
            ragDbId=data.get("ragDbId"),
            contextSummary=data.get("contextSummary") or "",
            summarisedThroughTurn=throughTurn,
            summarisedThroughContext=throughContext,
            turnCount=int(data.get("turnCount") or 0),
            contextCount=int(data.get("contextCount") or 0),
            context=context,
            messages=messages,
        )

    # --- writing ----------------------------------------------------------

    async def appendTurn(
        self,
        *,
        window: ConversationWindow,
        question: str,
        answer: str,
        context: list[ContextEntry],
        reviewScore: float | None = None,
        retried: bool = False,
    ) -> ConversationWindow:
        updated = await asyncio.to_thread(
            self._appendTurn, window, question, answer, context, reviewScore, retried
        )
        await self._cacheSet(updated)
        return updated

    def _appendTurn(
        self,
        window: ConversationWindow,
        question: str,
        answer: str,
        context: list[ContextEntry],
        reviewScore: float | None,
        retried: bool,
    ) -> ConversationWindow:
        """One transaction: both messages, every retrieval, and the counters.

        A transaction and not a batch, because the indices come from the conversation
        document's own counters -- two questions sent into one conversation at the same
        moment would otherwise both read ``turnCount`` as 6, both write
        ``messages/000006``, and one exchange would simply vanish. Reading the
        counter inside the transaction makes the second attempt see the first's
        write and retry against 8.

        Atomic also means a conversation is never left showing a question with no
        answer, or an answer whose retrieved passages did not survive beside it.
        """
        from firebase_admin import firestore

        reference = self._conversationRef(window.projectId, window.conversationId)

        @firestore.transactional
        def appendInTransaction(transaction) -> ConversationWindow:
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists:
                raise ConversationStoreError(
                    f"No conversation '{window.conversationId}' in project '{window.projectId}'."
                )

            data = snapshot.to_dict() or {}
            turnIndex = int(data.get("turnCount") or 0)
            entryIndex = int(data.get("contextCount") or 0)
            now = _now()

            userTurn = ConversationMessage(
                turnIndex=turnIndex,
                role="user",
                content=question[:MAX_MESSAGE_CHARS],
                createdAt=_isoOf(now),
            )
            assistantTurn = ConversationMessage(
                turnIndex=turnIndex + 1,
                role="assistant",
                content=answer[:MAX_MESSAGE_CHARS],
                createdAt=_isoOf(now),
                reviewScore=reviewScore,
                retried=retried,
            )
            for message in (userTurn, assistantTurn):
                transaction.set(
                    reference.collection("messages").document(f"{message.turnIndex:06d}"),
                    _messageDocument(message, now),
                )

            stored: list[ContextEntry] = []
            for offset, entry in enumerate(context):
                recorded = ContextEntry(
                    entryIndex=entryIndex + offset,
                    # The turn that caused the retrieval, so the summariser can
                    # fold a search away at the same time as the exchange it
                    # was run for. Keeping passages whose question has already
                    # been summarised away is how a "window" stops being one.
                    turnIndex=turnIndex,
                    query=entry.query,
                    passages=[p[:MAX_PASSAGE_CHARS] for p in entry.passages],
                    kind=entry.kind,
                    createdAt=_isoOf(now),
                )
                transaction.set(
                    reference.collection("context").document(f"{recorded.entryIndex:06d}"),
                    _contextDocument(recorded, now),
                )
                stored.append(recorded)

            transaction.update(
                reference,
                {
                    "turnCount": turnIndex + 2,
                    "contextCount": entryIndex + len(context),
                    "updatedAt": now,
                    "expiresAt": _expiry(),
                    # A conversation list wants a line per conversation, not a subcollection
                    # read per conversation. Both of these are denormalised for that.
                    "lastMessage": answer[:TITLE_CHARS],
                    **({"title": question[:TITLE_CHARS]} if not data.get("title") else {}),
                },
            )

            return ConversationWindow(
                conversationId=window.conversationId,
                projectId=window.projectId,
                systemPrompt=data.get("systemPrompt") or window.systemPrompt,
                ragDbId=data.get("ragDbId", window.ragDbId),
                contextSummary=data.get("contextSummary") or "",
                summarisedThroughTurn=int(data.get("summarisedThroughTurn") or 0),
                summarisedThroughContext=int(data.get("summarisedThroughContext") or 0),
                turnCount=turnIndex + 2,
                contextCount=entryIndex + len(context),
                context=[*window.context, *stored],
                messages=[*window.messages, userTurn, assistantTurn],
            )

        return appendInTransaction(self._db.transaction())

    async def saveSummary(
        self,
        *,
        projectId: str,
        conversationId: str,
        summary: str,
        throughTurn: int,
        throughContext: int,
    ) -> None:
        await asyncio.to_thread(
            self._saveSummary, projectId, conversationId, summary, throughTurn, throughContext
        )
        # Dropped rather than rewritten. The caller is holding the window it
        # just summarised and is about to append a turn to it, and writing a
        # window here would race that append. The next turn re-reads -- and
        # re-reading is now cheap, which is the entire point of having written
        # the summary.
        await self._cacheDelete(projectId, conversationId)

    def _saveSummary(
        self,
        projectId: str,
        conversationId: str,
        summary: str,
        throughTurn: int,
        throughContext: int,
    ) -> None:
        self._conversationRef(projectId, conversationId).update(
            {
                "contextSummary": summary,
                "summarisedThroughTurn": throughTurn,
                "summarisedThroughContext": throughContext,
                "summarisedAt": _now(),
            }
        )

    # --- the cache --------------------------------------------------------
    #
    # Every method here swallows its own failures. Redis holds nothing that is
    # not in Firestore, so the worst a broken cache can do is make a turn cost
    # the reads it would have cost anyway.

    async def _cacheGet(self, projectId: str, conversationId: str) -> ConversationWindow | None:
        if self._redis is None:
            return None
        try:
            raw = await asyncio.to_thread(self._redis.get, self.cacheKey(projectId, conversationId))
            return ConversationWindow.fromJson(raw) if raw else None
        except Exception:
            logger.warning(
                "Conversation cache read failed for '%s'.", conversationId, exc_info=True
            )
            return None

    async def _cacheSet(self, window: ConversationWindow) -> None:
        if self._redis is None:
            return
        try:
            await asyncio.to_thread(
                self._redis.set,
                self.cacheKey(window.projectId, window.conversationId),
                window.toJson(),
                ex=CACHE_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Conversation cache write failed for '%s'.",
                window.conversationId,
                exc_info=True,
            )

    async def _cacheDelete(self, projectId: str, conversationId: str) -> None:
        if self._redis is None:
            return
        try:
            await asyncio.to_thread(self._redis.delete, self.cacheKey(projectId, conversationId))
        except Exception:
            logger.warning(
                "Conversation cache drop failed for '%s'.", conversationId, exc_info=True
            )


def buildConversationStore() -> ConversationStore:
    """The conversation store. Firestore, or nothing.

    No in-process fallback, for the same reason as
    ``app.stores.projectStore.buildProjectStore``: a conversation that dies with
    the process is not a cheaper version of one that does not, it is a service
    that silently forgets every conversation on each restart while looking
    perfectly healthy. Tests run against real Firestore as well, which is the
    only way the append transaction and the summary range queries are ever
    actually exercised.

    Redis is different, and stays optional: it caches the assembled window, so
    losing it costs reads rather than conversations. That is worth degrading
    for; forgetting the conversation is not.
    """
    if not os.environ.get("GCP_PROJECT_ID"):
        raise RuntimeError(
            "GCP_PROJECT_ID is not set, so conversation history has nowhere to live. Set it "
            "(and GOOGLE_APPLICATION_CREDENTIALS outside GCP) -- there is no "
            "in-process substitute, because one would drop every conversation on "
            "restart without anything looking wrong."
        )

    from app.infra.redisClient import redisClient

    redis = redisClient()
    if redis is None:
        logger.info(
            "No REDIS_URL: every question re-reads its conversation from Firestore."
        )
    return FirestoreConversationStore(redis=redis)


@lru_cache(maxsize=1)
def getConversationStore() -> ConversationStore:
    """FastAPI dependency: the process-wide conversation store.

    One instance for the life of the process, because it holds the Firestore
    and Redis clients and both own connection pools -- building one per request
    would open a new pool per request and leak connections until the server
    refused them.
    """
    return buildConversationStore()
