"""Conversation storage: turn indices, the summary watermarks, and the cache in front.

The storage tests run against real Firestore. What matters is that appending a
turn cannot lose an exchange -- which is a claim about a transaction, and a
claim no dict can be asked to support -- and that a summarised conversation stops
reading the documents it folded away, which is a claim about a range query
resolving with no composite index defined.

The cache tests below use a fake Firestore on purpose: there, the whole point
is proving Firestore is *not* reached, which needs a client that fails loudly if
it is.
"""

import asyncio

import fakeredis
import pytest

from app.stores.conversationStore import (
    ConversationMessage,
    ConversationStoreError,
    ConversationWindow,
    ContextEntry,
    FirestoreConversationStore,
    buildConversationStore,
)

PROMPT = "You are the handbook assistant."


@pytest.fixture
def projectId(scratch) -> str:
    return scratch.projectId("conversation")


@pytest.fixture
def conversation(conversations, projectId) -> ConversationWindow:
    return asyncio.run(
        conversations.createConversation(
            projectId=projectId, systemPrompt=PROMPT, ragDbId="handbook-abc123"
        )
    )


# --- storage ---------------------------------------------------------------


def testANewConversationStartsEmpty(conversation: ConversationWindow) -> None:
    assert conversation.messages == []
    assert conversation.context == []
    assert conversation.turnCount == 0
    assert conversation.systemPrompt == PROMPT


def testAnUnknownConversationIsNoneRatherThanAnError(conversations, projectId) -> None:
    """The caller turns None into a 404 and an exception into a degraded
    answer, so the two must not collapse into one."""
    assert asyncio.run(conversations.loadWindow(projectId, "never-existed")) is None


def testAConversationIsScopedToItsProject(conversations, conversation, scratch) -> None:
    other = scratch.projectId("other")

    assert asyncio.run(conversations.loadWindow(other, conversation.conversationId)) is None


def testAppendingStoresBothHalvesOfTheExchange(conversations, conversation, projectId) -> None:
    asyncio.run(
        conversations.appendTurn(
            window=conversation, question="Refunds?", answer="30 days.", context=[]
        )
    )

    stored = asyncio.run(conversations.loadWindow(projectId, conversation.conversationId))
    assert [(m.turnIndex, m.role, m.content) for m in stored.messages] == [
        (0, "user", "Refunds?"),
        (1, "assistant", "30 days."),
    ]
    assert stored.turnCount == 2


def testTurnIndicesKeepCountingAcrossTurns(conversations, conversation) -> None:
    """Indices come from the stored counter inside the transaction, not from
    the caller's window -- which is what stops two concurrent turns writing the
    same document id and losing one of them."""
    window = asyncio.run(
        conversations.appendTurn(
            window=conversation, question="One?", answer="A.", context=[]
        )
    )
    window = asyncio.run(
        conversations.appendTurn(window=window, question="Two?", answer="B.", context=[])
    )

    assert [m.turnIndex for m in window.messages] == [0, 1, 2, 3]
    assert window.turnCount == 4


def testRetrievalsAreStoredAgainstTheTurnThatCausedThem(
    conversations, conversation, projectId
) -> None:
    asyncio.run(
        conversations.appendTurn(
            window=conversation,
            question="Refunds?",
            answer="30 days.",
            context=[ContextEntry(0, 0, "refund window", ["Refunds: 30 days."])],
        )
    )

    stored = asyncio.run(conversations.loadWindow(projectId, conversation.conversationId))
    assert stored.context[0].turnIndex == 0
    assert stored.context[0].entryIndex == 0
    assert stored.context[0].passages == ["Refunds: 30 days."]
    assert stored.contextCount == 1


def testAppendingToAConversationThatIsNotThereIsAnError(conversations, projectId) -> None:
    ghost = ConversationWindow(conversationId="ghost", projectId=projectId, systemPrompt=PROMPT)

    with pytest.raises(ConversationStoreError):
        asyncio.run(conversations.appendTurn(window=ghost, question="?", answer="!", context=[]))


def testTheSystemPromptIsSnapshotted(conversations, conversation, projectId) -> None:
    """A prompt edited mid-conversation must not rewrite the instructions the
    earlier answers were given under."""
    asyncio.run(conversations.appendTurn(window=conversation, question="Q", answer="A", context=[]))

    loaded = asyncio.run(
        conversations.loadWindow(projectId, conversation.conversationId)
    )
    assert loaded.systemPrompt == PROMPT


def testASummaryReplacesWhatItCovers(conversations, conversation, projectId) -> None:
    """The point of a summary: the documents below the watermark are never read
    again, so a long conversation stops getting more expensive."""
    window = conversation
    for number in range(3):
        window = asyncio.run(
            conversations.appendTurn(
                window=window,
                question=f"Q{number}",
                answer=f"A{number}",
                context=[ContextEntry(0, 0, f"search {number}", ["passage"])],
            )
        )

    asyncio.run(
        conversations.saveSummary(
            projectId=projectId,
            conversationId=conversation.conversationId,
            summary="They asked three things.",
            throughTurn=4,
            throughContext=2,
        )
    )

    stored = asyncio.run(conversations.loadWindow(projectId, conversation.conversationId))
    assert stored.contextSummary == "They asked three things."
    assert [m.turnIndex for m in stored.messages] == [4, 5]
    assert [c.entryIndex for c in stored.context] == [2]
    # The counters still describe the whole conversation, not the surviving tail.
    assert stored.turnCount == 6


def testAppendingContinuesPastASummary(conversations, conversation, projectId) -> None:
    window = conversation
    for number in range(3):
        window = asyncio.run(
            conversations.appendTurn(
                window=window, question=f"Q{number}", answer=f"A{number}", context=[]
            )
        )
    asyncio.run(
        conversations.saveSummary(
            projectId=projectId,
            conversationId=conversation.conversationId,
            summary="earlier",
            throughTurn=4,
            throughContext=0,
        )
    )

    folded = asyncio.run(conversations.loadWindow(projectId, conversation.conversationId))
    after = asyncio.run(
        conversations.appendTurn(window=folded, question="Q3", answer="A3", context=[])
    )

    assert after.turnCount == 8
    assert after.contextSummary == "earlier"


@pytest.mark.slow
def testConcurrentTurnsDoNotLoseAnExchange(conversations, conversation, projectId) -> None:
    """The transaction is the only thing stopping two questions arriving
    together from both claiming turn 0 and one exchange vanishing."""

    async def race() -> None:
        await asyncio.gather(
            conversations.appendTurn(window=conversation, question="One?", answer="A.", context=[]),
            conversations.appendTurn(window=conversation, question="Two?", answer="B.", context=[]),
        )

    asyncio.run(race())

    stored = asyncio.run(conversations.loadWindow(projectId, conversation.conversationId))
    assert stored.turnCount == 4
    assert sorted(m.turnIndex for m in stored.messages) == [0, 1, 2, 3]


# --- the window itself -----------------------------------------------------


def testAWindowSurvivesAJsonRoundTrip() -> None:
    """It goes through Redis as JSON, so the dataclasses have to come back as
    dataclasses rather than as dicts."""
    window = ConversationWindow(
        conversationId="c1",
        projectId="p1",
        systemPrompt=PROMPT,
        contextSummary="earlier",
        turnCount=2,
        context=[ContextEntry(0, 0, "q", ["p"])],
        messages=[ConversationMessage(0, "user", "hello")],
    )

    restored = ConversationWindow.fromJson(window.toJson())

    assert restored == window
    assert isinstance(restored.messages[0], ConversationMessage)
    assert isinstance(restored.context[0], ContextEntry)


# --- the Redis cache in front of Firestore ---------------------------------
#
# Fake clients here, because the assertion is about what is *not* called.


class ExplodingFirestore:
    """Any use at all is a failure. Proves a cache hit never reaches Firestore."""

    def collection(self, name):  # pragma: no cover - only reached when the test fails
        raise AssertionError("Firestore was read on what should have been a cache hit.")


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def testACachedWindowNeverReachesFirestore(redis) -> None:
    store = FirestoreConversationStore(firestore=ExplodingFirestore(), redis=redis)
    window = ConversationWindow(
        conversationId="c1",
        projectId="p1",
        systemPrompt=PROMPT,
        messages=[ConversationMessage(0, "user", "hello")],
    )
    redis.set(store.cacheKey("p1", "c1"), window.toJson())

    assert asyncio.run(store.loadWindow("p1", "c1")) == window


def testAnUnreadableStoreRaisesRatherThanAnsweringEmpty(redis) -> None:
    """"Firestore is down" and "there is no such conversation" lead to a degraded
    answer and a 404 respectively, so they must not be the same value."""

    class BrokenFirestore:
        def collection(self, name):
            raise RuntimeError("unreachable")

    store = FirestoreConversationStore(firestore=BrokenFirestore(), redis=redis)

    with pytest.raises(ConversationStoreError):
        asyncio.run(store.loadWindow("p1", "c1"))


def testACacheFailureIsNotAnError() -> None:
    """Redis holds nothing that is not in Firestore, so a broken cache costs
    reads and nothing else."""

    class BrokenRedis:
        def get(self, key):
            raise RuntimeError("no redis")

        def set(self, *args, **kwargs):
            raise RuntimeError("no redis")

    class EmptyFirestore:
        def collection(self, name):
            return self

        def document(self, name):
            return self

        def get(self):
            return type("Snapshot", (), {"exists": False})()

    store = FirestoreConversationStore(firestore=EmptyFirestore(), redis=BrokenRedis())

    assert asyncio.run(store.loadWindow("p1", "c1")) is None


# --- the factory -----------------------------------------------------------


def testWithoutAGcpProjectTheStoreRefusesToBuild(monkeypatch) -> None:
    """No in-process fallback: one would drop every conversation on restart
    while the service looked perfectly healthy."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        buildConversationStore()
