"""Chat storage: turn indices, the summary watermarks, and the cache in front.

The storage tests run against real Firestore. What matters is that appending a
turn cannot lose an exchange -- which is a claim about a transaction, and a
claim no dict can be asked to support -- and that a summarised chat stops
reading the documents it folded away, which is a claim about a range query
resolving with no composite index defined.

The cache tests below use a fake Firestore on purpose: there, the whole point
is proving Firestore is *not* reached, which needs a client that fails loudly if
it is.
"""

import asyncio

import fakeredis
import pytest

from app.stores.chatStore import (
    ChatMessage,
    ChatStoreError,
    ChatWindow,
    ContextEntry,
    FirestoreChatStore,
    buildChatStore,
)

PROMPT = "You are the handbook assistant."


@pytest.fixture
def projectId(scratch) -> str:
    return scratch.projectId("chat")


@pytest.fixture
def chat(chats, projectId) -> ChatWindow:
    return asyncio.run(
        chats.createChat(projectId=projectId, systemPrompt=PROMPT, ragDbId="handbook-abc123")
    )


# --- storage ---------------------------------------------------------------


def testANewChatStartsEmpty(chat: ChatWindow) -> None:
    assert chat.messages == []
    assert chat.context == []
    assert chat.turnCount == 0
    assert chat.systemPrompt == PROMPT


def testAnUnknownChatIsNoneRatherThanAnError(chats, projectId) -> None:
    """The caller turns None into a 404 and an exception into a degraded
    answer, so the two must not collapse into one."""
    assert asyncio.run(chats.loadWindow(projectId, "never-existed")) is None


def testAChatIsScopedToItsProject(chats, chat, scratch) -> None:
    other = scratch.projectId("other")

    assert asyncio.run(chats.loadWindow(other, chat.chatId)) is None


def testAppendingStoresBothHalvesOfTheExchange(chats, chat, projectId) -> None:
    asyncio.run(chats.appendTurn(window=chat, question="Refunds?", answer="30 days.", context=[]))

    stored = asyncio.run(chats.loadWindow(projectId, chat.chatId))
    assert [(m.turnIndex, m.role, m.content) for m in stored.messages] == [
        (0, "user", "Refunds?"),
        (1, "assistant", "30 days."),
    ]
    assert stored.turnCount == 2


def testTurnIndicesKeepCountingAcrossTurns(chats, chat) -> None:
    """Indices come from the stored counter inside the transaction, not from
    the caller's window -- which is what stops two concurrent turns writing the
    same document id and losing one of them."""
    window = asyncio.run(chats.appendTurn(window=chat, question="One?", answer="A.", context=[]))
    window = asyncio.run(chats.appendTurn(window=window, question="Two?", answer="B.", context=[]))

    assert [m.turnIndex for m in window.messages] == [0, 1, 2, 3]
    assert window.turnCount == 4


def testRetrievalsAreStoredAgainstTheTurnThatCausedThem(chats, chat, projectId) -> None:
    asyncio.run(
        chats.appendTurn(
            window=chat,
            question="Refunds?",
            answer="30 days.",
            context=[ContextEntry(0, 0, "refund window", ["Refunds: 30 days."])],
        )
    )

    stored = asyncio.run(chats.loadWindow(projectId, chat.chatId))
    assert stored.context[0].turnIndex == 0
    assert stored.context[0].entryIndex == 0
    assert stored.context[0].passages == ["Refunds: 30 days."]
    assert stored.contextCount == 1


def testAppendingToAChatThatIsNotThereIsAnError(chats, projectId) -> None:
    ghost = ChatWindow(chatId="ghost", projectId=projectId, systemPrompt=PROMPT)

    with pytest.raises(ChatStoreError):
        asyncio.run(chats.appendTurn(window=ghost, question="?", answer="!", context=[]))


def testTheSystemPromptIsSnapshotted(chats, chat, projectId) -> None:
    """A prompt edited mid-conversation must not rewrite the instructions the
    earlier answers were given under."""
    asyncio.run(chats.appendTurn(window=chat, question="Q", answer="A", context=[]))

    assert asyncio.run(chats.loadWindow(projectId, chat.chatId)).systemPrompt == PROMPT


def testASummaryReplacesWhatItCovers(chats, chat, projectId) -> None:
    """The point of a summary: the documents below the watermark are never read
    again, so a long conversation stops getting more expensive."""
    window = chat
    for number in range(3):
        window = asyncio.run(
            chats.appendTurn(
                window=window,
                question=f"Q{number}",
                answer=f"A{number}",
                context=[ContextEntry(0, 0, f"search {number}", ["passage"])],
            )
        )

    asyncio.run(
        chats.saveSummary(
            projectId=projectId,
            chatId=chat.chatId,
            summary="They asked three things.",
            throughTurn=4,
            throughContext=2,
        )
    )

    stored = asyncio.run(chats.loadWindow(projectId, chat.chatId))
    assert stored.contextSummary == "They asked three things."
    assert [m.turnIndex for m in stored.messages] == [4, 5]
    assert [c.entryIndex for c in stored.context] == [2]
    # The counters still describe the whole chat, not the surviving tail.
    assert stored.turnCount == 6


def testAppendingContinuesPastASummary(chats, chat, projectId) -> None:
    window = chat
    for number in range(3):
        window = asyncio.run(
            chats.appendTurn(window=window, question=f"Q{number}", answer=f"A{number}", context=[])
        )
    asyncio.run(
        chats.saveSummary(
            projectId=projectId,
            chatId=chat.chatId,
            summary="earlier",
            throughTurn=4,
            throughContext=0,
        )
    )

    folded = asyncio.run(chats.loadWindow(projectId, chat.chatId))
    after = asyncio.run(chats.appendTurn(window=folded, question="Q3", answer="A3", context=[]))

    assert after.turnCount == 8
    assert after.contextSummary == "earlier"


@pytest.mark.slow
def testConcurrentTurnsDoNotLoseAnExchange(chats, chat, projectId) -> None:
    """The transaction is the only thing stopping two questions arriving
    together from both claiming turn 0 and one exchange vanishing."""

    async def race() -> None:
        await asyncio.gather(
            chats.appendTurn(window=chat, question="One?", answer="A.", context=[]),
            chats.appendTurn(window=chat, question="Two?", answer="B.", context=[]),
        )

    asyncio.run(race())

    stored = asyncio.run(chats.loadWindow(projectId, chat.chatId))
    assert stored.turnCount == 4
    assert sorted(m.turnIndex for m in stored.messages) == [0, 1, 2, 3]


# --- the window itself -----------------------------------------------------


def testAWindowSurvivesAJsonRoundTrip() -> None:
    """It goes through Redis as JSON, so the dataclasses have to come back as
    dataclasses rather than as dicts."""
    window = ChatWindow(
        chatId="c1",
        projectId="p1",
        systemPrompt=PROMPT,
        contextSummary="earlier",
        turnCount=2,
        context=[ContextEntry(0, 0, "q", ["p"])],
        messages=[ChatMessage(0, "user", "hello")],
    )

    restored = ChatWindow.fromJson(window.toJson())

    assert restored == window
    assert isinstance(restored.messages[0], ChatMessage)
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
    store = FirestoreChatStore(firestore=ExplodingFirestore(), redis=redis)
    window = ChatWindow(
        chatId="c1",
        projectId="p1",
        systemPrompt=PROMPT,
        messages=[ChatMessage(0, "user", "hello")],
    )
    redis.set(store.cacheKey("p1", "c1"), window.toJson())

    assert asyncio.run(store.loadWindow("p1", "c1")) == window


def testAnUnreadableStoreRaisesRatherThanAnsweringEmpty(redis) -> None:
    """"Firestore is down" and "there is no such chat" lead to a degraded
    answer and a 404 respectively, so they must not be the same value."""

    class BrokenFirestore:
        def collection(self, name):
            raise RuntimeError("unreachable")

    store = FirestoreChatStore(firestore=BrokenFirestore(), redis=redis)

    with pytest.raises(ChatStoreError):
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

    store = FirestoreChatStore(firestore=EmptyFirestore(), redis=BrokenRedis())

    assert asyncio.run(store.loadWindow("p1", "c1")) is None


# --- the factory -----------------------------------------------------------


def testWithoutAGcpProjectTheStoreRefusesToBuild(monkeypatch) -> None:
    """No in-process fallback: one would drop every conversation on restart
    while the service looked perfectly healthy."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        buildChatStore()
