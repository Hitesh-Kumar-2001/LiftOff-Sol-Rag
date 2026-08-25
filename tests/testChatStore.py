"""Chat storage: turn indices, the summary watermarks, and the cache in front.

What matters here is that appending a turn cannot lose an exchange, that a
summarised chat stops reading the documents it folded away, and that a cache hit
does not touch Firestore at all -- that last one is the whole reason the cache
exists, and it is invisible from the outside unless a test asserts it.
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
    InMemoryChatStore,
)

PROJECT = "handbook"
PROMPT = "You are the handbook assistant."


@pytest.fixture
def store() -> InMemoryChatStore:
    return InMemoryChatStore()


@pytest.fixture
def chat(store: InMemoryChatStore) -> ChatWindow:
    return asyncio.run(
        store.createChat(projectId=PROJECT, systemPrompt=PROMPT, ragDbId="handbook-abc123")
    )


# --- the in-memory store ---------------------------------------------------


def testANewChatStartsEmpty(chat: ChatWindow) -> None:
    assert chat.messages == []
    assert chat.context == []
    assert chat.turnCount == 0
    assert chat.systemPrompt == PROMPT


def testAnUnknownChatIsNoneRatherThanAnError(store: InMemoryChatStore) -> None:
    """The caller turns None into a 404 and an exception into a degraded
    answer, so the two must not collapse into one."""
    assert asyncio.run(store.loadWindow(PROJECT, "never-existed")) is None


def testAChatIsScopedToItsProject(store: InMemoryChatStore, chat: ChatWindow) -> None:
    assert asyncio.run(store.loadWindow("another-project", chat.chatId)) is None


def testAppendingStoresBothHalvesOfTheExchange(
    store: InMemoryChatStore, chat: ChatWindow
) -> None:
    asyncio.run(store.appendTurn(window=chat, question="Refunds?", answer="30 days.", context=[]))

    stored = asyncio.run(store.loadWindow(PROJECT, chat.chatId))
    assert [(m.turnIndex, m.role, m.content) for m in stored.messages] == [
        (0, "user", "Refunds?"),
        (1, "assistant", "30 days."),
    ]
    assert stored.turnCount == 2


def testTurnIndicesKeepCountingAcrossTurns(
    store: InMemoryChatStore, chat: ChatWindow
) -> None:
    """Indices come from the stored counter, not from the caller's window --
    which is what stops two concurrent turns writing the same document id."""
    window = asyncio.run(store.appendTurn(window=chat, question="One?", answer="A.", context=[]))
    window = asyncio.run(store.appendTurn(window=window, question="Two?", answer="B.", context=[]))

    assert [m.turnIndex for m in window.messages] == [0, 1, 2, 3]
    assert window.turnCount == 4


def testRetrievalsAreStoredAgainstTheTurnThatCausedThem(
    store: InMemoryChatStore, chat: ChatWindow
) -> None:
    asyncio.run(
        store.appendTurn(
            window=chat,
            question="Refunds?",
            answer="30 days.",
            context=[ContextEntry(0, 0, "refund window", ["Refunds: 30 days."])],
        )
    )

    stored = asyncio.run(store.loadWindow(PROJECT, chat.chatId))
    assert stored.context[0].turnIndex == 0
    assert stored.context[0].entryIndex == 0
    assert stored.context[0].passages == ["Refunds: 30 days."]
    assert stored.contextCount == 1


def testAppendingToAChatThatIsNotThereIsAnError(store: InMemoryChatStore) -> None:
    ghost = ChatWindow(chatId="ghost", projectId=PROJECT, systemPrompt=PROMPT)

    with pytest.raises(ChatStoreError):
        asyncio.run(store.appendTurn(window=ghost, question="?", answer="!", context=[]))


def testALoadedWindowIsDetachedFromTheStore(
    store: InMemoryChatStore, chat: ChatWindow
) -> None:
    """The summariser mutates the window it is handed. Without a copy that
    would edit the stored conversation behind the store's back."""
    asyncio.run(store.appendTurn(window=chat, question="Q", answer="A", context=[]))

    window = asyncio.run(store.loadWindow(PROJECT, chat.chatId))
    window.messages.clear()

    assert len(asyncio.run(store.loadWindow(PROJECT, chat.chatId)).messages) == 2


def testASummaryReplacesWhatItCovers(store: InMemoryChatStore, chat: ChatWindow) -> None:
    """The point of a summary: the documents below the watermark are never
    read again, so a long conversation stops getting more expensive."""
    window = chat
    for number in range(3):
        window = asyncio.run(
            store.appendTurn(
                window=window,
                question=f"Q{number}",
                answer=f"A{number}",
                context=[ContextEntry(0, 0, f"search {number}", ["passage"])],
            )
        )

    asyncio.run(
        store.saveSummary(
            projectId=PROJECT,
            chatId=chat.chatId,
            summary="They asked three things.",
            throughTurn=4,
            throughContext=2,
        )
    )

    stored = asyncio.run(store.loadWindow(PROJECT, chat.chatId))
    assert stored.contextSummary == "They asked three things."
    assert [m.turnIndex for m in stored.messages] == [4, 5]
    assert [c.entryIndex for c in stored.context] == [2]
    # The counters still describe the whole chat, not the surviving tail.
    assert stored.turnCount == 6


# --- the window itself -----------------------------------------------------


def testAWindowSurvivesAJsonRoundTrip() -> None:
    """It goes through Redis as JSON, so the dataclasses have to come back as
    dataclasses rather than as dicts."""
    window = ChatWindow(
        chatId="c1",
        projectId=PROJECT,
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


class ExplodingFirestore:
    """Any use at all is a failure. Proves a cache hit never reaches Firestore."""

    def collection(self, name):  # pragma: no cover - only called when the test fails
        raise AssertionError("Firestore was read on what should have been a cache hit.")


@pytest.fixture
def redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def testACachedWindowNeverReachesFirestore(redis) -> None:
    store = FirestoreChatStore(firestore=ExplodingFirestore(), redis=redis)
    window = ChatWindow(
        chatId="c1",
        projectId=PROJECT,
        systemPrompt=PROMPT,
        messages=[ChatMessage(0, "user", "hello")],
    )
    redis.set(store.cacheKey(PROJECT, "c1"), window.toJson())

    loaded = asyncio.run(store.loadWindow(PROJECT, "c1"))

    assert loaded == window


def testAnUnreadableStoreRaisesRatherThanAnsweringEmpty(redis) -> None:
    """"Firestore is down" and "there is no such chat" lead to a degraded
    answer and a 404 respectively, so they must not be the same value."""

    class BrokenFirestore:
        def collection(self, name):
            raise RuntimeError("unreachable")

    store = FirestoreChatStore(firestore=BrokenFirestore(), redis=redis)

    with pytest.raises(ChatStoreError):
        asyncio.run(store.loadWindow(PROJECT, "c1"))


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

    assert asyncio.run(store.loadWindow(PROJECT, "c1")) is None
