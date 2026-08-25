"""Check FirestoreChatStore against real Firestore.

    python scripts/liveChatCheck.py

Configuration comes from .env, the same as ``liveFirestoreCheck``.

Covers the three things a stand-in client cannot answer for, all of which are
about the *service* rather than about this code:

* the range queries. ``where(turnIndex >= watermark)`` has to work with no
  composite index defined, or a summarised chat 404s its own history on the
  first deployment that has never had one created by hand.
* the transaction in ``appendTurn``. Two questions arriving in one chat at the
  same moment must not both claim turn 6 and lose an exchange -- a fake
  transaction always agrees with itself, so only the real one proves this.
* the round trip of the stored types. ``expiresAt`` goes in as a datetime and
  comes back as ``DatetimeWithNanoseconds``, and the window has to survive
  being JSON-encoded for the cache afterwards.

Runs without Redis on purpose: the cache would answer the reads and none of the
above would be exercised.

Deletes the chat, and both of its subcollections, on the way out.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Imported for its side effect -- app/__init__.py calls load_dotenv() -- so that
# the guard below reads the configuration .env already holds. See
# scripts/liveFirestoreCheck.py.
import app  # noqa: F401

PROJECT_ID = "live-chat-check"

failures: list[str] = []


def check(description: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


def deleteChat(store, projectId: str, chatId: str) -> None:
    """Firestore does not delete subcollections with their parent."""
    reference = store._chatRef(projectId, chatId)
    for name in ("messages", "context"):
        for document in reference.collection(name).stream():
            document.reference.delete()
    reference.delete()


async def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("Set GCP_PROJECT_ID (and GOOGLE_APPLICATION_CREDENTIALS).")

    from app.stores.chatStore import (
        COLLECTION,
        ChatWindow,
        ContextEntry,
        FirestoreChatStore,
    )

    # No Redis: a cache hit would answer the reads below without Firestore ever
    # being asked, which is exactly what must not happen here.
    store = FirestoreChatStore(redis=None)
    created: list[str] = []

    try:
        print("\n=== creating a chat")
        chat = await store.createChat(
            projectId=PROJECT_ID,
            systemPrompt="You are the live-check assistant.",
            ragDbId="live-chat-check-abc123",
            title="first question",
        )
        created.append(chat.chatId)
        check("a chatId was minted", bool(chat.chatId))
        check("it starts with no turns", chat.turnCount == 0)

        print("\n=== an unknown chat is None, not an error")
        check("resolves to nothing", await store.loadWindow(PROJECT_ID, "no-such-chat") is None)

        print("\n=== appending a turn with what it retrieved")
        window = await store.appendTurn(
            window=chat,
            question="What is the refund window?",
            answer="Thirty days from purchase.",
            context=[ContextEntry(0, 0, "refund window", ["Refunds: 30 days."])],
            reviewScore=0.9,
        )
        check("both halves were counted", window.turnCount == 2)
        check("the retrieval was counted", window.contextCount == 1)

        print("\n=== reading it back through the service")
        reloaded = await store.loadWindow(PROJECT_ID, chat.chatId)
        check("the prompt survived", reloaded.systemPrompt.startswith("You are the live-check"))
        check(
            "both messages came back in order",
            [m.role for m in reloaded.messages] == ["user", "assistant"],
        )
        check("the passages came back verbatim", reloaded.context[0].passages == ["Refunds: 30 days."])
        check("the review score came back", reloaded.messages[1].reviewScore == 0.9)
        # Firestore hands timestamps back as DatetimeWithNanoseconds, which is
        # not JSON-serialisable -- if that reached the cache every write to it
        # would fail, silently, forever.
        check("the window still serialises for the cache", bool(ChatWindow.fromJson(reloaded.toJson())))

        print("\n=== two turns racing the same chat")
        # The transaction is the only thing stopping these both claiming turn 2
        # and one exchange disappearing without trace.
        await asyncio.gather(
            store.appendTurn(window=reloaded, question="Gift cards?", answer="No refund.", context=[]),
            store.appendTurn(window=reloaded, question="Vouchers?", answer="No refund.", context=[]),
        )
        raced = await store.loadWindow(PROJECT_ID, chat.chatId)
        check("both exchanges survived", raced.turnCount == 6)
        check(
            "every turn has its own index",
            sorted(m.turnIndex for m in raced.messages) == [0, 1, 2, 3, 4, 5],
        )

        print("\n=== summarising, and the range queries behind it")
        await store.saveSummary(
            projectId=PROJECT_ID,
            chatId=chat.chatId,
            summary="They asked about refunds on several products.",
            throughTurn=4,
            throughContext=1,
        )
        folded = await store.loadWindow(PROJECT_ID, chat.chatId)
        check("the summary came back", folded.contextSummary.startswith("They asked about refunds"))
        # This is the assertion the whole mechanism rests on: below the
        # watermark, those documents are never read again.
        check("only the tail was read", [m.turnIndex for m in folded.messages] == [4, 5])
        check("the folded retrieval was not read", folded.context == [])
        check("the counters still describe the whole chat", folded.turnCount == 6)

        print("\n=== appending after a summary")
        after = await store.appendTurn(
            window=folded, question="And deposits?", answer="Non-refundable.", context=[]
        )
        check("indices continue past the summary", after.turnCount == 8)
        check("the summary is still there", after.contextSummary.startswith("They asked"))
    finally:
        print("\n=== cleanup")
        for chatId in created:
            deleteChat(store, PROJECT_ID, chatId)
            print(f"  deleted chat {chatId}")
        store._db.collection(COLLECTION).document(PROJECT_ID).delete()
        print(f"  deleted {PROJECT_ID}")

    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
