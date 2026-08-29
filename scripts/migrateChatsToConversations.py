"""Copy the old ``ragChats`` tree onto the new ``ragConversations`` one.

The service used to call a conversation a *chat*, in the URLs, the code and the
Firestore layout. The rename to *conversation* went all the way down, which
means the collection names and one field name changed::

    ragChats/{projectId}/chats/{chatId}
        -> ragConversations/{projectId}/conversations/{conversationId}

    field  chatId  ->  conversationId

Firestore has no rename. A collection is defined by the path its documents are
written under, so the new code simply reads an empty tree and every conversation
held under the old names becomes unreachable -- present, billable, and invisible
to the service that wrote it. This script is what stops that being a data loss.

**Copies. Deletes nothing.** ``ragChats`` is left exactly as it was, so a rollback
is redeploying the previous image rather than a restore, and a mistake here costs
storage rather than conversations. Delete the old tree by hand once the new one
has been checked -- and note that Firestore does not remove subcollections with
their parent, so that means walking ``messages`` and ``context`` too.

**Idempotent.** Every write is a ``set()`` at a deterministic path, so running it
twice is the same as running it once. Safe to re-run after a partial failure.

Usage::

    python scripts/migrateChatsToConversations.py            # report only
    python scripts/migrateChatsToConversations.py --apply    # actually copy
"""

import sys

import app  # noqa: F401  -- runs load_dotenv() before anything reads config.
from app.infra.firestoreClient import firestoreClient
from app.infra.redisClient import redisClient

OLD_ROOT = "ragChats"
OLD_CHILD = "chats"
NEW_ROOT = "ragConversations"
NEW_CHILD = "conversations"

# The old cache prefix. Entries under it are orphaned by the rename -- the new
# code reads `ragConversation:` -- so they would sit there until their TTL ran
# out. Harmless, but there is no reason to leave them.
OLD_CACHE_PREFIX = "ragChat:"

SUBCOLLECTIONS = ("messages", "context")


def renamedFields(data: dict) -> dict:
    """The stored document with ``chatId`` carried over as ``conversationId``.

    Only that one key changes. Everything else -- systemPrompt, contextSummary,
    the watermarks, the counters, title, lastMessage, the timestamps -- keeps
    its name, because only the word "chat" was ever the problem.
    """
    moved = dict(data)
    if "chatId" in moved:
        moved["conversationId"] = moved.pop("chatId")
    return moved


def migrate(apply: bool) -> int:
    db = firestoreClient()
    conversations = 0
    documents = 0

    for parent in db.collection(OLD_ROOT).stream():
        projectId = parent.id
        children = list(parent.reference.collection(OLD_CHILD).stream())
        print(f"{OLD_ROOT}/{projectId}: {len(children)} conversation(s)")

        newParent = db.collection(NEW_ROOT).document(projectId)
        if apply:
            newParent.set(renamedFields(parent.to_dict() or {}), merge=True)
        documents += 1

        for child in children:
            data = child.to_dict() or {}
            target = newParent.collection(NEW_CHILD).document(child.id)
            counts = {}

            for name in SUBCOLLECTIONS:
                entries = list(child.reference.collection(name).stream())
                counts[name] = len(entries)
                for entry in entries:
                    if apply:
                        target.collection(name).document(entry.id).set(entry.to_dict() or {})
                    documents += 1

            if apply:
                target.set(renamedFields(data))
            documents += 1
            conversations += 1

            print(
                f"  {child.id}  turns={data.get('turnCount')}  "
                f"messages={counts['messages']}  context={counts['context']}  "
                f"title={str(data.get('title') or '')[:40]!r}"
            )

    if apply:
        redis = redisClient()
        if redis is not None:
            stale = list(redis.scan_iter(match=f"{OLD_CACHE_PREFIX}*"))
            if stale:
                redis.delete(*stale)
            print(f"\ndropped {len(stale)} orphaned '{OLD_CACHE_PREFIX}' cache key(s)")

    print(
        f"\n{'copied' if apply else 'would copy'} {conversations} conversation(s), "
        f"{documents} document(s) in total"
    )
    if not apply:
        print("\nDry run. Re-run with --apply to write.")
    return conversations


if __name__ == "__main__":
    sys.exit(0 if migrate("--apply" in sys.argv) >= 0 else 1)
