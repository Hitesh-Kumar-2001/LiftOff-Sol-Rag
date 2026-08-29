# Conversation schema

How a conversation is stored in Firestore, how a turn reads and writes it, and
what has to be configured in the console for it to behave.

Code: [`app/stores/conversationStore.py`](../app/stores/conversationStore.py) (storage),
[`app/agent/summariser.py`](../app/agent/summariser.py) (rendering and folding),
[`app/api/routes.py`](../app/api/routes.py) (`query`, which orchestrates both).
Verified against the real service by `python scripts/liveConversationCheck.py`.

---

> **Renamed from `chat`.** This tree was `ragChats/{projectId}/chats/{chatId}` with a
> `chatId` field until the rename to `conversation`. Firestore has no rename — a
> collection is defined by the path its documents are written under — so an existing
> deployment needs `scripts/migrateChatsToConversations.py` before the new code can see
> its conversations. It copies and deletes nothing, so `ragChats` stays put and a
> rollback is redeploying the previous image.

## Collections

```
ragProjects/{projectId}                    the projectId -> ragDbId mapping
                                           (pre-existing; app/stores/projectStore.py)

ragConversations/{projectId}                       one project's conversations
  conversations/{conversationId}                           one conversation
    messages/{turnIndex}                   one turn each
    context/{entryIndex}                   one retrieval each
```

Collection names come from `FIRESTORE_PROJECTS_COLLECTION` and
`FIRESTORE_CONVERSATIONS_COLLECTION`; the subcollection names are fixed.

### Conversations are keyed by `projectId`, not `ragDbId`

Three reasons, and all three are load-bearing:

1. `/query` runs against a project that has never been ingested into —
   `ragDbId` is `None` there, and there would be no key to write the
   conversation under. Minting one is forbidden: only `/document` may create a
   mapping, or every mistyped project leaves an empty vector namespace behind.
2. A `ragDbId` is deliberately allowed to change — that is why it is random
   rather than derived. Rebuilding a project into a fresh namespace would orphan
   every conversation keyed by the old one, and unlike vectors, conversations
   cannot be regenerated.
3. It buys no isolation. The service has no authentication, and `ragDbId` is
   resolved *from* `projectId` on every request anyway.

The `ragDbId` a conversation was created against is still recorded on the conversation document,
as **audit only** — which namespace its answers were grounded in. Nothing
resolves retrieval from it.

---

## `ragConversations/{projectId}`

The parent document. It exists so a project's conversations can be found by listing
rather than by collection-group query — Firestore is perfectly happy to hold a
subcollection under a document that was never written, and such a conversation is
reachable but invisible in the console.

| Field | Type | Notes |
| --- | --- | --- |
| `projectId` | string | Same as the document id. |
| `updatedAt` | timestamp | Touched when a conversation is created. |

---

## `ragConversations/{projectId}/conversations/{conversationId}`

`conversationId` is `uuid4().hex`. Random, never derived from the project, the question,
or a timestamp.

| Field | Type | Notes |
| --- | --- | --- |
| `conversationId` | string | Same as the document id. |
| `projectId` | string | Denormalised, so a collection-group query can filter. |
| `systemPrompt` | string | **Snapshotted at creation.** See below. |
| `ragDbId` | string \| null | Audit only. Never resolve retrieval from this. |
| `title` | string | The first question, 80 chars. Written once. |
| `lastMessage` | string | The latest answer, 80 chars — so a conversation list is one read per conversation. |
| `contextSummary` | string | `""` until the conversation is first folded. |
| `summarisedThroughTurn` | number | Messages below this index are inside the summary. |
| `summarisedThroughContext` | number | Retrievals below this index are inside the summary. |
| `summarisedAt` | timestamp \| null | When the fold last ran. |
| `turnCount` | number | The **next** turn index — 2 per exchange. |
| `contextCount` | number | The **next** context index. |
| `createdAt` | timestamp | |
| `updatedAt` | timestamp | Order a conversation list by this, descending. |
| `expiresAt` | timestamp | For a TTL policy. See *Expiry*. |

### Why `systemPrompt` is a snapshot

A project's prompt is editable (`app/agent/promptStore.py`). A conversation whose
instructions change underneath it half way through stops being one conversation:
the model is asked to keep faith with earlier answers it would no longer have
given. New conversations pick up the new prompt; running conversations finish under the one they
started with.

---

## `.../conversations/{conversationId}/messages/{turnIndex}`

Document id is the zero-padded turn index — `000000`, `000001`, … User turns are
even, assistant turns odd.

| Field | Type | Notes |
| --- | --- | --- |
| `turnIndex` | number | Also the document id. Query on the field, not the id. |
| `role` | string | `"user"` or `"assistant"`. |
| `content` | string | Capped at `RAG_CONVERSATION_MAX_MESSAGE_CHARS` (20 000). |
| `createdAt` | timestamp | |
| `expiresAt` | timestamp | |
| `reviewScore` | number \| null | Assistant turns only. Never shown to the model. |
| `retried` | bool | Assistant turns only. |

## `.../conversations/{conversationId}/context/{entryIndex}`

One document per search the agent ran — the `conversationHistory`'s expensive twin.

| Field | Type | Notes |
| --- | --- | --- |
| `entryIndex` | number | Also the document id. |
| `turnIndex` | number | The turn that caused the search, so a retrieval is folded away with the exchange it belongs to. |
| `kind` | string | `"search"`. |
| `query` | string | What the agent searched for. |
| `passages` | array\<string\> | The chunk text, verbatim. Each capped at `RAG_CONVERSATION_MAX_PASSAGE_CHARS` (4 000). |
| `createdAt` | timestamp | |
| `expiresAt` | timestamp | |

**Passages are stored as text, not as chunk ids.** The entire point is that a
follow-up never pays for the same vector search again; an id would mean a
Pinecone round trip to turn it back into text, which is the cost being avoided.

**A search that found nothing is still recorded** (`passages: []`). "The
documents do not cover this" is a real finding and an expensive one to
rediscover on every follow-up.

Web search results are deliberately **not** stored: they are somebody else's
pages, they go stale, and re-fetching one is a Tavily call rather than the vector
search this mechanism exists to avoid paying for twice.

---

## Why subcollections and not fields

A Firestore document is capped at **1 MiB**, and every write rewrites the whole
document.

- `conversationHistory` as an **array on the conversation document** makes each turn cost the
  length of the conversation so far, and hard-stops a few hundred turns in.
  Retrieved passages are worse — six chunks a search, several searches a turn.
- A nested map `ragId { conversationId { … } }` in **one document** is worse still: every
  conversation in a project shares that 1 MiB and contends on Firestore's ~1 sustained
  write per second per document.

One document per item makes appending O(1) and unbounded, at the price of one
read per item — which is exactly the price the summariser caps.

**Document ids are the zero-padded index, and the index is also a field.** The id
makes a write idempotent: turn 7 is always `messages/000007`, so a retried append
overwrites rather than duplicating. The field is what queries use —
`where(turnIndex >= n)` needs no composite index and no ordering trick.

---

## The read path

**Redis first, Firestore only on a miss.** This is what makes a follow-up cheap:
the Firestore path is a document read plus two range queries, and the Redis path
is a single GET of the whole assembled window — the prompt, the summary, the
recent turns and the passages already retrieved.

```
/query or /conversation/message arrives with a conversationId
        |
        v
  Redis  ragConversation:{projectId}:{conversationId}        one GET, the whole window
        |  miss
        v
  Firestore  conversations/{conversationId}                  one document read
        |
        +-- contextSummary present?  ------> read only messages/context AT OR ABOVE
        |                                     the watermarks; everything below is
        |                                     already inside the summary and is
        |                                     never read again
        |
        +-- no summary yet          ------> read all messages and all context
        |
        v
  cache the assembled window back into Redis, then answer
```

Two range queries do the tail read:

```python
messages.where(filter=FieldFilter("turnIndex", ">=", summarisedThroughTurn)).order_by("turnIndex")
context.where(filter=FieldFilter("entryIndex", ">=", summarisedThroughContext)).order_by("entryIndex")
```

Range and order on the same single field, so Firestore's automatic single-field
indexes cover both — **no composite index needs to be created.** Verified
against the real service by `scripts/liveConversationCheck.py`.

Only a window that was actually resolved is cached. A Firestore failure degrades
one turn; it is not written down and inflicted on the next hour of them.

### Failure behaviour

| Situation | Result |
| --- | --- |
| No `conversationId` sent | **`/query` only.** One is created; its id comes back on the response. `/conversation/message` requires the id and answers 422 without it. |
| `conversationId` exists | Its window is loaded — Redis, then Firestore. |
| `conversationId` does not exist | **404.** Not a new conversation — a typo silently starting a fresh one is indistinguishable, to the caller, from a model that has forgotten everything. |
| Conversation store unreachable | The question is answered without history, and the caller's own `conversationId` is echoed back. |
| The turn cannot be written | The answer is still returned, and the failure is logged loudly. |

The model call is the expensive, irreversible step. Nothing to do with storing a
conversation may cost a caller an answer that has already been paid for.

---

## The write path

One Firestore **transaction** per turn:

1. read the conversation document → `turnCount`, `contextCount`
2. write `messages/{turnCount}` (user) and `messages/{turnCount+1}` (assistant)
3. write one `context/{n}` per search the agent ran this turn
4. update `turnCount`, `contextCount`, `updatedAt`, `expiresAt`, `lastMessage`,
   and `title` if it is still empty

A transaction and not a batch, because the indices come from the document's own
counters: two questions sent into one conversation at the same moment would otherwise
both read `turnCount` as 6, both write `messages/000006`, and one exchange would
vanish. Reading the counter inside the transaction makes the second attempt see
the first's write and retry against 8.

Atomic also means a conversation is never left showing a question with no answer, or an
answer whose retrieved passages did not survive beside it.

---

## Summarising

Context and history both grow without bound. Past a budget, everything except the
most recent turns is folded into one paragraph of prose and the documents behind
it stop being read.

```
approximate tokens > RAG_CONTEXT_SUMMARY_TOKENS (6000)
  and more than RAG_CONTEXT_KEEP_TURNS (4) turns exist above the last watermark
        |
        v
  summarise (prior summary + folded turns + folded passages) -> contextSummary
  advance summarisedThroughTurn / summarisedThroughContext
```

- **Runs before the answer, not after.** Summarising afterwards would leave the
  turn that tripped the limit to be answered with the oversized prompt that
  tripped it. The cost lands on the same turn as the benefit.
- **The recent turns are kept verbatim.** "Make that shorter" only means
  something next to the text it refers to.
- **A single enormous exchange is not folded.** It is over budget and
  summarising cannot help — there is nothing outside the keep-verbatim tail to
  fold, so the call would change nothing and be made again next turn.
- **Failure is not an error.** The fallback is `trimToBudget`, which drops the
  oldest *retrievals* from what is sent without touching what is stored.
  Messages are never dropped: passages can be searched for again, while a
  missing turn silently rewrites the conversation the user can see they had.

Token count is approximated at four characters per token. An exact count means
BPE-encoding the whole conversation on every question — real CPU on the event
loop, for precision a threshold does not need.

### How a stored conversation reaches the model

Retrieved passages go into the **system prompt**, not back into the transcript as
replayed tool calls. Replaying them faithfully means reconstructing
provider-specific tool-call ids and pairing each call with its result — four
different shapes across four providers, and broken the moment one changes. A
prompt section is provider-independent and achieves the actual goal.

The block ends with an instruction, not just material: passages sitting in a
prompt with nothing said about them get treated as background and searched for
again, which is the one outcome the whole mechanism exists to prevent.

Review scores are stored but never rendered. Telling a model its earlier answer
scored 0.4 invites it to apologise for that answer instead of answering the
question in front of it.

---

## Expiry

Every document carries `expiresAt` (`RAG_CONVERSATION_TTL_SECONDS`, default 90 days,
refreshed on each turn).

**The field alone deletes nothing.** A TTL policy has to be created in the
Firestore console on each of the three collection groups — `conversations`, `messages`,
`context` — with `expiresAt` as the timestamp field. Writing the field
regardless means turning expiry on later is a console change rather than a
backfill over every conversation ever held.

## Security rules

The Admin SDK bypasses security rules entirely, so locking them costs this
service nothing and stops anything else reading conversations. Worth having
while the API itself has no authentication:

```
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} { allow read, write: if false; }
  }
}
```

---

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `FIRESTORE_CONVERSATIONS_COLLECTION` | `ragConversations` | Root collection. |
| `RAG_CONVERSATION_CACHE_PREFIX` | `ragConversation:` | Redis key prefix. |
| `RAG_CONVERSATION_CACHE_TTL_SECONDS` | `3600` | How long an assembled window is served before Firestore is asked again. |
| `RAG_CONVERSATION_TTL_SECONDS` | `7776000` | 90 days. Written as `expiresAt`; needs a TTL policy to act. |
| `RAG_CONVERSATION_MAX_MESSAGE_CHARS` | `20000` | Per-message cap. |
| `RAG_CONVERSATION_MAX_PASSAGE_CHARS` | `4000` | Per-passage cap. |
| `RAG_CONTEXT_SUMMARY_TOKENS` | `6000` | Fold above this. |
| `RAG_CONTEXT_KEEP_TURNS` | `4` | Messages kept verbatim through a fold. |
| `RAG_CONTEXT_SUMMARY_MAX_CHARS` | `6000` | Ceiling on the summary itself. |
| `RAG_SUMMARISER_PROVIDER` / `RAG_SUMMARISER_MODEL` | agent defaults | Configured apart from the agent and the reviewer — summarising is the cheapest call this service makes and the one most worth pointing at a small fast model. |

Storage backend follows the same switch as everything else durable:
`GCP_PROJECT_ID` set means Firestore, unset means a dict that dies with the
process. `REDIS_URL` adds the cache; without it every question re-reads its
conversation from Firestore, which is slower rather than wrong.

---

## Wire contract

### `POST /api/v1/conversation` — start one

```jsonc
// request
{
  "serverId": "billing-service",
  "projectId": "handbook",
  "title": "Refund policy"   // optional; the first question fills it in if omitted
}

// response, 201
{
  "conversationId": "3f2a…",
  "projectId": "handbook",
  "systemPrompt": "You are a helpful assistant…"   // the snapshot it will answer under
}
```

For the caller that needs the id before there is a question — a UI opening a new
conversation has a window to render and a record to attach to the user at the
moment the user clicks, and none of that can wait on a model call. `/query`
still starts a conversation for a question that names none, so this is an addition to
that path and not a replacement for it.

Two things are decided by this call and cannot be decided later: the
`systemPrompt` snapshot (see *Why `systemPrompt` is a snapshot*) and the
`ragDbId` recorded for audit. The project is resolved **read-only** — starting a
conversation never brings a RAG database into existence, or every mistyped
`projectId` would leave an empty one behind forever.

A conversation store failure here is a **503**, not a degraded success. That is the
opposite of `/query`, and deliberately so: `/query` degrades because it is
holding an answer that was already paid for, whereas this request *is* the
creation. A 201 carrying an id nothing stored would be a conversation the caller
can address and every later request will 404.

### `POST /api/v1/conversation/message` — post a question to it

```jsonc
// request
{
  "serverId": "billing-service",
  "projectId": "handbook",
  "conversationId": "3f2a…",     // required here, unlike on /query
  "question": "And for gift cards?"
}

// response
{
  "answer": "Gift cards are non-refundable.",
  "projectId": "handbook",
  "conversationId": "3f2a…"
}
```

`conversationId` is the only difference from `/query` — both run the same
`_runTurn` in `app/api/routes.py`. Requiring it is what lets this endpoint 404
an id it does not recognise instead of quietly opening a new conversation under
a name that says it is addressing an existing one.

### `POST /api/v1/query` — the one-call form

```jsonc
// request
{
  "serverId": "billing-service",
  "projectId": "handbook",
  "question": "And for gift cards?",
  "conversationId": "3f2a…"   // omit to start a new one
}

// response
{
  "answer": "Gift cards are non-refundable.",
  "projectId": "handbook",
  "conversationId": "3f2a…"   // always returned, including on the turn that created it
}
```

Which of the two forms to use is about who keeps the id. A UI that renders a
conversation list wants one before there is a question, and uses
`/conversation` then `/conversation/message`. A caller that just wants an answer,
or is happy to learn the id from the first response, uses `/query`.

`ragDbId` still appears nowhere on the wire.

---

## Not built

- **No endpoint lists a project's conversations, or reads one back.** The data is shaped
  for it — `title`, `lastMessage` and `updatedAt` are denormalised onto the conversation
  document precisely so a list is one read per conversation — but nothing serves it yet.
  This is the gap that matters most now that conversations can be created: a UI can open
  a conversation and continue it, and still cannot show the user the ones they
  already had.
- **No endpoint deletes a conversation.** Deleting one means deleting both
  subcollections first; Firestore does not remove them with the parent. See
  `deleteConversation` in `scripts/liveConversationCheck.py` for the shape.
- **No access control.** Anyone who can reach the service can read any
  conversation in any project by naming its ids. Same standing gap as the rest
  of the API.
