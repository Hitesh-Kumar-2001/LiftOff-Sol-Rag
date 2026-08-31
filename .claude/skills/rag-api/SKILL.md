---
name: rag-api
description: Context for this repo — an unauthenticated FastAPI RAG service (async document ingestion, Gemini chunking, Pinecone vector store, Redis job table and queue, Firestore project mapping and conversation history, a worker process, and a deepagents answering agent over anthropic/openai/groq/gemini). Load before reading, changing, debugging, or extending any code under app/, api/, scripts/, or tests/, or when asked how ingestion, jobs, projects, search, conversations, the agent, or deployment work here.
---

# RAG API — project context

A FastAPI service that ingests documents into per-project RAG databases and searches
them. Python 3.12, managed with `uv`.

**It has no authentication.** Every endpoint takes a `serverId`, but nothing verifies
it — no secret, no signature, no registry. It is a log label. Anyone who can reach the
service can ingest into any project and read any project's chunks under any name. This
was removed deliberately; `app/security.py` and `app/credentials.py` are gone and
recoverable from git history if it needs to come back.

## Commands

```bash
uv run uvicorn app.main:app --reload      # local API, docs at /docs
uv run pytest                             # the unit suite -- tests/ only
uv run pytest -m "not slow"               # skip the ~2min corpus tests
uv run pytest e2e -s                      # the LIVE suite. Costs money. See below.
python -m app.jobs.worker                 # the ingestion worker (needs REDIS_URL)
docker compose up --build                 # api + worker + redis
```

Layout: `app/api` (routes, schemas), `app/agent` (LLM manager, prompt store, tools,
reviewer, summariser, the loop), `app/jobs` (record, manager, Redis store, queue,
worker), `app/ingestion` (documents, selector, pipeline, processor), `app/stores`
(chunk stores, project mapping, conversation store), `app/infra` (Redis/Firestore clients,
machine stats).

`scripts/live*Check.py` run against real infrastructure and are deliberately outside
the test suite (they cost money / need network): `liveFirestoreCheck` (project mapping),
`liveConversationCheck` (conversation store — the append transaction and the summary range queries),
`livePineconeCheck` (store), `liveIngestionCheck <url>` (whole pipeline, real Gemini
and Pinecone). `scripts/migrateChatsToConversations.py` is the one-off that carried the
old `ragChats` tree onto `ragConversations`; it is a dry run without `--apply`, copies
rather than moves, and is idempotent. They read `.env`. **The suite reads it too** — there are no in-process
stores, so `tests/conftest.py` requires `GCP_PROJECT_ID` and runs against real Firestore,
prefixing every id per run and deleting what it created. Redis is `fakeredis`, in-process,
so the real `QueuedJobManager` and `RedisJobStore` are exercised without a server.

**`e2e/` is a second suite and is not part of `uv run pytest`** — `testpaths = ["tests"]`
in `pyproject.toml` keeps it out, because every run costs real money at a real provider
and needs the API, the worker and Redis actually running (it *skips* with an instruction
if they are not; it never starts them). It ingests
`e2e/documents/wanderlynTravel.txt` — a ~16.5k-token fictional travel brochure, sized
deliberately into the **AI chunking** band — through the real worker into a real vector
database, then runs a twenty-turn sales conversation through the `web` gateway and reads
`ragUsage` back to check the arithmetic. One session-scoped fixture runs the conversation
and twenty-five separate tests read the transcript, so a failure names one property rather
than cascading. It asserts grounding (the October price, the single supplement, the module
price), memory (turn 14 asks about a room without naming the trip — only the remembered
trip gives $520), and honesty (it must not invent a discount, and must decline the drone
question the corpus does not answer). Tone and phrasing are deliberately not asserted.
`RAG_E2E_PROJECT_ID` reuses an ingested project so iterating does not re-pay for ~220
chunker calls; full details in [e2e/README.md](e2e/README.md). **WhatsApp and LINE are not
covered** — their answers arrive by push rather than on the response.

## Endpoints

All are POST, taking their input in the body. **Every endpoint takes a `projectId` and
an unverified `serverId`; none takes or returns a `ragDbId`.** There is no 401 anywhere
in the API.

| Route | Does |
| --- | --- |
| `POST /api/v1/conversations/{projectId}` | Creates an empty conversation, returns its `conversationId` and the `systemPrompt` snapshotted onto it. 201. Read-only on the project — starting one never mints a `ragDbId`. |
| `POST /api/v1/conversations/{projectId}/{gateway}` | **One route, every gateway.** `web` takes `{serverId, question, conversationId?}` and answers on the response. `whatsapp` / `line` take the platform's signed body, verify it, 200 immediately, and push the answer back afterwards. 403 bad signature, 404 unknown gateway or unknown `conversationId`, 502/503/504 as `web` only. |
| `GET /api/v1/conversations/{projectId}/{gateway}` | The gateway's URL-verification handshake. Meta's `hub.challenge`, echoed as plain text. 405 for gateways with no handshake (LINE, web). |
| `POST /api/v1/document` | Resolves-or-creates the project's database, claims it, queues ingestion, returns 202. 409 on conflict, 503 on dispatch failure. |
| `POST /api/v1/document/status` | Job status. Returns `projectId` normally, or `documentLink` when the strategy was RAW. 404 if the project or its job is unknown. |
| `POST /api/v1/search` | Retrieval only — the matching chunks, not a generated answer. An unresolved project gives empty hits, not a 404. |
| `GET /`, `GET /health` | Liveness. `/health` also reports CPU / memory / disk — informational only, never changes the outcome, and a reading that fails is omitted rather than 500ing. |

## Module map

**API layer** — [app/main.py](app/main.py) builds the app (the lifespan runs
`checkConfiguration()` so a bad config fails the deploy, and awaits in-flight jobs on
shutdown) and owns the 422 handler that allowlists `type`/`loc`/`msg`
so a malformed request is not echoed back. [app/api/routes.py](app/api/routes.py) has the three
document and search routes and is where `projectId` becomes `ragDbId`;
[app/api/conversationRoutes.py](app/api/conversationRoutes.py) is the whole conversation
surface, one route for every gateway. [app/api/schemas.py](app/api/schemas.py) is
the wire contract. [app/stores/projectStore.py](app/stores/projectStore.py) holds the mapping —
`ProjectStore` protocol, the Firestore implementation, `buildProjectStore()`
factory. [app/infra/machineStats.py](app/infra/machineStats.py) reads CPU/memory/disk via `psutil`
for `/health`.

**Agent** — [app/agent/agent.py](app/agent/agent.py) is the entry point
(`answerQuestion`): resolve prompt → build tools → run → review once → at most one retry,
continued from the first run's transcript. [app/agent/llmManager.py](app/agent/llmManager.py)
turns `(provider, model)` into a `BaseChatModel` for anthropic/openai/groq/gemini; vendor
imports live *inside* the builder functions, `model` is required, and clients are
`lru_cache`d because each owns a connection pool. **It holds no model names** — the four
role functions (`agentModel`, `reviewerModel`, `summariserModel`, `chunkerModel`) read
`config/models.toml` through [app/modelConfig.py](app/modelConfig.py); see *Model configuration*.
**OpenAI is built with `use_responses_api=True`** — `/v1/responses`, not Chat Completions —
because OpenAI's reasoning models refuse *function tools* on `/v1/chat/completions` with a
400. Only the agent binds one (`searchProject`); every other role uses
`with_structured_output`, which is a response format and works on either endpoint. So the
symptom was ingestion succeeding, startup checks passing, `/health` green, and **every
question 502ing**. `reasoning_effort='none'` also fixes it and turns reasoning off, which is
the wrong trade for a persona that must ground every claim. `setdefault`, so an explicit
override still wins. Content then arrives as blocks rather than a string;
`agent._lastText` and `summariser.summarise` already handle both.
[app/agent/promptStore.py](app/agent/promptStore.py) resolves a project's system prompt from
two Firestore collections and caches the *resolved text keyed by projectId* in Redis; the
default it falls back to is the **persona** in `config/prompts.toml`, read by
[app/promptConfig.py](app/promptConfig.py) — see *Persona*.
[app/agent/tools.py](app/agent/tools.py) builds `searchProject` (closed over one `ragDbId`
*and* an optional `searchLog` that records what it retrieved) and Tavily.
[app/agent/reviewer.py](app/agent/reviewer.py) grades an answer 0.0–1.0 with an
optional suggestion. [app/agent/summariser.py](app/agent/summariser.py) renders a stored
conversation into prompt + messages, and folds it down when it outgrows
`RAG_CONTEXT_SUMMARY_TOKENS`.

**Conversations** — one word, all the way down: the URL, the wire field, the class, the
Firestore collections and the stored field are all `conversation`. It was `chat` in the
storage layer until the rename; `scripts/migrateChatsToConversations.py` is what carried
the existing `ragChats` tree across, and it copies rather than moves, so the old tree is
still there to roll back to.
[app/stores/conversationStore.py](app/stores/conversationStore.py) is the storage:
`ragConversations/{projectId}/conversations/{conversationId}` with `messages` and `context` subcollections,
Redis in front, `ConversationStore` protocol + Firestore + factory. Full schema and
rationale in [docs/conversationSchema.md](docs/conversationSchema.md) — read that before changing
anything about how a conversation is stored.

**Channels** — the messaging gateways, in [app/channels/](app/channels/).
[channel.py](app/channels/channel.py) is the `Channel` protocol every gateway
satisfies (`verify` / `parse` / `send`) plus `IncomingMessage` and the shared HMAC
helpers; [whatsapp.py](app/channels/whatsapp.py) and [line.py](app/channels/line.py) are
the two adapters; [registry.py](app/channels/registry.py) is the one place a gateway is
registered; [sender.py](app/channels/sender.py) is the single outbound path, and the only
thing that turns "an answer was produced" into "an answer was delivered".
The routes live with the web ones in
[app/api/conversationRoutes.py](app/api/conversationRoutes.py); per-project credentials are
in [app/stores/channelStore.py](app/stores/channelStore.py). See *Messaging channels*.

**Usage** — [app/agent/usage.py](app/agent/usage.py) collects what an answer cost while
it is being produced (`UsageLog`, threaded like `searchLog`; `trackUsage` wraps each role
in its own `get_usage_metadata_callback`), and
[app/stores/usageStore.py](app/stores/usageStore.py) writes it. See *Token accounting*.

**Auth** — none, except on the webhooks. There is no auth module; see the note at the
top and the signature checks in *Messaging channels*.

**Jobs** — [app/jobs/job.py](app/jobs/job.py) is the `Job` record, `JobStatus`, `runJob`, the
`JobStore` protocol, and `resolveSubmission` (the shared NEW/REUSE/CONFLICT rules).
[app/jobs/jobManager.py](app/jobs/jobManager.py) is the `JobManager` Protocol, the
environment-driven `buildJobManager()`, and the lazily built, lru_cached `getJobManager()`. [app/jobs/redisJobStore.py](app/jobs/redisJobStore.py) is the table —
storage only, and the claim is a WATCH/MULTI transaction.
[app/jobs/queuedJobManager.py](app/jobs/queuedJobManager.py) claims then enqueues;
[app/jobs/jobQueue.py](app/jobs/jobQueue.py) is the Redis list and its crash recovery;
[app/jobs/worker.py](app/jobs/worker.py) is the process on the far end.

**Ingestion** — [app/ingestion/ragProcessor.py](app/ingestion/ragProcessor.py) is the composition root:
download → analyze → select strategy → extract text → chunk → store.
[app/ingestion/documents.py](app/ingestion/documents.py) handles download, format detection, archive walking,
and metadata (pdf/docx/csv/txt/md; zip/rar). [app/ingestion/ragSelector.py](app/ingestion/ragSelector.py)
picks a strategy from token count. [app/ingestion/ragIngestionPipeline.py](app/ingestion/ragIngestionPipeline.py)
is chunking plus the `ChunkStore` protocol plus `lexicalSearch`.
[app/stores/chunkStoreFactory.py](app/stores/chunkStoreFactory.py) picks the store;
[app/stores/pineconeChunkStore.py](app/stores/pineconeChunkStore.py) and
[app/stores/localChunkStore.py](app/stores/localChunkStore.py) implement it.

**Deploy** — the `Dockerfile` runs uvicorn under `uv` (`uv sync --frozen --no-dev`
from `pyproject.toml` + `uv.lock`; there is no `requirements.txt`). The worker is a
second process from the same image: `python -m app.jobs.worker`. `app/__init__.py` calls
`load_dotenv()` before any submodule reads config.

## Chunking strategies

`RagSelector.score` reads `DocumentMetadata.tokenCount` (cl100k_base):

- `< 2000` → **RAW** — stored whole, one chunk. *No vector DB is meaningfully populated*,
  so `/document/status` answers with `documentLink` instead of a queryable project.
- `< 10000` → **NON_AI** — 400-token windows, 40-token overlap.
- otherwise → **AI** — the `[chunker]` model picks boundaries, one call per blank-line
  section, 8 concurrent, through `with_structured_output(ChunkList)`. A section whose
  answer will not parse, or comes back empty, falls back to non-AI chunking rather than
  losing the document; a *call* that fails fails the whole job, because a bad key or an
  exhausted quota fails every section identically and silently non-AI-chunking a whole
  corpus stores a database that looks successful. `>= 100000` tokens logs a warning and
  proceeds.

  This used to call Gemini through `google-genai` directly, with its own key and its own
  pinned model name. It was the only AI call in the service that could not be repointed
  without editing code, and the one whose per-day quota stopped a large ingestion dead.

Every strategy's output then passes `enforceEmbedLimit` — anything over
`RAG_MAX_EMBED_TOKENS` (2048) is re-split, because Pinecone's embedder silently truncates
the tail instead of erroring.

## Model configuration

Four roles, one file: [config/models.toml](config/models.toml).

| Role | Called | Notes |
| --- | --- | --- |
| `agent` | once per `/query` | answers the question |
| `reviewer` | once per `/query` | grades the answer 0.0–1.0 |
| `summariser` | when a conversation outgrows its budget | on the critical path; worth a small fast model |
| `chunker` | **once per blank-line section of every AI-chunked document** | by far the highest call volume; the role a request quota stops first |

Each section names a `provider` (anthropic/openai/groq/gemini) and a `model`, and both
are required. [app/modelConfig.py](app/modelConfig.py) resolves them; nothing else in the
codebase names a model.

- **Precedence is environment, then file, then error.** `RAG_<ROLE>_PROVIDER` /
  `RAG_<ROLE>_MODEL` override one role so a container can be repointed without a rebuild.
  `.env` deliberately leaves all eight unset, because with them set, editing the file
  would appear to do nothing.
- **Provider and model always move together.** Half-overriding a role — provider from the
  environment, model from the file — is refused, because a model name is not portable
  between vendors and the failure would otherwise arrive as a 404 from the vendor.
- **No defaults.** A default model here would be four vendors' current model names living
  in code and going stale the first time any of them ships a new one — which is exactly
  how `gemini-2.0-flash` came to be pinned in the chunker until Google retired it.
- **`app.main.checkConfiguration` resolves all four at startup** and logs what each
  resolved to and where from, so a missing `[chunker]` fails the deploy rather than a
  worker job hours later.
- The API keys stay in the environment (`llmManager.API_KEY_ENV`). The file is committed;
  keys are not.

`config/` is copied into the image by the `Dockerfile`; `app/modelConfig.py` locates it
relative to the package, not the working directory, because uvicorn, the worker and
pytest all start from somewhere different.

## Persona — what the agent is

[config/prompts.toml](config/prompts.toml) holds named personas. Each is a `systemPrompt`
and an optional `reviewCriteria`; `default` at the top picks the active one.
[app/promptConfig.py](app/promptConfig.py) reads it.

**The service ships selling.** `default = "sales"` — a consultative sales representative
that qualifies the need, connects an answer to a concrete outcome, and closes on one next
step. `default = "support"` is the original neutral retrieval-grounded assistant, kept for
an internal handbook or a policy corpus where a sales register would be actively wrong.

Where a prompt comes from, highest first:

1. the project's own prompt in Firestore, if assigned — `promptStore` resolves that and
   never reads this file;
2. `RAG_DEFAULT_SYSTEM_PROMPT` — a whole prompt in an env var. Predates the file, still
   wins. Suppresses `reviewCriteria` too, since criteria describing one prompt would
   penalise a different one;
3. `RAG_PERSONA` naming a persona in the file;
4. the file's `default`;
5. `FALLBACK_SYSTEM_PROMPT` in `promptConfig` — the neutral assistant, deliberately *not*
   the sales one, on the grounds that if configuration has gone missing the safe thing to
   be is accurate and dull rather than a salesperson improvising without its script.

**Two entry points, and the split is the point.** `validatePersona()` raises and is called
by `checkConfiguration`, so a `RAG_PERSONA` naming something that does not exist stops the
deploy — the failure is otherwise silent, because the service answers perfectly well as
somebody other than the assistant that was configured. `defaultSystemPrompt()` and
`reviewCriteria()` never raise: they are on the path of a live question, and a prompt
lookup must not cost an answer.

**`reviewCriteria` exists because the reviewer otherwise pulls against the persona.** It
grades "does this answer the question", so a flat factual dump that ignores everything the
sales persona was told scores 1.0 and the persona has no feedback loop. The shipped
criteria are worded so honesty outranks selling — see invariant 29. `tests/testPromptConfig.py`
asserts the guardrails in the shipped copy (retrieve before claiming, no invented urgency,
no payment details, never claim to be human) so an edit that drops one fails a test rather
than a customer.

Unlike `config/models.toml` there *is* a built-in fallback here: a default model name goes
stale the moment a vendor retires it, whereas a default prompt is prose with no vendor
behind it and works forever.

## Messaging channels

The same agent, reached from WhatsApp or LINE instead of over HTTP. One thing is
different and it shapes everything else: **the answer does not go back on the response.**

```
POST /api/v1/conversations/{projectId}            start one
POST /api/v1/conversations/{projectId}/{gateway}  say something
GET  /api/v1/conversations/{projectId}/{gateway}  the gateway's handshake
```

`gateway` is `web`, `whatsapp`, `line`, or whatever is registered next. **One route**,
dispatching on the last segment. What differs per gateway is only this:

```
web       question in the body  -> answer in the response
whatsapp  signed platform body  -> 200 now, answer pushed back later
line      signed platform body  -> 200 now, answer pushed back later
```

Everything then runs the same `runTurn`. `web` is deliberately **not** a `Channel` — it
has no signature to verify, no envelope to parse and no outbound API, so implementing
that protocol for it would mean three methods that lie.

Every one of these platforms treats a slow or non-2xx webhook as a failure and
redelivers; Meta disables one that keeps failing. Answering takes up to
`RAG_ANSWER_TIMEOUT_SECONDS`, far past that. So the delivery is verified, acknowledged,
and answered in a `BackgroundTasks` job that pushes the reply back through the platform's
own API.

**Adding a gateway is two steps**: an adapter module implementing `Channel`, and a line in
`registry.CHANNELS`. There is no third — the route is parameterised, so a gateway becomes
reachable the moment it is registered. Nothing else knows the list; the store keys its
config map on the same names and the sender dispatches on them. `getChannel` raising for
an unknown name is what the route turns into a 404 naming the gateways that do exist.

**The project is in the path, not the body.** The one place this API breaks its
everything-in-the-body convention, and it has to: the body is written by the platform and
nothing can be added to it. The URL is the only part of the request this service controls,
because it is what gets pasted into the platform's console — so one deployment serves any
number of projects, each with its own webhook URL and credentials.

**The prefix is a literal segment**, not `/api/v1/{projectId}/{gateway}`. A bare
two-segment pattern under `/api/v1` also matches `/api/v1/document/status` —
`projectId="document", gateway="status"` — and which wins is registration order. The
document and search routes keep the older everything-in-the-body convention: nothing
forces them to move, and changing a working contract to match a constraint that does not
apply to it would be churn.

**Neither platform gives you a conversation id.** WhatsApp identifies a *person*
(`from` / `wa_id`, plus your `phone_number_id`); its `conversation.id` appears only on
outbound status callbacks and is a 24-hour **billing** window that rotates. LINE gives
`source.userId`, stable but scoped to the Official Account. Neither has any notion of a
thread, and neither ever signals that a conversation started or ended — which is exactly
why the `threads` subcollection exists: we mint the `conversationId` and remember it
against the person. A consequence worth knowing: a conversation never ends on its own, so
"start fresh after N hours of silence" is a rule we would have to add.

LINE does hand us `deliveryContext.isRedelivery`. It is carried on `IncomingMessage` for
the log but does **not** gate the answer: a redelivery means the platform did not get a
2xx last time, which usually means the answer was never queued.

**This is the only real authentication in the service.** Every other endpoint trusts an
unverified `serverId`. A webhook is a public URL anyone can find and POST to, and what
arrives is answered by a paid model call, so every delivery is HMAC-checked against the
project's own secret over the *raw bytes* — parsing and re-serialising changes key order
and the signature then fails in a way that reads as a wrong secret. `verify` returns False
when the secret is missing: **fails closed**, because a gateway that accepts everything
until somebody adds a secret looks exactly like one that works.

**A redelivery must not be answered twice.** `_alreadyHandled` claims the platform's
message id in Redis with `SET NX`. That one fails *open* — answering twice is worse than
not answering — which is the opposite of the signature check, deliberately.

Storage is `ragChannels/{projectId}`: a `channels` map (one document, because a project
has a handful of gateways and each config is a few short strings) and a `threads`
subcollection mapping `{channel}:{userId}` to a `conversationId`, so a person's history
follows them between messages. A thread is linked only after a *successful* send — a
conversation they never received an answer from is one they will ask again from.

## Token accounting

Three separately billed models run on the way to one answer, and only one of them is
visible from outside:

```
summariser   folds the conversation, when it has outgrown its budget
agent        answers -- several internal turns, and twice if the review fails
reviewer     grades the draft, on every question
```

`runTurn` opens a `UsageLog`, threads it through all three, and writes it after the turn
is recorded. Storage mirrors the conversation tree:

```
ragUsage/{projectId}                    the project's running total
  conversations/{conversationId}        the conversation's total, by role
    messages/{messageId}                one answered question
```

`messageId` is the assistant's own zero-padded turn index, so a usage row joins straight
onto the `messages/{turnIndex}` document in `ragConversations`.

- **A callback per role, not per request.** The handler aggregates by *model name*, and
  all three roles are usually pointed at the same model — one shared context would merge
  them and lose the breakdown that is the entire point.
- **The callback, not the return value.** `with_structured_output` hands back the parsed
  object and the `AIMessage` carrying `usage_metadata` is gone, so the reviewer and the
  summariser would both report zero if usage were read off what they return.
- **Rollups are `Increment`, leaves are `set`.** Two answers in one conversation at the
  same moment would otherwise both read the old total and one would vanish. The trade is
  that an increment applied twice counts twice, where the message document — written once
  under a deterministic id — is idempotent.
- **Recorded in a `finally`.** A call that raised after the provider produced tokens was
  still billed for them, and an expensive failure is the one worth being able to see.
- **Isolated by contextvar.** Each request runs in its own asyncio task, so concurrent
  answers do not bill each other. `tests/testUsage.py` pins that, because the failure
  would be silent and unexplainable after the fact.
- **Nothing here may fail a request.** The model call is already paid for; a write that
  does not land costs a number in a report.
- `cachedInputTokens` and `reasoningTokens` are kept alongside the totals that contain
  them, because they are priced differently and a total alone cannot be turned into money.
- The **chunker** is absent: it runs during ingestion, against a document rather than a
  conversation, so its cost belongs to a job and not to anybody's message. Nothing records
  it yet.

## Where things run

**There is exactly one shape, and no in-process fallback anywhere.** `REDIS_URL` and
`GCP_PROJECT_ID` are both required; each of `buildJobManager`, `buildProjectStore` and
`buildConversationStore` raises without them, and `app.main.checkConfiguration` calls all of them
in the lifespan so a misconfigured deployment fails at **startup** rather than 500ing every
request behind a green `/health`.

| Needs | Holds | Work runs |
| --- | --- | --- |
| `REDIS_URL` | job table + queue | `python -m app.jobs.worker` |
| `GCP_PROJECT_ID` | `projectId`→`ragDbId` mapping, conversations, prompts | — |

The dict-backed job manager, `InMemoryProjectStore`, `InMemoryConversationStore` and
`InMemoryChunkStore` are all gone. They were two problems, not one: state died with the
process, and the job manager ran chunking on the API's own event loop, so a large document
froze every other request — health check included — and the platform killed a task that was
working. `tests/fakeJobManager.py` keeps the dict version as a *test double*, which is what
it was always suited to; it calls `resolveSubmission` rather than reimplementing the rules.

Redis-without-Firestore is refused because durable jobs resolved through a mapping that
dies on restart is the worst of both: `/document/status` would 404 running jobs, and a
resubmitted project would mint a second `ragDbId` and orphan the first one's vectors.

**Every CPU-bound ingestion step runs in a thread** (`split`, `chunkWithoutAi`,
`enforceEmbedLimit`, `buildChunks`). tiktoken releases the GIL and is nearly all of the
cost; `re` does not, so `split` still blocks briefly — one pass over the text rather than
one per chunk.

Celery was removed as overkill for one node. It bought horizontal scaling, routing, a
result backend and a scheduler, none of which were used, and cost a broker abstraction
plus time limits that silently do not work on Windows.

The agent runs **inside the request**, on the API's event loop — unlike ingestion. It is
bounded by `RAG_ANSWER_TIMEOUT_SECONDS` (120) in `routes.query`, because nothing inside a
model client caps a whole answer.

Chunk store: Pinecone unless `RAG_TEST_MODE` is set, in which case `LocalChunkStore`
writes JSON files. `LocalChunkStore.__init__` *refuses to construct* without the flag,
so no misconfiguration can quietly downgrade a real deployment.

## Invariants — do not break these

1. **Callers name a `projectId`; the `ragDbId` is internal and never on the wire.**
   Resolution happens in `routes.py` and nowhere deeper — every layer inward (job
   manager, claim, chunk store, Pinecone namespace) still keys on `ragDbId` exactly as
   before. Do not thread `projectId` past the route.
2. **A project's `ragDbId` is minted once and never changes.** Mint a second and
   everything ingested under the first is stranded in Pinecone: billable, unreachable,
   invisible. `resolveOrCreate` must therefore be atomic — a Firestore transaction, or
   — or two concurrent first submissions mint two ids.
3. **Only `/document` may create a mapping.** `/search`, `/query`, and `/status` resolve
   read-only, or every mistyped `projectId` leaves an empty database behind forever.
4. **`ragDbId` is random, not derived from `projectId`.** Nothing may recompute one from
   the other; a derived id is one that can never be changed, which forfeits rebuilding a
   project into a fresh namespace, versioning it, or splitting it later.
5. **`jobId` IS the `ragDbId`.** Not a generated id. One job per RAG database.
6. **The claim happens in the API, before dispatch**, and it is atomic (a Firestore
   transaction). If a worker claimed, the 202 would already have been returned and the
   caller could never learn about a 409.
7. **The queue message carries only the `ragDbId`.** The worker re-reads the job from
   the table. A copy in the message would be a second source of truth.
8. **`resolveSubmission` is the single copy of the reuse/conflict rules.** The test double in `tests/fakeJobManager.py`
   manager calls it directly; `RedisJobStore.claim` calls it inside its WATCH/MULTI
   retry. Do not reimplement it per backend, and do not move it into a Lua script —
   drift would surface as the same request accepted on one deployment and refused on
   another.
9. **`jobQueue` uses `BLMOVE` into a processing list, not `BLPOP`.** A plain pop drops
   the id the instant the worker takes it, so a worker killed mid-ingestion loses the
   job with nothing recording it ever existed. `requeueAbandoned` at startup is what
   puts those back — and it assumes **one worker**; two sharing a processing list would
   requeue each other's live jobs.

   **`RAG_QUEUE_POP_TIMEOUT` must stay below `RAG_REDIS_TIMEOUT`** (2 vs 5 today). Two
   clocks run during a `BLMOVE`: the server holding the reply for the pop timeout, and
   the client's own socket read timeout. Both defaulted to 5, the socket usually won, and
   the exception came out of a call the worker makes *before* its `try` — so every idle
   worker died on its first poll, ingestion only worked when a job happened to already be
   queued at startup, and nothing said so: `/document/status` answered `queued` forever
   for a job with nobody left to pick it up. `takeNext` now also catches
   `redis.TimeoutError` and returns None, which is what the caller means by it anyway.
10. **`RAG_STALE_JOB_SECONDS` must exceed the longest legitimate runtime**, and is off
   by default. Celery's `task_time_limit` used to guarantee an upper bound; nothing now
   can kill CPU-bound work, so reclaiming early starts a second ingestion beside a live
   one — the interleaving the conflict check exists to prevent, arriving through the
   check itself.
11. **Pinecone `save` upserts first, then sweeps stale ids.** Clearing first would empty
   the database for the length of the upsert and lose everything if the upsert failed.
   Re-ingesting a *shorter* document must still remove the old tail records.
12. **A failed dispatch releases the claim** (job to FAILED, response 503). A stranded
   claim would 409 that `ragDbId` forever.
13. **`runJob` records processor failures on the job rather than raising** — a dead link
   is a permanent failure, not a transient one to retry. Resubmitting a FAILED job is how
   a retry is requested.
14. **Ingestion writes and search reads must be the same store instance** —
    `getChunkStore()` is `lru_cache`d for exactly that reason.
15. **Startup fails loudly on any misconfiguration**, via `app.main.checkConfiguration`:
    the job manager and all three stores are built in the lifespan. None of those
    constructors touch the network, so this checks configuration rather than
    availability. A config typo must surface on deploy, not on the first request — left
    lazy, every request 500s while `/health` answers `ok` and the platform routes all
    traffic to a task that cannot serve any of it. The counterpart rule is that
    everything *after* startup degrades instead of crashing: see 24, and
    `primeCpuPercent`, which is guarded precisely because it runs in the lifespan.
16. **`/health` must not fail, and must not be slow.** Every machine reading is
    taken independently and dropped on error — a liveness endpoint that 500s because
    it could not stat a filesystem pulls a working service out of a load balancer.
    CPU uses `cpu_percent(interval=None)`, primed once in the lifespan hook, because
    `interval=0.1` would block the event loop on every poll.
17. **The agent's search tool is bound to one `ragDbId` and takes no project
    argument.** That closure is the entire authorisation story for retrieval — with no
    authentication, it is what stops a prompt talking the model into another project's
    documents. Never add a `projectId` parameter to `searchProject`.
18. **The deep agent's host tools stay off.** `create_deep_agent` merges caller tools
    *into* its built-ins, so the only supported way to drop `read_file`/`write_file`/
    `execute`/… is a `HarnessProfile(excluded_tools=…)` registered per provider *name*
    (`google_genai`, not `gemini`). The auto-added general-purpose subagent is disabled
    too. `tests/testAgent.py` builds a real agent per provider and asserts the model sees
    exactly `{"searchProject"}` — that test is the guard against a deepagents upgrade
    silently handing the host back.
19. **The review runs exactly once**, and the retried answer is returned ungraded. A loop
    has no guaranteed exit: an honest "the documents do not cover this" scores badly every
    time, because the reviewer grades the answer and not the corpus.
20. **A tool must not raise into the graph.** An exception out of a tool aborts the run
    and 500s a question the agent could still have answered honestly — `searchProject`
    catches and reports the failure to the model as text instead.
21. **Only a prompt that was actually resolved is cached.** A Firestore failure answers
    with the default but must not write it, or a two-second outage pins the default onto
    a project for a full `RAG_PROMPT_TTL_SECONDS`. "No prompt assigned" *is* cached.
22. **`serverId` is never trusted.** It is logged, and it reaches `resolveSubmission`,
    where same-caller-same-document means REUSE and a different caller means CONFLICT —
    the one place an unverified value still steers behaviour. Do not add anything that
    grants access based on it without adding authentication first.
23. **Conversations are keyed by `projectId`, never by `ragDbId`.** `/query` runs with
   `ragDbId` None, so there would be no key for a project's first conversation; and a
   `ragDbId` is deliberately changeable (invariant 4), so keying on it would orphan
   every conversation the day a project is rebuilt. The `ragDbId` on a conversation document is
   audit only — nothing resolves retrieval from it.
24. **Nothing about storing a conversation may fail a `/query`.** The model call is the
   expensive, irreversible step. An unreachable conversation store answers without history; a
   failed write logs and still returns the answer. Only an unknown `conversationId` is an error
   (404), and that is thrown *before* the model runs.
25. **An unknown `conversationId` is a 404, not a new conversation.** A typo silently
   opening a fresh one is indistinguishable, to the caller, from a model that forgot
   everything. `/conversation/message` additionally *requires* the id, so it can never
   create one.
26. **`appendTurn` is a transaction, and turn indices come from the stored counter.**
   Two questions racing one conversation would otherwise both write `messages/000006` and lose
   an exchange. Do not turn it into a batch.
27. **The summariser runs before the answer, and its failure is never fatal.**
   Summarising afterwards leaves the turn that tripped the budget to be answered with
   the prompt that tripped it. Every failure path falls through to `trimToBudget`, which
   drops the oldest *retrievals* from what is sent and never a message.
28. **A conversation answers with the `systemPrompt` snapshotted onto it**, not the project's
   current one. Re-resolving per turn lets an edit rewrite the instructions earlier
   answers were given under.
29. **In a persona's `reviewCriteria`, honesty outranks selling.** The sales criteria
   explicitly say an answer that declines to invent a price, a feature, a discount or a
   deadline is a *good* answer. Do not reverse that. The reviewer already has one known
   failure mode where an honest "the documents do not cover this" is marked down and
   retried (invariant 19); criteria that rewarded selling harder would turn that into an
   agent whose fastest route to a high score is to make something up, and an invented
   price is the most expensive thing this configuration can produce.

## Conventions

- **camelCase everywhere in Python** — file names (`ragIngestionPipeline.py`), functions
  (`runJob`, `getChunkStore`), locals, and test names (`testAQueryReturnsTheMatchingChunk`).
  Deliberately not PEP 8. Match it.
- **Pydantic models are the one exception**: snake_case fields with a camelCase alias
  generator, so the wire is camelCase and Python is snake_case. Request models set
  `extra="forbid"`.
- **Comments explain *why*, at length, and usually name the failure they prevent.** This
  is the dominant style of the codebase — module docstrings run 10–25 lines and justify
  the design. New code should carry the same density of rationale, not bare "what" notes.
- Protocol plus factory-from-environment is the repeated seam: `ChunkStore` /
  `buildChunkStore`, `ProjectStore` / `buildProjectStore`, `JobStore`,
  `DocumentProcessor`. Add a backend by writing the class and returning it from the factory.
- Blocking SDKs (Firestore, Pinecone, kombu publish) are always wrapped in
  `asyncio.to_thread`.
- Process-wide singletons are **all** `lru_cache`d and built lazily: `getJobManager()`,
  `getProjectStore()`, `getConversationStore()`, `getChunkStore()`. Importing a module no longer
  requires a working environment — that used to be true of `JOB_MANAGER` and was a
  standing footgun. `app.main.checkConfiguration` builds all four during startup, so
  laziness costs nothing in fail-fast behaviour.
- Lines wrap around 100 chars. No linter is configured.
- Tests are `tests/test*.py` (per `pyproject.toml`), use `TestClient` plus
  `app.dependency_overrides`, and mark corpus tests `pytest.mark.slow`.

## Configuration

| Variable | Effect |
| --- | --- |
| `GCP_PROJECT_ID` | Move the job table to Firestore. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account key path. Omit inside GCP. |
| `FIRESTORE_PROJECTS_COLLECTION` | The `projectId` → `ragDbId` mapping. Default `ragProjects`. |
| `FIRESTORE_DATABASE_ID` | A database *named* `default` is not the special `(default)` one — set this if the client claims the database does not exist. |
| `REDIS_URL` | The job table and the worker queue. Requires `GCP_PROJECT_ID`. Unset means everything runs in the API process. |
| `RAG_JOB_TTL_SECONDS` | How long a job record survives its last write. Default 7 days; refreshed on every save. |
| `RAG_REDIS_QUEUE` / `RAG_REDIS_JOB_PREFIX` | Key names. Default `ragQueue` / `ragJob:`. |
| `RAG_QUEUE_POP_TIMEOUT` / `RAG_REDIS_TIMEOUT` | Worker block, and socket timeout. 2s and 5s. The first must stay under the second — see invariant 9. |
| `RAG_STALE_JOB_SECONDS` | Reclaim age for a stuck job. See invariant 6. |
| `PINECONE_API_KEY`, `RAG_PINECONE_INDEX`/`_CLOUD`/`_REGION`/`_EMBED_MODEL` | Index `rag-chunks`, aws/us-east-1, `llama-text-embed-v2` — server-side embedding, so nothing here embeds. |
| `RAG_TEST_MODE`, `RAG_LOCAL_STORE_DIR` | Local JSON chunk store instead of Pinecone. |
| `RAG_MAX_EMBED_TOKENS`, `RAG_AI_CHUNK_CONCURRENCY` | 2048, 8. |
| `RAG_UNRAR_TOOL` | `.rar` needs an external unrar/bsdtar/7z binary on the machine. |
| `TIKTOKEN_CACHE_DIR` | tiktoken downloads its BPE data on first use; point this somewhere that survives restarts. |
| `RAG_HEALTH_DISK_PATH` | Which filesystem `/health` reports on. Defaults to the working directory. |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` | One per provider any role names. A role whose key is missing raises `LlmConfigError`, which is a 503 on `/query`. |
| `RAG_MODEL_CONFIG` | Where `config/models.toml` lives. Defaults to `config/models.toml` beside the `app` package. |
| `RAG_AGENT_PROVIDER` / `RAG_AGENT_MODEL` | Overrides `[agent]`. Both or neither. |
| `RAG_REVIEWER_PROVIDER` / `RAG_REVIEWER_MODEL` | Overrides `[reviewer]`. |
| `RAG_CHUNKER_PROVIDER` / `RAG_CHUNKER_MODEL` | Overrides `[chunker]` — the ingestion chunker. |
| `RAG_REVIEW_THRESHOLD` | Retry below this. Default 0.7; the reviewer prompt is interpolated from it. |
| `RAG_ANSWER_TIMEOUT_SECONDS` | Whole-question bound. Default 120. |
| `TAVILY_API_KEY` | Unset → the agent never sees a web-search tool. |
| `RAG_AGENT_SEARCH_TOP_K` / `RAG_TAVILY_MAX_RESULTS` | 6, 5. |
| `RAG_PERSONA` | Which persona in `config/prompts.toml` is active. Overrides its `default`. An unknown name fails startup. |
| `RAG_PROMPT_CONFIG` | Where `config/prompts.toml` lives. |
| `RAG_DEFAULT_SYSTEM_PROMPT`, `RAG_PROMPT_TTL_SECONDS`, `RAG_PROMPT_CACHE_PREFIX` | A whole prompt, overriding the persona entirely (and suppressing its review criteria); 3600s; `ragPrompt:`. |
| `FIRESTORE_PROMPTS_COLLECTION` / `FIRESTORE_PROJECT_PROMPTS_COLLECTION` | `systemPrompts` / `projectPrompts`. |
| `FIRESTORE_USAGE_COLLECTION` | `ragUsage`. Per-project, per-conversation and per-message token cost. |
| `FIRESTORE_CHANNELS_COLLECTION` | `ragChannels`. Per-project gateway credentials and the platform-user → conversation map. |
| `FIRESTORE_CONVERSATIONS_COLLECTION` | `ragConversations`. The conversation root; see [docs/conversationSchema.md](docs/conversationSchema.md). |
| `RAG_CONVERSATION_CACHE_PREFIX` / `RAG_CONVERSATION_CACHE_TTL_SECONDS` | `ragConversation:`, 3600. The assembled window in Redis. |
| `RAG_CONVERSATION_TTL_SECONDS` | 90 days, written as `expiresAt`. Deletes nothing without a TTL policy on `conversations`, `messages` and `context`. |
| `RAG_CONVERSATION_MAX_MESSAGE_CHARS` / `RAG_CONVERSATION_MAX_PASSAGE_CHARS` | 20000, 4000. Keeps one huge turn away from the 1 MiB document limit. |
| `RAG_CONTEXT_SUMMARY_TOKENS` / `RAG_CONTEXT_KEEP_TURNS` / `RAG_CONTEXT_SUMMARY_MAX_CHARS` | 6000, 4, 6000. When to fold, what survives verbatim, and the cap on the summary. |
| `RAG_SUMMARISER_PROVIDER` / `RAG_SUMMARISER_MODEL` | Overrides `[summariser]`. |

No credential configuration exists — the API authenticates nobody. The keys above are
this service's own credentials for the infrastructure it calls, not its callers'.

## Known gaps and drift

- **Conversations can be created and continued, but not listed, read back, or deleted.**
  `POST /api/v1/conversation` starts one and `/conversation/message` continues it;
  nothing serves the list, the transcript, or a delete. The data is shaped
  for a list (`title`, `lastMessage`, `updatedAt` are denormalised onto the conversation document
  for exactly that) but no endpoint reads them, so a UI can open a conversation and
  continue it, and cannot show the user the ones they already had. Deleting a conversation means
  deleting both subcollections first — Firestore does not remove them with the parent;
  see `deleteConversation` in `scripts/liveConversationCheck.py`.
- **Request bodies are uncapped on `/document`, `/document/status` and `/search`.** The
  conversation routes read their own bodies and cap them at `MAX_WEBHOOK_BODY_BYTES`;
  those three are parsed by FastAPI before this code sees them, and uvicorn has no
  default limit. A service-wide cap belongs in a proxy or an ASGI middleware.
- **Archive expansion has no size budget.** `MAX_ARCHIVE_DEPTH = 1` bounds *nesting* and
  `MAX_DOWNLOAD_BYTES` (200MB) bounds the *download*, but `_analyzeArchiveEntries` calls
  `archive.read(info)` per member with nothing tracking the total decompressed bytes — so
  a single-level archive inside the download limit can still expand without bound and OOM
  the worker. The comment on `MAX_ARCHIVE_DEPTH` already says to raise it "only alongside
  a size budget"; the budget is the part that does not exist.
- **A chunk that does not name its subject is a chunk that will be misattributed.**
  Not a code defect -- a corpus one, and worth recording because it looked exactly
  like a model defect for three runs. `e2e`'s travel corpus had ten paragraphs
  reading "Shoulder season is late March, June, September and November, and it
  costs $4,180 per person" with nothing saying *which trip*. Chunking splits on
  blank lines, so the index held a price with no owner, and the agent duly
  attached it to whichever trip was under discussion -- quoting one journey's real
  price against another. Every figure was genuine and retrieved; only the pairing
  was wrong, which no per-answer grounding check can see, and a real price on the
  wrong item reads to a customer as a quote. Naming the trip in each paragraph
  fixed it. The general rule for any corpus ingested here: **a chunk has to stand
  alone, because retrieval will make it.** `testNoPriceIsAttachedToTheWrongTrip`
  guards it going forward.
- **Nothing reads the usage back.** `projectTotal` and `conversationTotal` exist on the
  store and are tested; no endpoint serves them, and there is no per-message listing, no
  date bucketing and no cost-in-currency conversion — only tokens.
- **Ingestion's token cost is not recorded.** The chunker is by far the highest call
  volume in the service and none of it reaches `ragUsage`, so a project's total is its
  *answering* cost only.
- **A webhook answer is not durable.** It runs in a FastAPI `BackgroundTasks` job in
  the API process, so a restart between the 200 and the send loses that person's answer
  with nothing recording it — and the platform will not redeliver, because it was already
  acknowledged. The ingestion queue is not reusable for this (its `jobId` *is* a
  `ragDbId`, one job per database); a second Redis queue is the fix when it matters.
- **Channel credentials sit in Firestore in plaintext.** No field-level encryption, and an
  access token there is enough to send messages as somebody else's business. Acceptable
  only because nothing real is stored yet — before it is, put them in Secret Manager and
  keep a resource name here.
- **Only text is handled.** Images, stickers, locations and button replies are dropped in
  `parse` rather than answered, and outbound is text only.
- **`expiresAt` is written on every conversation document but deletes nothing** until a TTL
  policy is created in the Firestore console on each of `conversations`, `messages`, `context`.
- **Web search results are not stored on a conversation.** Only `searchProject` retrievals are.
- **No endpoint writes system prompts.** `PromptStore.savePrompt` / `assignPrompt` exist
  and are tested; the Firestore collections are edited by hand today.
- **No authentication or access control of any kind.** Removed deliberately. Every
  project is readable and writable by anyone who can reach the service, and the mapping
  is not scoped by `serverId`, so two callers picking the same project name share one
  database. Must come back before this is exposed to anything.
- **Job eviction is now the Redis TTL** (`RAG_JOB_TTL_SECONDS`, 7 days). A project whose record expired reads as never submitted, and resubmitting starts fresh.
- **No retry classification.** A Pinecone blip and a dead link are both just FAILED.
- **`RagIngestionPipeline.load` only reads plain text**, so anything binary must come in
  through `runText` with text already extracted (which is what `ragProcessor` does).
- **`MAX_ARCHIVE_DEPTH = 1`** in `app/ingestion/documents.py` — one level of archive
  nesting, deliberately. Raising it multiplies the worst-case work per level.
- **`/health`'s machine figures describe the host, not the container.** `psutil` reads
  `/proc`, so under Docker they show host CPU and memory rather than the cgroup limit —
  a 512MB container on a 32GB host reports 32GB and looks idle until it is OOM-killed.
  Making them cgroup-aware means reading `/sys/fs/cgroup` directly.
- **At-least-once delivery is accepted, not solved.** A DONE job is skipped at pickup; a
  redelivery racing a *live* run is not prevented (the runs converge except under AI
  chunking, where two runs can segment differently).
