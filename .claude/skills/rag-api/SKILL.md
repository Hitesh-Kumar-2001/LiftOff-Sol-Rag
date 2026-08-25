---
name: rag-api
description: Context for this repo — an unauthenticated FastAPI RAG service (async document ingestion, Gemini chunking, Pinecone vector store, Redis job table and queue, Firestore project mapping and chat history, a worker process, and a deepagents answering agent over anthropic/openai/groq/gemini). Load before reading, changing, debugging, or extending any code under app/, api/, scripts/, or tests/, or when asked how ingestion, jobs, projects, search, chats, the agent, or deployment work here.
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
uv run pytest                             # full suite
uv run pytest -m "not slow"               # skip the ~2min corpus tests
python -m app.jobs.worker                 # the ingestion worker (needs REDIS_URL)
docker compose up --build                 # api + worker + redis
```

Layout: `app/api` (routes, schemas), `app/agent` (LLM manager, prompt store, tools,
reviewer, summariser, the loop), `app/jobs` (record, manager, Redis store, queue,
worker), `app/ingestion` (documents, selector, pipeline, processor), `app/stores`
(chunk stores, project mapping, chat store), `app/infra` (Redis/Firestore clients,
machine stats).

`scripts/live*Check.py` run against real infrastructure and are deliberately outside
the test suite (they cost money / need network): `liveFirestoreCheck` (project mapping),
`liveChatCheck` (chat store — the append transaction and the summary range queries),
`livePineconeCheck` (store), `liveIngestionCheck <url>` (whole pipeline, real Gemini
and Pinecone). They read `.env`; `tests/conftest.py` deliberately unsets
`GCP_PROJECT_ID` so the *suite* never does.

## Endpoints

All are POST, taking their input in the body. **Every endpoint takes a `projectId` and
an unverified `serverId`; none takes or returns a `ragDbId`.** There is no 401 anywhere
in the API.

| Route | Does |
| --- | --- |
| `POST /api/v1/query` | Runs the answering agent (`app/agent/`). Takes an optional `chatId` and always returns one — see *Chats*. 404 unknown `chatId`, 502 provider failure, 503 no model configured, 504 past `RAG_ANSWER_TIMEOUT_SECONDS`. |
| `POST /api/v1/document` | Resolves-or-creates the project's database, claims it, queues ingestion, returns 202. 409 on conflict, 503 on dispatch failure. |
| `POST /api/v1/document/status` | Job status. Returns `projectId` normally, or `documentLink` when the strategy was RAW. 404 if the project or its job is unknown. |
| `POST /api/v1/search` | Retrieval only — the matching chunks, not a generated answer. An unresolved project gives empty hits, not a 404. |
| `GET /`, `GET /health` | Liveness. `/health` also reports CPU / memory / disk — informational only, never changes the outcome, and a reading that fails is omitted rather than 500ing. |

## Module map

**API layer** — [app/main.py](app/main.py) builds the app (lifespan only awaits
in-flight jobs on shutdown) and owns the 422 handler that allowlists `type`/`loc`/`msg`
so a malformed request is not echoed back. [app/api/routes.py](app/api/routes.py) has the four
routes and is where `projectId` becomes `ragDbId`. [app/api/schemas.py](app/api/schemas.py) is
the wire contract. [app/stores/projectStore.py](app/stores/projectStore.py) holds the mapping —
`ProjectStore` protocol, in-memory and Firestore implementations, `buildProjectStore()`
factory. [app/infra/machineStats.py](app/infra/machineStats.py) reads CPU/memory/disk via `psutil`
for `/health`.

**Agent** — [app/agent/agent.py](app/agent/agent.py) is the entry point
(`answerQuestion`): resolve prompt → build tools → run → review once → at most one retry,
continued from the first run's transcript. [app/agent/llmManager.py](app/agent/llmManager.py)
turns `(provider, model)` into a `BaseChatModel` for anthropic/openai/groq/gemini; vendor
imports live *inside* the builder functions, `model` is required, and clients are
`lru_cache`d because each owns a connection pool.
[app/agent/promptStore.py](app/agent/promptStore.py) resolves a project's system prompt from
two Firestore collections and caches the *resolved text keyed by projectId* in Redis.
[app/agent/tools.py](app/agent/tools.py) builds `searchProject` (closed over one `ragDbId`
*and* an optional `searchLog` that records what it retrieved) and Tavily.
[app/agent/reviewer.py](app/agent/reviewer.py) grades an answer 0.0–1.0 with an
optional suggestion. [app/agent/summariser.py](app/agent/summariser.py) renders a stored
conversation into prompt + messages, and folds it down when it outgrows
`RAG_CONTEXT_SUMMARY_TOKENS`.

**Chats** — [app/stores/chatStore.py](app/stores/chatStore.py) is the storage:
`ragChats/{projectId}/chats/{chatId}` with `messages` and `context` subcollections,
Redis in front, `ChatStore` protocol + in-memory + Firestore + factory. Full schema and
rationale in [docs/chatSchema.md](docs/chatSchema.md) — read that before changing
anything about how a conversation is stored.

**Auth** — none. There is no auth module; see the note at the top.

**Jobs** — [app/jobs/job.py](app/jobs/job.py) is the `Job` record, `JobStatus`, `runJob`, the
`JobStore` protocol, and `resolveSubmission` (the shared NEW/REUSE/CONFLICT rules).
[app/jobs/jobManager.py](app/jobs/jobManager.py) picks the manager from the environment and
holds the in-memory one. [app/jobs/redisJobStore.py](app/jobs/redisJobStore.py) is the table —
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
- otherwise → **AI** — Gemini (`gemini-3.5-flash-lite`, pinned) picks boundaries, one
  call per blank-line section, 8 concurrent, `response_schema=list[str]`. A section whose
  answer will not parse falls back to non-AI chunking rather than losing the document.
  `>= 100000` tokens logs a warning and proceeds.

Every strategy's output then passes `enforceEmbedLimit` — anything over
`RAG_MAX_EMBED_TOKENS` (2048) is re-split, because Pinecone's embedder silently truncates
the tail instead of erroring.

## Where things run

`REDIS_URL` picks the whole shape. **There is no silent fallback** — Redis without
`GCP_PROJECT_ID` raises at startup.

| `REDIS_URL` | `GCP_PROJECT_ID` | Job table | Work runs | Survives restart |
| --- | --- | --- | --- | --- |
| — | — | dict in-process | API's event loop | no |
| — | set | dict in-process | API's event loop | mapping does; jobs do not |
| set | set | Redis | `python -m app.jobs.worker` | yes |

Redis-without-Firestore is refused because durable jobs resolved through a mapping that
dies on restart is the worst of both: `/document/status` would 404 running jobs, and a
resubmitted project would mint a second `ragDbId` and orphan the first one's vectors.

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
   the in-memory lock — or two concurrent first submissions mint two ids.
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
8. **`resolveSubmission` is the single copy of the reuse/conflict rules.** The in-memory
   manager calls it directly; `RedisJobStore.claim` calls it inside its WATCH/MULTI
   retry. Do not reimplement it per backend, and do not move it into a Lua script —
   drift would surface as the same request accepted on one deployment and refused on
   another.
9. **`jobQueue` uses `BLMOVE` into a processing list, not `BLPOP`.** A plain pop drops
   the id the instant the worker takes it, so a worker killed mid-ingestion loses the
   job with nothing recording it ever existed. `requeueAbandoned` at startup is what
   puts those back — and it assumes **one worker**; two sharing a processing list would
   requeue each other's live jobs.
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
15. **Startup fails loudly** on a misconfigured job backend. A config typo should
    surface on deploy, not on the first request.
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
23. **Chats are keyed by `projectId`, never by `ragDbId`.** `/query` runs with
   `ragDbId` None, so there would be no key for a project's first conversation; and a
   `ragDbId` is deliberately changeable (invariant 4), so keying on it would orphan
   every conversation the day a project is rebuilt. The `ragDbId` on a chat document is
   audit only — nothing resolves retrieval from it.
24. **Nothing about storing a conversation may fail a `/query`.** The model call is the
   expensive, irreversible step. An unreachable chat store answers without history; a
   failed write logs and still returns the answer. Only an unknown `chatId` is an error
   (404), and that is thrown *before* the model runs.
25. **An unknown `chatId` is a 404, not a new chat.** A typo silently opening a fresh
   conversation is indistinguishable, to the caller, from a model that forgot everything.
26. **`appendTurn` is a transaction, and turn indices come from the stored counter.**
   Two questions racing one chat would otherwise both write `messages/000006` and lose
   an exchange. Do not turn it into a batch.
27. **The summariser runs before the answer, and its failure is never fatal.**
   Summarising afterwards leaves the turn that tripped the budget to be answered with
   the prompt that tripped it. Every failure path falls through to `trimToBudget`, which
   drops the oldest *retrievals* from what is sent and never a message.
28. **A chat answers with the `systemPrompt` snapshotted onto it**, not the project's
   current one. Re-resolving per turn lets an edit rewrite the instructions earlier
   answers were given under.

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
- Process-wide singletons: `JOB_MANAGER` is built at import time, so **env vars must be
  set before importing `app.jobs.jobManager`** — the live scripts set them at the top of the
  file for this reason. `getChunkStore()` and `getProjectStore()` are `lru_cache`d and
  built lazily instead.
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
| `RAG_QUEUE_POP_TIMEOUT` / `RAG_REDIS_TIMEOUT` | Worker block, and socket timeout. Default 5s each. |
| `RAG_STALE_JOB_SECONDS` | Reclaim age for a stuck job. See invariant 6. |
| `PINECONE_API_KEY`, `RAG_PINECONE_INDEX`/`_CLOUD`/`_REGION`/`_EMBED_MODEL` | Index `rag-chunks`, aws/us-east-1, `llama-text-embed-v2` — server-side embedding, so nothing here embeds. |
| `GEMINI_API_KEY`, `RAG_GEMINI_MODEL` | AI chunking. |
| `RAG_TEST_MODE`, `RAG_LOCAL_STORE_DIR` | Local JSON chunk store instead of Pinecone. |
| `RAG_MAX_EMBED_TOKENS`, `RAG_AI_CHUNK_CONCURRENCY` | 2048, 8. |
| `RAG_UNRAR_TOOL` | `.rar` needs an external unrar/bsdtar/7z binary on the machine. |
| `TIKTOKEN_CACHE_DIR` | tiktoken downloads its BPE data on first use; point this somewhere that survives restarts. |
| `RAG_HEALTH_DISK_PATH` | Which filesystem `/health` reports on. Defaults to the working directory. |
| `ANTHROPIC_API_KEY` | Default agent + reviewer provider. Without it `/query` is 503. |
| `OPENAI_API_KEY` / `GROQ_API_KEY` | Only for those providers. Gemini reuses `GEMINI_API_KEY`. |
| `RAG_AGENT_PROVIDER` / `RAG_AGENT_MODEL` | Default `anthropic` / `claude-opus-5`. A non-default provider **must** name a model. |
| `RAG_REVIEWER_PROVIDER` / `RAG_REVIEWER_MODEL` | Same defaults, configured separately. |
| `RAG_REVIEW_THRESHOLD` | Retry below this. Default 0.7; the reviewer prompt is interpolated from it. |
| `RAG_ANSWER_TIMEOUT_SECONDS` | Whole-question bound. Default 120. |
| `TAVILY_API_KEY` | Unset → the agent never sees a web-search tool. |
| `RAG_AGENT_SEARCH_TOP_K` / `RAG_TAVILY_MAX_RESULTS` | 6, 5. |
| `RAG_DEFAULT_SYSTEM_PROMPT`, `RAG_PROMPT_TTL_SECONDS`, `RAG_PROMPT_CACHE_PREFIX` | Default prompt, 3600s, `ragPrompt:`. |
| `FIRESTORE_PROMPTS_COLLECTION` / `FIRESTORE_PROJECT_PROMPTS_COLLECTION` | `systemPrompts` / `projectPrompts`. |
| `FIRESTORE_CHATS_COLLECTION` | `ragChats`. The chat root; see [docs/chatSchema.md](docs/chatSchema.md). |
| `RAG_CHAT_CACHE_PREFIX` / `RAG_CHAT_CACHE_TTL_SECONDS` | `ragChat:`, 3600. The assembled window in Redis. |
| `RAG_CHAT_TTL_SECONDS` | 90 days, written as `expiresAt`. Deletes nothing without a TTL policy on `chats`, `messages` and `context`. |
| `RAG_CHAT_MAX_MESSAGE_CHARS` / `RAG_CHAT_MAX_PASSAGE_CHARS` | 20000, 4000. Keeps one huge turn away from the 1 MiB document limit. |
| `RAG_CONTEXT_SUMMARY_TOKENS` / `RAG_CONTEXT_KEEP_TURNS` / `RAG_CONTEXT_SUMMARY_MAX_CHARS` | 6000, 4, 6000. When to fold, what survives verbatim, and the cap on the summary. |
| `RAG_SUMMARISER_PROVIDER` / `RAG_SUMMARISER_MODEL` | Same defaults as the agent; configured separately because this is the call most worth pointing at a small fast model. |

No credential configuration exists — the API authenticates nobody. The keys above are
this service's own credentials for the infrastructure it calls, not its callers'.

## Known gaps and drift

- **Chats have no list, read-back, or delete endpoint.** The data is shaped for a list
  (`title`, `lastMessage`, `updatedAt` are denormalised onto the chat document for
  exactly that) but nothing serves it. Deleting a chat means deleting both
  subcollections first — Firestore does not remove them with the parent; see
  `deleteChat` in `scripts/liveChatCheck.py`.
- **`expiresAt` is written on every chat document but deletes nothing** until a TTL
  policy is created in the Firestore console on each of `chats`, `messages`, `context`.
- **Web search results are not stored on a chat.** Only `searchProject` retrievals are.
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
- **`Dockerfile`, `.dockerignore`, and `docker-compose.yml` are untracked.** Compose
  defines `api`, `worker`, and `redis`; `worker` exits immediately when `REDIS_URL` is
  unset, which is correct — the API is ingesting instead.
- **`/health`'s machine figures describe the host, not the container.** `psutil` reads
  `/proc`, so under Docker they show host CPU and memory rather than the cgroup limit —
  a 512MB container on a 32GB host reports 32GB and looks idle until it is OOM-killed.
  Making them cgroup-aware means reading `/sys/fs/cgroup` directly.
- **At-least-once delivery is accepted, not solved.** A DONE job is skipped at pickup; a
  redelivery racing a *live* run is not prevented (the runs converge except under AI
  chunking, where two runs can segment differently).
