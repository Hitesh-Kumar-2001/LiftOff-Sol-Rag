# RAG API

A FastAPI service that ingests documents into per-project RAG databases and
searches them. Managed with `uv`.

> **There is no authentication.** Every endpoint takes a `serverId`, but nothing
> verifies it — there is no secret, no signature, and no registry behind it. It is
> a label for the log. Anyone who can reach this API can ingest into any project
> and read any project's chunks, under any name they choose. Do not expose it
> publicly without putting authentication in front of it first.

**Status:** ingestion, chunking, embedding, retrieval, and answering are built
and tested. What is not is listed under
[What isn't built yet](#what-isnt-built-yet) — chat history is the main gap:
every question is a single turn today.

## Run locally

```powershell
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

## Health

`GET /health` answers liveness, and reports what the machine is doing:

```json
{
  "status": "ok",
  "machine": {
    "cpu":    {"usedPercent": 4.8, "count": 16},
    "memory": {"totalBytes": 16309932032, "availableBytes": 1582424064,
               "usedBytes": 14727507968, "usedPercent": 90.3},
    "disk":   {"path": "/app", "totalBytes": 1081101176832,
               "freeBytes": 1079174234112, "usedBytes": 1926942720,
               "usedPercent": 0.2}
  }
}
```

`status` is unchanged and always first — anything already polling this endpoint
is reading that field, and the machine figures are an addition to it. They never
change the outcome: a machine under load is still `ok`, because this endpoint
answers *am I running*, and failing it would pull a working service out of a load
balancer. Nor can reading them fail the request — each of the three is taken
independently, and one the platform refuses is simply left out of the response
rather than sent as null. If none can be read, `machine` disappears entirely.

Two caveats before reading anything into the numbers, both in
[app/infra/machineStats.py](app/infra/machineStats.py):

- **They describe the machine, not this process — and in a container, the
  *host*.** `psutil` reads `/proc`, which inside Docker still shows the host's
  CPU count and total memory rather than the cgroup limit the container was
  given. A container capped at 512MB on a 32GB host reports 32GB here and looks
  idle right up until the OOM killer takes it.
- **On any request-scoped host the CPU figure is close to meaningless**, since
  the sandbox is frozen between requests.

The disk figures describe the filesystem the process is running from. Point them
somewhere else with `RAG_HEALTH_DISK_PATH`.

## Asking a question

`POST /api/v1/query` takes the calling server's id and the question in one JSON
body:

```json
{
  "serverId": "billing-service",
  "question": "What is the refund window?",
  "projectId": "handbook"
}
```

```json
{
  "answer": "...",
  "projectId": "handbook"
}
```

| Status | Meaning |
| ------ | ------- |
| 200 | Answer returned |
| 422 | A field is missing, blank, or unexpected |
| 502 | The model provider failed. The request was fine; the failure is behind the API |
| 503 | No model is configured — usually a missing or wrong API key |
| 504 | The agent did not finish inside `RAG_ANSWER_TIMEOUT_SECONDS` |

`serverId` says *who is asking*, for the log and nothing else; `question` +
`projectId` say *what to answer and from where*.

A 502's `detail` names the exception type and nothing else. Provider errors
quote the prompt back, and the prompt is another project's configuration; the
full error goes to the log instead.

### How an answer is produced

```
question ─► system prompt (Firestore, cached in Redis)
              │
              ▼
            agent ⇄ searchProject (this project only) ⇄ Tavily (if configured)
              │
              ▼
            review ──► score ≥ threshold ──► answer
              │
              └──► below ──► one more attempt, continued from the same
                             transcript ──► answer (not reviewed again)
```

The agent is a [deepagents](https://github.com/langchain-ai/deepagents) deep
agent, in [app/agent/](app/agent/). Four things about it are deliberate:

- **Retrieval is the agent's decision, not the route's.** It gets a search tool
  and calls it when the question needs it. A follow-up like "shorter, please"
  should not cost a vector search.
- **The search tool is bound to one `ragDbId` and takes no project argument.**
  That is the entire authorisation story for retrieval: no prompt reaching the
  agent can talk it into reading another project's documents. Given there is no
  authentication, this is the boundary worth knowing about.
- **The built-in filesystem and shell tools are switched off**, along with the
  auto-added general-purpose subagent. A deep agent ships with
  `read_file`/`write_file`/`execute` and friends, which are right for a coding
  agent and wrong at the end of an unauthenticated endpoint. The agent is left
  holding exactly `searchProject`, and web search when `TAVILY_API_KEY` is set.
  [tests/testAgent.py](tests/testAgent.py) builds a real agent per provider and
  asserts it.
- **The review runs exactly once.** A low score buys one more attempt, with the
  reviewer's suggestion and continued from the same transcript, and that second
  answer is returned ungraded. A loop has no guaranteed exit: on a question the
  documents genuinely cannot answer, an honest "the documents do not cover this"
  scores badly every time, because the reviewer grades the answer and not the
  corpus.

### System prompts

Each project answers with its own system prompt, in two Firestore collections:

| Collection | Document | Holds |
| --- | --- | --- |
| `systemPrompts` | `{promptId}` | `{"prompt": "..."}` — the text, stored once |
| `projectPrompts` | `{projectId}` | `{"promptId": "..."}` — which prompt a project uses |

Split so twenty projects can share one prompt and be re-tuned by editing one
document. Redis caches the **resolved text keyed by projectId**, so the hot path
is a single `GET` rather than both hops; the cost is that editing a prompt takes
up to `RAG_PROMPT_TTL_SECONDS` to be seen everywhere. Re-*assigning* a project's
prompt invalidates immediately — that is a deliberate switch, and waiting an
hour would read as a failed call. Nothing is cached in process memory, so a
second API instance sees the first's invalidation.

A prompt that could not be read falls back to the default and is **not** cached,
so a momentary Firestore outage cannot pin the default onto a project for a full
TTL. A project that simply has no prompt assigned is cached, because that is a
real answer.

There is no endpoint for writing prompts yet — `PromptStore.savePrompt` and
`assignPrompt` exist and are tested, but today the collections are edited
directly. See [What isn't built yet](#what-isnt-built-yet).

## Projects and databases

A caller names a **project** and never a database. Internally every project
resolves to a `ragDbId` — the job's id, the Pinecone namespace, and what the
conflict check claims — and that id never appears on the wire.

They are one-to-one today, and deliberately not the same string. A project is a
caller's permanent name for "my documents"; a `ragDbId` is where those documents
happen to live right now, and that should stay ours to change: rebuilding a
project into a fresh namespace, versioning it, or one day splitting it across
several databases are all changes to the mapping alone, invisible to whoever is
calling. The id is random rather than derived precisely so that nothing can
recompute one from the other and quietly depend on the two staying in step.

The mapping lives in [app/stores/projectStore.py](app/stores/projectStore.py) — in Firestore
when `GCP_PROJECT_ID` is set, in memory otherwise, the same durability switch the
job table uses. It is **write-once**: resolving a project must return the same
`ragDbId` every time, because handing back a different one leaves everything
already ingested under the old id sitting in Pinecone, billable and unreachable.
Only `POST /api/v1/document` may create one; everything else resolves read-only,
so a mistyped `projectId` on a search cannot leave an empty database behind.

## Submitting a document

`POST /api/v1/document` queues a document for ingestion and returns right away
— it does not wait for processing:

```json
{
  "serverId": "billing-service",
  "documentLink": "https://example.com/handbook.pdf",
  "projectId": "handbook"
}
```

```json
{
  "projectId": "handbook",
  "status": "queued"
}
```

| Status | Meaning |
| ------ | ------- |
| 202 | Job created and running in the background |
| 409 | A different document is already being ingested into this project |
| 422 | A field is missing, blank, or unexpected |
| 503 | The queue was unreachable; nothing was started, so resubmit as-is |

No job id comes back, deliberately. A job is keyed by the `ragDbId` behind the
project — a job exists to populate one RAG database, so there is nothing else
meaningful to key it by — and that id is internal. The `projectId` the caller
already sent is what `/document/status` takes, so it is the only id worth
returning. A project's first submission is what brings its database into
existence; every later one resolves to the same database and lands on the same
job. Deduplicating, queueing, or cancelling on resubmission is a policy decision
for later.

What "processing" a document means beyond metadata extraction — chunking,
embedding, writing into the `ragDbId` — is not fully decided yet. See
[app/ingestion/documents.py](app/ingestion/documents.py): `DocumentProcessor` is the contract real
ingestion plugs into; `StubDocumentProcessor` just marks the job done without
doing anything. `POST /api/v1/document/status` polls a job's progress.

Where jobs live depends on configuration (see [Where jobs run](#where-jobs-run));
with `REDIS_URL` unset they are held in memory and do not
survive a restart. On shutdown, the app
*waits* for any job still running rather than cancelling it — a job mid-download
or mid-analysis has no safe halfway point to stop at. This can't hang forever:
a download is capped at `DOWNLOAD_TIMEOUT_SECONDS`, so every job finishes (or
fails) on its own well within that. There is no job eviction yet — old jobs
stay in memory until the process restarts; when that's needed it'll be its own
endpoint, not part of submission.

A `.zip` or `.rar` is searched recursively — an archive containing another
archive is unpacked in turn, however many levels deep (capped at
`MAX_ARCHIVE_DEPTH`, currently 1, to bound a maliciously nested archive). A
nested file's `filename` carries its full path, e.g.
`outer.zip/inner.rar/doc.pdf`, so the origin stays visible even though the
result is a flat list. `DocumentMetadata` also totals `pageCount`,
`imageCount`, `tableCount`, and `tokenCount` across every file found, at every
depth.

Token counts use `tiktoken`'s `cl100k_base` encoding — a general-purpose proxy
for "how much content is here" rather than a count tied to whichever model
ends up doing embedding or generation. `tiktoken` fetches that encoding's data
over the network the first time any process asks for it, then caches it on
disk; a deployment with no outbound network on a cold start (or a wiped temp
dir between restarts) will fail to count tokens until it succeeds once. That
failure doesn't fail the file — `tokenCount` is `null` for that file rather
than the whole analysis erroring out. Set `TIKTOKEN_CACHE_DIR` to a path that
survives restarts if this matters for your deployment.

## How it works

### The pieces

| File | Responsibility |
| ---- | -------------- |
| [app/main.py](app/main.py) | Builds the app, and waits for in-flight jobs on shutdown |
| [app/api/routes.py](app/api/routes.py) | The endpoints: resolve the project, then answer or queue |
| [app/stores/projectStore.py](app/stores/projectStore.py) | `projectId` → `ragDbId`, and nothing else. The indirection stops at the route |
| [app/jobs/jobManager.py](app/jobs/jobManager.py) | What the API talks to for jobs — create, look up by id, shut down cleanly. Picks which manager from the environment |
| [app/jobs/job.py](app/jobs/job.py) | A job's data (`Job`, `JobStatus`), `runJob` which executes one, and `resolveSubmission` — the reuse/conflict rules every manager shares |
| [app/jobs/redisJobStore.py](app/jobs/redisJobStore.py) | The job table in Redis. Storage only; the claim is a WATCH/MULTI transaction |
| [app/jobs/queuedJobManager.py](app/jobs/queuedJobManager.py) | Claims the id, then hands it to the queue. Nothing runs in the API |
| [app/jobs/jobQueue.py](app/jobs/jobQueue.py) | The Redis list between API and worker, and its crash recovery |
| [app/jobs/worker.py](app/jobs/worker.py) | The process on the far end: take an id, ingest, write the status back |
| [app/ingestion/documents.py](app/ingestion/documents.py) | The `DocumentProcessor` contract, plus the real (and stub) implementations |
| [app/api/schemas.py](app/api/schemas.py) | The wire contract — camelCase JSON in and out, snake_case in Python |
| [app/agent/agent.py](app/agent/agent.py) | The answering loop: run, review once, retry at most once. Also what switches the host tools off |
| [app/agent/llmManager.py](app/agent/llmManager.py) | `(provider, model)` → a chat model object. The only file that knows which SDK backs a provider |
| [app/agent/promptStore.py](app/agent/promptStore.py) | A project's system prompt: Firestore for the record, Redis for the round trip |
| [app/agent/tools.py](app/agent/tools.py) | What the agent may reach for — this project's documents, and web search |
| [app/agent/reviewer.py](app/agent/reviewer.py) | Grades a draft answer, and why that happens exactly once |
| [app/infra/machineStats.py](app/infra/machineStats.py) | CPU, memory, and disk for `/health`. Nothing here may fail the health check |

### A request, start to finish

1. **Validation.** FastAPI checks the body against `QueryRequest` before any of
   our code runs. A missing, blank, over-long, or unexpected field is a 422 —
   the handler never sees a malformed request.
2. **Logging.** `serverId` is written to the log line for the request. Nothing
   checks it; see the warning at the top of this file.
3. **Resolution.** The `projectId` resolves to its `ragDbId`, read-only —
   asking a question never creates a database. A project nothing was ingested
   into still answers; the agent simply gets no search tool.
4. **Answering.** The agent runs with that project's system prompt and a search
   tool bound to that one `ragDbId`, is reviewed once, and gets at most one
   retry. Bounded by `RAG_ANSWER_TIMEOUT_SECONDS`.
5. **Response.** `QueryResponse` is serialized back to camelCase.

### Where jobs run

Ingestion is minutes of work — download, extract, chunk (sometimes through
Gemini), then a few hundred Pinecone upserts — kicked off by a request that has
to return in milliseconds. Where that work runs, and where the job table lives,
are two separate decisions, and the environment picks both:

| `REDIS_URL` | `GCP_PROJECT_ID` | Job table | Work runs | Survives a restart |
| --- | --- | --- | --- | --- |
| — | — | dict in the process | on the API's event loop | no |
| — | set | dict in the process | on the API's event loop | the mapping does; jobs do not |
| set | set | Redis | on the worker | yes |

Redis without a GCP project **fails at startup** rather than falling back. Redis makes
the job table shared, but the `projectId` → `ragDbId` mapping is a separate store, and
in memory it lives only in the API process — jobs would survive a restart while the
mapping that resolves them did not, so `/document/status` would 404 running jobs and a
resubmitted project would mint a second database, orphaning the first one's vectors. A
config mistake should surface on deploy, not in production.

```
POST /document ─► resolve project ─► claim ragDbId ─► enqueue id ─► 202
                                        (atomic)      │
                                                      ▼
                                              worker ─► read job ─► ingest ─► write status
                                                                                  │
POST /document/status ──────────────────────────────────────────► read job ◄──────┘
```

Two things about that shape are deliberate:

- **The claim happens in the API, before dispatch.** If the worker claimed, the
  API would have returned 202 before anything checked for a conflict, and the
  caller would never learn its document was refused. The 409 has to be decidable
  while the request is still open.
- **The message carries only the `ragDbId`.** The worker reads the rest from the
  table, which is the record that stays current. A message carrying a copy of the
  job would be a second source of truth, stale the moment anything writes.

Running a worker:

```powershell
python -m app.jobs.worker
```

Run **one**. Crash recovery moves an id to a processing list while it is being worked
and requeues whatever is left there at startup; two workers sharing that list would
requeue each other's live jobs. Give the worker the same environment as the API — it
needs the Redis, Pinecone, Gemini, and Firestore credentials, since it is the process
that actually does the work. It stops on SIGINT/SIGTERM once the current job finishes.

[scripts/liveFirestoreCheck.py](scripts/liveFirestoreCheck.py) checks the project
mapping against real Firestore; it needs only credentials.

> **A project can hold more than one database, and one *named* `default` is not
> the special `(default)` one.** If the client reports `The database (default)
> does not exist for project X` while the console plainly shows a database, that
> is the reason — set `FIRESTORE_DATABASE_ID` to the name shown there.

Celery used to sit here and was removed as overkill for a single node: it bought
horizontal scaling, routing, a result backend and a scheduler, none of which were used,
and cost a broker abstraction plus time limits that silently do not work on Windows.
What was actually wanted — get the work off the API process, and do not lose it if that
process dies — is what [app/jobs/jobQueue.py](app/jobs/jobQueue.py) does in about
twenty lines.

Jobs are not retried automatically. `runJob` records a processor failure on the
job rather than raising, so a document that can never succeed (dead link,
unsupported type) is written back as `FAILED` with the reason instead of being
re-attempted as though it were a transient fault. Resubmitting a `FAILED` job is
how a retry is requested.

### When something breaks mid-flight

The claim is written before the work is dispatched, which means a failure in
between could strand the `ragDbId` — claimed, with nothing running and nothing
coming to run it. Only a finished or failed job frees an id, so that would be a
permanent 409 on a database nobody is using. Two things prevent it:

- **A failed dispatch releases the claim.** The job is recorded `FAILED` with
  the reason and the request gets a **503** — not a 500, because the request was
  fine and resubmitting it unchanged is the right response.
- **A job stuck past `RAG_STALE_JOB_SECONDS` can be reclaimed.** Off unless a
  threshold is set. Nothing can hard-kill CPU-bound work — a supervisor can
  SIGKILL the worker, but the application cannot stop a job mid-parse — so the
  threshold is a judgement about the longest a document could legitimately take.
  It must stay above that: reclaiming early starts a second ingestion alongside a
  live one, which is the interleaving the conflict check exists to prevent,
  arriving through the check itself. Failing that, a job whose worker died holds
  its project until the record's TTL expires.

Two things are known and accepted rather than fixed:

- **Delivery is at-least-once.** A worker killed between finishing and acking
  leaves its message to be handed out again. A job already `DONE` at pickup is
  skipped, which covers that case. A redelivery racing a *live* run is not
  prevented — both workers ingest the same document, and because record ids come
  from chunk position they converge, except under AI chunking, where two runs can
  segment the document differently. Preventing it needs a renewable lease, and a
  lease would break the crash recovery that late acks exist to provide.
- **Nothing can hard-kill a running job.** A supervisor can SIGKILL the worker,
  but the application cannot stop a job mid-parse, which is why
  `RAG_STALE_JOB_SECONDS` is off by default and must exceed the longest a
  document could legitimately take.

## Configuration

| Variable | Effect |
| -------- | ------ |
| `GCP_PROJECT_ID` | Set to keep the job table in Firestore instead of process memory. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service account key. Omit inside GCP, where the platform supplies credentials. |
| `FIRESTORE_PROJECTS_COLLECTION` | Collection holding the `projectId` → `ragDbId` mapping. Default `ragProjects`. |
| `FIRESTORE_DATABASE_ID` | Which database in the project. Leave unset for the special `(default)` one. |
| `REDIS_URL` | The job table and the worker queue, e.g. `redis://localhost:6379/0`. Requires `GCP_PROJECT_ID`. Unset runs everything in the API process. |
| `RAG_JOB_TTL_SECONDS` | How long a job record survives its last write. Default 7 days, refreshed on every save. |
| `RAG_REDIS_QUEUE` / `RAG_REDIS_JOB_PREFIX` | Redis key names. Default `ragQueue` and `ragJob:`. |
| `RAG_QUEUE_POP_TIMEOUT` | How long the worker blocks before checking for a stop signal. Default 5s. |
| `RAG_REDIS_TIMEOUT` | Socket timeout, so an unreachable Redis fails fast instead of hanging a request. Default 5s. |
| `RAG_STALE_JOB_SECONDS` | Age past which a stuck job's `ragDbId` may be reclaimed. **Off by default** — nothing can hard-kill CPU-bound work now, so this must exceed the longest a job could legitimately run. |
| `RAG_HEALTH_DISK_PATH` | Which filesystem `/health` reports on. Defaults to the working directory. |

### The agent

| Variable | Effect |
| -------- | ------ |
| `ANTHROPIC_API_KEY` | The default provider's key. Without it, `/query` answers 503. |
| `OPENAI_API_KEY` / `GROQ_API_KEY` / `GEMINI_API_KEY` | Only needed for the providers actually named below. `GEMINI_API_KEY` is shared with the chunking pipeline. |
| `RAG_AGENT_PROVIDER` / `RAG_AGENT_MODEL` | Which model answers. Default `anthropic` / `claude-opus-5`. Naming a non-default provider **requires** a model — a model name is not portable between vendors. |
| `RAG_REVIEWER_PROVIDER` / `RAG_REVIEWER_MODEL` | Which model grades. Same defaults, set separately so the judge can be a cheaper model without moving the agent. |
| `RAG_REVIEW_THRESHOLD` | Score below which an answer is retried once. Default `0.7`. The reviewer's prompt is interpolated from this, so the two cannot disagree. |
| `RAG_ANSWER_TIMEOUT_SECONDS` | Upper bound on one whole question — turns, tool calls, review, retry. Default 120. |
| `TAVILY_API_KEY` | Enables web search. Unset, the agent simply never sees the tool. |
| `RAG_AGENT_SEARCH_TOP_K` | Chunks per retrieval call. Default 6. |
| `RAG_TAVILY_MAX_RESULTS` | Web results per search. Default 5. |
| `RAG_DEFAULT_SYSTEM_PROMPT` | What a project with no prompt assigned answers with. |
| `RAG_PROMPT_TTL_SECONDS` | How long a resolved prompt is served before Firestore is re-read. Default 1 hour. |
| `FIRESTORE_PROMPTS_COLLECTION` / `FIRESTORE_PROJECT_PROMPTS_COLLECTION` | Default `systemPrompts` and `projectPrompts`. |
| `RAG_PROMPT_CACHE_PREFIX` | Redis key prefix for cached prompts. Default `ragPrompt:`. |

There is no credential configuration: the API does not authenticate anyone.
`GOOGLE_APPLICATION_CREDENTIALS`, `PINECONE_API_KEY`, `GEMINI_API_KEY`, and the
model keys above are this service's own credentials for the infrastructure it
calls, not its callers'.

## Docker

```powershell
docker compose up --build
```

Three services: `api`, `worker`, and `redis`. Which one ingests depends on
`REDIS_URL` — unset, the API does it on its own event loop and `worker` exits
saying it has no queue to read; set, the API only claims and enqueues and
`worker` is the process that does the work. Setting it **also requires**
`GCP_PROJECT_ID`, or the API refuses to start.

Inside the compose network Redis is `redis`, never localhost:
`REDIS_URL=redis://redis:6379/0`.

`keys/` is in `.dockerignore` — a service account key must never be baked into
an image — so it is mounted read-only at run time instead. Point
`GOOGLE_APPLICATION_CREDENTIALS` at `/app/keys/<file>.json`.

## What isn't built yet

- **Chat history.** Every `/query` is a single turn. The retry continues that
  turn's conversation, but nothing persists between requests, so a caller cannot
  ask a follow-up. This is where a `chatHistory` collection would land.
- **Prompt administration.** `PromptStore.savePrompt` and `assignPrompt` are
  written and tested but no endpoint calls them; the two Firestore collections
  are edited by hand today.
- **Authentication, and any access control at all.** `serverId` is unverified,
  so every caller is anonymous in practice and every project is readable and
  writable by anyone who can reach the service. The project mapping is not scoped
  by `serverId` either, so two callers picking the same project name share one
  database. This has to come back before the API is exposed to anything.
- **Job eviction on the in-memory path.** Redis records expire
  (`RAG_JOB_TTL_SECONDS`), so the deployed path evicts itself. The dict the
  no-Redis path uses grows until the process restarts.
- **Automatic retry of transient failures.** A Pinecone blip and a dead link are
  both recorded as `FAILED` and both need resubmitting, because nothing yet tells
  them apart.

## Test

```powershell
uv run pytest
```
