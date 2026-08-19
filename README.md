# RAG API

A FastAPI service that ingests documents into per-project RAG databases and
searches them. Managed with `uv`.

> **There is no authentication.** Every endpoint takes a `serverId`, but nothing
> verifies it — there is no secret, no signature, and no registry behind it. It is
> a label for the log. Anyone who can reach this API can ingest into any project
> and read any project's chunks, under any name they choose. Do not expose it
> publicly without putting authentication in front of it first.

**Status:** ingestion, chunking, embedding, and retrieval are built and tested.
`POST /api/v1/query` still returns a placeholder rather than a generated answer.
See [What isn't built yet](#what-isnt-built-yet).

## Run locally

```powershell
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

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

`serverId` says *who is asking*, for the log and nothing else; `question` +
`projectId` say *what to answer and from where*.

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

The mapping lives in [app/projectStore.py](app/projectStore.py) — in Firestore
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
[app/documents.py](app/documents.py): `DocumentProcessor` is the contract real
ingestion plugs into; `StubDocumentProcessor` just marks the job done without
doing anything. `POST /api/v1/document/status` polls a job's progress.

Where jobs live depends on configuration (see [Where jobs run](#where-jobs-run));
with neither `GCP_PROJECT_ID` nor a broker set they are held in memory and do not
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
| [app/routes.py](app/routes.py) | The endpoints: resolve the project, then answer or queue |
| [app/projectStore.py](app/projectStore.py) | `projectId` → `ragDbId`, and nothing else. The indirection stops at the route |
| [app/jobManager.py](app/jobManager.py) | What the API talks to for jobs — create, look up by id, shut down cleanly. Picks which manager from the environment |
| [app/jobs.py](app/jobs.py) | A job's data (`Job`, `JobStatus`), `runJob` which executes one, and `resolveSubmission` — the reuse/conflict rules every manager shares |
| [app/firestoreJobStore.py](app/firestoreJobStore.py) | The job table in Firestore. Storage only; the claim is a transaction |
| [app/firestoreJobManager.py](app/firestoreJobManager.py) | Jobs in Firestore, run in this process |
| [app/celeryJobManager.py](app/celeryJobManager.py) | Jobs in Firestore, dispatched to workers. Differs from the above by one method |
| [app/celeryApp.py](app/celeryApp.py) / [app/celeryTasks.py](app/celeryTasks.py) | The broker configuration, and what a worker does with a queued job |
| [app/documents.py](app/documents.py) | The `DocumentProcessor` contract, plus the real (and stub) implementations |
| [app/schemas.py](app/schemas.py) | The wire contract — camelCase JSON in and out, snake_case in Python |

### A request, start to finish

1. **Validation.** FastAPI checks the body against `QueryRequest` before any of
   our code runs. A missing, blank, over-long, or unexpected field is a 422 —
   the handler never sees a malformed request.
2. **Logging.** `serverId` is written to the log line for the request. Nothing
   checks it; see the warning at the top of this file.
3. **Retrieval.** Not built. The `ragDbId` the `projectId` resolves to, and
   `question`, are where it will plug in.
4. **Response.** `QueryResponse` is serialized back to camelCase.

### Where jobs run

Ingestion is minutes of work — download, extract, chunk (sometimes through
Gemini), then a few hundred Pinecone upserts — kicked off by a request that has
to return in milliseconds. Where that work runs, and where the job table lives,
are two separate decisions, and the environment picks both:

| `CELERY_BROKER_URL` | `GCP_PROJECT_ID` | Table | Work runs | Survives a restart |
| --- | --- | --- | --- | --- |
| — | — | dict in the process | in the API process | no |
| — | set | Firestore | in the API process | the table does; a running job does not |
| set | set | Firestore | Celery workers | yes |

A broker without a project **fails at startup** rather than falling back. The
worker is a different process, so a dict the API owns is a table the worker
cannot see: the API would write jobs the worker never reads, the worker's status
writes would land nowhere, and `/document/status` would 404 every running job —
while the deployment looked healthy. The principle holds generally here: a
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
celery -A app.celeryApp worker --loglevel=info              # Linux/macOS
celery -A app.celeryApp worker --loglevel=info --pool=solo  # Windows
```

`--pool=solo` is not optional on Windows — the prefork pool does not work there.
Give the worker the same environment as the API: it needs the Pinecone, Gemini,
and Firestore credentials, since it is the process that actually does the work.

Two scripts check this against real infrastructure rather than stubs.
[scripts/liveFirestoreCheck.py](scripts/liveFirestoreCheck.py) needs only
credentials; [scripts/liveCeleryCheck.py](scripts/liveCeleryCheck.py) starts its
own broker, worker, and document server, so it needs nothing else installed
(except `pywin32` on Windows) and cleans up after itself.

> **A project can hold more than one database, and one *named* `default` is not
> the special `(default)` one.** If the client reports `The database (default)
> does not exist for project X` while the console plainly shows a database, that
> is the reason — set `FIRESTORE_DATABASE_ID` to the name shown there.

The settings worth knowing about are in [app/celeryApp.py](app/celeryApp.py).
The one with teeth is `CELERY_VISIBILITY_TIMEOUT`, which **must** stay above
`CELERY_TIME_LIMIT`: if Redis decides a message was dropped while its job is
still legitimately running, it hands the job to a second worker, and the two
ingest into one namespace — the exact interleaving the conflict check exists to
prevent, arriving by a route that bypasses it.

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
- **A job stuck past `RAG_STALE_JOB_SECONDS` can be reclaimed.** This is off
  unless a threshold is set, and it is only set where something guarantees an
  upper bound on runtime — under Celery, `task_time_limit` kills a task outright,
  so a job still `PROCESSING` well past it has definitively lost its worker. The
  threshold must stay above the longest a job could legitimately run: reclaiming
  early starts a second ingestion alongside a live one, which is the interleaving
  the conflict check exists to prevent, arriving through the check itself.

Two things are known and accepted rather than fixed:

- **Delivery is at-least-once.** A worker killed between finishing and acking
  leaves its message to be handed out again. A job already `DONE` at pickup is
  skipped, which covers that case. A redelivery racing a *live* run is not
  prevented — both workers ingest the same document, and because record ids come
  from chunk position they converge, except under AI chunking, where two runs can
  segment the document differently. Preventing it needs a renewable lease, and a
  lease would break the crash recovery that late acks exist to provide.
- **Time limits are not enforced on Windows or the solo pool.** The staleness
  reclaim assumes they are, so leave `RAG_STALE_JOB_SECONDS` unset when running
  workers there.

## Configuration

| Variable | Effect |
| -------- | ------ |
| `GCP_PROJECT_ID` | Set to keep the job table in Firestore instead of process memory. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service account key. Omit inside GCP, where the platform supplies credentials. |
| `FIRESTORE_JOBS_COLLECTION` | Collection holding the job table. Default `ragJobs`. |
| `FIRESTORE_PROJECTS_COLLECTION` | Collection holding the `projectId` → `ragDbId` mapping. Default `ragProjects`. |
| `FIRESTORE_DATABASE_ID` | Which database in the project. Leave unset for the special `(default)` one. |
| `CELERY_BROKER_URL` | Set to dispatch ingestion to workers, e.g. `redis://localhost:6379/0`. Requires `GCP_PROJECT_ID`. |
| `CELERY_TIME_LIMIT` | Hard kill for one ingestion, in seconds. Default 1800. |
| `CELERY_SOFT_TIME_LIMIT` | Raises inside the task first, so the job is marked failed with a reason. Default 1500. |
| `CELERY_VISIBILITY_TIMEOUT` | How long the broker waits before redelivering. Must exceed `CELERY_TIME_LIMIT`. Default is twice it. |
| `RAG_STALE_JOB_SECONDS` | Age past which a stuck job's `ragDbId` may be reclaimed. Defaults to twice `CELERY_TIME_LIMIT`; only applies under Celery. Must exceed the longest a job could legitimately run. |

There is no credential configuration: the API does not authenticate anyone.
`GOOGLE_APPLICATION_CREDENTIALS`, `PINECONE_API_KEY`, and `GEMINI_API_KEY` are
this service's own credentials for the infrastructure it calls, not its callers'.

## What isn't built yet

- **Generated answers.** `POST /api/v1/search` returns the matching chunks, but
  `POST /api/v1/query` still returns a placeholder — nothing yet writes an answer
  from what was retrieved. Ingestion, chunking, embedding, and retrieval
  themselves are built (see [app/ragIngestionPipeline.py](app/ragIngestionPipeline.py)
  and [app/pineconeChunkStore.py](app/pineconeChunkStore.py)).
- **Authentication, and any access control at all.** `serverId` is unverified,
  so every caller is anonymous in practice and every project is readable and
  writable by anyone who can reach the service. The project mapping is not scoped
  by `serverId` either, so two callers picking the same project name share one
  database. This has to come back before the API is exposed to anything.
- **Job eviction.** Finished jobs are never removed from the table. In memory
  they leak; in Firestore they accumulate. Planned as its own endpoint, separate
  from submission.
- **Automatic retry of transient failures.** A Pinecone blip and a dead link are
  both recorded as `FAILED` and both need resubmitting, because nothing yet tells
  them apart.

## Test

```powershell
uv run pytest
```
