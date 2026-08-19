---
name: rag-api
description: Context for this repo — an unauthenticated FastAPI RAG service (async document ingestion, Gemini chunking, Pinecone vector store, Firestore job table and project mapping, Celery workers). Load before reading, changing, debugging, or extending any code under app/, api/, scripts/, or tests/, or when asked how ingestion, jobs, projects, search, or deployment work here.
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
celery -A app.celeryApp worker --loglevel=info              # worker (Linux/macOS)
celery -A app.celeryApp worker --loglevel=info --pool=solo  # worker (Windows; --pool=solo is required)
```

`scripts/live*Check.py` run against real infrastructure and are deliberately outside
the test suite (they cost money / need network): `liveFirestoreCheck` (job table),
`liveCeleryCheck` (self-contained broker + worker, needs `pywin32` on Windows),
`livePineconeCheck` (store), `liveIngestionCheck <url>` (whole pipeline, real Gemini
and Pinecone).

## Endpoints

All are POST, taking their input in the body. **Every endpoint takes a `projectId` and
an unverified `serverId`; none takes or returns a `ragDbId`.** There is no 401 anywhere
in the API.

| Route | Does |
| --- | --- |
| `POST /api/v1/query` | **Placeholder.** Returns a canned string. Retrieval is not wired in. |
| `POST /api/v1/document` | Resolves-or-creates the project's database, claims it, queues ingestion, returns 202. 409 on conflict, 503 on dispatch failure. |
| `POST /api/v1/document/status` | Job status. Returns `projectId` normally, or `documentLink` when the strategy was RAW. 404 if the project or its job is unknown. |
| `POST /api/v1/search` | Retrieval only — the matching chunks, not a generated answer. An unresolved project gives empty hits, not a 404. |
| `GET /`, `GET /health` | Liveness. |

## Module map

**API layer** — [app/main.py](app/main.py) builds the app (lifespan only awaits
in-flight jobs on shutdown) and owns the 422 handler that allowlists `type`/`loc`/`msg`
so a malformed request is not echoed back. [app/routes.py](app/routes.py) has the four
routes and is where `projectId` becomes `ragDbId`. [app/schemas.py](app/schemas.py) is
the wire contract. [app/projectStore.py](app/projectStore.py) holds the mapping —
`ProjectStore` protocol, in-memory and Firestore implementations, `buildProjectStore()`
factory.

**Auth** — none. There is no auth module; see the note at the top.

**Jobs** — [app/jobs.py](app/jobs.py) is the `Job` record, `JobStatus`, `runJob`, the
`JobStore` protocol, and `resolveSubmission` (the shared NEW/REUSE/CONFLICT rules).
[app/jobManager.py](app/jobManager.py) picks the manager from the environment and holds
the in-memory one. [app/firestoreJobManager.py](app/firestoreJobManager.py) runs jobs
in-process with the table in Firestore; [app/celeryJobManager.py](app/celeryJobManager.py)
subclasses it and overrides exactly one method (`_start`).
[app/firestoreJobStore.py](app/firestoreJobStore.py) is storage only — the claim is a
transaction. [app/celeryApp.py](app/celeryApp.py) and [app/celeryTasks.py](app/celeryTasks.py)
are the broker config and the worker end.

**Ingestion** — [app/ragProcessor.py](app/ragProcessor.py) is the composition root:
download → analyze → select strategy → extract text → chunk → store.
[app/documents.py](app/documents.py) handles download, format detection, archive walking,
and metadata (pdf/docx/csv/txt/md; zip/rar). [app/ragSelector.py](app/ragSelector.py)
picks a strategy from token count. [app/ragIngestionPipeline.py](app/ragIngestionPipeline.py)
is chunking plus the `ChunkStore` protocol plus `lexicalSearch`.
[app/chunkStoreFactory.py](app/chunkStoreFactory.py) picks the store;
[app/pineconeChunkStore.py](app/pineconeChunkStore.py) and
[app/localChunkStore.py](app/localChunkStore.py) implement it.

**Deploy** — [api/index.py](api/index.py) exposes the ASGI app for Vercel;
`vercel.json` rewrites everything to it. The untracked `Dockerfile` runs uvicorn
under `uv`. `app/__init__.py` calls `load_dotenv()` before any submodule reads config.

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

The environment picks both the table and the executor. **There is no silent fallback** —
a broker without a project raises at startup.

| `CELERY_BROKER_URL` | `GCP_PROJECT_ID` | Table | Work runs | Survives restart |
| --- | --- | --- | --- | --- |
| — | — | dict in-process | API process | no |
| — | set | Firestore | API process | table does; running job does not |
| set | set | Firestore | Celery workers | yes |

The `projectId` → `ragDbId` mapping follows the same switch: Firestore when
`GCP_PROJECT_ID` is set, in-memory otherwise (with a startup warning). The two must not
disagree — a durable job table over a mapping that dies on restart would resolve a
project to a *new* database after every restart.

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
7. **The Celery message carries only the `ragDbId`.** The worker re-reads the job from
   the table. A copy in the message would be a second source of truth.
8. **`resolveSubmission` is the single copy of the reuse/conflict rules.** The in-memory
   manager calls it directly; `FirestoreJobStore.claim` calls it for both the Firestore
   and Celery managers. Do not reimplement it per backend — drift would only surface in
   production, as the same request accepted on one deployment and refused on another.
9. **`CELERY_VISIBILITY_TIMEOUT` must exceed `CELERY_TIME_LIMIT`.** Otherwise Redis
   redelivers a live job to a second worker and two ingestions interleave into one
   namespace — the exact thing the conflict check exists to prevent, arriving by a route
   that bypasses it.
10. **`RAG_STALE_JOB_SECONDS` must exceed the longest legitimate runtime.** Same failure
   mode, arriving *through* the check. It is only safe under Celery, where
   `task_time_limit` guarantees an upper bound; leave it unset on Windows or the solo
   pool, where time limits are not enforced.
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
16. **`serverId` is never trusted.** It is logged, and it reaches `resolveSubmission`,
    where same-caller-same-document means REUSE and a different caller means CONFLICT —
    the one place an unverified value still steers behaviour. Do not add anything that
    grants access based on it without adding authentication first.

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
  set before importing `app.jobManager`** — the live scripts set them at the top of the
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
| `FIRESTORE_JOBS_COLLECTION` | Default `ragJobs`. |
| `FIRESTORE_PROJECTS_COLLECTION` | The `projectId` → `ragDbId` mapping. Default `ragProjects`. |
| `FIRESTORE_DATABASE_ID` | A database *named* `default` is not the special `(default)` one — set this if the client claims the database does not exist. |
| `CELERY_BROKER_URL` | Dispatch to workers. Requires `GCP_PROJECT_ID`. |
| `CELERY_TIME_LIMIT` / `CELERY_SOFT_TIME_LIMIT` | 1800 / 1500s. |
| `CELERY_VISIBILITY_TIMEOUT` | Default twice the hard limit. See invariant 5. |
| `CELERY_BROKER_TRANSPORT_OPTIONS` | JSON, merged over the defaults. |
| `RAG_STALE_JOB_SECONDS` | Reclaim age for a stuck job. See invariant 6. |
| `PINECONE_API_KEY`, `RAG_PINECONE_INDEX`/`_CLOUD`/`_REGION`/`_EMBED_MODEL` | Index `rag-chunks`, aws/us-east-1, `llama-text-embed-v2` — server-side embedding, so nothing here embeds. |
| `GEMINI_API_KEY`, `RAG_GEMINI_MODEL` | AI chunking. |
| `RAG_TEST_MODE`, `RAG_LOCAL_STORE_DIR` | Local JSON chunk store instead of Pinecone. |
| `RAG_MAX_EMBED_TOKENS`, `RAG_AI_CHUNK_CONCURRENCY` | 2048, 8. |
| `RAG_UNRAR_TOOL` | `.rar` needs an external unrar/bsdtar/7z binary on the machine. |
| `TIKTOKEN_CACHE_DIR` | tiktoken downloads its BPE data on first use; point this somewhere that survives restarts. |

No credential configuration exists — the API authenticates nobody. The keys above are
this service's own credentials for the infrastructure it calls, not its callers'.

## Known gaps and drift

- **`/query` does no retrieval.** `/search` works; joining the two into a generated
  answer is the obvious next piece. The route carries a comment showing exactly where
  `projects.resolve(...)` goes when it lands — deliberately not wired yet, so the stub
  does not pay for a Firestore read it discards.
- **No authentication or access control of any kind.** Removed deliberately. Every
  project is readable and writable by anyone who can reach the service, and the mapping
  is not scoped by `serverId`, so two callers picking the same project name share one
  database. Must come back before this is exposed to anything.
- **No job eviction.** Finished jobs accumulate in the table forever.
- **No retry classification.** A Pinecone blip and a dead link are both just FAILED.
- **`RagIngestionPipeline.load` only reads plain text**, so anything binary must come in
  through `runText` with text already extracted (which is what `ragProcessor` does).
- **`MAX_ARCHIVE_DEPTH = 1`** in `app/documents.py`, but the constant's own comment
  still says five. The README was updated for both the `projectId` change and the auth
  removal; sections not touched by either have not been audited, so trust the code where
  the two disagree.
- **`Dockerfile.save`** is a stray 1-byte file at the repo root; `Dockerfile` and
  `.dockerignore` are untracked.
- **At-least-once delivery is accepted, not solved.** A DONE job is skipped at pickup; a
  redelivery racing a *live* run is not prevented (the runs converge except under AI
  chunking, where two runs can segment differently).
