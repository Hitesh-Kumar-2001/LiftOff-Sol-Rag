# RAG API

A FastAPI service that answers questions from a chosen RAG database, callable
only by servers it can verify. Managed with `uv`.

**Status:** the API surface and server authentication are built and tested.
Retrieval is not — the handler validates everything, then returns a placeholder
answer. See [What isn't built yet](#what-isnt-built-yet).

## Run locally

```powershell
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive API documentation.

## Asking a question

`POST /api/v1/query` takes the calling server's credentials and the question in
one JSON body:

```json
{
  "serverId": "billing-service",
  "serverSecret": "top-secret",
  "question": "What is the refund window?",
  "ragDbId": "handbook"
}
```

```json
{
  "answer": "...",
  "ragDbId": "handbook"
}
```

| Status | Meaning |
| ------ | ------- |
| 200 | Verified caller; answer returned |
| 401 | `serverId` unknown, or `serverSecret` wrong (the two are not distinguished) |
| 422 | A field is missing, blank, or unexpected |

`serverId` + `serverSecret` say *who is asking*; `question` + `ragDbId` say
*what to answer and from where*.

## Submitting a document

`POST /api/v1/document` queues a document for ingestion and returns right away
— it does not wait for processing:

```json
{
  "serverId": "billing-service",
  "serverSecret": "top-secret",
  "documentLink": "https://example.com/handbook.pdf",
  "ragDbId": "handbook"
}
```

```json
{
  "jobId": "handbook",
  "status": "queued"
}
```

| Status | Meaning |
| ------ | ------- |
| 202 | Verified caller; job created and running in the background |
| 401 | `serverId` unknown, or `serverSecret` wrong |
| 422 | A field is missing, blank, or unexpected |

`jobId` is the `ragDbId` itself, not a generated id — a job exists to populate
one RAG database, so there's nothing else meaningful to key it by. A second
submission for a `ragDbId` that already has a job reuses that same job id and
replaces the previous record; the earlier task is *not* cancelled, it just runs
to completion writing into a `Job` nothing can look up anymore. Deduplicating,
queueing, or cancelling on resubmission is a policy decision for later.

What "processing" a document means beyond metadata extraction — chunking,
embedding, writing into the `ragDbId` — is not fully decided yet. See
[app/documents.py](app/documents.py): `DocumentProcessor` is the contract real
ingestion plugs into; `StubDocumentProcessor` just marks the job done without
doing anything. There is no endpoint yet to poll a job's status — a natural
next addition.

Jobs live in memory only and do not survive a restart. On shutdown, the app
*waits* for any job still running rather than cancelling it — a job mid-download
or mid-analysis has no safe halfway point to stop at. This can't hang forever:
a download is capped at `DOWNLOAD_TIMEOUT_SECONDS`, so every job finishes (or
fails) on its own well within that. There is no job eviction yet — old jobs
stay in memory until the process restarts; when that's needed it'll be its own
endpoint, not part of submission.

A `.zip` or `.rar` is searched recursively — an archive containing another
archive is unpacked in turn, however many levels deep (capped at
`MAX_ARCHIVE_DEPTH`, currently 5, to bound a maliciously nested archive). A
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
| [app/main.py](app/main.py) | Builds the app; loads credentials at startup, keeps them fresh, and cancels in-flight jobs on shutdown |
| [app/routes.py](app/routes.py) | The endpoints: verify the caller, then answer or queue |
| [app/jobManager.py](app/jobManager.py) | What the API talks to for jobs — create, look up by id, shut down cleanly |
| [app/jobs.py](app/jobs.py) | A job's data (`Job`, `JobStatus`) and `runJob`, which knows how to execute one |
| [app/documents.py](app/documents.py) | The `DocumentProcessor` contract, plus the real (and stub) implementations |
| [app/schemas.py](app/schemas.py) | The wire contract — camelCase JSON in and out, snake_case in Python |
| [app/security.py](app/security.py) | Holds credentials in RAM and checks them |
| [app/credentials.py](app/credentials.py) | Where credentials come from, and whether that source can change |

The split that matters is the last two: `security.py` decides *whether a caller
is who it says*, `credentials.py` decides *where the answer to that comes from*.
Changing storage touches one file.

### A request, start to finish

1. **Validation.** FastAPI checks the body against `QueryRequest` before any of
   our code runs. A missing, blank, over-long, or unexpected field is a 422 —
   the handler never sees a malformed request.
2. **Verification.** The route hands `serverId` and `serverSecret` to the
   registry, which is a dict lookup plus one constant-time digest compare
   against the in-memory copy. No I/O, so a flood of bad credentials costs a
   hash each and never reaches the store. A failure raises a 401 whose message
   deliberately does not reveal which half was wrong.
3. **Retrieval.** Not built. `ragDbId` and `question` are where it will plug in.
4. **Response.** `QueryResponse` is serialized back to camelCase.

The secret is typed `SecretStr`, so it does not appear in logs, tracebacks, or
error responses even if something goes wrong mid-request.

### Startup and shutdown

```
startup ──► read every credential ──► hold in memory ──┐
                                                       │
(shared stores only)                                   │
every 30s ─► read every credential ──► replace ────────┤
                                                       ▼
request ─────────────────────────────────────────► check memory
                                                       │
                                              hit ─────► proceed
                                             miss ─────► 401
```

The lifespan hook fills memory before the first request lands. If the store is
unreachable or malformed, **startup fails** — a process that cannot verify
callers should not accept traffic, and a config typo should surface on deploy
rather than on the first request.

### Keeping the copy honest

Whether the in-memory copy is re-read on a timer is the *store's* decision,
declared as `refreshInterval`:

| Store | `refreshInterval` | Why |
| ----- | ------------------ | --- |
| File, env var | `None` | Nothing else can change them while the node runs, so polling would re-read the same bytes forever. Edit and restart. |
| Firestore, any shared DB | 30s | Another process can add, rotate, or revoke a credential at any moment. |

For a shared store the timer is the only thing that keeps memory honest, and
that interval is the window in which a revoked server is still served. Checking
the store only when auth *fails* would never catch a revocation at all: a
revoked server still sends a secret that matches the stale copy in memory, so
the check passes and the store is never consulted.

If a refresh fails, the last known-good copy keeps serving traffic and the next
tick retries — a store outage must not lock out every caller.

### Adding a store

`CredentialSource` in [app/credentials.py](app/credentials.py) is one method and
one attribute. A Firestore adapter is the whole of it:

```python
class FirestoreCredentialSource:
    refreshInterval = DEFAULT_REFRESH_INTERVAL_SECONDS  # shared: must be polled

    async def loadAll(self):
        return [
            _toCredential(doc.id, doc.to_dict())
            async for doc in self._collection.stream()
        ]
```

Return it from `buildCredentialSource()` and it starts being polled because it
declares an interval. Nothing in `security.py`, `routes.py`, or `main.py`
changes.

## Configuration

| Variable | Effect |
| -------- | ------ |
| `RAG_CREDENTIALS_FILE` | Path to a JSON credentials file. Takes precedence. |
| `RAG_SERVER_CREDENTIALS` | The same JSON, inline. Used when no file is named. |

Both hold the servers allowed to call the API:

```json
{
  "billing-service": {"secret": "top-secret"},
  "admin-service":   {"secretSha256": "9f86d0818884..."}
}
```

- Use `secretSha256` in production; `secret` is accepted so a freshly generated
  key can be pasted in as-is. Generate a pair with:
  `python -c "import secrets,hashlib;s=secrets.token_urlsafe(32);print(s, hashlib.sha256(s.encode()).hexdigest())"`
- Secrets must be high-entropy random strings, not chosen passwords — SHA-256 is
  not a password KDF.
- With neither variable set, the store is empty and every request gets a 401.
- Malformed credentials fail startup rather than surfacing on the first request.

## What isn't built yet

- **Retrieval.** No vector store, embeddings, or chunking. `ragDbId` is accepted
  and echoed back but nothing resolves it to an actual database yet.
- **Per-server access control.** Any verified server may name any `ragDbId`.
  Worth adding before two tenants share the deployment.
- **A Firestore adapter** for credentials. The seam is ready; the adapter is not
  written.
- **Ingestion past metadata.** `POST /document` downloads and analyzes the
  document (see [app/documents.py](app/documents.py)) but nothing chunks,
  embeds, or writes it into the `ragDbId` yet.
- **Job status.** No way to poll a `jobId` after submission, and no persistence
  — jobs vanish on restart.
- **Job eviction.** Old jobs never leave memory. Planned as its own endpoint,
  separate from submission.

## Test

```powershell
uv run pytest
```
