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

## How it works

### The pieces

| File | Responsibility |
| ---- | -------------- |
| [app/main.py](app/main.py) | Builds the app; loads credentials at startup and keeps them fresh for the process lifetime |
| [app/routes.py](app/routes.py) | The endpoint: verify the caller, then answer |
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
declared as `refresh_interval`:

| Store | `refresh_interval` | Why |
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
    refresh_interval = DEFAULT_REFRESH_INTERVAL_SECONDS  # shared: must be polled

    async def load_all(self):
        return [
            _to_credential(doc.id, doc.to_dict())
            async for doc in self._collection.stream()
        ]
```

Return it from `build_credential_source()` and it starts being polled because it
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
- **A Firestore adapter.** The seam is ready; the adapter is not written.

## Test

```powershell
uv run pytest
```
