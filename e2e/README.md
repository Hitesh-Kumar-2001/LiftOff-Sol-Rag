# The live end-to-end suite

One test file. It ingests a real document through the real worker into a real
vector database, then has a twenty-turn sales conversation with the agent over
HTTP and checks what the whole thing cost.

It is **not** part of `uv run pytest`. `testpaths = ["tests"]` in
`pyproject.toml` keeps it out, because every run of this costs money and needs
three processes that a unit suite has no business requiring.

```bash
uv run pytest e2e -s          # the live suite. -s so you can watch it.
uv run pytest                 # the unit suite. Never touches this.
```

## What has to be running

Three things, and you start them — the fixtures deliberately do not. A fixture
that launched the server would be convenient right up until something broke, at
which point it would hide which half broke.

```bash
docker compose up redis              # or any Redis on REDIS_URL
uv run uvicorn app.main:app          # the API
python -m app.jobs.worker            # the ingestion worker
```

Then, in a fourth terminal:

```bash
uv run pytest e2e -s
```

If nothing is listening the whole suite **skips** with an instruction rather
than failing. A red suite for "you did not start uvicorn" teaches people to
ignore red suites.

You also need everything the service itself needs — `GCP_PROJECT_ID`,
`REDIS_URL`, `PINECONE_API_KEY`, and an API key for whichever provider
`config/models.toml` names. They come from `.env` like everywhere else.

## What it actually does

1. Serves `e2e/documents/wanderlynTravel.txt` on an ephemeral local port.
2. `POST /api/v1/document` with that URL, then polls `/document/status` until
   the job is `done`. This is the slow part: the corpus is ~16,500 tokens, which
   puts it in the **AI chunking** band, so the worker makes roughly 220 chunker
   calls at eight concurrent.
3. `POST /api/v1/conversations/{projectId}` to open a conversation.
4. Twenty turns through `POST /api/v1/conversations/{projectId}/web`, in order,
   as one customer who starts undecided and ends up booking.
5. Reads `ragUsage` back out of Firestore and checks the arithmetic.

Web gateway only. WhatsApp and LINE answer through a push to a platform API
rather than on the response, so testing them live needs credentials and a
public callback URL — a different file, when there is something to point at.

## The corpus

`e2e/documents/wanderlynTravel.txt` is a fictional travel company: five trips
with day-by-day itineraries, seasonal pricing, single supplements, what is and
is not included, ten add-on modules, and the full policy set — booking, the
cancellation sliding scale, refunds, insurance, accidents and medical
evacuation, liability, force majeure, complaints, visas, health, ages,
accessibility, dietary, tipping, luggage, weather by month, and how to book.

Three things about it are deliberate and worth knowing before you edit it.

**It has to stay above 2,000 tokens.** Below that, `RagSelector` picks the RAW
strategy, the document is stored whole, *no vector database is populated*, and
every one of the twenty answers would be improvised from nothing. The `ingested`
fixture asserts this rather than letting it become a mysterious twenty-way
failure.

**Every paragraph names its trip**, and this one is not theoretical — the corpus
broke it and three runs blamed the agent for it. Ten seasonal-price paragraphs
originally read *"Shoulder season is late March, June, September and November,
and it costs $4,180 per person"* with nothing saying which trip. Chunking splits
on blank lines, so the index held a price with no owner, and the agent attached
it to whatever trip was under discussion — quoting one journey's real price
against another. Naming the trip in each paragraph fixed it completely.

A chunk has to stand alone, because retrieval will make it stand alone. If you
add to this corpus, the check is: read any single paragraph with the rest of the
file covered up, and see whether it still says what it is about.

**It is pure ASCII**, so nothing in the run can turn into an encoding question.

Some facts in it are load-bearing for assertions — the October Morocco price
($2,860), the single supplement ($520), the cooking module ($145), the deposit
($600), and the three published reductions (5% group, 4% returning, $200 early,
9% combined cap). They are listed at the top of `testSalesConversation.py` so an
edit to the corpus and a stale test are one diff apart.

## The twenty turns

| # | Turn | What it is really testing |
| --- | --- | --- |
| 1–3 | opener, qualify, catalogue | does it qualify before pitching |
| 4–5 | chooseTrip, itinerary | retrieval on the trip that was named |
| 6–8 | price, included, excluded | grounding, and whether it repeats the bad news |
| 9–10 | landmark, freeTime | is it using our copy or its own knowledge of Morocco |
| 11–13 | difficulty, dietary, module | honest caveats, and the upsell |
| 14 | singleRoom | **memory** — never names the trip; $520 is only reachable by remembering turn 4 |
| 15–17 | cancellation, accident, insurance | the answers that are a contract |
| 18 | discount | **must not invent one** |
| 19 | uncovered | drones are nowhere in the corpus; it has to say so |
| 20 | close | twenty turns is wasted if there is no next step |

Twenty-six checks read the finished transcript, each as its own test, so a
failure names exactly one property that did not hold and the rest still report.
One of them, `testNoPriceIsAttachedToTheWrongTrip`, checks line by line that a
real price is never quoted against the wrong journey — a live run offered
Kerala's $2,540 as the Amalfi Coast's shoulder price, which no per-turn price
check can see, because the figure is genuine and only the pairing is wrong.
Tone, length, ordering and phrasing are deliberately not asserted — pinning
those produces a test that fails on every model upgrade and catches nothing.

## What it costs, and how long it takes

Measured, on a real run against `openai` / `gpt-5.6-luna` through the Responses
API, reusing an already-ingested project:

```
twenty turns in 315s (15.8s per turn)

total       264,401   (13,220 per turn)
agent       226,565   openai/gpt-5.6-luna  calls=20
reviewer     14,571   openai/gpt-5.6-luna  calls=20
summariser   23,265   openai/gpt-5.6-luna  calls=6
cached in   138,913
reasoning     4,556
```

Three things in that are worth reading twice. The **summariser ran six times** —
a twenty-turn conversation outgrows `RAG_CONTEXT_SUMMARY_TOKENS` repeatedly, and
it is 7.5% of the bill that nothing outside this ledger would have shown. **The
reviewer is cheap**, 4% for a call on every single question. And **more than half
the input was cached**, which is the prompt cache doing its job on a conversation
that resends its history every turn.

Across six runs the total ranged 264k-496k tokens: the same twenty questions,
varying by how often the agent retried and how much history was resent. Turn
latency settled at 8-30s, but the very first turn of a cold process took **96s**,
and `RAG_ANSWER_TIMEOUT_SECONDS` is 120 -- less headroom on that first question
than is comfortable.

On top of that, a run that ingests costs ~220 chunker calls. The per-role numbers
are printed at the end of every run, from the ledger the service wrote itself.

Ingest once and reuse it while you iterate on the conversation:

```bash
uv run pytest e2e -s                                   # note the project id it prints
RAG_E2E_PROJECT_ID=e2e-abc123def0-travel uv run pytest e2e -s
```

That also stops each run minting a new Pinecone namespace, which is the one
thing the cleanup below does *not* remove.

## Environment

| Variable | Default | Why you would set it |
| --- | --- | --- |
| `RAG_E2E_BASE_URL` | `http://127.0.0.1:8000` | The service is somewhere else. |
| `RAG_E2E_DOCUMENT_URL` | a local file server | **The worker downloads the corpus, not the test.** If the worker is in Docker, `127.0.0.1` is the container — point this at `http://host.docker.internal:<port>/wanderlynTravel.txt` or anywhere else it can GET. |
| `RAG_E2E_PROJECT_ID` | a fresh id per run | Reuse an ingested project and skip step 2. Checked to be `done` first. |
| `RAG_E2E_SERVER_ID` | `e2e-sales-check` | The label in the API log. Unverified, like every `serverId`. |
| `RAG_E2E_INGEST_TIMEOUT` | `600` | AI chunking is slow, and slower when the provider throttles. |
| `RAG_E2E_TURN_TIMEOUT` | `180` | Must stay above the server's `RAG_ANSWER_TIMEOUT_SECONDS` (120), or the client gives up on an answer that was about to arrive. |
| `RAG_E2E_CLEANUP` | unset | Delete this run's Firestore documents afterwards. Off by default on purpose — see below. |

## Cleanup

Off by default. The run's real value is that it leaves behind a genuine
conversation and a genuine usage ledger you can go and read in the Firestore
console, and deleting them at the end of a green run throws that away.

`RAG_E2E_CLEANUP=1` removes the project mapping, the conversation with both its
subcollections, and the whole `ragUsage` tree for the run. It does **not** touch
the Pinecone namespace — use `RAG_E2E_PROJECT_ID` if you care about that, rather
than minting one per run and deleting it by hand.

## When it fails

- **Everything skipped** — nothing is listening on `RAG_E2E_BASE_URL`.
- **Ingestion `failed`** — almost always one of two things. The worker could not
  download the corpus (see `RAG_E2E_DOCUMENT_URL`), or the chunker's provider
  rejected the calls. One failed chunker call fails the whole job by design: a
  bad key fails every section identically, and silently falling back to non-AI
  chunking would store a corpus that looks fine and answers badly.
- **Ingestion stuck at `queued`** — the worker is not running.
- **Every turn 503** — a role in `config/models.toml` has no API key behind it.
- **Turns pass, grounding assertions fail** — retrieval is the suspect, not the
  agent. `POST /api/v1/search` with the same `projectId` and the failing
  question shows you what the agent was actually given.
- **`testTheConversationOpenedUnderTheSalesPersona`** — `default` in
  `config/prompts.toml`, `RAG_PERSONA`, or a prompt assigned to this project in
  Firestore.
- **`testItStillKnowsWhichTripTenTurnsLater`** — conversation history is not
  reaching the model. Nothing else in this suite would notice; every individual
  answer still looks fine.
