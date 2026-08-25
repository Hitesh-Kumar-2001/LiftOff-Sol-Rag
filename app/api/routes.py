"""Public API routes.

Every endpoint takes a ``projectId``. The ``ragDbId`` behind it is resolved
here, at the edge, and everything inward -- the job manager, the conflict
check, the chunk store -- goes on keying on that id exactly as before. The
indirection stops in this file, which is what keeps it cheap: see
``app.stores.projectStore`` for why it exists at all.

**There is no authentication.** Every endpoint takes a ``serverId``, but
nothing verifies it: there is no secret, no signature, and no registry to check
it against. It is a label the caller picked, recorded in the log so a request
can be traced back to whoever says they made it, and treated as nothing more.
Anyone who can reach this API can therefore ingest into any project and read
any project's chunks, and can name themselves anything while doing it. That is
a deliberate choice for a service that is not yet exposed; putting it behind
anything public means adding authentication back first -- in front of these
routes, or in front of the whole app.

One place still *uses* ``serverId`` for a decision rather than a log line:
``app.jobs.job.resolveSubmission`` treats the same document from the same
``serverId`` as a retry and a different caller's document as a conflict. With
nothing verifying the id, a caller can put itself on either side of that by
choosing what to send. The consequence is a 409 avoided or provoked, not access
gained -- but it is the one spot where an unverified value still steers
behaviour.
"""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.agent.agent import ANSWER_TIMEOUT_SECONDS, answerQuestion
from app.agent.llmManager import LlmConfigError
from app.agent.promptStore import PromptStore, getPromptStore
from app.agent.summariser import (
    applySummary,
    contextFromSearches,
    needsSummary,
    summariseConversation,
    trimToBudget,
)
from app.api.schemas import (
    DocumentIngestRequest,
    ErrorResponse,
    JobResponse,
    JobStatusRequest,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.ingestion.ragIngestionPipeline import ChunkingStrategy, ChunkStore
from app.jobs.jobManager import JobConflictError, JobDispatchError, JobManager, getJobManager
from app.stores.chatStore import ChatStore, ChatStoreError, ChatWindow, getChatStore
from app.stores.chunkStoreFactory import getChunkStore
from app.stores.projectStore import ProjectStore, getProjectStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


async def _openChat(
    chats: ChatStore, prompts: PromptStore, payload: QueryRequest, ragDbId: str | None
) -> tuple[ChatWindow | None, str | None]:
    """The conversation this question belongs to, and the id to answer with.

    Three outcomes, and they are deliberately different:

    * no ``chatId`` -- a chat is created, snapshotting the project's current
      system prompt onto it, and its window comes back empty.
    * a ``chatId`` that exists -- its window is loaded: Redis first, Firestore
      behind it, and only the part the summary does not already cover.
    * a ``chatId`` that does not exist -- 404. Not a new chat: a mistyped id
      quietly starting a fresh conversation is indistinguishable, from the
      caller's side, from a model that has forgotten everything they said.

    A store that cannot be *reached* is none of those. It degrades to a
    single-turn answer, because an unreachable chat store is a reason for a
    worse answer and not a reason for no answer at all -- the model call is the
    expensive part and it has not been made yet.
    """
    try:
        if payload.chat_id is None:
            prompt = await prompts.systemPromptFor(payload.project_id)
            window = await chats.createChat(
                projectId=payload.project_id,
                systemPrompt=prompt,
                ragDbId=ragDbId,
                title=payload.question,
            )
            return window, window.chatId

        window = await chats.loadWindow(payload.project_id, payload.chat_id)
        if window is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chat '{payload.chat_id}' in project '{payload.project_id}'.",
            )
        return window, window.chatId
    except HTTPException:
        raise
    except ChatStoreError:
        logger.exception(
            "The chat store is unreachable; answering project '%s' without history.",
            payload.project_id,
        )
        # The caller's own id is echoed back rather than dropped: their chat
        # still exists, this one turn simply did not reach it.
        return None, payload.chat_id
    except Exception:
        logger.exception("Could not open a chat for project '%s'.", payload.project_id)
        return None, payload.chat_id


async def _foldIfNeeded(chats: ChatStore, window: ChatWindow) -> ChatWindow:
    """Summarise the conversation if it has outgrown its budget.

    Runs before the answer rather than after it, so the window the model sees
    is always inside budget -- summarising afterwards would leave the turn that
    tripped the limit to be answered with the oversized prompt that tripped it.
    The cost lands on the same turn as the benefit.

    Every failure here falls through to ``trimToBudget``, which shortens what is
    *sent* without touching what is stored. An over-budget prompt is the one
    outcome worth avoiding at any cost, because it is the one that turns a
    question that was merely expensive into a provider error.
    """
    if not needsSummary(window):
        return window

    folded = await summariseConversation(window)
    if folded is None:
        return trimToBudget(window)

    summary, throughTurn, throughContext = folded
    try:
        await chats.saveSummary(
            projectId=window.projectId,
            chatId=window.chatId,
            summary=summary,
            throughTurn=throughTurn,
            throughContext=throughContext,
        )
    except Exception:
        # The summary is still usable for this turn even if it was not stored;
        # the next question simply pays to make it again.
        logger.exception("Could not store the summary for chat '%s'.", window.chatId)

    logger.info(
        "Folded chat '%s' down to a summary through turn %d.", window.chatId, throughTurn
    )
    return applySummary(window, summary, throughTurn, throughContext)


async def _recordTurn(
    chats: ChatStore,
    window: ChatWindow | None,
    payload: QueryRequest,
    answer: str,
    searchLog: list[dict],
    reviewOutcome,
) -> None:
    """Store the exchange and everything it retrieved. Never raises.

    Deliberately after the answer is in hand and deliberately unable to fail the
    request. The model call is paid for and the caller is owed its result; a
    Firestore write that did not land costs the *next* turn some context, which
    is a far smaller loss than turning a completed answer into a 500. It is
    logged loudly because a chat that is silently not being written is a bug
    that would otherwise only surface as a model with no memory.
    """
    if window is None:
        return
    try:
        await chats.appendTurn(
            window=window,
            question=payload.question,
            answer=answer,
            context=contextFromSearches(searchLog),
            reviewScore=reviewOutcome.score,
            retried=reviewOutcome.retried,
        )
    except Exception:
        logger.exception(
            "Answered project '%s' but could not record the turn on chat '%s'.",
            payload.project_id,
            window.chatId,
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against a project's RAG database",
    tags=["query"],
    responses={
        502: {
            "model": ErrorResponse,
            "description": "The model provider failed; the question itself was fine",
        },
        503: {
            "model": ErrorResponse,
            "description": "The agent could not run — usually a missing or wrong model key",
        },
        504: {
            "model": ErrorResponse,
            "description": "The agent did not finish inside RAG_ANSWER_TIMEOUT_SECONDS",
        },
        404: {
            "model": ErrorResponse,
            "description": "A chatId was sent that this project has no chat for",
        },
    },
)
async def query(
    payload: QueryRequest,
    store: Annotated[ChunkStore, Depends(getChunkStore)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
    chats: Annotated[ChatStore, Depends(getChatStore)],
    prompts: Annotated[PromptStore, Depends(getPromptStore)],
) -> QueryResponse:
    """Answer a question from one project, through the agent.

    Retrieval is the agent's decision, not this route's: it gets a search tool
    bound to this project and calls it when the question needs it. A follow-up
    like "shorter, please" should not cost a vector search.

    The project is resolved read-only -- asking a question does not bring a
    database into existence. A project nothing was ingested into still answers:
    the agent simply gets no search tool (see ``app.agent.tools.buildTools``).

    The conversation around the question is this route's job and not the
    agent's::

        open the chat -> fold it if oversized -> answer -> record the turn

    Everything in that line except "answer" degrades rather than fails. The
    model call is the expensive, irreversible step, so nothing to do with
    storing a conversation may cost a caller an answer that has already been
    paid for -- see ``_openChat`` and ``_recordTurn``.
    """
    logger.debug("Query from '%s' on project '%s'.", payload.server_id, payload.project_id)

    ragDbId = await projects.resolve(payload.project_id)

    window, chatId = await _openChat(chats, prompts, payload, ragDbId)
    if window is not None:
        window = await _foldIfNeeded(chats, window)

    # Filled by the search tool as the agent runs, and read afterwards. A
    # mutable argument rather than a return value because the agent's own
    # return is the answer, and the retrievals have to come back from a run the
    # agent does not otherwise report on.
    #
    # Only a *completed* turn is stored, so retrievals made by a run that then
    # timed out or 502'd are dropped. They were paid for and could in principle
    # be kept, but there is no exchange to hang them on -- storing them would
    # mean a context entry belonging to a turn the chat has no record of.
    searchLog: list[dict] = []

    try:
        # Bounded. Nothing inside the model client caps a whole answer -- turns,
        # tool calls, the review and a retry -- so without this a provider that
        # stalls holds the connection until the caller gives up, and this
        # process holds a worker slot for exactly as long.
        async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
            result = await answerQuestion(
                projectId=payload.project_id,
                question=payload.question,
                ragDbId=ragDbId,
                chunkStore=store,
                chatWindow=window,
                searchLog=searchLog,
            )
    except LlmConfigError as exc:
        # The request was fine; this deployment cannot reach a model. A 503 says
        # "fix the configuration and try again", which a 500 does not.
        logger.exception("The agent is not configured to run.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from None
    except TimeoutError:
        logger.warning(
            "The agent did not answer project '%s' within %ss.",
            payload.project_id,
            ANSWER_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"The question was not answered within {ANSWER_TIMEOUT_SECONDS:.0f}s. "
                f"Nothing was changed by the attempt; try a narrower question."
            ),
        ) from None
    except Exception as exc:
        # A provider outage, a rate limit, a graph that ran out of steps. The
        # detail is deliberately the exception's type and not its text: a
        # provider message can quote the prompt back, and that prompt is another
        # project's configuration. The full error goes to the log.
        logger.exception("The agent failed to answer for project '%s'.", payload.project_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"The question could not be answered ({type(exc).__name__}). This is "
                f"a failure behind the API, not a problem with the request."
            ),
        ) from None

    if result.reviewOutcome.retried:
        logger.info(
            "Project '%s': first answer scored %.2f and was retried once.",
            payload.project_id,
            result.reviewOutcome.score,
        )

    await _recordTurn(chats, window, payload, result.answer, searchLog, result.reviewOutcome)

    return QueryResponse(
        answer=result.answer, project_id=payload.project_id, chat_id=chatId
    )


@router.post(
    "/document",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue a document for ingestion into a project",
    tags=["documents"],
    responses={
        409: {
            "model": ErrorResponse,
            "description": "A different document is already being ingested into this project",
        },
        503: {
            "model": ErrorResponse,
            "description": "The queue was unreachable; nothing was started, so resubmit as-is",
        },
    },
)
async def submitDocument(
    payload: DocumentIngestRequest,
    jobs: Annotated[JobManager, Depends(getJobManager)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> JobResponse:
    """Queue the document and return immediately.

    The only endpoint that may *create* a project's database: a first
    submission is what brings one into existence, and every later request for
    that project resolves to the same one. What processing means downstream is
    not fully decided yet (see ``app.ingestion.documents.DocumentProcessor``); the
    response only confirms the job was queued.
    """
    # Info, unlike the read routes: this one spends money and changes what is
    # stored, so it is the line worth having when working out who filled a
    # project with something unexpected.
    logger.info(
        "Ingestion requested by '%s' for project '%s': %s",
        payload.server_id,
        payload.project_id,
        payload.document_link,
    )

    # Resolved before the claim, because the claim is keyed on the ragDbId. A
    # mapping is permanent and idempotent, so minting one for a submission that
    # then turns out to conflict costs nothing -- the next submission for this
    # project resolves to the same id rather than a second one.
    ragDbId = await projects.resolveOrCreate(payload.project_id)

    try:
        job = await jobs.create(
            # Unverified, and recorded as given. It reaches resolveSubmission,
            # which is the one place it still decides something -- see the
            # module docstring.
            serverId=payload.server_id,
            documentLink=payload.document_link,
            ragDbId=ragDbId,
        )
    except JobConflictError:
        # Not a 202: nothing was queued, and running it anyway would leave the
        # database holding half of each document. The manager's message names
        # the ragDbId, which is internal, so it is reworded here in the terms
        # the caller actually sent.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Another document is already being ingested into project "
                f"'{payload.project_id}'. Wait for it to finish before submitting another."
            ),
        ) from None
    except JobDispatchError:
        # The request was fine; the queue behind it was not. A 503 says
        # "resubmit this exact request later", which a 500 does not.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Project '{payload.project_id}' could not be queued for processing. "
                f"Nothing was started; resubmit when the queue is reachable."
            ),
        ) from None

    return JobResponse(project_id=payload.project_id, status=job.status.value)


@router.post(
    "/document/status",
    response_model=JobStatusResponse,
    response_model_exclude_none=True,
    summary="Check how a project's ingestion is going",
    tags=["documents"],
    responses={
        404: {"model": ErrorResponse, "description": "No ingestion job for that project"},
    },
)
async def documentStatus(
    payload: JobStatusRequest,
    jobs: Annotated[JobManager, Depends(getJobManager)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> JobStatusResponse:
    """Report the job's status and where to go next -- nothing else.

    A POST rather than a GET because the rest of this API takes its input in
    the body and answering a status check the same way keeps one shape across
    every endpoint.

    An unknown project is a 404 rather than some 'unknown' status: that
    distinction is what a caller needs to decide whether to resubmit. How long
    a project stays known depends on where the mapping and the job table live
    -- in memory that is until the next restart, in Firestore it is
    indefinitely (see app.stores.projectStore and app.jobs.jobManager).
    """
    logger.debug(
        "Status check from '%s' on project '%s'.", payload.server_id, payload.project_id
    )

    # One 404 for both misses. A project nobody ever submitted and a project
    # whose job has since gone are the same answer to the caller.
    notFound = HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No ingestion job for projectId '{payload.project_id}'.",
    )

    ragDbId = await projects.resolve(payload.project_id)
    if ragDbId is None:
        raise notFound

    job = await jobs.get(ragDbId)
    if job is None:
        raise notFound

    # A document small enough to keep whole was never written to a vector
    # database, so answering "this project is queryable" would name something
    # there is nothing behind. That caller is sent back to the source instead.
    if job.strategy is ChunkingStrategy.RAW:
        return JobStatusResponse(status=job.status.value, document_link=job.documentLink)

    return JobStatusResponse(status=job.status.value, project_id=payload.project_id)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search a project's RAG database for the chunks nearest a query",
    tags=["query"],
)
async def search(
    payload: SearchRequest,
    store: Annotated[ChunkStore, Depends(getChunkStore)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
) -> SearchResponse:
    """Return the chunks matching the query.

    Retrieval only -- the matching passages, not an answer written from them.
    Generating an answer is ``/query``'s job, and keeping the two apart means
    a caller can see what was actually retrieved rather than inferring it
    from a summary.

    Which store answers is not decided here: under test mode it is the local
    one and the ranking is keyword overlap, otherwise it is Pinecone and the
    ranking is embedding similarity. Both satisfy the same protocol.

    A project that was never ingested into has nothing to match, so it comes
    back with no hits rather than a 404 -- an empty database and an empty
    result set are the same answer to "what matches this query". Searching
    does not create a database either: minting one here would leave an empty
    namespace behind for every mistyped projectId that ever arrived.
    """
    logger.debug("Search from '%s' on project '%s'.", payload.server_id, payload.project_id)

    ragDbId = await projects.resolve(payload.project_id)
    results = await store.search(ragDbId, payload.query, payload.top_k) if ragDbId else []

    return SearchResponse(
        project_id=payload.project_id,
        hits=[
            SearchHit(text=r.text, chunk_index=r.index, score=r.score) for r in results
        ],
    )
