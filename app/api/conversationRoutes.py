"""One conversation surface, reached through any gateway.

    POST /api/v1/conversations/{projectId}              start one
    POST /api/v1/conversations/{projectId}/{gateway}    say something
    GET  /api/v1/conversations/{projectId}/{gateway}    the gateway's handshake

``gateway`` is ``web``, ``whatsapp``, ``line``, or whatever is registered next.
One route, dispatching on the last segment, rather than a path per platform --
adding Facebook Messenger is an adapter and a line in
``app.channels.registry``, and no route at all.

**The project is in the path, for every gateway.** It has to be for a webhook:
the body is written by WhatsApp or LINE and nothing can be added to it, so the
URL is the only part of the request this service controls -- it is what gets
pasted into the platform's console. Having done that, the web gateway follows,
because one shape across all of them is worth more than the old convention of
putting every id in the body. The document and search routes keep that
convention; they are not conversational and nothing was gained by moving them.

The prefix is a literal segment rather than ``/api/v1/{projectId}/{gateway}``
on purpose. A bare two-segment pattern under ``/api/v1`` also matches
``/api/v1/document/status`` -- ``projectId="document", gateway="status"`` -- and
which one wins is registration order. That is a trap, not a design.

What differs per gateway, and it is only this
---------------------------------------------
::

    web       question in the body  -> answer in the response
    whatsapp  signed platform body  -> 200 now, answer pushed back later
    line      signed platform body  -> 200 now, answer pushed back later

Every one of them then runs the same ``runTurn``: open the conversation, fold it
if oversized, answer, record the turn. The gateway decides how the question
arrives and how the answer leaves, and nothing else.

``web`` is deliberately **not** a ``Channel``. It has no signature to verify, no
webhook envelope to parse, and no outbound API to send through, so implementing
that protocol for it would mean three methods that lie. It is handled directly
below; the registry holds only real gateways.

Authentication
--------------
The webhook gateways are the only authenticated surface in this service. Every
other endpoint takes an unverified ``serverId`` and trusts it, including ``web``
here -- but a webhook URL is public, anyone can find it, and what arrives is
answered by a paid model call, so every delivery is HMAC-checked against that
project's own secret over the raw bytes before anything is parsed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from app.agent.agent import ANSWER_TIMEOUT_SECONDS, answerQuestion
from app.agent.llmManager import LlmConfigError
from app.agent.promptStore import PromptStore, getPromptStore
from app.agent.usage import UsageLog
from app.agent.summariser import (
    applySummary,
    contextFromSearches,
    needsSummary,
    summariseConversation,
    trimToBudget,
)
from app.api.schemas import (
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ErrorResponse,
    WebhookAck,
    checkProjectId,
)
from app.channels.channel import ChannelError, IncomingMessage
from app.channels.registry import channelNames, getChannel
from app.channels.sender import sendReply
from app.ingestion.ragIngestionPipeline import ChunkStore
from app.stores.channelStore import ChannelStore, ChannelStoreError, getChannelStore
from app.stores.chunkStoreFactory import getChunkStore
from app.stores.conversationStore import (
    ConversationStore,
    ConversationStoreError,
    ConversationWindow,
    getConversationStore,
)
from app.stores.projectStore import ProjectStore, getProjectStore
from app.stores.usageStore import UsageStore, getUsageStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

# The one gateway that is not a messaging platform: an ordinary HTTP caller,
# answered on the response it asked on. Named rather than special-cased by
# absence, so a project cannot register a channel called "web" and shadow it.
WEB = "web"

# What a person is told when the answer could not be produced at all. Only the
# webhook gateways use it -- a web caller gets a status code, which is more
# useful to a program. Somebody waiting in a chat app gets nothing at all
# otherwise, and silence reads as the service being broken.
FAILURE_REPLY = (
    "Sorry, something went wrong answering that. Please try again in a moment."
)

# How long a delivered message id is remembered, so a redelivery is recognised.
# These platforms retry for minutes, not days.
SEEN_TTL_SECONDS = 3600

# What a validation error may say about itself. The same allowlist app.main uses
# on the app-wide handler: pydantic reports a bad field by attaching the whole
# request body as ``input``, and a 422 must not reflect that back.
SAFE_ERROR_KEYS = ("type", "loc", "msg")

# The most a body on these routes may be before it is refused unread. A real
# webhook delivery is single-digit kilobytes and a web question is capped at
# 4000 characters, so this is three orders of magnitude of headroom.
#
# It exists because both routes read the body themselves rather than letting
# FastAPI parse a declared model, and both are public. On the webhook side the
# body must be buffered *before* the signature over it can be checked -- the
# check comes after the read, so no amount of signature verification helps.
#
# The document, status and search routes are not covered: FastAPI reads those
# bodies before this module sees them. Capping request size across the whole
# service belongs in front of it -- a proxy, or an ASGI middleware -- and is
# noted in the known gaps rather than half-solved here.
MAX_WEBHOOK_BODY_BYTES = 1_000_000


def _projectIdOrRefuse(projectId: str) -> str:
    """The path's projectId, or a 422 saying what is wrong with it.

    A projectId becomes a Firestore document id. Reaching Firestore with one it
    will not accept -- too long, "..", wrapped in underscores -- surfaces as an
    InvalidArgument several layers down and escapes as a 500. The body models
    have carried this rule for a long time; the path did not, because the
    constraint lived on the pydantic field and moving the project into the URL
    left it behind. Both now use ``checkProjectId``.
    """
    try:
        return checkProjectId(projectId)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"projectId {exc}.",
        ) from None


async def _readBody(request: Request) -> bytes:
    """The raw body, refusing anything past the cap without buffering it.

    Streamed rather than ``await request.body()``: the point is to stop reading,
    and a length check after the fact has already paid the memory. Content-Length
    is not trusted for it either -- it is absent on a chunked request and can
    simply be wrong.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_WEBHOOK_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Webhook body too large.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _openConversation(
    conversations: ConversationStore,
    prompts: PromptStore,
    projectId: str,
    conversationId: str | None,
    question: str,
    ragDbId: str | None,
) -> tuple[ConversationWindow | None, str | None]:
    """The conversation this question belongs to, and the id to answer with.

    Three outcomes, and they are deliberately different:

    * no ``conversationId`` -- one is created, snapshotting the project's
      current system prompt onto it, and its window comes back empty. Only
      ``/query`` reaches this: ``/conversation/message`` requires an id, and
      creating one is ``/conversation``'s whole job.
    * a ``conversationId`` that exists -- **its window is loaded from Redis,
      and from Firestore only on a miss.** See ``FirestoreConversationStore.loadWindow``
      for the two halves; what comes back is the tail the summary does not
      already cover, not the whole transcript, which is what stops a long
      conversation costing more every time somebody adds to it.
    * a ``conversationId`` that does not exist -- 404. Not a new conversation:
      a mistyped id quietly starting a fresh one is indistinguishable, from the
      caller's side, from a model that has forgotten everything they said.

    A store that cannot be *reached* is none of those. It degrades to a
    single-turn answer, because an unreachable conversation store is a reason for a
    worse answer and not a reason for no answer at all -- the model call is the
    expensive part and it has not been made yet.

    Takes plain values rather than a request model because two different
    request shapes arrive here now -- ``QueryRequest`` and
    ``ConversationMessageRequest`` -- and threading both through would put a
    union type in the one function neither endpoint can afford to get wrong.
    """
    try:
        if conversationId is None:
            prompt = await prompts.systemPromptFor(projectId)
            window = await conversations.createConversation(
                projectId=projectId,
                systemPrompt=prompt,
                ragDbId=ragDbId,
                title=question,
            )
            return window, window.conversationId

        window = await conversations.loadWindow(projectId, conversationId)
        if window is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No conversation '{conversationId}' in project '{projectId}'.",
            )
        return window, window.conversationId
    except HTTPException:
        raise
    except ConversationStoreError:
        logger.exception(
            "The conversation store is unreachable; answering project '%s' without history.",
            projectId,
        )
        # The caller's own id is echoed back rather than dropped: their
        # conversation still exists, this one turn simply did not reach it.
        return None, conversationId
    except Exception:
        logger.exception("Could not open a conversation for project '%s'.", projectId)
        return None, conversationId


async def _foldIfNeeded(
    conversations: ConversationStore,
    window: ConversationWindow,
    usage: UsageLog | None = None,
) -> ConversationWindow:
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

    folded = await summariseConversation(window, usage=usage)
    if folded is None:
        return trimToBudget(window)

    summary, throughTurn, throughContext = folded
    try:
        await conversations.saveSummary(
            projectId=window.projectId,
            conversationId=window.conversationId,
            summary=summary,
            throughTurn=throughTurn,
            throughContext=throughContext,
        )
    except Exception:
        # The summary is still usable for this turn even if it was not stored;
        # the next question simply pays to make it again.
        logger.exception(
            "Could not store the summary for conversation '%s'.", window.conversationId
        )

    logger.info(
        "Folded conversation '%s' down to a summary through turn %d.",
        window.conversationId,
        throughTurn,
    )
    return applySummary(window, summary, throughTurn, throughContext)


async def _recordTurn(
    conversations: ConversationStore,
    window: ConversationWindow | None,
    projectId: str,
    question: str,
    answer: str,
    searchLog: list[dict],
    reviewOutcome,
) -> str | None:
    """Store the exchange and everything it retrieved. Never raises.

    Deliberately after the answer is in hand and deliberately unable to fail the
    request. The model call is paid for and the caller is owed its result; a
    Firestore write that did not land costs the *next* turn some context, which
    is a far smaller loss than turning a completed answer into a 500. It is
    logged loudly because a conversation that is silently not being written is a
    bug that would otherwise only surface as a model with no memory.

    This is also what refreshes the Redis copy: ``appendTurn`` writes the
    updated window back to the cache, so the next question on this conversation
    is answered from Redis rather than from a Firestore read of two
    subcollections.
    """
    if window is None:
        return None
    try:
        updated = await conversations.appendTurn(
            window=window,
            question=question,
            answer=answer,
            context=contextFromSearches(searchLog),
            reviewScore=reviewOutcome.score,
            retried=reviewOutcome.retried,
        )
    except Exception:
        logger.exception(
            "Answered project '%s' but could not record the turn on conversation '%s'.",
            projectId,
            window.conversationId,
        )
        return None

    # The assistant's own turn index, zero-padded to match the document it was
    # just written to. Returned so the cost of producing it can be filed under
    # the same id -- a usage row and a message row that cannot be joined are
    # two reports rather than one.
    return f"{updated.turnCount - 1:06d}"


async def runTurn(
    *,
    projectId: str,
    question: str,
    conversationId: str | None,
    store: ChunkStore,
    projects: ProjectStore,
    conversations: ConversationStore,
    prompts: PromptStore,
    usageStore: UsageStore | None = None,
    channel: str = WEB,
    externalMessageId: str = "",
) -> tuple[str, str | None]:
    """One question answered inside a conversation. Returns (answer, id).

    The whole pipeline, shared by ``/query`` and ``/conversation/message``::

        resolve project -> open the conversation -> fold if oversized
                        -> answer -> record the turn

    One copy, because the two endpoints differ only in whether a
    ``conversationId`` is optional. Two copies would drift, and the half that
    drifts first is the error handling -- which is the half where a caller gets
    a 500 for something that should have been a 504.

    Everything in that line except "answer" degrades rather than fails. The
    model call is the expensive, irreversible step, so nothing to do with
    storing a conversation may cost a caller an answer already paid for -- see
    ``_openConversation`` and ``_recordTurn``.

    What it cost is accumulated as it goes and written at the end. Three models
    are billed here -- the summariser folds, the agent answers, the reviewer
    grades, and the agent may answer twice -- and only one of those is visible
    from outside, so a single total would hide the two that surprise people.
    ``channel`` and ``externalMessageId`` are carried purely so a bill can be
    traced back to the WhatsApp or LINE message that caused it.
    """
    usage = UsageLog()
    # Read-only. Asking a question does not bring a RAG database into existence;
    # only /document may do that. A project nothing was ingested into still
    # answers -- the agent simply gets no search tool.
    ragDbId = await projects.resolve(projectId)

    window, resolvedId = await _openConversation(
        conversations, prompts, projectId, conversationId, question, ragDbId
    )
    if window is not None:
        window = await _foldIfNeeded(conversations, window, usage)

    # Filled by the search tool as the agent runs, and read afterwards. A
    # mutable argument rather than a return value because the agent's own
    # return is the answer, and the retrievals have to come back from a run the
    # agent does not otherwise report on.
    #
    # Only a *completed* turn is stored, so retrievals made by a run that then
    # timed out or 502'd are dropped. They were paid for and could in principle
    # be kept, but there is no exchange to hang them on -- storing them would
    # mean a context entry belonging to a turn the conversation has no record of.
    searchLog: list[dict] = []

    try:
        # Bounded. Nothing inside the model client caps a whole answer -- turns,
        # tool calls, the review and a retry -- so without this a provider that
        # stalls holds the connection until the caller gives up, and this
        # process holds a worker slot for exactly as long.
        async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
            result = await answerQuestion(
                projectId=projectId,
                question=question,
                ragDbId=ragDbId,
                chunkStore=store,
                conversationWindow=window,
                searchLog=searchLog,
                usage=usage,
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
            projectId,
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
        logger.exception("The agent failed to answer for project '%s'.", projectId)
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
            projectId,
            result.reviewOutcome.score,
        )

    messageId = await _recordTurn(
        conversations, window, projectId, question, result.answer, searchLog, result.reviewOutcome
    )

    await _recordUsage(
        usageStore, projectId, resolvedId, messageId, usage, channel, externalMessageId
    )

    return result.answer, resolvedId


async def _recordUsage(
    usageStore: UsageStore | None,
    projectId: str,
    conversationId: str | None,
    messageId: str | None,
    usage: UsageLog,
    channel: str,
    externalMessageId: str,
) -> None:
    """File what the answer cost. Never raises, and never blocks the answer.

    Skipped only when there is no conversation to file it against -- which
    means the conversation store was unreachable and there is no id that would
    still mean anything tomorrow. The tokens are logged in that case rather
    than attributed to something invented.

    A turn whose *message* could not be written still gets a row, under a
    generated id: the money was spent either way, and losing it from the
    project total to keep the join tidy is the wrong trade.
    """
    logger.info("Answer for project '%s' cost %s.", projectId, usage.summary())

    if usageStore is None or conversationId is None:
        if usage.entries and conversationId is None:
            logger.warning(
                "No conversation to attribute %s to on project '%s'.",
                usage.summary(),
                projectId,
            )
        return

    await usageStore.recordTurn(
        projectId=projectId,
        conversationId=conversationId,
        messageId=messageId or f"unrecorded-{uuid.uuid4().hex[:12]}",
        byRole=usage.byRole(),
        channel=channel,
        externalMessageId=externalMessageId,
    )



# --- the webhook half ------------------------------------------------------


async def _alreadyHandled(message: IncomingMessage) -> bool:
    """Whether this exact message has been answered already.

    Every one of these platforms redelivers on a failure, and some redeliver
    without one. Without this a retry costs a second model call and sends the
    person a second copy of the same answer.

    Redis is the natural home -- already required, and this is exactly the
    short-lived working state it holds. If it cannot be reached the message is
    treated as new: answering twice is a worse outcome than not answering, so
    this fails **open**, which is the deliberate opposite of the signature check.

    ``message.isRedelivery`` is not consulted. LINE sets it, and it means "we
    did not get a 2xx last time" -- which usually means the answer was never
    queued, so a redelivery is exactly the case that *should* be answered. It is
    carried for the log, not for this decision.
    """
    if not message.messageId:
        return False

    from app.infra.redisClient import redisClient

    redis = redisClient()
    if redis is None:
        return False

    key = f"ragChannelSeen:{message.channel}:{message.messageId}"
    try:
        # SET NX: claims the id and says whether it was already claimed, in one
        # round trip and without a read-then-write race between two deliveries
        # arriving at once.
        claimed = await asyncio.to_thread(redis.set, key, "1", nx=True, ex=SEEN_TTL_SECONDS)
        return not claimed
    except Exception:
        logger.warning(
            "Could not check whether %s was already handled; answering it.",
            message.messageId,
            exc_info=True,
        )
        return False


async def _runTurnFor(
    message: IncomingMessage,
    conversationId: str | None,
    store: ChunkStore,
    projects: ProjectStore,
    conversations: ConversationStore,
    prompts: PromptStore,
    usageStore: UsageStore,
) -> tuple[str, str | None]:
    """``runTurn``, with one retry when the stored conversation has gone.

    A linked conversation can expire -- ``RAG_CONVERSATION_TTL_SECONDS`` is 90
    days -- and ``runTurn`` answers a missing one with a 404, which is right for
    an HTTP caller and useless here. Nobody messaging on WhatsApp should be told
    their conversation expired; they should just be answered, from a fresh one.
    """
    try:
        return await runTurn(
            projectId=message.projectId,
            question=message.text,
            conversationId=conversationId,
            store=store,
            projects=projects,
            conversations=conversations,
            prompts=prompts,
            usageStore=usageStore,
            channel=message.channel,
            externalMessageId=message.messageId,
        )
    except HTTPException as exc:
        if exc.status_code != status.HTTP_404_NOT_FOUND or conversationId is None:
            raise
        logger.info(
            "Conversation '%s' for %s is gone; starting a new one.",
            conversationId,
            message.threadKey,
        )
        return await runTurn(
            projectId=message.projectId,
            question=message.text,
            conversationId=None,
            store=store,
            projects=projects,
            conversations=conversations,
            prompts=prompts,
            usageStore=usageStore,
            channel=message.channel,
            externalMessageId=message.messageId,
        )


async def _answerAndSend(
    message: IncomingMessage,
    store: ChunkStore,
    projects: ProjectStore,
    conversations: ConversationStore,
    prompts: PromptStore,
    channels: ChannelStore,
    usageStore: UsageStore,
) -> None:
    """Produce the answer and push it back. Runs after the webhook was answered.

    Never raises. There is no caller left to raise to -- this runs in a
    background task, where an exception is logged by the framework at best and
    swallowed at worst. Every failure ends with *something* going back to the
    person instead, because they are sitting in a chat app waiting and silence
    is indistinguishable from the service being down.

    **Neither platform has a conversation id**, so which conversation this
    belongs to is resolved from who is speaking -- see
    ``app.stores.channelStore``. WhatsApp identifies a person by their phone
    number and LINE by a userId scoped to the Official Account; neither offers
    any notion of a thread, and neither ever says a conversation has ended.
    """
    conversationId = await channels.conversationFor(message.projectId, message.threadKey)

    try:
        answer, resolvedId = await _runTurnFor(
            message, conversationId, store, projects, conversations, prompts, usageStore
        )
    except Exception:
        logger.exception(
            "Could not answer %s on project '%s'.", message.threadKey, message.projectId
        )
        await sendReply(message, FAILURE_REPLY, channels)
        return

    delivered = await sendReply(message, answer, channels)

    # Linked only after a successful send. A conversation the person never
    # received an answer from is one they will ask again from; pointing them at
    # it would mean the model believes it already answered.
    if delivered and resolvedId and resolvedId != conversationId:
        await channels.linkConversation(message.projectId, message.threadKey, resolvedId)


async def _receiveWebhook(
    gateway: str,
    projectId: str,
    request: Request,
    background: BackgroundTasks,
    store: ChunkStore,
    projects: ProjectStore,
    conversations: ConversationStore,
    prompts: PromptStore,
    channels: ChannelStore,
    usageStore: UsageStore,
) -> WebhookAck:
    """Verify a delivery, queue its messages, and acknowledge.

    The order is the point: nothing is parsed, and certainly nothing is
    answered, until the signature over the raw bytes checks out.
    """
    channel = getChannel(gateway)

    try:
        config = await channels.configFor(projectId, gateway)
    except ChannelStoreError:
        # A 503 rather than a 404. The platform redelivers a 5xx, which is what
        # should happen while Firestore is briefly unreachable -- a 404 tells it
        # the URL is wrong, and Meta eventually disables a webhook that keeps
        # answering that way.
        logger.exception("Could not read the %s config for '%s'.", gateway, projectId)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Channel configuration is unavailable; retry.",
        ) from None

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{projectId}' is not configured for {gateway}.",
        )

    # The exact bytes the platform signed. Parsing first and re-serialising
    # would change key order and whitespace, and the signature would then fail
    # in a way that looks like a wrong secret.
    body = await _readBody(request)

    if not channel.verify(body, {k.lower(): v for k, v in request.headers.items()}, config):
        logger.warning(
            "Rejected an unverified %s delivery for project '%s'.", gateway, projectId
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Signature verification failed."
        )

    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        # Signed, so it really is from the platform -- but unreadable. A 200
        # stops it being redelivered forever; no retry would fix it.
        logger.warning("A verified %s delivery was not JSON; ignoring.", gateway)
        return WebhookAck(accepted=0)

    queued = 0
    for message in channel.parse(payload, projectId):
        if await _alreadyHandled(message):
            logger.info("Skipping %s: already handled.", message.messageId)
            continue
        background.add_task(
            _answerAndSend,
            message,
            store,
            projects,
            conversations,
            prompts,
            channels,
            usageStore,
        )
        queued += 1

    logger.info("Accepted %d message(s) from %s for project '%s'.", queued, gateway, projectId)
    return WebhookAck(accepted=queued)


# --- the routes ------------------------------------------------------------


@router.post(
    "/{projectId}",
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversation and get its id back",
    responses={
        503: {
            "model": ErrorResponse,
            "description": "The conversation store was unreachable; nothing was created",
        },
    },
)
async def createConversation(
    projectId: str,
    payload: ConversationCreateRequest,
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
    conversations: Annotated[ConversationStore, Depends(getConversationStore)],
    prompts: Annotated[PromptStore, Depends(getPromptStore)],
) -> ConversationCreateResponse:
    """Create an empty conversation and return the id to post questions to.

    For the caller that needs the id *before* there is a question. A UI opening
    a new conversation has a window to render, a URL to route to, and a record
    to attach to the user the moment the user clicks, and none of that can wait
    on a model call. Posting to the ``web`` gateway without a ``conversationId``
    also creates one, so this is a convenience rather than a required first step.

    The webhook gateways never call it: WhatsApp and LINE have no conversation
    id of their own to offer, so one is minted on their behalf the first time
    somebody speaks and remembered against them.

    Two things are decided here and cannot be decided later: the **system
    prompt** is snapshotted onto the conversation, so it answers under the
    instructions it was opened with even if the project's prompt is edited half
    way through; and the **ragDbId** is recorded for audit, resolved read-only,
    because starting a conversation must not bring a RAG database into
    existence.

    A store failure here is a 503 rather than a degraded success -- this request
    *is* the creation, and a 201 carrying an id nothing stored would be a
    conversation the caller can address and every later request will 404.
    """
    projectId = _projectIdOrRefuse(projectId)
    logger.info(
        "New conversation requested by '%s' on project '%s'.", payload.server_id, projectId
    )

    ragDbId = await projects.resolve(projectId)

    # Degrades to the default prompt on its own if Firestore is unreachable --
    # see invariant 21 in app.agent.promptStore -- so this needs no guard. The
    # conversation is then snapshotted with the default, which is a worse
    # conversation and not a broken one.
    systemPrompt = await prompts.systemPromptFor(projectId)

    try:
        window = await conversations.createConversation(
            projectId=projectId,
            systemPrompt=systemPrompt,
            ragDbId=ragDbId,
            title=payload.title,
        )
    except Exception:
        logger.exception("Could not create a conversation for project '%s'.", projectId)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"A conversation could not be started for project '{projectId}'. "
                f"Nothing was created; retry, or post to the web gateway without a "
                f"conversationId to have one started with the question."
            ),
        ) from None

    return ConversationCreateResponse(
        conversation_id=window.conversationId,
        project_id=projectId,
        system_prompt=window.systemPrompt,
    )


@router.get(
    "/{projectId}/{gateway}",
    response_class=PlainTextResponse,
    summary="A gateway's webhook verification handshake",
    responses={
        403: {"model": ErrorResponse, "description": "Verification failed"},
        404: {"model": ErrorResponse, "description": "Unknown gateway, or project not on it"},
        405: {"model": ErrorResponse, "description": "This gateway has no handshake"},
    },
)
async def gatewayHandshake(
    projectId: str,
    gateway: str,
    request: Request,
    channels: Annotated[ChannelStore, Depends(getChannelStore)],
) -> PlainTextResponse:
    """Answer a platform's URL-verification GET.

    Meta calls this once, when the webhook URL is saved, and delivers nothing
    until it succeeds: a GET carrying ``hub.challenge``, which has to come back
    as the bare string with no JSON around it. There is no body to sign, which
    is what ``verifyToken`` stands in for.

    LINE has no handshake -- its console's "Verify" button sends a real, signed
    POST instead -- so this answers 405 for it rather than pretending.
    """
    projectId = _projectIdOrRefuse(projectId)
    channel = _gatewayOrRefuse(gateway)

    if not getattr(channel, "usesHandshake", False):
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail=f"The {gateway} gateway has no GET handshake.",
        )

    config = await channels.configFor(projectId, gateway)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{projectId}' is not configured for {gateway}.",
        )

    challenge = channel.handshake(dict(request.query_params), config)
    if challenge is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Verification failed."
        )
    return PlainTextResponse(challenge)


@router.post(
    "/{projectId}/{gateway}",
    summary="Say something in a conversation, through any gateway",
    responses={
        403: {"model": ErrorResponse, "description": "A webhook signature did not verify"},
        404: {
            "model": ErrorResponse,
            "description": "Unknown gateway, unknown conversationId, or project not on it",
        },
        502: {"model": ErrorResponse, "description": "The model provider failed (web only)"},
        503: {"model": ErrorResponse, "description": "The agent could not run"},
        504: {"model": ErrorResponse, "description": "Past RAG_ANSWER_TIMEOUT_SECONDS (web only)"},
    },
)
async def postMessage(
    projectId: str,
    gateway: str,
    request: Request,
    background: BackgroundTasks,
    store: Annotated[ChunkStore, Depends(getChunkStore)],
    projects: Annotated[ProjectStore, Depends(getProjectStore)],
    conversations: Annotated[ConversationStore, Depends(getConversationStore)],
    prompts: Annotated[PromptStore, Depends(getPromptStore)],
    channels: Annotated[ChannelStore, Depends(getChannelStore)],
    usageStore: Annotated[UsageStore, Depends(getUsageStore)],
):
    """One entry point, three arrival shapes.

    ``web`` takes ``{serverId, question, conversationId?}`` and answers on this
    response. Omit ``conversationId`` and one is started; send one this project
    does not have and it is a 404, never a new conversation -- a typo silently
    opening a fresh one looks, from the caller's side, exactly like a model that
    has forgotten everything.

    ``whatsapp`` and ``line`` take the platform's own signed body, verify it,
    acknowledge with 200, and answer afterwards -- the reply is pushed back
    through the platform's API by ``app.channels.sender``. The response says
    only how many messages were accepted, because the platform reads nothing
    else and a webhook response is not a place to describe this service.

    The body is not declared as a typed parameter, because it differs per
    gateway and FastAPI would have to pick one. Web bodies are validated
    against ``ConversationMessageRequest`` below; platform bodies are validated
    by their signature, which is stronger.
    """
    projectId = _projectIdOrRefuse(projectId)

    if gateway == WEB:
        return await _answerOnTheResponse(
            projectId, request, store, projects, conversations, prompts, usageStore
        )

    _gatewayOrRefuse(gateway)
    return await _receiveWebhook(
        gateway,
        projectId,
        request,
        background,
        store,
        projects,
        conversations,
        prompts,
        channels,
        usageStore,
    )


def _gatewayOrRefuse(gateway: str):
    """The channel adapter for ``gateway``, or a 404 naming what does exist.

    A single parameterised route means an unknown gateway reaches code rather
    than being refused by the router, so this is where it stops -- and the
    message names the registered gateways, which a path-per-platform shape
    could not do.

    ``web`` is refused here rather than returned: it is not a ``Channel``, and
    both callers have already handled it -- the POST path returns before
    reaching this, and the GET path has no handshake to offer for it.
    """
    if gateway == WEB:
        raise HTTPException(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            detail="The web gateway has no handshake.",
        )
    try:
        return getChannel(gateway)
    except ChannelError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown gateway '{gateway}'. Available: "
                f"{', '.join([WEB, *channelNames()])}."
            ),
        ) from None


async def _answerOnTheResponse(
    projectId: str,
    request: Request,
    store: ChunkStore,
    projects: ProjectStore,
    conversations: ConversationStore,
    prompts: PromptStore,
    usageStore: UsageStore,
) -> ConversationMessageResponse:
    """The web gateway: a question in the body, the answer in the response.

    The only gateway that can do this, because it is the only one whose caller
    is still on the other end of the connection when the answer exists.
    """
    try:
        payload = ConversationMessageRequest.model_validate(json.loads(await _readBody(request)))
    except ValidationError as exc:
        # Hand-validated rather than declared, because this route's body shape
        # depends on the gateway -- FastAPI would have to pick one. The detail
        # is built the same way as the app-wide 422 handler in ``app.main``:
        # an allowlist of type/loc/msg, so which field was wrong comes back but
        # pydantic's ``input`` -- the caller's whole body -- never does.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[
                {key: value for key, value in error.items() if key in SAFE_ERROR_KEYS}
                for error in exc.errors()
            ],
        ) from None
    except Exception:
        # Not JSON at all.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The web gateway takes a JSON body with serverId and question.",
        ) from None

    logger.debug("Web message from '%s' on project '%s'.", payload.server_id, projectId)

    answer, conversationId = await runTurn(
        projectId=projectId,
        question=payload.question,
        conversationId=payload.conversation_id,
        store=store,
        projects=projects,
        conversations=conversations,
        prompts=prompts,
        usageStore=usageStore,
    )

    return ConversationMessageResponse(
        answer=answer, project_id=projectId, conversation_id=conversationId
    )
