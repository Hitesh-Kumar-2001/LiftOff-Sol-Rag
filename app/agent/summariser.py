"""Turning a stored conversation into model input, and stopping it growing.

Two halves of one problem, which is why they share a module.

**Rendering.** A ``ConversationWindow`` is storage: a summary, a list of retrievals, a
list of turns. ``renderContext`` and ``renderHistory`` turn it into the two
things a model actually takes -- a system prompt and a message list.

Retrieved passages go into the **system prompt**, not back into the transcript
as replayed tool calls. Replaying them faithfully would mean reconstructing
provider-specific tool-call ids and pairing every call with its result, which
is four different shapes across our four providers and breaks the moment one of
them changes. A prompt section is provider-independent and achieves the actual
goal: the passages are in front of the model, so a follow-up does not pay for
the same vector search twice.

**Summarising.** Context and history both grow without bound, and a
conversation whose prompt gets longer every turn costs more every turn and
eventually stops fitting at all. Past a budget, everything except the last few
exchanges is folded into one paragraph of prose and the documents behind it are
never read again. See ``app.stores.conversationStore`` for the watermarks that record
how far the fold reached.

**The recent turns are kept verbatim.** Summarising the whole conversation
including the question just asked would blur exactly the part the model needs
in full -- "make that shorter" only means something next to the text it refers
to. Old material is where detail is cheap to lose and expensive to carry.

**A summariser failure is not an error.** The question is still answerable; it
just costs more tokens this turn. The fallback is ``trimToBudget``, which drops
the oldest retrievals from *what is sent* without touching what is stored --
because a window that will not fit is the one failure mode that turns a
recoverable cost problem into a provider error.
"""

from __future__ import annotations

import logging
import os

from app.agent.llmManager import summariserModel
from app.stores.conversationStore import ConversationWindow, ContextEntry

logger = logging.getLogger(__name__)

# Above this, the conversation is folded down. Configured in tokens because
# that is the unit the limit actually lives in, but measured in characters at
# roughly four per token: an exact count means BPE-encoding the whole
# conversation on every single question, which is real CPU on the event loop in
# exchange for precision a *threshold* does not need.
SUMMARY_TRIGGER_TOKENS = int(os.environ.get("RAG_CONTEXT_SUMMARY_TOKENS", 6000))
CHARS_PER_TOKEN = 4

# How many messages survive the fold untouched -- four is two exchanges, which
# is what "shorter, please" and "what about the second one" need in full.
KEEP_RECENT_TURNS = int(os.environ.get("RAG_CONTEXT_KEEP_TURNS", 4))

# Ceiling on the summary itself, so folding a long conversation cannot simply
# move the growth into the thing that was meant to bound it.
SUMMARY_MAX_CHARS = int(os.environ.get("RAG_CONTEXT_SUMMARY_MAX_CHARS", 6000))

SUMMARISER_SYSTEM_PROMPT = (
    "You compress the earlier part of an assistant conversation into a brief "
    "another assistant will answer from. It replaces the material entirely -- "
    "whatever you leave out is gone.\n\n"
    "Keep: concrete facts, figures, names, dates and definitions taken from the "
    "project's documents; what the user is trying to do; decisions, corrections "
    "and preferences they stated; questions raised but not yet answered; and "
    "anything the documents were searched for and did NOT contain.\n\n"
    "Drop: greetings, acknowledgements, restatements, and the assistant's own "
    "hedging.\n\n"
    "Write plain prose in the third person, under 300 words. Do not address the "
    "user, do not offer to help, and do not invent anything that is not in the "
    "material you were given."
)


def approximateTokens(window: ConversationWindow) -> int:
    """Roughly what this window will cost to send. See CHARS_PER_TOKEN."""
    characters = len(window.contextSummary)
    characters += sum(len(message.content) for message in window.messages)
    for entry in window.context:
        characters += len(entry.query) + sum(len(passage) for passage in entry.passages)
    return characters // CHARS_PER_TOKEN


def needsSummary(window: ConversationWindow) -> bool:
    """Whether this window is over budget and worth folding down.

    The second condition matters: a single enormous exchange is over budget and
    cannot be helped by summarising, because there is nothing outside the
    keep-verbatim tail to fold. Summarising it anyway would pay for a model
    call, change nothing, and do it again on the very next question.
    """
    if approximateTokens(window) <= SUMMARY_TRIGGER_TOKENS:
        return False
    return window.turnCount - window.summarisedThroughTurn > KEEP_RECENT_TURNS


def _foldPoint(window: ConversationWindow) -> tuple[int, int]:
    """The turn and context watermarks a fold of this window would reach.

    Everything strictly below them is summarised away; everything at or above
    stays verbatim. Context is folded by the *turn that caused it*, so a
    retrieval never outlives the exchange it was run for.
    """
    throughTurn = max(window.summarisedThroughTurn, window.turnCount - KEEP_RECENT_TURNS)

    folded = [entry for entry in window.context if entry.turnIndex < throughTurn]
    throughContext = (
        folded[-1].entryIndex + 1 if folded else window.summarisedThroughContext
    )
    return throughTurn, throughContext


def renderContext(window: ConversationWindow) -> str:
    """The retrieval block for the system prompt, or "" when there is none.

    Ends with an instruction, not just material. Passages sitting in a prompt
    with nothing said about them get treated as background and searched for
    again anyway, which is the one outcome this whole mechanism exists to
    prevent -- so the model is told plainly that this ground is already covered.
    """
    sections: list[str] = []

    if window.contextSummary:
        sections.append(f"### Earlier in this conversation\n{window.contextSummary}")

    for entry in window.context:
        passages = "\n".join(
            f"[passage {index}] {passage}" for index, passage in enumerate(entry.passages)
        )
        sections.append(f'### Already searched: "{entry.query}"\n{passages or "(no matches)"}')

    if not sections:
        return ""

    return (
        "\n\n## What this conversation has already established\n\n"
        "The material below was retrieved earlier in this same conversation and "
        "is still current. Treat it as already searched: use it directly, and "
        "only call searchProject for ground it does not cover.\n\n"
        + "\n\n".join(sections)
    )


def renderHistory(window: ConversationWindow) -> list[dict[str, str]]:
    """The stored turns as messages, oldest first.

    Only ``role`` and ``content``: the review score and the retried flag are
    recorded for whoever reads the conversation later, and telling a model that its own
    earlier answer was graded 0.4 invites it to apologise for it instead of
    answering the question in front of it.
    """
    return [
        {"role": message.role, "content": message.content}
        for message in sorted(window.messages, key=lambda message: message.turnIndex)
        if message.content.strip()
    ]


def trimToBudget(window: ConversationWindow) -> ConversationWindow:
    """Drop the oldest retrievals until the window fits. Storage is untouched.

    The fallback for a window that is over budget and could not be summarised,
    and the reason a broken summariser costs money rather than answers. Context
    goes first and messages are never dropped: passages are bulk that can be
    searched for again, while a missing turn silently rewrites the conversation
    the user can see they had.
    """
    trimmed = window.copy()
    dropped = 0
    while trimmed.context and approximateTokens(trimmed) > SUMMARY_TRIGGER_TOKENS:
        trimmed.context.pop(0)
        dropped += 1

    if dropped:
        logger.warning(
            "Conversation '%s' was over budget and could not be summarised; %d retrieval(s) "
            "were left out of this turn's prompt. They are still stored.",
            window.conversationId,
            dropped,
        )
    return trimmed


def _foldedMaterial(window: ConversationWindow, throughTurn: int) -> str:
    """What the summariser is asked to compress: prior summary, turns, passages."""
    parts: list[str] = []
    if window.contextSummary:
        parts.append(f"Summary of everything before this:\n{window.contextSummary}")

    exchanges = [
        f"{message.role}: {message.content}"
        for message in sorted(window.messages, key=lambda message: message.turnIndex)
        if message.turnIndex < throughTurn and message.content.strip()
    ]
    if exchanges:
        parts.append("Conversation:\n" + "\n\n".join(exchanges))

    retrievals: list[str] = []
    for entry in window.context:
        if entry.turnIndex >= throughTurn:
            continue
        passages = "\n".join(entry.passages)
        retrievals.append(f'Searched "{entry.query}" and found:\n{passages}')
    if retrievals:
        parts.append("Retrieved from the project's documents:\n" + "\n\n".join(retrievals))

    return "\n\n---\n\n".join(parts)


async def summariseConversation(
    window: ConversationWindow, model=None
) -> tuple[str, int, int] | None:
    """Fold this window down. Returns (summary, throughTurn, throughContext).

    Returns None when there was nothing to fold or the model could not be
    reached. Both are ordinary outcomes -- the caller answers from the
    un-summarised window either way, and tries again next turn.
    """
    throughTurn, throughContext = _foldPoint(window)
    material = _foldedMaterial(window, throughTurn)
    if not material.strip():
        return None

    try:
        response = await (model or summariserModel()).ainvoke(
            [
                {"role": "system", "content": SUMMARISER_SYSTEM_PROMPT},
                {"role": "user", "content": material},
            ]
        )
    except Exception:
        # Not fatal. The conversation is still answerable, it just costs this
        # turn more tokens than it should have -- and the caller falls back to
        # trimToBudget so "more" cannot become "too many".
        logger.exception(
            "Could not summarise conversation '%s'; answering unsummarised.",
            window.conversationId,
        )
        return None

    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Some providers return content blocks rather than a bare string.
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    summary = str(content or "").strip()[:SUMMARY_MAX_CHARS]
    if not summary:
        logger.warning(
            "The summariser returned nothing for conversation '%s'.", window.conversationId
        )
        return None

    return summary, throughTurn, throughContext


def applySummary(
    window: ConversationWindow, summary: str, throughTurn: int, throughContext: int
) -> ConversationWindow:
    """The window as it looks once a summary has been stored.

    Kept beside the store's own version of this so the caller can answer from a
    folded window in the same request that wrote the fold, rather than paying
    for a re-read to see its own write.
    """
    folded = window.copy()
    folded.contextSummary = summary
    folded.summarisedThroughTurn = throughTurn
    folded.summarisedThroughContext = throughContext
    folded.messages = [m for m in folded.messages if m.turnIndex >= throughTurn]
    folded.context = [c for c in folded.context if c.entryIndex >= throughContext]
    return folded


def contextFromSearches(searches: list[dict]) -> list[ContextEntry]:
    """Turn what the search tool recorded into storable context entries.

    Indices are placeholders: the store assigns the real ones inside the
    transaction that appends them, because only it knows how many entries the
    conversation already holds. See
    ``app.stores.conversationStore.FirestoreConversationStore._appendTurn``.

    Searches that returned nothing are kept, not filtered. "The documents do
    not cover this" is a real finding and the expensive one to rediscover --
    drop it and every follow-up runs the same fruitless search again.
    """
    return [
        ContextEntry(
            entryIndex=0,
            turnIndex=0,
            query=str(search.get("query", "")),
            passages=[str(passage) for passage in search.get("passages") or []],
        )
        for search in searches
        if str(search.get("query", "")).strip()
    ]
