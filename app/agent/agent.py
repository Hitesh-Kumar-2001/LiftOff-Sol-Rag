"""The agent entry point: a question in, a reviewed answer out.

    resolve project -> prompt + conversation -> run the agent -> review once
                                                      ^              |
                                                      +--- retry ----+  (at most once)

The agent is a `deepagents` deep agent given three things the base harness does
not have: a search tool bound to this project's documents, this chat's system
prompt, and whatever the conversation has already established. Whether to
search, how many times, and with what wording are the model's decisions -- there
is no forced retrieval step, because a follow-up like "say that more briefly"
should not cost a vector search.

The conversation arrives as a ``ChatWindow`` and leaves as an entry in
``searchLog``; neither is loaded or stored here. This module answers one
question -- ``app.api.routes.query`` owns the chat around it, because durability
is not the agent's concern and a failure to write history must not lose an
answer that was already paid for.

**The built-in filesystem and shell tools are switched off.** A deep agent ships
with `ls`/`read_file`/`write_file`/`edit_file`/`delete`/`glob`/`grep`/`execute`,
which are the right tools for a coding agent and entirely wrong at the end of an
unauthenticated HTTP endpoint -- anyone who can reach `/query` could otherwise
ask the model to read or write files on the host. See ``_excludeHostTools``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_core.messages import AIMessage

from app.agent.llmManager import agentModel
from app.agent.promptStore import getPromptStore
from app.agent.reviewer import (
    ReviewOutcome,
    needsAnotherAttempt,
    retryInstruction,
    review,
)
from app.agent.summariser import renderContext, renderHistory
from app.agent.tools import buildTools
from app.stores.chatStore import ChatWindow

logger = logging.getLogger(__name__)

# Everything the deep agent harness would otherwise put on the host.
HOST_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)

# The harness keys its profile off the model's own provider name, which is not
# always our enum's spelling -- Gemini reports `google_genai`. Registering under
# every name any of our four models can report is what makes the exclusion below
# actually apply; a name we miss silently gets the full built-in tool suite.
PROFILE_KEYS = ("anthropic", "openai", "groq", "gemini", "google_genai", "google")

# What a caller gets when the agent produced no text at all -- a provider that
# stopped on a tool call it never completed, or a content filter. Better than
# answering 200 with an empty string, which reads as "the documents say nothing"
# rather than "this went wrong".
NO_ANSWER = (
    "No answer could be produced for that question. Please try rephrasing it."
)

# The longest one question may take end to end: the agent's turns, its tool
# calls, the review, and a retry. Nothing inside the model client bounds this,
# so without it a provider that stalls holds an HTTP connection open until the
# client gives up. Generous, because a question that searches several times
# legitimately takes tens of seconds.
ANSWER_TIMEOUT_SECONDS = float(os.environ.get("RAG_ANSWER_TIMEOUT_SECONDS", 120))

_profilesRegistered = False


def _excludeHostTools() -> None:
    """Strip the filesystem and shell tools from every provider's harness.

    `create_deep_agent` merges caller tools *into* its built-in suite rather
    than replacing it, so the only supported way to drop a built-in is a harness
    profile. Registration is additive and keyed per provider, so this registers
    the same exclusion under each name a provider can report.

    The auto-added ``general-purpose`` subagent goes too. It inherits this same
    exclusion, so it is not a way back to the host -- it is simply dead weight
    here: it would hand a copy of the question to a second agent holding the one
    tool this one already has, doubling the tokens and the latency of every
    question to reach the same passages.
    """
    global _profilesRegistered
    if _profilesRegistered:
        return

    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        register_harness_profile,
    )

    profile = HarnessProfile(
        excluded_tools=frozenset(HOST_TOOLS),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    for key in PROFILE_KEYS:
        register_harness_profile(key, profile)

    _profilesRegistered = True


@dataclass
class AgentAnswer:
    """What the agent produced, and what the reviewer made of it."""

    answer: str
    reviewOutcome: ReviewOutcome


def buildAgent(systemPrompt: str, tools, model=None):
    """Compile a deep agent for one request.

    Per request, not cached: the search tool is closed over this project's
    ragDbId, so a shared agent would hand one project's tool to another's
    question. The chat model underneath *is* cached -- see the LLM manager.
    """
    _excludeHostTools()

    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model or agentModel(),
        tools=tools,
        system_prompt=systemPrompt,
    )


def _lastText(result) -> str:
    """The answer out of a LangGraph result.

    The final message is the agent's reply; its content may be a plain string or
    a list of content blocks depending on the provider, so both are handled
    rather than assuming whichever one the current provider happens to return.
    """
    messages = (result or {}).get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str):
            if content.strip():
                return content.strip()
            continue
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
            if text:
                return text
    return ""


async def _run(agent, messages) -> tuple[str, list]:
    """Run one turn. Returns the answer text and the full transcript.

    The transcript is what a retry continues from: it holds the tool calls and
    the passages they returned, so a second attempt does not have to search the
    project again to keep the part of the first answer that was already right.
    """
    result = await agent.ainvoke({"messages": messages})
    return _lastText(result), list((result or {}).get("messages") or [])


async def answerQuestion(
    *,
    projectId: str,
    question: str,
    ragDbId: str | None,
    chunkStore,
    promptStore=None,
    chatWindow: ChatWindow | None = None,
    searchLog: list[dict] | None = None,
) -> AgentAnswer:
    """Answer ``question`` for ``projectId``, reviewing the result once.

    ``ragDbId`` None means nothing has been ingested into this project yet. The
    agent still runs -- it can answer from the system prompt, the conversation,
    or web search -- it just gets no retrieval tool. See ``buildTools``.

    ``chatWindow`` is the conversation this question belongs to: its prompt, the
    passages it has already retrieved, and its previous turns. None is a
    single-turn question, which is what every question was before chats existed
    and still what a caller sending no ``chatId`` gets.

    ``searchLog`` is where this turn's retrievals are recorded for the caller to
    store. Passed in rather than returned because this function's return value
    is the answer, and the tool fills the log while the graph is still running.
    """
    prompts = promptStore or getPromptStore()

    # A chat answers with the prompt it was created under, not the project's
    # current one. Re-resolving per turn would let an edit rewrite the
    # instructions mid-conversation, leaving the model bound to earlier answers
    # it would no longer have given. See app.stores.chatStore.
    systemPrompt = (
        chatWindow.systemPrompt
        if chatWindow is not None and chatWindow.systemPrompt
        else await prompts.systemPromptFor(projectId)
    )

    # Retrieved passages ride in the system prompt rather than as replayed tool
    # calls -- see app.agent.summariser for why. The effect is that a follow-up
    # starts already holding what the conversation has found.
    if chatWindow is not None:
        systemPrompt += renderContext(chatWindow)

    tools = buildTools(chunkStore, ragDbId, searchLog)
    agent = buildAgent(systemPrompt, tools)

    history = renderHistory(chatWindow) if chatWindow is not None else []
    messages = [*history, {"role": "user", "content": question}]
    answer, transcript = await _run(agent, messages)

    verdict = await review(question, answer)
    if not needsAnotherAttempt(verdict):
        return AgentAnswer(
            answer=answer or NO_ANSWER,
            reviewOutcome=ReviewOutcome(
                score=verdict.score if verdict else None,
                suggestion="",
                retried=False,
            ),
        )

    logger.info(
        "Answer for project '%s' scored %.2f; retrying once with the reviewer's "
        "suggestion.",
        projectId,
        verdict.score,
    )

    # Continued from the *transcript*, not from the question again: the tool
    # calls and the passages they returned are in there, so the second attempt
    # starts holding what the first retrieved instead of paying for the same
    # searches over. The fallback covers an agent that returned no transcript
    # at all, where replaying the question is the only thing left to do.
    priorTurn = transcript or [*messages, {"role": "assistant", "content": answer}]
    retried, _ = await _run(
        agent,
        [*priorTurn, {"role": "user", "content": retryInstruction(question, answer, verdict.suggestion)}],
    )

    # Deliberately not reviewed again -- see app.agent.reviewer. If the second
    # attempt came back empty, the first one is still the better answer.
    return AgentAnswer(
        answer=retried or answer or NO_ANSWER,
        reviewOutcome=ReviewOutcome(
            score=verdict.score, suggestion=verdict.suggestion, retried=True
        ),
    )
