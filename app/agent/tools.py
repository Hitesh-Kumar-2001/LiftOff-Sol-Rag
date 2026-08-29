"""What the agent can reach for when the question needs more than it knows.

Two tools, and the agent decides whether either is worth calling -- there is no
forced retrieval step. A question about the conversation, or a request to
rephrase the last answer, should not cost a vector search.

``searchProject`` is built per request and closed over one ``ragDbId``. That is
the whole authorisation story for retrieval: the agent is handed a tool that can
only see the project the request named, so no prompt reaching it can talk it
into reading another project's documents. Do not add a projectId argument to
this tool.

It also closes over an optional ``searchLog``, which is how a retrieval survives
the request that made it: what the tool found is appended there, and the route
stores it on the conversation so a follow-up is answered from the passages rather than
by searching for them again. That is a second closed-over value and *not* a
second tool argument -- the model's view of this tool stays exactly one string
in, one string out, which is what ``tests/testAgent.py`` pins.
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

ENV_TAVILY_API_KEY = "TAVILY_API_KEY"

# Chunks per retrieval call. Enough to answer from, small enough that several
# searches in one turn do not crowd out the conversation.
SEARCH_TOP_K = int(os.environ.get("RAG_AGENT_SEARCH_TOP_K", 6))

TAVILY_MAX_RESULTS = int(os.environ.get("RAG_TAVILY_MAX_RESULTS", 5))


def buildProjectSearchTool(store, ragDbId: str, searchLog: list[dict] | None = None) -> BaseTool:
    """A retrieval tool bound to one project's database.

    ``searchLog`` collects ``{"query", "passages"}`` for every search that
    reached the store, so the caller can persist them onto the conversation. None means
    nobody is recording -- the tool behaves identically either way.
    """

    @tool("searchProject")
    async def searchProject(query: str) -> str:
        """Search this project's own documents and return the passages that
        match. Use this before answering anything factual about the project.
        Call it more than once with different wordings if the first results
        look thin."""
        try:
            results = await store.search(ragDbId, query, SEARCH_TOP_K)
        except Exception:
            # Reported to the model rather than raised. An exception out of a
            # tool aborts the graph, so a Pinecone timeout would turn a question
            # the agent could still have answered honestly into a 500. Told
            # instead, it can say what it could not do.
            logger.exception("Search of '%s' failed.", ragDbId)
            return (
                "The document search is unavailable right now, so this project's "
                "documents could not be consulted. Do not retry the search. Answer "
                "from what you already have and say plainly that you could not "
                "search the project's documents."
            )

        # Recorded before the empty check, and only on the path where the store
        # actually answered. A search that found nothing is worth keeping -- it
        # is the record that this project's documents do not cover the topic,
        # and without it every follow-up re-runs the same fruitless search. A
        # search that *failed* is not: nothing was learned, and storing it
        # would tell the next turn the documents are silent on something that
        # was never really looked for.
        if searchLog is not None:
            searchLog.append({"query": query, "passages": [result.text for result in results]})

        if not results:
            return (
                "No passages matched. The project may have nothing on this topic, "
                "or the wording may need to be different."
            )

        # Numbered and separated so the model can refer to a passage, and so two
        # chunks are never silently read as one piece of prose.
        return "\n\n".join(
            f"[passage {result.index}] {result.text}" for result in results
        )

    return searchProject


def buildWebSearchTool() -> BaseTool | None:
    """Tavily search, or None when no key is configured.

    Returning None rather than raising is deliberate: web search is an
    enhancement, and a deployment without a Tavily key should still answer from
    its own documents. The agent simply never sees the tool.
    """
    if not os.environ.get(ENV_TAVILY_API_KEY, "").strip():
        logger.info("No %s: the agent will answer without web search.", ENV_TAVILY_API_KEY)
        return None

    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=TAVILY_MAX_RESULTS)


def buildTools(
    store, ragDbId: str | None, searchLog: list[dict] | None = None
) -> list[BaseTool]:
    """Every tool this request's agent gets.

    ``ragDbId`` is None for a project nothing has been ingested into. It gets no
    search tool at all rather than one that always answers "no passages" --
    a tool that never works is worse than an absent one, because the model will
    keep trying it.

    ``searchLog`` is passed to the project search tool only. Web results are
    deliberately not recorded onto a conversation: they are somebody else's pages,
    they go stale, and re-fetching one is a Tavily call rather than the vector
    search this mechanism exists to avoid paying for twice.
    """
    tools: list[BaseTool] = []
    if ragDbId is not None:
        tools.append(buildProjectSearchTool(store, ragDbId, searchLog))

    web = buildWebSearchTool()
    if web is not None:
        tools.append(web)

    return tools
