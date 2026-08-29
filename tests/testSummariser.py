"""Rendering a conversation for the model, and folding it down when it grows.

Two things worth pinning. First, that retrieved passages actually reach the
prompt with an instruction attached -- material sitting there unannounced gets
searched for again, which defeats the entire mechanism. Second, that every
failure path still produces a window inside budget: a summariser that cannot
run must cost tokens, never an answer.
"""

import asyncio

import pytest

from app.agent import summariser
from app.agent.summariser import (
    applySummary,
    approximateTokens,
    contextFromSearches,
    needsSummary,
    renderContext,
    renderHistory,
    summariseConversation,
    trimToBudget,
)
from app.stores.conversationStore import ConversationMessage, ConversationWindow, ContextEntry

PROJECT = "handbook"


def window(**overrides) -> ConversationWindow:
    base = {
        "conversationId": "c1",
        "projectId": PROJECT,
        "systemPrompt": "You are the handbook assistant.",
    }
    return ConversationWindow(**(base | overrides))


# --- rendering -------------------------------------------------------------


def testAnEmptyConversationRendersNothing() -> None:
    """A first turn must look exactly like a single-turn question did before
    conversations existed -- no empty scaffolding in the prompt."""
    assert renderContext(window()) == ""
    assert renderHistory(window()) == []


def testRetrievedPassagesReachThePrompt() -> None:
    rendered = renderContext(
        window(context=[ContextEntry(0, 0, "refund window", ["Refunds: 30 days."])])
    )

    assert "Refunds: 30 days." in rendered
    assert "refund window" in rendered


def testThePassagesComeWithAnInstructionNotJustMaterial() -> None:
    """Passages with nothing said about them get treated as background and
    searched for again, which is the one outcome this exists to prevent."""
    rendered = renderContext(window(context=[ContextEntry(0, 0, "q", ["p"])]))

    assert "already searched" in rendered.lower()
    assert "searchProject" in rendered


def testASearchThatFoundNothingStillRenders() -> None:
    """Recording that the documents are silent on a topic is only useful if the
    model is told about it."""
    rendered = renderContext(window(context=[ContextEntry(0, 0, "parental leave", [])]))

    assert "parental leave" in rendered
    assert "no matches" in rendered


def testTheSummaryRendersAheadOfTheLiveContext() -> None:
    rendered = renderContext(
        window(
            contextSummary="They asked about refunds.",
            context=[ContextEntry(0, 0, "gift cards", ["Gift cards: no refund."])],
        )
    )

    assert rendered.index("They asked about refunds.") < rendered.index("Gift cards")


def testHistoryComesBackInTurnOrder() -> None:
    rendered = renderHistory(
        window(
            messages=[
                ConversationMessage(2, "user", "second question"),
                ConversationMessage(0, "user", "first question"),
                ConversationMessage(1, "assistant", "first answer"),
            ]
        )
    )

    assert [message["content"] for message in rendered] == [
        "first question",
        "first answer",
        "second question",
    ]


def testHistoryCarriesNoGrades() -> None:
    """Telling a model its earlier answer scored 0.4 invites it to apologise
    for that answer instead of answering the question in front of it."""
    rendered = renderHistory(
        window(
            messages=[
                ConversationMessage(
                    0, "assistant", "an answer", reviewScore=0.4, retried=True
                )
            ]
        )
    )

    assert rendered == [{"role": "assistant", "content": "an answer"}]


def testEmptyMessagesAreDropped() -> None:
    rendered = renderHistory(
        window(
            messages=[
                ConversationMessage(0, "user", "  "),
                ConversationMessage(1, "assistant", "real"),
            ]
        )
    )

    assert [message["content"] for message in rendered] == ["real"]


# --- deciding to fold ------------------------------------------------------


def testASmallConversationIsLeftAlone() -> None:
    assert not needsSummary(window(messages=[ConversationMessage(0, "user", "short")], turnCount=2))


def testAnOversizedConversationIsFolded() -> None:
    big = "x" * (summariser.SUMMARY_TRIGGER_TOKENS * summariser.CHARS_PER_TOKEN + 1)

    assert needsSummary(
        window(
            turnCount=10,
            messages=[ConversationMessage(index, "user", big) for index in range(10)],
        )
    )


def testOneEnormousExchangeIsNotWorthFolding() -> None:
    """It is over budget and summarising cannot help: there is nothing outside
    the keep-verbatim tail to fold, so a call would change nothing and be made
    again on the very next question."""
    big = "x" * (summariser.SUMMARY_TRIGGER_TOKENS * summariser.CHARS_PER_TOKEN + 1)

    assert not needsSummary(
        window(
            turnCount=2,
            messages=[
                ConversationMessage(0, "user", big),
                ConversationMessage(1, "assistant", big),
            ],
        )
    )


def testRetrievedPassagesCountTowardsTheBudget() -> None:
    """They are the bulk of a long conversation -- a budget that ignored them
    would never trigger on the thing actually filling the prompt."""
    passages = ["y" * 4000] * 10

    assert approximateTokens(window(context=[ContextEntry(0, 0, "q", passages)])) > 9000


# --- folding ---------------------------------------------------------------


def testTheRecentTurnsSurviveTheFold() -> None:
    """"Make that shorter" only means something next to the text it refers to."""
    folded = applySummary(
        window(
            turnCount=10,
            messages=[ConversationMessage(index, "user", f"turn {index}") for index in range(10)],
        ),
        summary="earlier",
        throughTurn=6,
        throughContext=0,
    )

    assert [message.turnIndex for message in folded.messages] == [6, 7, 8, 9]
    assert folded.contextSummary == "earlier"


def testFoldingDropsTheContextItCovers() -> None:
    folded = applySummary(
        window(
            turnCount=6,
            context=[ContextEntry(index, index * 2, f"q{index}", ["p"]) for index in range(3)],
        ),
        summary="earlier",
        throughTurn=4,
        throughContext=2,
    )

    assert [entry.entryIndex for entry in folded.context] == [2]


def testAFoldedWindowIsACopy() -> None:
    original = window(
        turnCount=6,
        messages=[ConversationMessage(index, "user", "t") for index in range(6)],
    )

    applySummary(original, summary="s", throughTurn=4, throughContext=0)

    assert len(original.messages) == 6


def testTheSummariserFoldsTheOldTurnsIntoProse() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.seen: list = []

        async def ainvoke(self, messages):
            self.seen = messages
            return type("Response", (), {"content": "They asked about refunds."})()

    model = FakeModel()
    result = asyncio.run(
        summariseConversation(
            window(
                turnCount=10,
                messages=[
                    ConversationMessage(index, "user", f"turn {index}")
                    for index in range(10)
                ],
            ),
            model=model,
        )
    )

    summary, throughTurn, _ = result
    assert summary == "They asked about refunds."
    assert throughTurn == 10 - summariser.KEEP_RECENT_TURNS
    # The turns being folded are what it was asked to compress; the ones being
    # kept verbatim are not, or the fold would blur what it just preserved.
    asked = model.seen[1]["content"]
    assert "turn 0" in asked
    assert "turn 9" not in asked


def testABrokenSummariserIsNotAnError() -> None:
    """The question is still answerable; it just costs more tokens this turn."""

    class BrokenModel:
        async def ainvoke(self, messages):
            raise RuntimeError("provider down")

    result = asyncio.run(
        summariseConversation(
            window(
                turnCount=10,
                messages=[ConversationMessage(i, "user", f"t{i}") for i in range(10)],
            ),
            model=BrokenModel(),
        )
    )

    assert result is None


def testThereIsNothingToFoldInAFreshConversation() -> None:
    class Unused:
        async def ainvoke(self, messages):  # pragma: no cover - must not be called
            raise AssertionError("The summariser was called with nothing to summarise.")

    assert asyncio.run(summariseConversation(window(), model=Unused())) is None


# --- the fallback ----------------------------------------------------------


def testTrimmingDropsTheOldestRetrievals() -> None:
    """The fallback when summarising fails. An over-budget prompt is the one
    outcome worth avoiding at any cost -- it turns an expensive question into a
    provider error."""
    passages = ["z" * 4000] * 3
    oversized = window(
        turnCount=4,
        context=[ContextEntry(index, 0, f"q{index}", passages) for index in range(8)],
    )

    trimmed = trimToBudget(oversized)

    assert approximateTokens(trimmed) <= summariser.SUMMARY_TRIGGER_TOKENS
    # The newest retrievals are the ones kept.
    assert trimmed.context[-1].entryIndex == 7


def testTrimmingNeverDropsAMessage() -> None:
    """Passages can be searched for again; a missing turn silently rewrites the
    conversation the user can see they had."""
    big = "w" * (summariser.SUMMARY_TRIGGER_TOKENS * summariser.CHARS_PER_TOKEN)
    oversized = window(turnCount=2, messages=[ConversationMessage(0, "user", big)])

    assert len(trimToBudget(oversized).messages) == 1


def testTrimmingLeavesTheStoredWindowAlone() -> None:
    oversized = window(
        turnCount=2, context=[ContextEntry(index, 0, "q", ["v" * 4000]) for index in range(20)]
    )

    trimToBudget(oversized)

    assert len(oversized.context) == 20


# --- what the tool recorded ------------------------------------------------


def testRecordedSearchesBecomeContextEntries() -> None:
    entries = contextFromSearches([{"query": "refunds", "passages": ["30 days."]}])

    assert entries == [ContextEntry(0, 0, "refunds", ["30 days."])]


def testAFruitlessSearchIsKept() -> None:
    """"The documents do not cover this" is a real finding, and the expensive
    one to rediscover."""
    assert contextFromSearches([{"query": "parental leave", "passages": []}])


def testAQuerylessRecordIsDropped() -> None:
    assert contextFromSearches([{"query": "  ", "passages": ["something"]}]) == []


@pytest.mark.parametrize("searches", [[], [{}]])
def testNothingRecordedIsNoContext(searches: list) -> None:
    assert contextFromSearches(searches) == []
