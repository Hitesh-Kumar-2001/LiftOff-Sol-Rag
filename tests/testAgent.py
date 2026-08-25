"""The agent loop: answer, review once, retry at most once.

No provider is reached. The agent and the reviewer are both stubbed, because
what is being pinned down is the control flow -- how many times each runs, and
what comes back when one of them misbehaves.
"""

import asyncio

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import agent as agentModule
from app.agent.agent import (
    HOST_TOOLS,
    NO_ANSWER,
    AgentAnswer,
    _lastText,
    answerQuestion,
    buildAgent,
)
from app.agent.promptStore import PromptStore
from app.agent.reviewer import Review, needsAnotherAttempt, retryInstruction
from app.agent.tools import buildTools
from app.ingestion.ragIngestionPipeline import SearchResult

QUESTION = "What is the refund window?"


class FakeAgent:
    """Records each turn it is asked for and answers from a script.

    Returns the messages it was given plus its own reply, which is what
    LangGraph does -- and what the retry path depends on, since it continues
    from the transcript rather than from the question again.
    """

    def __init__(self, answers: list[str], toolTurns: list | None = None) -> None:
        self.answers = list(answers)
        self.toolTurns = list(toolTurns or [])
        self.calls: list[list] = []

    async def ainvoke(self, state):
        incoming = list(state["messages"])
        self.calls.append(incoming)
        text = self.answers.pop(0) if self.answers else ""
        extra = [self.toolTurns.pop(0)] if self.toolTurns else []
        return {"messages": [*incoming, *extra, AIMessage(content=text)]}


class FakeChunkStore:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.searched: list[tuple[str, str]] = []

    async def search(self, ragDbId, query, topK=5):
        self.searched.append((ragDbId, query))
        return self.results


@pytest.fixture
def prompts() -> PromptStore:
    return PromptStore(redis=None, firestore=None)


def install(monkeypatch, fake: FakeAgent, verdicts: list[Review | None]):
    """Swap in the stub agent and a scripted reviewer; return the review call log."""
    monkeypatch.setattr(agentModule, "buildAgent", lambda *a, **k: fake)
    reviews: list[tuple[str, str]] = []

    async def fakeReview(question, answer, model=None):
        reviews.append((question, answer))
        return verdicts.pop(0) if verdicts else None

    monkeypatch.setattr(agentModule, "review", fakeReview)
    return reviews


def run(prompts, chunkStore=None, ragDbId="acme-db") -> AgentAnswer:
    return asyncio.run(
        answerQuestion(
            projectId="acme",
            question=QUESTION,
            ragDbId=ragDbId,
            chunkStore=chunkStore or FakeChunkStore(),
            promptStore=prompts,
        )
    )


def testAGoodAnswerIsReturnedWithoutARetry(monkeypatch, prompts) -> None:
    fake = FakeAgent(["Thirty days from purchase."])
    reviews = install(monkeypatch, fake, [Review(score=0.9)])

    result = run(prompts)

    assert result.answer == "Thirty days from purchase."
    assert len(fake.calls) == 1, "the agent should have run once"
    assert len(reviews) == 1
    assert result.reviewOutcome.retried is False


def testAPoorAnswerIsRetriedOnce(monkeypatch, prompts) -> None:
    fake = FakeAgent(["Not sure.", "Thirty days from purchase."])
    install(monkeypatch, fake, [Review(score=0.2, suggestion="Give the actual window.")])

    result = run(prompts)

    assert result.answer == "Thirty days from purchase."
    assert len(fake.calls) == 2
    assert result.reviewOutcome.retried is True
    assert result.reviewOutcome.score == 0.2


def testTheReviewHappensExactlyOnce(monkeypatch, prompts) -> None:
    """The whole point of the ceiling. Reviewing the retry too would have no
    guaranteed exit on a question the documents cannot answer."""
    fake = FakeAgent(["Not sure.", "Still not sure."])
    reviews = install(
        monkeypatch, fake, [Review(score=0.1, suggestion="Be specific."), Review(score=0.1)]
    )

    result = run(prompts)

    assert len(reviews) == 1, "the retried answer must not be graded again"
    assert result.answer == "Still not sure."


def testTheRetryCarriesTheSuggestionAndThePreviousAnswer(monkeypatch, prompts) -> None:
    fake = FakeAgent(["Not sure.", "Thirty days."])
    install(monkeypatch, fake, [Review(score=0.2, suggestion="Quote the policy.")])

    run(prompts)

    secondTurn = str(fake.calls[1])
    assert "Quote the policy." in secondTurn
    assert "Not sure." in secondTurn


def testTheRetryContinuesFromTheTranscript(monkeypatch, prompts) -> None:
    """Whatever was retrieved the first time has to still be in hand.

    Rebuilding the conversation from the question and the answer text would
    drop the tool calls, and the second attempt would pay for the same searches
    again to get back to where the first one already was.
    """
    retrieved = ToolMessage(content="Refunds within 30 days.", tool_call_id="1")
    fake = FakeAgent(["Not sure.", "Thirty days."], toolTurns=[retrieved])
    install(monkeypatch, fake, [Review(score=0.2, suggestion="Be specific.")])

    run(prompts)

    assert retrieved in fake.calls[1], "the retry threw away what was retrieved"
    assert len(fake.calls[1]) > len(fake.calls[0])


def testAnUngradableAnswerIsReturnedAsIs(monkeypatch, prompts) -> None:
    """A broken judge must not swallow a good answer."""
    fake = FakeAgent(["Thirty days from purchase."])
    install(monkeypatch, fake, [None])

    result = run(prompts)

    assert result.answer == "Thirty days from purchase."
    assert len(fake.calls) == 1
    assert result.reviewOutcome.reviewed is False


def testAnEmptyRetryFallsBackToTheFirstAnswer(monkeypatch, prompts) -> None:
    """A second attempt that produced nothing is worse than a mediocre first."""
    fake = FakeAgent(["A thin answer.", ""])
    install(monkeypatch, fake, [Review(score=0.3, suggestion="More detail.")])

    result = run(prompts)

    assert result.answer == "A thin answer."


def testNoAnswerAtAllIsSaidRatherThanReturnedBlank(monkeypatch, prompts) -> None:
    """An empty string reads as 'the documents say nothing about that', which
    is a different claim from 'this went wrong'."""
    fake = FakeAgent(["", ""])
    install(monkeypatch, fake, [Review(score=0.0, suggestion="Answer it.")])

    assert run(prompts).answer == NO_ANSWER


# --- the pieces around the loop -------------------------------------------


def testTheSearchToolIsBoundToOneProject() -> None:
    """The whole authorisation story for retrieval: no prompt reaching the agent
    can talk it into reading another project's documents."""
    store = FakeChunkStore([SearchResult(text="Refunds within 30 days.", index=0, score=0.9)])

    tools = buildTools(store, "acme-db")
    searchProject = next(t for t in tools if t.name == "searchProject")
    asyncio.run(searchProject.ainvoke({"query": "refunds"}))

    assert store.searched == [("acme-db", "refunds")]
    assert "ragDbId" not in (searchProject.args_schema.model_json_schema()["properties"])


def testAProjectWithNoDatabaseGetsNoSearchTool() -> None:
    """A tool that can only ever fail is worse than an absent one -- the model
    keeps trying it."""
    assert [t.name for t in buildTools(FakeChunkStore(), None)] == []


def testTheAgentNeverGetsHostTools() -> None:
    """deepagents ships filesystem and shell tools. Behind an unauthenticated
    endpoint they would let anyone who can reach /query read or write files."""
    assert HOST_TOOLS >= {"read_file", "write_file", "execute", "delete"}
    assert [t.name for t in buildTools(FakeChunkStore(), "acme-db")] == ["searchProject"]


@pytest.mark.parametrize(
    "verdict, expected",
    [(Review(score=0.69), True), (Review(score=0.7), False), (Review(score=1.0), False), (None, False)],
)
def testTheThresholdDecidesTheRetry(verdict, expected) -> None:
    assert needsAnotherAttempt(verdict) is expected


def testTheAnswerIsReadFromContentBlocks() -> None:
    """Providers differ: some return a string, some a list of blocks."""
    blocks = {"messages": [AIMessage(content=[{"type": "text", "text": "Thirty days."}])]}

    assert _lastText(blocks) == "Thirty days."


def testTheRetryInstructionDoesNotLeakIntoTheAnswer() -> None:
    instruction = retryInstruction(QUESTION, "Not sure.", "Be specific.")

    assert "do not mention this feedback" in instruction.lower()


# --- the real harness ------------------------------------------------------
#
# Everything above stubs buildAgent, which leaves the one claim that actually
# matters unproven: that a deep agent built here cannot touch the host. These
# build a real one. No provider is reached -- the model is a recorder that
# reports a provider name and answers with a fixed message -- but
# create_deep_agent, the harness profile, and the tool filtering are the real
# ones, so a deepagents upgrade that renames a tool or changes how profiles
# resolve fails here rather than in production.


class RecordingModel(BaseChatModel):
    """A chat model that records the tools it was bound with."""

    provider: str = "anthropic"
    bound: list = []

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _get_ls_params(self, stop=None, **kwargs):
        # How deepagents decides which harness profile applies to a pre-built
        # model. The provider name, not the class, is what it keys on.
        return {"ls_provider": self.provider, "ls_model_name": "test", "ls_model_type": "chat"}

    def bind_tools(self, tools, **kwargs):
        self.bound.extend(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Thirty days."))])


def toolNames(model: RecordingModel) -> set[str]:
    return {
        getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else str(t))
        for t in model.bound
    }


@pytest.mark.parametrize("provider", ["anthropic", "openai", "groq", "google_genai"])
def testARealAgentIsNeverGivenHostTools(provider: str) -> None:
    """The claim the unauthenticated endpoint rests on.

    Parametrized over the provider names our four models actually report --
    the profile is keyed on that string, and one we failed to register under
    would silently get the full built-in suite, filesystem and shell included.
    """
    model = RecordingModel(provider=provider, bound=[])

    agent = buildAgent("Answer questions.", buildTools(FakeChunkStore(), "acme-db"), model=model)
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))

    assert toolNames(model) & HOST_TOOLS == set(), f"{provider} kept host tools"


def testARealAgentGetsOnlyTheProjectSearchTool() -> None:
    """Not just "no host tools" -- nothing else either.

    The auto-added general-purpose subagent brings a `task` tool that would
    hand the question to a second agent holding this same single tool, paying
    twice to reach the same passages.
    """
    model = RecordingModel(bound=[])

    agent = buildAgent("Answer questions.", buildTools(FakeChunkStore(), "acme-db"), model=model)
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))

    assert toolNames(model) == {"searchProject"}


# --- retrieval failures ----------------------------------------------------


def testASearchFailureIsReportedToTheModelNotRaised() -> None:
    """An exception out of a tool aborts the graph, turning a question the
    agent could still have answered honestly into a 500."""

    class BrokenStore:
        async def search(self, ragDbId, query, topK=5):
            raise ConnectionError("pinecone is unreachable")

    searchProject = buildTools(BrokenStore(), "acme-db")[0]

    answer = asyncio.run(searchProject.ainvoke({"query": "refunds"}))

    assert "unavailable" in answer.lower()
