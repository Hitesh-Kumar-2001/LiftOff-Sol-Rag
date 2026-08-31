"""Token accounting: what an answer cost, and where it is filed.

Two halves. ``app.agent.usage`` collects the numbers while three separately
billed models run, and ``app.stores.usageStore`` writes them as
project -> conversation -> message with atomic rollups.

The collection tests use a fake model rather than a real one, because what is
being checked is the plumbing -- that a callback per role keeps the roles apart,
that a failed call still reports what it burned, and that two concurrent answers
do not bill each other. The store tests use real Firestore, like the rest of the
suite, because the thing worth testing there is `Increment` behaving under
concurrent writers, which a stand-in would agree with by construction.
"""

import asyncio
from collections.abc import Iterator

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.usage import ANSWER_ROLES, RoleUsage, UsageLog, trackUsage
from app.stores.usageStore import COLLECTION, FirestoreUsageStore, _sum


class CountingModel(BaseChatModel):
    """A model that reports a fixed token count and never calls anybody."""

    tokens: int = 100
    modelName: str = "fake-model"
    explode: bool = False

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.explode:
            raise RuntimeError("the provider fell over")
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": self.tokens,
                "output_tokens": self.tokens // 2,
                "total_tokens": self.tokens + self.tokens // 2,
            },
            response_metadata={"model_name": self.modelName},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def metadata(total: int = 30, model: str = "gpt-5.6-luna") -> dict:
    return {
        model: {
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": total,
            "input_token_details": {"cache_read": 5},
            "output_token_details": {"reasoning": 4},
        }
    }


# --- the log ---------------------------------------------------------------


def testOneRoleIsRecordedWithItsProviderAndModel() -> None:
    usage = UsageLog()

    usage.record("agent", "openai", "gpt-5.6-luna", metadata())

    assert usage.totalTokens == 30
    assert usage.byRole()["agent"]["provider"] == "openai"
    assert usage.byRole()["agent"]["model"] == "gpt-5.6-luna"


def testCachedAndReasoningTokensAreKept() -> None:
    """They are priced differently from the totals containing them, so a total
    alone cannot be turned back into money."""
    usage = UsageLog()

    usage.record("agent", "openai", "gpt-5.6-luna", metadata())

    assert usage.byRole()["agent"]["cachedInputTokens"] == 5
    assert usage.byRole()["agent"]["reasoningTokens"] == 4


def testRolesStaySeparateEvenOnOneModel() -> None:
    """The whole reason for a callback per role. The agent, reviewer and
    summariser are usually the same model, and a shared context would merge all
    three into one number."""
    usage = UsageLog()

    usage.record("agent", "openai", "m", metadata(total=100))
    usage.record("reviewer", "openai", "m", metadata(total=10))
    usage.record("summariser", "openai", "m", metadata(total=5))

    byRole = usage.byRole()
    assert byRole["agent"]["totalTokens"] == 100
    assert byRole["reviewer"]["totalTokens"] == 10
    assert byRole["summariser"]["totalTokens"] == 5
    assert usage.totalTokens == 115


def testARetriedAnswerCollapsesIntoOneRowWithTwoCalls() -> None:
    usage = UsageLog()

    usage.record("agent", "openai", "m", metadata(total=100))
    usage.record("agent", "openai", "m", metadata(total=80))

    assert usage.byRole()["agent"] == {
        "provider": "openai",
        # The model the vendor reported, not the alias it was asked for.
        "model": "gpt-5.6-luna",
        "inputTokens": 40,
        "outputTokens": 20,
        "totalTokens": 180,
        "cachedInputTokens": 10,
        "reasoningTokens": 8,
        "calls": 2,
    }


def testTheModelTheVendorBilledWins() -> None:
    """A configured name can be an alias for a dated build. What actually ran is
    what a bill has to be reconciled against."""
    usage = UsageLog()

    usage.record("agent", "openai", "gpt-5.6-luna", metadata(model="gpt-5.6-luna-2026-01-01"))

    assert usage.byRole()["agent"]["model"] == "gpt-5.6-luna-2026-01-01"


def testAnEmptyLogSummarisesWithoutFailing() -> None:
    assert "no usage" in UsageLog().summary()


def testTheChunkerIsNotAnAnswerRole() -> None:
    """It runs during ingestion, against a document rather than a conversation,
    so its cost belongs to a job and not to anybody's message."""
    assert "chunker" not in ANSWER_ROLES
    assert set(ANSWER_ROLES) == {"agent", "reviewer", "summariser"}


# --- the tracker -----------------------------------------------------------


def testTrackingIsOffWhenNobodyIsCounting() -> None:
    """None disables it entirely, which is what every caller that does not care
    about cost gets -- and every test that stubs a model out."""
    with trackUsage(None, "agent"):
        pass  # No handler registered, nothing raised.


def testACallInsideTheBlockIsCounted() -> None:
    usage = UsageLog()

    with trackUsage(usage, "agent"):
        CountingModel(tokens=100).invoke("hello")

    assert usage.byRole()["agent"]["totalTokens"] == 150


def testAFailedCallStillReportsWhatItBurned() -> None:
    """A provider that raises after producing tokens was still billed for them,
    and an answer that failed expensively is the one worth seeing."""
    usage = UsageLog()

    with pytest.raises(RuntimeError):
        with trackUsage(usage, "agent"):
            CountingModel(tokens=40).invoke("counted")
            CountingModel(explode=True).invoke("never returns")

    assert usage.byRole()["agent"]["totalTokens"] == 60


def testConcurrentAnswersDoNotBillEachOther() -> None:
    """The failure this guards against is silent and unexplainable after the
    fact: one project's total quietly carrying another's traffic."""
    first, second = UsageLog(), UsageLog()

    async def answer(usage: UsageLog, tokens: int) -> None:
        with trackUsage(usage, "agent"):
            await asyncio.sleep(0)  # force interleaving
            await CountingModel(tokens=tokens).ainvoke("go")
            await asyncio.sleep(0)

    async def both() -> None:
        await asyncio.gather(answer(first, 100), answer(second, 10))

    asyncio.run(both())

    assert first.byRole()["agent"]["totalTokens"] == 150
    assert second.byRole()["agent"]["totalTokens"] == 15


# --- the store -------------------------------------------------------------


def testTheRollupSumsOnlyTheCounters() -> None:
    byRole = {
        "agent": {"totalTokens": 100, "inputTokens": 60, "outputTokens": 40, "model": "m"},
        "reviewer": {"totalTokens": 10, "inputTokens": 8, "outputTokens": 2, "model": "m"},
    }

    assert _sum(byRole)["totalTokens"] == 110
    assert _sum(byRole)["inputTokens"] == 68
    assert "model" not in _sum(byRole)


@pytest.fixture
def usageStore(scratch) -> FirestoreUsageStore:
    return FirestoreUsageStore()


@pytest.fixture
def projectId(scratch) -> Iterator[str]:
    yield scratch.projectId("usage")


def role(total: int, calls: int = 1) -> dict:
    return {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "inputTokens": total // 2,
        "outputTokens": total // 2,
        "totalTokens": total,
        "cachedInputTokens": 0,
        "reasoningTokens": 0,
        "calls": calls,
    }


def testATurnIsFiledUnderItsProjectConversationAndMessage(usageStore, projectId) -> None:
    asyncio.run(
        usageStore.recordTurn(
            projectId=projectId,
            conversationId="conv-1",
            messageId="000001",
            byRole={"agent": role(100), "reviewer": role(10)},
        )
    )

    conversation = asyncio.run(usageStore.conversationTotal(projectId, "conv-1"))
    project = asyncio.run(usageStore.projectTotal(projectId))

    assert conversation["totalTokens"] == 110
    assert conversation["turns"] == 1
    assert conversation["roles"]["agent"]["totalTokens"] == 100
    assert conversation["roles"]["reviewer"]["model"] == "gpt-5.6-luna"
    assert project["totalTokens"] == 110


def testTotalsAccumulateAcrossTurns(usageStore, projectId) -> None:
    """`Increment`, not read-modify-write: two answers in one conversation at
    the same moment would otherwise both read the old total and one would
    vanish."""
    for index in range(3):
        asyncio.run(
            usageStore.recordTurn(
                projectId=projectId,
                conversationId="conv-1",
                messageId=f"{index:06d}",
                byRole={"agent": role(100)},
            )
        )

    conversation = asyncio.run(usageStore.conversationTotal(projectId, "conv-1"))
    assert conversation["totalTokens"] == 300
    assert conversation["turns"] == 3
    assert conversation["roles"]["agent"]["calls"] == 3


def testTwoConversationsRollUpIntoOneProject(usageStore, projectId) -> None:
    for conversation in ("conv-1", "conv-2"):
        asyncio.run(
            usageStore.recordTurn(
                projectId=projectId,
                conversationId=conversation,
                messageId="000001",
                byRole={"agent": role(50)},
            )
        )

    assert asyncio.run(usageStore.projectTotal(projectId))["totalTokens"] == 100
    assert asyncio.run(usageStore.conversationTotal(projectId, "conv-1"))["totalTokens"] == 50


def testTheChannelAndPlatformMessageIdAreKept(usageStore, projectId) -> None:
    """So a WhatsApp bill can be traced back to the message the customer sent."""
    asyncio.run(
        usageStore.recordTurn(
            projectId=projectId,
            conversationId="conv-1",
            messageId="000001",
            byRole={"agent": role(10)},
            channel="whatsapp",
            externalMessageId="wamid.abc",
        )
    )

    from app.infra.firestoreClient import firestoreClient

    stored = (
        firestoreClient()
        .collection(COLLECTION)
        .document(projectId)
        .collection("conversations")
        .document("conv-1")
        .collection("messages")
        .document("000001")
        .get()
        .to_dict()
    )
    assert stored["channel"] == "whatsapp"
    assert stored["externalMessageId"] == "wamid.abc"


def testNothingIsWrittenWhenNothingWasSpent(usageStore, projectId) -> None:
    """Every model stubbed, or none reported usage. An empty row would make
    "which answers cost nothing" a real question with a wrong answer."""
    asyncio.run(
        usageStore.recordTurn(
            projectId=projectId, conversationId="conv-1", messageId="000001", byRole={}
        )
    )

    assert asyncio.run(usageStore.projectTotal(projectId)) is None


def testAFailedWriteDoesNotRaise(usageStore) -> None:
    """The model call is already paid for by the time this runs. An accounting
    write that does not land costs a number in a report; raising would cost the
    caller an answer they are owed."""
    asyncio.run(
        usageStore.recordTurn(
            projectId="..",  # Firestore refuses this as a document id.
            conversationId="conv-1",
            messageId="000001",
            byRole={"agent": role(10)},
        )
    )
