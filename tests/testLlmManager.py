"""The provider factory. Nothing here reaches a vendor -- the point is that the
right client is constructed, with the right key, and that a missing key fails
with a message naming the variable to set."""

import pytest

from app.agent.llmManager import (
    API_KEY_ENV,
    DEFAULT_MODEL,
    LlmConfigError,
    Provider,
    _cachedChatModel,
    agentModel,
    asProvider,
    chatModel,
    reviewerModel,
)


@pytest.fixture(autouse=True)
def isolatedCache(monkeypatch):
    """The factory caches clients per process; tests must not share them."""
    _cachedChatModel.cache_clear()
    for name in API_KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)
    for name in (
        "RAG_AGENT_PROVIDER",
        "RAG_AGENT_MODEL",
        "RAG_REVIEWER_PROVIDER",
        "RAG_REVIEWER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    _cachedChatModel.cache_clear()


@pytest.mark.parametrize(
    "provider, model, expected",
    [
        (Provider.ANTHROPIC, "claude-opus-5", "ChatAnthropic"),
        (Provider.OPENAI, "gpt-5.1", "ChatOpenAI"),
        (Provider.GROQ, "llama-3.3-70b-versatile", "ChatGroq"),
        (Provider.GEMINI, "gemini-3.5-flash-lite", "ChatGoogleGenerativeAI"),
    ],
)
def testEachProviderBuildsItsOwnClient(monkeypatch, provider, model, expected) -> None:
    """The whole point of the layer: ask for a provider and a model, get that
    vendor's chat model object back."""
    monkeypatch.setenv(API_KEY_ENV[provider], "test-key")

    built = chatModel(provider, model)

    assert type(built).__name__ == expected


def testAProviderCanBeNamedAsAString(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    assert type(chatModel("anthropic", "claude-opus-5")).__name__ == "ChatAnthropic"


def testAnUnknownProviderSaysWhatIsSupported() -> None:
    with pytest.raises(LlmConfigError) as raised:
        asProvider("bedrock")

    for provider in Provider:
        assert provider.value in str(raised.value)


def testAMissingKeyNamesTheVariableToSet() -> None:
    """Better than the vendor's 401 three layers down."""
    with pytest.raises(LlmConfigError) as raised:
        chatModel(Provider.GROQ, "llama-3.3-70b-versatile")

    assert "GROQ_API_KEY" in str(raised.value)


def testABlankModelIsRefused(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    with pytest.raises(LlmConfigError):
        chatModel(Provider.ANTHROPIC, "   ")


def testTheSameProviderAndModelIsOneClient(monkeypatch) -> None:
    """Each client owns a connection pool; one per request would leak sockets
    until the vendor refused them."""
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    assert chatModel(Provider.ANTHROPIC, "claude-opus-5") is chatModel(
        Provider.ANTHROPIC, "claude-opus-5"
    )


def testDifferentModelsAreDifferentClients(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    assert chatModel(Provider.ANTHROPIC, "claude-opus-5") is not chatModel(
        Provider.ANTHROPIC, "claude-sonnet-5"
    )


def testOverridesBypassTheCache(monkeypatch) -> None:
    """Two callers wanting different settings must not share an instance."""
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    plain = chatModel(Provider.ANTHROPIC, "claude-opus-5")
    tuned = chatModel(Provider.ANTHROPIC, "claude-opus-5", temperature=0.9)

    assert plain is not tuned


def testTheAgentDefaultsToAnthropic(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")

    built = agentModel()

    assert type(built).__name__ == "ChatAnthropic"
    assert DEFAULT_MODEL in repr(built.model)


def testTheAgentAndReviewerAreConfiguredSeparately(monkeypatch) -> None:
    """So the judge can be pointed at a cheaper model without moving the agent."""
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")
    monkeypatch.setenv(API_KEY_ENV[Provider.GROQ], "test-key")
    monkeypatch.setenv("RAG_REVIEWER_PROVIDER", "groq")
    monkeypatch.setenv("RAG_REVIEWER_MODEL", "llama-3.3-70b-versatile")

    assert type(agentModel()).__name__ == "ChatAnthropic"
    assert type(reviewerModel()).__name__ == "ChatGroq"


def testANonDefaultProviderMustNameItsModel(monkeypatch) -> None:
    """A model name is not portable between vendors, so naming a provider
    without one is a configuration mistake, not something to guess at."""
    monkeypatch.setenv(API_KEY_ENV[Provider.OPENAI], "test-key")
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "openai")

    with pytest.raises(LlmConfigError) as raised:
        agentModel()

    assert "RAG_AGENT_MODEL" in str(raised.value)
