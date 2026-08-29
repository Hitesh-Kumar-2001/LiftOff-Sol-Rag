"""The provider factory. Nothing here reaches a vendor -- the point is that the
right client is constructed, with the right key, and that a missing key fails
with a message naming the variable to set."""

import pytest

from app.agent.llmManager import (
    API_KEY_ENV,
    LlmConfigError,
    Provider,
    _cachedChatModel,
    agentModel,
    asProvider,
    chatModel,
    chunkerModel,
    reviewerModel,
)
from app.modelConfig import ENV_CONFIG_PATH, ENV_OVERRIDES, _readConfig


@pytest.fixture(autouse=True)
def isolatedCache(monkeypatch, tmp_path):
    """No shared clients, no ambient configuration, no real config file.

    The factory caches clients per process, so tests must not share them. The
    file is redirected at a tmp_path that does not exist, so a test says what
    the configuration is or gets a clean "nothing is configured" -- otherwise
    every assertion here would depend on whatever config/models.toml currently
    says, and editing that file would break this suite.
    """
    _cachedChatModel.cache_clear()
    _readConfig.cache_clear()
    for name in API_KEY_ENV.values():
        monkeypatch.delenv(name, raising=False)
    for providerVar, modelVar in ENV_OVERRIDES.values():
        monkeypatch.delenv(providerVar, raising=False)
        monkeypatch.delenv(modelVar, raising=False)
    monkeypatch.setenv(ENV_CONFIG_PATH, str(tmp_path / "absent.toml"))
    yield
    _cachedChatModel.cache_clear()
    _readConfig.cache_clear()


def configure(monkeypatch, tmp_path, **roles: tuple[str, str]) -> None:
    """Write a models.toml naming the given roles, and point the loader at it."""
    body = "\n".join(
        f'[{role}]\nprovider = "{provider}"\nmodel = "{model}"'
        for role, (provider, model) in roles.items()
    )
    path = tmp_path / "models.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
    _readConfig.cache_clear()


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


def testEachRoleComesFromTheConfigFile(monkeypatch, tmp_path) -> None:
    """No model name is hardcoded any more -- config/models.toml is the source."""
    monkeypatch.setenv(API_KEY_ENV[Provider.OPENAI], "test-key")
    configure(
        monkeypatch,
        tmp_path,
        agent=("openai", "gpt-5.6-luna"),
        chunker=("openai", "gpt-5.4-mini"),
    )

    assert agentModel().model_name == "gpt-5.6-luna"
    assert chunkerModel().model_name == "gpt-5.4-mini"


def testEveryRoleIsConfiguredSeparately(monkeypatch, tmp_path) -> None:
    """So the judge, the summariser and the chunker can each be pointed at a
    cheaper model without moving the one that answers."""
    monkeypatch.setenv(API_KEY_ENV[Provider.ANTHROPIC], "test-key")
    monkeypatch.setenv(API_KEY_ENV[Provider.GROQ], "test-key")
    configure(
        monkeypatch,
        tmp_path,
        agent=("anthropic", "claude-opus-5"),
        reviewer=("groq", "llama-3.3-70b-versatile"),
    )

    assert type(agentModel()).__name__ == "ChatAnthropic"
    assert type(reviewerModel()).__name__ == "ChatGroq"


def testTheEnvironmentOverridesTheFile(monkeypatch, tmp_path) -> None:
    """What lets one container be pointed elsewhere without a rebuild."""
    monkeypatch.setenv(API_KEY_ENV[Provider.OPENAI], "test-key")
    configure(monkeypatch, tmp_path, agent=("openai", "gpt-5.6-luna"))
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("RAG_AGENT_MODEL", "gpt-5.4-mini")

    assert agentModel().model_name == "gpt-5.4-mini"


def testAnOverriddenProviderMustNameItsModel(monkeypatch, tmp_path) -> None:
    """A model name is not portable between vendors, so half-overriding a role
    -- provider from the environment, model from the file -- is refused rather
    than silently pairing the two."""
    monkeypatch.setenv(API_KEY_ENV[Provider.OPENAI], "test-key")
    configure(monkeypatch, tmp_path, agent=("anthropic", "claude-opus-5"))
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "openai")

    with pytest.raises(LlmConfigError) as raised:
        agentModel()

    assert "RAG_AGENT_MODEL" in str(raised.value)


def testAnUnconfiguredRoleNamesTheFileAndTheRole(monkeypatch) -> None:
    """A 503 that says where to go is worth more than one that says what broke."""
    monkeypatch.setenv(API_KEY_ENV[Provider.OPENAI], "test-key")

    with pytest.raises(LlmConfigError) as raised:
        chunkerModel()

    message = str(raised.value)
    assert "chunker" in message
    # The path it actually looked at, not a generic "config file": with
    # RAG_MODEL_CONFIG in play the two can differ, and the one worth printing
    # is the one that was read.
    assert ".toml" in message
    assert "RAG_CHUNKER_PROVIDER" in message
