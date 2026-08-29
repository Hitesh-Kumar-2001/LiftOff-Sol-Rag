"""Reading config/models.toml. No vendor is reached and no client is built --
what this covers is where a (provider, model) pair comes from, which of two
sources wins, and that a half-configured role is refused rather than guessed at.

Building the client from the pair is tests/testLlmManager.py.
"""

import pytest

from app.modelConfig import (
    ENV_CONFIG_PATH,
    ENV_OVERRIDES,
    ROLES,
    ModelConfigError,
    _readConfig,
    configuredModels,
    describeModels,
    modelFor,
)

FULL_CONFIG = """
[agent]
provider = "openai"
model = "gpt-5.6-luna"

[reviewer]
provider = "openai"
model = "gpt-5.4-mini"

[summariser]
provider = "groq"
model = "llama-3.3-70b-versatile"

[chunker]
provider = "anthropic"
model = "claude-haiku-4-5-20251001"
"""


@pytest.fixture(autouse=True)
def noAmbientConfiguration(monkeypatch, tmp_path):
    """Point the loader at a file that does not exist, and clear the overrides.

    Without this every test here would read the repository's real
    config/models.toml and pick up whatever .env happens to set, so editing
    either would break this suite for reasons having nothing to do with it.
    """
    _readConfig.cache_clear()
    for providerVar, modelVar in ENV_OVERRIDES.values():
        monkeypatch.delenv(providerVar, raising=False)
        monkeypatch.delenv(modelVar, raising=False)
    monkeypatch.setenv(ENV_CONFIG_PATH, str(tmp_path / "absent.toml"))
    yield
    _readConfig.cache_clear()


@pytest.fixture
def configFile(monkeypatch, tmp_path):
    """Write a config file and point the loader at it."""

    def write(body: str):
        path = tmp_path / "models.toml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
        _readConfig.cache_clear()
        return path

    return write


def testEveryRoleIsReadFromTheFile(configFile) -> None:
    configFile(FULL_CONFIG)

    assert modelFor("agent").provider == "openai"
    assert modelFor("agent").model == "gpt-5.6-luna"
    assert modelFor("summariser").provider == "groq"
    assert modelFor("chunker").model == "claude-haiku-4-5-20251001"


def testTheFourRolesAreIndependent(configFile) -> None:
    """The whole reason there are four sections rather than one setting: the
    call that answers and the call that picks chunk boundaries have nothing in
    common except that both are models."""
    configFile(FULL_CONFIG)

    assert len({modelFor(role).model for role in ROLES}) == 4


def testTheEnvironmentWinsOverTheFile(configFile, monkeypatch) -> None:
    """What lets a container be pointed at a different model without a rebuild."""
    configFile(FULL_CONFIG)
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "groq")
    monkeypatch.setenv("RAG_AGENT_MODEL", "llama-3.3-70b-versatile")

    choice = modelFor("agent")

    assert (choice.provider, choice.model) == ("groq", "llama-3.3-70b-versatile")
    # And only that role. An override is not a global switch.
    assert modelFor("reviewer").provider == "openai"


def testAnOverrideMustSetBothHalves(configFile, monkeypatch) -> None:
    """Provider from the environment and model from the file would pair a vendor
    with a model name it has never heard of, and the failure would surface as a
    404 from that vendor rather than as the configuration mistake it is."""
    configFile(FULL_CONFIG)
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "anthropic")

    with pytest.raises(ModelConfigError) as raised:
        modelFor("agent")

    assert "RAG_AGENT_MODEL" in str(raised.value)


def testAModelWithoutAProviderIsRefused(configFile, monkeypatch) -> None:
    """Nothing can tell which vendor serves a given model name."""
    configFile(FULL_CONFIG)
    monkeypatch.setenv("RAG_REVIEWER_MODEL", "some-model")

    with pytest.raises(ModelConfigError) as raised:
        modelFor("reviewer")

    assert "RAG_REVIEWER_PROVIDER" in str(raised.value)


def testAHalfWrittenRoleIsRefused(configFile) -> None:
    """The same rule inside the file. A section naming only a provider is a
    half-finished edit, not a request to guess the model."""
    configFile('[agent]\nprovider = "openai"\n')

    with pytest.raises(ModelConfigError) as raised:
        modelFor("agent")

    assert "a model" in str(raised.value)


def testAMissingRoleNamesTheFileAndTheRole(configFile) -> None:
    configFile('[agent]\nprovider = "openai"\nmodel = "gpt-5.6-luna"\n')

    with pytest.raises(ModelConfigError) as raised:
        modelFor("chunker")

    message = str(raised.value)
    assert "chunker" in message
    assert "models.toml" in message
    assert "RAG_CHUNKER_PROVIDER" in message


def testTheEnvironmentAloneIsEnough(monkeypatch) -> None:
    """A deployment that sets all four pairs never needs the file, so an absent
    file is not an error on its own -- only a role nothing names is."""
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("RAG_AGENT_MODEL", "gpt-5.6-luna")

    assert modelFor("agent").model == "gpt-5.6-luna"


def testMalformedTomlIsRaisedRatherThanIgnored(configFile) -> None:
    """Unlike an absent file, broken TOML means somebody meant to configure
    something and it did not land. Falling back to the environment there would
    run the service on a configuration nobody chose."""
    configFile("[agent\nprovider =")

    with pytest.raises(ModelConfigError) as raised:
        modelFor("agent")

    assert "not valid TOML" in str(raised.value)


def testAnUnknownRoleIsRefused() -> None:
    with pytest.raises(ModelConfigError) as raised:
        modelFor("embedder")

    for role in ROLES:
        assert role in str(raised.value)


def testAMisspeltSectionIsWarnedAbout(configFile, caplog) -> None:
    """[summarizer] with a z reads as correct to most people, and the symptom
    without this warning is an error about summariser being unconfigured while
    the file visibly configures it."""
    configFile(FULL_CONFIG.replace("[summariser]", "[summarizer]"))

    with caplog.at_level("WARNING"):
        _readConfig.cache_clear()
        with pytest.raises(ModelConfigError):
            modelFor("summariser")

    assert any("summarizer" in record.getMessage() for record in caplog.records)


def testConfiguredModelsReportsEveryProblemAtOnce(configFile) -> None:
    """Startup calls this. A deployment with two roles missing should learn both
    on the first boot rather than one per restart."""
    configFile('[agent]\nprovider = "openai"\nmodel = "gpt-5.6-luna"\n')

    with pytest.raises(ModelConfigError) as raised:
        configuredModels()

    message = str(raised.value)
    assert "reviewer" in message
    assert "summariser" in message
    assert "chunker" in message


def testConfiguredModelsReturnsAllFourWhenComplete(configFile) -> None:
    configFile(FULL_CONFIG)

    assert [choice.role for choice in configuredModels()] == list(ROLES)


def testDescribeSaysWhereEachChoiceCameFrom(configFile, monkeypatch) -> None:
    """Knowing the agent is on gpt-5.6-luna is half the answer; knowing whether
    that came from the committed file or from something set in the container is
    the half nobody can reconstruct afterwards."""
    configFile(FULL_CONFIG)
    monkeypatch.setenv("RAG_AGENT_PROVIDER", "groq")
    monkeypatch.setenv("RAG_AGENT_MODEL", "llama-3.3-70b-versatile")

    described = describeModels()

    assert "RAG_AGENT_PROVIDER" in described
    assert "models.toml" in described


def testDescribeNeverRaises() -> None:
    """It is a log line. A log line that throws during startup replaces the
    problem it was describing with itself."""
    assert "NOT CONFIGURED" in describeModels()
