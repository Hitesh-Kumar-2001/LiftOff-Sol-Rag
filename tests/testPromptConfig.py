"""Reading config/prompts.toml -- which persona the agent answers as.

Two behaviours are being pinned here and they pull in opposite directions on
purpose: a misconfigured persona must stop the *deployment* (``validatePersona``),
and must never stop a *question* (``defaultSystemPrompt``). The split is why
there are two entry points at all.

The shipped file is checked at the bottom, because the sales persona's
guardrails against inventing prices are a property of that copy and not of this
loader -- an edit that removes them should fail a test rather than a customer.
"""

import pytest

from app.promptConfig import (
    ENV_CONFIG_PATH,
    ENV_PERSONA,
    ENV_PROMPT_OVERRIDE,
    FALLBACK_SYSTEM_PROMPT,
    PromptConfigError,
    _readConfig,
    activePersonaName,
    defaultSystemPrompt,
    describePersona,
    personaNames,
    reviewCriteria,
    validatePersona,
)

TWO_PERSONAS = """
default = "seller"

[seller]
systemPrompt = '''Sell the thing. Never invent a price.'''
reviewCriteria = '''Also weigh whether it moves the sale forward.'''

[plain]
systemPrompt = '''Answer the question.'''
"""


@pytest.fixture(autouse=True)
def noAmbientConfiguration(monkeypatch, tmp_path):
    """No inherited persona, and no real prompts.toml unless a test writes one."""
    _readConfig.cache_clear()
    monkeypatch.delenv(ENV_PERSONA, raising=False)
    monkeypatch.delenv(ENV_PROMPT_OVERRIDE, raising=False)
    monkeypatch.setenv(ENV_CONFIG_PATH, str(tmp_path / "absent.toml"))
    yield
    _readConfig.cache_clear()


@pytest.fixture
def configFile(monkeypatch, tmp_path):
    def write(body: str):
        path = tmp_path / "prompts.toml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv(ENV_CONFIG_PATH, str(path))
        _readConfig.cache_clear()
        return path

    return write


# --- resolution ------------------------------------------------------------


def testTheDefaultPersonaIsUsed(configFile) -> None:
    configFile(TWO_PERSONAS)

    assert activePersonaName() == "seller"
    assert defaultSystemPrompt() == "Sell the thing. Never invent a price."


def testAPersonaCanBeSwitchedByName(configFile, monkeypatch) -> None:
    """One word changes what the service is -- the whole point of naming them."""
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PERSONA, "plain")

    assert defaultSystemPrompt() == "Answer the question."


def testEveryPersonaIsListed(configFile) -> None:
    configFile(TWO_PERSONAS)

    assert sorted(personaNames()) == ["plain", "seller"]


def testReviewCriteriaComeFromTheSamePersona(configFile) -> None:
    """They switch together. Grading one persona's answers against another's
    criteria would mark them down for not doing what they were never told."""
    configFile(TWO_PERSONAS)

    assert "moves the sale forward" in reviewCriteria()


def testAPersonaWithoutCriteriaAddsNothing(configFile, monkeypatch) -> None:
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PERSONA, "plain")

    assert reviewCriteria() == ""


def testAWholePromptCanBeOverriddenFromTheEnvironment(configFile, monkeypatch) -> None:
    """Predates the file and still wins, so nothing that already set it breaks."""
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PROMPT_OVERRIDE, "You are a haiku generator.")

    assert defaultSystemPrompt() == "You are a haiku generator."


def testAnOverriddenPromptIsNotGradedAgainstAPersona(configFile, monkeypatch) -> None:
    """The criteria in the file describe the prompt in the file. Applied to an
    unrelated prompt they penalise an answer for not selling when nothing ever
    asked it to."""
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PROMPT_OVERRIDE, "You are a haiku generator.")

    assert reviewCriteria() == ""


# --- failing loudly, at startup -------------------------------------------


def testAnUnknownPersonaIsRefusedAtStartup(configFile, monkeypatch) -> None:
    """The failure worth catching early: the service would answer perfectly
    well, as somebody other than the assistant that was configured, and nobody
    finds out until they read a transcript."""
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PERSONA, "salesman")

    with pytest.raises(PromptConfigError) as raised:
        validatePersona()

    message = str(raised.value)
    assert "salesman" in message
    assert "seller" in message  # names what IS defined


def testAPersonaWithNoPromptIsRefused(configFile) -> None:
    configFile('default = "empty"\n\n[empty]\nreviewCriteria = "grade it"\n')

    with pytest.raises(PromptConfigError) as raised:
        validatePersona()

    assert "systemPrompt" in str(raised.value)


def testNoDefaultAndNoOverrideIsRefused(configFile) -> None:
    configFile('[seller]\nsystemPrompt = "Sell."\n')

    with pytest.raises(PromptConfigError):
        validatePersona()


# --- never failing, at request time ---------------------------------------


def testAMissingFileStillAnswers() -> None:
    """A prompt lookup must not cost an answer. A worse assistant beats none."""
    assert defaultSystemPrompt() == FALLBACK_SYSTEM_PROMPT


def testAnUnknownPersonaStillAnswers(configFile, monkeypatch) -> None:
    """Same misconfiguration as the startup test above, reached at request time
    because the process was already running when the environment changed."""
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PERSONA, "salesman")

    assert defaultSystemPrompt() == FALLBACK_SYSTEM_PROMPT
    assert reviewCriteria() == ""


def testMalformedTomlStillAnswers(configFile) -> None:
    configFile("default = \n[seller")

    assert defaultSystemPrompt() == FALLBACK_SYSTEM_PROMPT


def testTheFallbackIsTheNeutralAssistant() -> None:
    """Deliberately not the sales persona: if configuration has gone missing,
    the safe thing to be is accurate and dull, not a salesperson improvising
    without its script."""
    assert "retrieval-grounded" in FALLBACK_SYSTEM_PROMPT
    assert "sell" not in FALLBACK_SYSTEM_PROMPT.lower()


def testDescribeNeverRaises(configFile, monkeypatch) -> None:
    configFile(TWO_PERSONAS)
    monkeypatch.setenv(ENV_PERSONA, "salesman")

    assert "NOT CONFIGURED" in describePersona()


# --- the copy that actually ships -----------------------------------------


@pytest.fixture
def shippedConfig(monkeypatch):
    """The repository's own config/prompts.toml, not a tmp_path one."""
    monkeypatch.delenv(ENV_CONFIG_PATH, raising=False)
    _readConfig.cache_clear()
    yield
    _readConfig.cache_clear()


def testTheShippedConfigResolves(shippedConfig) -> None:
    persona = validatePersona()

    assert persona.name == "sales"
    assert persona.systemPrompt
    assert persona.reviewCriteria


def testTheNeutralPersonaIsStillAvailable(shippedConfig) -> None:
    """One word back to a non-selling assistant, for an internal handbook or a
    policy corpus where a sales register would be actively wrong."""
    assert "support" in personaNames()


@pytest.mark.parametrize(
    "guardrail",
    [
        "searchProject",  # it must retrieve before claiming
        "from memory",  # and not answer product questions from memory
        "invented urgency",  # no manufactured scarcity or deadlines
        "payment details",  # it does not take them
        "claim to be human",  # and does not pretend to be a person
    ],
)
def testTheSalesPersonaKeepsItsGuardrails(shippedConfig, guardrail) -> None:
    """These are the lines that keep a sales agent from being a liability. An
    edit to the copy that drops one should fail here rather than in front of a
    customer -- an invented price is the single most expensive thing this file
    can produce."""
    assert guardrail in validatePersona().systemPrompt


def testTheSalesCriteriaPutHonestyAboveSelling(shippedConfig) -> None:
    """The reviewer already has one known failure mode where an honest "the
    documents do not cover this" is marked down and retried. Criteria that
    rewarded selling harder would make that worse, so they say the opposite."""
    criteria = validatePersona().reviewCriteria

    assert "GOOD answer" in criteria
    assert "Never mark an answer down" in criteria
