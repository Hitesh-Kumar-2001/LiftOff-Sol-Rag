"""Which model runs each of this service's four AI jobs.

There are four places this service calls a language model, and they are
genuinely different decisions rather than one setting used four times:

======================  ===========================================
``agent``               answers the question
``reviewer``            grades the answer
``summariser``          folds a long conversation down
``chunker``             picks chunk boundaries during ingestion
======================  ===========================================

They used to be configured in three different styles -- two environment
variable pairs read through ``llmManager``, and a fourth model name hardcoded in
the ingestion pipeline against a vendor SDK that only that one call used. Which
meant there was no single place to answer "what is this service calling, and
how much of it", and moving off a provider was a code change in one of the four.

``config/models.toml`` is now that single place, and this module reads it.

Precedence
----------
Environment first, file second, and an error if neither names both halves.
Environment winning is what lets one container be pointed at a different model
without rebuilding the image; the file is what makes the ordinary case a
committed, reviewable, greppable fact rather than a variable somebody set once.
The two do not fight in practice because ``.env`` deliberately no longer sets
any of them.

**Provider and model move together, always.** Naming one without the other --
in the file or in the environment -- is refused rather than filled in. A model
name is not portable between vendors, so a half-override silently pairing
``anthropic`` with ``gpt-5.6-luna`` produces a 404 from the vendor several
layers below the actual mistake, and the message it comes back with names
neither the role nor the file.

**No defaults, on purpose.** A default model here would be this file carrying a
list of current model names for four vendors, going quietly stale the first
time any of them ships a new one -- which is exactly how ``gemini-2.0-flash``
came to be pinned in the chunker until Google retired it out from under us. A
missing role is an error naming the file and the role, which is a thirty-second
fix; a stale default is an outage that looks like a vendor problem.

The API keys are not here. They stay in the environment (see
``llmManager.API_KEY_ENV``), because this file is committed and keys are not.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# app/modelConfig.py -> app/ -> the tree root. That root is the repository in a
# checkout and /app inside the image, and config/ sits directly under it in
# both, so nothing here depends on the working directory. It cannot: uvicorn,
# `python -m app.jobs.worker` and pytest are all started from somewhere
# different, and a relative path would work for whichever one was tried first.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.toml"

ENV_CONFIG_PATH = "RAG_MODEL_CONFIG"

# The environment pair that overrides each role. Both halves or neither.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "agent": ("RAG_AGENT_PROVIDER", "RAG_AGENT_MODEL"),
    "reviewer": ("RAG_REVIEWER_PROVIDER", "RAG_REVIEWER_MODEL"),
    "summariser": ("RAG_SUMMARISER_PROVIDER", "RAG_SUMMARISER_MODEL"),
    "chunker": ("RAG_CHUNKER_PROVIDER", "RAG_CHUNKER_MODEL"),
}

ROLES = tuple(ENV_OVERRIDES)


class ModelConfigError(Exception):
    """No usable (provider, model) could be found for a role.

    Translated to ``LlmConfigError`` at the ``llmManager`` seam, which is what
    ``routes.query`` turns into a 503 -- "this deployment is misconfigured, fix
    it and retry", rather than a 500 that reads as a bug in the request.
    """


@dataclass(frozen=True)
class ModelChoice:
    """One role's provider and model, from wherever they were found.

    ``source`` is carried for the log line at startup. Knowing that the agent
    is on gpt-5.6-luna is half the answer; knowing whether that came from the
    committed file or from something set in the container is the other half,
    and it is the half nobody can reconstruct after the fact.
    """

    role: str
    provider: str
    model: str
    source: str


def _configPath() -> Path:
    """Where the file is. Read per call, not at import.

    An import-time ``os.environ`` read freezes whatever was set at the moment
    this module first happened to be imported, which in a test session is
    whatever the first test to touch it wanted. This codebase has been bitten by
    exactly that before -- see the note on ``_useFirestore`` in
    ``app.jobs.jobManager``.
    """
    override = (os.environ.get(ENV_CONFIG_PATH) or "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def _readConfig(path: str) -> dict[str, dict[str, str]]:
    """The parsed file, once per process. Keyed by path so a test can point
    elsewhere without poisoning the entry the application will use.

    A missing file is not an error here, only an empty result: a deployment
    that sets all four environment pairs never needs the file at all, and the
    error worth raising is "nothing names a model for the agent", not "a file
    you were not using is absent". ``modelFor`` names this path in that error,
    so a mistyped ``RAG_MODEL_CONFIG`` still says where it looked.

    Malformed TOML *is* raised. Unlike an absent file, it means somebody
    intended to configure something and the intent did not land -- falling back
    to the environment there would run the service on a configuration nobody
    chose.
    """
    file = Path(path)
    try:
        raw = file.read_bytes()
    except FileNotFoundError:
        logger.warning(
            "No model configuration at %s; every role must come from the environment.",
            file,
        )
        return {}
    except OSError as exc:
        raise ModelConfigError(f"Could not read the model configuration at {file}: {exc}") from exc

    try:
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ModelConfigError(f"{file} is not valid TOML: {exc}") from exc

    # A section nobody reads is almost always a typo in the section name, and
    # the symptom without this is the *other* kind of error -- "no model
    # configured for summariser" while the file plainly contains a
    # [summarizer] block that reads as correct to anyone who spells it with a
    # z. Warned rather than raised: an unknown key is inert, and refusing to
    # start over one would make adding a role a breaking change for anyone
    # running an older image against a newer file.
    for section in parsed:
        if section not in ENV_OVERRIDES:
            logger.warning(
                "%s has a [%s] section, which is not one of the roles this service "
                "uses (%s). It is being ignored -- check the spelling.",
                file,
                section,
                ", ".join(ROLES),
            )

    return {
        role: values for role, values in parsed.items() if isinstance(values, dict)
    }


def _fromEnvironment(role: str) -> tuple[str, str] | None:
    """This role's environment override, or None if it is not being overridden.

    Refuses a half-set pair. Setting only ``RAG_AGENT_PROVIDER`` and letting the
    model come from the file is the specific mistake worth blocking: it pairs a
    provider from one place with a model name from another, and the result is a
    model name sent to a vendor that has never heard of it.
    """
    providerVar, modelVar = ENV_OVERRIDES[role]
    provider = (os.environ.get(providerVar) or "").strip()
    model = (os.environ.get(modelVar) or "").strip()

    if provider and not model:
        raise ModelConfigError(
            f"{providerVar} is set to '{provider}' but {modelVar} is not. An override "
            f"must name both -- a model name is not portable between providers, so "
            f"taking the model from {_configPath()} would pair '{provider}' with a "
            f"model it does not serve."
        )
    if model and not provider:
        raise ModelConfigError(
            f"{modelVar} is set to '{model}' but {providerVar} is not. An override must "
            f"name both, because nothing can tell which vendor serves '{model}'."
        )
    if not provider:
        return None
    return provider, model


def modelFor(role: str) -> ModelChoice:
    """The provider and model for one role: environment, then file, then error."""
    if role not in ENV_OVERRIDES:
        raise ModelConfigError(
            f"'{role}' is not a role this service configures. Known roles: "
            f"{', '.join(ROLES)}."
        )

    override = _fromEnvironment(role)
    if override is not None:
        providerVar, modelVar = ENV_OVERRIDES[role]
        return ModelChoice(role, override[0], override[1], source=f"${providerVar}/${modelVar}")

    path = _configPath()
    section = _readConfig(str(path)).get(role) or {}
    provider = str(section.get("provider") or "").strip()
    model = str(section.get("model") or "").strip()

    if not provider or not model:
        providerVar, modelVar = ENV_OVERRIDES[role]
        missing = "provider and model" if not provider and not model else (
            "a provider" if not provider else "a model"
        )
        raise ModelConfigError(
            f"No model is configured for '{role}'. Add {missing} under [{role}] in "
            f"{path}, or set {providerVar} and {modelVar} together."
        )

    return ModelChoice(role, provider, model, source=str(path))


def configuredModels() -> list[ModelChoice]:
    """Every role's choice, for the startup log and for ``checkConfiguration``.

    Deliberately resolves all four rather than stopping at the first failure:
    a deployment with two roles misconfigured should learn both on the first
    boot, not one per restart.
    """
    resolved: list[ModelChoice] = []
    problems: list[str] = []
    for role in ROLES:
        try:
            resolved.append(modelFor(role))
        except ModelConfigError as exc:
            problems.append(str(exc))

    if problems:
        raise ModelConfigError(" ".join(problems))
    return resolved


def describeModels() -> str:
    """One line per role, for the log. Never raises -- it is a log line."""
    lines = []
    for role in ROLES:
        try:
            choice = modelFor(role)
            lines.append(f"  {role}: {choice.provider}/{choice.model}  (from {choice.source})")
        except ModelConfigError as exc:
            lines.append(f"  {role}: NOT CONFIGURED -- {exc}")
    return "\n".join(lines)
