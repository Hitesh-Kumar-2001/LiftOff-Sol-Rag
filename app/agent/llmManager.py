"""One layer that turns (provider, model) into a chat model object.

Ask it for "OpenAI, this model" and it hands back a configured
``BaseChatModel``. Nothing above this knows which SDK backs a provider, which
environment variable holds its key, or what its constructor is called --
adding a fifth provider is a row in ``_BUILDERS`` and a row in ``API_KEY_ENV``.

``model`` is deliberately required, and this module has no default for it. Every
model name the service uses lives in ``config/models.toml`` and is resolved by
``app.modelConfig``; the four role functions at the bottom of this file are the
only bridge between the two. A default here would mean this file carrying
current model names for four vendors and going stale the first time any of them
ships a new one.

The API key for a provider is still read from the environment rather than that
file, because the file is committed and keys are not.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.modelConfig import ModelConfigError, modelFor

logger = logging.getLogger(__name__)


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"


class LlmConfigError(Exception):
    """A provider was asked for that this process cannot build.

    Also what a ``ModelConfigError`` becomes at this seam -- see ``_roleModel``.
    ``app.api.routes.query`` turns this one type into a 503, and a caller does
    not care whether the model could not be named or could not be built: both
    mean "this deployment cannot answer until somebody fixes its configuration".
    """


API_KEY_ENV: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GROQ: "GROQ_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
}


def _buildAnthropic(model: str, apiKey: str, **overrides: Any) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # max_tokens is required by ChatAnthropic and has a small default; an answer
    # truncated mid-sentence is worse than a slightly more expensive call.
    overrides.setdefault("max_tokens", 8192)
    return ChatAnthropic(model=model, anthropic_api_key=apiKey, **overrides)


def _buildOpenai(model: str, apiKey: str, **overrides: Any) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=apiKey, **overrides)


def _buildGroq(model: str, apiKey: str, **overrides: Any) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, groq_api_key=apiKey, **overrides)


def _buildGemini(model: str, apiKey: str, **overrides: Any) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=apiKey, **overrides)


# Imports live inside the builders on purpose: importing this module must not
# pull four vendor SDKs into memory when a deployment only ever uses one.
_BUILDERS = {
    Provider.ANTHROPIC: _buildAnthropic,
    Provider.OPENAI: _buildOpenai,
    Provider.GROQ: _buildGroq,
    Provider.GEMINI: _buildGemini,
}


def asProvider(name: str | Provider) -> Provider:
    """Parse a provider name, with an error that lists the real options."""
    if isinstance(name, Provider):
        return name
    try:
        return Provider(name.strip().lower())
    except ValueError:
        raise LlmConfigError(
            f"Unknown provider '{name}'. Supported: {', '.join(p.value for p in Provider)}."
        ) from None


def chatModel(
    provider: str | Provider, model: str, **overrides: Any
) -> BaseChatModel:
    """The chat model object for one provider and model.

    ``overrides`` go straight to the underlying constructor (``temperature``,
    ``timeout``, whatever that vendor takes). Passing any bypasses the cache
    below, since two callers asking for different settings must not share an
    instance.
    """
    if overrides:
        return _build(asProvider(provider), model, overrides)
    return _cachedChatModel(asProvider(provider), model)


@lru_cache(maxsize=16)
def _cachedChatModel(provider: Provider, model: str) -> BaseChatModel:
    """One instance per (provider, model) for the life of the process.

    Not the prompt cache -- these are client objects, and each holds an HTTP
    connection pool. Building one per request would open a new pool per request
    and leak sockets until the vendor refused them. Anything that is *data*
    (system prompts) is cached in Redis instead; see ``app.agent.promptStore``.
    """
    return _build(provider, model, {})


def _build(provider: Provider, model: str, overrides: dict[str, Any]) -> BaseChatModel:
    if not model or not model.strip():
        raise LlmConfigError(f"No model named for provider '{provider.value}'.")

    keyName = API_KEY_ENV[provider]
    apiKey = os.environ.get(keyName, "").strip()
    if not apiKey:
        # Failing here rather than at the vendor's 401 means the message names
        # the variable to go and set.
        raise LlmConfigError(
            f"{keyName} is not set, so provider '{provider.value}' cannot be used."
        )

    logger.debug("Building chat model %s:%s.", provider.value, model)
    return _BUILDERS[provider](model, apiKey, **overrides)


def _roleModel(role: str) -> BaseChatModel:
    """The model for one configured role -- see ``config/models.toml``.

    The one place a ``ModelConfigError`` becomes an ``LlmConfigError``. Callers
    above this line catch the latter and nothing else; keeping the translation
    here rather than making them catch both means adding a fifth role, or
    changing where configuration comes from, does not ripple into the routes.
    """
    try:
        choice = modelFor(role)
    except ModelConfigError as exc:
        raise LlmConfigError(str(exc)) from exc
    return chatModel(choice.provider, choice.model)


def agentModel() -> BaseChatModel:
    """The model the agent reasons and answers with. Role ``[agent]``."""
    return _roleModel("agent")


def reviewerModel() -> BaseChatModel:
    """The model that grades an answer. Role ``[reviewer]``.

    Configured apart from the agent so a cheaper model can grade without moving
    the one that answers -- grading is easier than answering. That is a cost
    decision for whoever runs this, made in the config file, not one hardcoded
    here.
    """
    return _roleModel("reviewer")


def summariserModel() -> BaseChatModel:
    """The model that folds a long conversation down. Role ``[summariser]``.

    Configured apart from the other two on purpose. Summarising is the cheapest
    thing this service asks a model to do and the one most often worth pointing
    at a small fast model -- it runs on the critical path of a question that is
    already long, so latency here is latency the user feels.
    """
    return _roleModel("summariser")


def chunkerModel() -> BaseChatModel:
    """The model that picks chunk boundaries during ingestion. Role ``[chunker]``.

    Configured apart from the rest because its cost profile is nothing like
    theirs: the other three are one call per question, this one is a call per
    blank-line section of every document ingested -- hundreds of calls in a few
    minutes on a large corpus. It is the role a per-day request quota stops
    first, and the one where a small fast model pays for itself immediately.
    """
    return _roleModel("chunker")
