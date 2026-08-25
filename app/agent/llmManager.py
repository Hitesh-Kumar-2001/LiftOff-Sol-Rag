"""One layer that turns (provider, model) into a chat model object.

Ask it for "Anthropic, this model" and it hands back a configured
``BaseChatModel``. Nothing above this knows which SDK backs a provider, which
environment variable holds its key, or what its constructor is called --
adding a fifth provider is a row in ``_BUILDERS`` and a row in ``API_KEY_ENV``.

``model`` is deliberately required. A default per provider would mean this file
carrying a list of current model names for four vendors and going quietly stale
the moment any of them ships a new one; the caller knows what it wants. The two
places the *application* needs a default -- the agent and the reviewer -- read
theirs from the environment, below.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    GEMINI = "gemini"


class LlmConfigError(Exception):
    """A provider was asked for that this process cannot build."""


API_KEY_ENV: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GROQ: "GROQ_API_KEY",
    # Shared with the chunking pipeline, which already reads this key.
    Provider.GEMINI: "GEMINI_API_KEY",
}

# What the application itself uses when nothing names a model. Anthropic and
# `claude-opus-5` because that is the model this codebase is developed against;
# both halves are environment-overridable, so pointing the agent at Groq for
# cost or Gemini for latency is configuration, not a code change.
DEFAULT_PROVIDER = Provider.ANTHROPIC
DEFAULT_MODEL = "claude-opus-5"


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


def _fromEnvironment(providerVar: str, modelVar: str) -> tuple[Provider, str]:
    provider = asProvider(os.environ.get(providerVar) or DEFAULT_PROVIDER)
    model = (os.environ.get(modelVar) or "").strip()
    if not model:
        # Only meaningful alongside the default provider -- a model name is not
        # portable between vendors, so naming one without the other is a
        # configuration mistake worth refusing.
        if provider is not DEFAULT_PROVIDER:
            raise LlmConfigError(
                f"{providerVar} is '{provider.value}', so {modelVar} must name one of "
                f"its models -- there is no default model outside {DEFAULT_PROVIDER.value}."
            )
        model = DEFAULT_MODEL
    return provider, model


def agentModel() -> BaseChatModel:
    """The model the agent reasons and answers with."""
    provider, model = _fromEnvironment("RAG_AGENT_PROVIDER", "RAG_AGENT_MODEL")
    return chatModel(provider, model)


def reviewerModel() -> BaseChatModel:
    """The model that grades an answer.

    Same default as the agent. A cheaper model is a reasonable thing to point
    this at -- grading is easier than answering -- but that is a cost decision
    for whoever runs this, made with RAG_REVIEWER_PROVIDER/MODEL, not one
    hardcoded here.
    """
    provider, model = _fromEnvironment("RAG_REVIEWER_PROVIDER", "RAG_REVIEWER_MODEL")
    return chatModel(provider, model)


def summariserModel() -> BaseChatModel:
    """The model that folds a long conversation down to a brief.

    Configured apart from the other two on purpose. Summarising is the cheapest
    thing this service asks a model to do and the one most often worth pointing
    at a small fast model -- it runs on the critical path of a question that is
    already long, so latency here is latency the user feels.
    """
    provider, model = _fromEnvironment("RAG_SUMMARISER_PROVIDER", "RAG_SUMMARISER_MODEL")
    return chatModel(provider, model)
