"""Who the agent is, read from one file.

``config/prompts.toml`` holds named personas. Each is a ``systemPrompt`` -- what
the agent is told it is and how to behave -- and an optional ``reviewCriteria``,
which the reviewer weighs on top of its usual accuracy checks. One is marked
``default`` and that is the persona a project answers with when it has no prompt
of its own.

**This is copy, not code.** The sales pitch is the thing whoever owns the pitch
will want to tune, repeatedly, without opening a Python file or shipping a
release -- and it is the thing most likely to be edited by somebody who is not a
programmer. A prompt living as a string literal in ``promptStore`` made every
change to it a code change; that was the wrong home for it.

Where a prompt comes from, highest precedence first
---------------------------------------------------
1. the project's own prompt in Firestore, if one is assigned -- ``promptStore``
   resolves that and never reaches this module;
2. ``RAG_DEFAULT_SYSTEM_PROMPT``, a whole prompt as an environment variable,
   which predates this file and still wins so nothing that used it breaks;
3. ``RAG_PERSONA`` naming a persona in the file;
4. the file's own ``default``;
5. ``FALLBACK_SYSTEM_PROMPT`` below, if the file is missing or unreadable.

Failing loudly, and not failing at all
--------------------------------------
Both, in different places, and the split matters.

``validatePersona`` raises, and ``app.main.checkConfiguration`` calls it, so a
``RAG_PERSONA`` naming a persona that does not exist stops the deployment
instead of quietly serving something nobody chose.

``defaultSystemPrompt`` and ``reviewCriteria`` never raise. They are on the path
of a live question, and a prompt lookup must not cost an answer -- the same rule
``promptStore`` already follows for an unreachable Firestore. A broken file after
startup degrades to the fallback and logs; it does not 500 the question.

Unlike ``app.modelConfig``, there *is* a built-in fallback here. A default model
name goes stale the moment a vendor retires it, so there must not be one; a
default prompt is prose with no vendor behind it and works forever. The right
answer to "the config file is missing" is a worse assistant, not no assistant.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# app/promptConfig.py -> app/ -> the tree root: the repository in a checkout and
# /app inside the image. Same reasoning as app.modelConfig -- uvicorn, the worker
# and pytest all start from different directories, so a relative path would work
# for whichever was tried first.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "prompts.toml"

ENV_CONFIG_PATH = "RAG_PROMPT_CONFIG"
ENV_PERSONA = "RAG_PERSONA"
# Predates the file. Still the highest precedence below a per-project prompt,
# because a deployment already setting it is expecting it to be obeyed.
ENV_PROMPT_OVERRIDE = "RAG_DEFAULT_SYSTEM_PROMPT"

# Used only when the file cannot be read at all. Deliberately the neutral,
# retrieval-grounded assistant rather than the sales persona: if configuration
# has gone missing, the safe thing for the service to be is accurate and dull,
# not a salesperson improvising without its script.
FALLBACK_SYSTEM_PROMPT = (
    "You are a retrieval-grounded assistant for one project's documents.\n"
    "Answer from what the project's own documents say. Use the searchProject "
    "tool to find them before answering anything factual -- do not answer from "
    "memory when the answer should come from the documents.\n"
    "If the documents do not contain the answer, say so plainly rather than "
    "guessing, and say what they do cover. Keep answers direct and specific."
)


class PromptConfigError(Exception):
    """The persona configuration names something that does not exist."""


@dataclass(frozen=True)
class Persona:
    """One resolved persona, and where it came from.

    ``source`` is for the startup log. Knowing the service is selling is half
    the answer; knowing whether that came from the committed file or from
    something set on one container is the half nobody can reconstruct later.
    """

    name: str
    systemPrompt: str
    reviewCriteria: str
    source: str


def _configPath() -> Path:
    """Read per call, not at import -- see the note in ``app.modelConfig``."""
    override = (os.environ.get(ENV_CONFIG_PATH) or "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def _readConfig(path: str) -> dict:
    """The parsed file, once per process. Keyed by path so a test pointing
    elsewhere cannot poison the entry the application uses.

    Returns ``{}`` when the file is absent or will not parse, rather than
    raising: every caller in this module has a working fallback, and the one
    place a bad file should stop the world is ``validatePersona`` at startup.
    Both failures are logged, because a service quietly running on the fallback
    prompt looks exactly like one running on the prompt somebody wrote.
    """
    file = Path(path)
    try:
        raw = file.read_bytes()
    except FileNotFoundError:
        logger.warning(
            "No persona configuration at %s; using the built-in fallback prompt.", file
        )
        return {}
    except OSError:
        logger.exception("Could not read the persona configuration at %s.", file)
        return {}

    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        logger.exception("%s is not valid TOML; using the built-in fallback prompt.", file)
        return {}


def personaNames() -> list[str]:
    """Every persona the file defines, for error messages and the log."""
    return [
        name
        for name, section in _readConfig(str(_configPath())).items()
        if isinstance(section, dict)
    ]


def activePersonaName() -> str:
    """Which persona is selected: the environment, then the file's ``default``."""
    override = (os.environ.get(ENV_PERSONA) or "").strip()
    if override:
        return override
    return str(_readConfig(str(_configPath())).get("default") or "").strip()


def _resolve() -> Persona:
    """The active persona, or raise. The strict half -- see the module docstring."""
    path = _configPath()
    config = _readConfig(str(path))
    name = activePersonaName()

    if not name:
        raise PromptConfigError(
            f"No persona is selected. Set `default = \"<name>\"` at the top of {path}, "
            f"or set {ENV_PERSONA}."
        )

    section = config.get(name)
    if not isinstance(section, dict):
        known = ", ".join(personaNames()) or "none"
        raise PromptConfigError(
            f"Persona '{name}' is not defined in {path}. Defined personas: {known}."
        )

    systemPrompt = str(section.get("systemPrompt") or "").strip()
    if not systemPrompt:
        raise PromptConfigError(
            f"Persona '{name}' in {path} has no systemPrompt, so the agent would be "
            f"given no instructions at all."
        )

    return Persona(
        name=name,
        systemPrompt=systemPrompt,
        reviewCriteria=str(section.get("reviewCriteria") or "").strip(),
        source=str(path),
    )


def validatePersona() -> Persona:
    """Resolve the persona and raise if it cannot be. Called at startup.

    An override naming a persona that does not exist is the failure worth
    catching here: it is silent otherwise, and the symptom -- an assistant that
    is not the one you configured -- is invisible until somebody reads a
    transcript.
    """
    return _resolve()


def activePersona() -> Persona | None:
    """The active persona, or None if it could not be resolved. Never raises."""
    try:
        return _resolve()
    except PromptConfigError:
        logger.warning(
            "The persona could not be resolved; falling back to the built-in prompt.",
            exc_info=True,
        )
        return None


def defaultSystemPrompt() -> str:
    """What a project with no prompt of its own answers with. Never raises.

    On the path of a live question, so every failure degrades: an environment
    override wins, then the persona, then the built-in fallback.
    """
    override = (os.environ.get(ENV_PROMPT_OVERRIDE) or "").strip()
    if override:
        return override

    persona = activePersona()
    return persona.systemPrompt if persona else FALLBACK_SYSTEM_PROMPT


def reviewCriteria() -> str:
    """What the reviewer weighs on top of accuracy, or "" for nothing extra.

    Empty whenever the prompt itself is being overridden by
    ``RAG_DEFAULT_SYSTEM_PROMPT``: the criteria in the file describe the persona
    in the file, and grading an unrelated prompt against them would mark answers
    down for not doing something they were never told to do.
    """
    if (os.environ.get(ENV_PROMPT_OVERRIDE) or "").strip():
        return ""

    persona = activePersona()
    return persona.reviewCriteria if persona else ""


def describePersona() -> str:
    """One line for the startup log. Never raises -- it is a log line."""
    if (os.environ.get(ENV_PROMPT_OVERRIDE) or "").strip():
        return f"custom prompt from ${ENV_PROMPT_OVERRIDE} (no review criteria)"

    persona = activePersona()
    if persona is None:
        return "NOT CONFIGURED -- answering with the built-in fallback prompt"

    graded = "with review criteria" if persona.reviewCriteria else "no review criteria"
    return f"{persona.name} ({graded}, from {persona.source})"
