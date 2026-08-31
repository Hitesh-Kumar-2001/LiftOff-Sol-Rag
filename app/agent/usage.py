"""What one answer cost, in tokens, broken down by which model did what.

Three models run on the way to a single answer, and they are billed
separately::

    summariser   folds the conversation, when it has outgrown its budget
    agent        answers -- several turns and tool calls, sometimes twice
    reviewer     grades the draft

Only the agent's cost is obvious. The summariser is invisible until a long
conversation makes it the expensive one, and the reviewer runs on *every*
question. Recording a single total would hide both, so everything here is kept
per role.

How the numbers are obtained
----------------------------
``get_usage_metadata_callback`` from langchain-core, one context per role. It
registers a handler in the ambient callback context, so any model call inside
the block is counted without threading a config down through deepagents and
LangGraph -- which is the only practical way to see the agent's internal turns.

**Structured output is why the callback is used rather than the return value.**
``with_structured_output`` hands back the parsed Pydantic object, and the
``AIMessage`` that carried ``usage_metadata`` is gone by then -- so the reviewer
and the summariser's schema-constrained calls would report zero if this read
usage off what they return. The callback sees the raw message either way.

**One context per role, not one per request.** The handler aggregates by *model
name*, and the agent, reviewer and summariser are usually pointed at the same
model, so a single shared context would merge all three into one number and
lose exactly the breakdown this exists to provide.

Isolation
---------
The handler lives in a ``contextvar``, and every request runs in its own asyncio
task with its own context, so two questions answered concurrently do not see
each other's tokens. ``tests/testUsage.py`` pins that, because the failure --
one project billed for another's traffic -- would be silent and would show up
as a number nobody could explain.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# The roles that answer a question. `chunker` is deliberately absent: it runs
# during ingestion, against a document rather than a conversation, so its cost
# belongs to a job and not to anybody's message.
ANSWER_ROLES = ("agent", "reviewer", "summariser")


@dataclass(frozen=True)
class RoleUsage:
    """One role's spend on one answer, on one model.

    ``cachedInputTokens`` and ``reasoningTokens`` are carried because they are
    priced differently from the totals that contain them -- cached input is
    billed at a discount and reasoning tokens are billed as output -- so a
    total alone cannot be turned back into money.
    """

    role: str
    provider: str
    model: str
    inputTokens: int = 0
    outputTokens: int = 0
    totalTokens: int = 0
    cachedInputTokens: int = 0
    reasoningTokens: int = 0

    def asDocument(self) -> dict:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "inputTokens": self.inputTokens,
            "outputTokens": self.outputTokens,
            "totalTokens": self.totalTokens,
            "cachedInputTokens": self.cachedInputTokens,
            "reasoningTokens": self.reasoningTokens,
        }


@dataclass
class UsageLog:
    """What this answer has spent so far. Mutable, and passed down.

    An accumulator threaded through the call rather than a return value, for
    the same reason ``searchLog`` is: the functions doing the spending already
    have a return value that is the point of calling them, and the cost has to
    come back from calls the caller does not otherwise see inside.
    """

    entries: list[RoleUsage] = field(default_factory=list)

    def record(self, role: str, provider: str, model: str, usageMetadata: dict) -> None:
        """Fold one role's callback result in. One entry per model it used."""
        for reportedModel, counts in (usageMetadata or {}).items():
            if not isinstance(counts, dict):
                continue
            inputDetails = counts.get("input_token_details") or {}
            outputDetails = counts.get("output_token_details") or {}
            self.entries.append(
                RoleUsage(
                    role=role,
                    provider=provider,
                    # What the vendor actually billed, which can be more
                    # specific than what was configured -- "gpt-5.6-luna" may
                    # come back as a dated build. The configured name is
                    # recorded as the provider's peer, not as a substitute.
                    model=str(reportedModel or model),
                    inputTokens=int(counts.get("input_tokens") or 0),
                    outputTokens=int(counts.get("output_tokens") or 0),
                    totalTokens=int(counts.get("total_tokens") or 0),
                    cachedInputTokens=int(inputDetails.get("cache_read") or 0),
                    reasoningTokens=int(outputDetails.get("reasoning") or 0),
                )
            )

    @property
    def totalTokens(self) -> int:
        return sum(entry.totalTokens for entry in self.entries)

    def byRole(self) -> dict[str, dict]:
        """Per-role sums, which is the shape the store writes.

        A role that ran more than once -- the agent, when an answer is retried
        -- collapses into one row with ``calls`` counting the attempts.
        """
        rolled: dict[str, dict] = {}
        for entry in self.entries:
            row = rolled.setdefault(
                entry.role,
                {
                    "provider": entry.provider,
                    "model": entry.model,
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 0,
                    "cachedInputTokens": 0,
                    "reasoningTokens": 0,
                    "calls": 0,
                },
            )
            row["inputTokens"] += entry.inputTokens
            row["outputTokens"] += entry.outputTokens
            row["totalTokens"] += entry.totalTokens
            row["cachedInputTokens"] += entry.cachedInputTokens
            row["reasoningTokens"] += entry.reasoningTokens
            row["calls"] += 1
        return rolled

    def summary(self) -> str:
        """One line for the log: what this answer cost and where it went."""
        if not self.entries:
            return "no usage reported"
        parts = [
            f"{role}={row['totalTokens']}" for role, row in sorted(self.byRole().items())
        ]
        return f"{self.totalTokens} tokens ({', '.join(parts)})"


@contextmanager
def trackUsage(usage: UsageLog | None, role: str):
    """Count every model call made inside this block against ``role``.

    ``usage`` of None disables tracking entirely and costs nothing -- which is
    what every existing caller that does not care about cost gets, and what the
    tests that stub a model out get for free.

    The recording happens in a ``finally``: a call that raised after the
    provider had already produced tokens was still billed for them, and an
    answer that failed expensively is exactly the one worth being able to see.
    """
    if usage is None:
        yield
        return

    from langchain_core.callbacks import get_usage_metadata_callback

    provider, model = _configuredFor(role)
    with get_usage_metadata_callback() as handler:
        try:
            yield
        finally:
            try:
                usage.record(role, provider, model, handler.usage_metadata)
            except Exception:
                # Accounting must never be the reason an answer is lost.
                logger.warning("Could not record %s usage.", role, exc_info=True)


def _configuredFor(role: str) -> tuple[str, str]:
    """The provider and model this role was configured with.

    Read from configuration rather than inferred from the response, because a
    provider is not something a response reports -- only a model name is, and
    two vendors can serve the same one.
    """
    try:
        from app.modelConfig import modelFor

        choice = modelFor(role)
        return choice.provider, choice.model
    except Exception:
        # A role that could not be resolved did not run, so there is nothing to
        # attribute; the entry still gets written with what is known.
        return "", ""
