"""Grades an answer against the question it was meant to answer.

A second model reads (question, answer) and returns a score in 0.0-1.0 plus,
when the score is low, a concrete suggestion for what a better answer would do.
Below ``REVIEW_THRESHOLD`` the agent is asked once more with that suggestion in
hand.

**Review happens exactly once.** The retried answer is returned as it stands and
is never graded again. That is a deliberate ceiling, not an oversight: a loop
that keeps reviewing until it is satisfied has no guaranteed exit, doubles cost
per lap, and on a question the documents genuinely cannot answer it never
terminates -- the honest "the documents do not cover this" scores badly every
time, because the reviewer is grading the answer rather than the corpus.

A reviewer failure is not an error. If the grading call itself fails, the
original answer is returned ungraded; a broken judge must not swallow a good
answer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.agent.llmManager import reviewerModel
from app.agent.usage import trackUsage
from app.promptConfig import reviewCriteria

logger = logging.getLogger(__name__)

# Below this, the answer goes back for one more attempt. 0.7 asks for "clearly
# answers the question", not perfection -- set it too high and every honest
# "the documents do not cover this" pays for a retry that cannot improve it.
REVIEW_THRESHOLD = float(os.environ.get("RAG_REVIEW_THRESHOLD", 0.7))

REVIEW_RUBRIC = (
    "You grade a draft answer against the question it was written for. Score "
    "0.0 to 1.0 on whether it actually answers the question: is it responsive, "
    "specific, and self-consistent?\n\n"
    "Grade the answer, not the underlying documents. An answer that clearly and "
    "honestly states the available material does not cover the question is a "
    "GOOD answer -- score it high. Reserve low scores for answers that are "
    "evasive, off-topic, self-contradictory, padded without substance, or that "
    "leave an obviously answerable part of the question untouched."
)

# Interpolated rather than written into the text: the threshold is
# configurable, and a prompt naming a different number from the one the code
# compares against would have the reviewer withholding suggestions on exactly
# the answers that are about to be retried.
SUGGESTION_INSTRUCTION = (
    f"When the score is below {REVIEW_THRESHOLD}, put one concrete instruction "
    "in `suggestion` saying what a better attempt should do differently. "
    "Otherwise leave `suggestion` empty."
)


def reviewerSystemPrompt() -> str:
    """The rubric, plus whatever the active persona adds to it.

    Built per call rather than frozen at import, for the same reason the agent's
    own prompt is: the persona is configuration and a restart should not be
    needed to change it.

    The persona's criteria go *between* the rubric and the suggestion
    instruction, not after it. Appended at the end they would sit below "put one
    instruction in `suggestion`", which reads as a qualifier on the output format
    rather than as something to grade on.

    Why a persona gets a say here at all: without it the reviewer pulls against
    the agent. It grades "does this answer the question", so a flat factual dump
    that ignores everything a sales persona was told to do scores 1.0, and the
    persona has no feedback loop. What a persona must NOT do is outrank honesty
    -- see the note at the bottom of config/prompts.toml.
    """
    criteria = reviewCriteria()
    if not criteria:
        return f"{REVIEW_RUBRIC}\n\n{SUGGESTION_INSTRUCTION}"

    return (
        f"{REVIEW_RUBRIC}\n\n"
        f"Additionally, for this deployment:\n{criteria}\n\n"
        f"{SUGGESTION_INSTRUCTION}"
    )


class Review(BaseModel):
    """The reviewer's structured verdict."""

    score: float = Field(
        ge=0.0, le=1.0, description="How well the answer answers the question."
    )
    suggestion: str = Field(
        default="", description="What a better attempt should do. Empty if the answer is fine."
    )


@dataclass
class ReviewOutcome:
    score: float | None
    suggestion: str
    retried: bool

    @property
    def reviewed(self) -> bool:
        """False when grading could not run at all."""
        return self.score is not None


def retryInstruction(question: str, answer: str, suggestion: str) -> str:
    """The follow-up turn sent to the agent after a poor review."""
    return (
        f"Your previous answer to this question was judged not good enough.\n\n"
        f"Question: {question}\n\n"
        f"Your previous answer:\n{answer}\n\n"
        f"What to do differently: {suggestion}\n\n"
        f"Write an improved answer. Search the project's documents again if that "
        f"is what the feedback calls for. Reply with the answer only -- do not "
        f"mention this feedback or that you are revising."
    )


async def review(question: str, answer: str, model=None, usage=None) -> Review | None:
    """Grade ``answer``. Returns None if grading could not run.

    ``usage`` collects what this call cost. Structured output is why it has to
    be a callback rather than a look at the return value -- see
    ``app.agent.usage``.
    """
    if not answer.strip():
        # Nothing to grade, and an empty answer is bad by definition -- say so
        # without paying for a model call.
        return Review(score=0.0, suggestion="The previous attempt produced no answer at all.")

    try:
        judge = (model or reviewerModel()).with_structured_output(Review)
        with trackUsage(usage, "reviewer"):
            verdict = await judge.ainvoke(
                [
                    {"role": "system", "content": reviewerSystemPrompt()},
                    {
                        "role": "user",
                        "content": f"Question:\n{question}\n\nDraft answer:\n{answer}",
                    },
                ]
            )
    except Exception:
        logger.exception("The answer reviewer failed; returning the answer ungraded.")
        return None

    if isinstance(verdict, dict):  # Some providers hand back a plain dict.
        verdict = Review(**verdict)
    return verdict


def needsAnotherAttempt(verdict: Review | None) -> bool:
    """Whether a verdict is bad enough to be worth one more try."""
    return verdict is not None and verdict.score < REVIEW_THRESHOLD
