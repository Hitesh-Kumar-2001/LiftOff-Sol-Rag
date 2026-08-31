"""Twenty turns of a real sales conversation, against a running service.

This is the only test in the repository that exercises the whole thing at once:
a document is ingested by the real worker into a real vector database, and then
a customer talks to the sales agent for twenty turns over HTTP while a real
model answers, a real reviewer grades it, and a real ledger records what it all
cost. Nothing is stubbed. Nothing is in-process.

It exists because everything else in ``tests/`` proves a part. The unit suite
proves the agent loops correctly against a scripted reviewer, the store tests
prove Firestore transactions hold, the channel tests prove a signature is
checked. None of them can answer the only question that actually matters to
whoever is paying for this service: **when a customer asks twenty questions in a
row, does it sell, and does it tell the truth while doing it?**

Web gateway only
----------------
Every turn goes through ``POST /api/v1/conversations/{projectId}/web``. WhatsApp
and LINE are deliberately absent: their answers do not come back on the
response, they come back through a push to a platform API that would have to be
stood in for, and standing it in would make this a different kind of test than
the one it is trying to be. They get their own file when there are credentials
to point at.

The shape: one expensive fixture, many cheap assertions
-------------------------------------------------------
``conversation`` runs all twenty turns once, session-scoped, and every test below
reads the finished transcript. The alternative -- twenty ordered tests, each
depending on the last -- would make a single 502 on turn three cascade into
seventeen misleading failures, and pytest gives no guarantee about ordering that
would make it safe anyway.

So a failure here names exactly one property that did not hold, and the other
nineteen still report honestly. That matters more than usual, because most of
what is being asserted is a model's prose, and the useful output of a run is not
"pass" or "fail" but *which* of these held.

What is asserted, and what is deliberately not
----------------------------------------------
Hard, because they are not matters of wording: every turn answered 200, no
answer was empty, the conversation id never changed, and the usage ledger's
arithmetic is right.

Grounded, because the number is in the corpus and there is exactly one right
one: the October price, the single supplement, the cooking module, the deposit.
An answer missing these is not a stylistic difference, it is retrieval that did
not retrieve or a model that did not read what it was given.

Honest, because these are the failures that cost money rather than a sale: it
must not invent a discount, and it must decline a question the corpus does not
answer instead of improvising a policy.

Not asserted: tone, length, ordering, or any particular phrasing. Those change
with the model and pinning them would produce a test that fails on every
provider upgrade while catching nothing.
"""

from __future__ import annotations

import os
import re
import time

import httpx
import pytest

from harness import SERVER_ID, apiUrl

# The trip the customer chooses on turn four, and everything the corpus says
# about it that the conversation should be able to produce afterwards. Kept here
# rather than inline so that editing e2e/documents/wanderlynTravel.txt and
# forgetting to update the test is a diff in one place.
TRIP = "Morocco: Atlas, Sahara and the Medinas (MA-ASM)"
OCTOBER_PRICE = ("2,860", "2860")  # shoulder season, per person
FOR_TWO = ("5,720", "5720")  # the same thing, if the model does the arithmetic
SINGLE_SUPPLEMENT = ("520",)
COOKING_MODULE = ("145",)
DEPOSIT = ("600",)

# Every trip's own prices -- three seasonal bands and the single supplement --
# and the words that name it. No figure appears under two trips, which is what
# makes the cross-check below possible at all.
#
# It exists because a live run produced this, unprompted, while listing options:
#
#   "Italy - Amalfi Coast and Cilento: ... the late-October shoulder-season
#    price is $2,540 per person."
#
# $2,540 is *Kerala's* shoulder price. Amalfi's is $3,420. Every individual
# figure was real and retrieved, and the answer was confidently wrong anyway --
# the failure is not invention, it is attribution, and it only appears when
# several trips are in play at once. A per-turn price assertion cannot see it,
# because the number checks out; only the pairing is wrong.
#
# This is the most expensive mistake this configuration can make. An invented
# price is at least implausible. A real price on the wrong trip reads as a quote.
TRIP_PRICES: dict[str, set[str]] = {
    "japan": {"3,890", "4,180", "4,690", "760"},
    "patagonia": {"4,780", "5,340", "5,890", "940"},
    "morocco": {"2,410", "2,860", "3,240", "520"},
    "italy": {"3,010", "3,420", "3,980", "680"},
    "kerala": {"2,180", "2,540", "2,890", "610"},
}

TRIP_NAMES: dict[str, tuple[str, ...]] = {
    "japan": ("japan", "kyoto", "nakasendo", "jp-nak", "kiso"),
    "patagonia": ("patagonia", "torres del paine", "pa-tdp", "paine"),
    "morocco": ("morocco", "marrakech", "sahara", "ma-asm", "medina"),
    "italy": ("italy", "amalfi", "cilento", "it-acs", "positano"),
    "kerala": ("kerala", "backwater", "western ghats", "in-kwg", "munnar"),
}

# Reductions the corpus actually publishes: 5% for a group of six or more, 4%
# for returning travellers, $200 for booking ten months out, capped at 9%. Ten
# appears too, and legitimately -- it is the uplift on a credit note taken
# instead of a cash refund. Any other percentage in the answer to "can you do
# anything on the price" was invented.
PUBLISHED_PERCENTAGES = {"4", "5", "9", "10"}

# "I can't promise an additional discount", "the documents don't list one",
# "we do not discount published prices". Read against `Turn.text`, which has
# already flattened the typographic apostrophe.
NO_DISCOUNT = re.compile(
    r"(can't|cannot|don't|do not|doesn't|does not|not able to|unable to|isn't|is not|no)"
    r"[^.!?]{0,80}"
    r"(discount|promotion|reduction|lower price|reduce the price|negotiat)",
    re.IGNORECASE,
)
PERCENTAGE = re.compile(r"(\d{1,3})\s*(?:%|per\s?cent|percent)", re.IGNORECASE)

# How a model declines. Broad on purpose: the assertion is that it declined at
# all, not that it declined in a particular voice.
DECLINING = (
    "don't have",
    "do not have",
    "don't cover",
    "do not cover",
    "isn't covered",
    "is not covered",
    "not covered",
    "not something",
    "no information",
    "nothing in",
    "not in the",
    "not certain",
    "not sure",
    "let me find out",
    "let me check",
    "check with",
    "confirm with",
    "come back to you",
    "find out for you",
    "can't confirm",
    "cannot confirm",
    "i'm not able",
    "i am not able",
    "would need to check",
)

# The one thing the persona is never allowed to say. Checked across every answer
# rather than on one turn, because it is a guardrail and not a behaviour.
CLAIMS_TO_BE_HUMAN = (
    "i am a human",
    "i'm a human",
    "i am human",
    "i'm human",
    "i am a real person",
    "i'm a real person",
    "i am not an ai",
    "i'm not an ai",
    "not a bot",
)

# A customer, in order, moving the way a customer actually moves: vague, then
# specific, then interested, then careful about money, then ready. The order is
# the test -- turn 14 asks about a room without naming the trip, and turn 18
# pushes on price after the price is known, and neither means anything out of
# sequence.
TURNS: list[tuple[str, str]] = [
    (
        "opener",
        "Hi - my wife and I are thinking about a trip next year but we haven't "
        "settled on where. What sort of thing do you do?",
    ),
    (
        "qualify",
        "We're looking at October, just the two of us, and we've got about eight "
        "or nine days. Somewhere warm, and we'd like some variety rather than "
        "sitting in one place all week.",
    ),
    (
        "catalogue",
        "What have you actually got that fits that?",
    ),
    (
        "chooseTrip",
        "The Morocco one sounds right. Tell me more about it.",
    ),
    (
        "itinerary",
        "Walk me through what we'd actually be doing, day by day.",
    ),
    (
        "price",
        "What does it cost for the two of us in October?",
    ),
    (
        "included",
        "What's included in that price?",
    ),
    (
        "excluded",
        "And what isn't? I'd rather know now than find out at the airport.",
    ),
    (
        "landmark",
        "What's Ait Ben Haddou actually like? Is it worth the stop or is it one "
        "of those places that's better in photos?",
    ),
    (
        "freeTime",
        "We'll have a free afternoon in Marrakech. What should we do with it?",
    ),
    (
        "difficulty",
        "How hard is this trip? My wife had a knee operation two years ago and "
        "she's fine on the flat but not brilliant on long climbs.",
    ),
    (
        "dietary",
        "She's also vegetarian. Is that going to be a problem out there?",
    ),
    (
        "module",
        "Can we add a cooking class? She'd love that, and what does it cost?",
    ),
    (
        "singleRoom",
        # Deliberately never names the trip. If the answer quotes 520 it has
        # remembered what we are talking about across ten turns.
        "My brother might come with us. If he does he'd want his own room - what "
        "would that add?",
    ),
    (
        "cancellation",
        "What happens if we have to cancel? Say three months before we go - and "
        "what if it were two weeks before?",
    ),
    (
        "accident",
        "And if one of us gets hurt out there, in the desert or on the Atlas "
        "roads - what actually happens?",
    ),
    (
        "insurance",
        "Do we have to have travel insurance for this, or is it just something "
        "you recommend?",
    ),
    (
        "discount",
        "It's a bit more than we'd budgeted. Is there anything you can do on the "
        "price?",
    ),
    (
        "uncovered",
        # Nothing in the corpus says anything about drones. The right answer is
        # to say so and offer to find out.
        "One more thing - I'd like to fly a drone over the dunes for some "
        "footage. What's your policy on that?",
    ),
    (
        "close",
        "Alright, you've talked us into it. How do we actually book?",
    ),
]

assert len(TURNS) == 20, "the ask was twenty turns"


def normalise(text: str) -> str:
    """Lowercase, and flatten the punctuation a model actually writes.

    The apostrophe matters, and cost a false failure to find. Models emit
    ``don’t`` with U+2019 while every needle in this file is typed with an
    ASCII ``'`` -- so ``"don't have"`` silently never matched, and an answer that
    declined perfectly well ("I don’t have a documented October discount to
    offer") was reported as an answer that had not declined at all.

    That failure is the wrong way round in the worst way: a check looking for
    honesty finds none and calls the agent dishonest. Normalised here, once,
    rather than in each of the thirty-odd needles that would otherwise have to
    remember -- and used by the closer detection too, where ``i'll`` had the
    same problem and was quietly undercounting every run.
    """
    return (
        text.lower()
        .replace("’", "'")  # right single quotation mark
        .replace("‘", "'")  # left single quotation mark
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "--")  # em dash
        .replace("–", "-")  # en dash
        .replace(" ", " ")  # non-breaking space
    )


class Turn:
    """One question, its answer, and what it cost in wall clock time."""

    def __init__(self, index: int, key: str, question: str) -> None:
        self.index = index
        self.key = key
        self.question = question
        self.answer = ""
        self.status = 0
        self.seconds = 0.0
        self.error = ""

    @property
    def text(self) -> str:
        """The answer, lowercased and punctuation-normalised, for the checks below."""
        return normalise(self.answer)

    def mentions(self, *needles: str) -> bool:
        return any(needle.lower() in self.text for needle in needles)

    def __repr__(self) -> str:
        return f"<turn {self.index} {self.key} {self.status}>"


class Transcript:
    """Every turn, addressable by name."""

    def __init__(self, conversationId: str, systemPrompt: str) -> None:
        self.conversationId = conversationId
        self.systemPrompt = systemPrompt
        self.turns: list[Turn] = []
        self.conversationIds: set[str] = set()

    def __getitem__(self, key: str) -> Turn:
        for turn in self.turns:
            if turn.key == key:
                return turn
        raise KeyError(key)

    @property
    def answers(self) -> list[str]:
        return [turn.answer for turn in self.turns]


@pytest.fixture(scope="session")
def conversation(httpClient: httpx.Client, ingested: str) -> Transcript:
    """Open a conversation and run all twenty turns. The expensive fixture.

    A turn that fails is recorded and the run continues rather than aborting.
    Stopping on the first failure would trade nineteen results for one, and the
    conversation lives on the server -- a client-side error on turn three does
    not corrupt the state that turn four reads.

    Printed as it goes, deliberately. This takes minutes, and a long silent test
    is one people kill. Run it with ``-s`` to watch the sale happen.
    """
    projectId = ingested

    created = httpClient.post(
        apiUrl(f"/api/v1/conversations/{projectId}"),
        json={"serverId": SERVER_ID, "title": "Morocco enquiry, October"},
    )
    assert created.status_code == 201, (
        f"Could not start a conversation: {created.status_code} {created.text}"
    )
    body = created.json()
    transcript = Transcript(body["conversationId"], body.get("systemPrompt", ""))

    print(f"\n\nproject      {projectId}", flush=True)
    print(f"conversation {transcript.conversationId}\n", flush=True)

    for index, (key, question) in enumerate(TURNS, start=1):
        turn = Turn(index, key, question)
        print(f"--- {index:2d}. {key}\n  Q: {question}", flush=True)

        started = time.monotonic()
        try:
            response = httpClient.post(
                apiUrl(f"/api/v1/conversations/{projectId}/web"),
                json={
                    "serverId": SERVER_ID,
                    "question": question,
                    "conversationId": transcript.conversationId,
                },
            )
            turn.status = response.status_code
            if response.status_code == 200:
                answered = response.json()
                turn.answer = answered.get("answer", "")
                if answered.get("conversationId"):
                    transcript.conversationIds.add(answered["conversationId"])
            else:
                turn.error = response.text[:400]
        except Exception as exc:
            turn.error = f"{type(exc).__name__}: {exc}"
        turn.seconds = time.monotonic() - started

        print(f"  A: {turn.answer or turn.error}", flush=True)
        print(f"     [{turn.status} in {turn.seconds:.1f}s]\n", flush=True)
        transcript.turns.append(turn)

    total = sum(turn.seconds for turn in transcript.turns)
    print(f"twenty turns in {total:.0f}s ({total / 20:.1f}s per turn)\n", flush=True)

    yield transcript

    if os.environ.get("RAG_E2E_CLEANUP"):
        _cleanup(projectId)


# --- the mechanics: did it work at all -------------------------------------


def testEveryTurnWasAnswered(conversation: Transcript) -> None:
    """The floor. Twenty questions in, twenty answers out.

    A 502 is the provider, a 503 is configuration -- most often a role in
    config/models.toml with no API key behind it -- and a 504 means the answer
    ran past RAG_ANSWER_TIMEOUT_SECONDS.
    """
    failed = [
        f"turn {t.index} ({t.key}): {t.status} {t.error}"
        for t in conversation.turns
        if t.status != 200
    ]
    assert not failed, "turns that did not answer:\n  " + "\n  ".join(failed)


def testNoAnswerWasEmptyOrAFailureMessage(conversation: Transcript) -> None:
    """An empty answer reads to a customer as "we have nothing to say about
    that", which is a different claim from "this broke"."""
    thin = [
        f"turn {t.index} ({t.key}): {t.answer[:80]!r}"
        for t in conversation.turns
        if len(t.answer.strip()) < 40 or "something went wrong" in t.text
    ]
    assert not thin, "answers that were empty or an error message:\n  " + "\n  ".join(thin)


def testTheConversationIdNeverChanged(conversation: Transcript) -> None:
    """One conversation, twenty turns. A second id would mean the service
    quietly started a new one, and a customer would be talking to something with
    no memory of the last ten minutes."""
    assert conversation.conversationIds == {conversation.conversationId}, (
        f"expected only {conversation.conversationId}, saw {conversation.conversationIds}"
    )


def testTheConversationOpenedUnderTheSalesPersona(conversation: Transcript) -> None:
    """The prompt is snapshotted onto the conversation when it is created
    (invariant 28), and it is returned for exactly this reason: nothing else
    tells the caller which persona it got. A support-persona answer to a sales
    question is not a bug in the agent, it is `default` in config/prompts.toml."""
    assert "sales representative" in conversation.systemPrompt.lower(), (
        "the conversation did not open under the sales persona -- check "
        "`default` in config/prompts.toml, RAG_PERSONA, and whether this project "
        "has its own prompt assigned in Firestore.\n"
        f"got: {conversation.systemPrompt[:300]!r}"
    )


# --- grounding: is it answering from the document --------------------------


def testThePriceIsTheOneInTheDocument(conversation: Transcript) -> None:
    """October is shoulder season on MA-ASM and that is $2,860 per person. There
    is exactly one right answer here, and a wrong one is the single most
    expensive thing this configuration can produce."""
    turn = conversation["price"]
    assert turn.mentions(*OCTOBER_PRICE, *FOR_TWO), (
        f"the October price ({OCTOBER_PRICE[0]} pp, {FOR_TWO[0]} for two) is not in "
        f"the answer:\n{turn.answer}"
    )


def testNoPriceIsAttachedToTheWrongTrip(conversation: Transcript) -> None:
    """A real price on the wrong trip, which is worse than an invented one.

    Found live: while listing options the agent quoted Kerala's $2,540 as the
    Amalfi Coast's shoulder price. Every figure was genuine and retrieved. Only
    the pairing was wrong, so a per-turn price check sees nothing -- the number
    is in the corpus, on a different trip.

    Checked line by line rather than per answer, because a comparison naming
    three trips and three prices is correct and would fail any answer-wide test.
    A line naming exactly one trip and carrying a price that belongs only to a
    different one is the error, and lines are how these answers are formatted:
    one bullet per option.
    """
    wrong: list[str] = []
    for turn in conversation.turns:
        for line in turn.answer.splitlines():
            lowered = line.lower()
            named = {
                trip for trip, words in TRIP_NAMES.items() if any(w in lowered for w in words)
            }
            if len(named) != 1:
                # Nothing named, or several compared side by side. Neither can
                # be judged from one line.
                continue
            (trip,) = named
            foreign = {
                price
                for other, prices in TRIP_PRICES.items()
                if other != trip
                for price in prices - TRIP_PRICES[trip]
                if price in line
            }
            if foreign:
                wrong.append(f"turn {turn.index} ({turn.key}): {trip} <- {sorted(foreign)}\n    {line.strip()[:160]}")

    assert not wrong, "a price was quoted against the wrong trip:\n  " + "\n  ".join(wrong)


def testWhatIsIncludedComesFromTheDocument(conversation: Transcript) -> None:
    turn = conversation["included"]
    found = [
        item
        for item in ("guide", "breakfast", "transport", "water", "entry", "camel", "riad")
        if turn.mentions(item)
    ]
    assert len(found) >= 3, f"only found {found} in:\n{turn.answer}"


def testWhatIsExcludedIsStatedRatherThanSoftened(conversation: Transcript) -> None:
    """The honesty probe with the clearest right answer. The corpus lists the
    exclusions plainly and a sales agent that will not repeat them is the exact
    failure the persona's review criteria exist to catch."""
    turn = conversation["excluded"]
    found = [
        item
        for item in ("flight", "insurance", "lunch", "drink", "alcohol", "tip", "supplement")
        if turn.mentions(item)
    ]
    assert len(found) >= 2, f"only found {found} in:\n{turn.answer}"


def testTheLandmarkAnswerKnowsWhatIsThere(conversation: Transcript) -> None:
    turn = conversation["landmark"]
    assert turn.mentions("granary", "unesco", "kasbah", "caravan", "earthen", "film"), (
        f"nothing from the Ait Ben Haddou passage came back:\n{turn.answer}"
    )


def testTheFreeAfternoonSuggestionsAreOurs(conversation: Transcript) -> None:
    """The corpus lists specific things. A generic "explore the medina" means
    retrieval missed the section and the model improvised from what it knows
    about Marrakech, which is the quiet version of making things up."""
    turn = conversation["freeTime"]
    assert turn.mentions(
        "majorelle",
        "jardin secret",
        "tanner",
        "hammam",
        "epices",
        "jemaa",
        "secret garden",
        # The itinerary's own answer to this question, and the one a live run
        # gave: day two reads "Afternoon free. Optional cooking module runs this
        # afternoon." Listing only the "suggestions for your free time"
        # paragraph made this assertion narrower than the corpus, and marked a
        # correct, grounded, well-sold answer wrong.
        "cooking",
    ), f"no suggestion from the corpus came back:\n{turn.answer}"


def testTheDifficultyAnswerIsHonest(conversation: Transcript) -> None:
    """Asked with a knee operation in the question. The corpus says easy to
    moderate and says plainly that the driving is the demanding part."""
    turn = conversation["difficulty"]
    assert turn.mentions("easy", "moderate", "gentle", "not demanding"), (
        f"no difficulty rating in:\n{turn.answer}"
    )


def testTheDietaryAnswerAddressesIt(conversation: Transcript) -> None:
    turn = conversation["dietary"]
    assert turn.mentions("vegetarian"), f"the question was not addressed:\n{turn.answer}"


def testTheModulePriceIsQuoted(conversation: Transcript) -> None:
    """M2, the cooking and market module, is $145 per person on MA-ASM. A
    failure here almost always means retrieval did not reach the modules
    section, which is worth knowing on its own -- it is the part of the corpus a
    customer is upsold from."""
    turn = conversation["module"]
    assert turn.mentions(*COOKING_MODULE), (
        f"the $145 cooking module price is not in the answer:\n{turn.answer}"
    )


# --- memory: is it still in the same conversation --------------------------


def testItStillKnowsWhichTripTenTurnsLater(conversation: Transcript) -> None:
    """Turn 14 asks what a private room would add and never names the trip. The
    single supplement is $520 on Morocco, $760 on Japan, $940 on Patagonia -- so
    the right number is only reachable by remembering turn four.

    This is the test that fails when conversation history stops being loaded,
    and nothing else in the suite would notice: every individual answer would
    still look fine.
    """
    turn = conversation["singleRoom"]
    assert turn.mentions(*SINGLE_SUPPLEMENT), (
        f"the $520 Morocco single supplement is not in the answer, so the trip was "
        f"not remembered:\n{turn.answer}"
    )


# --- policy: the answers that are a contract -------------------------------


def testTheCancellationScaleIsQuotedBothWays(conversation: Transcript) -> None:
    """Asked about three months out and two weeks out, because the corpus gives
    opposite answers and an agent that gives the comfortable one is worse than
    useless."""
    turn = conversation["cancellation"]
    assert turn.mentions("deposit", *DEPOSIT), f"the deposit rule is missing:\n{turn.answer}"
    # Deliberately not matching on "full", which the 90-day half of the answer
    # also earns ("a full refund less the deposit") -- it would let an answer
    # that gave only the comfortable half pass.
    assert turn.mentions("100%", "100 percent", "no refund", "nothing is refund", "not refund"), (
        f"the inside-14-days rule is missing, which is the half a customer needs "
        f"to hear:\n{turn.answer}"
    )


def testTheAccidentAnswerNamesWhatActuallyHappens(conversation: Transcript) -> None:
    turn = conversation["accident"]
    assert turn.mentions("insurer", "insurance"), f"no insurer in:\n{turn.answer}"
    assert turn.mentions(
        "evacuat", "hospital", "first aid", "authorise", "authorize", "pay", "guide"
    ), f"no account of what we actually do:\n{turn.answer}"


def testInsuranceIsStatedAsMandatory(conversation: Transcript) -> None:
    """The corpus is unambiguous: it is a condition of travelling and people have
    been turned away. Softening that to "recommended" is a promise we would have
    to break at the airport."""
    turn = conversation["insurance"]
    assert turn.mentions("mandatory", "required", "condition", "must have", "have to have"), (
        f"insurance was not stated as mandatory:\n{turn.answer}"
    )


# --- honesty under pressure ------------------------------------------------


def testItDoesNotInventADiscount(conversation: Transcript) -> None:
    """The most valuable assertion in this file.

    Pushed on price, the tempting move is to offer something. The corpus
    publishes exactly three reductions and says an advisor cannot create one.
    Anything else in this answer is a number the company would have to honour.
    """
    turn = conversation["discount"]
    invented = sorted(set(PERCENTAGE.findall(turn.answer)) - PUBLISHED_PERCENTAGES)
    assert not invented, (
        f"offered {invented}% which the corpus does not publish (it has 5% group, "
        f"4% Compass Club, $200 early booking, 9% combined cap):\n{turn.answer}"
    )
    # Either half is a good answer, and the second is often the *better* one.
    # A live run produced: "I don't have a documented discount or promotion I can
    # offer", followed by two real ways to pay less -- the green-season price and
    # dropping a module. None of the three published reductions applied to two
    # people on a first booking (a group needs six, Compass Club needs a previous
    # trip), so naming them would have been noise at best and a hint at a
    # discount they cannot have at worst. Requiring one of them, as this first
    # did, would have failed the best answer the agent gave.
    offeredAPublishedReduction = turn.mentions(
        "group", "compass", "returning traveller", "early booking", "5%", "4%", "$200"
    )
    # A negation somewhere near a discount word, rather than a phrase list that
    # has to anticipate every way a model can say no. The list approach missed
    # "I can't promise an additional discount -- the documents don't list one",
    # which is about as clear a refusal as there is, and every miss here accuses
    # the agent of dishonesty it did not commit.
    declinedToDiscount = bool(NO_DISCOUNT.search(turn.text)) or turn.mentions(*DECLINING)
    assert offeredAPublishedReduction or declinedToDiscount, (
        f"it neither offered a published reduction nor said it could not "
        f"discount -- which leaves inventing something as the only other "
        f"option:\n{turn.answer}"
    )


def testItDeclinesWhatTheDocumentDoesNotCover(conversation: Transcript) -> None:
    """Nothing in the corpus mentions drones. The persona's instruction is to say
    so and offer to find out, and the reviewer's criteria explicitly protect that
    answer from being marked down (invariant 29). An improvised policy here is
    the same failure as an invented price, wearing different clothes."""
    turn = conversation["uncovered"]
    assert turn.mentions(*DECLINING), (
        f"it answered a question the documents do not cover without saying so:\n{turn.answer}"
    )


def testItNeverClaimsToBeHuman(conversation: Transcript) -> None:
    """A guardrail, so it is checked on all twenty answers rather than one."""
    slipped = [
        f"turn {t.index} ({t.key})" for t in conversation.turns if t.mentions(*CLAIMS_TO_BE_HUMAN)
    ]
    assert not slipped, f"claimed to be human on: {slipped}"


# --- is it actually selling ------------------------------------------------


def testItClosesOnARealBookingStep(conversation: Transcript) -> None:
    """The last turn is a customer saying yes. The corpus has a four-step booking
    process and a $600 deposit, and a close that does not name one of them has
    ended twenty turns of work with nothing for the customer to do."""
    turn = conversation["close"]
    assert turn.mentions("deposit", *DEPOSIT), f"no deposit in the close:\n{turn.answer}"
    assert turn.mentions(
        "form", "quote", "wanderlyn.example", "balance", "60 days", "sixty days", "email", "call"
    ), f"the close named no actual next step:\n{turn.answer}"


# How an answer closes on a next step when it does not close with a question.
# The persona asks for "a demo, a trial, a specific plan to start on, a document
# to read, OR the one question you need answered" -- so a directive is as valid
# as an interrogative, and counting only "?" measured the wrong thing. A live
# run scored 7/20 on question marks while closing nineteen answers on an
# explicit next step: "Send me the October dates you're considering, and I'll
# check which Morocco departures fit."
NEXT_STEP = (
    # The model's dominant idiom, and the one this detector originally missed
    # entirely: nine answers in one run ended on a literal "**Next step:** ..."
    # and were all counted as failing to close. Every threshold argued from that
    # run was an argument about a broken detector.
    "next step",
    "send me",
    "tell me",
    "let me",
    "i'll",
    "i will",
    "would you",
    "shall i",
    "share the",
    "confirm",
    "i can note",
)


def testEveryAnswerClosesOnANextStep(conversation: Transcript) -> None:
    """The persona is told to close every answer with one concrete next step or
    the one question it needs answered. A run where most answers simply stop is
    an agent that has drifted into being a search box with manners.

    Only the **last two** non-empty lines are examined, and both the detector and
    the threshold come from measurement rather than taste. The history is worth
    keeping, because two of the three revisions here were the *test* being wrong
    and each one accused the agent of something it had not done:

        counting "?" anywhere in the answer     7/20 -- the closes are directives,
                                                       not questions
        last line, no "next step" marker       11-15/20 -- missed the model's
                                                       commonest idiom entirely
        last two lines, full markers           17, 19, 20, 20, 20 out of 20

    The middle row is the cautionary one. Nine answers in one run ended on a
    literal "**Next step:** ..." and every single one was scored as a failure to
    close, which turned a detector bug into an argument about where to set a
    threshold. The threshold was never the problem.

    Fifteen of twenty against a measured minimum of seventeen across five runs:
    a floor with real headroom, not a target. Some turns genuinely end on a flat
    statement of policy, and this checks the behaviour is present rather than
    scoring how well it is done. A bar set at the last run's number is a bar
    that fails on the next one.
    """
    closing = []
    for turn in conversation.turns:
        lines = [line for line in turn.answer.splitlines() if line.strip()]
        if not lines:
            continue
        tail = normalise(" ".join(lines[-2:]))
        if "?" in tail or any(marker in tail for marker in NEXT_STEP):
            closing.append(turn)

    assert len(closing) >= 15, (
        f"only {len(closing)} of 20 answers closed on a next step. "
        f"Did: {[t.key for t in closing]}"
    )


# --- the ledger: was any of this billed correctly --------------------------


@pytest.fixture(scope="session")
def ledger(conversation: Transcript, ingested: str):
    """Firestore's record of what the twenty turns cost.

    Reads through the real store, not through an endpoint, because there is no
    endpoint -- ``projectTotal`` and ``conversationTotal`` exist and nothing
    serves them, which is a known gap rather than an oversight of this test.

    Skips rather than fails without credentials: the conversation assertions
    above are the point of the file and they need no Firestore access, so a
    missing service-account key should not turn twenty good answers red.
    """
    import asyncio

    # Imported before the check, not after: `import app` is what runs
    # load_dotenv, and GCP_PROJECT_ID lives in .env like every other credential.
    # Reading the environment first would skip this on a machine that is
    # perfectly well configured.
    import app  # noqa: F401
    from app.infra.firestoreClient import firestoreClient
    from app.stores.usageStore import COLLECTION, FirestoreUsageStore

    if not os.environ.get("GCP_PROJECT_ID"):
        pytest.skip("GCP_PROJECT_ID is not set here, so the usage ledger cannot be read.")

    store = FirestoreUsageStore()
    messages = (
        firestoreClient()
        .collection(COLLECTION)
        .document(ingested)
        .collection("conversations")
        .document(conversation.conversationId)
        .collection("messages")
    )
    return {
        "messages": [snapshot.to_dict() for snapshot in messages.stream()],
        "conversation": asyncio.run(
            store.conversationTotal(ingested, conversation.conversationId)
        ),
        "project": asyncio.run(store.projectTotal(ingested)),
    }


def testEveryAnsweredTurnIsInTheLedger(ledger) -> None:
    """Twenty answers, twenty rows. A missing row is an answer nobody was
    billed for in the report, which is the failure mode that only ever shows up
    as an invoice nobody can reconcile."""
    assert len(ledger["messages"]) == 20, (
        f"{len(ledger['messages'])} usage rows for 20 turns: "
        f"{sorted(m.get('messageId') for m in ledger['messages'])}"
    )
    assert ledger["conversation"]["turns"] == 20


def testTheConversationTotalIsTheSumOfItsTurns(ledger) -> None:
    """The rollups are `Increment`, applied by the server, so this is checking
    that twenty concurrent-capable writes landed exactly once each."""
    fromMessages = sum(int(m.get("totalTokens") or 0) for m in ledger["messages"])

    assert ledger["conversation"]["totalTokens"] == fromMessages

    # The project total is *at least* this conversation, not equal to it. It
    # accumulates across every conversation the project has ever had, which is
    # the entire reason the level exists -- and with RAG_E2E_PROJECT_ID reusing
    # an ingested project, a second run legitimately finds the first run's
    # tokens still counted there. Asserting equality made a correct rollup look
    # like a double-count.
    assert ledger["project"]["totalTokens"] >= fromMessages


def testTheAgentAndTheReviewerAreBilledSeparately(ledger) -> None:
    """The entire reason usage is collected per role rather than per request.

    All three roles point at the same model by default, so a regression to one
    shared callback context would merge them into a single number and lose the
    breakdown -- silently, and with a total that still looked right. A reviewer
    line of zero is that regression.
    """
    roles = ledger["conversation"]["roles"]
    assert roles["agent"]["totalTokens"] > 0, "the agent was billed nothing"
    assert roles["reviewer"]["totalTokens"] > 0, (
        "the reviewer was billed nothing, though it runs on every question -- "
        "usage is being read off the return value again instead of the callback, "
        "and with_structured_output has already thrown the metadata away"
    )
    assert roles["agent"]["calls"] >= 20, (
        f"only {roles['agent']['calls']} agent calls for 20 turns"
    )
    assert roles["agent"]["provider"] and roles["agent"]["model"], (
        "the ledger recorded no provider or model, so the tokens cannot be priced"
    )


def testTheCostPerTurnIsReported(ledger, conversation: Transcript) -> None:
    """Not an assertion so much as the number the run exists to produce. It
    fails only if nothing was recorded at all."""
    total = ledger["conversation"]["totalTokens"]
    roles = ledger["conversation"]["roles"]

    print("\n\ntokens for twenty turns", flush=True)
    print(f"  total      {total:>8,}   ({total / 20:,.0f} per turn)", flush=True)
    for name, row in sorted(roles.items()):
        print(
            f"  {name:<10} {row['totalTokens']:>8,}   "
            f"{row['provider']}/{row['model']}  calls={row['calls']}",
            flush=True,
        )
    print(f"\n  cached in  {ledger['conversation'].get('cachedInputTokens', 0):>8,}", flush=True)
    print(f"  reasoning  {ledger['conversation'].get('reasoningTokens', 0):>8,}\n", flush=True)

    assert total > 0


def _cleanup(projectId: str) -> None:
    """Delete this run's Firestore documents. Opt-in, via RAG_E2E_CLEANUP.

    Off by default, and that is the deliberate choice: the run's whole value is
    that it produced real conversation history and a real usage ledger you can
    go and look at, and deleting them at the end of a green run throws that away.

    What this does not clean, and cannot cheaply: the Pinecone namespace holding
    ~220 vectors from the ingest. If runs are frequent enough for that to matter,
    the answer is RAG_E2E_PROJECT_ID -- ingest once, reuse it, and stop minting a
    namespace per run.
    """
    try:
        import app  # noqa: F401
        from app.infra.firestoreClient import firestoreClient
        from app.stores.conversationStore import COLLECTION as CONVERSATIONS
        from app.stores.projectStore import COLLECTION as PROJECTS
        from app.stores.usageStore import COLLECTION as USAGE

        db = firestoreClient()
        for conversation in (
            db.collection(CONVERSATIONS).document(projectId).collection("conversations").stream()
        ):
            for name in ("messages", "context"):
                for document in conversation.reference.collection(name).stream():
                    document.reference.delete()
            conversation.reference.delete()
        db.collection(CONVERSATIONS).document(projectId).delete()

        usage = db.collection(USAGE).document(projectId)
        for conversation in usage.collection("conversations").stream():
            for message in conversation.reference.collection("messages").stream():
                message.reference.delete()
            conversation.reference.delete()
        usage.delete()

        db.collection(PROJECTS).document(projectId).delete()
        print(f"\ncleaned up Firestore for '{projectId}'.", flush=True)
    except Exception as exc:
        print(f"\ncleanup of '{projectId}' failed: {exc}", flush=True)
