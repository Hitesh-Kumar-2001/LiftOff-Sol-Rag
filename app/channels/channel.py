"""What every messaging gateway has to look like from the outside.

A **channel** is one messaging platform this service can be reached through --
WhatsApp, LINE, and whatever is added next. The word is deliberate: these are
two-way, so "sender" would name only half of one, and ``provider`` is already
taken in this codebase for LLM vendors (``app.agent.llmManager``). Confusing
those two would be expensive, because a stack trace naming "provider" would no
longer say which kind.

Three operations, and adding a gateway means writing them once::

    verify(body, headers, config)   is this really from the platform?
    parse(payload, projectId)       what did the person actually say?
    send(message, text, config)     put the answer back in front of them

Everything else -- the route, the ack, the agent, the conversation, the
delivery failure handling -- is shared, and lives in ``app.api.conversationRoutes``
and ``app.channels.sender``. See ``app.channels.registry`` for the one place a
new gateway is registered.

Why the three are one Protocol and not three
--------------------------------------------
They share the platform's credentials, its base URL, and its id formats. A
WhatsApp access token is used by ``send`` and its app secret by ``verify``, and
both come out of the same config document; splitting them across files would
mean two places to touch per gateway and two chances to leave one half behind.
The user's instinct to keep sending separate is honoured in
``app.channels.sender``, which is the single outbound entry point that the
routes call -- what is *per platform* stays together, what is *shared* is
factored out.

**The raw request body matters.** Every platform signs the bytes it sent, so
``verify`` takes ``bytes`` and never a parsed dict: re-serialising JSON changes
key order and whitespace, and the signature then fails for reasons that look
like a wrong secret.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ChannelError(Exception):
    """A gateway could not be used: bad configuration, or a failed delivery."""


@dataclass(frozen=True)
class IncomingMessage:
    """One thing one person said, normalised across platforms.

    ``replyToken`` is LINE's and nothing else's -- a short-lived token that
    makes a reply free and which expires in about a minute. It is carried here
    rather than hidden inside the LINE adapter because the answer is produced
    *after* the webhook has been acknowledged, so the token has to survive the
    trip; see ``app.api.conversationRoutes``.

    ``messageId`` is the platform's own id for the message. Every one of these
    platforms redelivers on a non-2xx, and some redeliver anyway, so it is what
    makes "have I already answered this?" answerable.
    """

    channel: str
    projectId: str
    userId: str
    text: str
    messageId: str
    replyToken: str = ""
    # LINE says outright when it is retrying (``deliveryContext.isRedelivery``);
    # WhatsApp does not, and leaves it False. Carried for the log rather than
    # for a decision: a redelivery means the platform did not get a 2xx last
    # time, which usually means the answer was never queued -- so it is exactly
    # the case that *should* be answered. The Redis claim in the route is what
    # actually prevents a double answer.
    isRedelivery: bool = False

    @property
    def threadKey(self) -> str:
        """What this person's conversation is filed under.

        Scoped by channel as well as user id, because the two id spaces are
        unrelated: a LINE userId and a WhatsApp wa_id could collide, and the
        person behind them is not the same person.
        """
        return f"{self.channel}:{self.userId}"


class Channel(Protocol):
    """One messaging platform. Stateless -- the config arrives per call."""

    name: str

    # Whether this platform verifies its webhook URL with a GET before it will
    # deliver anything. Meta does; LINE sends a real signed POST instead. A flag
    # rather than inferring it from ``handshake`` returning None, because the
    # route has to tell "this gateway has no handshake" (405) from "the token
    # was wrong" (403), and one return value cannot say both.
    usesHandshake: bool

    def verify(self, body: bytes, headers: Mapping[str, str], config: Mapping[str, Any]) -> bool:
        """Whether this request really came from the platform.

        **This is the only authentication anywhere in this service.** Every
        other endpoint takes an unverified ``serverId`` and trusts it. A webhook
        is different in kind: it is a public URL that anyone can find and POST
        to, and what arrives is fed to a model and answered at this deployment's
        expense, to a person who did not ask.

        Must return False when the secret is missing, never True. "Not
        configured" has to fail closed -- a gateway that accepts everything
        until somebody remembers to add a secret is worse than one that accepts
        nothing, because it looks like it is working.
        """
        ...

    def parse(self, payload: Mapping[str, Any], projectId: str) -> list[IncomingMessage]:
        """The text messages in one webhook delivery, in order.

        A list, because these platforms batch: LINE sends an ``events`` array
        and WhatsApp nests messages under ``entry[].changes[]``. Anything that
        is not a text message from a person -- a delivery receipt, a read
        receipt, a sticker, a platform verification ping -- is dropped here
        rather than reaching the agent.
        """
        ...

    def handshake(self, params: Mapping[str, str], config: Mapping[str, Any]) -> str | None:
        """The body to echo for a URL-verification GET, or None to refuse it.

        Only meaningful when ``usesHandshake`` is True. Refuse when no verifying
        secret is configured -- otherwise anyone who guessed the URL could point
        their own app at this deployment.
        """
        ...

    async def send(
        self, message: IncomingMessage, text: str, config: Mapping[str, Any]
    ) -> None:
        """Deliver ``text`` back to whoever sent ``message``. Raises on failure."""
        ...


def hmacDigest(secret: str, body: bytes, encoding: str) -> str:
    """HMAC-SHA256 of the raw body, in the encoding the platform expects.

    Shared because both gateways implemented so far sign the same way and
    differ only in how they spell the result -- Meta sends hex, LINE sends
    base64. Getting this wrong fails in a way that is very hard to read: a
    correct secret with the wrong encoding looks exactly like a wrong secret.
    """
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    if encoding == "hex":
        return digest.hexdigest()
    if encoding == "base64":
        import base64

        return base64.b64encode(digest.digest()).decode("ascii")
    raise ChannelError(f"Unknown signature encoding '{encoding}'.")


def signaturesMatch(expected: str, received: str) -> bool:
    """Constant-time comparison, and False for anything missing.

    ``compare_digest`` rather than ``==`` because a plain comparison returns as
    soon as two bytes differ, and the time it took is a measurement of how much
    of the signature was right -- enough, over many requests, to reconstruct a
    valid one byte by byte.
    """
    if not expected or not received:
        return False
    return hmac.compare_digest(expected, received)
