"""LINE, through the LINE Messaging API.

Config document (``ragChannels/{projectId}.channels.line``)::

    channelSecret        signs every webhook; from the LINE Developers console
    channelAccessToken   a Bearer token for the reply and push endpoints

Two things about this platform that shape the code:

**A reply token is free, and it expires.** LINE gives every inbound event a
``replyToken`` that can be used once, within about a minute, at no cost against
the monthly message quota. A push message has no token, works any time, and is
metered. Answering a RAG question can take tens of seconds, so the token is
often still valid and sometimes not -- ``send`` tries the reply first and falls
back to push, which is the only way to be both cheap and reliable.

**LINE verifies a webhook by sending a real delivery.** The console's "Verify"
button POSTs a properly signed event whose ``replyToken`` is all zeroes.
Replying to it fails, so it is dropped in ``parse``: a verification ping is not
somebody asking a question, and answering it would spend a model call on nobody.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx

from app.channels.channel import (
    ChannelError,
    IncomingMessage,
    hmacDigest,
    signaturesMatch,
)

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "x-line-signature"

REPLY_URL = "https://api.line.me/v2/bot/message/reply"
PUSH_URL = "https://api.line.me/v2/bot/message/push"

# The token LINE sends when the console's "Verify" button is pressed. A real
# event never has this value.
VERIFICATION_REPLY_TOKEN = "0" * 32

# LINE truncates a text message over 5000 characters. Cut here instead, with a
# marker, so it is visible that something was removed.
MAX_TEXT_CHARS = 5000
TRUNCATION_MARKER = "\n\n[...]"

SEND_TIMEOUT_SECONDS = 15.0


class LineChannel:
    name = "line"
    # LINE has no GET challenge: the console's "Verify" button sends a real,
    # signed POST whose reply token is all zeroes, which `parse` drops.
    usesHandshake = False

    def handshake(self, params, config) -> None:
        """Never. Present so the protocol is satisfied without a special case."""
        return None

    def verify(self, body: bytes, headers: Mapping[str, str], config: Mapping[str, Any]) -> bool:
        secret = str(config.get("channelSecret") or "").strip()
        if not secret:
            logger.warning(
                "No channelSecret configured for LINE; refusing the delivery. A "
                "webhook that cannot be verified must not be answered."
            )
            return False

        # LINE sends the base64 digest bare -- no algorithm prefix, unlike Meta.
        return signaturesMatch(
            hmacDigest(secret, body, "base64"), str(headers.get(SIGNATURE_HEADER) or "")
        )

    def parse(self, payload: Mapping[str, Any], projectId: str) -> list[IncomingMessage]:
        messages: list[IncomingMessage] = []

        for event in payload.get("events") or []:
            event = event or {}
            if event.get("type") != "message":
                # follow, unfollow, join, postback, delivery receipts. None of
                # them is a question.
                continue

            replyToken = str(event.get("replyToken") or "")
            if replyToken == VERIFICATION_REPLY_TOKEN:
                logger.info("LINE webhook verification ping for project '%s'.", projectId)
                continue

            payloadMessage = event.get("message") or {}
            if payloadMessage.get("type") != "text":
                continue

            text = str(payloadMessage.get("text") or "").strip()
            userId = str((event.get("source") or {}).get("userId") or "").strip()
            if not text or not userId:
                continue

            messages.append(
                IncomingMessage(
                    channel=self.name,
                    projectId=projectId,
                    # LINE's only durable identity. Scoped to this Official
                    # Account, so the same person messaging two of them is two
                    # different users -- which happens to match how projects are
                    # keyed here. There is no thread or conversation id in the
                    # payload at all; see app.stores.channelStore for what
                    # stands in for one.
                    userId=userId,
                    text=text,
                    messageId=str(payloadMessage.get("id") or ""),
                    replyToken=replyToken,
                    isRedelivery=bool(
                        (event.get("deliveryContext") or {}).get("isRedelivery")
                    ),
                )
            )

        return messages

    async def send(
        self, message: IncomingMessage, text: str, config: Mapping[str, Any]
    ) -> None:
        token = str(config.get("channelAccessToken") or "").strip()
        if not token:
            raise ChannelError(
                f"LINE needs a channelAccessToken to send; project "
                f"'{message.projectId}' has none."
            )

        headers = {"Authorization": f"Bearer {token}"}
        body = [{"type": "text", "text": trimForLine(text)}]

        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            if message.replyToken:
                response = await client.post(
                    REPLY_URL,
                    headers=headers,
                    json={"replyToken": message.replyToken, "messages": body},
                )
                if response.status_code < 400:
                    return
                # Almost always an expired token: the answer took longer than
                # the ~1 minute LINE allows. Worth an info line rather than an
                # error, because the push below is the designed outcome and not
                # a degradation.
                logger.info(
                    "LINE reply token for project '%s' was not usable (%s); pushing instead.",
                    message.projectId,
                    response.status_code,
                )

            response = await client.post(
                PUSH_URL, headers=headers, json={"to": message.userId, "messages": body}
            )

        if response.status_code >= 400:
            logger.error(
                "LINE rejected a send for project '%s': %s %s",
                message.projectId,
                response.status_code,
                response.text[:500],
            )
            raise ChannelError(f"LINE send failed with {response.status_code}; see the log.")


def trimForLine(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
