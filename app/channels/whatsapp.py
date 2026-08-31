"""WhatsApp, through Meta's WhatsApp Cloud API.

Config document (``ragChannels/{projectId}.channels.whatsapp``)::

    appSecret       signs every webhook; from the Meta app's Basic Settings
    accessToken     a Bearer token for the Graph API send call
    phoneNumberId   which number is replying -- part of the send URL
    verifyToken     a string you choose, echoed during the GET handshake

Nothing here has a default. A missing ``appSecret`` makes ``verify`` return
False rather than skipping the check, so a half-configured project accepts no
traffic at all; see the note in ``app.channels.channel.Channel.verify``.

Two things about this platform that shape the code:

**It retries.** A non-2xx, or a response that takes longer than a handful of
seconds, is redelivered -- and after enough failures Meta disables the webhook
for the app. So the route acknowledges before the answer exists, and
``messageId`` is what lets a redelivery be recognised rather than answered
twice.

**The GET handshake is separate.** Meta verifies a webhook URL by GETting it
with ``hub.challenge`` and expects that value echoed back as plain text. It is
not signed -- there is no body to sign -- which is what ``verifyToken`` is for.
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

SIGNATURE_HEADER = "x-hub-signature-256"
GRAPH_VERSION = "v21.0"

# WhatsApp rejects a text body over 4096 characters outright, and an answer that
# long is a failed delivery rather than a long answer. Truncated with a marker
# so the person can see something was cut rather than reading a sentence that
# stops mid-word.
MAX_TEXT_CHARS = 4096
TRUNCATION_MARKER = "\n\n[...]"

SEND_TIMEOUT_SECONDS = 15.0


class WhatsAppChannel:
    name = "whatsapp"
    # Meta will not deliver anything until the URL answers a GET challenge.
    usesHandshake = True

    def verify(self, body: bytes, headers: Mapping[str, str], config: Mapping[str, Any]) -> bool:
        secret = str(config.get("appSecret") or "").strip()
        if not secret:
            logger.warning(
                "No appSecret configured for WhatsApp; refusing the delivery. A "
                "webhook that cannot be verified must not be answered."
            )
            return False

        # Meta sends "sha256=<hex>". The prefix is part of the header value, not
        # of the digest.
        received = str(headers.get(SIGNATURE_HEADER) or "")
        if received.startswith("sha256="):
            received = received[len("sha256=") :]

        return signaturesMatch(hmacDigest(secret, body, "hex"), received)

    def handshake(self, params: Mapping[str, str], config: Mapping[str, Any]) -> str | None:
        """The GET challenge, or None when it should be refused.

        Returns the challenge string to echo. Refuses when the token does not
        match or when none is configured -- otherwise anyone who guessed the URL
        could point their own Meta app at this deployment.
        """
        expected = str(config.get("verifyToken") or "").strip()
        if not expected:
            logger.warning("No verifyToken configured for WhatsApp; refusing the handshake.")
            return None

        if params.get("hub.mode") != "subscribe":
            return None
        if not signaturesMatch(expected, str(params.get("hub.verify_token") or "")):
            logger.warning("WhatsApp handshake presented the wrong verify token.")
            return None
        return str(params.get("hub.challenge") or "")

    def parse(self, payload: Mapping[str, Any], projectId: str) -> list[IncomingMessage]:
        """Text messages out of Meta's nesting.

        The shape is ``entry[].changes[].value.messages[]``, and ``value`` also
        carries ``statuses`` -- delivery and read receipts for messages *we*
        sent. Those arrive far more often than actual messages and must not be
        answered, which is why this filters on the message type rather than
        taking whatever is there.
        """
        messages: list[IncomingMessage] = []

        for entry in payload.get("entry") or []:
            for change in (entry or {}).get("changes") or []:
                value = (change or {}).get("value") or {}
                for raw in value.get("messages") or []:
                    if (raw or {}).get("type") != "text":
                        # Images, stickers, locations, button replies. Nothing
                        # here can read them yet, and answering a sticker with a
                        # RAG answer is worse than staying quiet.
                        continue
                    text = ((raw.get("text") or {}).get("body") or "").strip()
                    sender = str(raw.get("from") or "").strip()
                    if not text or not sender:
                        continue
                    messages.append(
                        IncomingMessage(
                            channel=self.name,
                            projectId=projectId,
                            userId=sender,
                            text=text,
                            messageId=str(raw.get("id") or ""),
                        )
                    )

        return messages

    async def send(
        self, message: IncomingMessage, text: str, config: Mapping[str, Any]
    ) -> None:
        token = str(config.get("accessToken") or "").strip()
        phoneNumberId = str(config.get("phoneNumberId") or "").strip()
        if not token or not phoneNumberId:
            raise ChannelError(
                "WhatsApp needs both accessToken and phoneNumberId to send; "
                f"project '{message.projectId}' has "
                f"{'no token' if not token else 'no phoneNumberId'}."
            )

        body = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": message.userId,
            "type": "text",
            "text": {"body": trimForWhatsApp(text)},
        }

        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"https://graph.facebook.com/{GRAPH_VERSION}/{phoneNumberId}/messages",
                headers={"Authorization": f"Bearer {token}"},
                json=body,
            )

        if response.status_code >= 400:
            # The body is logged, not raised: Meta's errors quote the request
            # back, and the request contains the answer, which is this project's
            # content. The exception says what failed and where to look.
            logger.error(
                "WhatsApp rejected a send for project '%s': %s %s",
                message.projectId,
                response.status_code,
                response.text[:500],
            )
            raise ChannelError(
                f"WhatsApp send failed with {response.status_code}; see the log."
            )


def trimForWhatsApp(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
