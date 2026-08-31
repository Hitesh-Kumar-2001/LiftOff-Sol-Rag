"""The gateway adapters: verify, parse, and the registry.

No network and no Firestore -- everything here is a pure function of bytes,
headers and a config dict, which is the whole point of the ``Channel`` protocol.
Sending is the one part that talks to a platform; it is covered in
tests/testChannelRoutes.py through the sender, with the HTTP call stubbed.

The signature tests matter more than their size suggests. This is the only
authentication in the service, and both of its failure modes are silent: a
check that passes everything looks exactly like a check that works, and a check
using the wrong encoding looks exactly like a wrong secret.
"""

import base64
import hashlib
import hmac
import json

import pytest

from app.channels.channel import ChannelError, IncomingMessage, hmacDigest, signaturesMatch
from app.channels.line import VERIFICATION_REPLY_TOKEN, LineChannel, trimForLine
from app.channels.registry import CHANNELS, channelNames, getChannel
from app.channels.whatsapp import WhatsAppChannel, trimForWhatsApp

WHATSAPP_SECRET = "meta-app-secret"
LINE_SECRET = "line-channel-secret"


def whatsappHeaders(secret: str, body: bytes) -> dict[str, str]:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"x-hub-signature-256": f"sha256={digest}"}


def lineHeaders(secret: str, body: bytes) -> dict[str, str]:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return {"x-line-signature": base64.b64encode(digest).decode()}


def whatsappBody(text: str = "What is the refund window?", sender: str = "447700900000") -> bytes:
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "messages": [
                                    {
                                        "from": sender,
                                        "id": "wamid.abc",
                                        "type": "text",
                                        "text": {"body": text},
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    ).encode()


def lineBody(text: str = "What is the refund window?", replyToken: str = "reply-1") -> bytes:
    return json.dumps(
        {
            "destination": "U000",
            "events": [
                {
                    "type": "message",
                    "replyToken": replyToken,
                    "source": {"type": "user", "userId": "Uabc123"},
                    "message": {"id": "msg-1", "type": "text", "text": text},
                }
            ],
        }
    ).encode()


# --- the shared signing helpers -------------------------------------------


def testTheTwoEncodingsDiffer() -> None:
    """Meta sends hex, LINE sends base64, from the same HMAC. Using one where
    the other belongs fails in a way that reads as a wrong secret."""
    body = b'{"a":1}'

    assert hmacDigest("s", body, "hex") != hmacDigest("s", body, "base64")


def testAnUnknownEncodingIsRefused() -> None:
    with pytest.raises(ChannelError):
        hmacDigest("s", b"{}", "rot13")


@pytest.mark.parametrize(
    "expected, received",
    [("abc", ""), ("", "abc"), ("", ""), ("abc", "abd")],
)
def testSignaturesDoNotMatch(expected, received) -> None:
    """A missing signature must never compare equal to a missing secret."""
    assert signaturesMatch(expected, received) is False


def testMatchingSignaturesMatch() -> None:
    assert signaturesMatch("abc", "abc") is True


# --- WhatsApp --------------------------------------------------------------


def testWhatsAppAcceptsItsOwnSignature() -> None:
    body = whatsappBody()

    assert WhatsAppChannel().verify(
        body, whatsappHeaders(WHATSAPP_SECRET, body), {"appSecret": WHATSAPP_SECRET}
    )


def testWhatsAppRejectsATamperedBody() -> None:
    """The signature is over the bytes, so changing the question invalidates it."""
    headers = whatsappHeaders(WHATSAPP_SECRET, whatsappBody())

    assert not WhatsAppChannel().verify(
        whatsappBody("something else"), headers, {"appSecret": WHATSAPP_SECRET}
    )


def testWhatsAppWithNoSecretRefusesEverything() -> None:
    """Fails closed. A gateway that accepts anything until somebody remembers to
    add a secret looks exactly like one that is working."""
    body = whatsappBody()

    assert not WhatsAppChannel().verify(body, whatsappHeaders(WHATSAPP_SECRET, body), {})


def testWhatsAppParsesATextMessage() -> None:
    messages = WhatsAppChannel().parse(json.loads(whatsappBody()), "handbook")

    assert len(messages) == 1
    assert messages[0].text == "What is the refund window?"
    assert messages[0].userId == "447700900000"
    assert messages[0].messageId == "wamid.abc"
    assert messages[0].channel == "whatsapp"
    assert messages[0].projectId == "handbook"


def testWhatsAppIgnoresDeliveryReceipts() -> None:
    """`statuses` arrives far more often than `messages` -- they are receipts for
    what we sent. Answering one would be answering ourselves."""
    payload = {
        "entry": [
            {"changes": [{"value": {"statuses": [{"id": "wamid.abc", "status": "read"}]}}]}
        ]
    }

    assert WhatsAppChannel().parse(payload, "handbook") == []


def testWhatsAppIgnoresNonTextMessages() -> None:
    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "447700900000", "type": "image"}]}}]}
        ]
    }

    assert WhatsAppChannel().parse(payload, "handbook") == []


def testWhatsAppHandshakeEchoesTheChallenge() -> None:
    params = {"hub.mode": "subscribe", "hub.verify_token": "tok", "hub.challenge": "12345"}

    assert WhatsAppChannel().handshake(params, {"verifyToken": "tok"}) == "12345"


def testWhatsAppHandshakeRefusesAWrongToken() -> None:
    params = {"hub.mode": "subscribe", "hub.verify_token": "guess", "hub.challenge": "12345"}

    assert WhatsAppChannel().handshake(params, {"verifyToken": "tok"}) is None


def testWhatsAppHandshakeRefusesWhenNoTokenIsConfigured() -> None:
    """Otherwise anyone who guessed the URL could point their own Meta app here."""
    params = {"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "12345"}

    assert WhatsAppChannel().handshake(params, {}) is None


def testWhatsAppTrimsAnOversizedAnswer() -> None:
    """4096 is a hard reject, so an untrimmed long answer is a failed delivery."""
    trimmed = trimForWhatsApp("x" * 6000)

    assert len(trimmed) <= 4096
    assert trimmed.endswith("[...]")


# --- LINE ------------------------------------------------------------------


def testLineAcceptsItsOwnSignature() -> None:
    body = lineBody()

    assert LineChannel().verify(
        body, lineHeaders(LINE_SECRET, body), {"channelSecret": LINE_SECRET}
    )


def testLineRejectsAHexSignature() -> None:
    """The exact confusion the shared helper exists to prevent: right secret,
    right algorithm, wrong encoding."""
    body = lineBody()
    hexDigest = hmac.new(LINE_SECRET.encode(), body, hashlib.sha256).hexdigest()

    assert not LineChannel().verify(
        body, {"x-line-signature": hexDigest}, {"channelSecret": LINE_SECRET}
    )


def testLineWithNoSecretRefusesEverything() -> None:
    body = lineBody()

    assert not LineChannel().verify(body, lineHeaders(LINE_SECRET, body), {})


def testLineParsesATextMessage() -> None:
    messages = LineChannel().parse(json.loads(lineBody()), "handbook")

    assert len(messages) == 1
    assert messages[0].text == "What is the refund window?"
    assert messages[0].userId == "Uabc123"
    assert messages[0].replyToken == "reply-1"


def testLineDropsTheVerificationPing() -> None:
    """The console's Verify button sends a real signed event with an all-zero
    reply token. Answering it would spend a model call on nobody."""
    payload = json.loads(lineBody(replyToken=VERIFICATION_REPLY_TOKEN))

    assert LineChannel().parse(payload, "handbook") == []


def testLineIgnoresNonMessageEvents() -> None:
    payload = {"events": [{"type": "follow", "source": {"userId": "Uabc"}}]}

    assert LineChannel().parse(payload, "handbook") == []


def testLineTrimsAnOversizedAnswer() -> None:
    trimmed = trimForLine("x" * 8000)

    assert len(trimmed) <= 5000
    assert trimmed.endswith("[...]")


# --- the registry ----------------------------------------------------------


def testBothGatewaysAreRegistered() -> None:
    assert channelNames() == ["line", "whatsapp"]


def testAnUnknownChannelNamesTheOnesThatExist() -> None:
    with pytest.raises(ChannelError) as raised:
        getChannel("telegram")

    for name in channelNames():
        assert name in str(raised.value)


@pytest.mark.parametrize("name", sorted(CHANNELS))
def testEveryRegisteredChannelSatisfiesTheProtocol(name) -> None:
    """The contract adding a gateway has to meet. A missing method would
    otherwise surface as an AttributeError inside a background task, where
    nobody is watching."""
    channel = getChannel(name)

    assert channel.name == name
    for method in ("verify", "parse", "send"):
        assert callable(getattr(channel, method, None))


# --- the normalised message ------------------------------------------------


def testThreadKeysAreScopedByChannel() -> None:
    """A LINE userId and a WhatsApp wa_id are unrelated id spaces. Keying on the
    user alone would let one collide with the other and hand somebody else's
    conversation to the wrong person."""
    common = {"projectId": "p", "userId": "123", "text": "hi", "messageId": "m"}

    line = IncomingMessage(channel="line", **common)
    whatsapp = IncomingMessage(channel="whatsapp", **common)

    assert line.threadKey != whatsapp.threadKey
