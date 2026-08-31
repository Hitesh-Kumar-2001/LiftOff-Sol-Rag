"""The webhook routes, the channel store, and the outbound sender.

Real Firestore for the store, like the rest of the suite. The agent is stubbed
-- these tests are about what happens *around* the answer -- and so is the HTTP
call to the platform, because sending is the one thing that cannot be exercised
without somebody else's API and a real credential.

What is worth pinning here, in order of how expensive it would be to get wrong:

* an unverified delivery is refused **before** anything is parsed or answered;
* the webhook is acknowledged without waiting for the answer;
* a redelivery does not produce a second answer;
* a person's conversation is continued rather than restarted every message.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import conversationRoutes
from app.channels.channel import ChannelError, IncomingMessage
from app.main import app
from app.stores.channelStore import FirestoreChannelStore, getChannelStore

WHATSAPP_SECRET = "meta-app-secret"
LINE_SECRET = "line-channel-secret"

PROJECT_ID = "unset"


@pytest.fixture(autouse=True)
def projectId(scratch) -> Iterator[str]:
    global PROJECT_ID
    PROJECT_ID = scratch.projectId("channel")
    yield PROJECT_ID
    PROJECT_ID = "unset"


@pytest.fixture
def channelStore(projectId) -> Iterator[FirestoreChannelStore]:
    store = FirestoreChannelStore()
    app.dependency_overrides[getChannelStore] = lambda: store
    yield store
    app.dependency_overrides.clear()


@pytest.fixture
def configured(channelStore) -> FirestoreChannelStore:
    """A project reachable on both gateways."""
    asyncio.run(
        channelStore.saveConfig(
            PROJECT_ID,
            "whatsapp",
            {
                "appSecret": WHATSAPP_SECRET,
                "accessToken": "token",
                "phoneNumberId": "1234",
                "verifyToken": "verify-me",
            },
        )
    )
    asyncio.run(
        channelStore.saveConfig(
            PROJECT_ID, "line", {"channelSecret": LINE_SECRET, "channelAccessToken": "token"}
        )
    )
    return channelStore


@pytest.fixture
def answered(monkeypatch) -> list[dict]:
    """Stub the agent. Records what it was asked, answers a fixed string."""
    asked: list[dict] = []

    async def fakeRunTurn(**kwargs):
        asked.append(kwargs)
        return "Thirty days from purchase.", kwargs.get("conversationId") or "conv-new"

    monkeypatch.setattr(conversationRoutes, "runTurn", fakeRunTurn)
    return asked


@pytest.fixture
def sent(monkeypatch) -> list[tuple]:
    """Stub the outbound send. Records (message, text)."""
    delivered: list[tuple] = []

    async def fakeSend(message, text, channels=None):
        delivered.append((message, text))
        return True

    monkeypatch.setattr(conversationRoutes, "sendReply", fakeSend)
    return delivered


@pytest.fixture
def client(configured, answered, sent) -> Iterator[TestClient]:
    with TestClient(app) as testClient:
        yield testClient


def newMessageId(prefix: str = "wamid") -> str:
    """A message id no other test has used.

    The dedupe in the route is keyed on (channel, messageId) in Redis, and
    conftest's fakeredis is session-scoped -- so a fixed id here would be
    claimed by the first test that used it and silently skipped by every test
    after. Which is the mechanism working correctly; it just has to be fed real
    platform behaviour, where ids are unique.
    """
    return f"{prefix}.{uuid.uuid4().hex[:12]}"


def whatsappBody(text: str = "What is the refund window?", messageId: str = "") -> bytes:
    messageId = messageId or newMessageId()
    return json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "447700900000",
                                        "id": messageId,
                                        "type": "text",
                                        "text": {"body": text},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    ).encode()


def lineBody(text: str = "What is the refund window?", messageId: str = "") -> bytes:
    messageId = messageId or newMessageId("msg")
    return json.dumps(
        {
            "events": [
                {
                    "type": "message",
                    "replyToken": "reply-1",
                    "source": {"userId": "Uabc123"},
                    "message": {"id": messageId, "type": "text", "text": text},
                }
            ]
        }
    ).encode()


def postWhatsApp(client: TestClient, body: bytes, secret: str = WHATSAPP_SECRET):
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        content=body,
        headers={
            "x-hub-signature-256": f"sha256={digest}",
            "content-type": "application/json",
        },
    )


def postLine(client: TestClient, body: bytes, secret: str = LINE_SECRET):
    digest = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return client.post(
        f"/api/v1/conversations/{PROJECT_ID}/line",
        content=body,
        headers={"x-line-signature": digest, "content-type": "application/json"},
    )


# --- verification ----------------------------------------------------------


def testAVerifiedDeliveryIsAccepted(client: TestClient, answered, sent) -> None:
    response = postWhatsApp(client, whatsappBody())

    assert response.status_code == 200
    assert response.json()["accepted"] == 1
    # TestClient runs background tasks before returning, so by here the answer
    # has been produced and handed to the sender.
    assert answered[0]["question"] == "What is the refund window?"
    assert sent[0][1] == "Thirty days from purchase."


def testAWrongSignatureIsRefusedBeforeAnythingIsAnswered(
    client: TestClient, answered, sent
) -> None:
    """The expensive half of the check: a forged delivery must not reach a
    model. Anyone can find this URL."""
    response = postWhatsApp(client, whatsappBody(), secret="not-the-secret")

    assert response.status_code == 403
    assert answered == []
    assert sent == []


def testAMissingSignatureIsRefused(client: TestClient, answered) -> None:
    response = client.post(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        content=whatsappBody(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 403
    assert answered == []


def testLineIsVerifiedIndependently(client: TestClient, answered, sent) -> None:
    response = postLine(client, lineBody())

    assert response.status_code == 200
    assert sent[0][0].channel == "line"
    assert sent[0][0].replyToken == "reply-1"


def testLineRefusesAWhatsAppSignature(client: TestClient, answered) -> None:
    """Each gateway checks against its own secret; one project's configuration
    is not interchangeable across platforms."""
    body = lineBody()
    digest = base64.b64encode(
        hmac.new(WHATSAPP_SECRET.encode(), body, hashlib.sha256).digest()
    ).decode()

    response = client.post(
        f"/api/v1/conversations/{PROJECT_ID}/line",
        content=body,
        headers={"x-line-signature": digest},
    )

    assert response.status_code == 403
    assert answered == []


def testAnUnconfiguredProjectIsA404(client: TestClient, scratch, answered) -> None:
    """Checked before the signature, because without a config there is no secret
    to check one against."""
    other = scratch.projectId("unconfigured")

    response = client.post(f"/api/v1/conversations/{other}/whatsapp", content=whatsappBody())

    assert response.status_code == 404
    assert answered == []


# --- the Meta handshake ----------------------------------------------------


def testTheHandshakeEchoesTheChallengeAsPlainText(client: TestClient) -> None:
    """Meta wants the bare string, not JSON, and will not deliver until it
    gets it."""
    response = client.get(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-me",
            "hub.challenge": "9876",
        },
    )

    assert response.status_code == 200
    assert response.text == "9876"


def testTheHandshakeRefusesAWrongToken(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "guess", "hub.challenge": "9876"},
    )

    assert response.status_code == 403


# --- delivery semantics ----------------------------------------------------


def testARedeliveryIsNotAnsweredTwice(client: TestClient, answered, sent) -> None:
    """Every one of these platforms retries. Without the dedupe a retry costs a
    second model call and sends the person the same answer again."""
    body = whatsappBody(messageId=newMessageId("redelivered"))

    first = postWhatsApp(client, body)
    second = postWhatsApp(client, body)

    assert first.json()["accepted"] == 1
    assert second.json()["accepted"] == 0
    assert len(answered) == 1
    assert len(sent) == 1


def testAVerifiedButUnreadableBodyIsAcknowledged(client: TestClient, answered) -> None:
    """A 200, not a 400. It was signed, so it really is from the platform -- and
    a non-2xx would have it redelivered forever for something no retry fixes."""
    body = b"not json at all"
    digest = hmac.new(WHATSAPP_SECRET.encode(), body, hashlib.sha256).hexdigest()

    response = client.post(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        content=body,
        headers={"x-hub-signature-256": f"sha256={digest}"},
    )

    assert response.status_code == 200
    assert answered == []


def testADeliveryWithNothingAnswerableIsAcknowledged(client: TestClient, answered) -> None:
    """Read receipts arrive constantly. They are accepted and ignored."""
    body = json.dumps(
        {"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]}
    ).encode()

    response = postWhatsApp(client, body)

    assert response.status_code == 200
    assert response.json()["accepted"] == 0
    assert answered == []


def testTheAnswerStillGoesOutWhenTheAgentFails(
    client: TestClient, monkeypatch, sent
) -> None:
    """Somebody is sitting in a chat app waiting. Silence is indistinguishable
    from the service being down, so a failure gets a reply too."""

    async def broken(**kwargs):
        raise RuntimeError("the provider is down")

    monkeypatch.setattr(conversationRoutes, "runTurn", broken)

    response = postWhatsApp(client, whatsappBody())

    assert response.status_code == 200
    assert sent[0][1] == conversationRoutes.FAILURE_REPLY


def testAnOversizedWebhookBodyIsRefusedUnread(client: TestClient, answered) -> None:
    """These URLs are public and the body is read into memory *before* the
    signature over it can be checked -- the check comes after the read, so no
    amount of signature verification helps. Streamed with a cap instead, so the
    memory is never paid."""
    huge = b"x" * (conversationRoutes.MAX_WEBHOOK_BODY_BYTES + 1024)

    response = client.post(
        f"/api/v1/conversations/{PROJECT_ID}/whatsapp",
        content=huge,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert answered == []


def testABodyInsideTheCapStillWorks(client: TestClient, answered, sent) -> None:
    """The cap has three orders of magnitude of headroom over a real delivery;
    a long question must not trip it."""
    response = postWhatsApp(client, whatsappBody(text="why? " * 2000))

    assert response.status_code == 200
    assert len(answered) == 1


# --- conversation continuity ----------------------------------------------


def testAPersonsConversationIsRemembered(
    client: TestClient, channelStore, answered, sent
) -> None:
    """Otherwise every message from the same number starts a new conversation
    and the agent has no memory at all between them."""
    postWhatsApp(client, whatsappBody())

    linked = asyncio.run(channelStore.conversationFor(PROJECT_ID, "whatsapp:447700900000"))
    assert linked == "conv-new"


def testTheNextMessageContinuesThatConversation(
    client: TestClient, answered, sent
) -> None:
    postWhatsApp(client, whatsappBody())

    postWhatsApp(client, whatsappBody(text="And gift cards?"))

    assert answered[1]["conversationId"] == "conv-new"


def testAConversationThatHasGoneIsRestartedRatherThanFailing(
    client: TestClient, channelStore, monkeypatch, sent
) -> None:
    """A linked conversation expires after RAG_CONVERSATION_TTL_SECONDS. Telling
    somebody on WhatsApp that their conversation expired is not an answer."""
    from fastapi import HTTPException

    asyncio.run(
        channelStore.linkConversation(PROJECT_ID, "whatsapp:447700900000", "long-gone")
    )

    attempts: list[str | None] = []

    async def fussy(**kwargs):
        attempts.append(kwargs.get("conversationId"))
        if kwargs.get("conversationId") == "long-gone":
            raise HTTPException(status_code=404, detail="no such conversation")
        return "Thirty days from purchase.", "conv-fresh"

    monkeypatch.setattr(conversationRoutes, "runTurn", fussy)

    response = postWhatsApp(client, whatsappBody())

    assert response.status_code == 200
    assert attempts == ["long-gone", None]
    assert sent[0][1] == "Thirty days from purchase."


def testAConversationIsNotLinkedWhenDeliveryFailed(
    client: TestClient, channelStore, answered, monkeypatch
) -> None:
    """A conversation they never got an answer from is one they will ask again
    from. Linking it would leave the model believing it already replied."""

    async def failed(message, text, channels=None):
        return False

    monkeypatch.setattr(conversationRoutes, "sendReply", failed)

    postWhatsApp(client, whatsappBody())

    assert asyncio.run(channelStore.conversationFor(PROJECT_ID, "whatsapp:447700900000")) is None


# --- the store -------------------------------------------------------------


def testConfigForReturnsOneGateway(configured, channelStore) -> None:
    config = asyncio.run(channelStore.configFor(PROJECT_ID, "whatsapp"))

    assert config["appSecret"] == WHATSAPP_SECRET
    assert config["phoneNumberId"] == "1234"


def testAnUnconfiguredGatewayIsNone(configured, channelStore) -> None:
    assert asyncio.run(channelStore.configFor(PROJECT_ID, "messenger")) is None


def testAProjectWithNoDocumentHasNoChannels(channelStore, scratch) -> None:
    assert asyncio.run(channelStore.allChannels(scratch.projectId("empty"))) == {}


def testSavingOneGatewayLeavesTheOthersAlone(configured, channelStore) -> None:
    """The merge that makes adding a gateway safe. A plain set() here would
    silently take the project off WhatsApp the moment LINE was configured."""
    asyncio.run(channelStore.saveConfig(PROJECT_ID, "line", {"channelSecret": "rotated"}))

    channels = asyncio.run(channelStore.allChannels(PROJECT_ID))
    assert set(channels) == {"whatsapp", "line"}
    assert channels["whatsapp"]["appSecret"] == WHATSAPP_SECRET
    assert channels["line"]["channelSecret"] == "rotated"


# --- the sender ------------------------------------------------------------


def message(channel: str = "whatsapp") -> IncomingMessage:
    return IncomingMessage(
        channel=channel,
        projectId=PROJECT_ID,
        userId="447700900000",
        text="hi",
        messageId="wamid.1",
    )


def testTheSenderDispatchesToTheRightGateway(configured, channelStore, monkeypatch) -> None:
    from app.channels import sender as senderModule

    calls: list[tuple] = []

    async def fakeSend(msg, text, config):
        calls.append((msg.channel, text, config["phoneNumberId"]))

    monkeypatch.setattr(senderModule.getChannel("whatsapp"), "send", fakeSend)

    assert asyncio.run(senderModule.sendReply(message(), "an answer", channelStore))
    assert calls == [("whatsapp", "an answer", "1234")]


def testTheSenderReportsAFailureRatherThanRaising(
    configured, channelStore, monkeypatch
) -> None:
    """There is no caller left to raise to -- the webhook returned 200 seconds
    ago. The return value is how a caller learns the answer never landed."""
    from app.channels import sender as senderModule

    async def broken(msg, text, config):
        raise ChannelError("the platform said no")

    monkeypatch.setattr(senderModule.getChannel("whatsapp"), "send", broken)

    assert asyncio.run(senderModule.sendReply(message(), "an answer", channelStore)) is False


def testTheSenderRefusesToSendNothing(configured, channelStore) -> None:
    assert asyncio.run(conversationRoutes.sendReply(message(), "   ", channelStore)) is False


def testTheSenderStopsWhenTheGatewayWasRemoved(channelStore, scratch) -> None:
    """Config is re-read at send time, not carried from the webhook: minutes can
    pass on a slow answer, and the credentials that matter are the current ones."""
    from app.channels import sender as senderModule

    gone = IncomingMessage(
        channel="whatsapp",
        projectId=scratch.projectId("removed"),
        userId="447700900000",
        text="hi",
        messageId="wamid.1",
    )

    assert asyncio.run(senderModule.sendReply(gone, "an answer", channelStore)) is False
