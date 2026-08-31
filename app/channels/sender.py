"""The single outbound path: an answer, on its way back to a person.

The one thing a web request never needs. ``/query`` hands its answer straight
back on the HTTP response it arrived on; a message from WhatsApp or LINE was
acknowledged long before the answer existed, so delivery is a second, separate
call to somebody else's API -- and that call can fail on its own, long after the
model has been paid for.

This module is where "an answer was produced" becomes "an answer was
delivered", for every gateway. The routes call ``sendReply`` and know nothing
about tokens, URLs, or which platform they are talking to.

**It does not raise.** A delivery failure is logged and reported as ``False``.
There is no caller left to fail: the webhook returned 200 several seconds ago
and the platform has moved on, so an exception here would climb into a
background task and be swallowed anyway -- but silently, which is the one
outcome worth avoiding. The return value exists so the caller can record that
the answer never landed.
"""

from __future__ import annotations

import logging

from app.channels.channel import ChannelError, IncomingMessage
from app.channels.registry import getChannel
from app.stores.channelStore import ChannelStore, getChannelStore

logger = logging.getLogger(__name__)


async def sendReply(
    message: IncomingMessage, text: str, channels: ChannelStore | None = None
) -> bool:
    """Deliver ``text`` to whoever sent ``message``. True if it landed.

    The config is re-read here rather than carried from the webhook, and
    deliberately: minutes can pass between the two on a slow answer, and the
    credentials that matter are the ones that are current when the send is
    actually made. It is one cached Firestore read.
    """
    if not text.strip():
        # Nothing to say. Reached when the agent produced no text at all -- see
        # NO_ANSWER in app.agent.agent, which normally prevents this.
        logger.warning(
            "Nothing to send to %s on '%s'; the answer was empty.",
            message.threadKey,
            message.projectId,
        )
        return False

    store = channels or getChannelStore()

    try:
        config = await store.configFor(message.projectId, message.channel)
    except Exception:
        logger.exception(
            "Could not read the %s configuration for project '%s'; the answer "
            "cannot be delivered.",
            message.channel,
            message.projectId,
        )
        return False

    if not config:
        # The webhook was verified against a config that existed, so reaching
        # here means it was removed mid-answer. Rare, and worth saying plainly.
        logger.error(
            "Project '%s' has no %s configuration any more; dropping the answer "
            "for %s.",
            message.projectId,
            message.channel,
            message.threadKey,
        )
        return False

    try:
        await getChannel(message.channel).send(message, text, config)
    except ChannelError:
        logger.exception(
            "Could not deliver the answer to %s on project '%s'.",
            message.threadKey,
            message.projectId,
        )
        return False
    except Exception:
        # A socket timeout, a DNS failure, a platform outage. Same outcome for
        # the person waiting, and the same non-event for this process.
        logger.exception(
            "Unexpected failure delivering to %s on project '%s'.",
            message.threadKey,
            message.projectId,
        )
        return False

    logger.info(
        "Delivered an answer to %s on project '%s' (%d chars).",
        message.threadKey,
        message.projectId,
        len(text),
    )
    return True
