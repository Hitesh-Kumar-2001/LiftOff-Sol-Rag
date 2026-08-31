"""The one place a messaging gateway is registered.

Adding Facebook Messenger, Telegram, or anything else is two steps and no
edits anywhere else:

1. write ``app/channels/messenger.py`` with a class satisfying
   ``app.channels.channel.Channel`` -- ``verify``, ``parse``, ``send``, plus
   ``usesHandshake`` and ``handshake``;
2. add a line to ``CHANNELS`` below.

There is no step three. The route is a single parameterised path --
``/api/v1/conversations/{projectId}/{gateway}`` -- so a gateway becomes
reachable the moment it is registered here, and the URL to paste into the
platform's console follows from its name.

Nothing else knows the list. The store keys its config map on the same names
and the sender dispatches on them. ``getChannel`` raising for an unknown name
is what the route turns into a 404 that names the gateways that do exist,
which a path-per-platform shape could not do.

The adapters are stateless -- every call takes the project's config -- so one
instance each is built here and shared. There is nothing per-request in them.
"""

from __future__ import annotations

from app.channels.channel import Channel, ChannelError
from app.channels.line import LineChannel
from app.channels.whatsapp import WhatsAppChannel

CHANNELS: dict[str, Channel] = {
    WhatsAppChannel.name: WhatsAppChannel(),
    LineChannel.name: LineChannel(),
}


def getChannel(name: str) -> Channel:
    """The adapter for one channel, or an error naming the ones that exist."""
    channel = CHANNELS.get(name)
    if channel is None:
        raise ChannelError(
            f"Unknown channel '{name}'. Registered: {', '.join(channelNames()) or 'none'}."
        )
    return channel


def channelNames() -> list[str]:
    return sorted(CHANNELS)
