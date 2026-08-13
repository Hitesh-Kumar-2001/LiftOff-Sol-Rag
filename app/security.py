"""Verification of the calling server.

The whole credential store is held in RAM. A request is therefore a dict lookup
plus one digest compare -- no I/O, and nothing a caller sends can make the
process talk to the store.

Whether that copy is re-read on a timer is the store's call, not this module's:
a file on one node cannot change without a restart, a shared database can change
at any moment. See ``refresh_interval`` in ``app.credentials``.
"""

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from app.credentials import CredentialSource, ServerCredential, build_credential_source

logger = logging.getLogger(__name__)

# Compared against when the serverId is unknown, so a lookup miss costs the same
# as a wrong secret.
_UNMATCHABLE = ServerCredential(
    server_id="", secret_hash=hashlib.sha256(os.urandom(32)).hexdigest()
)


class AuthenticationError(Exception):
    """The serverId / serverSecret pair could not be verified."""


class ServerRegistry:
    """The in-memory copy of the credential store."""

    def __init__(self, source: CredentialSource) -> None:
        self._source = source
        self._by_id: dict[str, ServerCredential] = {}

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def refresh_interval(self) -> float | None:
        """Seconds between re-reads, or None if the store cannot change."""
        return self._source.refresh_interval

    async def load_all(self) -> int:
        """Replace the in-memory copy with the current contents of the store.

        Deliberately lock-free: it builds a new dict and rebinds it in one
        atomic step, so a concurrent request sees either the old map or the new
        one, never a half-filled one. A raised exception leaves the previous
        copy untouched.
        """
        credentials = await self._source.load_all()
        self._by_id = {credential.server_id: credential for credential in credentials}
        return len(self._by_id)

    async def refresh_forever(self, interval: float) -> None:
        """Re-read the store every ``interval`` seconds until cancelled.

        A failed read is logged and skipped: the last known-good copy keeps
        serving traffic and the next tick tries again. The loop must outlive any
        store outage -- if this task dies, credentials silently freeze.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                count = await self.load_all()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Credential refresh failed; keeping %d cached.", len(self))
            else:
                logger.debug("Refreshed %d server credentials.", count)

    def authenticate(self, server_id: str, server_secret: str) -> ServerCredential:
        """Return the credential for the pair, or raise ``AuthenticationError``.

        Memory is the only authority here. Nothing in this path does I/O, so a
        flood of bad credentials costs a hash each and never reaches the store.
        """
        credential = self._by_id.get(server_id)
        # The compare runs even on a miss, so an unknown serverId and a wrong
        # secret are not distinguishable by timing.
        if (credential or _UNMATCHABLE).matches(server_secret) and credential is not None:
            return credential
        raise AuthenticationError("Unknown serverId or serverSecret.")


@asynccontextmanager
async def refreshing(registry: ServerRegistry) -> AsyncIterator[None]:
    """Keep ``registry`` current for as long as the block runs.

    A store that cannot change gets no task at all -- the point of asking the
    source rather than polling everything unconditionally.
    """
    interval = registry.refresh_interval
    if interval is None:
        logger.info("Credential store is static; no refresh loop started.")
        yield
        return

    logger.info("Refreshing credentials every %gs.", interval)
    poller = asyncio.create_task(registry.refresh_forever(interval))
    try:
        yield
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller


# One registry for the life of the process. Construction touches nothing; the
# store is read at startup (see the lifespan hook in app.main).
SERVER_REGISTRY = ServerRegistry(build_credential_source())


def get_server_registry() -> ServerRegistry:
    """FastAPI dependency: the process-wide registry."""
    return SERVER_REGISTRY
