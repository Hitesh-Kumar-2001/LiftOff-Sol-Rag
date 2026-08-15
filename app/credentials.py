"""Where server credentials come from.

``app.security`` keeps credentials in RAM. Everything about *storage* lives
behind ``CredentialSource``, so swapping a local file for Firestore -- or
anything else -- touches this file only.

Each source also says whether it needs re-reading. A file next to a single node
cannot change without someone restarting that node, so there is nothing to poll.
A shared database can change at any moment, from another process, and must be.

Secrets are expected to be high-entropy random strings (API keys, not
human-chosen passwords), so a plain SHA-256 digest is enough: there is nothing
to brute force offline even if the digest leaks.
"""

import hashlib
import hmac
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ENV_SERVER_CREDENTIALS = "RAG_SERVER_CREDENTIALS"
ENV_CREDENTIALS_FILE = "RAG_CREDENTIALS_FILE"

# What a shared store should use when it has no opinion of its own. This is the
# window in which a revoked server is still served, so it is the number to
# argue about.
DEFAULT_REFRESH_INTERVAL_SECONDS = 30.0


class CredentialConfigError(Exception):
    """The credential store is unreadable or holds something malformed."""


def hashSecret(secret: str) -> str:
    """Return the digest stored for ``secret``."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ServerCredential:
    serverId: str
    secretHash: str

    def matches(self, secret: str) -> bool:
        """Constant-time check of ``secret`` against the stored digest."""
        return hmac.compare_digest(self.secretHash, hashSecret(secret))


class CredentialSource(Protocol):
    """What the registry needs from a credential store: one read, plus whether
    that read has to be repeated.

    A Firestore adapter is the whole of it -- a collection of documents keyed by
    ``serverId``, re-read on a timer because other processes can change it::

        class FirestoreCredentialSource:
            refreshInterval = DEFAULT_REFRESH_INTERVAL_SECONDS

            async def loadAll(self):
                return [
                    _toCredential(doc.id, doc.to_dict())
                    async for doc in self._collection.stream()
                ]
    """

    # Seconds between re-reads, or None if the store cannot change while the
    # process runs -- in which case no refresh loop is started at all.
    refreshInterval: float | None

    async def loadAll(self) -> Iterable[ServerCredential]:
        """Every credential in the store, as it stands right now."""
        ...


def parseCredentials(raw: str, origin: str = "credential document") -> list[ServerCredential]:
    """Parse the JSON credential document.

    ```json
    {
      "billing-service": {"secret": "plaintext-api-key"},
      "support-bot":     {"secretSha256": "9f86d0818884..."}
    }
    ```

    ``secretSha256`` is preferred in production; ``secret`` is accepted so a
    freshly generated key can be dropped in as-is. ``origin`` names the store in
    error messages, so a bad document points at the thing to go fix.
    """
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialConfigError(f"{origin} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise CredentialConfigError(f"{origin} must be a JSON object.")

    return [_toCredential(serverId, entry) for serverId, entry in document.items()]


def _toCredential(serverId: str, entry: object) -> ServerCredential:
    """Build a credential from one store record, whatever the store."""
    if not isinstance(entry, dict):
        raise CredentialConfigError(f"Credential for '{serverId}' must be an object.")

    secretHash = entry.get("secretSha256")
    if secretHash is None:
        secret = entry.get("secret")
        if not secret:
            raise CredentialConfigError(
                f"Credential for '{serverId}' needs a 'secret' or 'secretSha256'."
            )
        secretHash = hashSecret(secret)

    return ServerCredential(serverId=serverId, secretHash=str(secretHash))


class FileCredentialSource:
    """Credentials from a JSON file on disk.

    Static: one node, one file, read at startup. Editing the file takes effect
    on the next restart -- there is no other process that could change it, so
    polling would only burn cycles re-reading the same bytes.
    """

    refreshInterval = None

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    async def loadAll(self) -> list[ServerCredential]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CredentialConfigError(f"Cannot read credential file '{self._path}': {exc}") from exc
        return parseCredentials(raw, str(self._path)) if raw.strip() else []


class EnvCredentialSource:
    """Credentials from a JSON blob in the environment.

    Static, for the same reason as the file: the environment of a running
    process does not change under it. Unset means an empty store, so every
    request fails authentication -- better than an open door.
    """

    refreshInterval = None

    def __init__(self, raw: str | None = None) -> None:
        self._raw = raw

    async def loadAll(self) -> list[ServerCredential]:
        raw = self._raw if self._raw is not None else os.environ.get(ENV_SERVER_CREDENTIALS, "")
        raw = raw.strip()
        return parseCredentials(raw, ENV_SERVER_CREDENTIALS) if raw else []


class InMemoryCredentialSource:
    """A fixed set of credentials. Useful in tests, and as the shortest example
    of what a source has to provide."""

    refreshInterval = None

    def __init__(self, credentials: Iterable[ServerCredential]) -> None:
        self._byId = {c.serverId: c for c in credentials}

    async def loadAll(self) -> list[ServerCredential]:
        return list(self._byId.values())


def buildCredentialSource() -> CredentialSource:
    """Pick the credential store from the environment.

    Today: one node reading a file, or the inline blob if no file is named.
    When this grows past one node, a Firestore source slots in here and starts
    being polled because it declares a ``refreshInterval`` -- no other file
    changes.
    """
    path = os.environ.get(ENV_CREDENTIALS_FILE, "").strip()
    return FileCredentialSource(path) if path else EnvCredentialSource()
