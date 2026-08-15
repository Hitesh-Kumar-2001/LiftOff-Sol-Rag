import asyncio

import pytest

from app.credentials import (
    ENV_CREDENTIALS_FILE,
    CredentialConfigError,
    EnvCredentialSource,
    FileCredentialSource,
    InMemoryCredentialSource,
    ServerCredential,
    buildCredentialSource,
    hashSecret,
    parseCredentials,
)
from app.security import AuthenticationError, ServerRegistry, refreshing

# Fast enough to keep the suite quick, slow enough to survive a loaded machine.
TICK = 0.01
SETTLE = 0.15

CREDENTIAL_JSON = '{"svc": {"secret": "abc"}}'


def credential(serverId: str = "svc", secret: str = "abc") -> ServerCredential:
    return ServerCredential(serverId=serverId, secretHash=hashSecret(secret))


class FakeStore(InMemoryCredentialSource):
    """A shared store: editable, pollable, and able to go down like a real one."""

    def __init__(self, credentials=(), refreshInterval: float | None = TICK) -> None:
        super().__init__(credentials)
        self.refreshInterval = refreshInterval
        self.reads = 0
        self.down = False

    async def loadAll(self):
        self.reads += 1
        if self.down:
            raise ConnectionError("store unreachable")
        return await super().loadAll()

    def put(self, cred: ServerCredential) -> None:
        self._byId[cred.serverId] = cred

    def remove(self, serverId: str) -> None:
        self._byId.pop(serverId, None)


# --- parsing -----------------------------------------------------------------


def testParseCredentialsAcceptsAPlaintextSecret() -> None:
    [cred] = parseCredentials('{"svc": {"secret": "abc"}}')

    assert cred.serverId == "svc"
    assert cred.matches("abc")


def testParseCredentialsAcceptsAPrecomputedDigest() -> None:
    [cred] = parseCredentials('{"svc": {"secretSha256": "%s"}}' % hashSecret("abc"))

    assert cred.matches("abc")


def testParseCredentialsRejectsAnEntryWithoutASecret() -> None:
    with pytest.raises(CredentialConfigError):
        parseCredentials('{"svc": {}}')


def testParseCredentialsRejectsMalformedJson() -> None:
    with pytest.raises(CredentialConfigError):
        parseCredentials("not json")


def testAnUnsetEnvironmentYieldsAnEmptyStore() -> None:
    assert asyncio.run(EnvCredentialSource("").loadAll()) == []


def testABadDocumentNamesTheStoreItCameFrom() -> None:
    with pytest.raises(CredentialConfigError, match="creds.json"):
        parseCredentials("not json", "creds.json")


# --- choosing a store --------------------------------------------------------


def testCredentialsLoadFromAFile(tmp_path) -> None:
    path = tmp_path / "creds.json"
    path.write_text(CREDENTIAL_JSON, encoding="utf-8")

    [cred] = asyncio.run(FileCredentialSource(path).loadAll())

    assert cred.matches("abc")


def testAMissingCredentialFileIsAConfigError(tmp_path) -> None:
    with pytest.raises(CredentialConfigError, match="Cannot read"):
        asyncio.run(FileCredentialSource(tmp_path / "absent.json").loadAll())


def testAnEmptyCredentialFileYieldsAnEmptyStore(tmp_path) -> None:
    path = tmp_path / "creds.json"
    path.write_text("", encoding="utf-8")

    assert asyncio.run(FileCredentialSource(path).loadAll()) == []


def testANamedFileWinsOverTheInlineBlob(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_CREDENTIALS_FILE, str(tmp_path / "creds.json"))

    assert isinstance(buildCredentialSource(), FileCredentialSource)


def testTheInlineBlobIsTheFallback(monkeypatch) -> None:
    monkeypatch.delenv(ENV_CREDENTIALS_FILE, raising=False)

    assert isinstance(buildCredentialSource(), EnvCredentialSource)


def testLocalStoresDeclareThemselvesStatic(tmp_path) -> None:
    # Nothing else can change them mid-flight, so nothing should poll them.
    assert FileCredentialSource(tmp_path / "creds.json").refreshInterval is None
    assert EnvCredentialSource().refreshInterval is None


# --- verifying from memory ---------------------------------------------------


def testStartupLoadsTheWholeStoreIntoMemory() -> None:
    registry = ServerRegistry(FakeStore([credential()]))

    assert asyncio.run(registry.loadAll()) == 1
    assert len(registry) == 1


def testVerificationNeverTouchesTheStore() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)
    asyncio.run(registry.loadAll())

    for _ in range(50):
        registry.authenticate("svc", "abc")
        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "wrong")
        with pytest.raises(AuthenticationError):
            registry.authenticate("attacker", "guess")

    assert store.reads == 1  # Only the startup load.


def testAnEmptyRegistryAuthenticatesNobody() -> None:
    registry = ServerRegistry(FakeStore())

    with pytest.raises(AuthenticationError):
        registry.authenticate("svc", "abc")


# --- what the refresh timer catches -----------------------------------------


def testARevokedServerStopsBeingAccepted() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        registry.authenticate("svc", "abc")  # Works before revocation.

        store.remove("svc")  # Server decommissioned upstream.
        await registry.loadAll()  # What the timer does.

        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "abc")

    asyncio.run(scenario())


def testARotatedSecretTakesEffect() -> None:
    store = FakeStore([credential(secret="old")])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        store.put(credential(secret="new"))
        await registry.loadAll()

        registry.authenticate("svc", "new")
        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "old")

    asyncio.run(scenario())


def testANewlyIssuedCredentialIsPickedUp() -> None:
    store = FakeStore()
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        store.put(credential())
        await registry.loadAll()

        assert registry.authenticate("svc", "abc").serverId == "svc"

    asyncio.run(scenario())


# --- polling only when the store can change ----------------------------------


def testAStaticStoreIsNeverReRead() -> None:
    store = FakeStore([credential()], refreshInterval=None)
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)

    asyncio.run(scenario())

    assert store.reads == 1  # The startup load, and nothing since.


def testASharedStoreIsReReadOnItsOwnInterval() -> None:
    store = FakeStore([credential()], refreshInterval=TICK)
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)

    asyncio.run(scenario())

    assert store.reads > 2


def testTheRefreshLoopIsTornDownWithTheApp() -> None:
    store = FakeStore([credential()], refreshInterval=TICK)
    registry = ServerRegistry(store)

    async def scenario() -> int:
        await registry.loadAll()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)
        settled = store.reads
        await asyncio.sleep(SETTLE)  # Nothing should still be running.
        return store.reads - settled

    assert asyncio.run(scenario()) == 0


# --- the timer itself --------------------------------------------------------


def testThePollerKeepsReReadingTheStore() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        poller = asyncio.create_task(registry.refreshForever(interval=TICK))
        await asyncio.sleep(SETTLE)
        poller.cancel()

    asyncio.run(scenario())

    assert store.reads > 2


def testAStoreOutageLeavesTheCachedCredentialsServing() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        store.down = True
        poller = asyncio.create_task(registry.refreshForever(interval=TICK))
        await asyncio.sleep(SETTLE)
        poller.cancel()

    asyncio.run(scenario())

    assert registry.authenticate("svc", "abc").serverId == "svc"


def testThePollerSurvivesAnOutageAndCatchesUpAfterwards() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.loadAll()
        poller = asyncio.create_task(registry.refreshForever(interval=TICK))

        store.down = True
        await asyncio.sleep(SETTLE)
        assert len(registry) == 1  # Still serving the last known-good copy.

        store.down = False
        store.remove("svc")  # Revoked while the store was unreachable.
        await asyncio.sleep(SETTLE)

        assert len(registry) == 0  # Poller lived through it and caught up.
        poller.cancel()

    asyncio.run(scenario())
