import asyncio

import pytest

from app.credentials import (
    ENV_CREDENTIALS_FILE,
    CredentialConfigError,
    EnvCredentialSource,
    FileCredentialSource,
    InMemoryCredentialSource,
    ServerCredential,
    build_credential_source,
    hash_secret,
    parse_credentials,
)
from app.security import AuthenticationError, ServerRegistry, refreshing

# Fast enough to keep the suite quick, slow enough to survive a loaded machine.
TICK = 0.01
SETTLE = 0.15

CREDENTIAL_JSON = '{"svc": {"secret": "abc"}}'


def credential(server_id: str = "svc", secret: str = "abc") -> ServerCredential:
    return ServerCredential(server_id=server_id, secret_hash=hash_secret(secret))


class FakeStore(InMemoryCredentialSource):
    """A shared store: editable, pollable, and able to go down like a real one."""

    def __init__(self, credentials=(), refresh_interval: float | None = TICK) -> None:
        super().__init__(credentials)
        self.refresh_interval = refresh_interval
        self.reads = 0
        self.down = False

    async def load_all(self):
        self.reads += 1
        if self.down:
            raise ConnectionError("store unreachable")
        return await super().load_all()

    def put(self, cred: ServerCredential) -> None:
        self._by_id[cred.server_id] = cred

    def remove(self, server_id: str) -> None:
        self._by_id.pop(server_id, None)


# --- parsing -----------------------------------------------------------------


def test_parse_credentials_accepts_a_plaintext_secret() -> None:
    [cred] = parse_credentials('{"svc": {"secret": "abc"}}')

    assert cred.server_id == "svc"
    assert cred.matches("abc")


def test_parse_credentials_accepts_a_precomputed_digest() -> None:
    [cred] = parse_credentials('{"svc": {"secretSha256": "%s"}}' % hash_secret("abc"))

    assert cred.matches("abc")


def test_parse_credentials_rejects_an_entry_without_a_secret() -> None:
    with pytest.raises(CredentialConfigError):
        parse_credentials('{"svc": {}}')


def test_parse_credentials_rejects_malformed_json() -> None:
    with pytest.raises(CredentialConfigError):
        parse_credentials("not json")


def test_an_unset_environment_yields_an_empty_store() -> None:
    assert asyncio.run(EnvCredentialSource("").load_all()) == []


def test_a_bad_document_names_the_store_it_came_from() -> None:
    with pytest.raises(CredentialConfigError, match="creds.json"):
        parse_credentials("not json", "creds.json")


# --- choosing a store --------------------------------------------------------


def test_credentials_load_from_a_file(tmp_path) -> None:
    path = tmp_path / "creds.json"
    path.write_text(CREDENTIAL_JSON, encoding="utf-8")

    [cred] = asyncio.run(FileCredentialSource(path).load_all())

    assert cred.matches("abc")


def test_a_missing_credential_file_is_a_config_error(tmp_path) -> None:
    with pytest.raises(CredentialConfigError, match="Cannot read"):
        asyncio.run(FileCredentialSource(tmp_path / "absent.json").load_all())


def test_an_empty_credential_file_yields_an_empty_store(tmp_path) -> None:
    path = tmp_path / "creds.json"
    path.write_text("", encoding="utf-8")

    assert asyncio.run(FileCredentialSource(path).load_all()) == []


def test_a_named_file_wins_over_the_inline_blob(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_CREDENTIALS_FILE, str(tmp_path / "creds.json"))

    assert isinstance(build_credential_source(), FileCredentialSource)


def test_the_inline_blob_is_the_fallback(monkeypatch) -> None:
    monkeypatch.delenv(ENV_CREDENTIALS_FILE, raising=False)

    assert isinstance(build_credential_source(), EnvCredentialSource)


def test_local_stores_declare_themselves_static(tmp_path) -> None:
    # Nothing else can change them mid-flight, so nothing should poll them.
    assert FileCredentialSource(tmp_path / "creds.json").refresh_interval is None
    assert EnvCredentialSource().refresh_interval is None


# --- verifying from memory ---------------------------------------------------


def test_startup_loads_the_whole_store_into_memory() -> None:
    registry = ServerRegistry(FakeStore([credential()]))

    assert asyncio.run(registry.load_all()) == 1
    assert len(registry) == 1


def test_verification_never_touches_the_store() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)
    asyncio.run(registry.load_all())

    for _ in range(50):
        registry.authenticate("svc", "abc")
        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "wrong")
        with pytest.raises(AuthenticationError):
            registry.authenticate("attacker", "guess")

    assert store.reads == 1  # Only the startup load.


def test_an_empty_registry_authenticates_nobody() -> None:
    registry = ServerRegistry(FakeStore())

    with pytest.raises(AuthenticationError):
        registry.authenticate("svc", "abc")


# --- what the refresh timer catches -----------------------------------------


def test_a_revoked_server_stops_being_accepted() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        registry.authenticate("svc", "abc")  # Works before revocation.

        store.remove("svc")  # Server decommissioned upstream.
        await registry.load_all()  # What the timer does.

        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "abc")

    asyncio.run(scenario())


def test_a_rotated_secret_takes_effect() -> None:
    store = FakeStore([credential(secret="old")])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        store.put(credential(secret="new"))
        await registry.load_all()

        registry.authenticate("svc", "new")
        with pytest.raises(AuthenticationError):
            registry.authenticate("svc", "old")

    asyncio.run(scenario())


def test_a_newly_issued_credential_is_picked_up() -> None:
    store = FakeStore()
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        store.put(credential())
        await registry.load_all()

        assert registry.authenticate("svc", "abc").server_id == "svc"

    asyncio.run(scenario())


# --- polling only when the store can change ----------------------------------


def test_a_static_store_is_never_re_read() -> None:
    store = FakeStore([credential()], refresh_interval=None)
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)

    asyncio.run(scenario())

    assert store.reads == 1  # The startup load, and nothing since.


def test_a_shared_store_is_re_read_on_its_own_interval() -> None:
    store = FakeStore([credential()], refresh_interval=TICK)
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)

    asyncio.run(scenario())

    assert store.reads > 2


def test_the_refresh_loop_is_torn_down_with_the_app() -> None:
    store = FakeStore([credential()], refresh_interval=TICK)
    registry = ServerRegistry(store)

    async def scenario() -> int:
        await registry.load_all()
        async with refreshing(registry):
            await asyncio.sleep(SETTLE)
        settled = store.reads
        await asyncio.sleep(SETTLE)  # Nothing should still be running.
        return store.reads - settled

    assert asyncio.run(scenario()) == 0


# --- the timer itself --------------------------------------------------------


def test_the_poller_keeps_re_reading_the_store() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        poller = asyncio.create_task(registry.refresh_forever(interval=TICK))
        await asyncio.sleep(SETTLE)
        poller.cancel()

    asyncio.run(scenario())

    assert store.reads > 2


def test_a_store_outage_leaves_the_cached_credentials_serving() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        store.down = True
        poller = asyncio.create_task(registry.refresh_forever(interval=TICK))
        await asyncio.sleep(SETTLE)
        poller.cancel()

    asyncio.run(scenario())

    assert registry.authenticate("svc", "abc").server_id == "svc"


def test_the_poller_survives_an_outage_and_catches_up_afterwards() -> None:
    store = FakeStore([credential()])
    registry = ServerRegistry(store)

    async def scenario() -> None:
        await registry.load_all()
        poller = asyncio.create_task(registry.refresh_forever(interval=TICK))

        store.down = True
        await asyncio.sleep(SETTLE)
        assert len(registry) == 1  # Still serving the last known-good copy.

        store.down = False
        store.remove("svc")  # Revoked while the store was unreachable.
        await asyncio.sleep(SETTLE)

        assert len(registry) == 0  # Poller lived through it and caught up.
        poller.cancel()

    asyncio.run(scenario())
