"""The projectId -> ragDbId mapping.

What matters here is not that the mapping works but that it is *stable*: a
project resolving to a second ragDbId would leave everything ingested under the
first one stranded in Pinecone, unreachable and still billable.
"""

import asyncio
import os
from collections.abc import Iterator

import pytest

from app.projectStore import (
    InMemoryProjectStore,
    buildProjectStore,
    newRagDbId,
)


@pytest.fixture
def store() -> InMemoryProjectStore:
    return InMemoryProjectStore()


@pytest.fixture
def noGcpProject(monkeypatch) -> Iterator[None]:
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    yield


def testAnUnknownProjectResolvesToNothing(store: InMemoryProjectStore) -> None:
    assert asyncio.run(store.resolve("acme")) is None


def testResolvingDoesNotCreate(store: InMemoryProjectStore) -> None:
    """Read-only really is read-only -- otherwise a mistyped projectId on a
    search would leave an empty database behind."""
    asyncio.run(store.resolve("acme"))

    assert len(store) == 0


def testAFirstResolveOrCreateMintsADatabase(store: InMemoryProjectStore) -> None:
    ragDbId = asyncio.run(store.resolveOrCreate("acme"))

    assert ragDbId
    assert asyncio.run(store.resolve("acme")) == ragDbId


def testAProjectKeepsTheSameDatabaseForever(store: InMemoryProjectStore) -> None:
    """The invariant the whole module exists to hold. A second id would strand
    everything ingested under the first."""
    first = asyncio.run(store.resolveOrCreate("acme"))
    second = asyncio.run(store.resolveOrCreate("acme"))

    assert first == second


def testDifferentProjectsGetDifferentDatabases(store: InMemoryProjectStore) -> None:
    acme = asyncio.run(store.resolveOrCreate("acme"))
    globex = asyncio.run(store.resolveOrCreate("globex"))

    assert acme != globex


def testConcurrentFirstSubmissionsAgreeOnOneDatabase(store: InMemoryProjectStore) -> None:
    """Two submissions for a brand-new project arriving together must not mint
    two ids -- the loser's ingestion would populate a namespace nothing ever
    resolves to again."""

    async def race() -> list[str]:
        return list(await asyncio.gather(*(store.resolveOrCreate("acme") for _ in range(25))))

    minted = asyncio.run(race())

    assert len(set(minted)) == 1
    assert len(store) == 1


def testARagDbIdIsNotTheProjectId() -> None:
    """Nothing may treat the two as interchangeable. They were the same string
    before this mapping existed, and code written then would still pass if they
    still were."""
    assert newRagDbId("acme") != "acme"


def testTwoIdsForOneProjectAreNeverTheSame() -> None:
    """Random, not derived. A derived id is one nothing can ever change, which
    would give away the ability to rebuild a project into a fresh namespace."""
    assert newRagDbId("acme") != newRagDbId("acme")


def testARagDbIdKeepsAReadablePrefix() -> None:
    """For whoever is reading a job table or a Firestore console by eye."""
    assert newRagDbId("acme").startswith("acme-")


@pytest.mark.parametrize("projectId", ["acme/corp", "acme corp", "acme:corp#1"])
def testAwkwardProjectIdsAreSanitizedIntoThePrefix(projectId: str) -> None:
    ragDbId = newRagDbId(projectId)

    assert set(ragDbId) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def testALongProjectIdDoesNotProduceAnUnboundedId() -> None:
    assert len(newRagDbId("x" * 500)) <= 64 + 1 + 12


def testWithoutAGcpProjectTheMappingIsInMemory(noGcpProject) -> None:
    """Same switch as the job table in app.jobManager -- the two are one
    durability decision and must not be able to disagree."""
    assert isinstance(buildProjectStore(), InMemoryProjectStore)
    assert not os.environ.get("GCP_PROJECT_ID")
