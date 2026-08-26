"""The projectId -> ragDbId mapping, against real Firestore.

What matters here is not that the mapping works but that it is *stable*: a
project resolving to a second ragDbId would leave everything ingested under the
first one stranded in Pinecone, unreachable and still billable.

Run against the real service on purpose. The transaction in ``resolveOrCreate``
exists for exactly one reason -- two first submissions arriving together must
not mint two ids -- and a dict with a lock satisfies that by construction, so a
stand-in would pass whether or not the transaction were there at all.
"""

import asyncio
import os
from collections.abc import Iterator

import pytest

from app.stores.projectStore import buildProjectStore, newRagDbId


def testAnUnknownProjectResolvesToNothing(projects, scratch) -> None:
    assert asyncio.run(projects.resolve(scratch.projectId())) is None


def testResolvingDoesNotCreate(projects, scratch) -> None:
    """Read-only really is read-only -- otherwise a mistyped projectId on a
    search would leave an empty database behind forever."""
    projectId = scratch.projectId()

    asyncio.run(projects.resolve(projectId))

    assert asyncio.run(projects.resolve(projectId)) is None


def testAFirstResolveOrCreateMintsADatabase(projects, scratch) -> None:
    projectId = scratch.projectId()

    ragDbId = asyncio.run(projects.resolveOrCreate(projectId))

    assert ragDbId
    assert asyncio.run(projects.resolve(projectId)) == ragDbId


def testAProjectKeepsTheSameDatabaseForever(projects, scratch) -> None:
    """The invariant the whole module exists to hold. A second id would strand
    everything ingested under the first."""
    projectId = scratch.projectId()

    first = asyncio.run(projects.resolveOrCreate(projectId))
    second = asyncio.run(projects.resolveOrCreate(projectId))

    assert first == second


def testASecondClientSeesTheMapping(projects, scratch) -> None:
    """The reason this is not in process memory at all: another API instance
    has to resolve a project to the same database."""
    from app.stores.projectStore import FirestoreProjectStore

    projectId = scratch.projectId()
    ragDbId = asyncio.run(projects.resolveOrCreate(projectId))

    assert asyncio.run(FirestoreProjectStore().resolve(projectId)) == ragDbId


def testDifferentProjectsGetDifferentDatabases(projects, scratch) -> None:
    acme = asyncio.run(projects.resolveOrCreate(scratch.projectId("acme")))
    globex = asyncio.run(projects.resolveOrCreate(scratch.projectId("globex")))

    assert acme != globex


@pytest.mark.slow
def testConcurrentFirstSubmissionsAgreeOnOneDatabase(scratch) -> None:
    """Two submissions for a brand-new project arriving together must not mint
    two ids -- the loser's ingestion would populate a namespace nothing ever
    resolves to again.

    Separate store instances, not one: a single instance could agree with
    itself through some accident of its own state, which is precisely the
    reassurance a stand-in used to give. Marked slow because it is a dozen
    concurrent transactions against a real service.
    """
    from app.stores.projectStore import FirestoreProjectStore

    projectId = scratch.projectId()

    async def race() -> list[str]:
        return list(
            await asyncio.gather(
                *(FirestoreProjectStore().resolveOrCreate(projectId) for _ in range(12))
            )
        )

    minted = asyncio.run(race())

    assert len(set(minted)) == 1


# --- the id itself, which needs no store ----------------------------------


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


# --- the factory -----------------------------------------------------------


@pytest.fixture
def noGcpProject(monkeypatch) -> Iterator[None]:
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    yield


def testWithoutAGcpProjectTheStoreRefusesToBuild(noGcpProject) -> None:
    """No in-process fallback, deliberately. A deployment that quietly started
    without Firestore would look healthy while holding the only record of where
    every project's vectors live in a dict that dies with the process."""
    assert not os.environ.get("GCP_PROJECT_ID")

    with pytest.raises(RuntimeError, match="GCP_PROJECT_ID"):
        buildProjectStore()
