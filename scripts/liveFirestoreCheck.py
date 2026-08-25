"""Check FirestoreProjectStore against real Firestore.

    python scripts/liveFirestoreCheck.py

Configuration comes from .env -- GCP_PROJECT_ID (the switch that selects
Firestore at all), GOOGLE_APPLICATION_CREDENTIALS (omit it inside GCP) and
FIRESTORE_DATABASE_ID. Environment variables set on the command line still win.

Firestore holds one thing now: the projectId -> ragDbId mapping. It is also the
one record whose loss cannot be recovered from -- hand a project a second
ragDbId and everything ingested under the first is stranded in Pinecone,
billable and unreachable -- so what matters here is that resolving is stable and
that two processes racing a project's first submission agree on one id.

Covers what a stand-in cannot: that the mapping round-trips through the real
service, that a second client sees the first's writes, and that the transaction
in resolveOrCreate actually holds when two of them collide.

Cleans up the documents it creates.
"""

from __future__ import annotations

import asyncio
import os
import sys

# Imported purely for its side effect: ``app/__init__.py`` calls
# ``load_dotenv()``, and without it the guard in ``main`` reads an environment
# that .env has not reached yet -- so a fully configured checkout would still
# exit claiming GCP_PROJECT_ID is unset, and the only way to run this would be
# to repeat on the command line what .env already says. Everything under ``app``
# that reads config at import time is imported below that guard, so this one
# line is enough to make the file the single source of configuration.
import app  # noqa: F401

PROJECT_ID = "live-firestore-check"
OTHER_PROJECT_ID = "live-firestore-check-second"

failures: list[str] = []


def check(description: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


async def main() -> None:
    if not os.environ.get("GCP_PROJECT_ID"):
        sys.exit("Set GCP_PROJECT_ID (and GOOGLE_APPLICATION_CREDENTIALS).")

    from app.infra.firestoreClient import firestoreClient
    from app.stores.projectStore import COLLECTION, FirestoreProjectStore

    store = FirestoreProjectStore()
    collection = firestoreClient().collection(COLLECTION)

    try:
        print("\n=== resolving a project that has never been seen")
        check("an unknown project resolves to nothing", await store.resolve(PROJECT_ID) is None)

        print("\n=== first submission mints a database")
        ragDbId = await store.resolveOrCreate(PROJECT_ID)
        check("a ragDbId was minted", bool(ragDbId))
        check("it is not the projectId", ragDbId != PROJECT_ID)
        check("resolving now finds it", await store.resolve(PROJECT_ID) == ragDbId)

        print("\n=== the mapping is stable")
        check("resolveOrCreate returns the same id", await store.resolveOrCreate(PROJECT_ID) == ragDbId)

        print("\n=== a second client sees it")
        # The whole reason this is not in process memory.
        check("another instance resolves the same id", await FirestoreProjectStore().resolve(PROJECT_ID) == ragDbId)

        print("\n=== two first submissions racing")
        # The transaction is the only thing stopping these minting two ids and
        # leaving one ingestion writing to a namespace nothing resolves to.
        first, second = await asyncio.gather(
            FirestoreProjectStore().resolveOrCreate(OTHER_PROJECT_ID),
            FirestoreProjectStore().resolveOrCreate(OTHER_PROJECT_ID),
        )
        check("both racers agree on one ragDbId", first == second)

        print("\n=== different projects get different databases")
        check("they are not the same", await store.resolve(PROJECT_ID) != await store.resolve(OTHER_PROJECT_ID))
    finally:
        print("\n=== cleanup")
        for projectId in (PROJECT_ID, OTHER_PROJECT_ID):
            collection.document(projectId).delete()
            print(f"  deleted {projectId}")

    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed: {failures}")
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
