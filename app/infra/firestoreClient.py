"""Reaching Firestore: one client, one place that knows how to build it.

Firestore holds the durable records -- today that is the ``projectId`` ->
``ragDbId`` mapping (see ``app.stores.projectStore``), which is the one thing whose
loss cannot be recovered from, since it is the only record of where a
project's vectors live. Jobs are *not* here: they are transient working state
and live in Redis (see ``app.jobs.redisJobStore``).

Split out from any one consumer because the client, the credential handling,
and the database-id trap below are not the business of whichever module
happens to need a collection.
"""

from __future__ import annotations

import os
import threading

import firebase_admin
from firebase_admin import credentials, firestore

# Where the service account key lives. Unset in a managed GCP environment,
# where application default credentials are supplied by the platform.
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# Which database in the project. A project's first database is usually the
# special one literally called "(default)", and the client assumes it -- but a
# project can hold several, and one *named* "default" is a different database
# from "(default)". The failure is confusing enough to be worth naming: the
# client reports 'The database (default) does not exist for project X' while
# the console plainly shows a database sitting there.
DATABASE_ID = os.environ.get("FIRESTORE_DATABASE_ID") or None


_initLock = threading.Lock()


def firestoreClient() -> firestore.Client:
    """The shared Firestore client, initialised once per process.

    Locked because the check and the initialisation are separate steps, and
    ``initialize_app`` raises if the default app already exists. Callers reach
    this from a thread pool (the client is synchronous, so everything above it
    wraps calls in ``asyncio.to_thread``), which makes two arriving together an
    ordinary occurrence rather than a theoretical one.
    """
    if not firebase_admin._apps:
        with _initLock:
            if not firebase_admin._apps:
                # An explicit key file when one is named, otherwise the
                # credentials the platform provides -- which is how this runs
                # on Cloud Run without a key file on disk at all.
                cred = credentials.Certificate(CREDENTIALS_PATH) if CREDENTIALS_PATH else None
                firebase_admin.initialize_app(cred)
    return firestore.client(database_id=DATABASE_ID)
