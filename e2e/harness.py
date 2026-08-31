"""Where the live service is, and how this suite addresses it.

A module of its own rather than constants in ``conftest.py``, so that the test
module and the fixtures import the same object from the same place. Importing
names *out of* a conftest works, but conftest is a file pytest owns and imports
by its own rules, and having application code depend on that is the kind of
thing that breaks on a pytest upgrade for no reason worth debugging.

Every value here is overridable from the environment, because the whole point of
this suite is that the service is somewhere else -- a container, another host,
a staging deployment -- and none of those are ``http://127.0.0.1:8000``.
"""

from __future__ import annotations

import os

BASE_URL = os.environ.get("RAG_E2E_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Who this suite says it is. Unverified by the service -- there is no
# authentication anywhere except the webhook signatures -- so it is a label that
# shows up in the API log and nothing more. Distinctive on purpose: it is what
# you grep the server log for when a run goes wrong.
SERVER_ID = os.environ.get("RAG_E2E_SERVER_ID", "e2e-sales-check")


def apiUrl(path: str) -> str:
    return f"{BASE_URL}{path}"
