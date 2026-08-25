"""Keeps the suite off real infrastructure, whatever ``.env`` happens to say.

``app/__init__.py`` calls ``load_dotenv()``, which is exactly right for running
the service and quietly wrong for running its tests: a developer who has
configured Firestore -- which is the normal state of a working checkout -- would
otherwise have ``GCP_PROJECT_ID`` set, and every factory in the codebase reads
that one variable to decide between a dict and the real service. The suite would
then read, write and *delete* documents in whichever project the developer
happened to be pointed at, differ between machines, and fail with no network.

So the variable is removed for the whole session. Tests that want a Firestore
backend build the class directly with a stand-in client (``tests/testPromptStore.py``
does this); tests that want the switch itself set it back through ``monkeypatch``
(``tests/testProjectStore.py`` does this). The live checks under ``scripts/``
are where real infrastructure is exercised, deliberately outside pytest.

Order matters: ``app`` is imported *first*, so ``load_dotenv`` has already run by
the time anything is removed. Removing first would achieve nothing -- dotenv
fills in variables that are absent, so the next ``app`` import would put it
straight back.
"""

import os

import app  # noqa: F401  -- imported for load_dotenv(); see above.

for _variable in ("GCP_PROJECT_ID", "GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ.pop(_variable, None)
